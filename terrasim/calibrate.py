"""Calibration harness: run the column physics at observation stations and score it.

Full-grid runs are too slow to evaluate hundreds of parameter sets, so the
harness runs the engine only at the station pixels ("point mode", routing
off) and keeps the full grid for the final spatial checks. Point mode is
exactly the full grid at those pixels except that drainage picks its
sub-step count from the wettest pixel present (``Terrasim.drain_frac_max`` /
``max_substeps``), so with the default limiter the top layer can differ by up
to ~1e-2; pin ``max_substeps`` to compare exactly. Pieces:

- :func:`groups`: cluster stations into bboxes a few km across (one static +
  forcing fetch per group, shared by every parameter set).
- :func:`point_subset`: the engine's arrays for just the station pixels.
- :func:`run_points`: the daily model series at those pixels.
- :func:`pair`: model theta averaged over each observation's depth interval
  (thickness-weighted over the model layers), joined to the observation.
- :func:`score`: bias, RMSE, unbiased RMSE, r, KGE per station and depth.
- :func:`evaluate`: all of the above for one site and one parameter set, with
  static datasets and forcing cached on disk under ``Paths.root/calib``.

Example:
    ```python
    from terrasim import calibrate
    from terrasim.params import Params
    from terrasim.physics.surface import SurfaceParams
    paired, scores = calibrate.evaluate('sasmas', params=Params(surface=SurfaceParams(seal_ksat_mm_h=5)))
    scores.groupby('depth').kge.median()
    ```
"""
import hashlib
import json
import os
import pickle
import time
import warnings
from datetime import date, timedelta

import attrs
import numpy as np
import pandas as pd
import xarray as xr

from terrasim import grid, obs
from terrasim.forcing import Forcing, ForcingBuilder
from terrasim.layers import LayerSpec, overlap_matrix
from terrasim.model import Model
from terrasim.params import Params
from terrasim.paths import Paths
from terrasim.physics.engine import Engine
from terrasim.physics.state import Fluxes, State, StaticArrays
from terrasim.static import StaticBuilder, to_arrays
from terrasim.terrasim import Terrasim
from terrasim.validate import metrics as _metrics

POINT_FLUXES = ('rain', 'infiltration', 'runoff_out', 'deep_drain', 'et_actual')
SENTINEL2_FROM = 2017      # first year pysentinel2 NDVI is used; earlier runs take the constant-Kcb fallback


# -- station groups ---------------------------------------------------------------

def _metres(lon, lat, lon0, lat0):
    return (np.asarray(lon) - lon0) * 111_320 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 111_320


def groups(stations: pd.DataFrame, max_span_m: float = 2500.0, margin_m: float = 150.0) -> list:
    """Greedy clustering of ``obs.stations()`` rows into groups whose stations lie within
    ``max_span_m`` of the first member; each group gets a bbox with ``margin_m`` around its
    stations. Returns ``[{'site', 'name', 'stations': [...], 'bbox': [W, S, E, N]}, ...]``."""
    out = []
    for site, st in stations.groupby('site'):
        st = st.sort_values(['lat', 'lon']).reset_index(drop=True)
        left = list(st.index)
        k = 0
        while left:
            i0 = left[0]
            dx, dy = _metres(st.lon[left], st.lat[left], st.lon[i0], st.lat[i0])
            members = [i for i, d in zip(left, np.hypot(dx, dy)) if d <= max_span_m]
            left = [i for i in left if i not in members]
            sub = st.loc[members]
            dlat = margin_m / 111_320
            dlon = margin_m / (111_320 * np.cos(np.radians(sub.lat.mean())))
            out.append({'site': site, 'name': f'{site}_g{k:02d}', 'stations': sub.station.tolist(),
                        'bbox': [float(sub.lon.min() - dlon), float(sub.lat.min() - dlat), float(sub.lon.max() + dlon), float(sub.lat.max() + dlat)],
                        'start': sub.start.min().date(), 'end': sub.end.max().date()})
            k += 1
    return out


def station_pixels(window, lons, lats):
    """``(iy, ix)`` of the model pixels containing the points (nearest centre)."""
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', grid.CRS, always_xy=True)
    X, Y = tr.transform(np.asarray(lons, float), np.asarray(lats, float))
    y, x = grid.coords(window)
    iy = np.abs(Y[:, None] - y[None, :]).argmin(axis=1)
    ix = np.abs(X[:, None] - x[None, :]).argmin(axis=1)
    return iy, ix


# -- point mode -------------------------------------------------------------------

def point_subset(static_ds: xr.Dataset, forcing: Forcing, iy, ix):
    """Engine arrays and forcing for the pixels ``(iy, ix)`` only, laid out as a ``(1, n)`` grid
    with no routing geometry (point mode is routing-free)."""
    a = to_arrays(static_ds)
    iy = np.asarray(iy); ix = np.asarray(ix)
    s3 = lambda v: np.ascontiguousarray(v[:, iy, ix][:, None, :])
    s2 = lambda v: np.ascontiguousarray(v[iy, ix][None, :])
    st = StaticArrays(dz_mm=a.dz_mm, theta_s=s3(a.theta_s), theta_dul=s3(a.theta_dul), theta_l15=s3(a.theta_l15), ksat=s3(a.ksat),
                      b=s3(a.b), psi_e=s3(a.psi_e), psi_f=s2(a.psi_f), rew=s2(a.rew), tew=s2(a.tew), ze_mm=a.ze_mm,
                      root_frac=s3(a.root_frac), k_bottom_scale=s2(a.k_bottom_scale), valid=s2(a.valid),
                      twi=None if a.twi is None else s2(a.twi), elevation=None if a.elevation is None else s2(a.elevation))
    f = attrs.evolve(forcing, cell_index=s2(forcing.cell_index), point_index=s2(forcing.point_index),
                     cell_weights=np.ascontiguousarray(forcing.cell_weights[:, iy, ix][:, None, :]),
                     point_weights=np.ascontiguousarray(forcing.point_weights[:, iy, ix][:, None, :]),
                     kcb=np.ascontiguousarray(forcing.kcb[:, iy, ix][:, None, :]), fc=np.ascontiguousarray(forcing.fc[:, iy, ix][:, None, :]))
    return st, f


def run_points(model: Model, static_ds: xr.Dataset, forcing: Forcing, iy, ix, names=None) -> xr.Dataset:
    """Daily theta per layer and a few fluxes at the station pixels, ``(time, layer, point)``."""
    st, f = point_subset(static_ds, forcing, iy, ix)
    if f.nstep != f.nday * model.settings.steps_per_day:
        raise ValueError(f'forcing has {f.nstep} steps for {f.nday} days but step={model.settings.step!r} needs {f.nday * model.settings.steps_per_day}')
    names = list(names) if names is not None else [f'p{i}' for i in range(st.shape[1])]
    bounds = (static_ds['layer'].values, static_ds['layer_top_m'].values, static_ds['layer_bottom_m'].values)
    return _run_point_arrays(model, st, f, names, bounds)


# -- pairing and scoring ----------------------------------------------------------

def pair(points: xr.Dataset, observations: pd.DataFrame) -> pd.DataFrame:
    """Join observations (``obs`` schema) to the model theta averaged over each observation's
    depth interval, by station name (= point name) and day. Depth intervals below the model
    profile take the bottom layer's value."""
    src = np.column_stack([points['layer_top_m'].values, points['layer_bottom_m'].values])
    th = points['theta']
    frames = []
    for (station, top, bot), o in observations.groupby(['station', 'depth_top_m', 'depth_bottom_m']):
        if station not in points['point'].values:
            continue
        w = overlap_matrix(src, [[top, bot]])[0]                              # (L,)
        series = pd.Series((th.sel(point=station).values * w[None, :]).sum(axis=1), index=pd.DatetimeIndex(points['time'].values))
        m = o[['site', 'station', 'lon', 'lat', 'time', 'depth_top_m', 'depth_bottom_m', 'vwc', 'n']].copy()
        m['mod'] = series.reindex(pd.DatetimeIndex(m['time'])).values
        frames.append(m.dropna(subset=['mod']))
    if not frames:
        return pd.DataFrame(columns=['site', 'station', 'lon', 'lat', 'time', 'depth_top_m', 'depth_bottom_m', 'vwc', 'n', 'mod'])
    return pd.concat(frames, ignore_index=True).rename(columns={'vwc': 'obs'})


def kge(o, m) -> float:
    """Kling-Gupta efficiency (2009 form)."""
    o = np.asarray(o, float); m = np.asarray(m, float)
    ok = np.isfinite(o) & np.isfinite(m); o, m = o[ok], m[ok]
    if o.size < 3 or o.std() == 0 or m.std() == 0:
        return np.nan
    r = np.corrcoef(o, m)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (m.std() / o.std() - 1) ** 2 + (m.mean() / o.mean() - 1) ** 2))


def score(paired: pd.DataFrame, by=('site', 'station', 'depth_top_m', 'depth_bottom_m')) -> pd.DataFrame:
    """Metrics per group of ``by``: n, bias (model - obs), rmse, ubrmse, r, nse, kge, plus the
    obs and model means. VWC units (m3 m-3)."""
    rows = []
    for key, g in paired.groupby(list(by)):
        m = _metrics(g['obs'], g['mod'])
        m.update(kge=kge(g['obs'], g['mod']), obs_mean=float(g['obs'].mean()), mod_mean=float(g['mod'].mean()))
        rows.append(dict(zip(by, key)) | m)
    df = pd.DataFrame(rows)
    if {'depth_top_m', 'depth_bottom_m'} <= set(df.columns):
        df['depth'] = df.apply(lambda r: f"{int(round(r.depth_top_m * 100))}-{int(round(r.depth_bottom_m * 100))}cm", axis=1)
    return df


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """Median metrics per site and depth class, with station counts."""
    return scores.groupby(['site', 'depth']).agg(stations=('station', 'nunique'), n=('n', 'sum'), bias=('bias', 'median'),
                                                   ubrmse=('ubrmse', 'median'), r=('r', 'median'), kge=('kge', 'median')).round(3).reset_index()


# -- caching and the site evaluation ------------------------------------------------

def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def forcing_for(bbox, start: date, end: date, settings: Terrasim, params: Params = None, config=None, email=None, verbose=True) -> Forcing:
    """Forcing for a group bbox, cached on disk. It is always built with the default
    vegetation parameters; :func:`remap_vegetation` applies a parameter set's Kcb mapping
    afterwards, so sweeps over vegetation parameters reuse the cache."""
    paths = Paths(config) if config is not None else Paths()
    use_ndvi = start.year >= SENTINEL2_FROM
    params = Params()
    key = _key(bbox, start, end, settings.step, settings.within_hour, settings.peak_fraction, settings.disaggregation,
               settings.et_method, settings.tz_offset_hours, attrs.asdict(params.veg), use_ndvi)
    path = f'{paths.root}/calib/forcing_{key}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    t0 = time.time()
    kw = {'config': config} if config is not None else {}
    f = ForcingBuilder(settings=settings, params=params, use_ndvi=use_ndvi, **kw).build(bbox, start, end, email=email)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(f, fh)
    if verbose:
        print(f'calibrate: forcing {bbox} {start}..{end} built in {time.time() - t0:.0f}s (ndvi={use_ndvi}, barra timing {f.barra_fraction_wet_days:.2f})')
    return f


def remap_vegetation(forcing: Forcing, veg, use_ndvi: bool, veg_built=None) -> Forcing:
    """Kcb and cover for a new :class:`VegParams` from a forcing built with ``veg_built``
    (default parameters). Without NDVI the constant ``veg.kcb_constant`` is used; with NDVI the
    linear Kcb map is inverted and re-applied (exact except where the old map was clipped)."""
    from terrasim.params import VegParams
    from terrasim.vegetation import fc_from_kcb
    veg_built = veg_built or VegParams()
    if not use_ndvi:
        kcb = np.full_like(forcing.kcb, veg.kcb_constant, dtype='float32')
    else:
        frac = (forcing.kcb.astype(float) - veg_built.kcb_min) / max(veg_built.kcb_max - veg_built.kcb_min, 1e-9)
        kcb = np.clip(veg.kcb_min + (veg.kcb_max - veg.kcb_min) * frac, veg.kcb_min, veg.kcb_max).astype('float32')
    return attrs.evolve(forcing, kcb=kcb, fc=fc_from_kcb(kcb, veg).astype('float32'))


def static_for(bbox, settings: Terrasim, params: Params, layers: LayerSpec = LayerSpec(), config=None, api_key=None) -> xr.Dataset:
    kw = {'config': config} if config is not None else {}
    return StaticBuilder(params=params, layers=layers, settings=settings, **kw).build(bbox, api_key=api_key)


def evaluate(site: str, settings: Terrasim = Terrasim(), params: Params = Params(), start: date = None, end: date = None,
             observations: pd.DataFrame = None, only_groups=None, max_span_m: float = 2500.0, config=None, email=None,
             api_key=None, verbose: bool = True):
    """Point-mode run of one parameter set at every station of ``site`` and the paired scores.

    Returns ``(paired, scores)``. ``start``/``end`` clip the period (default: the site's record,
    with ``end`` no later than 120 days ago so the forcing stores have the data).
    """
    o = observations if observations is not None else obs.load([site])
    o = o[o.site == site]
    if start is not None:
        o = o[o.time >= pd.Timestamp(start)]
    latest = pd.Timestamp(date.today() - timedelta(days=120))
    o = o[o.time <= min(pd.Timestamp(end) if end is not None else latest, latest)]
    st = obs.stations(o)
    gs = groups(st, max_span_m=max_span_m)
    if only_groups is not None:
        gs = [g for g in gs if g['name'] in set(only_groups)]
    paired = []
    for g in gs:
        t0 = time.time()
        g_start, g_end = (start or g['start']), min(end or g['end'], latest.date())
        g_start = max(g_start, g['start']) if start is None else g_start
        static = static_for(g['bbox'], settings, params, config=config, api_key=api_key)
        forcing = remap_vegetation(forcing_for(g['bbox'], g_start, g_end, settings, config=config, email=email, verbose=verbose),
                                   params.veg, g_start.year >= SENTINEL2_FROM)
        sub = st[st.station.isin(g['stations'])]
        iy, ix = station_pixels(grid.model_window(g['bbox']), sub.lon.values, sub.lat.values)
        pts = run_points(Model(settings=settings, params=params), static, forcing, iy, ix, names=sub.station.tolist())
        p = pair(pts, o[o.station.isin(g['stations'])])
        p['group'] = g['name']
        paired.append(p)
        if verbose:
            print(f"calibrate: {g['name']} {len(g['stations'])} stations {g_start}..{g_end} -> {len(p)} pairs in {time.time() - t0:.0f}s")
    paired = pd.concat(paired, ignore_index=True) if paired else pd.DataFrame()
    return paired, (score(paired) if len(paired) else pd.DataFrame())


# -- all stations of a site in one engine call -----------------------------------------

def merge_points(parts):
    """Stack per-group ``(StaticArrays, Forcing)`` point subsets (each ``(L, 1, n_g)``) into one
    ``(L, 1, N)`` set so a whole site runs in one engine call. Forcings must share the time
    axis; SILO points and BARRA cells are concatenated with block-diagonal weights."""
    sts, fs = zip(*parts)
    n0 = fs[0].nstep
    if any(f.nstep != n0 for f in fs) or any(f.nday != fs[0].nday for f in fs):
        raise ValueError('point forcings cover different periods; build them over one common period')
    cat3 = lambda name: np.concatenate([getattr(s, name) for s in sts], axis=2)
    cat2 = lambda name: np.concatenate([getattr(s, name) for s in sts], axis=1)
    a0 = sts[0]
    st = StaticArrays(dz_mm=a0.dz_mm, theta_s=cat3('theta_s'), theta_dul=cat3('theta_dul'), theta_l15=cat3('theta_l15'), ksat=cat3('ksat'),
                      b=cat3('b'), psi_e=cat3('psi_e'), psi_f=cat2('psi_f'), rew=cat2('rew'), tew=cat2('tew'), ze_mm=a0.ze_mm,
                      root_frac=cat3('root_frac'), k_bottom_scale=cat2('k_bottom_scale'), valid=cat2('valid'),
                      twi=None if a0.twi is None else cat2('twi'), elevation=None if a0.elevation is None else cat2('elevation'))
    ncell = [f.rain.shape[1] for f in fs]; npt = [f.et0.shape[1] for f in fs]; npix = [f.kcb.shape[2] for f in fs]
    N, C, P = sum(npix), sum(ncell), sum(npt)
    cell_w = np.zeros((C, 1, N), dtype='float32'); point_w = np.zeros((P, 1, N), dtype='float32')
    cell_index = np.zeros((1, N), dtype=fs[0].cell_index.dtype); point_index = np.zeros((1, N), dtype=fs[0].point_index.dtype)
    c = p = x = 0
    for f, nc, npnt, nx in zip(fs, ncell, npt, npix):
        cell_w[c:c + nc, :, x:x + nx] = f.cell_weights; point_w[p:p + npnt, :, x:x + nx] = f.point_weights
        cell_index[:, x:x + nx] = f.cell_index + c; point_index[:, x:x + nx] = f.point_index + p
        c += nc; p += npnt; x += nx
    daily = {k: np.concatenate([f.daily[k] for f in fs], axis=1) for k in fs[0].daily}
    f = attrs.evolve(fs[0], rain=np.concatenate([f.rain for f in fs], axis=1), rain_source_barra=np.concatenate([f.rain_source_barra for f in fs], axis=1),
                     silo_wet=np.concatenate([f.silo_wet for f in fs], axis=1), et0=np.concatenate([f.et0 for f in fs], axis=1), daily=daily,
                     cell_index=cell_index, point_index=point_index, cell_weights=cell_w, point_weights=point_w,
                     kcb=np.concatenate([f.kcb for f in fs], axis=2), fc=np.concatenate([f.fc for f in fs], axis=2))
    return st, f


class SitePoints:
    """A site's stations prepared for repeated point-mode evaluation: forcing per group over
    one common period (fetched once, cached on disk), station pixels, observations. Each
    :meth:`evaluate` builds the static datasets for the given parameters (cached per
    parameter key) and runs every station in one engine call.

    Example:
        ```python
        sp = calibrate.SitePoints('sasmas', start=date(2003, 1, 1), end=date(2006, 12, 31))
        paired, scores = sp.evaluate(Params())
        ```
    """

    def __init__(s, site: str, settings: Terrasim = Terrasim(), start: date = None, end: date = None, observations: pd.DataFrame = None,
                 max_span_m: float = 2500.0, config=None, email=None, api_key=None, verbose: bool = True):
        o = observations if observations is not None else obs.load([site])
        o = o[o.site == site]
        latest = date.today() - timedelta(days=120)
        s.start = start or o.time.min().date()
        s.end = min(end or o.time.max().date(), latest)
        o = o[(o.time >= pd.Timestamp(s.start)) & (o.time <= pd.Timestamp(s.end))]
        s.site, s.settings, s.config, s.api_key, s.verbose = site, settings, config, api_key, verbose
        s.obs = o.reset_index(drop=True)
        s.stations = obs.stations(s.obs)
        s.groups = groups(s.stations, max_span_m=max_span_m)
        s.forcings, s.pixels, s.names = [], [], []
        for g in s.groups:
            f = forcing_for(g['bbox'], s.start, s.end, settings, config=config, email=email, verbose=verbose)
            sub = s.stations[s.stations.station.isin(g['stations'])]
            iy, ix = station_pixels(grid.model_window(g['bbox']), sub.lon.values, sub.lat.values)
            s.forcings.append(f); s.pixels.append((iy, ix)); s.names += sub.station.tolist()
        s.use_ndvi = s.start.year >= SENTINEL2_FROM
        s._static_cache = {}

    def arrays(s, params: Params):
        key = _key(attrs.asdict(params.soil), attrs.asdict(params.veg), list(params.slga_attributes), params.pedotransfer_version)
        if key not in s._static_cache:
            parts = []
            for g, f, (iy, ix) in zip(s.groups, s.forcings, s.pixels):
                static = static_for(g['bbox'], s.settings, params, config=s.config, api_key=s.api_key)
                parts.append(point_subset(static, remap_vegetation(f, params.veg, s.use_ndvi), iy, ix))
            st, f = merge_points(parts)
            s._static_cache = {key: (st, f)}          # keep only the latest
            s._layer_bounds = (static['layer'].values, static['layer_top_m'].values, static['layer_bottom_m'].values)
        return s._static_cache[key]

    def run(s, params: Params = Params()) -> xr.Dataset:
        st, f = s.arrays(params)
        return _run_point_arrays(Model(settings=s.settings, params=params), st, f, s.names, s._layer_bounds)

    def evaluate(s, params: Params = Params()):
        pts = s.run(params)
        paired = pair(pts, s.obs)
        return paired, (score(paired) if len(paired) else pd.DataFrame())

    def sweep(s, param_sets: dict, out_dir: str = None) -> pd.DataFrame:
        """Evaluate several labelled ``Params``; per-run outputs are written to ``out_dir`` as they
        finish (restartable). Returns the stacked per-station scores with a ``run`` column."""
        frames = []
        for label, params in param_sets.items():
            if out_dir and os.path.exists(f'{out_dir}/{s.site}_{label}_scores.csv'):
                sc = pd.read_csv(f'{out_dir}/{s.site}_{label}_scores.csv'); sc['run'] = label; frames.append(sc); continue
            t0 = time.time()
            paired, sc = s.evaluate(params)
            sc['run'] = label
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                paired.to_parquet(f'{out_dir}/{s.site}_{label}_paired.parquet'); sc.to_csv(f'{out_dir}/{s.site}_{label}_scores.csv', index=False)
            if s.verbose:
                print(f'calibrate: {s.site} {label} in {time.time() - t0:.0f}s | median kge {sc.kge.median():.2f} r {sc.r.median():.2f} ubrmse {sc.ubrmse.median():.3f} bias {sc.bias.median():+.3f}', flush=True)
            frames.append(sc)
        return pd.concat(frames, ignore_index=True)


def _run_point_arrays(model: Model, st: StaticArrays, f: Forcing, names, layer_bounds) -> xr.Dataset:
    settings = attrs.evolve(model.settings, routing='none')
    engine = Engine(st, settings, model.params)
    state = State.initial(st, settings.init_theta)
    spd = settings.steps_per_day; nday = f.nday; L = st.nlayer; n = st.shape[1]
    for t in range(min(settings.spinup_days, nday) * spd):
        engine.step(state, f.step(t), None)
    theta = np.zeros((nday, L, n), dtype='float32')
    flux = {k: np.zeros((nday, n), dtype='float32') for k in POINT_FLUXES}
    acc = Fluxes.zeros(L, 1, n)
    for d in range(nday):
        for t in range(d * spd, (d + 1) * spd):
            acc.add_(engine.step(state, f.step(t), None))
        theta[d] = state.theta[:, 0, :]
        for k in POINT_FLUXES:
            flux[k][d] = (acc.et_actual if k == 'et_actual' else getattr(acc, k))[0]
        acc = Fluxes.zeros(L, 1, n)
    layer, top, bot = layer_bounds
    coords = {'time': f.days, 'layer': layer, 'point': list(names), 'layer_top_m': ('layer', top), 'layer_bottom_m': ('layer', bot)}
    data = {'theta': (('time', 'layer', 'point'), theta)}
    data.update({k: (('time', 'point'), v) for k, v in flux.items()})
    return xr.Dataset(data, coords=coords)


# -- parameter sweeps -----------------------------------------------------------------

def sweep(site: str, param_sets: dict, settings: Terrasim = Terrasim(), observations: pd.DataFrame = None, out_dir: str = None,
          verbose: bool = True, **kw) -> pd.DataFrame:
    """Evaluate several ``Params`` at one site; returns the per-station scores stacked with a
    ``run`` column. Static datasets and forcing are cached, so after the first set each run is
    engine time only. ``param_sets`` maps a run label to a :class:`Params`. With ``out_dir``
    each run's paired table and scores are written as they finish (restartable)."""
    frames = []
    for label, params in param_sets.items():
        if out_dir and os.path.exists(f'{out_dir}/{site}_{label}_scores.csv'):
            sc = pd.read_csv(f'{out_dir}/{site}_{label}_scores.csv'); sc['run'] = label; frames.append(sc)
            continue
        t0 = time.time()
        paired, sc = evaluate(site, settings, params, observations=observations, verbose=False, **kw)
        sc['run'] = label
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            paired.to_parquet(f'{out_dir}/{site}_{label}_paired.parquet'); sc.to_csv(f'{out_dir}/{site}_{label}_scores.csv', index=False)
        if verbose:
            s = summarise(sc)
            print(f'calibrate: {site} {label} in {time.time() - t0:.0f}s | median kge {sc.kge.median():.2f} ubrmse {sc.ubrmse.median():.3f} bias {sc.bias.median():+.3f}', flush=True)
        frames.append(sc)
    return pd.concat(frames, ignore_index=True)


def morris_sets(space: dict, n_paths: int = 6, levels: int = 4, seed: int = 0):
    """Morris one-at-a-time trajectories over ``space = {param: (lo, hi, 'lin'|'log')}``; returns
    ``(list of dict samples, list of (path, step, param) tags)``. ``n_paths * (k + 1)`` samples."""
    rng = np.random.default_rng(seed)
    names = list(space); k = len(names)
    grid_vals = np.linspace(0, 1, levels)
    delta = levels / (2 * (levels - 1))
    samples, tags = [], []
    for p in range(n_paths):
        base = rng.choice(grid_vals[: levels // 2], size=k)
        order = rng.permutation(k)
        x = base.copy(); samples.append(x.copy()); tags.append((p, 0, None))
        for step, j in enumerate(order, 1):
            x = x.copy(); x[j] = x[j] + delta if x[j] + delta <= 1 else x[j] - delta
            samples.append(x.copy()); tags.append((p, step, names[j]))
    out = []
    for x in samples:
        d = {}
        for u, name in zip(x, names):
            lo, hi, kind = space[name]
            d[name] = float(lo * (hi / lo) ** u) if kind == 'log' else float(lo + (hi - lo) * u)
        out.append(d)
    return out, tags


def params_from(sample: dict) -> Params:
    """``Params`` from a flat dict of SoilParams / VegParams / SurfaceParams field names."""
    from terrasim.params import SoilParams, VegParams
    from terrasim.physics.surface import SurfaceParams
    soil = {k: v for k, v in sample.items() if k in {f.name for f in attrs.fields(SoilParams)}}
    veg = {k: v for k, v in sample.items() if k in {f.name for f in attrs.fields(VegParams)}}
    surf = {k: v for k, v in sample.items() if k in {f.name for f in attrs.fields(SurfaceParams)}}
    unknown = set(sample) - set(soil) - set(veg) - set(surf)
    if unknown:
        raise KeyError(f'not parameters: {sorted(unknown)}')
    return Params(soil=SoilParams(**soil), veg=VegParams(**veg), surface=SurfaceParams(**surf))


def morris_effects(scores_by_run: pd.DataFrame, tags, metric: str = 'kge') -> pd.DataFrame:
    """Elementary effects of each parameter on the site-median ``metric``: mean |EE| (mu*) and
    std over trajectories. ``scores_by_run`` is the ``sweep`` output with runs labelled ``m{i}``."""
    med = scores_by_run.groupby('run')[metric].median()
    rows = []
    for i in range(1, len(tags)):
        p, step, name = tags[i]
        if name is None:
            continue
        rows.append(dict(param=name, path=p, ee=float(med.get(f'm{i}', np.nan) - med.get(f'm{i - 1}', np.nan))))
    ee = pd.DataFrame(rows)
    return ee.groupby('param').ee.agg(mu_star=lambda s: np.nanmean(np.abs(s)), mu=np.nanmean, sigma=np.nanstd, n='count').sort_values('mu_star', ascending=False).reset_index()


# -- tests -----------------------------------------------------------------------

def test_remap_vegetation():
    from terrasim.params import VegParams
    from terrasim.vegetation import fc_from_kcb
    base = VegParams(); kcb = np.linspace(base.kcb_min, base.kcb_max, 5, dtype='float32').reshape(5, 1, 1)
    f = attrs.make_class('F', ['kcb', 'fc'])(kcb=kcb, fc=fc_from_kcb(kcb, base).astype('float32'))
    new = VegParams(kcb_max=0.8, kcb_constant=0.3)
    with_ndvi = remap_vegetation(f, new, use_ndvi=True)
    without = remap_vegetation(f, new, use_ndvi=False)
    ends = abs(with_ndvi.kcb.min() - new.kcb_min) < 1e-6 and abs(with_ndvi.kcb.max() - 0.8) < 1e-6
    linear = np.allclose(np.diff(with_ndvi.kcb.ravel()), np.diff(with_ndvi.kcb.ravel())[0], atol=1e-6)
    const = np.allclose(without.kcb, 0.3) and np.allclose(without.fc, fc_from_kcb(np.float32(0.3), new))
    return ends and linear and const


def test_morris_and_params():
    space = {'seal_ksat_mm_h': (2.0, 50.0, 'log'), 'kcb_max': (0.8, 1.2, 'lin'), 'pond_max_mm': (0.5, 5.0, 'lin')}
    samples, tags = morris_sets(space, n_paths=3, levels=4, seed=1)
    right_count = len(samples) == 3 * 4 and tags[0] == (0, 0, None) and all(t[2] in space for t in tags if t[1] > 0)
    in_range = all(2.0 - 1e-9 <= s['seal_ksat_mm_h'] <= 50.0 + 1e-9 and 0.8 <= s['kcb_max'] <= 1.2 for s in samples)
    one_change = all(sum(abs(a[k] - b[k]) > 1e-12 for k in space) == 1 for a, b, t in zip(samples, samples[1:], tags[1:]) if t[1] > 0)
    p = params_from(samples[0])
    typed = p.surface.seal_ksat_mm_h == samples[0]['seal_ksat_mm_h'] and p.veg.kcb_max == samples[0]['kcb_max'] and p.soil.pond_max_mm == samples[0]['pond_max_mm']
    try:
        params_from({'nope': 1}); bad = False
    except KeyError:
        bad = True
    fake = pd.DataFrame({'run': [f'm{i}' for i in range(len(samples)) for _ in range(2)], 'kge': np.repeat([s['kcb_max'] for s in samples], 2)})
    eff = morris_effects(fake, tags)
    kcb_dominates = eff.iloc[0]['param'] == 'kcb_max' and eff.set_index('param').loc['seal_ksat_mm_h', 'mu_star'] < 1e-12
    return right_count and in_range and one_change and typed and bad and kcb_dominates


def test_groups_and_pixels():
    st = pd.DataFrame({'site': ['a'] * 3, 'station': ['s1', 's2', 's3'], 'lon': [150.0, 150.001, 150.1], 'lat': [-32.0, -32.0, -32.0],
                       'start': pd.to_datetime(['2003-01-01'] * 3), 'end': pd.to_datetime(['2004-01-01'] * 3)})
    gs = groups(st, max_span_m=1000)
    two_groups = len(gs) == 2 and gs[0]['stations'] == ['s1', 's2'] and gs[1]['stations'] == ['s3']
    w = grid.model_window(gs[0]['bbox'])
    iy, ix = station_pixels(w, [150.0, 150.001], [-32.0, -32.0])
    ny, nx = grid.shape(w)
    inside = (0 <= iy).all() and (iy < ny).all() and (0 <= ix).all() and (ix < nx).all() and ix[1] > ix[0]
    return two_groups and inside


def test_pair_thickness_weights_and_kge():
    t = pd.date_range('2022-01-01', periods=3)
    theta = np.array([[[0.1], [0.3], [0.5]]] * 3, dtype='float32')              # (time, layer, point)
    pts = xr.Dataset({'theta': (('time', 'layer', 'point'), theta)},
                     coords={'time': t, 'layer': [0, 1, 2], 'point': ['s1'], 'layer_top_m': ('layer', [0.0, 0.1, 0.3]), 'layer_bottom_m': ('layer', [0.1, 0.3, 0.6])})
    o = pd.DataFrame({'site': 'x', 'station': 's1', 'lon': 0.0, 'lat': 0.0, 'time': t, 'depth_top_m': 0.0, 'depth_bottom_m': 0.3, 'vwc': [0.2, 0.25, 0.3], 'n': 1})
    p = pair(pts, o)
    # 0-30 cm = 1/3 of layer 0 (0.1) + 2/3 of layer 1 (0.3) = 0.2333
    weighted = np.allclose(p['mod'], 0.1 / 3 + 0.3 * 2 / 3)
    deep = pair(pts, o.assign(depth_top_m=1.0, depth_bottom_m=1.2))
    extends = np.allclose(deep['mod'], 0.5)
    perfect = abs(kge([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    sc = score(p)
    return weighted and extends and perfect and len(sc) == 1 and sc.depth.iloc[0] == '0-30cm'


def test():
    return all([test_groups_and_pixels(), test_pair_thickness_weights_and_kge(), test_morris_and_params(), test_remap_vegetation()])


if __name__ == '__main__':
    print(test())
