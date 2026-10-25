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
    settings = attrs.evolve(model.settings, routing='none')
    engine = Engine(st, settings, model.params)
    state = State.initial(st, settings.init_theta)
    spd = settings.steps_per_day; nday = f.nday; L = st.nlayer; n = st.shape[1]
    if f.nstep != nday * spd:
        raise ValueError(f'forcing has {f.nstep} steps for {nday} days but step={settings.step!r} needs {nday * spd}')
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
    names = list(names) if names is not None else [f'p{i}' for i in range(n)]
    coords = {'time': f.days, 'layer': static_ds['layer'].values, 'point': names,
              'layer_top_m': static_ds['layer_top_m'], 'layer_bottom_m': static_ds['layer_bottom_m']}
    data = {'theta': (('time', 'layer', 'point'), theta)}
    data.update({k: (('time', 'point'), v) for k, v in flux.items()})
    return xr.Dataset(data, coords=coords)


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


def forcing_for(bbox, start: date, end: date, settings: Terrasim, params: Params, config=None, email=None, verbose=True) -> Forcing:
    """Forcing for a group bbox, cached on disk (it does not depend on soil or surface
    parameters, only on the step settings and the vegetation mapping)."""
    paths = Paths(config) if config is not None else Paths()
    use_ndvi = start.year >= SENTINEL2_FROM
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
        forcing = forcing_for(g['bbox'], g_start, g_end, settings, params, config=config, email=email, verbose=verbose)
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


# -- tests -----------------------------------------------------------------------

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
    return all([test_groups_and_pixels(), test_pair_thickness_weights_and_kge()])


if __name__ == '__main__':
    print(test())
