"""Assemble everything that drives the model into one :class:`Forcing`.

Sources (all through the lab's stores):

- SILO daily climate at every 0.05 deg point under the grid (one for a
  paddock; several for a large bbox) — ``daily_rain`` (the conserved daily
  total), ``et_short_crop`` (FAO-56 reference ET), temperatures, humidity,
  radiation and pressure for the DAESIM export;
- BARRA-C2 hourly rain for the cells under the grid — within-day timing;
- Sentinel-2 NDVI on the model grid — Kcb and cover fraction.

Rain and ET0 are kept *factored*: ``rain[step, cell]`` and
``et0[step, point]`` with ``(ny, nx)`` index maps, instead of full
``(step, y, x)`` cubes (1.4 GB per hourly year per variable for a 2 km
bbox). Vegetation is daily ``(day, y, x)``. :func:`assemble` is a pure
function of already-fetched inputs so it can be tested offline;
:class:`ForcingBuilder` does the fetching.

Time axis: local standard time (``round(centre_lon/15)`` hours, no DST,
matching SILO and BoM) from ``start`` 00:00 to ``end`` 23:xx. SILO's
09:00-09:00 day is honoured by :mod:`terrasim.disaggregate`.
"""
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from attrs import frozen, field

from troi import Config, config as default_config
from terrasim import grid
from terrasim.disaggregate import BarraWeights, METHODS as DISAGG, silo_day_of, hour_in_silo_day, hourly_to_steps
from terrasim.params import Params
from terrasim.terrasim import Terrasim, defaultterrasim
from terrasim.physics.engine import StepForcing

SILO_DAILY_VARS = ('daily_rain', 'et_short_crop', 'et_morton_potential', 'max_temp', 'min_temp',
                   'radiation', 'vp', 'vp_deficit', 'rh_tmax', 'rh_tmin', 'mslp')


@frozen
class Forcing:
    """Model forcing on a local-time step axis.

    Attributes:
        time: Local standard time of each step start, ``(nstep,)`` datetime64.
        day_of_step: Index into the daily arrays for each step, ``(nstep,)``.
        days: Calendar dates of the daily axis, ``(nday,)``.
        silo_days: Dates of the SILO daily table rows (``start .. end+1``).
        rain: mm per step per BARRA cell, ``(nstep, ncell)``.
        rain_source_barra: ``(nday_silo, ncell)`` bool: BARRA timing used (else fallback).
        silo_wet: ``(nday_silo, ncell)`` bool: the SILO day had rain at the cell's point.
        et0: mm per step per SILO point, ``(nstep, npoint)``.
        daily: SILO daily table per point, dict var -> ``(nday, npoint)``.
        cell_index, point_index: ``(ny, nx)`` int maps into the cell / point axes (nearest).
        cell_weights, point_weights: ``(ncell, ny, nx)`` / ``(npoint, ny, nx)`` inverse-distance
            blending weights (each pixel's column sums to 1), so coarse-cell boundaries do not
            print through as seams; ``step()`` uses these.
        kcb, fc: daily ``(nday, ny, nx)`` float32.
        tz_hours: local standard time offset used.
    """

    time: np.ndarray
    day_of_step: np.ndarray
    days: np.ndarray
    silo_days: np.ndarray
    rain: np.ndarray
    rain_source_barra: np.ndarray
    silo_wet: np.ndarray
    et0: np.ndarray
    daily: dict
    cell_index: np.ndarray
    point_index: np.ndarray
    cell_weights: np.ndarray
    point_weights: np.ndarray
    kcb: np.ndarray
    fc: np.ndarray
    tz_hours: float

    @property
    def nstep(s) -> int:
        return len(s.time)

    @property
    def nday(s) -> int:
        return len(s.days)

    @property
    def barra_fraction_wet_days(s) -> float:
        """Share of rainy SILO days whose timing came from BARRA-C2 (not a fallback)."""
        wet = s.silo_wet
        return float(s.rain_source_barra[wet].mean()) if wet.any() else float('nan')

    def step(s, t: int) -> StepForcing:
        """Broadcast step ``t`` onto the grid."""
        d = s.day_of_step[t]
        return StepForcing(rain=np.tensordot(s.rain[t], s.cell_weights, axes=(0, 0)),
                           et0=np.tensordot(s.et0[t], s.point_weights, axes=(0, 0)),
                           kcb=s.kcb[d].astype(float), fc=s.fc[d].astype(float))

    def rain_grid_daily(s) -> np.ndarray:
        """Daily rain on the grid ``(nday, ny, nx)`` (for outputs/validation)."""
        daily_cell = np.add.reduceat(s.rain, np.flatnonzero(np.r_[True, np.diff(s.day_of_step) != 0]), axis=0)
        return np.tensordot(daily_cell, s.cell_weights, axes=(1, 0))


# -- pure pieces ------------------------------------------------------------------

def et0_diurnal_weights(days: pd.DatetimeIndex, lat_deg: float, lon_deg: float, tz_hours: float,
                        steps_per_day: int) -> np.ndarray:
    """``(nday, steps_per_day)`` weights: half-sine over daylight, zero at night,
    each row summing to 1. Sunrise/sunset from declination and hour angle;
    solar noon shifted by the longitude offset from the time-zone meridian."""
    doy = days.dayofyear.values
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    phi = np.radians(lat_deg)
    cos_ws = np.clip(-np.tan(phi) * np.tan(decl), -1, 1)
    ws = np.arccos(cos_ws)                       # half day length, radians
    daylen = 24 * ws / np.pi
    noon = 12.0 - (lon_deg - 15.0 * tz_hours) / 15.0
    sunrise, sunset = noon - daylen / 2, noon + daylen / 2
    dt = 24.0 / steps_per_day
    mid = (np.arange(steps_per_day) + 0.5) * dt
    x = (mid[None, :] - sunrise[:, None]) / np.maximum(sunset - sunrise, 1e-6)[:, None]
    w = np.clip(np.sin(np.pi * np.clip(x, 0, 1)), 0, None)
    w = np.where((x > 0) & (x < 1), w, 0.0)
    tot = w.sum(axis=1, keepdims=True)
    return np.where(tot > 0, w / np.where(tot > 0, tot, 1), 1.0 / steps_per_day)


def silo_points_for_window(window) -> tuple[list[str], np.ndarray, list[tuple[float, float]]]:
    """Unique SILO point ids under the window and the ``(ny, nx)`` index map."""
    from pysilo import grid as sgrid
    lat, lon = grid.pixel_latlon(window)
    ids = np.vectorize(sgrid.point_id)(lat, lon)
    uniq, inverse = np.unique(ids, return_inverse=True)
    coords = [tuple(float(v) for v in pid.split(',')) for pid in uniq]
    return list(uniq), inverse.reshape(lat.shape), coords


def blend_weights(window, src_lat: np.ndarray, src_lon: np.ndarray, power: float = 2.0,
                  scale_deg: float | None = None) -> np.ndarray:
    """Inverse-distance weights ``(nsrc, ny, nx)`` of coarse points/cells for every
    pixel, each pixel's weights summing to 1. Pixels much closer to one source
    than ``scale_deg`` (default: the source spacing) take it almost entirely, so
    with a single source this is the identity and with several the field varies
    smoothly instead of jumping at cell boundaries."""
    lat, lon = grid.pixel_latlon(window)
    coslat = np.cos(np.radians(lat))
    d2 = (lat[None] - src_lat[:, None, None]) ** 2 + ((lon[None] - src_lon[:, None, None]) * coslat[None]) ** 2
    if len(src_lat) == 1:
        return np.ones((1,) + lat.shape)
    if scale_deg is None:
        pts = np.column_stack([src_lat, src_lon])
        dd = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)); dd[dd == 0] = np.inf
        scale_deg = float(dd.min())
    w = 1.0 / (d2 + (0.1 * scale_deg) ** 2) ** (power / 2)
    return w / w.sum(axis=0, keepdims=True)


def cell_index_for_window(window, cell_lat: np.ndarray, cell_lon: np.ndarray) -> np.ndarray:
    """Nearest BARRA cell (by centre) for each pixel, ``(ny, nx)``."""
    lat, lon = grid.pixel_latlon(window)
    d2 = (lat[..., None] - cell_lat) ** 2 + ((lon[..., None] - cell_lon) * np.cos(np.radians(lat[..., None]))) ** 2
    return d2.argmin(axis=-1)


def extraterrestrial_radiation(doy, lat_deg):
    """FAO-56 eq. 21: daily extraterrestrial radiation ``Ra``, MJ m-2 d-1."""
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    phi = np.radians(lat_deg)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws))


def priestley_taylor_et0(df: pd.DataFrame, lat_deg: float, elevation_m: float = 300.0, alpha: float = 1.26,
                         albedo: float = 0.23) -> pd.Series:
    """Daily Priestley-Taylor reference ET (mm d-1) from SILO radiation (MJ m-2 d-1),
    Tmax/Tmin (degC) and vapour pressure (hPa), with FAO-56 net radiation."""
    t = 0.5 * (df['max_temp'] + df['min_temp'])
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    delta = 4098 * es / (t + 237.3) ** 2
    p = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
    gamma = 0.000665 * p
    lam = 2.45
    ea = df['vp'] / 10.0
    ra = extraterrestrial_radiation(pd.to_datetime(df['date']).dt.dayofyear.values, lat_deg)
    rso = (0.75 + 2e-5 * elevation_m) * ra
    rs = df['radiation'].values
    tmaxk, tmink = df['max_temp'].values + 273.16, df['min_temp'].values + 273.16
    rnl = 4.903e-9 * (tmaxk ** 4 + tmink ** 4) / 2 * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0, None))) * \
        (1.35 * np.clip(rs / np.maximum(rso, 1e-6), 0.3, 1.0) - 0.35)
    rn = (1 - albedo) * rs - rnl
    et0 = alpha * (delta / (delta + gamma)) * np.clip(rn, 0, None) / lam
    return pd.Series(np.asarray(et0, dtype=float), index=df.index)


def assemble(window, start: date, end: date, silo: dict, point_index: np.ndarray, barra: xr.Dataset | None,
             ndvi: xr.DataArray | None, settings: Terrasim = defaultterrasim, params: Params = Params(),
             centre_lat: float = -33.5, centre_lon: float = 148.4, et0_source: str = 'silo') -> Forcing:
    """Pure assembly of a :class:`Forcing` from fetched inputs.

    Args:
        silo: ``{point_id: DataFrame}`` with a ``date`` column and SILO variables,
            covering ``start - 1 day .. end + 1 day`` (missing days become 0 rain
            / NaN else, with a warning).
        point_index: ``(ny, nx)`` map into the sorted ``silo`` keys.
        barra: :meth:`terrasim.barra.Store.get_ds` output (UTC hourly) or ``None``
            (then rain timing is uniform/climatological per ``settings``).
        ndvi: raw Sentinel-2 NDVI ``(time, y, x)`` on the model grid, or ``None``
            (then a constant Kcb of 0.5 is used).
    """
    tz = settings.tz(centre_lon)
    spd = settings.steps_per_day
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq='D')
    # extended hourly local axis covering every SILO day the run touches in full — 09:00 on
    # start-1 (SILO day `start` begins) to 08:00 on end+1 (SILO day `end+1` ends) — so the
    # BARRA weights are normalised over complete 24 h windows; SILO days then run start .. end+1
    ext_start = pd.Timestamp(start) - pd.Timedelta(days=1) + pd.Timedelta(hours=9)
    ext_end = pd.Timestamp(end) + pd.Timedelta(days=1, hours=8)
    hours = pd.date_range(ext_start, ext_end, freq='h')
    silo_day = silo_day_of(hours)
    hour_idx = hour_in_silo_day(hours)
    silo_days = pd.DatetimeIndex(np.unique(silo_day))

    # -- SILO daily tables per point ------------------------------------------
    pids = sorted(silo)
    daily = {}
    for var in SILO_DAILY_VARS:
        cols = []
        for pid in pids:
            df = silo[pid]
            if var not in df.columns:
                cols.append(np.full(len(silo_days), np.nan)); continue
            ser = pd.Series(df[var].values, index=pd.to_datetime(df['date']).values.astype('datetime64[D]'))
            cols.append(ser.reindex(silo_days.values.astype('datetime64[D]')).values.astype(float))
        daily[var] = np.column_stack(cols)
    rain_daily = daily['daily_rain']
    if np.isnan(rain_daily).any():
        warnings.warn('SILO daily_rain missing for some days (archive lag?); treated as 0 mm')
        rain_daily = np.nan_to_num(rain_daily)

    # -- BARRA weights per cell -------------------------------------------------
    if barra is not None and barra.sizes['cell'] > 0:
        utc = hours - pd.Timedelta(hours=tz)
        hourly = barra['pr_mm'].reindex(time=utc.values).values          # (nhour, ncell), NaN where missing
        cell_lat, cell_lon = barra['cell_lat'].values, barra['cell_lon'].values
    else:
        hourly = np.full((len(hours), 1), np.nan)
        cell_lat, cell_lon = np.array([centre_lat]), np.array([centre_lon])
    ncell = hourly.shape[1]
    disagg = DISAGG[settings.disaggregation]()
    weights = np.zeros_like(hourly, dtype=float)
    used = np.zeros_like(hourly, dtype=bool)
    for c in range(ncell):
        weights[:, c], used[:, c] = disagg.weights(hourly[:, c], hour_idx, silo_day)
    cell_index = cell_index_for_window(window, cell_lat, cell_lon)
    cell_weights = blend_weights(window, cell_lat, cell_lon)

    # every pixel's rain comes from its SILO point (daily total) and its BARRA cell (weights).
    # rain[hour, cell] needs a daily total per cell: use the SILO point nearest the cell centre.
    pt_coords = np.array([[float(v) for v in pid.split(',')] for pid in pids])     # (P, 2) lat, lon
    point_weights = blend_weights(window, pt_coords[:, 0], pt_coords[:, 1])
    cell_point = np.array([np.argmin((pt_coords[:, 0] - la) ** 2 + (pt_coords[:, 1] - lo) ** 2)
                           for la, lo in zip(cell_lat, cell_lon)])
    day_pos = pd.Index(silo_days.values.astype('datetime64[D]')).get_indexer(silo_day)
    rain_hourly = weights * rain_daily[day_pos][:, cell_point]                      # (nhour, ncell)
    used_daily = pd.DataFrame(used).groupby(silo_day).any().values
    silo_wet = rain_daily[:, cell_point] > 0

    # -- ET0 per point -----------------------------------------------------------
    if et0_source == 'priestley_taylor':
        et0_daily = np.column_stack([priestley_taylor_et0(silo[pid].assign(date=pd.to_datetime(silo[pid]['date'])), centre_lat)
                                     .set_axis(pd.to_datetime(silo[pid]['date']).values.astype('datetime64[D]'))
                                     .reindex(days.values.astype('datetime64[D]')).values for pid in pids])
    else:
        src = 'et_short_crop' if not np.isnan(daily['et_short_crop']).all() else 'et_morton_potential'
        pos = pd.Index(silo_days.values.astype('datetime64[D]')).get_indexer(days.values.astype('datetime64[D]'))
        et0_daily = daily[src][pos]
    et0_daily = np.nan_to_num(et0_daily)
    wts = et0_diurnal_weights(days, centre_lat, centre_lon, tz, spd)               # (nday, spd)
    et0 = (et0_daily[:, None, :] * wts[:, :, None]).reshape(len(days) * spd, len(pids))

    # -- to model steps, trimmed to [start, end] -----------------------------------
    keep = (hours >= pd.Timestamp(start)) & (hours < pd.Timestamp(end) + pd.Timedelta(days=1))
    rain_h = rain_hourly[keep]
    rain = hourly_to_steps(rain_h, settings.steps_per_hour, int(round(settings.dt_hours)) if settings.dt_hours > 1 else 1,
                           within_hour=settings.within_hour, peak_fraction=settings.peak_fraction)
    time = pd.date_range(pd.Timestamp(start), periods=len(days) * spd, freq=pd.Timedelta(hours=settings.dt_hours)).values
    day_of_step = np.repeat(np.arange(len(days)), spd)

    # -- vegetation ----------------------------------------------------------------
    ny, nx = grid.shape(window)
    if ndvi is not None:
        from terrasim.vegetation import crop_coefficients
        kcb, fc, _ = crop_coefficients(ndvi, start, end, params.veg)
    else:
        kcb = np.full((len(days), ny, nx), params.veg.kcb_constant, dtype='float32')
        from terrasim.vegetation import fc_from_kcb
        fc = fc_from_kcb(kcb, params.veg).astype('float32')

    return Forcing(time=time, day_of_step=day_of_step, days=days.values, silo_days=silo_days.values, rain=rain,
                   rain_source_barra=used_daily, silo_wet=silo_wet, et0=et0, daily=daily, cell_index=cell_index,
                   point_index=point_index, cell_weights=cell_weights.astype('float32'),
                   point_weights=point_weights.astype('float32'), kcb=kcb, fc=fc, tz_hours=tz)


# -- the fetching builder ---------------------------------------------------------

@frozen
class ForcingBuilder:
    """Fetch SILO, BARRA-C2 and Sentinel-2 NDVI through the lab stores, then :func:`assemble`."""

    config: Config = default_config
    settings: Terrasim = defaultterrasim
    params: Params = Params()
    et0_source: str | None = None      # 'silo' | 'priestley_taylor'; None follows settings.et_method
    use_ndvi: bool = True
    use_barra: bool = True

    @property
    def resolved_et0_source(s) -> str:
        if s.et0_source is not None:
            return s.et0_source
        return 'priestley_taylor' if s.settings.et_method == 'priestley_taylor' else 'silo'

    def build(s, bbox: list[float], start: date, end: date, email: str = None) -> Forcing:
        from pysilo.store import Store as SiloStore
        window = grid.model_window(bbox)
        pids, point_index, coords = silo_points_for_window(window)
        silo_store = SiloStore(config=s.config)
        silo = {pid: silo_store.get_df(la, lo, start - timedelta(days=1), end + timedelta(days=1), email=email)
                for pid, (la, lo) in zip(pids, coords)}
        barra = None
        if s.use_barra and s.settings.disaggregation == 'barra':
            from terrasim.barra import Store as BarraStore
            try:
                barra = BarraStore(config=s.config).get_ds(bbox, start - timedelta(days=2), end + timedelta(days=2))
            except Exception as ex:   # noqa: BLE001
                warnings.warn(f'BARRA-C2 unavailable ({type(ex).__name__}: {ex}); using fallback rain timing')
        ndvi = None
        if s.use_ndvi:
            from pysentinel2.cube import Cube
            ds = Cube(config=s.config).get_ds(bbox, start - timedelta(days=60), end + timedelta(days=30),
                                             clean=True, indices=('NDVI',))
            ndvi = ds['NDVI']
            y, x = grid.coords(window)
            if ndvi.sizes['y'] != len(y) or ndvi.sizes['x'] != len(x):
                ndvi = ndvi.reindex(y=y, x=x, method='nearest')
        centre_lat, centre_lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
        return assemble(window, start, end, silo, point_index, barra, ndvi, s.settings, s.params,
                        centre_lat, centre_lon, s.resolved_et0_source)

    def build_troi(s, troi, email: str = None) -> Forcing:
        return s.build(troi.bbox, troi.start, troi.end, email=email)


# -- tests -----------------------------------------------------------------------

def _synthetic_inputs(start=date(2022, 1, 1), end=date(2022, 1, 10), bbox=None):
    bbox = bbox or [148.36265, -33.52606, 148.38265, -33.50606]
    window = grid.model_window(bbox)
    pids, point_index, coords = silo_points_for_window(window)
    days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=1), pd.Timestamp(end) + pd.Timedelta(days=1), freq='D')
    rng = np.random.default_rng(1)
    silo = {}
    for pid in pids:
        silo[pid] = pd.DataFrame({'date': days, 'daily_rain': rng.choice([0.0, 12.0], len(days)),
                                  'et_short_crop': 5.0, 'et_morton_potential': 6.0, 'max_temp': 30.0, 'min_temp': 15.0,
                                  'radiation': 25.0, 'vp': 12.0, 'vp_deficit': 20.0, 'rh_tmax': 30.0, 'rh_tmin': 80.0,
                                  'mslp': 1013.0})
    utc = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=2), pd.Timestamp(end) + pd.Timedelta(days=2, hours=23), freq='h')
    pr = np.zeros((len(utc), 1)); pr[utc.hour == 4, 0] = 3.0      # 04 UTC = 14:00 AEST daily shower
    barra = xr.Dataset({'pr_mm': (('time', 'cell'), pr)},
                       coords={'time': utc, 'cell': [0], 'cell_lat': ('cell', [-33.51]), 'cell_lon': ('cell', [148.37]),
                               'cell_iy': ('cell', [0]), 'cell_ix': ('cell', [0])})
    return window, silo, point_index, barra


def test_rain_conserves_silo_daily_totals():
    start, end = date(2022, 1, 1), date(2022, 1, 10)
    window, silo, point_index, barra = _synthetic_inputs(start, end)
    f = assemble(window, start, end, silo, point_index, barra, None, centre_lat=-33.516, centre_lon=148.373)
    hours = pd.DatetimeIndex(f.time)
    sd = silo_day_of(hours)
    sums = pd.Series(f.rain[:, 0]).groupby(sd).sum()
    df = next(iter(silo.values())).set_index('date')['daily_rain']
    # complete SILO days inside the window: 2022-01-02 .. 2022-01-10
    ok = all(abs(sums[pd.Timestamp(d)] - df[pd.Timestamp(d)]) < 1e-9 for d in pd.date_range('2022-01-02', '2022-01-10'))
    # BARRA shower at 14:00 local -> rain lands in the 14:00 step
    wet_steps = hours[f.rain[:, 0] > 0]
    return ok and set(wet_steps.hour.tolist()) == {14} and f.rain_source_barra.all()


def test_et0_weights_sum_to_one_and_zero_at_night():
    days = pd.date_range('2022-01-01', '2022-01-03')
    w = et0_diurnal_weights(days, -33.5, 148.4, 10.0, 24)
    return np.allclose(w.sum(axis=1), 1) and np.all(w[:, :4] == 0) and np.all(w[:, 22:] == 0) and w[:, 12].max() > 0.05


def test_forcing_step_broadcasts_to_grid():
    start, end = date(2022, 1, 1), date(2022, 1, 3)
    window, silo, point_index, barra = _synthetic_inputs(start, end)
    f = assemble(window, start, end, silo, point_index, barra, None, centre_lat=-33.516, centre_lon=148.373)
    sf = f.step(14)
    ny, nx = grid.shape(window)
    return sf.rain.shape == (ny, nx) and sf.et0.shape == (ny, nx) and sf.kcb.shape == (ny, nx) and f.nstep == 72


def test_blend_weights_sum_to_one_and_are_smooth():
    window = grid.model_window([148.36265, -33.52606, 148.38265, -33.50606])
    w1 = blend_weights(window, np.array([-33.51]), np.array([148.37]))
    w2 = blend_weights(window, np.array([-33.51, -33.51]), np.array([148.35, 148.39]))
    col = w2[0, 0, :]                                        # west cell's weight along a row
    return (np.allclose(w1, 1.0) and np.allclose(w2.sum(axis=0), 1.0) and col[0] > col[-1]
            and np.abs(np.diff(col)).max() < 0.02)           # no jump at the midline


def test_twenty_minute_step_conserves():
    start, end = date(2022, 1, 1), date(2022, 1, 3)
    window, silo, point_index, barra = _synthetic_inputs(start, end)
    f1 = assemble(window, start, end, silo, point_index, barra, None, Terrasim(step='1h'), centre_lat=-33.516, centre_lon=148.373)
    f3 = assemble(window, start, end, silo, point_index, barra, None, Terrasim(step='20min'), centre_lat=-33.516, centre_lon=148.373)
    f8 = assemble(window, start, end, silo, point_index, barra, None, Terrasim(step='3h'), centre_lat=-33.516, centre_lon=148.373)
    return abs(f1.rain.sum() - f3.rain.sum()) < 1e-9 and abs(f1.rain.sum() - f8.rain.sum()) < 1e-9 and f3.nstep == 216 and f8.nstep == 24


def test_peaked_within_hour_conserves_and_sharpens():
    start, end = date(2022, 1, 1), date(2022, 1, 3)
    window, silo, point_index, barra = _synthetic_inputs(start, end)
    kw = dict(centre_lat=-33.516, centre_lon=148.373)
    f1 = assemble(window, start, end, silo, point_index, barra, None, Terrasim(step='1h'), **kw)
    fu = assemble(window, start, end, silo, point_index, barra, None, Terrasim(step='20min'), **kw)
    fp = assemble(window, start, end, silo, point_index, barra, None, Terrasim(step='20min', within_hour='peaked'), **kw)
    hourly_from_peaked = fp.rain.reshape(-1, 3, fp.rain.shape[1]).sum(axis=1)
    conserved = np.allclose(hourly_from_peaked, f1.rain) and abs(fp.rain.sum() - fu.rain.sum()) < 1e-9
    # the synthetic shower is a single 12 mm hour: uniform gives 4 mm per step, peaked 7.2 mm in one step
    sharpened = abs(fp.rain.max() - 0.6 * f1.rain.max()) < 1e-9 and abs(fu.rain.max() - f1.rain.max() / 3) < 1e-9
    return conserved and sharpened


def test():
    return all([test_rain_conserves_silo_daily_totals(), test_et0_weights_sum_to_one_and_zero_at_night(),
                test_forcing_step_broadcasts_to_grid(), test_blend_weights_sum_to_one_and_are_smooth(),
                test_twenty_minute_step_conserves(), test_peaked_within_hour_conserves_and_sharpens()])


if __name__ == '__main__':
    print(test())
