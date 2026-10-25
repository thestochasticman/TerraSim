"""Soil-moisture observations from the lab's dense validation sites, in one table.

Every loader returns the same long format, one row per station, day and depth
interval, so the calibration code never looks at a site's raw layout:

====================  ==========================================================
column                meaning
====================  ==========================================================
``site``              'sasmas' | 'mri' | 'llara' | 'esdale' | 'phenode' | 'tarrawarra'
``station``           station / logger / probe / point id (string)
``lon``, ``lat``      WGS84
``time``              calendar date (daily mean for continuous loggers)
``depth_top_m``       top of the depth interval the value represents
``depth_bottom_m``    bottom of that interval
``vwc``               volumetric water content, m3 m-3 (0-1)
``n``                 number of raw values behind the daily mean (1 for campaigns)
====================  ==========================================================

Depth intervals: CS616 rods (SASMAS) integrate over their 30 cm length;
capacitance sensors (MRI, Llara) are taken as +-5 cm around the nominal depth;
handheld and vertical TDR campaigns (Esdale, Tarrawarra) integrate over the
rod length. Values are converted from percent to fraction and screened to
0 < vwc < 0.7 (anything else is a sensor fault code).

Site data root: the folder the user downloaded from Drive ("Dense_sites"),
``obs.DEFAULT_ROOT`` or ``TERRASIM_OBS_ROOT``. Nothing here fetches anything.

Caveats recorded per site in the loader docstrings; the Llara channel-depth
and device-location mappings are inferred from naming patterns, not documented.
"""
import os
import re
from datetime import date

import numpy as np
import pandas as pd

DEFAULT_ROOT = os.environ.get('TERRASIM_OBS_ROOT', os.path.expanduser('~/Downloads/Dense_sites'))
COLUMNS = ['site', 'station', 'lon', 'lat', 'time', 'depth_top_m', 'depth_bottom_m', 'vwc', 'n']


def _finish(df: pd.DataFrame, site: str) -> pd.DataFrame:
    df = df.copy()
    df['site'] = site
    df['time'] = pd.to_datetime(df['time']).dt.normalize()
    df['vwc'] = pd.to_numeric(df['vwc'], errors='coerce')
    df = df[(df['vwc'] > 0) & (df['vwc'] < 0.7)]
    df['station'] = df['station'].astype(str)
    return df[COLUMNS].sort_values(['station', 'depth_top_m', 'time']).reset_index(drop=True)


# -- SASMAS -----------------------------------------------------------------------

def sasmas(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """SASMAS (Goulburn/Krui/Merriwa/Stanley, NSW, 2003-2006): the prepared daily root-zone
    series, the mean of the available CS616 layers (0-30, 30-60, 60-90 cm). ``n_layers`` tells
    how many layers went into the mean; the interval is taken as 0 to ``0.3 * n_layers`` m,
    which assumes layers fail from the bottom up. Per-layer series would need the raw zips."""
    d = pd.read_csv(f'{root}/SASMAS/observations/sasmas_daily_rootzone.csv', parse_dates=['time'])
    c = pd.read_csv(f'{root}/SASMAS/observations/sasmas_coordinates.csv')[['station', 'lat', 'lon']]
    d = d.merge(c, on='station', how='inner')
    d['depth_top_m'] = 0.0
    d['depth_bottom_m'] = 0.3 * d['n_layers'].clip(1, 3)
    d['vwc'] = d['sm_rootzone_pct'] / 100.0
    d['n'] = d['n_subdaily']
    return _finish(d, 'sasmas')


# -- MRI (Mulloon) ----------------------------------------------------------------

def mri(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Mulloon Rehydration Initiative, NSW (Oct 2020 to Jun 2026): 18 loggers, 6 sensors per
    profile (10-150 cm), 30-min, converted to daily means in local time (UTC+10). Coordinates
    come from the logger crosswalk built from ``SM_metadata.pdf`` and the GeoPackage probe layer."""
    import glob
    cands = [f for f in glob.glob(f'{root}/MRI_data/SM_combined_cleaned/SM(*)_combined_cleaned.csv') if '(mm)' not in f]   # '(%)' or '(_)' after a Drive download
    if not cands:
        raise FileNotFoundError(f'no SM(%)_combined_cleaned.csv under {root}/MRI_data/SM_combined_cleaned')
    wide = pd.read_csv(cands[0], na_values=['NA'])
    wide['time'] = pd.to_datetime(wide['Date Time'], utc=True).dt.tz_convert('Etc/GMT-10').dt.tz_localize(None)
    long = wide.drop(columns=['Date Time']).melt(id_vars='time', var_name='col', value_name='pct').dropna(subset=['pct'])
    parts = long['col'].str.extract(r'^(\d+)_SM_(\d+)cm$')
    long['station'] = parts[0].str.lstrip('0')          # '05893' -> '5893', as the metadata table names them; survives CSV round trips
    long['depth_cm'] = parts[1].astype(float)
    long = long[(long['pct'] > 0) & (long['pct'] < 70)]
    long['day'] = long['time'].dt.normalize()
    daily = long.groupby(['station', 'depth_cm', 'day']).agg(pct=('pct', 'mean'), n=('pct', 'size')).reset_index()
    cw = pd.read_csv(f'{root}/MRI_data/mri_logger_coordinate_crosswalk.csv', dtype={'logger': str})
    cw['station'] = cw['logger'].astype(str).str.lstrip('0')
    daily = daily.merge(cw[['station', 'lon', 'lat']], on='station', how='inner')
    daily['time'] = daily['day']
    daily['depth_top_m'] = (daily['depth_cm'] - 5.0).clip(lower=0) / 100.0
    daily['depth_bottom_m'] = (daily['depth_cm'] + 5.0) / 100.0
    daily['vwc'] = daily['pct'] / 100.0
    return _finish(daily, 'mri')


# -- Llara (Narrabri) ---------------------------------------------------------------

_LLARA_DEVICE = re.compile(r'^(WW|WE) (\d\d)-(\d\d) ')


def llara(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Llara, Narrabri NSW (Oct 2021 to Jun 2024): two paddocks (WW, WE), capacitance probes
    with 15 channels. INFERRED, not documented: device ``'WW 12-05 95'`` is probe location
    ``'WW MP 1205'`` (paddock, probe length in dm, probe number); channel ``v_k`` sits at
    ``10 (k-1)`` cm, so 1.2 m probes have no signal on v13-v16 (they read 0 and are dropped).
    Whether values are calibrated VWC or raw sensor percent is unknown; values above 70 are
    dropped. Coordinates are MGA zone 55 (EPSG:28355)."""
    from pyproj import Transformer
    locs = pd.read_csv(f'{root}/Llara_data/sm_probe_locs.csv')
    tr = Transformer.from_crs('EPSG:28355', 'EPSG:4326', always_xy=True)
    locs['lon'], locs['lat'] = tr.transform(locs['Easting'].values, locs['Northing'].values)
    locs['key'] = locs['Name'].str.replace(' MP ', ' ', regex=False)          # 'WW 1205'
    frames = []
    for pad in ('WW', 'WE'):
        d = pd.read_csv(f'{root}/Llara_data/{pad}_cleanedData_20241022.csv', parse_dates=['Date']).dropna(subset=['soil_moisture'])
        m = d['device'].str.extract(_LLARA_DEVICE)
        d['key'] = m[0] + ' ' + m[1] + m[2]
        d['probe_len_m'] = m[1].astype(float) / 10.0
        d['depth_cm'] = (d['channel'].str[1:].astype(int) - 1) * 10.0
        d = d[(d['depth_cm'] / 100.0 < d['probe_len_m']) & (d['soil_moisture'] > 0) & (d['soil_moisture'] < 70)]
        frames.append(d)
    d = pd.concat(frames).merge(locs[['key', 'lon', 'lat']], on='key', how='inner')
    d['station'] = d['key'].str.replace(' ', '_')
    d['time'] = d['Date']
    d['depth_top_m'] = (d['depth_cm'] - 5.0).clip(lower=0) / 100.0
    d['depth_bottom_m'] = (d['depth_cm'] + 5.0) / 100.0
    d['vwc'] = d['soil_moisture'] / 100.0
    d['n'] = 1
    return _finish(d, 'llara')


# -- Esdale -------------------------------------------------------------------------

def esdale(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Esdale (near Murrumbateman NSW), Apr-Jul 2025: 79 handheld TDR points on nine dates,
    7.6 cm rods, so the interval is 0 to 0.076 m."""
    e = pd.read_csv(f'{root}/Esdale/observations/esdale_observations.csv', parse_dates=['date'])
    e = e.rename(columns={'point_id': 'station', 'date': 'time'})
    e['depth_top_m'] = 0.0
    e['depth_bottom_m'] = 0.076
    e['vwc'] = e['obs_sm_pct'] / 100.0
    e['n'] = 1
    return _finish(e, 'esdale')


def phenode(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Esdale Phenode wireless soil sensors, daily, Jun 2025 to Jul 2026. Sensor depth is not
    documented; taken as 0 to 0.1 m. Coordinates are the per-device medians from the export
    inventory."""
    p = pd.read_csv(f'{root}/Esdale/Phenode/observations/phenode_observations.csv', parse_dates=['time']).dropna(subset=['sm_pct'])
    inv = pd.read_csv(f'{root}/Esdale/Phenode/export_inventory_2026-10-07.csv')
    inv = inv.dropna(subset=['latitude_median']).drop_duplicates('file')
    inv['device'] = inv['file'].str.replace('.csv', '', regex=False)
    p = p.merge(inv[['device', 'latitude_median', 'longitude_median']], on='device', how='inner')
    p = p.rename(columns={'device': 'station', 'latitude_median': 'lat', 'longitude_median': 'lon'})
    p['time'] = p['time'].dt.tz_localize(None)
    p['depth_top_m'] = 0.0
    p['depth_bottom_m'] = 0.1
    p['vwc'] = p['sm_pct'] / 100.0
    p['n'] = 1
    return _finish(p, 'phenode')


# -- Tarrawarra ---------------------------------------------------------------------

def tarrawarra(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Tarrawarra, Victoria, Sep 1995 to Nov 1996 (Western & Grayson): vertical 30 cm TDR on a
    regular grid, 19 campaign dates; interval 0 to 0.3 m."""
    t = pd.read_csv(f'{root}/Tarrawarra/observations/tarrawarra_tdr_observations.csv', parse_dates=['date'])
    t = t.rename(columns={'point_id': 'station', 'date': 'time'})
    t['depth_top_m'] = 0.0
    t['depth_bottom_m'] = t['depth_cm'].fillna(30.0) / 100.0
    t['vwc'] = t['obs_sm_pct'] / 100.0
    t['n'] = 1
    return _finish(t, 'tarrawarra')


LOADERS = {'sasmas': sasmas, 'mri': mri, 'llara': llara, 'esdale': esdale, 'phenode': phenode, 'tarrawarra': tarrawarra}


def load(sites=None, root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """All (or the named) sites in the common format."""
    names = list(LOADERS) if sites is None else list(sites)
    return pd.concat([LOADERS[n](root) for n in names], ignore_index=True)


def stations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per site/station: coordinates, record span, depth intervals and row count."""
    g = df.groupby(['site', 'station'])
    out = g.agg(lon=('lon', 'first'), lat=('lat', 'first'), start=('time', 'min'), end=('time', 'max'), rows=('vwc', 'size'),
                depths=('depth_bottom_m', lambda s: sorted(set(np.round(s, 3)))))
    return out.reset_index()


def station_bbox(lon: float, lat: float, half_width_m: float = 150.0) -> list:
    """``[W, S, E, N]`` box of ``2 * half_width_m`` around a point."""
    dlat = half_width_m / 111_320.0
    dlon = half_width_m / (111_320.0 * np.cos(np.radians(lat)))
    return [float(lon - dlon), float(lat - dlat), float(lon + dlon), float(lat + dlat)]


# -- tests -----------------------------------------------------------------------

def _synthetic_root(tmp):
    """A minimal Dense_sites layout with two tiny files, to test the schema without real data."""
    os.makedirs(f'{tmp}/SASMAS/observations', exist_ok=True)
    pd.DataFrame({'site': ['S', 'S'], 'station': ['SASMAS_K1', 'SASMAS_K1'], 'time': ['2003-01-01', '2003-01-02'],
                  'sm_rootzone_pct': [20.0, 150.0], 'n_layers': [3, 3], 'n_subdaily': [72, 72]}).to_csv(f'{tmp}/SASMAS/observations/sasmas_daily_rootzone.csv', index=False)
    pd.DataFrame({'station': ['SASMAS_K1'], 'lat': [-32.1], 'lon': [150.07]}).to_csv(f'{tmp}/SASMAS/observations/sasmas_coordinates.csv', index=False)
    os.makedirs(f'{tmp}/Tarrawarra/observations', exist_ok=True)
    pd.DataFrame({'point_id': ['p1'], 'date': ['1996-05-02'], 'lon': [145.43], 'lat': [-37.65], 'obs_sm_pct': [45.9], 'depth_cm': [30.0]}).to_csv(
        f'{tmp}/Tarrawarra/observations/tarrawarra_tdr_observations.csv', index=False)
    return tmp


def test_schema_and_screening():
    import tempfile
    root = _synthetic_root(tempfile.mkdtemp(prefix='terrasim_obs_'))
    df = load(['sasmas', 'tarrawarra'], root)
    ok_cols = list(df.columns) == COLUMNS
    screened = len(df[df.site == 'sasmas']) == 1 and abs(df[df.site == 'sasmas'].vwc.iloc[0] - 0.20) < 1e-12
    depth = abs(df[df.site == 'sasmas'].depth_bottom_m.iloc[0] - 0.9) < 1e-12 and abs(df[df.site == 'tarrawarra'].depth_bottom_m.iloc[0] - 0.3) < 1e-12
    st = stations(df)
    box = station_bbox(150.07, -32.1, 150.0)
    return ok_cols and screened and depth and len(st) == 2 and box[0] < 150.07 < box[2] and box[1] < -32.1 < box[3]


def test():
    return test_schema_and_screening()


if __name__ == '__main__':
    print(test())
