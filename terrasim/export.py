"""Export a terrasim run as DAESIM2 forcing.

DAESIM2 (``daesim2_analysis.forcing_data.ForcingDataX``, "type 3") reads a
daily CSV whose soil-moisture columns are named
``Soil moisture interp (a)-(b)`` with the depth range in cm, plus
meteorology. terrasim's per-layer moisture is re-layered onto the target
DAESIM discretisation (default: DAESIM's own 5 x 0.2 m) with
thickness-weighted averaging, and the meteorology comes from the SILO
daily table carried in the :class:`terrasim.forcing.Forcing`.

Also exported: the soil hydraulic parameters DAESIM's ``SoilLayers`` takes
(``soilThetaMax``, ``Psi_e`` in MPa, ``b_soil``, ``K_sat`` in
mol m-1 s-1 MPa-1), averaged over the exported area and re-layered.

Units flagged for confirmation against the DAESIM reader: radiation is a
daily mean in W m-2; CO2/O2 are given in ppm; wind speed defaults to a
constant when no series is supplied.
"""
import json

import numpy as np
import pandas as pd
import xarray as xr

from terrasim import grid
from terrasim.layers import LayerSpec, DAESIM_DEFAULT, relayer
from terrasim.pedotransfer import to_daesim_units

MJ_PER_DAY_TO_W = 1e6 / 86400.0


def layers_of(ds: xr.Dataset) -> LayerSpec:
    """Reconstruct the model LayerSpec from a run/static dataset's layer coords."""
    thick = ds['layer_bottom_m'].values - ds['layer_top_m'].values
    return LayerSpec(nlayer=len(thick), z_max=float(ds['layer_bottom_m'].values[-1]), method='horizon',
                     z_horizon=tuple(float(t) for t in thick))


def select_area(ds: xr.Dataset, where) -> xr.Dataset:
    """Reduce ``(y, x)`` to a single column.

    ``where`` is ``('mean',)``, ``('pixel', lat, lon)`` or a boolean ``(y, x)`` mask.
    """
    if isinstance(where, np.ndarray):
        m = xr.DataArray(where, dims=('y', 'x'))
        return ds.where(m).mean(('y', 'x'))
    if where[0] == 'mean':
        return ds.mean(('y', 'x'))
    if where[0] == 'pixel':
        _, lat, lon = where
        window = tuple(json.loads(ds.attrs['window'])) if 'window' in ds.attrs else None
        if window is None:
            raise ValueError("dataset lacks a 'window' attr; pass a static dataset or a mask")
        plat, plon = grid.pixel_latlon(window)
        iy, ix = np.unravel_index(np.argmin((plat - lat) ** 2 + (plon - lon) ** 2), plat.shape)
        return ds.isel(y=iy, x=ix)
    raise ValueError(f'unknown where {where!r}')


def paddock_masks(gpkg_path: str, static: xr.Dataset, label_col: str = None) -> dict:
    """Rasterise polygons onto the model grid: ``{label: (y, x) bool mask}``."""
    import geopandas as gpd
    from rasterio.features import rasterize
    window = tuple(json.loads(static.attrs['window']))
    gdf = gpd.read_file(gpkg_path).to_crs(grid.CRS)
    labels = gdf[label_col].astype(str).tolist() if label_col else [str(i + 1) for i in range(len(gdf))]
    shape = grid.shape(window)
    out = {}
    for lab, geom in zip(labels, gdf.geometry):
        out[lab] = rasterize([(geom, 1)], out_shape=shape, transform=grid.transform(window), fill=0, dtype='uint8').astype(bool)
    return out


def extraterrestrial_radiation(doy, lat_deg):
    """FAO-56 eq. 21, MJ m-2 d-1."""
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    phi = np.radians(lat_deg)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws))


def erbs_diffuse_fraction(kt):
    kt = np.asarray(kt, dtype=float)
    fd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4, 0.165))
    return np.clip(fd, 0.0, 1.0)


def daesim_forcing_frame(theta_daily: np.ndarray, days, src: LayerSpec, target: LayerSpec, met: pd.DataFrame,
                         lat_deg: float, elevation_m: float = 300.0, wind=None, co2_ppm: float = 400.0,
                         o2_ppm: float = 209460.0) -> pd.DataFrame:
    """Build the DAESIM type-3 forcing table.

    Args:
        theta_daily: ``(nday, nlayer_src)`` moisture; ``days``: datetime64 (nday,).
        met: daily SILO rows indexed by date with ``max_temp, min_temp, rh_tmax, rh_tmin,
            radiation, mslp`` (missing columns are filled with NaN).
    """
    th = relayer(theta_daily, src, target, axis=1)
    days = pd.DatetimeIndex(days)
    df = pd.DataFrame({'Date': days.strftime('%Y-%m-%d'), 'DOY': days.dayofyear, 'Year': days.year})
    met = met.reindex(days)
    rs = met.get('radiation', pd.Series(np.nan, index=days)).values.astype(float)
    ra = extraterrestrial_radiation(days.dayofyear.values, lat_deg)
    fd = erbs_diffuse_fraction(np.clip(rs / np.maximum(ra, 1e-6), 0, 1))
    df['Downwelling shortwave beam radiation'] = (1 - fd) * rs * MJ_PER_DAY_TO_W
    df['Downwelling shortwave diffuse radiation'] = fd * rs * MJ_PER_DAY_TO_W
    mslp = met.get('mslp', pd.Series(1013.25, index=days)).fillna(1013.25).values
    df['Atmospheric pressure'] = mslp * 100.0 * np.exp(-elevation_m / 8434.0)
    rh = np.nanmean(np.column_stack([met.get('rh_tmax', pd.Series(np.nan, index=days)).values,
                                     met.get('rh_tmin', pd.Series(np.nan, index=days)).values]), axis=1)
    df['Relative humidity'] = rh
    df['Wind speed'] = np.asarray(wind, dtype=float) if wind is not None else 2.0
    df['Atmospheric carbon dioxide concentration'] = co2_ppm
    df['Atmospheric oxygen concentration'] = o2_ppm
    df['Minimum temperature'] = met.get('min_temp', pd.Series(np.nan, index=days)).values
    df['Maximum temperature'] = met.get('max_temp', pd.Series(np.nan, index=days)).values
    for j, (a, b) in enumerate(target.bounds_cm()):
        df[f'Soil moisture interp {a}-{b}'] = th[:, j]
    return df


def met_table_from_forcing(forcing, which: str = 'centre') -> pd.DataFrame:
    """Daily SILO table (one point: the first/only) indexed by date."""
    idx = pd.DatetimeIndex(forcing.silo_days)
    p = 0
    return pd.DataFrame({k: v[:, p] for k, v in forcing.daily.items()}, index=idx)


def export_daesim_forcing(run_ds: xr.Dataset, forcing, out_csv: str, static: xr.Dataset = None,
                          where=('mean',), target: LayerSpec = DAESIM_DEFAULT, wind=None, co2_ppm: float = 400.0,
                          o2_ppm: float = 209460.0) -> pd.DataFrame:
    """Write the DAESIM forcing CSV for an area of a run and return the frame."""
    src = layers_of(run_ds)
    ds = run_ds if 'window' in run_ds.attrs or static is None else run_ds.assign_attrs(window=static.attrs['window'])
    col = select_area(ds, where)
    theta = col['theta'].values                    # (nday, L)
    lat = float(static['lat'].mean()) if static is not None else -33.5
    elev = float(static['elevation'].mean()) if static is not None else 300.0
    df = daesim_forcing_frame(theta, run_ds['time'].values, src, target, met_table_from_forcing(forcing), lat, elev,
                              wind=wind, co2_ppm=co2_ppm, o2_ppm=o2_ppm)
    df.to_csv(out_csv, index=False)
    return df


def daesim_soil_params(static: xr.Dataset, target: LayerSpec = DAESIM_DEFAULT, mask=None) -> dict:
    """DAESIM ``SoilLayers`` hydraulic parameters for the area (mean), on ``target`` layers."""
    src = layers_of(static)
    sel = static if mask is None else static.where(xr.DataArray(mask, dims=('y', 'x')))
    mean = lambda n: sel[n].mean(('y', 'x')).values
    theta_s = relayer(mean('theta_s'), src, target)
    b = relayer(mean('b'), src, target)
    psi_e_mm = relayer(mean('psi_e'), src, target)
    ksat = relayer(mean('ksat'), src, target)
    K_sat, Psi_e = to_daesim_units(ksat, psi_e_mm)
    return {**target.to_daesim_kwargs(),
            'soilThetaMax': [float(v) for v in theta_s], 'b_soil': [float(v) for v in b],
            'Psi_e': [float(v) for v in Psi_e], 'K_sat': [float(v) for v in K_sat],
            'units': {'soilThetaMax': 'm3 m-3', 'Psi_e': 'MPa', 'K_sat': 'mol m-1 s-1 MPa-1'},
            'source': 'terrasim static dataset (SLGA v2 + pedotransfer)'}


def write_daesim_soil_params(static: xr.Dataset, out_json: str, target: LayerSpec = DAESIM_DEFAULT, mask=None) -> dict:
    d = daesim_soil_params(static, target, mask)
    with open(out_json, 'w') as f:
        json.dump(d, f, indent=2)
    return d


# -- tests ---------------------------------------------------------------------------

def test_frame_columns_and_water_conservation():
    import re
    src = LayerSpec()
    days = pd.date_range('2022-01-01', periods=3)
    theta = np.array([[0.40, 0.35, 0.30, 0.28, 0.25, 0.22]] * 3)
    met = pd.DataFrame({'max_temp': 30.0, 'min_temp': 15.0, 'rh_tmax': 30.0, 'rh_tmin': 80.0, 'radiation': 25.0, 'mslp': 1013.0}, index=days)
    df = daesim_forcing_frame(theta, days.values, src, DAESIM_DEFAULT, met, -33.5, 300.0)
    sm_cols = [c for c in df.columns if c.startswith('Soil moisture interp')]
    pattern = re.compile(r'Soil moisture interp (\d+)-(\d+)')
    starts = [int(pattern.match(c).group(1)) for c in sm_cols]
    water_src = float(np.sum(theta[0, :5] * src.thicknesses()[:5]))
    water_dst = float(np.sum(df[sm_cols].iloc[0].values * DAESIM_DEFAULT.thicknesses()))
    rad_ok = abs(df['Downwelling shortwave beam radiation'][0] + df['Downwelling shortwave diffuse radiation'][0] - 25 * MJ_PER_DAY_TO_W) < 1e-9
    return (all(pattern.match(c) for c in sm_cols) and starts == sorted(starts) and len(sm_cols) == 5
            and abs(water_src - water_dst) < 1e-9 and rad_ok and 90000 < df['Atmospheric pressure'][0] < 101325)


def test():
    return test_frame_columns_and_water_conservation()


if __name__ == '__main__':
    print(test())
