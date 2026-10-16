"""Sentinel-2 NDVI -> daily FAO-56 basal crop coefficient and cover fraction.

NDVI observations are irregular (5-day revisit, cloud gaps). They are
despiked with a rolling median over observations, linearly interpolated to
a daily axis (edges held), smoothed with a Savitzky-Golay filter, and mapped
linearly to ``Kcb`` between bare soil and full canopy. The cover fraction
follows FAO-56 eq. 76. Everything is vectorised over pixels with xarray.
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter

from terrasim.params import VegParams


def daily_ndvi(ndvi: xr.DataArray, start, end, despike_window: int = 3, smooth_days: int = 31,
               max_gap_warn_days: int = 60) -> xr.DataArray:
    """Daily, gap-filled, smoothed NDVI ``(time, y, x)`` for ``[start, end]``."""
    da = ndvi.where((ndvi > -0.2) & (ndvi <= 1.0))
    da = da.dropna('time', how='all')
    if da.sizes['time'] == 0:
        raise ValueError('no valid NDVI observations')
    if despike_window > 1 and da.sizes['time'] >= despike_window:
        da = da.rolling(time=despike_window, center=True, min_periods=1).median()
    gaps = np.diff(da['time'].values).astype('timedelta64[D]').astype(int)
    if gaps.size and gaps.max() > max_gap_warn_days:
        warnings.warn(f'NDVI gap of {gaps.max()} days; Kcb is interpolated across it')
    daily = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq='D')
    full = pd.DatetimeIndex(np.union1d(daily.values, da['time'].values))
    da = da.reindex(time=full).interpolate_na('time', method='linear')
    if bool(da.isnull().any()):   # leading/trailing gaps: hold the nearest observation
        da = da.interpolate_na('time', method='nearest', fill_value='extrapolate')
    da = da.sel(time=daily)
    if smooth_days > 2 and da.sizes['time'] > smooth_days:
        win = smooth_days if smooth_days % 2 else smooth_days + 1
        vals = savgol_filter(da.values, win, 2, axis=0, mode='nearest')
        da = da.copy(data=vals)
    return da.clip(-0.2, 1.0).astype('float32')


def kcb_from_ndvi(ndvi, veg: VegParams = VegParams()):
    """Linear NDVI -> Kcb between (ndvi_soil, kcb_min) and (ndvi_veg, kcb_max)."""
    frac = (np.asarray(ndvi, dtype=float) - veg.ndvi_soil) / (veg.ndvi_veg - veg.ndvi_soil)
    return np.clip(veg.kcb_min + (veg.kcb_max - veg.kcb_min) * frac, veg.kcb_min, veg.kcb_max)


def fc_from_kcb(kcb, veg: VegParams = VegParams()):
    """FAO-56 eq. 76: fc = ((Kcb - Kc_min) / (Kc_max - Kc_min)) ** (1 + 0.5 h)."""
    ratio = np.clip((np.asarray(kcb, dtype=float) - veg.kcb_min) / max(veg.kc_max - veg.kcb_min, 1e-9), 0.0, 1.0)
    return np.clip(ratio ** (1.0 + 0.5 * veg.h_crop_m), 0.0, 0.99)


def crop_coefficients(ndvi: xr.DataArray, start, end, veg: VegParams = VegParams()):
    """``(kcb, fc)`` daily float32 arrays ``(nday, y, x)`` from raw NDVI observations."""
    nd = daily_ndvi(ndvi, start, end)
    kcb = kcb_from_ndvi(nd.values, veg).astype('float32')
    return kcb, fc_from_kcb(kcb, veg).astype('float32'), nd['time'].values


# -- tests -----------------------------------------------------------------------

def _synthetic_ndvi(nobs=20, ny=2, nx=2, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range('2022-01-03', periods=nobs, freq='5D')
    doy = t.dayofyear.values
    curve = 0.2 + 0.5 * np.sin(np.pi * (doy - doy.min()) / (doy.max() - doy.min()))
    data = np.broadcast_to(curve[:, None, None], (nobs, ny, nx)).copy()
    data[3] = np.nan                       # a cloudy date
    data[7, 0, 0] = 0.95                   # a spike
    return xr.DataArray(data.astype('float32'), dims=('time', 'y', 'x'), coords={'time': t})


def test_daily_series_covers_range_without_nans():
    da = daily_ndvi(_synthetic_ndvi(), '2022-01-01', '2022-04-30')
    return da.sizes['time'] == 120 and not np.isnan(da.values).any() and float(da.max()) <= 1.0


def test_kcb_fc_bounds_and_monotone():
    k = kcb_from_ndvi(np.array([0.0, 0.5, 1.0]))
    f = fc_from_kcb(k)
    return k[0] == 0.15 and k[-1] == 1.10 and np.all(np.diff(k) > 0) and np.all(np.diff(f) > 0) and f[0] == 0.0


def test():
    return all([test_daily_series_covers_range_without_nans(), test_kcb_fc_bounds_and_monotone()])


if __name__ == '__main__':
    print(test())
