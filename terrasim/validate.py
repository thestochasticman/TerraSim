"""Compare a terrasim run with independent soil-moisture observations.

- **OzNet** (Murrumbidgee in-situ network): 20/30-minute layered soil
  moisture (0-30, 30-60, 60-90 cm). Parsed with ``emt.insitu.oznet.parse_xls``
  from the lab's DownscalingMoistureModel repo when installed; the model
  column (pixel nearest the station, or the area mean) is re-layered onto
  the OzNet depths and scored.
- **OzWALD Ssoil** (8-day, ~500 m): correlation of the run's domain-mean
  column storage with OzWALD's soil water storage, as a coarse sanity check.

Metrics follow ``emt.evaluation.metrics`` (RMSE, unbiased RMSE, bias, r,
Nash-Sutcliffe efficiency), re-implemented here so validation does not
pull in scikit-learn.
"""
import numpy as np
import pandas as pd
import xarray as xr

from terrasim.export import layers_of, select_area
from terrasim.layers import LayerSpec, relayer

OZNET_LAYERS = LayerSpec(nlayer=3, z_max=0.9, method='horizon', z_horizon=(0.3, 0.3, 0.3))
OZNET_COLUMNS = ('SM 0-30cm', 'SM 30-60cm', 'SM 60-90cm')


def metrics(y_true, y_pred) -> dict:
    """rmse, ubrmse, bias (pred - obs), r, nse, n — as in emt.evaluation.metrics."""
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    n = y_true.size
    if n < 2:
        return dict(rmse=np.nan, ubrmse=np.nan, bias=np.nan, r=np.nan, nse=np.nan, n=n)
    err = y_pred - y_true
    bias = float(err.mean()); rmse = float(np.sqrt((err ** 2).mean()))
    ubrmse = float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0)))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 and y_pred.std() > 0 else np.nan
    ss_res = float(((y_true - y_pred) ** 2).sum()); ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(rmse=rmse, ubrmse=ubrmse, bias=bias, r=r, nse=nse, n=n)


def oznet_daily(xls_path: str) -> pd.DataFrame:
    """Daily mean volumetric moisture (m3 m-3) per OzNet layer from a station .xls."""
    try:
        from emt.insitu.oznet import parse_xls, MISSING_FLAG
    except ImportError as ex:
        raise ImportError('OzNet validation needs the DownscalingMoistureModel repo installed as `emt` '
                          '(pip install -e ../DownscalingMoistureModel)') from ex
    df = parse_xls(xls_path)
    cols = [c for c in OZNET_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError(f'{xls_path} has none of {OZNET_COLUMNS}')
    # keep every OzNet layer as a column (NaN where the station lacks it) so the re-layered model lines up
    sm = df[cols].replace(MISSING_FLAG, np.nan).astype(float).reindex(columns=list(OZNET_COLUMNS))
    if sm.max().max() > 1.5:
        sm = sm / 100.0          # percent -> fraction
    sm.index = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df.iloc[:, 0])
    return sm.resample('D').mean()


def compare_profile(model_theta_daily: np.ndarray, model_days, src: LayerSpec, obs_daily: pd.DataFrame,
                    obs_layers: LayerSpec = OZNET_LAYERS) -> pd.DataFrame:
    """Score the model (re-layered onto the observation depths) per layer."""
    th = relayer(model_theta_daily, src, obs_layers, axis=1)
    model = pd.DataFrame(th, index=pd.DatetimeIndex(model_days), columns=list(obs_daily.columns)[:obs_layers.nlayer])
    both = model.join(obs_daily, how='inner', lsuffix='_model', rsuffix='_obs')
    rows = []
    for col in model.columns:
        m = metrics(both[f'{col}_obs'], both[f'{col}_model'])
        rows.append({'layer': col, **m})
    return pd.DataFrame(rows)


def compare_oznet(run_ds: xr.Dataset, xls_path: str, where=('mean',), static: xr.Dataset = None) -> pd.DataFrame:
    ds = run_ds if 'window' in run_ds.attrs or static is None else run_ds.assign_attrs(window=static.attrs['window'])
    col = select_area(ds, where)
    return compare_profile(col['theta'].values, run_ds['time'].values, layers_of(run_ds), oznet_daily(xls_path))


def compare_ozwald_ssoil(run_ds: xr.Dataset, lat: float, lon: float, config=None) -> dict:
    """Pearson r between domain-mean storage (mm) and OzWALD 8-day Ssoil at the point."""
    from datetime import date
    from pyozwald.store import Store
    from troi import config as default_config
    store = Store(config=config or default_config)
    days = pd.to_datetime(run_ds['time'].values)
    df = store.get_df(lat, lon, days[0].date(), days[-1].date(), cadence='8day', variables=['Ssoil'])
    obs = pd.Series(df['Ssoil'].values, index=pd.to_datetime(df['time']))
    model = pd.Series(run_ds['storage_mm'].mean(('y', 'x')).values, index=days)
    both = pd.concat([model.rename('model'), obs.rename('obs')], axis=1).dropna()
    return {**metrics(both['obs'], both['model']), 'note': 'OzWALD Ssoil is a different quantity (total column store); compare dynamics (r), not levels'}


# -- tests ---------------------------------------------------------------------------

def test_metrics_perfect_and_biased():
    y = np.array([0.2, 0.3, 0.4, 0.3])
    perfect = metrics(y, y)
    biased = metrics(y, y + 0.05)
    return perfect['rmse'] == 0 and perfect['nse'] == 1 and abs(biased['bias'] - 0.05) < 1e-12 and biased['ubrmse'] < 1e-9


def test_compare_profile_relayers_and_scores():
    src = LayerSpec()
    days = pd.date_range('2022-01-01', periods=10)
    theta = np.tile(np.array([0.30, 0.30, 0.30, 0.25, 0.25, 0.20]), (10, 1))
    obs = pd.DataFrame({c: 0.3 for c in OZNET_COLUMNS}, index=days)
    out = compare_profile(theta, days.values, src, obs)
    # layer 0-30 cm exactly matches (0.30); 30-60 is 0.25 -> bias -0.05
    r0 = out[out.layer == 'SM 0-30cm'].iloc[0]; r1 = out[out.layer == 'SM 30-60cm'].iloc[0]
    return abs(r0.rmse) < 1e-12 and abs(r1.bias + 0.05) < 1e-12 and (out.n == 10).all()


def test():
    return all([test_metrics_perfect_and_biased(), test_compare_profile_relayers_and_scores()])


if __name__ == '__main__':
    print(test())
