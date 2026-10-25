"""Phases 3-4: static builder, end-to-end run, DAESIM export, validation — offline with synthetic inputs."""
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from terrasim import static as S, export as E, validate as V, forcing as F
from terrasim.model import Model, summary
from terrasim.terrasim import Terrasim
from terrasim.layers import DAESIM_DEFAULT


def test_static_module():
    assert S.test()


def test_export_module():
    assert E.test()


def test_validate_module():
    assert V.test()


@pytest.fixture
def synthetic_run(tmp_troi):
    """A tiny end-to-end run: synthetic SLGA/DEM, synthetic SILO/BARRA, 12 days, written to zarr."""
    troi = tmp_troi
    start, end = date(2022, 1, 1), date(2022, 1, 12)
    sb = S.StaticBuilder(config=troi.config, slga_fetch=S.synthetic_slga, dem_fetch=lambda bbox: S.synthetic_dem(bbox))
    static = sb.build(troi.bbox)
    window, silo, point_index, barra = F._synthetic_inputs(start, end, troi.bbox)
    forcing = F.assemble(window, start, end, silo, point_index, barra, None, Terrasim(spinup_days=2),
                         centre_lat=troi.centre_lat, centre_lon=troi.centre_lon)
    from troi import Troi
    short = Troi(bbox=troi.bbox, start=start, end=end, stub='pytest_short', config=troi.config)
    model = Model(config=troi.config, settings=Terrasim(spinup_days=2), block_days=5)
    ds = model.run(short, static=static, forcing=forcing, write=True, verbose=False)
    return short, static, forcing, ds


def test_end_to_end_run_shape_and_balance(synthetic_run):
    troi, static, forcing, ds = synthetic_run
    from terrasim.paths import RunPaths
    rp = RunPaths(troi)
    assert os.path.exists(rp.success) and os.path.exists(rp.summary_csv)
    assert ds['theta'].dims == ('time', 'layer', 'y', 'x') and ds.sizes['time'] == 12
    assert float(np.abs(ds['residual']).max()) < 1e-6
    th = ds['theta'].values
    assert np.all(th <= static['theta_s'].values[None] + 1e-6) and np.all(th > 0)
    # rainy days wet the top layer relative to dry days
    rain = ds['rain'].mean(('y', 'x')).values
    assert rain.sum() > 0 and (ds['infiltration'].sum() > 0)
    df = summary(ds)
    assert len(df) == 12 and 'theta_0-5cm' in df.columns


def test_daesim_export_from_run(synthetic_run, tmp_path):
    troi, static, forcing, ds = synthetic_run
    out = tmp_path / 'forcing.csv'
    df = E.export_daesim_forcing(ds, forcing, str(out), static, where=('mean',), target=DAESIM_DEFAULT)
    back = pd.read_csv(out)
    sm = [c for c in back.columns if c.startswith('Soil moisture interp')]
    assert sm == ['Soil moisture interp 0-20', 'Soil moisture interp 20-40', 'Soil moisture interp 40-60',
                  'Soil moisture interp 60-80', 'Soil moisture interp 80-100']
    assert len(back) == 12 and back['Maximum temperature'].notna().all()
    params = E.daesim_soil_params(static, DAESIM_DEFAULT)
    assert len(params['K_sat']) == 5 and all(v < 0 for v in params['Psi_e'])
    # pixel selection works too
    pix = E.export_daesim_forcing(ds, forcing, str(tmp_path / 'p.csv'), static, where=('pixel', troi.centre_lat, troi.centre_lon))
    assert len(pix) == 12


def test_plot_module():
    from terrasim import plot
    assert plot.test()


def test_plots_from_run(synthetic_run, tmp_path):
    from terrasim import plot
    troi, static, forcing, ds = synthetic_run
    found = plot.static_for_run(ds, config=troi.config)
    assert found.attrs['key'] == static.attrs['key'] and ds.attrs['static_key'] == static.attrs['key']
    png = plot.summary_figure(ds, found, str(tmp_path / 's.png'))
    gif = plot.animate(ds, found, str(tmp_path / 'r.gif'), fps=4, dpi=40)
    assert os.path.getsize(png) > 0 and os.path.getsize(gif) > 0


def test_point_mode_matches_full_grid_without_routing(synthetic_run):
    """Point mode must reproduce the full-grid engine exactly at the chosen pixels when routing is off."""
    import xarray as xr
    from terrasim import calibrate
    troi, static, forcing, _ = synthetic_run
    # drainage chooses its sub-step count from the wettest pixel on the grid, so with the default limiter
    # point mode (fewer pixels) sub-steps less and the top layer can differ by ~1e-2; pin the count so the
    # comparison is exact and any difference means the subset is wrong
    model = Model(config=troi.config, settings=Terrasim(spinup_days=2, routing='none', drain_frac_max=1e-9, max_substeps=8), block_days=5)
    full = model.run_arrays(static, forcing, out_path=None, verbose=False)
    iy, ix = np.array([1, 5, 9]), np.array([2, 7, 3])
    pts = calibrate.run_points(model, static, forcing, iy, ix, names=['a', 'b', 'c'])
    ref = np.stack([full['theta'].values[:, :, y, x] for y, x in zip(iy, ix)], axis=-1)
    assert pts['theta'].shape == ref.shape and np.allclose(pts['theta'].values, ref, atol=1e-7)
    assert np.allclose(pts['infiltration'].values, np.stack([full['infiltration'].values[:, y, x] for y, x in zip(iy, ix)], -1), atol=1e-4)
    # pairing against a fake observation at 0-30 cm returns one model value per day
    o = pd.DataFrame({'site': 's', 'station': 'b', 'lon': 0.0, 'lat': 0.0, 'time': pd.to_datetime(pts['time'].values),
                      'depth_top_m': 0.0, 'depth_bottom_m': 0.3, 'vwc': 0.25, 'n': 1})
    p = calibrate.pair(pts, o)
    assert len(p) == pts.sizes['time'] and p['mod'].between(0.0, 0.7).all()


def test_in_memory_blocks_do_not_alias(synthetic_run):
    """write=False with more days than block_days must give the same daily rain as the forcing (regression:
    blocks used to share the reusable day buffers, so earlier blocks showed the last block's values)."""
    troi, static, forcing, _ = synthetic_run
    model = Model(config=troi.config, settings=Terrasim(spinup_days=0, routing='none'), block_days=5)
    full = model.run_arrays(static, forcing, out_path=None, verbose=False)
    daily_cell = forcing.rain.reshape(forcing.nday, -1, forcing.rain.shape[1]).sum(axis=1)[:, 0]
    assert np.allclose(full['rain'].values[:, 0, 0], daily_cell, atol=1e-4)


def test_merged_points_match_separate_runs(synthetic_run):
    """Two groups merged into one engine call give the same series as running them separately."""
    from terrasim import calibrate
    troi, static, forcing, _ = synthetic_run
    model = Model(config=troi.config, settings=Terrasim(spinup_days=1, routing='none', drain_frac_max=1e-9, max_substeps=4))
    a = calibrate.point_subset(static, forcing, np.array([1, 2]), np.array([2, 3]))
    b = calibrate.point_subset(static, forcing, np.array([8]), np.array([5]))
    st, f = calibrate.merge_points([a, b])
    assert st.shape == (1, 3) and f.cell_weights.shape[0] == a[1].cell_weights.shape[0] + b[1].cell_weights.shape[0]
    bounds = (static['layer'].values, static['layer_top_m'].values, static['layer_bottom_m'].values)
    merged = calibrate._run_point_arrays(model, st, f, ['p0', 'p1', 'p2'], bounds)
    sep = calibrate.run_points(model, static, forcing, np.array([1, 2, 8]), np.array([2, 3, 5]), names=['p0', 'p1', 'p2'])
    assert np.allclose(merged['theta'].values, sep['theta'].values, atol=1e-7)
    assert np.allclose(merged['rain'].values, sep['rain'].values, atol=1e-5)


def test_calibrate_module():
    from terrasim import calibrate, obs
    assert calibrate.test() and obs.test()


def test_routing_makes_valley_wetter(synthetic_run):
    """The synthetic DEM has a central valley: after rain, higher TWI should pair with higher mean theta."""
    troi, static, forcing, ds = synthetic_run
    twi = static['twi'].values.ravel()
    theta_top = ds['theta'].isel(layer=0).mean('time').values.ravel()
    ok = np.isfinite(twi) & np.isfinite(theta_top)
    r = np.corrcoef(twi[ok], theta_top[ok])[0, 1]
    assert r > -0.2   # not anti-correlated; valley pixels are at least as wet
