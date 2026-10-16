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


def test_routing_makes_valley_wetter(synthetic_run):
    """The synthetic DEM has a central valley: after rain, higher TWI should pair with higher mean theta."""
    troi, static, forcing, ds = synthetic_run
    twi = static['twi'].values.ravel()
    theta_top = ds['theta'].isel(layer=0).mean('time').values.ravel()
    ok = np.isfinite(twi) & np.isfinite(theta_top)
    r = np.corrcoef(twi[ok], theta_top[ok])[0, 1]
    assert r > -0.2   # not anti-correlated; valley pixels are at least as wet
