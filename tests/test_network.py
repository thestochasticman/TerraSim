"""Live checks against the data services. Run with ``pytest -m network``."""
from datetime import date

import numpy as np
import pytest

BBOX = [148.36265, -33.52606, 148.38265, -33.50606]


@pytest.mark.network
def test_barra_one_month(tmp_config):
    from terrasim.barra import Store
    store = Store(config=tmp_config)
    ds = store.get_ds(BBOX, date(2022, 2, 1), date(2022, 2, 28))
    assert ds.pr_mm.shape[0] == 28 * 24 and ds.sizes['cell'] >= 1
    assert np.isfinite(ds.pr_mm.values).all() and float(ds.pr_mm.max()) < 200
    assert store.fill(BBOX, date(2022, 2, 1), date(2022, 2, 28)) == 0


@pytest.mark.network
def test_silo_points_and_year():
    from troi import config
    from pysilo.store import Store
    from terrasim import grid
    from terrasim.forcing import silo_points_for_window
    if not config.email:
        pytest.skip('SILO needs config.email')
    pids, idx, coords = silo_points_for_window(grid.model_window(BBOX))
    assert len(pids) == 1
    df = Store().get_df(coords[0][0], coords[0][1], date(2022, 1, 1), date(2022, 12, 31))
    assert len(df) == 365 and 'et_short_crop' in df.columns


@pytest.mark.network
def test_static_build_real():
    from troi import config
    from terrasim.static import StaticBuilder
    if not config.tern_api_key:
        pytest.skip('SLGA needs config.tern_api_key')
    ds = StaticBuilder().build(BBOX)
    assert ds['theta_s'].shape[0] == 6 and bool((ds['theta_s'] > ds['theta_dul']).all())
