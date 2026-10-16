"""Phase 0: settings, paths, layers, pedotransfer, grid, terrain — all offline."""
import numpy as np

from terrasim import terrasim, params, paths, layers, pedotransfer, grid, terrain


def test_settings_module():
    assert terrasim.test()


def test_params_module():
    assert params.test()


def test_paths_module():
    assert paths.test()


def test_layers_module():
    assert layers.test()


def test_pedotransfer_module():
    assert pedotransfer.test()


def test_grid_module():
    assert grid.test()


def test_terrain_module():
    assert terrain.test()


def test_runpaths_under_troi(tmp_troi):
    rp = paths.RunPaths(tmp_troi)
    assert rp.run_zarr.startswith(tmp_troi.out_dir)
    assert rp.daesim_csv.endswith('pytest_troi_DAESim_forcing.csv')


def test_pysheds_flowdir_on_tilted_plane():
    """The real pysheds chain agrees with the simple D8 on a clean plane."""
    ny, nx = 12, 10
    dem = np.arange(ny)[:, None] * 1.0 + np.zeros((1, nx)) + 100.0
    from affine import Affine
    fdir = terrain.d8_flowdir(dem, Affine(10, 0, 0, 0, -10, 0))
    assert fdir.shape == (ny, nx)
    interior = fdir[1:-1, 1:-1]
    assert (interior == 64).all()          # every interior pixel drains north
    ds, outlet = terrain.downstream_index(fdir)
    order, ptr = terrain.topological_levels(ds)
    assert len(order) == ny * nx and len(np.unique(order)) == ny * nx
