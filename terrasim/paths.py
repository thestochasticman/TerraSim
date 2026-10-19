"""Derived on-disk locations for terrasim's machine-wide cache and per-run outputs.

Rule of thumb across the lab's packages: user-settable inputs -> Config,
derived locations -> Paths. No inheritance — composition only.

Two scopes:

- :class:`Paths` (keyed by :class:`troi.Config`): the machine-wide store,
  shared by every run — BARRA-C2 hourly rain ledger and the static
  terrain+soil datasets, cached by region.
- :class:`RunPaths` (keyed by a ``Troi``): where one run's outputs go
  under ``troi.out_dir``.
"""
from attrs import frozen, field
from troi import Config, config as default_config


@frozen
class Paths:
    """Where the terrasim store lives for a given Config.

    Attributes:
        config: The :class:`troi.Config` supplying the data root.
        root: Store directory (``{config.tmp_dir}/terrasim_store``).
        barra_db: SQLite ledger + hourly rain table for BARRA-C2 cells.
        static_dir: Directory of cached static datasets, one zarr per region key.

    Example:
        ```python
        from terrasim.paths import Paths

        Paths().barra_db  # '~/Downloads/Troi-Tmp/terrasim_store/barra.db'
        ```
    """

    config: Config = default_config

    root: str = field(init=False)
    barra_db: str = field(init=False)
    static_dir: str = field(init=False)

    root.default(lambda s: f'{s.config.tmp_dir}/terrasim_store')
    barra_db.default(lambda s: f'{s.root}/barra.db')
    static_dir.default(lambda s: f'{s.root}/static')

    def static_zarr(s, key: str) -> str:
        """Zarr path of the cached static dataset for a region/layers/PTF key."""
        return f'{s.static_dir}/{key}.zarr'


@frozen
class RunPaths:
    """Where one run's outputs go (``{troi.out_dir}/terrasim/...``)."""

    troi: object

    run_dir: str = field(init=False)
    run_zarr: str = field(init=False)
    success: str = field(init=False)
    summary_csv: str = field(init=False)
    daesim_csv: str = field(init=False)
    daesim_json: str = field(init=False)
    checkpoint: str = field(init=False)
    summary_png: str = field(init=False)
    gif: str = field(init=False)

    run_dir.default(lambda s: f'{s.troi.out_dir}/terrasim')
    run_zarr.default(lambda s: f'{s.run_dir}/run.zarr')
    success.default(lambda s: f'{s.run_dir}/run.zarr/_SUCCESS')
    summary_csv.default(lambda s: f'{s.run_dir}/{s.troi.stub}_terrasim_daily.csv')
    daesim_csv.default(lambda s: f'{s.run_dir}/{s.troi.stub}_DAESim_forcing.csv')
    daesim_json.default(lambda s: f'{s.run_dir}/{s.troi.stub}_DAESim_soil_params.json')
    checkpoint.default(lambda s: f'{s.run_dir}/checkpoint.npz')
    summary_png.default(lambda s: f'{s.run_dir}/{s.troi.stub}_terrasim_summary.png')
    gif.default(lambda s: f'{s.run_dir}/{s.troi.stub}_terrasim.gif')


def test_paths_derive_from_config():
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix='terrasim_paths_test_')
    cfg = Config(out_dir=tmpdir, tmp_dir=tmpdir)
    paths = Paths(cfg)
    return (
        paths.root == f'{tmpdir}/terrasim_store'
        and paths.barra_db == f'{tmpdir}/terrasim_store/barra.db'
        and paths.static_zarr('abc') == f'{tmpdir}/terrasim_store/static/abc.zarr'
    )


def test():
    return test_paths_derive_from_config()


if __name__ == '__main__':
    print(test())
