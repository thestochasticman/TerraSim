"""Run terrasim over a region: static + forcing -> engine loop -> daily outputs.

``Model.run(troi)`` builds (or loads) the static dataset, assembles the
forcing, spins the state up by replaying the first ``spinup_days`` of
forcing, then steps through the period accumulating daily fluxes and
end-of-day moisture. Outputs stream to zarr in 30-day blocks under
``{troi.out_dir}/terrasim/run.zarr`` with a ``_SUCCESS`` marker, plus a
daily domain-mean CSV. ``Model.run_arrays`` is the troi-agnostic core.
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from attrs import frozen

from troi import Config, config as default_config
from terrasim.forcing import Forcing, ForcingBuilder
from terrasim.layers import LayerSpec
from terrasim.params import Params
from terrasim.paths import RunPaths
from terrasim.physics.engine import Engine
from terrasim.physics.state import State, Fluxes
from terrasim.static import StaticBuilder, to_arrays
from terrasim.terrasim import Terrasim, defaultterrasim

DAILY_FLUXES = ('rain', 'interception_evap', 'infiltration', 'runon', 'runoff_out', 'deep_drain',
                'soil_evap', 'transpiration', 'et_actual', 'residual')


@frozen
class Model:
    """A configured terrasim model.

    Example:
        ```python
        from datetime import date
        from troi import Troi
        from terrasim.model import Model

        troi = Troi(bbox=[148.36, -33.53, 148.38, -33.51], start=date(2022, 1, 1), end=date(2022, 12, 31), stub='demo')
        ds = Model().run(troi)          # xr.Dataset (time, layer, y, x)
        ds.theta.isel(layer=0).mean(('y', 'x')).plot()
        ```
    """

    config: Config = default_config
    settings: Terrasim = defaultterrasim
    params: Params = Params()
    layers: LayerSpec = LayerSpec()
    block_days: int = 30

    def run(s, troi, static: xr.Dataset = None, forcing: Forcing = None, uptake_source=None,
            write: bool = True, email: str = None, api_key: str = None, verbose: bool = True) -> xr.Dataset:
        paths = RunPaths(troi)
        if static is None:
            static = StaticBuilder(s.config, s.params, s.layers, s.settings).build_troi(troi, api_key=api_key)
        if forcing is None:
            forcing = ForcingBuilder(s.config, s.settings, s.params).build_troi(troi, email=email)
        out_path = paths.run_zarr if write else None
        run_attrs = {'stub': troi.stub, 'bbox': json.dumps(list(troi.bbox)), 'start': str(troi.start), 'end': str(troi.end)}
        ds = s.run_arrays(static, forcing, uptake_source=uptake_source, out_path=out_path, verbose=verbose, extra_attrs=run_attrs)
        if write:
            summary(ds).to_csv(paths.summary_csv, index=False)
        return ds

    def run_arrays(s, static: xr.Dataset, forcing: Forcing, uptake_source=None, out_path: str = None,
                   verbose: bool = True, extra_attrs: dict = None) -> xr.Dataset:
        arrays = to_arrays(static)
        attrs = {**_run_attrs(s, forcing, 0.0), 'window': static.attrs.get('window', ''), **(extra_attrs or {})}
        engine = Engine(arrays, s.settings, s.params)
        state = State.initial(arrays, s.settings.init_theta)
        spd = s.settings.steps_per_day
        L, (ny, nx) = arrays.nlayer, arrays.shape
        nday = forcing.nday
        t_start = time.time()

        # spin-up: replay the first spinup_days of forcing (not recorded)
        n_spin = min(s.settings.spinup_days, nday) * spd
        for t in range(n_spin):
            engine.step(state, forcing.step(t), None)
        if verbose and n_spin:
            print(f'terrasim: spin-up {n_spin // spd} days in {time.time() - t_start:.1f}s')

        writer = _ZarrWriter(out_path, static, forcing, L, ny, nx, s.block_days, attrs) if out_path else None
        blocks = []
        day_theta = np.zeros((s.block_days, L, ny, nx), dtype='float32')
        day_flux = {k: np.zeros((s.block_days, ny, nx), dtype='float32') for k in DAILY_FLUXES}
        day_pond = np.zeros((s.block_days, ny, nx), dtype='float32')
        day_storage = np.zeros((s.block_days, ny, nx), dtype='float32')
        acc = Fluxes.zeros(L, ny, nx)
        bi = 0
        substep_theta = [] if s.settings.output_cadence == 'substep' else None
        if substep_theta is not None:
            warnings.warn(f'substep output: ~{forcing.nstep * L * ny * nx * 4 / 1e9:.2f} GB of theta will be kept in memory')

        for d in range(nday):
            for t in range(d * spd, (d + 1) * spd):
                uptake = uptake_source.uptake(state, t, None) if uptake_source is not None else None
                fx = engine.step(state, forcing.step(t), uptake)
                acc.add_(fx)
                if substep_theta is not None:
                    substep_theta.append(state.theta.astype('float32'))
            day_theta[bi] = state.theta
            for k in DAILY_FLUXES:
                day_flux[k][bi] = acc.transpiration.sum(axis=0) if k == 'transpiration' else (acc.et_actual if k == 'et_actual' else getattr(acc, k))
            day_pond[bi] = state.pond
            day_storage[bi] = state.water_mm(arrays.dz_mm)
            acc = Fluxes.zeros(L, ny, nx)
            bi += 1
            if bi == s.block_days or d == nday - 1:
                block = _block_dataset(static, forcing, d - bi + 1, d + 1, day_theta[:bi], {k: v[:bi] for k, v in day_flux.items()},
                                       day_pond[:bi], day_storage[:bi])
                if writer:
                    writer.write(block, d - bi + 1)
                else:
                    blocks.append(block)
                bi = 0
                if verbose:
                    print(f'terrasim: day {d + 1}/{nday} ({time.time() - t_start:.0f}s)')

        if writer:
            writer.finish()
            ds = xr.open_zarr(out_path).load()
        else:
            ds = xr.concat(blocks, dim='time')
        ds.attrs.update({**attrs, 'runtime_s': round(time.time() - t_start, 1)})
        if substep_theta is not None:
            ds['theta_substep'] = (('step', 'layer', 'y', 'x'), np.stack(substep_theta))
            ds = ds.assign_coords(step=forcing.time)
        return ds


def _block_dataset(static, forcing, d0, d1, theta, flux, pond, storage) -> xr.Dataset:
    coords = {'time': forcing.days[d0:d1], 'layer': static['layer'].values, 'y': static['y'].values, 'x': static['x'].values,
              'layer_top_m': static['layer_top_m'], 'layer_bottom_m': static['layer_bottom_m']}
    data = {'theta': (('time', 'layer', 'y', 'x'), theta), 'pond': (('time', 'y', 'x'), pond),
            'storage_mm': (('time', 'y', 'x'), storage)}
    for k, v in flux.items():
        data[k] = (('time', 'y', 'x'), v)
    ds = xr.Dataset(data, coords=coords)
    ds['theta'].attrs['units'] = 'm3 m-3'
    for k in flux:
        ds[k].attrs['units'] = 'mm d-1'
    ds['storage_mm'].attrs['units'] = 'mm'
    return ds


def _run_attrs(model, forcing, seconds) -> dict:
    return {'step': model.settings.step, 'routing': model.settings.routing, 'infiltration': model.settings.infiltration,
            'et_method': model.settings.et_method, 'disaggregation': model.settings.disaggregation,
            'spinup_days': model.settings.spinup_days, 'tz_hours': forcing.tz_hours,
            'barra_timing_fraction': forcing.barra_fraction_wet_days,
            'layers': str(model.layers.to_daesim_kwargs()), 'runtime_s': round(seconds, 1),
            'units': 'theta m3 m-3; fluxes mm d-1; storage mm'}


class _ZarrWriter:
    """Streams 30-day blocks into a pre-allocated zarr store, then marks _SUCCESS."""

    def __init__(self, path, static, forcing, L, ny, nx, block_days, attrs):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        nday = forcing.nday
        import dask.array as da
        chunks = (block_days, L, ny, nx)
        coords = {'time': forcing.days, 'layer': static['layer'].values, 'y': static['y'].values, 'x': static['x'].values,
                  'layer_top_m': static['layer_top_m'], 'layer_bottom_m': static['layer_bottom_m']}
        data = {'theta': (('time', 'layer', 'y', 'x'), da.zeros((nday, L, ny, nx), chunks=chunks, dtype='float32')),
                'pond': (('time', 'y', 'x'), da.zeros((nday, ny, nx), chunks=chunks[:1] + chunks[2:], dtype='float32')),
                'storage_mm': (('time', 'y', 'x'), da.zeros((nday, ny, nx), chunks=chunks[:1] + chunks[2:], dtype='float32'))}
        for k in DAILY_FLUXES:
            data[k] = (('time', 'y', 'x'), da.zeros((nday, ny, nx), chunks=chunks[:1] + chunks[2:], dtype='float32'))
        template = xr.Dataset(data, coords=coords, attrs=attrs)
        template.to_zarr(path, mode='w', compute=False)

    def write(self, block: xr.Dataset, d0: int):
        n = block.sizes['time']
        block.drop_vars([c for c in ('layer_top_m', 'layer_bottom_m') if c in block.coords]) \
             .drop_vars(['time', 'layer', 'y', 'x']).to_zarr(self.path, region={'time': slice(d0, d0 + n)})

    def finish(self):
        open(f'{self.path}/_SUCCESS', 'w').close()


def summary(ds: xr.Dataset) -> pd.DataFrame:
    """Daily domain means: one row per day, theta per layer plus every flux."""
    out = {'date': pd.to_datetime(ds['time'].values)}
    for i in range(ds.sizes['layer']):
        top, bot = ds['layer_top_m'].values[i], ds['layer_bottom_m'].values[i]
        out[f'theta_{int(round(top * 100))}-{int(round(bot * 100))}cm'] = ds['theta'].isel(layer=i).mean(('y', 'x')).values
    for k in list(DAILY_FLUXES) + ['storage_mm', 'pond']:
        if k in ds:
            out[k] = ds[k].mean(('y', 'x')).values
    return pd.DataFrame(out)


def run_troi(troi, **kwargs) -> xr.Dataset:
    """One-liner: ``run_troi(troi)`` with default settings."""
    return Model().run(troi, **kwargs)
