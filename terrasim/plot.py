"""Figures and animations from a finished run (``terrasim plot``).

Three products, all from the run zarr plus the static dataset it was built on:

- :func:`summary_figure`: rain and ET, per-layer moisture, routing and drainage,
  the DEM, the season's run-on map and mean surface moisture (one PNG).
- :func:`animate`: a daily GIF of 0-5 cm moisture over DEM contours, the DEM,
  run-on accumulated to date, and the rain / mean-moisture trace with a cursor.
- :func:`compare_figure`: surface moisture and run-on of several runs on
  shared axes (sensitivity runs, routing or step comparisons).

The static dataset is found through the ``static_key`` the run records, so a
run made with non-default soil parameters is plotted against the soils it
actually used (not the default-parameter static for the same bbox).

Example:
    ```python
    import xarray as xr
    from terrasim import plot
    run = xr.open_zarr('.../milgadara/terrasim/run.zarr').load()
    static = plot.static_for_run(run)
    plot.summary_figure(run, static, 'milgadara_summary.png', title='Milgadara Oct-Dec 2022')
    plot.animate(run, static, 'milgadara.gif')
    ```
"""
import glob
import json
import os

import numpy as np
import xarray as xr


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def static_for_run(run: xr.Dataset, config=None) -> xr.Dataset:
    """The cached static dataset a run was built on.

    Uses the run's ``static_key`` attribute (runs written since the key was
    recorded); older runs fall back to the first cached static whose pixel
    window matches, which may differ in soil parameters.
    """
    from terrasim.paths import Paths
    paths = Paths(config) if config is not None else Paths()
    key = run.attrs.get('static_key', '')
    if key:
        path = paths.static_zarr(key)
        if os.path.exists(path):
            return xr.open_zarr(path).load()
    window = run.attrs.get('window', '')
    for path in sorted(glob.glob(f'{paths.static_dir}/*.zarr')):
        if xr.open_zarr(path).attrs.get('window') == window:
            return xr.open_zarr(path).load()
    raise FileNotFoundError(f'no cached static dataset for this run (key={key!r}, window={window!r}) under {paths.static_dir}')


def _label(run: xr.Dataset) -> str:
    a = run.attrs
    parts = [a.get('stub', ''), f"{a.get('start', '')} to {a.get('end', '')}"]
    for k in ('step', 'routing', 'within_hour'):
        if a.get(k) and a[k] not in ('uniform',):
            parts.append(f'{k}={a[k]}')
    return ', '.join(p for p in parts if p)


def _mean(run, var):
    return run[var].mean(('y', 'x'))


def layer_labels(run: xr.Dataset) -> list:
    """``['0-5cm', '5-15cm', ...]`` from the run's layer bounds (falls back to the layer coord)."""
    if 'layer_top_m' in run and 'layer_bottom_m' in run:
        top, bot = run['layer_top_m'].values, run['layer_bottom_m'].values
        return [f'{int(round(a * 100))}-{int(round(b * 100))}cm' for a, b in zip(top, bot)]
    return [str(v) for v in run['layer'].values]


def summary_figure(run: xr.Dataset, static: xr.Dataset, path: str, title: str = None, dpi: int = 110) -> str:
    """Six-panel overview of a run; returns ``path``."""
    plt = _mpl()
    t = run['time'].values
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    a = ax[0, 0]
    a.bar(t, _mean(run, 'rain'), color='steelblue', width=1.0, label='rain')
    a.plot(t, _mean(run, 'et_actual'), 'r-', lw=1, label='ET actual')
    a.legend(); a.set_title('rain and ET (domain mean, mm/day)')
    a = ax[0, 1]
    for i, lab in enumerate(layer_labels(run)):
        a.plot(t, run['theta'].isel(layer=i).mean(('y', 'x')), label=lab)
    a.legend(fontsize=7, title='layer'); a.set_title('theta per layer (domain mean)')
    a = ax[0, 2]
    a.bar(t, _mean(run, 'runoff_out'), color='firebrick', width=1.0, label='runoff out (mm/day)')
    a.plot(t, _mean(run, 'deep_drain'), color='seagreen', lw=1, label='deep drainage (mm/day)')
    a.set_ylabel('mm/day'); a.set_title('runoff leaving the domain, drainage, run-on')
    ab = a.twinx(); ab.plot(t, _mean(run, 'runon'), color='steelblue', lw=1, alpha=0.8, label='run-on (pass-through count)')
    ab.set_yscale('symlog', linthresh=1.0); ab.set_ylabel('run-on, mm/day (symlog)')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = ab.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, fontsize=8)
    im = ax[1, 0].imshow(static['elevation'].values, cmap='terrain'); plt.colorbar(im, ax=ax[1, 0], fraction=0.046)
    ax[1, 0].set_title('DEM (m)')
    runon = run['runon'].sum('time').values
    im = ax[1, 1].imshow(np.log10(runon + 1), cmap='Blues'); plt.colorbar(im, ax=ax[1, 1], fraction=0.046, label='log10(mm+1)')
    ax[1, 1].set_title('run-on received over the run')
    im = ax[1, 2].imshow(run['theta'].isel(layer=0).mean('time').values, cmap='viridis'); plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
    ax[1, 2].set_title(f'mean {layer_labels(run)[0]} theta')
    for b in ax[1]:
        b.set_xticks([]); b.set_yticks([])
    fig.suptitle(title or f'terrasim: {_label(run)}')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi); plt.close(fig)
    return path


def animate(run: xr.Dataset, static: xr.Dataset, path: str, fps: int = 6, dpi: int = 72, title: str = None) -> str:
    """Daily GIF: surface moisture over DEM contours, DEM, run-on to date, rain + mean moisture; returns ``path``."""
    plt = _mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    dem = static['elevation'].values
    theta = run['theta'].isel(layer=0).values
    runon_cum = np.cumsum(run['runon'].values, axis=0)
    rain = _mean(run, 'rain').values; runoff = _mean(run, 'runoff_out').values
    mean_th = theta.mean(axis=(1, 2)); t = run['time'].values
    vmin, vmax = np.nanpercentile(theta, 1), np.nanpercentile(theta, 99)
    vmax_r = np.log10(runon_cum[-1].max() + 1) or 1.0
    fig = plt.figure(figsize=(14, 8.5)); gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.1])
    a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3)); a3 = fig.add_subplot(gs[1, :])
    im0 = a0.imshow(theta[0], cmap='viridis', vmin=vmin, vmax=vmax)
    a0.contour(dem, levels=12, colors='w', linewidths=0.4, alpha=0.7)
    plt.colorbar(im0, ax=a0, fraction=0.046, label=f'theta {layer_labels(run)[0]}'); title0 = a0.set_title('')
    im1 = a1.imshow(dem, cmap='terrain'); plt.colorbar(im1, ax=a1, fraction=0.046, label='m'); a1.set_title('DEM')
    im2 = a2.imshow(np.log10(runon_cum[0] + 1), cmap='Blues', vmin=0, vmax=vmax_r)
    plt.colorbar(im2, ax=a2, fraction=0.046, label='log10(mm+1)'); a2.set_title('run-on received to date')
    for a in (a0, a1, a2):
        a.set_xticks([]); a.set_yticks([])
    a3.bar(t, rain, color='steelblue', width=1.0, label='rain (mm/day)')
    a3.bar(t, runoff, color='firebrick', width=1.0, label=f'runoff out (mm/day, {runoff.sum():.0f} mm total)')
    a3.set_ylabel('mm/day')
    a3b = a3.twinx(); a3b.plot(t, mean_th, color='darkorange', label='mean surface theta'); a3b.set_ylabel('theta')
    cursor = a3.axvline(t[0], color='k', lw=1)
    a3.legend(loc='upper left', fontsize=8); a3b.legend(loc='upper right', fontsize=8)
    fig.suptitle(title or f'terrasim: {_label(run)}'); fig.tight_layout()

    def update(i):
        im0.set_data(theta[i]); im2.set_data(np.log10(runon_cum[i] + 1)); cursor.set_xdata([t[i], t[i]])
        title0.set_text(f'surface moisture  {str(t[i])[:10]}')
        return im0, im2, cursor, title0

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    FuncAnimation(fig, update, frames=len(t), blit=False).save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def compare_figure(runs: dict, path: str, title: str = None, dpi: int = 110) -> str:
    """Surface moisture, 30-60 cm-ish moisture and run-on of several runs ``{label: dataset}``; returns ``path``."""
    plt = _mpl()
    fig, ax = plt.subplots(2, 2, figsize=(15, 8.5)); ax = ax.ravel()
    for label, ds in runs.items():
        t = ds['time'].values
        mid = min(3, ds.sizes['layer'] - 1)
        ax[0].plot(t, ds['theta'].isel(layer=0).mean(('y', 'x')), label=label)
        ax[1].plot(t, ds['theta'].isel(layer=mid).mean(('y', 'x')), label=label)
        ro = _mean(ds, 'runoff_out').values
        ax[2].plot(t, np.cumsum(ro), label=f'{label} ({ro.sum():.0f} mm)')
        ax[3].plot(t, _mean(ds, 'runon'), label=label)
    first = next(iter(runs.values())); labels = layer_labels(first)
    ax[0].set_title(f'{labels[0]} theta, domain mean')
    ax[1].set_title(f"{labels[min(3, first.sizes['layer'] - 1)]} theta, domain mean")
    ax[2].set_title('cumulative runoff leaving the domain, mm'); ax[2].set_ylabel('mm')
    ax[3].set_title('run-on, domain mean mm/day (pass-through count)'); ax[3].set_yscale('symlog', linthresh=0.1)
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle(title or 'terrasim: run comparison'); fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi); plt.close(fig)
    return path


# -- tests -----------------------------------------------------------------------

def _tiny_run(nday=5, ny=6, nx=5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange('2022-01-01', '2022-01-06', dtype='datetime64[D]')[:nday].astype('datetime64[ns]')
    layer = ['0-5cm', '5-15cm', '15-30cm']
    coords = {'time': t, 'layer': layer, 'y': np.arange(ny), 'x': np.arange(nx)}
    day = lambda: (('time', 'y', 'x'), rng.uniform(0, 5, (nday, ny, nx)).astype('float32'))
    run = xr.Dataset({'theta': (('time', 'layer', 'y', 'x'), rng.uniform(0.1, 0.4, (nday, len(layer), ny, nx)).astype('float32')),
                      'rain': day(), 'et_actual': day(), 'runon': day(), 'runoff_out': day(), 'deep_drain': day(),
                      'layer_top_m': ('layer', [0.0, 0.05, 0.15]), 'layer_bottom_m': ('layer', [0.05, 0.15, 0.30])},
                     coords=coords, attrs={'stub': 'tiny', 'start': '2022-01-01', 'end': '2022-01-05', 'step': '1h', 'routing': 'd8'})
    static = xr.Dataset({'elevation': (('y', 'x'), np.arange(ny)[:, None] * 1.0 + np.zeros((1, nx)))}, coords={'y': np.arange(ny), 'x': np.arange(nx)})
    return run, static


def test_figures_and_gif_render():
    import tempfile
    d = tempfile.mkdtemp(prefix='terrasim_plot_')
    run, static = _tiny_run()
    png = summary_figure(run, static, f'{d}/summary.png')
    gif = animate(run, static, f'{d}/run.gif', fps=4, dpi=40)
    cmp = compare_figure({'a': run, 'b': run}, f'{d}/compare.png')
    from PIL import Image
    return all(os.path.getsize(p) > 0 for p in (png, gif, cmp)) and Image.open(gif).n_frames == run.sizes['time']


def test_static_lookup_reports_missing():
    run, _ = _tiny_run()
    run.attrs['static_key'] = 'nope'; run.attrs['window'] = '[0, 0, 0, 0]'
    try:
        static_for_run(run)
    except FileNotFoundError:
        return True
    return False


def test():
    return all([test_figures_and_gif_render(), test_static_lookup_reports_missing(),
                layer_labels(_tiny_run()[0]) == ['0-5cm', '5-15cm', '15-30cm']])


if __name__ == '__main__':
    print(test())
