"""Command line: ``terrasim run | static | export-daesim | validate | plot``.

    terrasim run --bbox 148.36 -33.53 148.38 -33.51 --start 2022-01-01 --end 2022-12-31 --stub demo
    terrasim export-daesim --stub demo [--where mean | --pixel LAT LON | --paddocks farm.gpkg --label paddock]
    terrasim validate --stub demo --oznet path/to/station.xls [--pixel LAT LON]
    terrasim plot --stub demo [--gif] [--compare other_stub ...]
"""
import argparse
import json
import sys
from datetime import date

from terrasim.terrasim import Terrasim


def _troi_from_args(a):
    from troi import Troi
    if a.bbox is not None and a.start and a.end:
        return Troi(bbox=a.bbox, start=date.fromisoformat(a.start), end=date.fromisoformat(a.end),
                    **({'stub': a.stub} if a.stub else {}))
    registry = _lookup_stub(a.stub) if a.stub else None
    if registry is None:
        sys.exit('give --bbox, --start and --end, or a --stub already in the troi registry')
    return Troi(bbox=registry['bbox'], start=date.fromisoformat(registry['start']), end=date.fromisoformat(registry['end']), stub=a.stub)


def _lookup_stub(stub):
    """``{'bbox', 'start', 'end'}`` for a stub in the troi registry (``config.hash_file``), else None.

    The registry is keyed by bbox hash: ``{hash: {'bbox': [...], 'queries': [{'stub', 'start', 'end', ...}, ...]}}``;
    a flat ``{'stub', 'bbox', 'start', 'end'}`` entry (older layout) is accepted too.
    """
    from troi import config
    try:
        data = json.load(open(config.hash_file))
    except (OSError, ValueError):
        return None
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        if entry.get('stub') == stub and 'bbox' in entry:
            return {'bbox': entry['bbox'], 'start': entry['start'], 'end': entry['end']}
        for q in entry.get('queries', []):
            if isinstance(q, dict) and q.get('stub') == stub:
                return {'bbox': entry['bbox'], 'start': q['start'], 'end': q['end']}
    return None


def _settings_from_args(a) -> Terrasim:
    kw = {k: getattr(a, k) for k in ('step', 'routing', 'infiltration', 'disaggregation', 'within_hour', 'peak_fraction',
                                     'output_cadence', 'init_theta')
          if getattr(a, k, None) is not None}
    if a.spinup_days is not None:
        kw['spinup_days'] = a.spinup_days
    try:
        return Terrasim(**kw)
    except ValueError as e:
        sys.exit(f'invalid settings: {e}')


def cmd_run(a):
    from terrasim.model import Model
    troi = _troi_from_args(a)
    ds = Model(settings=_settings_from_args(a)).run(troi, email=a.email, api_key=a.api_key, verbose=not a.quiet)
    from terrasim.paths import RunPaths
    rp = RunPaths(troi)
    print(f'wrote {rp.run_zarr}\nwrote {rp.summary_csv}\nruntime {ds.attrs.get("runtime_s")} s; '
          f'BARRA timing used on {100 * ds.attrs.get("barra_timing_fraction", 0):.0f}% of rainy days')


def cmd_static(a):
    from terrasim.static import StaticBuilder
    troi = _troi_from_args(a)
    ds = StaticBuilder().build_troi(troi, api_key=a.api_key, force=a.force)
    print(ds)


def cmd_export(a):
    import xarray as xr
    from terrasim.export import export_daesim_forcing, write_daesim_soil_params, paddock_masks
    from terrasim.forcing import ForcingBuilder
    from terrasim.layers import LayerSpec, DAESIM_DEFAULT
    from terrasim.paths import RunPaths
    from terrasim.static import StaticBuilder
    troi = _troi_from_args(a)
    rp = RunPaths(troi)
    run = xr.open_zarr(rp.run_zarr).load()
    static = StaticBuilder().build_troi(troi, api_key=a.api_key)
    forcing = ForcingBuilder(use_ndvi=False, use_barra=False).build_troi(troi, email=a.email)
    target = DAESIM_DEFAULT if a.layers == 'daesim' else LayerSpec()
    if a.paddocks:
        for label, mask in paddock_masks(a.paddocks, static, a.label).items():
            out = rp.daesim_csv.replace('.csv', f'_{label}.csv')
            export_daesim_forcing(run, forcing, out, static, where=mask, target=target)
            write_daesim_soil_params(static, out.replace('_forcing', '_soil_params').replace('.csv', '.json'), target, mask)
            print('wrote', out)
    else:
        where = ('pixel', a.pixel[0], a.pixel[1]) if a.pixel else ('mean',)
        export_daesim_forcing(run, forcing, rp.daesim_csv, static, where=where, target=target)
        write_daesim_soil_params(static, rp.daesim_json, target)
        print('wrote', rp.daesim_csv, 'and', rp.daesim_json)


def cmd_validate(a):
    import xarray as xr
    from terrasim.paths import RunPaths
    from terrasim.static import StaticBuilder
    from terrasim.validate import compare_oznet, compare_ozwald_ssoil
    troi = _troi_from_args(a)
    run = xr.open_zarr(RunPaths(troi).run_zarr).load()
    static = StaticBuilder().build_troi(troi, api_key=a.api_key)
    if a.oznet:
        where = ('pixel', a.pixel[0], a.pixel[1]) if a.pixel else ('mean',)
        print(compare_oznet(run, a.oznet, where=where, static=static).to_string(index=False))
    if a.ozwald:
        print(compare_ozwald_ssoil(run, troi.centre_lat, troi.centre_lon))


def cmd_plot(a):
    import xarray as xr
    from terrasim import plot
    from terrasim.paths import RunPaths
    troi = _troi_from_args(a)
    rp = RunPaths(troi)
    run = xr.open_zarr(rp.run_zarr).load()
    static = plot.static_for_run(run)
    title = a.title or None
    print('wrote', plot.summary_figure(run, static, rp.summary_png, title=title))
    if a.gif:
        print('wrote', plot.animate(run, static, rp.gif, fps=a.fps, title=title))
    if a.compare:
        from troi import Troi
        runs = {troi.stub: run}
        for stub in a.compare:
            if stub == troi.stub:
                continue
            entry = _lookup_stub(stub)
            if entry is None:
                sys.exit(f'--compare: stub {stub!r} is not in the troi registry')
            try:
                other = Troi(bbox=entry['bbox'], start=date.fromisoformat(entry['start']), end=date.fromisoformat(entry['end']), stub=stub)
                runs[stub] = xr.open_zarr(RunPaths(other).run_zarr).load()
            except Exception as e:
                sys.exit(f'--compare: cannot open run for stub {stub!r}: {e}')
        out = f'{rp.run_dir}/{troi.stub}_terrasim_compare.png'
        print('wrote', plot.compare_figure(runs, out, title=title))


def build_parser():
    p = argparse.ArgumentParser(prog='terrasim', description='10 m process-based soil moisture for Australian paddocks')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--bbox', nargs=4, type=float, metavar=('W', 'S', 'E', 'N'))
        sp.add_argument('--start'); sp.add_argument('--end'); sp.add_argument('--stub')
        sp.add_argument('--email', help='SILO registration email (else troi config)')
        sp.add_argument('--api-key', dest='api_key', help='TERN API key for SLGA (else troi config)')

    r = sub.add_parser('run', help='run the model and write outputs'); common(r)
    r.add_argument('--step', choices=['20min', '1h', '3h']); r.add_argument('--routing', choices=['d8', 'twi', 'none'])
    r.add_argument('--infiltration', choices=['green_ampt', 'capacity'])
    r.add_argument('--disaggregation', choices=['barra', 'climatology', 'uniform'])
    r.add_argument('--within-hour', dest='within_hour', choices=['uniform', 'peaked'],
                   help="with --step 20min: 'peaked' concentrates each hour's rain into one sub-step")
    r.add_argument('--peak-fraction', dest='peak_fraction', type=float, help='share of the hour in the peak sub-step (0.6)')
    r.add_argument('--output-cadence', dest='output_cadence', choices=['daily', 'substep'])
    r.add_argument('--init-theta', dest='init_theta', choices=['dul', 'midpoint', 'l15'])
    r.add_argument('--spinup-days', dest='spinup_days', type=int); r.add_argument('--quiet', action='store_true')
    r.set_defaults(func=cmd_run)

    s = sub.add_parser('static', help='build (or show) the static terrain+soil dataset'); common(s)
    s.add_argument('--force', action='store_true'); s.set_defaults(func=cmd_static)

    e = sub.add_parser('export-daesim', help='write DAESIM forcing CSV + soil parameter JSON'); common(e)
    e.add_argument('--pixel', nargs=2, type=float, metavar=('LAT', 'LON'))
    e.add_argument('--paddocks', help='GeoPackage/shapefile of paddocks (one CSV each)'); e.add_argument('--label')
    e.add_argument('--layers', choices=['daesim', 'model'], default='daesim'); e.set_defaults(func=cmd_export)

    v = sub.add_parser('validate', help='compare with OzNet / OzWALD'); common(v)
    v.add_argument('--oznet', help='OzNet station .xls'); v.add_argument('--ozwald', action='store_true')
    v.add_argument('--pixel', nargs=2, type=float, metavar=('LAT', 'LON')); v.set_defaults(func=cmd_validate)

    g = sub.add_parser('plot', help='summary figure (and optional GIF) of a finished run'); common(g)
    g.add_argument('--gif', action='store_true', help='also write the daily animation')
    g.add_argument('--fps', type=int, default=6); g.add_argument('--title')
    g.add_argument('--compare', nargs='+', metavar='STUB', help='other run stubs to overlay in a comparison figure')
    g.set_defaults(func=cmd_plot)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
