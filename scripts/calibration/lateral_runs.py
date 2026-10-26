"""Full-grid runs with lateral throughflow on vs off: Milgadara (runoff) and the MRI mid cluster (valley stations)."""
import sys, time, warnings, json, pandas as pd, numpy as np, xarray as xr
from datetime import date
from troi import Troi
from terrasim import calibrate, obs, grid
from terrasim.model import Model
from terrasim.terrasim import Terrasim
from terrasim.params import Params
from terrasim.physics.surface import SurfaceParams
from terrasim.physics.lateral import LateralParams
warnings.simplefilter('ignore')
O = '/home/yasar/Documents/Troi-Outputs/calibration/lateral'
import os; os.makedirs(O, exist_ok=True)
which = sys.argv[1]; aniso = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
surface = SurfaceParams(seal_ksat_mm_h=5.0, seal_rate_per_mm=0.05, capacity_cv=0.5)
if which == 'milgadara':
    bbox, start, end = [148.46, -34.39, 148.50, -34.36], date(2022, 10, 1), date(2022, 12, 31)
    for name, params in (('seal', Params(surface=surface)), (f'seal_lat{aniso:g}', Params(surface=surface, lateral=LateralParams(anisotropy=aniso)))):
        troi = Troi(bbox=bbox, start=start, end=end, stub=f'milgadara_{name}')
        t0 = time.time(); ds = Model(params=params).run(troi, verbose=False)
        m = lambda v: float(ds[v].mean(('y', 'x')).sum())
        print(f"{name}: {time.time()-t0:.0f}s | rain {m('rain'):.0f} infil {m('infiltration'):.0f} runoff_out {m('runoff_out'):.1f} lat_out {m('lateral_out'):.0f} exfil {m('exfiltration'):.1f} deep {m('deep_drain'):.0f} | days runoff>0.1: {int((ds.runoff_out.mean(('y','x'))>0.1).sum())} | max|res| {float(abs(ds.residual).max()):.0e}", flush=True)
else:
    allobs = pd.read_parquet('/home/yasar/Downloads/Dense_sites/all_sites_observations.parquet')
    o = allobs[allobs.site == 'mri']; st = obs.stations(o); gs = calibrate.groups(st)
    g = [g for g in gs if g['name'] == which][0]
    start, end = g['start'], min(g['end'], date.today().replace(day=1) - pd.Timedelta(days=150))
    settings = Terrasim()
    static = calibrate.static_for(g['bbox'], settings, Params())
    forcing = calibrate.forcing_for(g['bbox'], start, end, settings, verbose=True)
    sub = st[st.station.isin(g['stations'])]
    iy, ix = calibrate.station_pixels(grid.model_window(g['bbox']), sub.lon.values, sub.lat.values)
    print(f"{which}: {len(sub)} stations, grid {grid.shape(grid.model_window(g['bbox']))}, {start}..{end}", flush=True)
    for name, params in (('seal', Params(surface=surface)), (f'seal_lat{aniso:g}', Params(surface=surface, lateral=LateralParams(anisotropy=aniso)))):
        t0 = time.time()
        ds = Model(settings=settings, params=params).run_arrays(static, forcing, out_path=f'{O}/{which}_{name}.zarr', verbose=False)
        pts = xr.Dataset({'theta': (('time', 'layer', 'point'), np.stack([ds['theta'].values[:, :, y, x] for y, x in zip(iy, ix)], axis=-1))},
                         coords={'time': ds['time'].values, 'layer': ds['layer'].values, 'point': sub.station.tolist(),
                                 'layer_top_m': ('layer', ds['layer_top_m'].values), 'layer_bottom_m': ('layer', ds['layer_bottom_m'].values)})
        paired = calibrate.pair(pts, o[o.station.isin(g['stations'])]); sc = calibrate.score(paired)
        paired.to_parquet(f'{O}/{which}_{name}_paired.parquet'); sc.to_csv(f'{O}/{which}_{name}_scores.csv', index=False)
        m = lambda v: float(ds[v].mean(('y', 'x')).sum())
        print(f"{name}: {time.time()-t0:.0f}s | median kge {sc.kge.median():.2f} r {sc.r.median():.2f} ubrmse {sc.ubrmse.median():.3f} bias {sc.bias.median():+.3f} | deep layers (>=1 m) bias {sc[sc.depth_bottom_m>=1.0].bias.median():+.3f} | lat_out {m('lateral_out'):.0f} exfil {m('exfiltration'):.1f} runoff_out {m('runoff_out'):.1f} | max|res| {float(abs(ds.residual).max()):.0e}", flush=True)
print('=== done', flush=True)
