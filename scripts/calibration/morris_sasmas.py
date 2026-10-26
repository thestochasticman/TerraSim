import time, warnings, json, pandas as pd
from datetime import date
from terrasim import calibrate
from terrasim.params import Params
warnings.simplefilter('ignore')
O = '/home/yasar/Documents/Troi-Outputs/calibration/morris_sasmas'
allobs = pd.read_parquet('/home/yasar/Downloads/Dense_sites/all_sites_observations.parquet')
t0 = time.time()
sp = calibrate.SitePoints('sasmas', start=date(2003, 1, 1), end=date(2006, 12, 31), observations=allobs, verbose=True)
print(f'site prepared: {len(sp.names)} stations, {len(sp.groups)} groups, {sp.forcings[0].nstep} steps, in {time.time()-t0:.0f}s', flush=True)
t0 = time.time(); paired, sc = sp.evaluate(Params()); print(f'baseline merged run in {time.time()-t0:.0f}s: median kge {sc.kge.median():.2f} r {sc.r.median():.2f}', flush=True)
space = {'seal_ksat_mm_h': (2.0, 60.0, 'log'), 'seal_rate_per_mm': (0.01, 0.2, 'log'), 'capacity_cv': (0.0, 0.8, 'lin'), 'pond_max_mm': (0.5, 5.0, 'lin'),
         'b_multiplier': (0.6, 1.6, 'lin'), 'bedrock_leak_frac': (0.01, 0.3, 'log'), 'ksat_multiplier': (0.2, 2.0, 'log'),
         'kcb_constant': (0.3, 0.9, 'lin'), 'p_depletion': (0.3, 0.7, 'lin'), 'root_depth_m': (0.5, 2.0, 'lin'), 'root_beta': (1.5, 5.0, 'lin')}
samples, tags = calibrate.morris_sets(space, n_paths=4, levels=4, seed=0)
json.dump({'space': space, 'samples': samples, 'tags': tags}, open(f'{O}_design.json', 'w'), default=str)
print(f'morris: {len(samples)} parameter sets', flush=True)
sc_all = sp.sweep({f'm{i}': calibrate.params_from(s) for i, s in enumerate(samples)}, out_dir=O)
sc_all.to_csv(f'{O}_scores_all.csv', index=False)
for metric in ('kge', 'r', 'ubrmse', 'bias'):
    print(f'--- elementary effects on site-median {metric}'); print(calibrate.morris_effects(sc_all, tags, metric).round(4).to_string(index=False), flush=True)
print('=== morris done', flush=True)
