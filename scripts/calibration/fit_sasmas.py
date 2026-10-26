import time, warnings, json, pandas as pd
from datetime import date
from terrasim import calibrate
warnings.simplefilter('ignore')
O = '/home/yasar/Documents/Troi-Outputs/calibration'
allobs = pd.read_parquet('/home/yasar/Downloads/Dense_sites/all_sites_observations.parquet')
sp = calibrate.SitePoints('sasmas', start=date(2003, 1, 1), end=date(2006, 12, 31), observations=allobs, verbose=False)
print(f'site prepared: {len(sp.names)} stations', flush=True)
space = {'ksat_multiplier': (0.3, 6.0, 'log'), 'b_multiplier': (0.5, 2.0, 'lin'), 'p_depletion': (0.2, 0.8, 'lin'),
         'pond_max_mm': (0.5, 8.0, 'lin'), 'kcb_constant': (0.3, 1.0, 'lin'), 'root_beta': (1.0, 6.0, 'lin')}
fixed = {'seal_ksat_mm_h': 5.0, 'seal_rate_per_mm': 0.05, 'capacity_cv': 0.5}
json.dump({'space': space, 'fixed': fixed, 'fit_years': [2003, 2004], 'validate': [2005], 'test': [2006]}, open(f'{O}/sasmas_fit_design.json', 'w'))
t0 = time.time()
best, log = calibrate.fit(sp, space, fixed=fixed, years=[2003, 2004], metric='kge', maxiter=8, popsize=5, seed=0, out_dir=O, label='fit', verbose=True)
print(f'=== fit done in {time.time()-t0:.0f}s: best {best}', flush=True)
json.dump(best, open(f'{O}/sasmas_fit_best.json', 'w'))
b = calibrate.behavioural(log, 0.03)
print(f'behavioural set within 0.03 KGE of best: {len(b)} of {len(log)} evaluations'); print(b[list(space) + ['objective', 'obj_2005', 'obj_2006']].describe().loc[['min', '50%', 'max']].round(3).to_string(), flush=True)
