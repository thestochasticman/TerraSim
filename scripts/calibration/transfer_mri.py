import time, warnings, json, pandas as pd
from datetime import date
from terrasim import calibrate
from terrasim.params import Params
warnings.simplefilter('ignore')
O = '/home/yasar/Documents/Troi-Outputs/calibration'
allobs = pd.read_parquet('/home/yasar/Downloads/Dense_sites/all_sites_observations.parquet')
t0 = time.time()
sp = calibrate.SitePoints('mri', start=date(2020, 10, 21), end=date(2026, 6, 3), observations=allobs, verbose=True)
print(f'mri prepared: {len(sp.names)} stations in {time.time()-t0:.0f}s', flush=True)
best = json.load(open(f'{O}/sasmas_fit_best.json'))
best_mri = dict(best); best_mri.pop('kcb_constant')          # MRI has NDVI: the constant does not apply
sc = sp.sweep({'default': Params(), 'sasmas_fit': calibrate.params_from(best_mri)}, out_dir=f'{O}/transfer')
print('=== transfer done', flush=True)
