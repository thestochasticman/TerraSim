import sys, time, warnings, pandas as pd
from terrasim import calibrate, obs
from terrasim.params import Params
from terrasim.terrasim import Terrasim
warnings.simplefilter('ignore')
O = '/home/yasar/Documents/Troi-Outputs/calibration'
allobs = pd.read_parquet('/home/yasar/Downloads/Dense_sites/all_sites_observations.parquet')
for site in sys.argv[1:]:
    t0 = time.time()
    paired, scores = calibrate.evaluate(site, Terrasim(), Params(), observations=allobs, verbose=True)
    paired.to_parquet(f'{O}/{site}_baseline_paired.parquet'); scores.to_csv(f'{O}/{site}_baseline_scores.csv', index=False)
    print(f'=== {site} done in {time.time()-t0:.0f}s: {len(paired)} pairs, {scores.station.nunique() if len(scores) else 0} stations')
    print(calibrate.summarise(scores).to_string(index=False) if len(scores) else 'no scores', flush=True)
