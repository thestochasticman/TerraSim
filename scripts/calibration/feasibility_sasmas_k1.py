import time, warnings
from datetime import date
from troi import Troi
from terrasim.model import Model
from terrasim.forcing import ForcingBuilder
from terrasim.terrasim import Terrasim
lat, lon, h = -32.14861111111111, 150.07, 0.0015
troi = Troi(bbox=[lon - h, lat - h, lon + h, lat + h], start=date(2003, 1, 1), end=date(2006, 12, 31), stub='sasmas_K1_feas')
t0 = time.time()
forcing = ForcingBuilder(settings=Terrasim(), use_ndvi=False).build_troi(troi)
print(f'forcing built in {time.time()-t0:.0f}s: nstep {forcing.nstep}, barra fraction {forcing.barra_fraction_wet_days:.2f}, rain total {forcing.rain.sum():.0f} mm over cells {forcing.rain.shape}')
ds = Model().run(troi, forcing=forcing, verbose=True)
print(f'done total {time.time()-t0:.0f}s grid {dict(ds.sizes)} runtime {ds.attrs.get("runtime_s")}')
