# Calibration and validation: method

terrasim exists to predict soil moisture at paddocks nobody has instrumented,
so the test that matters is transfer to a site the model has never seen.
Time splits inside one site are the secondary check. This page records the
sites, how observations are prepared, the splits, what is fitted, how it is
scored, how the runs are made cheap enough to do it, and the exact workflow.
Results as they accumulate are in [calibration_log.md](calibration_log.md).
Code: `terrasim/obs.py` (observations), `terrasim/calibrate.py` (point-mode
runs, pairing, scores, screening, fitting); drivers in `scripts/calibration/`.

## Workflow in one picture

```
Dense_sites/ (CSV, xlsx, gpkg)      obs.load()        all_sites_observations.parquet
   per-site loaders  ───────────────────────────►  site, station, lon, lat, day,
                                                   depth_top_m, depth_bottom_m, vwc
                                                             │
 obs.stations() ─► calibrate.groups() ─► bbox per station cluster (≤ 2.5 km)
                                                             │
   per group, once:  StaticBuilder(bbox)  SLGA + DEM  ─► static zarr (cached by params key)
                     ForcingBuilder(bbox) SILO + BARRA (+ S2 NDVI ≥ 2017) ─► forcing pickle
                                                             │
 calibrate.SitePoints(site)  = station pixels of every group, merged ─► one (L, 1, N) column set
                                                             │
   per parameter set (≈ 15-40 s for 26 stations x 4 years):
     remap_vegetation ─► point_subset ─► Engine (routing off) ─► theta(day, layer, station)
     pair()  thickness-weighted theta over each observation's depth interval
     score() bias, RMSE, ubRMSE, r, NSE, KGE per station x depth
                                                             │
 morris_sets/morris_effects  (which parameters matter)  ─► fit() differential evolution on
 fit years ─► evaluations log ─► behavioural() ensemble ─► held-out years / stations / sites
```

## Sites

Local copy of the lab's "Dense_sites" collection (`obs.DEFAULT_ROOT`,
override with `TERRASIM_OBS_ROOT`). All values are volumetric water content.

| Site | Where, when | Observations | Role |
|---|---|---|---|
| SASMAS | Goulburn / Krui / Merriwa / Stanley NSW, 2003–2006 | 26 stations, CS616 0–30 / 30–60 / 60–90 cm, 20-min (prepared as a daily root-zone mean) | calibration |
| MRI | Mulloon Creek NSW, Oct 2020 – Jun 2026 | 18 loggers in three clusters, 6 sensors 10–150 cm, 30-min | calibration |
| Llara | Narrabri NSW, Oct 2021 – Jun 2024 | 36 capacitance probes, 11–15 depths, daily; cracking clay, cropping | transfer |
| Esdale | near Murrumbateman NSW, Apr–Jul 2025 (+ sensors to Jul 2026) | 79 handheld points on 9 dates; 7 Phenode sensors | transfer, spatial pattern |
| Tarrawarra | Victoria, Sep 1995 – Nov 1996 | ~8,600 TDR readings at 0–30 cm on 19 dates | transfer, spatial pattern |
| Nerrigundah | NSW, 1996–1998 | TDR transects and grid, met, rain, runoff (raw, local grid) | process test with runoff (not yet loaded) |

Depth support: CS616 rods integrate their 30 cm; capacitance sensors are
taken as ±5 cm around the nominal depth; campaign TDR integrates its rod
length. Model theta is averaged over the observation's interval with
thickness weights across the model layers (`calibrate.pair`).

Known data caveats: the SASMAS daily file is a mean over the layers that were
working that day (`n_layers`), taken here as 0 to 0.3·n_layers m; the Llara
channel-to-depth (10 (k−1) cm) and device-to-location mappings are inferred
from naming, and whether its values are calibrated VWC is unknown; MRI
coordinates come from the project GeoPackage via the logger crosswalk.
Sites before 2017 have no Sentinel-2 NDVI, so the crop coefficient falls back
to a constant (`ForcingBuilder(use_ndvi=False)`); BARRA-C2 and SILO cover
them.

## Observation preparation

`obs.py` turns each site's native layout into one long table. Steps, in
order, for every loader:

1. parse the native file(s); attach coordinates (SASMAS coordinate CSV; MRI
   logger crosswalk from the GeoPackage probe layer; Llara MGA55 → WGS84 with
   the inferred device-to-location mapping; Esdale and Tarrawarra carry lon/lat);
2. convert percent to fraction; drop values ≤ 0 or ≥ 0.7 (fault codes);
3. assign the depth interval each value represents (table above);
4. aggregate sub-daily records to local-day means (MRI is UTC → UTC+10) and
   keep the count behind each mean;
5. return the common schema; `obs.stations()` summarises one row per station.

No gap filling, no smoothing, no bias correction. Sensor-calibration offsets
are handled on the scoring side (unbiased metrics), not by editing the data.

## Model configuration held fixed

Unless a run says otherwise: hourly step, BARRA-C2 timing with uniform
within-hour split, FAO-56 dual-crop ET with SILO short-crop ET0, six horizon
layers to 2 m, 90-day spin-up replaying the first 90 days of forcing, initial
state at field capacity, routing **off** in point mode. Sites before 2017 use
the constant crop coefficient `VegParams.kcb_constant`; later sites use
Sentinel-2 NDVI. The surface seal is fixed at 5 mm/h with 0.05 per mm
development, capacity spread 0.5, no contributing fraction (screening showed
the seal has little effect on moisture scores at SASMAS; it is kept for
consistency with the runoff work).

## Splits

1. **Across sites, no retuning.** Fit on SASMAS and MRI together. Apply the
   result unchanged to Llara, Esdale, Tarrawarra (and Nerrigundah). That score
   is what a DAESIM user at a new paddock gets. Per-site refits then measure
   what is lost.
2. **Across stations within a network.** Three folds per network, stratified
   by landscape position (TWI, elevation), so each fold holds ridge, slope and
   valley. Calibrate on two folds, score the third.
3. **Across time within a network.** SASMAS: fit 2003–2004, validate 2005,
   test 2006. MRI: spin up 2020–2021, fit 2021–2023 (La Niña years and the
   2023 dry-down), validate 2024, test 2025–2026 once. Blocks of whole years,
   never random days (daily moisture is strongly autocorrelated).

If one parameter set cannot serve SASMAS and MRI together, that is a result:
it says which parameter must vary with site and why.

## What is fitted

Screen first (Morris, `calibrate.morris_sets` / `morris_effects`) over about
ten candidates, then fit at most five or six:

| Group | Parameters | What they control |
|---|---|---|
| surface | `seal_ksat_mm_h`, `seal_rate_per_mm`, `capacity_cv`, `pond_max_mm` | infiltration-excess runoff onset and amount |
| profile | `b_multiplier`, `bedrock_leak_frac` | drainage rate, deep loss |
| vegetation | `kcb_max` (NDVI sites) or `kcb_constant` (pre-2017 sites), `p_depletion`, `root_depth_m`, `root_beta` | ET rate and where it is taken from |
| events | `dry_reset_hours` | Green-Ampt event memory |

`ksat_multiplier` is kept as a reference knob but is a proxy for surface
sealing, not a soil property (see the README section on runoff).

Keep every parameter set within a tolerance of the best as an ensemble
rather than reporting one winner; several will be indistinguishable.

## Scores

Per station and depth interval (`calibrate.score`): bias, RMSE, unbiased
RMSE, r, NSE, KGE. Primary: median KGE across stations and depths. Secondary:
unbiased RMSE and r, because probe offsets of several percent VWC are normal
and should not drive the fit. For DAESIM relevance also score root-zone
plant-available water in mm. Report medians and spreads across stations, not
pooled regressions.

Before any fitting, three diagnostics that need no tuning and each point at
one parameter group: wetting-front arrival time by depth after big storms
(infiltration/drainage), recession rate at depth (drainage), and surface
dry-down slope (ET). In practice the baseline figure
(`plot.calibration_figure`: KGE by depth, bias vs ubRMSE per station, pooled
scatter, best/median/worst station series) shows these at a glance.

## Screening and fitting

**Morris screening** (`calibrate.morris_sets`, `morris_effects`): one-at-a-time
trajectories over the candidate space (4 levels, 4 paths, 11 parameters =
48 runs). For each parameter the elementary effect is the change in the
site-median score when only that parameter moves; its mean absolute value
(μ*) ranks influence and its spread (σ) flags interactions. Parameters with
μ* near zero are dropped from the fit.

**Fit** (`calibrate.fit`): differential evolution (scipy, population 5 per
parameter, up to 8 generations, no polishing) over the retained parameters
in unit space (log-scaled where the range spans decades), maximising the
site-median KGE over the **fit years only**. Every evaluation is appended to
`<site>_fit_evaluations.csv` with its parameters, the fit-year objective and
the objective in every other year, so validation and test scores are read
off the same log without re-running, and `calibrate.behavioural()` returns
the sets within a tolerance of the best. The reported result is that set's
parameter ranges and the held-out scores, not a single best point.

**Why not a gradient optimiser.** The objective is a median over stations of
a non-smooth score of a threshold-rich model; DE is robust to that and the
evaluation log doubles as the uncertainty sample.

## Scripts and outputs

| Script (`scripts/calibration/`) | What it does | Writes to `~/Documents/Troi-Outputs/calibration/` |
|---|---|---|
| `feasibility_sasmas_k1.py` | one 2003–2006 station run end to end | nothing (timing check) |
| `baseline.py <site> ...` | default parameters, per-group point runs, figure inputs | `<site>_baseline_paired.parquet`, `<site>_baseline_scores.csv` |
| `morris_sasmas.py` | SitePoints + 48-set Morris screening | `morris_sasmas/<site>_m<i>_*`, `morris_sasmas_design.json`, `morris_sasmas_scores_all.csv` |
| `fit_sasmas.py` | DE fit on 2003–2004 with per-year objectives | `sasmas_fit_evaluations.csv`, `sasmas_fit_best.json`, `sasmas_fit_design.json` |

Observations cache: `~/Downloads/Dense_sites/all_sites_observations.parquet`
(`obs.load()` output). Forcing and static caches:
`~/Downloads/Troi-Tmp/terrasim_store/calib/` and `.../static/`.

Long runs are launched detached (`setsid nohup python -u script.py > log &`)
so they survive the interactive session; progress is read from the log.

## Making it cheap (point mode)

Hundreds of parameter sets are needed; a multi-year full-grid run is tens of
minutes. `calibrate.run_points` runs the engine only at the station pixels
(routing off), which is about a minute per set for a network, and is exactly
the full-grid engine at those pixels (drainage sub-steps aside; pin
`max_substeps` to compare bit for bit). Static datasets and forcing are built
once per station group (`calibrate.groups`, bboxes a few km across) and
cached under `Paths.root/calib`. The full grid is run once per fold at the end
to confirm that routing and the spatial pattern hold.

```python
from terrasim import calibrate
from terrasim.params import Params
from terrasim.physics.surface import SurfaceParams

paired, scores = calibrate.evaluate('sasmas')                                     # defaults
paired2, scores2 = calibrate.evaluate('sasmas', params=Params(surface=SurfaceParams(seal_ksat_mm_h=5)))
calibrate.summarise(scores2)
```
