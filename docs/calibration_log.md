# Calibration log

Dated record of what was run and what came out. Method: [calibration.md](calibration.md).
Scores are site medians over stations x depth intervals unless stated.

## 2026-10-25 — data in hand

- Six sites loaded into the common table: 590,330 daily rows (SASMAS 26 stations,
  MRI 18, Llara 32, Esdale 79 points + 7 Phenode sensors, Tarrawarra 3,610 points).
  Nerrigundah not yet loaded (local grid, raw files).
- MRI coordinates recovered from the project GeoPackage (`Soil_Moisture_Probes`
  layer) via the logger-to-name table in `SM_metadata.pdf`; all 18 loggers matched.
- Llara channel depth (10 (k-1) cm) and device-to-location mapping inferred from
  naming; unconfirmed. Values up to 110 % suggest uncalibrated sensor output.

## 2026-10-25 — feasibility

- SASMAS K1, 2003–2006, 33 x 30 pixel grid, no NDVI: forcing fetched in 114 s
  (BARRA timing on 69 % of wet days), four model years in 27 s. Pre-Sentinel
  sites are workable.

## 2026-10-25 — baselines (default parameters, no tuning)

**SASMAS** (26 stations, 31,978 pairs, point mode per group)

| depth class | stations | bias | ubRMSE | r | KGE |
|---|---|---|---|---|---|
| 0–30 cm | 4 | −0.032 | 0.045 | 0.60 | 0.44 |
| 0–60 cm | 7 | −0.006 | 0.038 | 0.55 | 0.27 |
| 0–90 cm | 15 | −0.039 | 0.032 | 0.50 | 0.12 |

Median over all stations: r 0.57, KGE 0.20, variability ratio 0.75, mean ratio
0.92. Diagnosis from the series: the modelled root zone is too flashy (spikes
with every rain day, recedes in days; observations change over weeks) and the
deep summer dry-down at stations such as S1 is missed. Station biases run both
ways (SLGA soils at 90 m).

**MRI** (18 loggers, 179,287 pairs)

r 0.63–0.77 and KGE ≈ 0.4 in the top half metre; a −0.11 dry bias at every
depth. Several surface probes read 0.5–0.7 m³ m⁻³ for months (above mineral-soil
porosity), so much of the bias is sensor calibration; 150 cm sensors near 0.41
suggest a floodplain water table the model does not represent. For MRI, score
on ubRMSE and r.

Figures: `calibration/sasmas_baseline.png`, `calibration/mri_baseline.png`.

## 2026-10-25 — Morris screening, SASMAS (48 sets, 2003–2006)

Merged point mode: all 26 stations in one engine call, 21 s per set.

| parameter | μ* on median KGE | sign |
|---|---|---|
| ksat_multiplier | 0.055 | + (higher better, σ small) |
| b_multiplier | 0.029 | − |
| kcb_constant | 0.023 | mixed |
| pond_max_mm | 0.021 | mixed |
| p_depletion | 0.020 | − |
| seal_ksat_mm_h | 0.014 | − |
| root_beta | 0.014 | mixed |
| seal_rate_per_mm | 0.010 | + |
| capacity_cv | 0.007 | − |
| root_depth_m | 0.001 | none |
| bedrock_leak_frac | 0.000 | none |

Best screening set: median KGE 0.26 (baseline 0.16 over the same period),
bias −0.001. Dropped from the fit: root_depth_m, bedrock_leak_frac, the seal
parameters (fixed at 5 mm/h, 0.05 per mm, cv 0.5).

## 2026-10-26 — fit, SASMAS

Differential evolution over ksat_multiplier (0.3–6, log), b_multiplier
(0.5–2), p_depletion (0.2–0.8), pond_max_mm (0.5–8), kcb_constant (0.3–1.0),
root_beta (1–6); objective = median KGE over 2003–2004; 270 evaluations,
31 s each, 15 h wall. Best: ksat ×3.8, b ×1.35, p 0.23, pond 7.1 mm,
Kcb 0.32, root_beta 4.1 (`sasmas_fit_best.json`).

| period | default KGE | fitted KGE | default r | fitted r | var. ratio default → fitted |
|---|---|---|---|---|---|
| fit 2003–04 | 0.20 | 0.25 | 0.53 | 0.56 | 0.63 → 0.74 |
| validation 2005 | 0.26 | 0.31 | 0.66 | 0.69 | 1.07 → 1.09 |
| test 2006 | 0.06 | 0.10 | 0.33 | 0.41 | — |
| all years | 0.16 | 0.23 | 0.57 | 0.61 | 0.75 → 0.78 |

- Gain is real but small (+0.05 to +0.07 KGE), carries to the held-out years
  (2005 improves as much as the fit years), and 2006, a drought year, stays
  poor for both: the dry-end behaviour is structural.
- 15 of 26 stations improve, 11 get worse; the losers (G1, G2, S2) are the
  stations with large negative bias from SLGA soils, where the fitted faster
  drainage adds harm. One parameter set cannot serve all soils.
- Equifinality: 43 of 270 sets lie within 0.03 KGE of the best; across them
  ksat_multiplier spans 0.6–5.4 and kcb_constant 0.30–0.92. The data fix the
  direction (faster redistribution, less ET, more surface storage), not the values.
- Across all evaluations the fit-year objective correlates 0.45 with 2005 and
  0.56 with 2006: a better fit does generalise, weakly.

Figure: `calibration/sasmas_fit.png`.

## 2026-10-26 — transfer to MRI, no retuning

SASMAS-fitted parameters (minus `kcb_constant`, MRI has NDVI) applied to MRI
2020–2026 in merged point mode: median KGE 0.30 → 0.31, r 0.64 → 0.65,
ubRMSE 0.052 → 0.051, bias −0.107 → −0.102. The transfer neither helps nor
harms: the parameters that matter at SASMAS are not what limits MRI (sensor
level and the floodplain water table are). Preparing MRI's merged forcing over
the common period took 5.8 h (Sentinel-2 re-read for three clusters over six
years); cache it before any MRI sweep.

## What the first round says

1. Parameter tuning buys ~0.07 KGE at SASMAS and nothing at MRI. The ceiling is
   structural: the profile redistributes too fast at the surface and too slowly
   at depth, dry years are missed, and valley-floor sites need a water table.
2. The next model work, in order: profile drainage/redistribution (why the
   fit wants ksat ×4 with b ×1.35), a saturation-excess / lateral path for
   floodplain stations, and per-site soil hydraulics from field data where it
   exists (Esdale campaign bulk density; SASMAS site readmes with texture).
3. Scoring: use ubRMSE and r at MRI; treat SASMAS n_layers < 3 stations with
   care (depth support assumed).
