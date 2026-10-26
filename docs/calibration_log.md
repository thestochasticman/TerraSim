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

## 2026-10-26 — fit, SASMAS (in progress)

Differential evolution over ksat_multiplier (0.3–6, log), b_multiplier
(0.5–2), p_depletion (0.2–0.8), pond_max_mm (0.5–8), kcb_constant (0.3–1.0),
root_beta (1–6); objective = median KGE over 2003–2004; 2005 (validation) and
2006 (test) scored per evaluation but not used. After ~140 of ~270
evaluations the best fit-year KGE is 0.24 (default 0.16). The small gain from
six free parameters says the remaining misfit is structural (profile
redistribution speed, deep drying), not parametric.
