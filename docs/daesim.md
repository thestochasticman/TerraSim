# DAESIM interoperability

DAESIM2 (`daesim`, github NortonAlex/DAESIM) does not compute a soil water
balance: `PlantCH2O.calculate(W_L, W_R, soilTheta, ...)` takes volumetric
soil moisture *per layer* as forcing and converts it to soil water potential
with the Campbell curve of `SoilLayers` (`soilThetaMax`, `Psi_e`, `b_soil`,
`K_sat`). terrasim produces that forcing.

## Layers
`terrasim.layers.LayerSpec` reproduces `SoilLayers.discretise_layers()` for
`fixed`, `scaled_exp` and `horizon`; `LayerSpec.from_daesim(soil_layers)` and
`spec.to_daesim_kwargs()` convert both ways. terrasim's default is six
horizon layers to 2 m (5/10/15/30/40/100 cm); DAESIM's default is 5 × 0.2 m
(`DAESIM_DEFAULT`). Exports re-layer with thickness-weighted averaging,
which conserves stored water over the common depth range.

## Forcing CSV (`daesim2_analysis.forcing_data.ForcingDataX`, type 3)
`terrasim export-daesim --stub <stub>` (or `terrasim.export.export_daesim_forcing`) writes one row per day:

| Column | terrasim source |
|---|---|
| `Date`, `DOY`, `Year` | run day (end-of-day state, local midnight) |
| `Soil moisture interp 0-20`, `… 20-40`, … (cm, sorted) | re-layered theta; area mean, one pixel, or a paddock polygon |
| `Minimum temperature`, `Maximum temperature` (°C) | SILO |
| `Relative humidity` (%) | mean of SILO `rh_tmax`, `rh_tmin` |
| `Atmospheric pressure` (Pa) | SILO `mslp` → elevation-corrected `P = mslp·100·exp(−z/8434)` |
| `Downwelling shortwave beam/diffuse radiation` (W m⁻², **daily mean**) | SILO `radiation` split by the Erbs diffuse fraction |
| `Wind speed` | constant 2 m s⁻¹ unless a series is given |
| `Atmospheric carbon dioxide/oxygen concentration` | 400 / 209 460 **ppm** |

Flagged for confirmation against the DAESIM reader: radiation as daily vs
daytime mean; CO2/O2 units (the type-2 path converts ppm to bar internally).

## Soil parameters (`*_DAESim_soil_params.json`)
`terrasim.export.daesim_soil_params` gives `SoilLayers` kwargs plus per-layer
`soilThetaMax = theta_s`, `b_soil = b`, `Psi_e = −psi_e[mm] × 9.81e−6` MPa and
`K_sat = K_sat[m s⁻¹] × 5.66e6` mol m⁻¹ s⁻¹ MPa⁻¹ (DAESIM's default 12 ≙ 7.6 mm h⁻¹).

## Two-way coupling (hook in place, adapter later)
`Model.run(..., uptake_source=obj)` calls `obj.uptake(state, step_index, et_demand)`
each step and uses the returned `(nlayer, ny, nx)` mm array as the
transpiration sink instead of FAO-56 (clipped at wilting point). A
`DaesimUptakeAdapter` converting `PlantCH2O` canopy transpiration E
(mol m⁻² s⁻¹ → `E × 0.018 × dt_s` mm) spread by DAESIM's root fraction is the
next step once DAESIM runs alongside.
