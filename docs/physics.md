# terrasim physics

One step (default 1 h) for every 10 m pixel, in this order. Units: mm per
step unless noted; `theta` is volumetric water content (m³ m⁻³);
layer *i* has thickness `dz_i` (mm).

## 0. Soil hydraulic parameters (static, from SLGA via pedotransfer)

| Symbol | Source |
|---|---|
| `theta_s` (saturation / porosity) | `1 − BD/2650`, clipped [0.30, 0.70], ≥ DUL + 0.02 |
| `theta_dul` (field capacity, −33 kPa) | SLGA Drained Upper Limit |
| `theta_l15` (wilting point, −1500 kPa) | SLGA L15 |
| `b` (Campbell) | `ln(1500/33) / ln(DUL/L15)`, clipped [2, 20]; Cosby et al. (1984) where missing |
| `psi_e` (air-entry suction, mm) | `33 kPa · (DUL/theta_s)^b` so the Campbell curve passes exactly through DUL and L15 |
| `K_sat` (mm h⁻¹) | Saxton & Rawls (2006): `1930 (theta_s − DUL)^(3 − 1/b)`, clipped [0.1, 500] |
| `psi_f` (Green-Ampt wetting-front suction) | `psi_e (2b+3)/(2b+6)` (Rawls & Brakensiek 1983) |
| REW, TEW (FAO-56) | `TEW = 1000 (DUL − 0.5 L15) Ze`; REW by texture band, ≤ 0.8 TEW |

Campbell (1974), shared with DAESIM: `psi = psi_e (theta/theta_s)^(−b)`,
`K = K_sat (theta/theta_s)^(2b+3)`.

## 1. Interception
Canopy store capacity `S_max = 1 mm × fc`. Rain fills it first; the store
evaporates at ET0 before anything else; the remainder of ET0 drives soil
evaporation and transpiration.

## 2. ET demand (FAO-56 dual crop coefficient)
`De = Σ_{Ze} (DUL − theta) dz`, `Kr = 1` if `De ≤ REW` else `(TEW − De)/(TEW − REW)`,
`Ke = min(Kr (Kc_max − Kcb), (1 − fc) Kc_max)`, `E_pot = Ke ET0`, `T_pot = Kcb ET0`.
`Kcb` is linear in NDVI between bare soil (0.15) and full canopy (1.10);
`fc` follows FAO-56 eq. 76.

## 3. Infiltration (Green-Ampt, time compression)
With `S_f = psi_f (theta_s − theta_0)` and `F` the event's cumulative
infiltration, the capacity over a step solves
`F' − S_f ln(1 + F'/S_f) = F − S_f ln(1 + F/S_f) + K_sat dt`
(Newton, vectorised). Supply (throughfall + pond + run-on) below capacity
infiltrates entirely; otherwise the capacity increment infiltrates. Water is
placed by a fill-and-spill cascade (top layer to saturation, then the
next…) in which the transfer into each lower layer is capped at that
layer's `K_sat · dt`, so a slow subsoil perches water; what cannot be
stored or passed down is saturation excess. `F` resets after 6 dry hours.
Excess ponds up to 2 mm, the rest is local runoff.

**Surface seal and repellency** (`Params.surface`, off by default). The
step's capacity is further capped by a surface entry limit
`K_surf · r(theta_0) · dt`, with
`K_surf = K_seal + (K_sat,0 − K_seal) exp(−c E)` the conductivity of a crust
that develops with the rain `E` (mm) reaching the surface since renewal
(Morin & Benyamini-type exponential sealing; `E` relaxes with e-folding
`seal_recovery_days` during rain-free steps) and `r` a repellency factor
rising linearly from `repellency_factor` at wilting point to 1 at field
capacity. Both act only at the entry point: the profile keeps its own
`K_sat` for drainage and perching, unlike the `ksat_multiplier` calibration
knob, which scales every layer.

## 4. Routing (D8 run-on)
Pixels are visited in precomputed topological levels (ridges → outlets).
Run-on arriving at a level is offered for re-infiltration with the
receiving pixel's own Green-Ampt state; the unabsorbed part joins that
pixel's local runoff and moves to its D8 downstream pixel. Water at an
outlet (window edge, pit) leaves the domain. Mass-conserving by construction.

## 5. Drainage
Bottom-up, sub-stepped so `K(theta) dt_s ≤ 0.1 dz` everywhere (≤ 24
sub-steps). Per layer `q = min(K(theta_i) dt_s, (theta_i − DUL_i) dz_i, room_{i+1})`.
The bottom layer drains to deep drainage, scaled by 0.05 where it lies
below SLGA Depth to Rock. The limiters guarantee `0 ≤ theta ≤ theta_s`.

## 6. Soil evaporation
`E = min(E_pot, Σ_{Ze} (theta − 0.5 L15) dz)`, removed in proportion to each
Ze layer's available water.

## 7. Transpiration
`theta_crit = L15 + (1 − p)(DUL − L15)` with p = 0.5;
`Ks_i = clip((theta_i − L15)/(theta_crit − L15), 0, 1)`;
`T_i = min(T_pot · root_i · Ks_i, (theta_i − L15) dz_i)`; one compensation
pass gives the unmet demand to unstressed rooted layers. Roots decay
exponentially (β = 3 m⁻¹) to 1 m. An external per-layer uptake array (e.g.
DAESIM's plant hydraulics) replaces this when supplied.

## 8. Mass balance
Per pixel and step: `ΔS − (P + runon − E_i − E − ΣT − runoff_pass − deep) = residual`;
warn above 1e-6 mm, raise above 1e-3 mm. Reported in the outputs.
