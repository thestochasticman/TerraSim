"""Pedotransfer functions: SLGA soil properties -> hydraulic parameters.

SLGA gives bulk density, texture, drained upper limit (DUL, -33 kPa) and
the 15-bar lower limit (L15, -1500 kPa) but neither porosity nor
saturated conductivity. Everything the physics needs is derived here:

- porosity (theta_s) from bulk density;
- Campbell (1974) retention parameters ``b`` and ``psi_e`` fitted so the
  curve passes exactly through DUL at 33 kPa and L15 at 1500 kPa (this
  ``b`` is also Saxton & Rawls' ``B``);
- saturated conductivity from Saxton & Rawls (2006), with Cosby et al.
  (1984) texture regressions as the fallback where DUL/L15 are missing;
- Green-Ampt wetting-front suction (Rawls & Brakensiek 1983);
- FAO-56 soil-evaporation constants (REW, TEW);
- conversions to DAESIM's units for ``K_sat`` and ``Psi_e``.

Units are explicit in every name: ``_mm`` for suction head in mm of water,
``_mm_h`` for conductivity, fractions (0-1) for texture.

All functions are pure numpy and broadcast over arrays.
"""
import numpy as np

KPA_TO_MM = 101.97          # 1 kPa of water potential = 101.97 mm head
PSI_DUL_KPA = 33.0          # drained upper limit / field capacity
PSI_L15_KPA = 1500.0        # permanent wilting point
G = 9.81                    # m s-2
RHO_WATER = 1000.0          # kg m-3
M_WATER = 0.018015          # kg mol-1
# K [mol m-1 s-1 MPa-1] = K [m s-1] * (RHO/M) [mol m-3] / (RHO*G*1e-6) [MPa m-1]
_K_MS_TO_DAESIM = (RHO_WATER / M_WATER) / (RHO_WATER * G * 1e-6)   # ~5.66e6
_MM_HEAD_TO_MPA = RHO_WATER * G * 1e-3 * 1e-6                       # 1 mm head = 9.81e-6 MPa


def porosity(bd_g_cm3, rho_particle: float = 2650.0, lo: float = 0.30, hi: float = 0.70) -> np.ndarray:
    """Total porosity (m3 m-3) from bulk density (g cm-3): 1 - BD/rho_particle, clipped."""
    bd = np.asarray(bd_g_cm3, dtype=float)
    return np.clip(1.0 - bd * 1000.0 / rho_particle, lo, hi)


def campbell_from_limits(theta_s, theta_dul, theta_l15, b_min: float = 2.0, b_max: float = 20.0):
    """Campbell ``b`` and air-entry suction ``psi_e`` (mm) fitted through DUL and L15.

    Campbell: psi = psi_e * (theta/theta_s)^(-b). Two points give
    ``b = ln(1500/33) / ln(DUL/L15)`` and ``psi_e = 33 kPa * (DUL/theta_s)^b``.
    ``theta_s`` must exceed ``DUL`` (callers enforce this in the static builder).
    """
    theta_s = np.asarray(theta_s, dtype=float)
    dul = np.asarray(theta_dul, dtype=float)
    l15 = np.asarray(theta_l15, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        b = np.log(PSI_L15_KPA / PSI_DUL_KPA) / np.log(dul / l15)
        b = np.clip(b, b_min, b_max)
        psi_e_kpa = PSI_DUL_KPA * (dul / theta_s) ** b
    return b, psi_e_kpa * KPA_TO_MM


def campbell_cosby(sand_pct, clay_pct):
    """Cosby et al. (1984) texture regressions: ``(b, psi_e_mm, ksat_mm_h, theta_s)``.

    Fallback where DUL/L15 are unavailable. b = 2.91 + 0.159*clay;
    log10(psi_e [cm]) = 1.88 - 0.0131*sand; log10(Ks [in h-1]) = -0.6 + 0.0126*sand - 0.0064*clay;
    theta_s = 0.489 - 0.00126*sand.
    """
    sand = np.asarray(sand_pct, dtype=float)
    clay = np.asarray(clay_pct, dtype=float)
    b = 2.91 + 0.159 * clay
    psi_e_mm = 10.0 ** (1.88 - 0.0131 * sand) * 10.0            # cm -> mm
    ksat_mm_h = 10.0 ** (-0.6 + 0.0126 * sand - 0.0064 * clay) * 25.4
    theta_s = 0.489 - 0.00126 * sand
    return b, psi_e_mm, ksat_mm_h, theta_s


def retention_saxton_rawls(sand_frac, clay_frac, om_pct: float = 1.5):
    """Saxton & Rawls (2006) eqs 1-2: volumetric water content at -33 kPa (field
    capacity) and -1500 kPa (wilting point) from sand and clay fractions (0-1)
    and organic matter (% by weight). Used when SLGA's DUL/L15 layers are
    unavailable. Returns ``(theta_33, theta_1500)``, clipped to sane ranges."""
    S = np.asarray(sand_frac, dtype=float); C = np.asarray(clay_frac, dtype=float); OM = float(om_pct)
    t1500 = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    theta_1500 = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    theta_33 = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    theta_1500 = np.clip(theta_1500, 0.02, 0.40)
    theta_33 = np.clip(theta_33, theta_1500 + 0.02, 0.55)
    return theta_33, theta_1500


def ksat_saxton_rawls(theta_s, theta_dul, b, lo: float = 0.1, hi: float = 500.0) -> np.ndarray:
    """Saturated conductivity (mm h-1), Saxton & Rawls (2006) eq. 16:
    ``Ks = 1930 * (theta_s - theta_33)^(3 - lambda)`` with ``lambda = 1/B`` and
    ``B`` the Campbell b fitted through the 33 and 1500 kPa points."""
    theta_s = np.asarray(theta_s, dtype=float)
    dul = np.asarray(theta_dul, dtype=float)
    b = np.asarray(b, dtype=float)
    gap = np.clip(theta_s - dul, 1e-4, None)
    return np.clip(1930.0 * gap ** (3.0 - 1.0 / b), lo, hi)


def wetting_front_suction(psi_e_mm, b) -> np.ndarray:
    """Effective Green-Ampt wetting-front suction (mm), Rawls & Brakensiek (1983):
    ``psi_f = psi_e * (2 + 3*lambda) / (2*(1 + 3*lambda))`` with ``lambda = 1/b``,
    i.e. ``psi_e * (2b + 3) / (2b + 6)``."""
    b = np.asarray(b, dtype=float)
    return np.asarray(psi_e_mm, dtype=float) * (2 * b + 3) / (2 * b + 6)


def rew_tew(sand_frac, theta_dul, theta_l15, ze_m: float = 0.10, rew_mm=None):
    """FAO-56 soil-evaporation constants (mm): total evaporable water
    ``TEW = 1000*(DUL - 0.5*L15)*Ze`` and readily evaporable water REW, which
    follows the texture bands of FAO-56 Table 19 (coarser soils -> smaller REW)
    unless given explicitly. REW never exceeds 0.8*TEW."""
    dul = np.asarray(theta_dul, dtype=float)
    l15 = np.asarray(theta_l15, dtype=float)
    tew = 1000.0 * (dul - 0.5 * l15) * ze_m
    if rew_mm is None:
        rew = np.clip(10.0 - 6.0 * np.asarray(sand_frac, dtype=float), 2.0, 12.0)
    else:
        rew = np.full_like(tew, float(rew_mm))
    rew = np.minimum(rew, 0.8 * tew)
    return rew, tew


def to_daesim_units(ksat_mm_h, psi_e_mm):
    """Convert to DAESIM's ``K_sat`` (mol m-1 s-1 MPa-1) and ``Psi_e`` (MPa, negative).

    Sanity: DAESIM's default ``K_sat = 12`` corresponds to ~7.6 mm h-1 (a loam)."""
    k_ms = np.asarray(ksat_mm_h, dtype=float) / 1000.0 / 3600.0
    return k_ms * _K_MS_TO_DAESIM, -np.asarray(psi_e_mm, dtype=float) * _MM_HEAD_TO_MPA


def from_daesim_units(K_sat, Psi_e):
    """Inverse of :func:`to_daesim_units`: ``(ksat_mm_h, psi_e_mm)``."""
    ksat_mm_h = np.asarray(K_sat, dtype=float) / _K_MS_TO_DAESIM * 1000.0 * 3600.0
    psi_e_mm = -np.asarray(Psi_e, dtype=float) / _MM_HEAD_TO_MPA
    return ksat_mm_h, psi_e_mm


def root_fraction(bounds_m, root_depth_m: float, beta: float) -> np.ndarray:
    """Fraction of roots in each layer from an exponential density
    ``exp(-beta z)`` truncated at ``root_depth_m``; sums to 1 over layers
    with roots. ``bounds_m`` is ``(nlayer, 2)``."""
    bounds = np.asarray(bounds_m, dtype=float)
    top = np.minimum(bounds[:, 0], root_depth_m)
    bot = np.minimum(bounds[:, 1], root_depth_m)
    cum = np.exp(-beta * top) - np.exp(-beta * bot)
    total = 1.0 - np.exp(-beta * root_depth_m)
    return cum / total


# -- tests -----------------------------------------------------------------------

def test_campbell_fit_reproduces_limits():
    theta_s, dul, l15 = 0.45, 0.30, 0.15
    b, psi_e = campbell_from_limits(theta_s, dul, l15)
    psi_dul = psi_e * (dul / theta_s) ** (-b) / KPA_TO_MM
    psi_l15 = psi_e * (l15 / theta_s) ** (-b) / KPA_TO_MM
    return abs(psi_dul - 33) < 1e-9 and abs(psi_l15 - 1500) < 1e-6


def test_ksat_texture_ordering():
    # sand: large theta_s - DUL gap; clay: small gap
    b_s, _ = campbell_from_limits(0.42, 0.12, 0.05)
    b_l, _ = campbell_from_limits(0.46, 0.28, 0.13)
    b_c, _ = campbell_from_limits(0.50, 0.40, 0.27)
    k_sand = ksat_saxton_rawls(0.42, 0.12, b_s)
    k_loam = ksat_saxton_rawls(0.46, 0.28, b_l)
    k_clay = ksat_saxton_rawls(0.50, 0.40, b_c)
    return k_sand > k_loam > k_clay


def test_daesim_unit_roundtrip_and_sanity():
    k, p = to_daesim_units(7.6, 300.0)
    k_back, p_back = from_daesim_units(k, p)
    return abs(k - 12.0) / 12.0 < 0.02 and p < 0 and abs(k_back - 7.6) < 1e-9 and abs(p_back - 300.0) < 1e-9


def test_porosity_and_root_fraction():
    por = porosity(np.array([1.2, 1.6]))
    rf = root_fraction([[0, 0.1], [0.1, 0.5], [0.5, 1.0], [1.0, 2.0]], root_depth_m=1.0, beta=3.0)
    return np.allclose(por, [1 - 1200 / 2650, 1 - 1600 / 2650]) and abs(rf.sum() - 1) < 1e-12 and rf[-1] == 0


def test_retention_saxton_rawls_texture_ordering():
    fc_s, wp_s = retention_saxton_rawls(0.85, 0.05)      # sand
    fc_l, wp_l = retention_saxton_rawls(0.40, 0.20)      # loam
    fc_c, wp_c = retention_saxton_rawls(0.20, 0.50)      # clay
    # S&R Table 3 reference values: sand ~0.10/0.05, loam ~0.26/0.12 (OM 2.5%), clay ~0.40/0.30
    return (fc_s < fc_l < fc_c and wp_s < wp_l < wp_c and abs(fc_l - 0.26) < 0.05 and abs(wp_c - 0.30) < 0.06
            and fc_s > wp_s)


def test_wetting_front_and_rew_tew():
    psi_f = wetting_front_suction(100.0, 5.0)
    rew, tew = rew_tew(0.9, 0.25, 0.10, 0.10)
    return abs(psi_f - 100 * 13 / 16) < 1e-9 and abs(tew - 20.0) < 1e-9 and 2 <= rew <= 12


def test():
    return all([
        test_campbell_fit_reproduces_limits(),
        test_ksat_texture_ordering(),
        test_retention_saxton_rawls_texture_ordering(),
        test_daesim_unit_roundtrip_and_sanity(),
        test_porosity_and_root_fraction(),
        test_wetting_front_and_rew_tew(),
    ])


if __name__ == '__main__':
    print(test())
