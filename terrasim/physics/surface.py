"""Surface entry limit: sealing (crusting) and water repellency.

Field surfaces shed water at rain rates well below the bulk soil's K_sat.
Raindrop impact breaks aggregates and washes fines into the surface pores,
forming a crust a few millimetres thick whose conductivity is commonly an
order of magnitude below the soil beneath (hardsetting surfaces are typical
of south-eastern Australian duplex and sandy-loam soils). Dry sandy topsoils
under pasture are often water repellent, so the first rain onto a dry
surface beads and runs before the surface wets.

Both act only at the entry point, so they are modelled as a limit on the
surface infiltration capacity rather than by lowering the profile's K_sat:
drainage, perching and the deep-layer trajectory keep the soil's own
conductivities.

Seal conductivity follows the usual exponential development with cumulative
rainfall since the surface was last renewed (e.g. Morin & Benyamini 1977;
the form used in WEPP/CREAMS-type models):

    K_surf = K_seal + (K_sat,0 - K_seal) * exp(-c * E)

with ``E`` the rain (mm) that has reached the surface since renewal, ``c``
the seal development rate (per mm) and ``K_seal`` the fully sealed
conductivity. During rain-free steps ``E`` relaxes back toward zero with an
e-folding time ``seal_recovery_days`` (shrink-swell, biological reworking,
or tillage recovering the surface). The surface limit over a step is
``K_surf * r(theta_0) * dt`` where the repellency factor ``r`` rises linearly
from ``repellency_factor`` at a surface at wilting point to 1 at field
capacity (1 everywhere when the factor is 1).

Everything is flat over pixels and mutates ``E`` in place.
"""
import numpy as np
from attrs import frozen


@frozen
class SurfaceParams:
    """Surface sealing and repellency parameters (:class:`terrasim.params.Params`\ ``.surface``).

    Attributes:
        seal_ksat_mm_h: Conductivity of a fully developed seal (mm h-1); ``None``
            disables sealing (the surface takes water at the topsoil's K_sat).
            Sandy loams and loams crust to ~2-10 mm h-1.
        seal_rate_per_mm: Seal development rate ``c``; ``1/c`` mm of rain develops
            ~63% of the seal (0.05 -> ~60 mm for 95%).
        seal_recovery_days: e-folding time (days without rain) for the seal to
            recover toward the unsealed state.
        repellency_factor: Multiplier on the surface limit when the surface is at
            wilting point (1 disables; 0.1-0.3 for strongly repellent sands).
    """

    seal_ksat_mm_h: float | None = None
    seal_rate_per_mm: float = 0.05
    seal_recovery_days: float = 30.0
    repellency_factor: float = 1.0

    def __attrs_post_init__(s):
        if s.seal_ksat_mm_h is not None and s.seal_ksat_mm_h <= 0:
            raise ValueError('seal_ksat_mm_h must be > 0 (or None to disable)')
        if s.seal_rate_per_mm <= 0 or s.seal_recovery_days <= 0:
            raise ValueError('seal_rate_per_mm and seal_recovery_days must be > 0')
        if not 0 < s.repellency_factor <= 1:
            raise ValueError('repellency_factor must be in (0, 1]')

    @property
    def active(s) -> bool:
        return s.seal_ksat_mm_h is not None or s.repellency_factor < 1


def seal_conductivity(ksat_top: np.ndarray, E: np.ndarray, p: SurfaceParams) -> np.ndarray:
    """Surface conductivity (mm h-1) for cumulative rain ``E`` since renewal."""
    if p.seal_ksat_mm_h is None:
        return ksat_top
    k_seal = np.minimum(p.seal_ksat_mm_h, ksat_top)      # a seal never conducts more than the soil
    return k_seal + (ksat_top - k_seal) * np.exp(-p.seal_rate_per_mm * E)


def repellency(theta_top, theta_dul_top, theta_l15_top, p: SurfaceParams) -> np.ndarray:
    """Factor in ``[repellency_factor, 1]`` rising linearly from wilting point to field capacity."""
    if p.repellency_factor >= 1:
        return np.ones_like(np.asarray(theta_top, dtype=float))
    span = np.maximum(theta_dul_top - theta_l15_top, 1e-6)
    wet = np.clip((theta_top - theta_l15_top) / span, 0.0, 1.0)
    return p.repellency_factor + (1.0 - p.repellency_factor) * wet


def surface_limit(ksat_top, E, theta_top, theta_dul_top, theta_l15_top, dt: float, p: SurfaceParams) -> np.ndarray:
    """Most water (mm) the surface can admit this step; ``inf`` when sealing and repellency are off."""
    if not p.active:
        return np.full(np.shape(ksat_top), np.inf)
    return seal_conductivity(ksat_top, E, p) * repellency(theta_top, theta_dul_top, theta_l15_top, p) * dt


def update_seal(E: np.ndarray, rain_at_surface: np.ndarray, dt: float, p: SurfaceParams) -> None:
    """Advance the seal state in place: rain adds to ``E``; rain-free steps relax it."""
    if p.seal_ksat_mm_h is None:
        return
    wet = rain_at_surface > 0
    E[wet] += rain_at_surface[wet]
    E[~wet] *= np.exp(-dt / (24.0 * p.seal_recovery_days))


# -- tests -----------------------------------------------------------------------

def test_disabled_is_transparent():
    p = SurfaceParams()
    k = np.array([50.0, 10.0]); E = np.array([0.0, 100.0])
    lim = surface_limit(k, E, np.array([0.2, 0.2]), np.array([0.3, 0.3]), np.array([0.1, 0.1]), 1.0, p)
    update_seal(E, np.array([5.0, 0.0]), 1.0, p)
    return not p.active and np.all(np.isinf(lim)) and np.allclose(E, [0.0, 100.0])


def test_seal_develops_with_rain_and_recovers_when_dry():
    p = SurfaceParams(seal_ksat_mm_h=5.0, seal_rate_per_mm=0.05, seal_recovery_days=10.0)
    k = np.array([50.0]); E = np.zeros(1)
    k0 = seal_conductivity(k, E, p)[0]
    update_seal(E, np.array([60.0]), 1.0, p)                 # 60 mm of rain -> ~95% sealed
    k60 = seal_conductivity(k, E, p)[0]
    for _ in range(24 * 10):                                  # ten dry days -> E back to 1/e
        update_seal(E, np.zeros(1), 1.0, p)
    recovered = abs(E[0] - 60.0 * np.exp(-1)) < 1e-9
    return abs(k0 - 50.0) < 1e-12 and abs(k60 - (5.0 + 45.0 * np.exp(-3.0))) < 1e-9 and k60 < 8.0 and recovered


def test_seal_never_exceeds_soil_ksat():
    p = SurfaceParams(seal_ksat_mm_h=20.0)
    return seal_conductivity(np.array([5.0]), np.array([1e3]), p)[0] <= 5.0


def test_repellency_rises_from_wilting_to_field_capacity():
    p = SurfaceParams(repellency_factor=0.2)
    r = repellency(np.array([0.05, 0.10, 0.20, 0.30, 0.40]), 0.30, 0.10, p)
    return np.allclose(r, [0.2, 0.2, 0.6, 1.0, 1.0]) and p.active


def test_bad_params_raise():
    for kw in ({'seal_ksat_mm_h': 0.0}, {'seal_rate_per_mm': -1}, {'repellency_factor': 0.0}, {'repellency_factor': 1.5}):
        try:
            SurfaceParams(**kw)
            return False
        except ValueError:
            pass
    return True


def test():
    return all([test_disabled_is_transparent(), test_seal_develops_with_rain_and_recovers_when_dry(),
                test_seal_never_exceeds_soil_ksat(), test_repellency_rises_from_wilting_to_field_capacity(),
                test_bad_params_raise()])


if __name__ == '__main__':
    print(test())
