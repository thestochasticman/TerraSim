"""Surface infiltration: how much of the water standing on a pixel enters the soil.

Two formulations behind one function signature:

- :class:`GreenAmpt` — Green-Ampt with the time-compression approximation
  (Mein & Larson 1973): the capacity over a step follows the implicit
  cumulative-infiltration relation ``F - S_f ln(1 + F/S_f) = K_sat t``, with
  ``S_f = psi_f * (theta_s - theta_top)`` the sorptivity-like term. When
  supply is below capacity everything infiltrates and ``F`` advances by the
  supply; otherwise the capacity increment infiltrates and the rest ponds /
  runs off. ``F`` resets after a dry spell (a new storm).
- :class:`Capacity` — a plain ``min(supply, K_sat * dt)`` limit.

Infiltrated water is then placed in the column by :func:`cascade` — fill
the top layer to saturation, spill the remainder into the next, and so on;
whatever cannot fit anywhere comes back to the surface as saturation
excess. Together with the drainage limiters this keeps ``theta <= theta_s``
without any clipping.

All functions take flat 1-D arrays over the pixels being updated (so the
router can re-run infiltration on a subset) and mutate ``theta`` in place.
"""
import numpy as np
from attrs import frozen

from terrasim.physics import campbell


def cascade(theta: np.ndarray, dz_mm: np.ndarray, theta_s: np.ndarray, supply: np.ndarray,
            ksat: np.ndarray | None = None, dt: float | None = None) -> np.ndarray:
    """Place ``supply`` (mm per pixel) in the column top-down; mutate ``theta``
    (``(L, n)``) and return the saturation excess (mm) that comes back to the surface.

    Each layer stores up to saturation; what it cannot hold is passed to the
    layer below, but — when ``ksat`` and ``dt`` are given — no faster than that
    lower layer's saturated conductivity allows (``ksat[i+1] * dt``). A slow
    subsoil under a saturated topsoil therefore perches water and produces
    saturation-excess runoff, instead of the profile filling instantly."""
    inflow = supply.astype(float).copy()
    excess = np.zeros_like(inflow)
    L = theta.shape[0]
    for i in range(L):
        room = np.clip((theta_s[i] - theta[i]) * dz_mm[i], 0.0, None)
        store = np.minimum(inflow, room)
        theta[i] += store / dz_mm[i]
        rest = inflow - store
        if i < L - 1:
            cap = ksat[i + 1] * dt if (ksat is not None and dt is not None) else rest
            passed = np.minimum(rest, cap)
            excess += rest - passed
            inflow = passed
        else:
            excess += rest
    return excess


def _green_ampt_capacity(F: np.ndarray, S_f: np.ndarray, ksat: np.ndarray, dt: float, iters: int = 6) -> np.ndarray:
    """Cumulative infiltration after ``dt`` hours of unlimited supply, from
    ``F`` (mm): solves ``G(F') = F' - S_f ln(1 + F'/S_f) - RHS = 0`` by Newton
    (``G' = F'/(S_f + F')``, monotone). Where ``S_f`` ~ 0 the limit is ``F + K dt``."""
    kdt = ksat * dt
    small = S_f < 1e-9
    S = np.where(small, 1.0, S_f)
    rhs = F - S * np.log1p(F / S) + kdt
    # Start from the explicit-rate estimate; Newton on a convex increasing G converges monotonically.
    Fn = F + kdt * (1.0 + S / np.maximum(F, 0.1))
    for _ in range(iters):
        g = Fn - S * np.log1p(Fn / S) - rhs
        dg = Fn / (S + Fn)
        Fn = np.maximum(Fn - g / np.maximum(dg, 1e-12), F)
    return np.where(small, F + kdt, Fn)


@frozen
class GreenAmpt:
    """Green-Ampt infiltration with event reset. ``dry_reset_steps`` steps
    without infiltration start a new event (``F_cum = 0``)."""

    dry_reset_steps: int = 6

    def infiltrate(s, supply, theta, dz_mm, theta_s, ksat, psi_f, F_cum, t_dry, dt):
        """Infiltrate ``supply`` (mm, flat over pixels) into the column.

        Mutates ``theta`` (L, n), ``F_cum`` and ``t_dry`` (n,) in place and
        returns ``(infiltrated, excess)`` in mm.
        """
        new_event = t_dry >= s.dry_reset_steps
        F_cum[new_event] = 0.0
        d_theta = np.clip(theta_s[0] - theta[0], 1e-4, None)
        S_f = psi_f * d_theta
        F_new = _green_ampt_capacity(F_cum, S_f, ksat[0], dt)
        capacity = np.maximum(F_new - F_cum, 0.0)
        infil = np.minimum(supply, capacity)
        unstored = cascade(theta, dz_mm, theta_s, infil, ksat, dt)
        infil = infil - unstored
        excess = supply - infil
        F_cum += infil
        t_dry[:] = np.where(infil > 0, 0, t_dry + 1)
        return infil, excess


@frozen
class Capacity:
    """``min(supply, K_sat dt)`` infiltration; same signature as :class:`GreenAmpt`."""

    def infiltrate(s, supply, theta, dz_mm, theta_s, ksat, psi_f, F_cum, t_dry, dt):
        infil = np.minimum(supply, ksat[0] * dt)
        unstored = cascade(theta, dz_mm, theta_s, infil, ksat, dt)
        infil = infil - unstored
        F_cum += infil
        t_dry[:] = np.where(infil > 0, 0, t_dry + 1)
        return infil, supply - infil


METHODS = {'green_ampt': GreenAmpt, 'capacity': Capacity}


# -- tests -----------------------------------------------------------------------

def _analytic_F(S_f, ksat, t):
    """Reference: solve F - S_f ln(1+F/S_f) = K t by bisection."""
    lo, hi = 0.0, ksat * t + 10 * S_f + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - S_f * np.log1p(mid / S_f) < ksat * t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_green_ampt_matches_analytic_under_ponding():
    # one pixel, huge top layer so theta (hence S_f) stays constant
    dz = np.array([1e12, 1e12]); theta_s = np.full((2, 1), 0.45); theta = np.full((2, 1), 0.25)
    ksat = np.full((2, 1), 10.0); psi_f = np.array([150.0])
    F = np.zeros(1); t_dry = np.zeros(1, dtype=np.int64)
    ga = GreenAmpt()
    dt, n = 0.25, 16
    for _ in range(n):
        ga.infiltrate(np.array([1e9]), theta, dz, theta_s, ksat, psi_f, F, t_dry, dt)
    ref = _analytic_F(150.0 * 0.2, 10.0, dt * n)
    return abs(F[0] - ref) < 1e-5


def test_light_rain_all_infiltrates():
    dz = np.array([50.0, 150.0]); theta_s = np.full((2, 3), 0.45); theta = np.full((2, 3), 0.30)
    ksat = np.full((2, 3), 10.0); psi_f = np.full(3, 150.0)
    F = np.zeros(3); t_dry = np.zeros(3, dtype=np.int64)
    infil, excess = GreenAmpt().infiltrate(np.full(3, 2.0), theta, dz, theta_s, ksat, psi_f, F, t_dry, 1.0)
    return np.allclose(infil, 2.0) and np.allclose(excess, 0.0) and np.allclose(theta[0], 0.30 + 2.0 / 50)


def test_cascade_spills_and_reports_saturation_excess():
    dz = np.array([50.0, 50.0]); theta_s = np.full((2, 1), 0.40); theta = np.full((2, 1), 0.38)
    left = cascade(theta, dz, theta_s, np.array([5.0]))
    # room = 1 mm per layer -> 2 mm stored, 3 mm back to surface
    return abs(left[0] - 3.0) < 1e-12 and np.allclose(theta, 0.40)


def test_slow_subsoil_perches_water_and_sheds_runoff():
    # 50 mm topsoil (Ksat 50) already near saturation over a 1 mm/h clay subsoil, 20 mm/h storm
    dz = np.array([50.0, 500.0]); theta_s = np.full((2, 1), 0.45); theta = np.array([[0.44], [0.30]])
    ksat = np.array([[50.0], [1.0]]); psi_f = np.array([100.0])
    F = np.zeros(1); t_dry = np.zeros(1, dtype=np.int64)
    infil, excess = GreenAmpt().infiltrate(np.array([20.0]), theta, dz, theta_s, ksat, psi_f, F, t_dry, 1.0)
    # topsoil can take 0.5 mm, subsoil accepts 1 mm/h -> 1.5 mm infiltrates, 18.5 mm runs off
    return abs(infil[0] - 1.5) < 1e-9 and abs(excess[0] - 18.5) < 1e-9 and abs(theta[0, 0] - 0.45) < 1e-12


def test_event_reset_after_dry_spell():
    dz = np.array([1e9]); theta_s = np.full((1, 1), 0.45); theta = np.full((1, 1), 0.25)
    ksat = np.full((1, 1), 10.0); psi_f = np.array([150.0])
    F = np.array([50.0]); t_dry = np.array([10], dtype=np.int64)
    GreenAmpt(dry_reset_steps=6).infiltrate(np.array([0.5]), theta, dz, theta_s, ksat, psi_f, F, t_dry, 1.0)
    return abs(F[0] - 0.5) < 1e-12 and t_dry[0] == 0


def test():
    return all([
        test_green_ampt_matches_analytic_under_ponding(),
        test_light_rain_all_infiltrates(),
        test_cascade_spills_and_reports_saturation_excess(),
        test_slow_subsoil_perches_water_and_sheds_runoff(),
        test_event_reset_after_dry_spell(),
    ])


if __name__ == '__main__':
    print(test())
