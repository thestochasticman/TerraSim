"""Vertical gravity drainage between layers (Campbell K(theta), explicit, limited).

Water above field capacity drains downward at the unsaturated conductivity
``K(theta)`` of the donor layer, but no faster than the receiver's saturated
conductivity (a clay layer under a sandy one throttles percolation and
perches water above it), limited by what the donor holds above DUL and by
the room left in the receiver. Processing bottom-up within a
sub-step means a receiver has already made room before its donor drains.
The number of sub-steps is chosen so no layer moves more than
``drain_frac_max`` of its water-equivalent thickness per sub-step — a
Courant-like bound that keeps the explicit scheme accurate; the limiters
alone keep it bounded. The bottom layer drains to deep drainage, scaled by
``k_bottom_scale`` (<1 over bedrock).
"""
import math

import numpy as np

from terrasim.physics import campbell


def substeps(theta, static, dt: float, drain_frac_max: float, max_substeps: int) -> int:
    """Sub-step count so ``K(theta) dt_s <= drain_frac_max * dz`` everywhere."""
    k = campbell.k(theta, static.theta_s, static.ksat, static.b)
    ratio = (k * dt) / (drain_frac_max * static.dz_mm[:, None, None])
    worst = float(np.nanmax(ratio)) if ratio.size else 0.0
    return int(min(max(1, math.ceil(worst)), max_substeps))


def drain(theta: np.ndarray, static, dt: float, drain_frac_max: float = 0.1, max_substeps: int = 24):
    """Redistribute water downward for ``dt`` hours. Mutates ``theta`` (L,Y,X)
    and returns ``(deep_drain (Y,X) mm, n_substeps)``."""
    L = theta.shape[0]
    n = substeps(theta, static, dt, drain_frac_max, max_substeps)
    dts = dt / n
    dz = static.dz_mm
    deep = np.zeros(theta.shape[1:])
    for _ in range(n):
        for i in range(L - 1, -1, -1):
            k = campbell.k(theta[i], static.theta_s[i], static.ksat[i], static.b[i])
            if i < L - 1:
                k = np.minimum(k, static.ksat[i + 1])          # interface limited by the slower receiver
            q = np.minimum(k * dts, np.clip(theta[i] - static.theta_dul[i], 0.0, None) * dz[i])
            if i == L - 1:
                q = q * static.k_bottom_scale
                deep += q
            else:
                room = np.clip((static.theta_s[i + 1] - theta[i + 1]) * dz[i + 1], 0.0, None)
                q = np.minimum(q, room)
                theta[i + 1] += q / dz[i + 1]
            theta[i] -= q / dz[i]
    return deep, n


# -- tests -----------------------------------------------------------------------

def _static(ksat=20.0, nlayer=4, ny=2, nx=2):
    from terrasim.layers import LayerSpec
    from terrasim.physics.state import StaticArrays
    return StaticArrays.uniform(LayerSpec(nlayer=nlayer, z_max=1.0, method='fixed', z_horizon=None), ny, nx, ksat=ksat)


def test_conserves_mass_and_bounds():
    st = _static(ksat=500.0)
    rng = np.random.default_rng(0)
    theta = st.theta_dul + rng.uniform(0, 1, st.theta_s.shape) * (st.theta_s - st.theta_dul)
    total0 = np.tensordot(st.dz_mm, theta, axes=(0, 0))
    deep_total = np.zeros(st.shape)
    for _ in range(1000):
        deep, _ = drain(theta, st, 1.0)
        deep_total += deep
        if theta.max() > st.theta_s.max() + 1e-12 or theta.min() < 0:
            return False
    total1 = np.tensordot(st.dz_mm, theta, axes=(0, 0))
    return np.allclose(total0, total1 + deep_total, atol=1e-9)


def test_drains_to_dul_and_stops():
    st = _static(ksat=50.0)
    theta = st.theta_s.copy()
    for _ in range(2000):
        drain(theta, st, 1.0)
    at_dul = np.allclose(theta, st.theta_dul, atol=1e-6)
    deep, _ = drain(theta, st, 1.0)
    return at_dul and np.allclose(deep, 0.0, atol=1e-9)


def test_slow_receiver_throttles_percolation():
    from attrs import evolve
    st = _static(ksat=50.0, nlayer=3, ny=1, nx=1)
    ksat = st.ksat.copy(); ksat[1] = 1.0                      # clay in the middle
    st = evolve(st, ksat=ksat)
    theta = st.theta_dul.copy(); theta[0] = st.theta_s[0]      # saturated topsoil
    deep, _ = drain(theta, st, 1.0)
    moved = (st.theta_s[0, 0, 0] - theta[0, 0, 0]) * st.dz_mm[0]
    return abs(moved - 1.0) < 1e-9 and theta[0, 0, 0] > st.theta_dul[0, 0, 0]


def test_bedrock_scale_limits_deep_drainage():
    st_open = _static(ksat=50.0)
    from attrs import evolve
    st_rock = evolve(st_open, k_bottom_scale=np.full(st_open.shape, 0.05))
    th1, th2 = st_open.theta_s.copy(), st_open.theta_s.copy()
    d1, _ = drain(th1, st_open, 1.0)
    d2, _ = drain(th2, st_rock, 1.0)
    return bool(np.all(d2 < d1)) and np.allclose(d2, 0.05 * d1)


def test():
    return all([test_conserves_mass_and_bounds(), test_drains_to_dul_and_stops(), test_slow_receiver_throttles_percolation(),
                test_bedrock_scale_limits_deep_drainage()])


if __name__ == '__main__':
    print(test())
