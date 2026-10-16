"""Evapotranspiration partitioning: FAO-56 dual crop coefficient.

Reference ET (``ET0``, mm per step, already distributed through the day by
the forcing builder) is split into

- **soil evaporation** ``E = Ke * ET0`` from the evaporation layer ``Ze``,
  with ``Ke = min(Kr (Kc_max - Kcb), few Kc_max)``; ``Kr`` is 1 while the
  depletion ``De`` of the Ze layer is below the readily evaporable water
  (REW) and falls linearly to 0 at the total evaporable water (TEW);
  ``few = 1 - fc`` is the exposed, wetted soil fraction;
- **transpiration** ``T = Kcb * ET0`` taken from the layers in proportion
  to root fraction, each layer limited by a linear stress factor ``Ks``
  that is 1 above ``theta_crit = L15 + (1 - p)(DUL - L15)`` and 0 at L15.
  One compensation pass lets unstressed layers take up demand that
  stressed layers could not meet.

An external per-layer uptake array (e.g. from DAESIM's plant hydraulics)
can replace the FAO-56 transpiration; it is clipped at L15 and the
shortfall reported.
"""
import numpy as np
from attrs import frozen


@frozen
class FAO56DualCrop:
    kc_max: float = 1.20
    p_depletion: float = 0.5

    def demand(s, theta, static, kcb, fc, et0):
        """Potential soil evaporation and transpiration (mm per step, (Y,X))."""
        ze = static.ze_mm[:, None, None]
        de = np.clip(np.sum(ze * (static.theta_dul - theta), axis=0), 0.0, static.tew)
        span = np.maximum(static.tew - static.rew, 1e-9)
        kr = np.where(de <= static.rew, 1.0, (static.tew - de) / span)
        ke = np.minimum(kr * np.clip(s.kc_max - kcb, 0.0, None), (1.0 - fc) * s.kc_max)
        return ke * et0, kcb * et0

    def evaporate(s, theta, static, e_pot):
        """Remove soil evaporation from the Ze layers; mutate theta; return E (mm)."""
        ze = static.ze_mm[:, None, None]
        avail = np.clip(theta - 0.5 * static.theta_l15, 0.0, None) * ze
        total = avail.sum(axis=0)
        e = np.minimum(e_pot, total)
        share = np.where(total > 0, avail / np.where(total > 0, total, 1.0), 0.0)
        theta -= e * share / static.dz_mm[:, None, None]
        return e

    def transpire(s, theta, static, t_pot, uptake=None):
        """Remove transpiration per layer; mutate theta; return T (L,Y,X) mm.

        With ``uptake`` (L,Y,X) given, it replaces the FAO-56 demand and is
        clipped to the water available above L15 in each layer."""
        dz = static.dz_mm[:, None, None]
        avail = np.clip(theta - static.theta_l15, 0.0, None) * dz
        if uptake is not None:
            t = np.minimum(np.clip(uptake, 0.0, None), avail)
        else:
            crit = static.theta_l15 + (1.0 - s.p_depletion) * (static.theta_dul - static.theta_l15)
            ks = np.clip((theta - static.theta_l15) / np.maximum(crit - static.theta_l15, 1e-9), 0.0, 1.0)
            t = np.minimum(t_pot * static.root_frac * ks, avail)
            unmet = np.clip(t_pot - t.sum(axis=0), 0.0, None)
            # compensation: unstressed layers (ks == 1) with roots share the shortfall
            cap = np.where(ks >= 1.0, static.root_frac, 0.0)
            cap_sum = cap.sum(axis=0)
            extra = np.where(cap_sum > 0, unmet * cap / np.where(cap_sum > 0, cap_sum, 1.0), 0.0)
            t = np.minimum(t + extra, avail)
        theta -= t / dz
        return t


METHODS = {'fao56_dual': FAO56DualCrop}


# -- tests -----------------------------------------------------------------------

def _setup(theta_val=0.30, ny=1, nx=1):
    from terrasim.layers import LayerSpec
    from terrasim.physics.state import StaticArrays
    st = StaticArrays.uniform(LayerSpec(), ny, nx)
    return st, np.full(st.theta_s.shape, theta_val)


def test_ke_is_kcmax_minus_kcb_when_wet():
    st, theta = _setup(theta_val=0.30)   # at DUL -> De = 0 -> Kr = 1
    et = FAO56DualCrop()
    e_pot, t_pot = et.demand(theta, st, kcb=np.full(st.shape, 0.3), fc=np.zeros(st.shape), et0=np.ones(st.shape))
    return np.allclose(e_pot, 0.9) and np.allclose(t_pot, 0.3)


def test_no_evaporation_below_half_l15():
    st, theta = _setup(theta_val=0.07)   # below 0.5 * L15 (0.075)
    e = FAO56DualCrop().evaporate(theta, st, np.ones(st.shape))
    return np.allclose(e, 0.0)


def test_no_transpiration_at_l15():
    st, theta = _setup(theta_val=0.15)
    t = FAO56DualCrop().transpire(theta, st, np.ones(st.shape))
    return np.allclose(t, 0.0)


def test_transpiration_meets_demand_when_wet_and_compensates():
    st, theta = _setup(theta_val=0.30)
    theta[0] = 0.15            # top layer at wilting point: cannot contribute
    t = FAO56DualCrop().transpire(theta, st, np.full(st.shape, 2.0))
    return np.allclose(t.sum(axis=0), 2.0) and np.allclose(t[0], 0.0)


def test_external_uptake_used_and_clipped():
    st, theta = _setup(theta_val=0.16)
    uptake = np.full(theta.shape, 50.0)
    t = FAO56DualCrop().transpire(theta, st, np.zeros(st.shape), uptake=uptake)
    avail = (0.16 - 0.15) * st.dz_mm
    return np.allclose(t[:, 0, 0], avail) and np.allclose(theta, 0.15)


def test():
    return all([
        test_ke_is_kcmax_minus_kcb_when_wet(),
        test_no_evaporation_below_half_l15(),
        test_no_transpiration_at_l15(),
        test_transpiration_meets_demand_when_wet_and_compensates(),
        test_external_uptake_used_and_clipped(),
    ])


if __name__ == '__main__':
    print(test())
