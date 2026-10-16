"""Canopy interception: a small store that fills from rain and evaporates first.

Capacity scales with cover fraction (``canopy_store_mm_per_fc * fc``).
Intercepted water evaporates at the reference rate before any soil
evaporation or transpiration, reducing the residual demand.
"""
import numpy as np


def intercept(rain, canopy, fc, et0, store_mm_per_fc: float = 1.0):
    """Mutates ``canopy``; returns ``(throughfall, interception_evap, et0_residual)``."""
    smax = store_mm_per_fc * fc
    fill = np.clip(np.minimum(rain, smax - canopy), 0.0, None)
    canopy += fill
    throughfall = rain - fill
    ei = np.minimum(canopy, et0)
    canopy -= ei
    return throughfall, ei, et0 - ei


def test_fills_then_evaporates():
    canopy = np.zeros(2)
    tf, ei, res = intercept(np.array([3.0, 0.2]), canopy, np.array([0.5, 0.5]), np.array([0.3, 0.3]))
    # capacity 0.5 mm: pixel 0 stores 0.5, passes 2.5; evaporates 0.3 -> 0.2 left
    return np.allclose(tf, [2.5, 0.0]) and np.allclose(ei, [0.3, 0.2]) and np.allclose(canopy, [0.2, 0.0]) and np.allclose(res, [0.0, 0.1])


def test():
    return test_fills_then_evaporates()


if __name__ == '__main__':
    print(test())
