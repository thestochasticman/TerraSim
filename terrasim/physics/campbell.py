"""Campbell (1974) soil water retention and conductivity — shared with DAESIM.

    psi(theta) = psi_e * (theta / theta_s) ** (-b)        (suction, same units as psi_e)
    K(theta)   = K_sat * (theta / theta_s) ** (2*b + 3)

DAESIM writes K in terms of potential, ``K_sat (psi_e/psi)^(2 + 3/b)``, which
is the same curve. Pure numpy; every argument broadcasts.
"""
import numpy as np


def psi(theta, theta_s, psi_e, b):
    """Matric suction (positive, units of ``psi_e``); never below air entry."""
    s = np.clip(np.asarray(theta, dtype=float) / theta_s, 1e-6, 1.0)
    return psi_e * s ** (-b)


def k(theta, theta_s, ksat, b):
    """Unsaturated hydraulic conductivity (units of ``ksat``)."""
    s = np.clip(np.asarray(theta, dtype=float) / theta_s, 0.0, 1.0)
    return ksat * s ** (2.0 * b + 3.0)


def test_saturated_values():
    return np.isclose(k(0.45, 0.45, 20.0, 5.0), 20.0) and np.isclose(psi(0.45, 0.45, 200.0, 5.0), 200.0)


def test_monotone():
    th = np.linspace(0.05, 0.45, 50)
    return bool(np.all(np.diff(k(th, 0.45, 20.0, 5.0)) > 0)) and bool(np.all(np.diff(psi(th, 0.45, 200.0, 5.0)) < 0))


def test():
    return all([test_saturated_values(), test_monotone()])


if __name__ == '__main__':
    print(test())
