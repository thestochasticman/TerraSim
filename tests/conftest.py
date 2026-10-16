"""Shared fixtures: an isolated Troi and synthetic inputs.

Every fixture is offline — no network, no credentials. ``tmp_troi`` points
the Troi registry and all cache/output directories at a per-test temporary
directory so tests never touch ``~/.config/Troi.json`` or real data roots.
"""
from datetime import date

import numpy as np
import pytest

BBOX = [148.36265, -33.52606, 148.38265, -33.50606]   # ~2 km near Grenfell, NSW


@pytest.fixture
def tmp_config(tmp_path):
    from troi import Config
    return Config(out_dir=str(tmp_path / 'out'), tmp_dir=str(tmp_path / 'tmp'))


@pytest.fixture
def tmp_troi(tmp_config):
    from troi import Troi
    return Troi(bbox=BBOX, start=date(2022, 1, 1), end=date(2022, 3, 31),
                stub='pytest_troi', config=tmp_config)


def uniform_soil(nlayer, ny, nx, theta_s=0.45, dul=0.30, l15=0.15, ksat=20.0, b=5.0, psi_e=200.0):
    """Homogeneous soil hydraulic arrays shaped (nlayer, ny, nx)."""
    full = lambda v: np.full((nlayer, ny, nx), v, dtype=float)
    return dict(theta_s=full(theta_s), theta_dul=full(dul), theta_l15=full(l15),
                ksat=full(ksat), b=full(b), psi_e=full(psi_e))
