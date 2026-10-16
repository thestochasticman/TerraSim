"""Soil layer geometry, kept interchangeable with DAESIM's ``SoilLayers``.

DAESIM2 (``daesim.soillayers.SoilLayers``) discretises the profile with one
of three methods — ``fixed`` (equal thickness), ``scaled_exp`` (a thin top
layer then exponentially growing layers) and ``horizon`` (explicit
thicknesses). :class:`LayerSpec` reproduces those formulas exactly so a
terrasim run can be built from a DAESIM layer spec (or vice versa) and
per-layer moisture lines up without interpolation.

:func:`overlap_matrix` / :func:`relayer` move water-content profiles
between two discretisations by thickness-weighted averaging, which
conserves total stored water wherever the two profiles cover the same
depth range. All functions are pure numpy — no I/O.
"""
from hashlib import sha256

import numpy as np
from attrs import frozen

DISCRETISE_METHODS = ('fixed', 'scaled_exp', 'horizon')


@frozen
class LayerSpec:
    """Vertical discretisation of the soil column.

    Attributes:
        nlayer: Number of layers (DAESIM ``nlevmlsoil``).
        z_max: Total profile depth in metres (DAESIM ``z_max``).
        method: ``'fixed'`` | ``'scaled_exp'`` | ``'horizon'`` (DAESIM
            ``discretise_method``).
        z_top: Top-layer thickness (m) for ``scaled_exp``.
        z_horizon: Explicit layer thicknesses (m) for ``horizon``; their sum
            must not exceed ``z_max``.

    The default is six horizon layers to 2 m (5, 10, 15, 30, 40, 100 cm),
    thin near the surface where infiltration and evaporation act. DAESIM's
    own default is :data:`DAESIM_DEFAULT` (5 x 0.2 m).

    Example:
        ```python
        spec = LayerSpec()
        spec.thicknesses()   # array([0.05, 0.1, 0.15, 0.3, 0.4, 1.0])
        spec.bounds_cm()     # [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
        ```
    """

    nlayer: int = 6
    z_max: float = 2.0
    method: str = 'horizon'
    z_top: float | None = None
    z_horizon: tuple[float, ...] | None = (0.05, 0.10, 0.15, 0.30, 0.40, 1.00)

    def __attrs_post_init__(s):
        if s.method not in DISCRETISE_METHODS:
            raise ValueError(f'method must be one of {DISCRETISE_METHODS}, got {s.method!r}')
        if s.nlayer < 1:
            raise ValueError('nlayer must be >= 1')
        if s.method == 'scaled_exp':
            if s.z_top is None or not (0 < s.z_top < s.z_max):
                raise ValueError('scaled_exp needs 0 < z_top < z_max')
            if s.nlayer < 2:
                raise ValueError('scaled_exp needs nlayer >= 2')
        if s.method == 'horizon':
            if s.z_horizon is None or len(s.z_horizon) != s.nlayer:
                raise ValueError('horizon needs z_horizon with exactly nlayer entries')
            if any(z <= 0 for z in s.z_horizon):
                raise ValueError('z_horizon thicknesses must be positive')
            if sum(s.z_horizon) > s.z_max + 1e-9:
                raise ValueError(f'sum(z_horizon)={sum(s.z_horizon)} exceeds z_max={s.z_max}')

    # -- geometry ----------------------------------------------------------

    def thicknesses(s) -> np.ndarray:
        """Layer thicknesses in metres (DAESIM ``z_soil``), shape ``(nlayer,)``."""
        if s.method == 'fixed':
            return np.full(s.nlayer, s.z_max / s.nlayer)
        if s.method == 'scaled_exp':
            w = np.exp(np.linspace(0, 1, s.nlayer - 1))
            w = w / w.sum()
            return np.concatenate([[s.z_top], w * (s.z_max - s.z_top)])
        return np.asarray(s.z_horizon, dtype=float)

    def bottoms(s) -> np.ndarray:
        """Cumulative depth of each layer's bottom (DAESIM ``d_soil``), metres."""
        return np.cumsum(s.thicknesses())

    def tops(s) -> np.ndarray:
        b = s.bottoms()
        return np.concatenate([[0.0], b[:-1]])

    def bounds_m(s) -> np.ndarray:
        """``(nlayer, 2)`` array of (top, bottom) depths in metres."""
        return np.column_stack([s.tops(), s.bottoms()])

    def bounds_cm(s) -> list[tuple[int, int]]:
        """Integer (top, bottom) in cm — the labels DAESIM's forcing columns use."""
        return [(int(round(t * 100)), int(round(b * 100))) for t, b in s.bounds_m()]

    def depth(s) -> float:
        """Depth of the bottom of the last layer (m); equals z_max unless horizons are shorter."""
        return float(s.bottoms()[-1])

    def key(s) -> str:
        """Short stable hash of the geometry, for cache names."""
        payload = ','.join(f'{z:.6f}' for z in s.thicknesses())
        return sha256(payload.encode()).hexdigest()[:10]

    # -- DAESIM interop ----------------------------------------------------

    @classmethod
    def from_daesim(cls, soil_layers) -> 'LayerSpec':
        """Build from a ``daesim.soillayers.SoilLayers`` (duck-typed: reads
        ``nlevmlsoil``, ``z_max``, ``discretise_method``, ``z_top``, ``z_horizon``)."""
        method = getattr(soil_layers, 'discretise_method', 'fixed')
        z_horizon = getattr(soil_layers, 'z_horizon', None)
        return cls(
            nlayer=int(soil_layers.nlevmlsoil),
            z_max=float(soil_layers.z_max),
            method=method,
            z_top=getattr(soil_layers, 'z_top', None) if method == 'scaled_exp' else None,
            z_horizon=tuple(float(z) for z in z_horizon) if (method == 'horizon' and z_horizon is not None) else None,
        )

    def to_daesim_kwargs(s) -> dict:
        """Keyword arguments for ``daesim.soillayers.SoilLayers(...)``."""
        kw = {'nlevmlsoil': s.nlayer, 'z_max': s.z_max, 'discretise_method': s.method}
        if s.method == 'scaled_exp':
            kw['z_top'] = s.z_top
        if s.method == 'horizon':
            kw['z_horizon'] = list(s.z_horizon)
        return kw


DAESIM_DEFAULT = LayerSpec(nlayer=5, z_max=1.0, method='fixed', z_horizon=None)


# -- relayering ----------------------------------------------------------------

def overlap_matrix(src_bounds_m, dst_bounds_m, extend_last: bool = True) -> np.ndarray:
    """Thickness-weighted mapping from source layers to destination layers.

    Returns ``M`` of shape ``(ndst, nsrc)`` with ``theta_dst = M @ theta_src``:
    each row holds the fraction of the destination layer overlapped by each
    source layer, so rows sum to 1 wherever the destination lies within the
    source profile. With ``extend_last`` the deepest source layer is treated
    as extending indefinitely, so destinations below the source profile take
    its value (rows still sum to 1).
    """
    src = np.asarray(src_bounds_m, dtype=float).reshape(-1, 2).copy()
    dst = np.asarray(dst_bounds_m, dtype=float).reshape(-1, 2)
    if extend_last:
        src[-1, 1] = np.inf
    top = np.maximum(dst[:, None, 0], src[None, :, 0])
    bot = np.minimum(dst[:, None, 1], src[None, :, 1])
    overlap = np.clip(bot - top, 0, None)
    thick = (dst[:, 1] - dst[:, 0])[:, None]
    return overlap / thick


def relayer(theta: np.ndarray, src: LayerSpec, dst: LayerSpec, axis: int = 0) -> np.ndarray:
    """Re-express a per-layer profile on another discretisation (water-conserving
    where the depth ranges coincide). ``theta`` has the layer dimension on ``axis``."""
    m = overlap_matrix(src.bounds_m(), dst.bounds_m())
    moved = np.moveaxis(np.asarray(theta), axis, 0)
    out = np.tensordot(m, moved, axes=(1, 0))
    return np.moveaxis(out, 0, axis)


# -- tests -----------------------------------------------------------------------

def test_fixed_matches_daesim():
    spec = LayerSpec(nlayer=5, z_max=1.0, method='fixed', z_horizon=None)
    return np.allclose(spec.thicknesses(), 0.2) and np.allclose(spec.bottoms(), [0.2, 0.4, 0.6, 0.8, 1.0])


def test_scaled_exp_matches_daesim():
    n, z_max, z_top = 5, 1.0, 0.1
    spec = LayerSpec(nlayer=n, z_max=z_max, method='scaled_exp', z_top=z_top, z_horizon=None)
    # DAESIM: z_soil = [z_top] + list(normalised exp(linspace(0,1,n-1)) * (z_max - z_top))
    w = np.exp(np.linspace(0, 1, n - 1)); w = w / w.sum()
    expected = np.concatenate([[z_top], w * (z_max - z_top)])
    return np.allclose(spec.thicknesses(), expected) and abs(spec.depth() - z_max) < 1e-12


def test_horizon_default_and_bounds_cm():
    spec = LayerSpec()
    return spec.bounds_cm() == [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]


def test_from_daesim_roundtrip():
    class Stub:
        nlevmlsoil = 4; z_max = 1.2; discretise_method = 'horizon'; z_top = None
        z_horizon = [0.1, 0.2, 0.3, 0.6]
    spec = LayerSpec.from_daesim(Stub())
    kw = spec.to_daesim_kwargs()
    return kw == {'nlevmlsoil': 4, 'z_max': 1.2, 'discretise_method': 'horizon', 'z_horizon': [0.1, 0.2, 0.3, 0.6]}


def test_overlap_rows_sum_to_one_and_conserve_water():
    src, dst = LayerSpec(), DAESIM_DEFAULT
    m = overlap_matrix(src.bounds_m(), dst.bounds_m())
    theta = np.array([0.40, 0.35, 0.30, 0.28, 0.25, 0.22])
    out = relayer(theta, src, dst)
    # DAESIM layers cover 0-1 m; terrasim layers 0-1 m hold the same water.
    water_src = float(np.sum(theta[:5] * src.thicknesses()[:5]))
    water_dst = float(np.sum(out * dst.thicknesses()))
    return np.allclose(m.sum(axis=1), 1) and abs(water_src - water_dst) < 1e-12


def test_bad_horizon_raises():
    try:
        LayerSpec(nlayer=2, z_max=1.0, method='horizon', z_horizon=(0.6, 0.6))
    except ValueError:
        return True
    return False


def test():
    return all([
        test_fixed_matches_daesim(),
        test_scaled_exp_matches_daesim(),
        test_horizon_default_and_bounds_cm(),
        test_from_daesim_roundtrip(),
        test_overlap_rows_sum_to_one_and_conserve_water(),
        test_bad_horizon_raises(),
    ])


if __name__ == '__main__':
    print(test())
