"""Static hydraulic arrays, the mutable model state, and per-step fluxes.

:class:`StaticArrays` is the engine's view of the static dataset: plain numpy
arrays, already on the model grid and layer geometry, in the engine's units
(mm, mm h-1, m3 m-3). :class:`State` is everything that carries from one
step to the next. :class:`Fluxes` is what one step reports.
"""
import numpy as np
from attrs import define, frozen, field

from terrasim.layers import LayerSpec
from terrasim.pedotransfer import root_fraction, wetting_front_suction, rew_tew


def ze_overlap_mm(layers: LayerSpec, ze_m: float) -> np.ndarray:
    """Millimetres of each layer lying within the evaporation depth ``ze_m``."""
    b = layers.bounds_m()
    return np.clip(np.minimum(b[:, 1], ze_m) - np.minimum(b[:, 0], ze_m), 0, None) * 1000.0


@frozen
class StaticArrays:
    """Hydraulic parameters on the model grid (``L`` layers, ``Y`` x ``X`` pixels).

    Attributes:
        dz_mm: Layer thicknesses (L,), mm.
        theta_s, theta_dul, theta_l15: Saturation, field capacity, wilting point (L,Y,X), m3 m-3.
        ksat: Saturated conductivity (L,Y,X), mm h-1.
        b, psi_e: Campbell shape parameter and air-entry suction (L,Y,X); psi_e in mm.
        psi_f: Green-Ampt wetting-front suction of the top layer (Y,X), mm.
        rew, tew: FAO-56 readily / total evaporable water (Y,X), mm.
        ze_mm: Part of each layer inside the evaporation depth (L,), mm.
        root_frac: Root fraction per layer (L,Y,X), sums to 1.
        k_bottom_scale: Multiplier on bottom-layer drainage (Y,X); <1 over bedrock.
        valid: Pixels with usable soil data (Y,X).
        ds_index, level_order, level_ptr: D8 routing geometry (flat), or None.
        twi: Topographic wetness index (Y,X) or None (used by the TWI router).
    """

    dz_mm: np.ndarray
    theta_s: np.ndarray
    theta_dul: np.ndarray
    theta_l15: np.ndarray
    ksat: np.ndarray
    b: np.ndarray
    psi_e: np.ndarray
    psi_f: np.ndarray
    rew: np.ndarray
    tew: np.ndarray
    ze_mm: np.ndarray
    root_frac: np.ndarray
    k_bottom_scale: np.ndarray
    valid: np.ndarray
    ds_index: np.ndarray | None = None
    level_order: np.ndarray | None = None
    level_ptr: np.ndarray | None = None
    twi: np.ndarray | None = None

    @property
    def nlayer(s) -> int:
        return s.theta_s.shape[0]

    @property
    def shape(s) -> tuple[int, int]:
        return s.theta_s.shape[1:]

    @classmethod
    def uniform(cls, layers: LayerSpec, ny: int, nx: int, theta_s=0.45, theta_dul=0.30, theta_l15=0.15,
                ksat=20.0, b=5.0, psi_e=200.0, ze_m=0.10, root_depth_m=1.0, root_beta=3.0,
                sand_frac=0.4, k_bottom_scale=1.0, **routing) -> 'StaticArrays':
        """A homogeneous soil — the workhorse for tests and idealised runs."""
        L = layers.nlayer
        full = lambda v: np.full((L, ny, nx), float(v))
        rf = root_fraction(layers.bounds_m(), root_depth_m, root_beta)
        rew, tew = rew_tew(sand_frac, theta_dul, theta_l15, ze_m)
        return cls(
            dz_mm=layers.thicknesses() * 1000.0,
            theta_s=full(theta_s), theta_dul=full(theta_dul), theta_l15=full(theta_l15),
            ksat=full(ksat), b=full(b), psi_e=full(psi_e),
            psi_f=np.full((ny, nx), float(wetting_front_suction(psi_e, b))),
            rew=np.full((ny, nx), float(rew)), tew=np.full((ny, nx), float(tew)),
            ze_mm=ze_overlap_mm(layers, ze_m),
            root_frac=np.broadcast_to(rf[:, None, None], (L, ny, nx)).copy(),
            k_bottom_scale=np.full((ny, nx), float(k_bottom_scale)),
            valid=np.ones((ny, nx), dtype=bool),
            **routing,
        )


@define
class State:
    """Everything carried between steps (float64).

    Attributes:
        theta: Volumetric water content (L,Y,X).
        pond: Surface detention (Y,X), mm.
        canopy: Intercepted water on the canopy (Y,X), mm.
        F_cum: Green-Ampt cumulative infiltration in the current event (Y,X), mm.
        t_dry: Consecutive steps without infiltration (Y,X).
    """

    theta: np.ndarray
    pond: np.ndarray
    canopy: np.ndarray
    F_cum: np.ndarray
    t_dry: np.ndarray

    @classmethod
    def initial(cls, static: StaticArrays, how: str = 'dul') -> 'State':
        if how == 'dul':
            theta = static.theta_dul.copy()
        elif how == 'midpoint':
            theta = 0.5 * (static.theta_dul + static.theta_l15)
        elif how == 'l15':
            theta = static.theta_l15.copy()
        else:
            raise ValueError(f'unknown init_theta {how!r}')
        ny, nx = static.shape
        z = lambda dt=float: np.zeros((ny, nx), dtype=dt)
        return cls(theta=theta.astype(float), pond=z(), canopy=z(), F_cum=z(), t_dry=z(np.int64))

    def water_mm(s, dz_mm: np.ndarray) -> np.ndarray:
        """Total stored water per pixel (soil + pond + canopy), mm."""
        return np.tensordot(dz_mm, s.theta, axes=(0, 0)) + s.pond + s.canopy

    def copy(s) -> 'State':
        return State(s.theta.copy(), s.pond.copy(), s.canopy.copy(), s.F_cum.copy(), s.t_dry.copy())


@define
class Fluxes:
    """Water fluxes of one step, mm per pixel (``transpiration`` is per layer).

    ``runoff_pass`` is water leaving a pixel's surface (to a downslope pixel
    or out of the domain); ``runoff_out`` is the part that left the domain;
    ``runon`` is what arrived from upslope; ``residual`` is the per-pixel
    mass-balance error.
    """

    rain: np.ndarray
    interception_evap: np.ndarray
    throughfall: np.ndarray
    infiltration: np.ndarray
    runon: np.ndarray
    runoff_pass: np.ndarray
    runoff_out: np.ndarray
    deep_drain: np.ndarray
    soil_evap: np.ndarray
    transpiration: np.ndarray
    residual: np.ndarray

    @property
    def et_actual(s) -> np.ndarray:
        return s.interception_evap + s.soil_evap + s.transpiration.sum(axis=0)

    SCALARS = ('rain', 'interception_evap', 'throughfall', 'infiltration', 'runon',
               'runoff_pass', 'runoff_out', 'deep_drain', 'soil_evap', 'residual')

    def add_(s, other: 'Fluxes') -> 'Fluxes':
        """In-place accumulation (for daily totals)."""
        for name in s.SCALARS:
            getattr(s, name).__iadd__(getattr(other, name))
        s.transpiration += other.transpiration
        return s

    @classmethod
    def zeros(cls, nlayer: int, ny: int, nx: int) -> 'Fluxes':
        z = lambda: np.zeros((ny, nx))
        return cls(z(), z(), z(), z(), z(), z(), z(), z(), z(), np.zeros((nlayer, ny, nx)), z())
