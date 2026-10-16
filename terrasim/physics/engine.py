"""One model step for every pixel at once, with a mass-balance check.

Operator splitting, in this fixed order each step:

1. interception (canopy store fills, evaporates at ET0),
2. ET demand (FAO-56 dual crop: Ke from Ze depletion, Kcb from NDVI),
3. surface infiltration of throughfall + pond (Green-Ampt), excess ponds up
   to ``pond_max`` then becomes local runoff,
4. routing of local runoff downslope with re-infiltration at receivers,
5. vertical drainage (Campbell K, limited, sub-stepped), deep drainage out
   of the bottom layer,
6. soil evaporation from the Ze layers,
7. transpiration by root fraction and stress (or an external uptake sink),
8. per-pixel mass balance: storage change must equal the net flux.

Everything is mm per step on ``(ny, nx)`` or ``(nlayer, ny, nx)`` arrays.
"""
import warnings

import numpy as np
from attrs import frozen, field

from terrasim.params import Params
from terrasim.terrasim import Terrasim, defaultterrasim
from terrasim.physics import drainage, infiltration, interception
from terrasim.physics.evapotranspiration import FAO56DualCrop
from terrasim.physics.routing import NullRouter, D8Router, TWIRouter
from terrasim.physics.state import StaticArrays, State, Fluxes


@frozen
class StepForcing:
    """Forcing for one step, already on the model grid (mm / dimensionless)."""

    rain: np.ndarray
    et0: np.ndarray
    kcb: np.ndarray
    fc: np.ndarray


def _router_for(settings: Terrasim, static: StaticArrays):
    if settings.routing == 'none' or static.ds_index is None:
        if settings.routing != 'none':
            warnings.warn('routing requested but static has no flow geometry; using NullRouter')
        return NullRouter()
    if settings.routing == 'd8':
        return D8Router(static.ds_index, static.level_order, static.level_ptr)
    raise ValueError('TWI routing requires TWIRouter.from_terrain(); build it and pass router= explicitly')


@frozen
class Engine:
    """Steps a :class:`State` forward over a :class:`StaticArrays` grid."""

    static: StaticArrays
    settings: Terrasim = defaultterrasim
    params: Params = Params()
    infil: object = field()
    et: object = field()
    router: object = field()
    residual_warn_mm: float = 1e-6
    residual_raise_mm: float = 1e-3

    @infil.default
    def _default_infil(s):
        cls = infiltration.METHODS[s.settings.infiltration]
        if cls is infiltration.GreenAmpt:
            return cls(dry_reset_steps=max(1, int(round(s.settings.dry_reset_hours / s.settings.dt_hours))))
        return cls()

    @et.default
    def _default_et(s):
        return FAO56DualCrop(kc_max=s.params.veg.kc_max, p_depletion=s.params.veg.p_depletion)

    @router.default
    def _default_router(s):
        return _router_for(s.settings, s.static)

    # -- helpers -------------------------------------------------------------

    def _absorb_factory(s, state: State, dt: float):
        """Infiltration applied to a flat subset of pixels (used by the router)."""
        st = s.static
        L = st.nlayer
        theta_flat = state.theta.reshape(L, -1)
        theta_s = st.theta_s.reshape(L, -1)
        ksat = st.ksat.reshape(L, -1)
        psi_f = st.psi_f.ravel()
        F = state.F_cum.ravel()
        t_dry = state.t_dry.ravel()

        def absorb(idx, water):
            th = theta_flat[:, idx]
            Fi = F[idx].copy(); td = t_dry[idx].copy()
            got, _ = s.infil.infiltrate(water, th, st.dz_mm, theta_s[:, idx], ksat[:, idx], psi_f[idx], Fi, td, dt)
            theta_flat[:, idx] = th
            F[idx] = Fi; t_dry[idx] = td
            return got

        return absorb

    # -- the step --------------------------------------------------------------

    def step(s, state: State, f: StepForcing, uptake: np.ndarray | None = None) -> Fluxes:
        st, p = s.static, s.params
        dt = s.settings.dt_hours
        L, (ny, nx) = st.nlayer, st.shape
        n = ny * nx
        s0 = state.water_mm(st.dz_mm)

        # 1. interception
        throughfall, ei, et0_res = interception.intercept(f.rain, state.canopy, f.fc, f.et0, p.veg.canopy_store_mm_per_fc)

        # 2. ET demand (before the surface wets, as FAO-56 evaluates Ke on the previous day's state)
        e_pot, t_pot = s.et.demand(state.theta, st, f.kcb, f.fc, et0_res)

        # 3. infiltration of throughfall + pond
        supply = (throughfall + state.pond).ravel()
        state.pond[:] = 0.0
        theta_flat = state.theta.reshape(L, n)
        F = state.F_cum.ravel(); t_dry = state.t_dry.ravel()
        infil, excess = s.infil.infiltrate(supply, theta_flat, st.dz_mm, st.theta_s.reshape(L, n),
                                           st.ksat.reshape(L, n), st.psi_f.ravel(), F, t_dry, dt)
        pond = np.minimum(excess, p.soil.pond_max_mm)
        runoff_local = excess - pond

        # 4. routing with re-infiltration at receivers
        runon, passed, out, absorbed = s.router.route(runoff_local, s._absorb_factory(state, dt))
        infil_total = (infil + absorbed).reshape(ny, nx)
        state.pond[:] = pond.reshape(ny, nx)

        # 5. drainage
        deep, _ = drainage.drain(state.theta, st, dt, s.settings.drain_frac_max, s.settings.max_substeps)

        # 6. soil evaporation, 7. transpiration
        e = s.et.evaporate(state.theta, st, e_pot)
        t = s.et.transpire(state.theta, st, t_pot, uptake)

        # 8. mass balance
        s1 = state.water_mm(st.dz_mm)
        runon2 = runon.reshape(ny, nx); passed2 = passed.reshape(ny, nx)
        net = f.rain + runon2 - ei - e - t.sum(axis=0) - passed2 - deep
        residual = (s1 - s0) - net
        worst = float(np.nanmax(np.abs(residual[st.valid]))) if st.valid.any() else 0.0
        if worst > s.residual_raise_mm:
            raise RuntimeError(f'mass balance violated: max |residual| = {worst:.3e} mm')
        if worst > s.residual_warn_mm:
            warnings.warn(f'mass balance residual {worst:.3e} mm exceeds {s.residual_warn_mm}')

        return Fluxes(rain=f.rain.copy(), interception_evap=ei, throughfall=throughfall,
                      infiltration=infil_total, runon=runon2, runoff_pass=passed2,
                      runoff_out=out.reshape(ny, nx), deep_drain=deep, soil_evap=e,
                      transpiration=t, residual=residual)


# -- tests -----------------------------------------------------------------------

def _single_pixel(ksat=20.0, routing='none'):
    from terrasim.layers import LayerSpec
    st = StaticArrays.uniform(LayerSpec(), 1, 1, ksat=ksat)
    eng = Engine(st, Terrasim(routing=routing))
    return st, eng, State.initial(st, 'dul')


def test_steady_state_under_constant_rain():
    st, eng, state = _single_pixel()
    ones = np.ones((1, 1))
    forcing = StepForcing(rain=2.0 * ones, et0=0.1 * ones, kcb=0.5 * ones, fc=0.5 * ones)
    worst = 0.0
    daily = Fluxes.zeros(st.nlayer, 1, 1)
    for step in range(24 * 60):
        fx = eng.step(state, forcing)
        worst = max(worst, abs(float(fx.residual[0, 0])))
        if step >= 24 * 50:
            daily.add_(fx)
    rain = float(daily.rain.sum())
    losses = float(daily.deep_drain.sum() + daily.et_actual.sum() + daily.runoff_out.sum())
    return worst < 1e-9 and abs(losses - rain) / rain < 0.01


def test_dry_down_conserves_and_never_exceeds_bounds():
    st, eng, state = _single_pixel()
    ones = np.ones((1, 1))
    state.theta[:] = st.theta_s
    forcing = StepForcing(rain=0.0 * ones, et0=0.3 * ones, kcb=0.9 * ones, fc=0.8 * ones)
    s0 = state.water_mm(st.dz_mm).copy(); lost = 0.0
    for _ in range(24 * 30):
        fx = eng.step(state, forcing)
        lost += float((fx.deep_drain + fx.et_actual).sum())
        if state.theta.max() > st.theta_s.max() + 1e-12 or state.theta.min() < 0.5 * st.theta_l15.min() - 1e-12:
            return False
    return abs(float((state.water_mm(st.dz_mm) - s0).sum()) + lost) < 1e-8


def test_external_uptake_replaces_transpiration():
    st, eng, state = _single_pixel()
    ones = np.ones((1, 1))
    forcing = StepForcing(rain=0.0 * ones, et0=0.0 * ones, kcb=0.5 * ones, fc=0.5 * ones)
    uptake = np.zeros((st.nlayer, 1, 1)); uptake[2] = 0.7
    fx = eng.step(state, forcing, uptake=uptake)
    return abs(float(fx.transpiration[2].sum()) - 0.7) < 1e-12 and float(fx.transpiration.sum()) == 0.7


def test_routing_wets_downslope_pixels():
    from terrasim import terrain
    from terrasim.layers import LayerSpec
    ny, nx = 6, 3
    dem = np.arange(ny)[:, None] * 1.0 + np.zeros((1, nx))
    fdir = terrain.d8_flowdir_simple(dem, 10.0)
    ds, _ = terrain.downstream_index(fdir)
    order, ptr = terrain.topological_levels(ds)
    st = StaticArrays.uniform(LayerSpec(), ny, nx, ksat=5.0, ds_index=ds, level_order=order, level_ptr=ptr)
    eng = Engine(st, Terrasim(routing='d8'))
    state = State.initial(st, 'dul')
    state.theta[:, -2:, :] = st.theta_s[:, -2:, :]        # ridge rows (high rows) saturated -> they shed runoff
    forcing = StepForcing(rain=np.full((ny, nx), 20.0), et0=np.zeros((ny, nx)), kcb=np.zeros((ny, nx)), fc=np.zeros((ny, nx)))
    s_before = State.initial(st, 'dul'); s_before.theta[:, -2:, :] = st.theta_s[:, -2:, :]
    w0 = s_before.water_mm(st.dz_mm).sum()
    fx = eng.step(state, forcing)
    w1 = state.water_mm(st.dz_mm).sum()
    domain_closes = abs((w1 - w0) - (fx.rain.sum() - fx.runoff_out.sum() - fx.deep_drain.sum() - fx.et_actual.sum())) < 1e-9
    # run-on reached the lower rows, every pixel balances, and the domain balances
    return fx.runon[:-2].sum() > 0 and np.abs(fx.residual).max() < 1e-9 and domain_closes

def test():
    return all([
        test_steady_state_under_constant_rain(),
        test_dry_down_conserves_and_never_exceeds_bounds(),
        test_external_uptake_replaces_transpiration(),
        test_routing_wets_downslope_pixels(),
    ])


if __name__ == '__main__':
    print(test())
