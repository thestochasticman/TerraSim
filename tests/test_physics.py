"""Phase 1: column physics, offline."""
import numpy as np

from terrasim.physics import campbell, infiltration, drainage, evapotranspiration, interception, routing, engine, surface


def test_campbell():
    assert campbell.test()


def test_infiltration():
    assert infiltration.test()


def test_drainage():
    assert drainage.test()


def test_evapotranspiration():
    assert evapotranspiration.test()


def test_interception():
    assert interception.test()


def test_routing():
    assert routing.test()


def test_surface():
    assert surface.test()


def test_engine_sealed_surface():
    assert engine.test_sealed_surface_sheds_rain_the_soil_would_absorb()


def test_engine_capacity_spread():
    assert engine.test_capacity_spread_smooths_runoff_onset()


def test_engine_contributing_fraction():
    assert engine.test_contributing_fraction_sheds_some_of_every_rain()


def test_engine_contributing_fraction_with_seal():
    assert engine.test_contributing_fraction_scales_a_binding_seal()


def test_engine_steady_state():
    assert engine.test_steady_state_under_constant_rain()


def test_engine_dry_down():
    assert engine.test_dry_down_conserves_and_never_exceeds_bounds()


def test_engine_external_uptake():
    assert engine.test_external_uptake_replaces_transpiration()


def test_engine_routing():
    assert engine.test_routing_wets_downslope_pixels()


def test_rain_below_ksat_never_runs_off():
    from terrasim.layers import LayerSpec
    from terrasim.physics.state import StaticArrays, State
    from terrasim.physics.engine import Engine, StepForcing
    from terrasim.terrasim import Terrasim
    st = StaticArrays.uniform(LayerSpec(), 2, 2, ksat=10.0)
    eng = Engine(st, Terrasim(routing='none'))
    state = State.initial(st, 'dul')
    f = StepForcing(rain=np.full((2, 2), 3.0), et0=np.zeros((2, 2)), kcb=np.zeros((2, 2)), fc=np.zeros((2, 2)))
    for _ in range(48):
        fx = eng.step(state, f)
        assert fx.runoff_out.max() == 0.0
        assert fx.residual.max() < 1e-9
