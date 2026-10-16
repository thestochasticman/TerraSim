"""Phase 2: BARRA store ledger, disaggregation, vegetation, forcing assembly — offline."""
from terrasim import barra, disaggregate, vegetation, forcing


def test_barra_store_offline():
    assert barra.test()


def test_disaggregate():
    assert disaggregate.test()


def test_vegetation():
    assert vegetation.test()


def test_forcing_rain_conservation():
    assert forcing.test_rain_conserves_silo_daily_totals()


def test_forcing_et0_weights():
    assert forcing.test_et0_weights_sum_to_one_and_zero_at_night()


def test_forcing_step_broadcast():
    assert forcing.test_forcing_step_broadcasts_to_grid()


def test_forcing_substeps_conserve():
    assert forcing.test_twenty_minute_step_conserves()


def test_forcing_blend_weights():
    assert forcing.test_blend_weights_sum_to_one_and_are_smooth()
