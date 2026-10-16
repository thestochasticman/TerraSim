"""Run-time settings for a terrasim simulation.

Everything a user might tune about *how* the model runs (time step,
which process formulations, routing, output cadence) lives on one
frozen attrs class so a run is fully described by
``(Troi, Terrasim, Params, LayerSpec)``. Physical parameters live in
:mod:`terrasim.params`; layer geometry in :mod:`terrasim.layers`.
"""
from attrs import frozen

_STEP_HOURS = {'20min': 1 / 3, '1h': 1.0, '3h': 3.0}


@frozen
class Terrasim:
    """How a simulation is run.

    Attributes:
        step: Sub-daily model step: ``'20min'``, ``'1h'`` (default) or ``'3h'``.
        routing: Downslope redistribution of infiltration-excess runoff:
            ``'d8'`` (default, level-ordered D8 run-on), ``'twi'`` or ``'none'``.
        infiltration: ``'green_ampt'`` (default) or ``'capacity'``.
        et_method: ``'fao56_dual'`` (default) or ``'priestley_taylor'``.
        disaggregation: Sub-daily rain weights: ``'barra'`` (default; BARRA-C2
            hourly fractions with climatology/uniform fallback), ``'climatology'``
            or ``'uniform'``.
        output_cadence: ``'daily'`` (default) or ``'substep'`` (large!).
        spinup_days: Days of forcing repeated before ``start`` to settle the state.
        init_theta: Initial moisture: ``'dul'`` (field capacity), ``'midpoint'``
            (between DUL and L15) or ``'l15'``.
        buffer_m: Extra DEM margin used when computing flow accumulation.
        tz_offset_hours: Local standard time offset; ``None`` derives
            ``round(centre_lon / 15)`` (no daylight saving, matching SILO/BoM).
        drain_frac_max: Max fraction of a layer's water-equivalent thickness
            that may drain in one sub-step (Courant-like limiter).
        max_substeps: Cap on drainage sub-steps per model step.
        dry_reset_hours: Hours without infiltration after which the Green-Ampt
            cumulative infiltration resets (new storm event).
    """

    step: str = '1h'
    routing: str = 'd8'
    infiltration: str = 'green_ampt'
    et_method: str = 'fao56_dual'
    disaggregation: str = 'barra'
    output_cadence: str = 'daily'
    spinup_days: int = 90
    init_theta: str = 'dul'
    buffer_m: float = 500.0
    tz_offset_hours: float | None = None
    drain_frac_max: float = 0.1
    max_substeps: int = 24
    dry_reset_hours: float = 6.0

    def __attrs_post_init__(s):
        if s.step not in _STEP_HOURS:
            raise ValueError(f"step must be one of {sorted(_STEP_HOURS)}, got {s.step!r}")
        for name, allowed in (('routing', ('d8', 'twi', 'none')),
                              ('infiltration', ('green_ampt', 'capacity')),
                              ('et_method', ('fao56_dual', 'priestley_taylor')),
                              ('disaggregation', ('barra', 'climatology', 'uniform')),
                              ('output_cadence', ('daily', 'substep')),
                              ('init_theta', ('dul', 'midpoint', 'l15'))):
            if getattr(s, name) not in allowed:
                raise ValueError(f'{name} must be one of {allowed}, got {getattr(s, name)!r}')

    @property
    def dt_hours(s) -> float:
        """Length of one model step in hours."""
        return _STEP_HOURS[s.step]

    @property
    def steps_per_day(s) -> int:
        return int(round(24 / s.dt_hours))

    @property
    def steps_per_hour(s) -> int:
        """Model steps per hour (1 for '1h', 3 for '20min'); 0 for steps longer than an hour."""
        return int(round(1 / s.dt_hours)) if s.dt_hours <= 1 else 0

    def tz(s, centre_lon: float) -> float:
        return s.tz_offset_hours if s.tz_offset_hours is not None else float(round(centre_lon / 15))


defaultterrasim = Terrasim()


def test_defaults_valid():
    t = Terrasim()
    return t.dt_hours == 1.0 and t.steps_per_day == 24 and t.tz(148.37) == 10.0


def test_20min_step():
    t = Terrasim(step='20min')
    return t.steps_per_day == 72 and t.steps_per_hour == 3


def test_bad_option_raises():
    try:
        Terrasim(routing='magic')
    except ValueError:
        return True
    return False


def test():
    return all([test_defaults_valid(), test_20min_step(), test_bad_option_raises()])


if __name__ == '__main__':
    print(test())
