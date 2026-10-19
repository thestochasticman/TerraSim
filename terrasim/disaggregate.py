"""Split SILO daily rain into sub-daily steps using BARRA-C2 timing.

SILO's ``daily_rain`` for date D is the gauge-style total from 09:00
local standard time on D-1 to 09:00 on D. BARRA-C2 gives hourly rain in
UTC. For every SILO day and BARRA cell we take the 24 hourly values in the
SILO window, normalise them to weights and multiply by the SILO total, so
SILO's daily mass is preserved exactly while BARRA sets when within the
day it fell. When BARRA is dry (or missing) on a day SILO was wet, the
weights come from a fallback: the cell's climatological diurnal profile
over rainy days, or uniform if too few rainy days are known.

Weights are built at hourly resolution and then resampled to the model
step (20 min splits each hour evenly; 3 h sums three hours).
"""
import numpy as np
import pandas as pd
from attrs import frozen

SILO_DAY_START_HOUR = 9     # local standard time


def silo_day_of(local_times: pd.DatetimeIndex) -> np.ndarray:
    """SILO date each local hour belongs to (hours >= 09:00 belong to the next date)."""
    shifted = local_times + pd.Timedelta(hours=24 - SILO_DAY_START_HOUR)
    return shifted.normalize().values.astype('datetime64[D]')


def hour_in_silo_day(local_times: pd.DatetimeIndex) -> np.ndarray:
    """0 for the 09:00 hour, ..., 23 for the 08:00 hour."""
    return ((local_times.hour - SILO_DAY_START_HOUR) % 24).values


@frozen
class UniformWeights:
    def profile(s, cell_hourly: np.ndarray, hour_idx: np.ndarray, silo_day: np.ndarray) -> np.ndarray:
        return np.full(24, 1.0 / 24)


@frozen
class ClimatologyWeights:
    """Mean normalised hourly profile over rainy SILO days at this cell."""

    min_rainy_days: int = 30
    dry_threshold_mm: float = 0.2
    fallback: object = UniformWeights()

    def profile(s, cell_hourly: np.ndarray, hour_idx: np.ndarray, silo_day: np.ndarray) -> np.ndarray:
        df = pd.DataFrame({'v': cell_hourly, 'h': hour_idx, 'd': silo_day}).dropna()
        tot = df.groupby('d')['v'].transform('sum')
        wet = df[tot >= s.dry_threshold_mm].copy()
        n_days = wet['d'].nunique()
        if n_days < s.min_rainy_days:
            return s.fallback.profile(cell_hourly, hour_idx, silo_day)
        wet['w'] = wet['v'] / tot[tot >= s.dry_threshold_mm]
        prof = wet.groupby('h')['w'].mean().reindex(range(24)).fillna(0).values
        return prof / prof.sum() if prof.sum() > 0 else np.full(24, 1.0 / 24)


@frozen
class BarraWeights:
    """Per-day BARRA fractions with a fallback profile on dry/missing days."""

    dry_threshold_mm: float = 0.2
    fallback: object = ClimatologyWeights()

    def weights(s, cell_hourly: np.ndarray, hour_idx: np.ndarray, silo_day: np.ndarray):
        """Hourly weights (same length as ``cell_hourly``) summing to 1 within each
        complete SILO day, plus a per-hour source flag (``True`` where BARRA was used)."""
        v = np.where(np.isnan(cell_hourly), 0.0, cell_hourly)
        df = pd.DataFrame({'v': v, 'd': silo_day})
        tot = df.groupby('d')['v'].transform('sum').values
        use_barra = tot >= s.dry_threshold_mm
        prof = s.fallback.profile(cell_hourly, hour_idx, silo_day)
        w = np.where(use_barra, v / np.where(use_barra, tot, 1.0), prof[hour_idx])
        return w, use_barra


METHODS = {'barra': BarraWeights, 'climatology': lambda **kw: _ProfileOnly(ClimatologyWeights()),
           'uniform': lambda **kw: _ProfileOnly(UniformWeights())}


@frozen
class _ProfileOnly:
    inner: object

    def weights(s, cell_hourly, hour_idx, silo_day):
        prof = s.inner.profile(cell_hourly, hour_idx, silo_day)
        return prof[hour_idx], np.zeros(len(hour_idx), dtype=bool)


def hourly_to_steps(hourly: np.ndarray, steps_per_hour: int, hours_per_step: int,
                    within_hour: str = 'uniform', peak_fraction: float = 0.6) -> np.ndarray:
    """Resample a per-hour series (time first axis) to the model step: split each
    hour into ``steps_per_hour`` parts (evenly, or peaked: see
    :func:`peaked_within_hour`), or sum ``hours_per_step`` hours."""
    if steps_per_hour > 1:
        if within_hour == 'peaked':
            return peaked_within_hour(hourly, steps_per_hour, peak_fraction)
        return np.repeat(hourly / steps_per_hour, steps_per_hour, axis=0)
    if hours_per_step > 1:
        n = hourly.shape[0] // hours_per_step * hours_per_step
        return hourly[:n].reshape(-1, hours_per_step, *hourly.shape[1:]).sum(axis=1)
    return hourly


def peaked_within_hour(hourly: np.ndarray, steps_per_hour: int, peak_fraction: float = 0.6) -> np.ndarray:
    """Concentrate each hour's rain into one sub-step so sub-hourly intensities
    exceed the hourly mean, as they do in gauge records.

    BARRA-C2 is hourly, and an hour's rain rarely falls at a uniform rate: the
    heaviest 20 minutes of a wet hour typically carry ~0.55-0.65 of the hourly
    depth (the ratio of 20-min to 60-min depths at equal exceedance in the
    Bureau's IFD tables for temperate NSW). Splitting the hour evenly hides
    that peak from an infiltration-excess scheme and under-predicts runoff.

    Each hour's total is split so one sub-step receives ``peak_fraction`` and the
    other ``steps_per_hour - 1`` sub-steps share the remainder equally. The peak
    sits at the end of the hour when the next hour is wetter than the previous
    one, at the start when the previous hour is wetter, and in the middle
    otherwise, so storm cores stay contiguous across hour boundaries.
    Hourly totals are conserved exactly. Works per cell (``hourly`` is
    ``(nhour, ...)``).

    Args:
        hourly: Rain per hour, time first axis.
        steps_per_hour: Sub-steps per hour (3 for a 20-minute step).
        peak_fraction: Share of the hour's rain in the peak sub-step, in
            ``[1/steps_per_hour, 1]``; ``1/steps_per_hour`` reproduces uniform.
    """
    n = int(steps_per_hour)
    if n < 2:
        return hourly
    if not (1.0 / n - 1e-9 <= peak_fraction <= 1.0 + 1e-9):
        raise ValueError(f'peak_fraction must be in [1/{n}, 1], got {peak_fraction}')
    hourly = np.asarray(hourly, dtype=float)
    prev = np.concatenate([np.zeros_like(hourly[:1]), hourly[:-1]], axis=0)
    nxt = np.concatenate([hourly[1:], np.zeros_like(hourly[:1])], axis=0)
    pos = np.where(nxt > prev, n - 1, np.where(prev > nxt, 0, n // 2))          # (nhour, ...)
    rest = (1.0 - peak_fraction) / (n - 1)
    w = np.full((n,) + hourly.shape, rest)                                       # (n, nhour, ...)
    np.put_along_axis(w, pos[None], peak_fraction, axis=0)
    out = w * hourly[None]                                                       # (n, nhour, ...)
    return np.moveaxis(out, 0, 1).reshape((hourly.shape[0] * n,) + hourly.shape[1:])


# -- tests -----------------------------------------------------------------------

def _synthetic(days=40, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range('2022-01-01 00:00', periods=24 * days, freq='h')
    hourly = np.zeros(len(times))
    # storms in the afternoon (14:00-17:00) on even days
    for d in range(days):
        if d % 2 == 0:
            hourly[d * 24 + 14:d * 24 + 18] = rng.uniform(1, 5, 4)
    return times, hourly


def test_silo_day_assignment():
    t = pd.DatetimeIndex(['2022-01-05 08:00', '2022-01-05 09:00', '2022-01-05 23:00'])
    d = silo_day_of(t).astype(str).tolist()
    return d == ['2022-01-05', '2022-01-06', '2022-01-06'] and hour_in_silo_day(t).tolist() == [23, 0, 14]


def test_weights_sum_to_one_per_silo_day_and_fallback_on_dry_days():
    times, hourly = _synthetic()
    sd, hi = silo_day_of(times), hour_in_silo_day(times)
    w, used = BarraWeights().weights(hourly, hi, sd)
    sums = pd.Series(w).groupby(sd).sum()
    complete = sums.iloc[1:-1]          # first/last SILO days are partial in this window
    # even-day storms -> BARRA used on those days; odd days fall back to climatology (>= 30 rainy days? no: 20) -> uniform
    return np.allclose(complete.values, 1.0) and used.any() and (~used).any()


def test_climatology_profile_when_enough_rainy_days():
    times, hourly = _synthetic(days=80)
    sd, hi = silo_day_of(times), hour_in_silo_day(times)
    prof = ClimatologyWeights(min_rainy_days=30).profile(hourly, hi, sd)
    # storms at local 14:00-17:00 = SILO-day hours 5..8
    return abs(prof.sum() - 1) < 1e-12 and prof[5:9].sum() > 0.99


def test_resampling_conserves():
    hourly = np.arange(48.0)[:, None]
    a = hourly_to_steps(hourly, 3, 1); b = hourly_to_steps(hourly, 1, 3)
    return a.shape[0] == 144 and abs(a.sum() - hourly.sum()) < 1e-9 and b.shape[0] == 16 and abs(b.sum() - hourly.sum()) < 1e-9


def test_peaked_conserves_and_places_peak_toward_wetter_neighbour():
    hourly = np.array([[0.0], [3.0], [9.0], [3.0], [0.0], [6.0], [0.0]])           # rising, peak, falling, isolated
    out = peaked_within_hour(hourly, 3, 0.6).reshape(7, 3)
    conserved = np.allclose(out.sum(axis=1), hourly[:, 0])
    rising_peak_last = out[1].argmax() == 2            # hour 1: next (9) > prev (0)
    falling_peak_first = out[3].argmax() == 0          # hour 3: prev (9) > next (0)
    isolated_peak_middle = out[5].argmax() == 1        # hour 5: prev == next == 0
    amplified = abs(out[2].max() - 0.6 * 9.0) < 1e-12 and abs(out[2].min() - 0.2 * 9.0) < 1e-12
    uniform_limit = np.allclose(peaked_within_hour(hourly, 3, 1 / 3), np.repeat(hourly / 3, 3, axis=0))
    return conserved and rising_peak_last and falling_peak_first and isolated_peak_middle and amplified and uniform_limit


def test_peaked_rejects_bad_fraction():
    try:
        peaked_within_hour(np.ones((2, 1)), 3, 0.2)
    except ValueError:
        return True
    return False


def test():
    return all([test_silo_day_assignment(), test_weights_sum_to_one_per_silo_day_and_fallback_on_dry_days(),
                test_climatology_profile_when_enough_rainy_days(), test_resampling_conserves(),
                test_peaked_conserves_and_places_peak_toward_wetter_neighbour(), test_peaked_rejects_bad_fraction()])


if __name__ == '__main__':
    print(test())
