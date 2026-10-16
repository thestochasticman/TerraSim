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


def hourly_to_steps(hourly: np.ndarray, steps_per_hour: int, hours_per_step: int) -> np.ndarray:
    """Resample a per-hour series (time first axis) to the model step: split each
    hour into ``steps_per_hour`` equal parts, or sum ``hours_per_step`` hours."""
    if steps_per_hour > 1:
        return np.repeat(hourly / steps_per_hour, steps_per_hour, axis=0)
    if hours_per_step > 1:
        n = hourly.shape[0] // hours_per_step * hours_per_step
        return hourly[:n].reshape(-1, hours_per_step, *hourly.shape[1:]).sum(axis=1)
    return hourly


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


def test():
    return all([test_silo_day_assignment(), test_weights_sum_to_one_per_silo_day_and_fallback_on_dry_days(),
                test_climatology_profile_when_enough_rainy_days(), test_resampling_conserves()])


if __name__ == '__main__':
    print(test())
