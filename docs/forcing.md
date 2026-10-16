# Forcing

## Time
Local standard time, `round(centre_lon / 15)` hours (+10 for NSW), no daylight
saving — the convention SILO and the Bureau use. Steps run from `start` 00:00
to `end` 23:xx (20 min, 1 h or 3 h).

## Rain: SILO totals, BARRA-C2 timing
- SILO `daily_rain` on date D is the gauge-day total **09:00 (D−1) → 09:00 (D)**.
  It is the conserved quantity: the sum of model-step rain over every complete
  SILO day equals the SILO value exactly (asserted).
- BARRA-C2 (BoM reanalysis, 0.04° ≈ 4.4 km, hourly, UTC, 1979 → ~4 months
  ago) gives the fractions within the day. For each SILO day and BARRA cell the
  24 hourly values in the window are normalised to weights.
- If BARRA is dry (< 0.2 mm) or missing on a day SILO is wet, the weights come
  from the cell's climatological diurnal profile over rainy days (needs ≥ 30
  such days in the fetched record), else uniform. The run reports what share
  of days used BARRA timing (`barra_timing_fraction`).
- 20-minute steps split each hour evenly; 3-hour steps sum three hours.
- Each pixel blends the SILO points (0.05° grid) and BARRA cells under the
  window with inverse-distance weights, so coarse-cell boundaries do not print
  through as seams in runoff; a 2 km paddock usually has one of each.

## Reference ET
SILO `et_short_crop` (FAO-56 Penman-Monteith reference) by default, or
Priestley-Taylor from SILO radiation/temperature/vapour pressure
(`ForcingBuilder(et0_source='priestley_taylor')`). Daily ET0 is spread over
daylight as a half-sine (sunrise/sunset from latitude and day of year).

## Vegetation
Sentinel-2 NDVI (pysentinel2, cleaned, 10 m) → despiked (rolling median),
linearly interpolated to daily, Savitzky-Golay smoothed (31 d) → `Kcb` and
cover fraction `fc` (see physics.md §2). The NDVI window is fetched 60 days
before `start` so the first days are interpolated, not extrapolated.

## Stores
Every input goes through a machine-wide cache keyed by `troi.Config.tmp_dir`:
`pysilo` (SQLite), `pyslga`, `pycopdem`, `pysentinel2` (zarr + SQLite), and
terrasim's own `terrasim_store/barra.db` for BARRA-C2 (one row per cell per
UTC hour; months fetched once). Static terrain+soil datasets are cached as
zarr under `terrasim_store/static/` keyed by region, layer geometry and
pedotransfer version.
