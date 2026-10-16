"""One machine-wide BARRA-C2 hourly-rain store that fills itself on demand.

BARRA-C2 is the Bureau of Meteorology's 4.4 km (0.04 deg) convection-
permitting reanalysis, 1979 to a few months ago, served from NCI THREDDS
as one national NetCDF per variable per month (300-600 MB each) with
hourly precipitation flux ``pr`` (kg m-2 s-1). terrasim only needs the
within-day *timing* of rain for the handful of cells under a paddock, so
the unit of fetching — and of the ledger — is the ``(cell, month)`` pair:

    {config.tmp_dir}/terrasim_store/
    └── barra.db
        ├── axes(axis, idx, value)              # the file's lat/lon vectors, read once
        ├── hourly(iy, ix, time_utc, pr_mm)     # mm per hour, hour-start UTC
        └── coverage(iy, ix, ym)                # months fetched per cell

``Store.get_ds(bbox, start, end)`` finds the cells whose centres are
nearest to the pixels of ``bbox``, fetches the missing months with one
OPeNDAP box subset per month, and returns the hourly series per cell.
Months beyond the archive (or unreachable) are reported, not raised, so
runs can fall back to climatological timing.
"""
import sqlite3
import warnings
from datetime import date, datetime, timedelta
from os import makedirs

import numpy as np
import pandas as pd
import xarray as xr
from attrs import frozen, field

from troi import Config, config as default_config
from terrasim.paths import Paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS axes (
    axis  TEXT NOT NULL,
    idx   INTEGER NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (axis, idx)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS hourly (
    iy       INTEGER NOT NULL,
    ix       INTEGER NOT NULL,
    time_utc TEXT NOT NULL,
    pr_mm    REAL,
    PRIMARY KEY (iy, ix, time_utc)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS coverage (
    iy INTEGER NOT NULL,
    ix INTEGER NOT NULL,
    ym TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (iy, ix, ym)
) WITHOUT ROWID;
"""


@frozen
class BARRA:
    """Endpoint configuration for BARRA-C2 hourly precipitation on NCI THREDDS."""

    base_url: str = 'https://thredds.nci.org.au/thredds/dodsC'
    path: str = 'ob53/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/pr/latest'
    filename: str = 'pr_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{ym}-{ym}.nc'
    variable: str = 'pr'
    first_month: str = '197901'
    lat_range: tuple[float, float] = (-45.69, -5.01)
    lon_range: tuple[float, float] = (108.02, 159.9)
    flux_to_mm_per_hour: float = 3600.0

    def url(s, year: int, month: int) -> str:
        ym = f'{year:04d}{month:02d}'
        return f'{s.base_url}/{s.path}/{s.filename.format(ym=ym)}'

    def in_bounds(s, lat: float, lon: float) -> bool:
        return s.lat_range[0] <= lat <= s.lat_range[1] and s.lon_range[0] <= lon <= s.lon_range[1]


defaultbarra = BARRA()


def months_between(start: date, end: date) -> list[tuple[int, int]]:
    """Inclusive list of ``(year, month)`` touching ``[start, end]``."""
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


@frozen
class Store:
    """The machine-wide BARRA-C2 store: one ledger, zero re-fetches.

    Composed from :class:`troi.Config` (where the store lives) and
    :class:`BARRA` (endpoint). No inheritance.

    Example:
        ```python
        from datetime import date
        from terrasim.barra import Store

        store = Store()
        ds = store.get_ds([148.36, -33.53, 148.38, -33.51], date(2022, 1, 1), date(2022, 3, 31))
        ds.pr_mm            # (time, cell) hourly mm, UTC hour-start
        ```
    """

    config: Config = default_config
    barra: BARRA = defaultbarra
    paths: Paths = field(init=False)

    paths.default(lambda s: Paths(s.config))

    def __attrs_post_init__(s):
        makedirs(s.paths.root, exist_ok=True)

    def _db(s) -> sqlite3.Connection:
        db = sqlite3.connect(s.paths.barra_db)
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript(_SCHEMA)
        return db

    # -- axes -----------------------------------------------------------------

    def _axes(s, db) -> tuple[np.ndarray, np.ndarray]:
        """The file's lat/lon vectors, read from the archive at first contact."""
        lat = np.array([r[0] for r in db.execute("SELECT value FROM axes WHERE axis='lat' ORDER BY idx")])
        lon = np.array([r[0] for r in db.execute("SELECT value FROM axes WHERE axis='lon' ORDER BY idx")])
        if lat.size and lon.size:
            return lat, lon
        lat, lon = s._fetch_axes()
        with db:
            db.executemany('INSERT OR REPLACE INTO axes VALUES (?, ?, ?)',
                           [('lat', i, float(v)) for i, v in enumerate(lat)] +
                           [('lon', i, float(v)) for i, v in enumerate(lon)])
        return lat, lon

    def _fetch_axes(s):
        y, m = int(s.barra.first_month[:4]), int(s.barra.first_month[4:])
        ds = xr.open_dataset(s.barra.url(y, m), decode_times=False)
        try:
            return ds['lat'].values.astype(float), ds['lon'].values.astype(float)
        finally:
            ds.close()

    def cells_for_points(s, lats, lons, db=None) -> np.ndarray:
        """Nearest BARRA ``(iy, ix)`` for each (lat, lon); shape ``(n, 2)``."""
        own = db is None
        db = db or s._db()
        try:
            lat, lon = s._axes(db)
        finally:
            if own:
                db.close()
        iy = np.abs(np.asarray(lats, dtype=float)[:, None] - lat[None, :]).argmin(axis=1)
        ix = np.abs(np.asarray(lons, dtype=float)[:, None] - lon[None, :]).argmin(axis=1)
        return np.column_stack([iy, ix])

    def cells_for_bbox(s, bbox: list[float], db=None) -> np.ndarray:
        """All cells whose centres fall within ``bbox`` plus the nearest cells to
        its corners (so a sub-cell paddock still gets one cell); ``(n, 2)``."""
        west, south, east, north = bbox
        own = db is None
        db = db or s._db()
        try:
            lat, lon = s._axes(db)
        finally:
            if own:
                db.close()
        iys = np.flatnonzero((lat >= south) & (lat <= north))
        ixs = np.flatnonzero((lon >= west) & (lon <= east))
        corners = s.cells_for_points([south, south, north, north], [west, east, west, east])
        iys = np.union1d(iys, corners[:, 0]); ixs = np.union1d(ixs, corners[:, 1])
        return np.array([(int(a), int(b)) for a in iys for b in ixs], dtype=np.int64).reshape(-1, 2)

    # -- fill -----------------------------------------------------------------

    def fill(s, bbox: list[float], start: date, end: date) -> int:
        """Ensure every ``(cell, month)`` covering the request is populated.
        Returns the number of months actually fetched (0 = already local).
        Months the archive does not have yet are skipped with a warning."""
        db = s._db()
        try:
            cells = s.cells_for_bbox(bbox, db)
            fetched = 0
            for (y, m) in months_between(start, end):
                ym = f'{y:04d}{m:02d}'
                have = {(r[0], r[1]) for r in db.execute('SELECT iy, ix FROM coverage WHERE ym=?', (ym,))}
                missing = [(int(a), int(b)) for a, b in cells if (int(a), int(b)) not in have]
                if not missing:
                    continue
                if ym < s.barra.first_month:
                    warnings.warn(f'BARRA-C2 has no data before {s.barra.first_month}; skipping {ym}')
                    continue
                ok = s._fetch_month(db, np.array(missing), y, m)
                fetched += int(ok)
            return fetched
        finally:
            db.close()

    def _fetch_month(s, db, cells: np.ndarray, year: int, month: int) -> bool:
        """One OPeNDAP box subset covering ``cells`` for a month; store hourly mm."""
        iy0, iy1 = int(cells[:, 0].min()), int(cells[:, 0].max())
        ix0, ix1 = int(cells[:, 1].min()), int(cells[:, 1].max())
        url = s.barra.url(year, month)
        try:
            ds = xr.open_dataset(url)
        except Exception as ex:   # noqa: BLE001 — archive lag or THREDDS outage: report, don't fail the run
            warnings.warn(f'BARRA-C2 {year}-{month:02d} unavailable ({type(ex).__name__}: {ex}); '
                          'rain timing will use fallback weights for this month')
            return False
        try:
            da = ds[s.barra.variable].isel(lat=slice(iy0, iy1 + 1), lon=slice(ix0, ix1 + 1)).load()
        finally:
            ds.close()
        times = pd.to_datetime(da['time'].values).floor('h')
        vals = np.asarray(da.values, dtype=float) * s.barra.flux_to_mm_per_hour
        rows = []
        for (iy, ix) in map(tuple, cells):
            series = vals[:, iy - iy0, ix - ix0]
            rows += [(int(iy), int(ix), t.strftime('%Y-%m-%dT%H:00'), None if np.isnan(v) else float(v))
                     for t, v in zip(times, series)]
        now = datetime.utcnow().isoformat() + 'Z'
        with db:
            db.executemany('INSERT OR REPLACE INTO hourly VALUES (?, ?, ?, ?)', rows)
            db.executemany('INSERT OR REPLACE INTO coverage VALUES (?, ?, ?, ?)',
                           [(int(iy), int(ix), f'{year:04d}{month:02d}', now) for iy, ix in map(tuple, cells)])
        return True

    # -- read -----------------------------------------------------------------

    def get_ds(s, bbox: list[float], start: date, end: date) -> xr.Dataset:
        """Hourly rain (mm) per cell: ``pr_mm (time, cell)`` on a complete hourly
        UTC axis from ``start`` 00:00 to ``end`` 23:00 (NaN where not covered),
        with coords ``cell_iy, cell_ix, cell_lat, cell_lon``. Fills first."""
        s.fill(bbox, start, end)
        db = s._db()
        try:
            cells = s.cells_for_bbox(bbox, db)
            lat, lon = s._axes(db)
            t0, t1 = f'{start:%Y-%m-%d}T00:00', f'{end:%Y-%m-%d}T23:00'
            frames = []
            for (iy, ix) in map(tuple, cells):
                rows = db.execute('SELECT time_utc, pr_mm FROM hourly WHERE iy=? AND ix=? AND time_utc BETWEEN ? AND ? '
                                  'ORDER BY time_utc', (int(iy), int(ix), t0, t1)).fetchall()
                ser = pd.Series({pd.Timestamp(t): v for t, v in rows}, dtype=float, name=(iy, ix))
                frames.append(ser)
        finally:
            db.close()
        times = pd.date_range(f'{start:%Y-%m-%d} 00:00', f'{end:%Y-%m-%d} 23:00', freq='h')
        data = np.full((len(times), len(cells)), np.nan)
        for j, ser in enumerate(frames):
            if len(ser):
                data[:, j] = ser.reindex(times).values
        return xr.Dataset(
            {'pr_mm': (('time', 'cell'), data)},
            coords={'time': times, 'cell': np.arange(len(cells)),
                    'cell_iy': ('cell', cells[:, 0]), 'cell_ix': ('cell', cells[:, 1]),
                    'cell_lat': ('cell', lat[cells[:, 0]]), 'cell_lon': ('cell', lon[cells[:, 1]])},
            attrs={'source': 'BARRA-C2 v1 1hr pr (BoM via NCI THREDDS)', 'units': 'mm h-1', 'time': 'UTC hour start'},
        )

    def fill_troi(s, troi) -> int:
        return s.fill(troi.bbox, troi.start - timedelta(days=1), troi.end + timedelta(days=1))

    def get_ds_troi(s, troi) -> xr.Dataset:
        return s.get_ds(troi.bbox, troi.start - timedelta(days=1), troi.end + timedelta(days=1))


# -- tests -----------------------------------------------------------------------

def _tmp_store() -> Store:
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix='terrasim_barra_test_')
    return Store(config=Config(out_dir=tmpdir, tmp_dir=tmpdir))


def _prime_axes(store: Store):
    lat = np.round(np.arange(-45.69, -5.0, 0.04), 3)
    lon = np.round(np.arange(108.02, 159.9, 0.04), 3)
    db = store._db()
    with db:
        db.executemany('INSERT OR REPLACE INTO axes VALUES (?, ?, ?)',
                       [('lat', i, float(v)) for i, v in enumerate(lat)] + [('lon', i, float(v)) for i, v in enumerate(lon)])
    db.close()
    return lat, lon


def _prime_month(store: Store, cell, year, month, value=0.5):
    times = pd.date_range(f'{year}-{month:02d}-01', periods=24 * 28, freq='h')
    db = store._db()
    with db:
        db.executemany('INSERT OR REPLACE INTO hourly VALUES (?, ?, ?, ?)',
                       [(int(cell[0]), int(cell[1]), t.strftime('%Y-%m-%dT%H:00'), value) for t in times])
        db.execute('INSERT OR REPLACE INTO coverage VALUES (?, ?, ?, ?)', (int(cell[0]), int(cell[1]), f'{year}{month:02d}', 'test'))
    db.close()


def test_months_between():
    return months_between(date(2021, 11, 15), date(2022, 2, 1)) == [(2021, 11), (2021, 12), (2022, 1), (2022, 2)]


def test_cells_for_small_bbox_is_one_or_few_cells():
    store = _tmp_store(); _prime_axes(store)
    cells = store.cells_for_bbox([148.36265, -33.52606, 148.38265, -33.50606])
    return 1 <= len(cells) <= 4


def test_fill_skips_covered_months():
    store = _tmp_store(); _prime_axes(store)
    bbox = [148.36265, -33.52606, 148.38265, -33.50606]
    for c in map(tuple, store.cells_for_bbox(bbox)):
        _prime_month(store, c, 2022, 2)
    return store.fill(bbox, date(2022, 2, 1), date(2022, 2, 20)) == 0


def test_read_returns_complete_hourly_axis():
    store = _tmp_store(); _prime_axes(store)
    bbox = [148.36265, -33.52606, 148.38265, -33.50606]
    for c in map(tuple, store.cells_for_bbox(bbox)):
        _prime_month(store, c, 2022, 2)
    ds = store.get_ds(bbox, date(2022, 2, 1), date(2022, 2, 3))
    return ds.pr_mm.shape[0] == 72 and bool(np.allclose(ds.pr_mm.values, 0.5))


def test():
    return all([test_months_between(), test_cells_for_small_bbox_is_one_or_few_cells(),
                test_fill_skips_covered_months(), test_read_returns_complete_hourly_axis()])


if __name__ == '__main__':
    print(test())
