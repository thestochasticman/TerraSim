"""Flow routing geometry on the 10 m model grid.

pycopdem exposes slope/aspect/accumulation/TWI on its 30 m grid but not
flow direction, and routing must live on the model grid, so terrasim
derives D8 flow directions itself on the 10 m (metric, EPSG:6933) DEM via
pysheds' conditioning chain, then turns them into a downstream-index
array and a topological level ordering with plain numpy. The level
ordering is what makes run-on routing a handful of vectorised scatter-adds
per step instead of a per-pixel loop.

Conventions: pysheds' default dirmap (64 N, 128 NE, 1 E, 2 SE, 4 S, 8 SW,
16 W, 32 NW); ``0`` means no outflow (pit/flat/nodata). A downstream index
of ``-1`` marks an outlet: water leaving that pixel leaves the domain.
Slope and aspect reuse :mod:`pycopdem.derive`.
"""
import numpy as np

# pysheds 0.5 still calls np.in1d (removed in NumPy 2) — same alias pycopdem uses.
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

from pycopdem.derive import slope as slope_deg, aspect as aspect_deg  # noqa: E402  (re-exported)

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)                 # N NE E SE S SW W NW
_DROW = {64: -1, 128: -1, 1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1}
_DCOL = {64: 0, 128: 1, 1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1}


def d8_flowdir(dem: np.ndarray, transform, nodata: float = -9999.0) -> np.ndarray:
    """D8 flow direction (pysheds dirmap codes, int16) after pit filling,
    depression filling and flat resolution. ``transform`` is the window's
    affine in EPSG:6933 (metres), so no degree/metre mixing."""
    import os
    import tempfile
    import rasterio
    from pysheds.grid import Grid

    filled = np.where(np.isnan(dem), nodata, dem).astype('float32')
    fd, path = tempfile.mkstemp(suffix='.tif')
    os.close(fd)
    try:
        with rasterio.open(path, 'w', driver='GTiff', height=dem.shape[0], width=dem.shape[1],
                           count=1, dtype='float32', transform=transform, crs='EPSG:6933',
                           nodata=nodata) as dst:
            dst.write(filled, 1)
        grid = Grid.from_raster(path, nodata=nodata)
        raster = grid.read_raster(path, nodata=nodata)
        inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(raster)))
        fdir = grid.flowdir(inflated, dirmap=DIRMAP, nodata_out=0)
        return np.asarray(fdir).astype(np.int16)
    finally:
        os.remove(path)


def d8_flowdir_simple(dem: np.ndarray, cell_m: float) -> np.ndarray:
    """Steepest-descent D8 without conditioning (pure numpy; for tests and
    as a fallback). Cells with no lower neighbour get 0."""
    ny, nx = dem.shape
    pad = np.pad(dem.astype(float), 1, constant_values=np.nan)
    best = np.zeros((ny, nx), dtype=np.int16)
    best_drop = np.zeros((ny, nx))
    for code in DIRMAP:
        dr, dc = _DROW[code], _DCOL[code]
        dist = cell_m * (np.sqrt(2) if dr and dc else 1.0)
        nb = pad[1 + dr:1 + dr + ny, 1 + dc:1 + dc + nx]
        drop = (dem - nb) / dist
        drop = np.where(np.isnan(drop), -np.inf, drop)
        better = drop > best_drop
        best = np.where(better, code, best)
        best_drop = np.where(better, drop, best_drop)
    return best


def downstream_index(fdir: np.ndarray):
    """Flat index of each pixel's downstream pixel (``-1`` = outlet: pit/flat,
    flow off the window edge, or a pixel on a flow-direction cycle), plus the
    boolean outlet mask. Cycles are broken here so the result is always acyclic."""
    ny, nx = fdir.shape
    rows, cols = np.indices((ny, nx))
    drow = np.zeros_like(fdir, dtype=np.int64)
    dcol = np.zeros_like(fdir, dtype=np.int64)
    for code in DIRMAP:
        m = fdir == code
        drow[m] = _DROW[code]
        dcol[m] = _DCOL[code]
    r2, c2 = rows + drow, cols + dcol
    inside = (fdir != 0) & (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
    ds = np.where(inside, r2 * nx + c2, -1).ravel()
    ds, n_cut = break_cycles(ds)
    if n_cut:
        import warnings
        warnings.warn(f'{n_cut} pixels on flow-direction cycles were made outlets')
    return ds, ds < 0


def break_cycles(ds_index: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a copy of ``ds_index`` with every pixel that sits on a directed
    cycle (flats whose D8 arrows point at each other) turned into an outlet
    (``-1``), plus the number of pixels changed. Pixels upstream of a cycle
    keep their arrows and drain into the new outlet."""
    ds = ds_index.copy()
    n = ds.size
    indeg = np.bincount(ds[ds >= 0], minlength=n)
    alive = np.ones(n, dtype=bool)
    frontier = np.flatnonzero(indeg == 0)
    while frontier.size:                              # peel the trees away; what survives lies on/under cycles
        alive[frontier] = False
        tgt = ds[frontier]; tgt = tgt[tgt >= 0]
        indeg = indeg - np.bincount(tgt, minlength=n)
        frontier = np.flatnonzero(alive & (indeg == 0))
    survivors = np.flatnonzero(alive)
    if survivors.size == 0:
        return ds, 0
    # every survivor has out-degree 1 and in-degree >= 1 inside the survivor set,
    # so the survivor subgraph is exactly a union of simple cycles.
    ds[survivors] = -1
    return ds, int(survivors.size)


def topological_levels(ds_index: np.ndarray):
    """Order pixels so every pixel comes after all pixels draining into it.

    Kahn's algorithm, one level at a time: level 0 = pixels with no inflow
    (ridges), level k = pixels whose last upstream pixel sat in level k-1.
    Returns ``(order, level_ptr)`` with ``order[level_ptr[k]:level_ptr[k+1]]``
    the flat indices of level ``k``. ``ds_index`` must be acyclic (see
    :func:`break_cycles`); a cycle raises, because routing over it would
    lose water."""
    n = ds_index.size
    has_ds = ds_index >= 0
    indeg = np.bincount(ds_index[has_ds], minlength=n)
    current = np.flatnonzero(indeg == 0)
    order, ptr = [], [0]
    seen = 0
    while current.size:
        order.append(current)
        seen += current.size
        ptr.append(seen)
        targets = ds_index[current]
        targets = targets[targets >= 0]
        if targets.size == 0:
            break
        dec = np.bincount(targets, minlength=n)
        indeg = indeg - dec
        current = np.flatnonzero((indeg == 0) & (dec > 0))
    if seen < n:
        raise ValueError(f'{n - seen} pixels lie on or below a flow-direction cycle; call break_cycles() first')
    return np.concatenate(order), np.asarray(ptr, dtype=np.int64)


def accumulation(ds_index: np.ndarray, order: np.ndarray, level_ptr: np.ndarray) -> np.ndarray:
    """Number of pixels draining through each pixel (including itself)."""
    acc = np.ones(ds_index.size)
    for k in range(len(level_ptr) - 1):
        idx = order[level_ptr[k]:level_ptr[k + 1]]
        dst = ds_index[idx]
        ok = dst >= 0
        np.add.at(acc, dst[ok], acc[idx[ok]])
    return acc


def twi(acc_cells: np.ndarray, slope_degrees: np.ndarray, cell_m: float, min_slope_deg: float = 0.1) -> np.ndarray:
    """Topographic wetness index ln(a / tan(beta)) with specific catchment area
    ``a = acc * cell_m`` and slope floored at ``min_slope_deg``."""
    a = acc_cells * cell_m
    beta = np.radians(np.maximum(slope_degrees, min_slope_deg))
    return np.log(a / np.tan(beta))


# -- tests -----------------------------------------------------------------------

def _tilted(ny=6, nx=5, dz=1.0):
    # Elevation increases with row index -> water flows north (decreasing row).
    return np.arange(ny)[:, None] * dz + np.zeros((1, nx))


def test_simple_flowdir_on_tilted_plane():
    fdir = d8_flowdir_simple(_tilted(), 10.0)
    return bool((fdir[1:] == 64).all()) and bool((fdir[0] == 0).all())


def test_downstream_and_levels_on_plane():
    fdir = d8_flowdir_simple(_tilted(), 10.0)
    ds, outlet = downstream_index(fdir)
    order, ptr = topological_levels(ds)
    ny, nx = fdir.shape
    # ridge row (last) is level 0; outlet row (first) is the last level
    level0 = set(order[ptr[0]:ptr[1]].tolist())
    last = set(order[ptr[-2]:ptr[-1]].tolist())
    return (level0 == set(range((ny - 1) * nx, ny * nx)) and last == set(range(nx))
            and outlet.sum() == nx and len(ptr) - 1 == ny)


def test_accumulation_matches_brute_force_on_valley():
    # V-shaped valley draining north along the centre column.
    ny, nx = 7, 5
    cols = np.abs(np.arange(nx) - nx // 2)[None, :] * 1.0
    dem = np.arange(ny)[:, None] * 1.0 + cols
    fdir = d8_flowdir_simple(dem, 10.0)
    ds, _ = downstream_index(fdir)
    order, ptr = topological_levels(ds)
    acc = accumulation(ds, order, ptr)
    # brute force: follow each pixel downstream, counting visits
    ref = np.zeros(ny * nx)
    for i in range(ny * nx):
        j = i
        while j >= 0:
            ref[j] += 1
            j = ds[j]
    return np.array_equal(acc, ref) and acc.max() == ny * nx and acc.argmax() == nx // 2


def test_cycles_are_broken_and_levels_reject_cycles():
    # 3 pixels: 0 -> 1 -> 2 -> 1 (2-cycle between 1 and 2), plus 3 -> 0
    ds = np.array([1, 2, 1, 0])
    fixed, n_cut = break_cycles(ds)
    ok_cut = n_cut == 2 and fixed.tolist() == [1, -1, -1, 0]
    try:
        topological_levels(ds); raised = False
    except ValueError:
        raised = True
    order, ptr = topological_levels(fixed)
    return ok_cut and raised and len(order) == 4 and order[ptr[0]:ptr[1]].tolist() == [2, 3]


def test_twi_increases_with_accumulation():
    t = twi(np.array([1.0, 10.0, 100.0]), np.array([5.0, 5.0, 5.0]), 10.0)
    return bool(np.all(np.diff(t) > 0))


def test():
    return all([
        test_simple_flowdir_on_tilted_plane(),
        test_downstream_and_levels_on_plane(),
        test_accumulation_matches_brute_force_on_valley(),
        test_cycles_are_broken_and_levels_reject_cycles(),
        test_twi_increases_with_accumulation(),
    ])


if __name__ == '__main__':
    print(test())
