"""The model grid: the pysentinel2 10 m EPSG:6933 grid, so soil moisture
lines up pixel-for-pixel with Sentinel-2 NDVI and with paddock rasters.

Thin, pure wrappers around :mod:`pysentinel2.grid` plus the extras terrasim
needs: a buffered window for flow routing, pixel-centre lat/lon for mapping
pixels onto the coarse climate grids, and the static cache key. No I/O.
"""
import math

import numpy as np
from pysentinel2 import grid as s2grid

CRS = s2grid.CRS
RES = s2grid.RES

Window = tuple[int, int, int, int]   # (row0, row1, col0, col1), half-open


def model_window(bbox: list[float]) -> Window:
    """Pixel window covering ``bbox`` exactly (pixel-snapped, not chunk-snapped)."""
    return s2grid.tight_window_for_bbox(bbox)


def buffered_window(bbox: list[float], buffer_m: float) -> Window:
    """``model_window`` grown by ``buffer_m`` on every side (for DEM conditioning)."""
    row0, row1, col0, col1 = model_window(bbox)
    pad = int(math.ceil(buffer_m / RES))
    return (row0 - pad, row1 + pad, col0 - pad, col1 + pad)


def inner_slices(outer: Window, inner: Window) -> tuple[slice, slice]:
    """Array slices that crop an ``outer``-window array down to ``inner``."""
    return (slice(inner[0] - outer[0], inner[1] - outer[0]),
            slice(inner[2] - outer[2], inner[3] - outer[2]))


def shape(window: Window) -> tuple[int, int]:
    return (window[1] - window[0], window[3] - window[2])


def coords(window: Window):
    """Pixel-centre ``(y, x)`` coordinate arrays (y descending), metres."""
    return s2grid.coords_for_window(window)


def geobox(window: Window):
    """``odc.geo.geobox.GeoBox`` for the window (for ``rio.reproject_match``)."""
    return s2grid.geobox_for_window(window)


def transform(window: Window):
    """Affine transform of the window's top-left corner."""
    from affine import Affine
    row0, _, col0, _ = window
    return Affine(RES, 0, s2grid.X0 + col0 * RES, 0, -RES, s2grid.Y_TOP - row0 * RES)


def bbox_4326(window: Window) -> list[float]:
    """``[west, south, east, north]`` of the window."""
    return s2grid.window_to_4326(window)


def pixel_latlon(window: Window):
    """Pixel-centre latitude and longitude, each shape ``(ny, nx)``.

    EPSG:6933 is cylindrical: x depends only on lon and y only on lat, so
    two 1-D transforms are exact and are then broadcast."""
    y, x = coords(window)
    lon, _ = s2grid._to_4326.transform(x, np.full_like(x, y[0]))
    _, lat = s2grid._to_4326.transform(np.full_like(y, x[0]), y)
    return np.broadcast_to(np.asarray(lat)[:, None], (len(y), len(x))).copy(), \
        np.broadcast_to(np.asarray(lon)[None, :], (len(y), len(x))).copy()


def static_key(bbox_hash: str, layer_key: str, ptf_version: str) -> str:
    """Cache key of a static dataset: region x layer geometry x pedotransfer version."""
    return f'{bbox_hash[:16]}-{layer_key}-{ptf_version}'


# -- tests -----------------------------------------------------------------------

_BBOX = [148.36265, -33.52606, 148.38265, -33.50606]


def test_coords_match_pysentinel2():
    w = model_window(_BBOX)
    y, x = coords(w)
    y2, x2 = s2grid.coords_for_window(w)
    return np.array_equal(y, y2) and np.array_equal(x, x2) and shape(w) == (len(y), len(x))


def test_buffered_contains_model_and_slices_crop_back():
    w = model_window(_BBOX)
    b = buffered_window(_BBOX, 500)
    arr = np.zeros(shape(b))
    sl = inner_slices(b, w)
    return b[0] < w[0] and b[1] > w[1] and arr[sl].shape == shape(w) and (b[1] - b[0]) - (w[1] - w[0]) == 100


def test_pixel_latlon_inside_bbox():
    w = model_window(_BBOX)
    lat, lon = pixel_latlon(w)
    return (lat.shape == shape(w) and lat.max() <= _BBOX[3] + 1e-3 and lat.min() >= _BBOX[1] - 1e-3
            and lon.min() >= _BBOX[0] - 1e-3 and lon.max() <= _BBOX[2] + 1e-3
            and lat[0, 0] > lat[-1, 0] and lon[0, 0] < lon[0, -1])


def test_transform_matches_geobox():
    w = model_window(_BBOX)
    return transform(w) == geobox(w).transform


def test():
    return all([
        test_coords_match_pysentinel2(),
        test_buffered_contains_model_and_slices_crop_back(),
        test_pixel_latlon_inside_bbox(),
        test_transform_matches_geobox(),
    ])


if __name__ == '__main__':
    print(test())
