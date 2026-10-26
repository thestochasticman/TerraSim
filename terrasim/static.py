"""Build the static terrain + soil dataset for a region on the 10 m model grid.

Inputs (through the lab stores, or any callables with the same signatures
for offline tests):

- **SLGA** soil properties at six depths on a ~90 m EPSG:4326 grid
  (`pyslga`): clipped to the region, resampled bilinearly onto the model
  grid, thickness-weighted onto the model layers, then turned into
  hydraulic parameters by :mod:`terrasim.pedotransfer`.
- **Copernicus DEM** at 30 m (`pycopdem`): resampled onto a *buffered*
  model grid, lightly smoothed, run through D8 conditioning, and the flow
  geometry (downstream index, topological levels), accumulation, slope
  and TWI cropped back to the model window.

The result is one ``xr.Dataset`` with dims ``(layer, y, x)`` for layered
soil variables and ``(y, x)`` for terrain, cached as zarr under
``Paths.static_dir`` keyed by region, layer geometry and pedotransfer
version. :func:`to_arrays` turns it into the engine's :class:`StaticArrays`.
"""
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import xarray as xr
from attrs import frozen, field
from scipy.ndimage import gaussian_filter

from troi import Config, config as default_config
from troi.troi import encode as _sha
from terrasim import grid, terrain, pedotransfer as ptf
from terrasim.layers import LayerSpec, overlap_matrix
from terrasim.params import Params, SLGA_DEPTHS
from terrasim.paths import Paths
from terrasim.terrasim import Terrasim, defaultterrasim
from terrasim.physics.state import StaticArrays, ze_overlap_mm

SLGA_BOUNDS_M = np.array([[0, .05], [.05, .15], [.15, .30], [.30, .60], [.60, 1.0], [1.0, 2.0]])
_DEG_PAD = 0.003   # ~300 m of extra source pixels around the window before reprojection


def params_key(params: Params) -> str:
    """Short hash of every parameter that changes the static dataset (soil knobs,
    root profile, SLGA selections), so a calibration tweak never hits a stale cache."""
    from attrs import asdict
    payload = json.dumps({'soil': asdict(params.soil), 'root': [params.veg.root_depth_m, params.veg.root_beta],
                          'slga': [list(params.slga_attributes), list(params.slga_optional), list(params.slga_depths)]},
                         sort_keys=True, default=str)
    return _sha(payload)[:8]


def bbox_hash(bbox) -> str:
    """Same region identity as ``troi.Troi.bbox_hash`` (bbox snapped to ~100 m)."""
    return _sha(str([round(c, 3) for c in bbox]))


def _template(window) -> xr.DataArray:
    y, x = grid.coords(window)
    da = xr.DataArray(np.zeros((len(y), len(x)), dtype='float32'), dims=('y', 'x'), coords={'y': y, 'x': x})
    return da.rio.write_crs(grid.CRS)


def reproject_to_grid(da: xr.DataArray, window, resampling='bilinear') -> np.ndarray:
    """Reproject a ``(lat, lon)`` EPSG:4326 array onto the model window; NaN is
    carried as nodata and remaining holes are filled from the nearest valid pixel."""
    import rioxarray  # noqa: F401  (registers .rio)
    from rasterio.enums import Resampling
    src = da.astype('float64').rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.write_crs('EPSG:4326')
    src = src.rio.write_nodata(np.nan, inplace=False)
    out = src.rio.reproject_match(_template(window), resampling=getattr(Resampling, resampling), nodata=np.nan)
    arr = np.asarray(out.values, dtype='float64')
    if np.isnan(arr).any() and np.isfinite(arr).any():
        arr = _fill_nearest(arr)
    return arr


def _fill_nearest(arr: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(arr)
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def _clip_4326(ds: xr.Dataset, bbox, pad=_DEG_PAD) -> xr.Dataset:
    west, south, east, north = bbox
    lat = ds['lat']
    lat_slice = slice(north + pad, south - pad) if lat[0] > lat[-1] else slice(south - pad, north + pad)
    return ds.sel(lat=lat_slice, lon=slice(west - pad, east + pad))


def slga_to_layers(per_depth: np.ndarray, layers: LayerSpec) -> np.ndarray:
    """``(6, ny, nx)`` SLGA depth slices -> ``(nlayer, ny, nx)`` by thickness weighting
    (the 100-200 cm slice extends below 2 m)."""
    m = overlap_matrix(SLGA_BOUNDS_M, layers.bounds_m())
    return np.tensordot(m, per_depth, axes=(1, 0))


def _as_fraction(arr: np.ndarray, name: str) -> np.ndarray:
    """SLGA serves texture, DUL and L15 as percentages; accept either."""
    med = np.nanmedian(arr)
    return arr / 100.0 if med > 1.0 else arr


@frozen
class SLGACatalog:
    """pyslga's ``SLGA`` endpoint catalog with per-attribute versions and the
    depth codes of the single-layer products.

    TERN publishes each attribute on its own schedule and version: texture and
    bulk density are v2, but Drained Upper Limit and L15 only exist as v1, and
    Depth to Rock / Depth of Soil are single layers coded ``000_999`` /
    ``000_200``. This catalog is passed to ``pyslga.store.Store(slga=...)`` —
    composition, same duck-typed interface (``attribute_codes``, ``depth_codes``,
    ``layer_key``, ``listing_url``).
    """

    base_url: str = 'https://data.tern.org.au/model-derived/slga/NationalMaps/SoilAndLandscapeGrid'
    default_version: str = 'v2'
    versions: dict = {'Drained_Upper_Limit': 'v1', 'L15': 'v1'}
    attribute_codes: dict = {
        'Clay': 'CLY', 'Silt': 'SLT', 'Sand': 'SND', 'pH_Water': 'PHW', 'Bulk_Density': 'BDW',
        'Available_Water_Capacity': 'AWC', 'Cation_Exchange_Capacity': 'CEC',
        'Effective_Cation_Exchange_Capacity': 'ECE', 'Total_Nitrogen': 'NTO', 'Total_Phosphorus': 'PTO',
        'Coarse_Fragments': 'CFG', 'Depth_of_Soil': 'DES', 'Depth_to_Rock': 'DER',
        'Drained_Upper_Limit': 'DUL', 'L15': 'L15', 'Available_Phosphorus': 'AVP',
    }
    depth_codes: dict = {
        '0-5cm': ('000', '005'), '5-15cm': ('005', '015'), '15-30cm': ('015', '030'),
        '30-60cm': ('030', '060'), '60-100cm': ('060', '100'), '100-200cm': ('100', '200'),
        '0-200cm': ('000', '200'), '0-999cm': ('000', '999'),
    }

    def layer_key(s, attribute: str, depth: str) -> str:
        if attribute not in s.attribute_codes:
            raise ValueError(f'Unknown SLGA attribute {attribute!r}')
        if depth not in s.depth_codes:
            raise ValueError(f'Unknown SLGA depth {depth!r}')
        ds, de = s.depth_codes[depth]
        return f'{s.attribute_codes[attribute]}_{ds}_{de}'

    def listing_url(s, attribute: str) -> str:
        return f'{s.base_url}/{s.attribute_codes[attribute]}/{s.versions.get(attribute, s.default_version)}/'


SINGLE_LAYER_DEPTHS = {'Depth_of_Soil': ('0-200cm',), 'Depth_to_Rock': ('0-999cm',)}


def _default_slga_fetch(config: Config, catalog: SLGACatalog = SLGACatalog()):
    def fetch(bbox, attributes, depths, api_key=None):
        from pyslga.store import Store
        return Store(config=config, slga=catalog).get_ds(bbox, attributes=attributes, depths=depths, api_key=api_key)
    return fetch


def _default_dem_fetch(config: Config):
    def fetch(bbox):
        from pycopdem.store import Store
        return Store(config=config).get_ds(bbox)
    return fetch


@frozen
class StaticBuilder:
    """Builds (and caches) the static dataset for a bbox.

    Attributes:
        config: Where caches live (and the TERN key for SLGA).
        params, layers, settings: Determine derivations, geometry and DEM buffer.
        slga_fetch: ``(bbox, attributes, depths, api_key) -> xr.Dataset (lat, lon)``;
            defaults to pyslga.
        dem_fetch: ``(bbox) -> xr.Dataset`` with ``elevation (lat, lon)``; defaults to pycopdem.
    """

    config: Config = default_config
    params: Params = Params()
    layers: LayerSpec = LayerSpec()
    settings: Terrasim = defaultterrasim
    slga_fetch: object = field()
    dem_fetch: object = field()
    paths: Paths = field(init=False)

    paths.default(lambda s: Paths(s.config))

    @slga_fetch.default
    def _slga_default(s):
        return _default_slga_fetch(s.config)

    @dem_fetch.default
    def _dem_default(s):
        return _default_dem_fetch(s.config)

    def key(s, bbox) -> str:
        return grid.static_key(bbox_hash(bbox), s.layers.key(), f'{s.params.pedotransfer_version}-{params_key(s.params)}')

    def build(s, bbox: list[float], api_key: str = None, force: bool = False) -> xr.Dataset:
        path = s.paths.static_zarr(s.key(bbox))
        if not force and os.path.exists(f'{path}/_SUCCESS'):
            return xr.open_zarr(path).load()
        ds = s.build_uncached(bbox, api_key=api_key)
        os.makedirs(s.paths.static_dir, exist_ok=True)
        ds.to_zarr(path, mode='w')
        open(f'{path}/_SUCCESS', 'w').close()
        return ds

    def build_troi(s, troi, api_key: str = None, force: bool = False) -> xr.Dataset:
        return s.build(troi.bbox, api_key=api_key, force=force)

    # -- the work -------------------------------------------------------------

    def build_uncached(s, bbox: list[float], api_key: str = None) -> xr.Dataset:
        window = grid.model_window(bbox)
        soil = s._soil(bbox, window, api_key)
        terr = s._terrain(bbox, window)
        return assemble_static(window, s.layers, s.params, soil, terr, bbox=bbox, key=s.key(bbox))

    def _soil(s, bbox, window, api_key) -> dict:
        """Required SLGA layers (texture, bulk density) plus whatever optional
        layers the datastore actually has; each optional attribute fails soft."""
        depths = tuple(s.params.slga_depths)
        wb = grid.bbox_4326(window)
        raw = _clip_4326(s.slga_fetch(wb, tuple(s.params.slga_attributes), depths, api_key), wb)
        out = {}
        for attr in s.params.slga_attributes:
            out[attr] = np.stack([reproject_to_grid(raw[f'{attr}_{d}'], window) for d in depths])
        for attr in s.params.slga_optional:
            out[attr] = None
            want = SINGLE_LAYER_DEPTHS.get(attr, depths)
            try:
                opt = _clip_4326(s.slga_fetch(wb, (attr,), want, api_key), wb)
            except Exception as ex:   # noqa: BLE001 — listing/version gaps on TERN are common
                warnings.warn(f'SLGA {attr} unavailable ({type(ex).__name__}: {str(ex)[:120]}); '
                              + ('deriving from texture (Saxton & Rawls 2006)' if attr in ('Drained_Upper_Limit', 'L15')
                                 else 'deep drainage unrestricted'))
                continue
            names = [v for v in opt.data_vars if v.startswith(attr)]
            if attr in ('Depth_of_Soil', 'Depth_to_Rock'):
                out[attr] = reproject_to_grid(opt[names[0]], window) if names else None
            elif all(f'{attr}_{d}' in opt for d in depths):
                out[attr] = np.stack([reproject_to_grid(opt[f'{attr}_{d}'], window) for d in depths])
        return out

    def _terrain(s, bbox, window) -> dict:
        buffered = grid.buffered_window(bbox, s.settings.buffer_m)
        dem_ds = s.dem_fetch(grid.bbox_4326(buffered))
        dem_ds = _clip_4326(dem_ds, grid.bbox_4326(buffered))
        return terrain_from_dem(dem_ds['elevation'], buffered, window)


def terrain_from_dem(elevation: xr.DataArray, buffered, window, smooth_sigma: float = 1.0) -> dict:
    """Reproject a (lat, lon) DEM onto the buffered model grid and derive the
    routing geometry, cropped to the model window."""
    dem_b = reproject_to_grid(elevation, buffered)
    if np.isnan(dem_b).all():
        raise ValueError('DEM window is entirely nodata')
    dem_b = _fill_nearest(dem_b) if np.isnan(dem_b).any() else dem_b
    if smooth_sigma > 0:
        dem_b = gaussian_filter(dem_b, smooth_sigma)
    fdir_b = terrain.d8_flowdir(dem_b, grid.transform(buffered))
    ds_b, _ = terrain.downstream_index(fdir_b)
    order_b, ptr_b = terrain.topological_levels(ds_b)
    acc_b = terrain.accumulation(ds_b, order_b, ptr_b).reshape(dem_b.shape)
    slope_b = terrain.slope_deg(dem_b, grid.RES, grid.RES)
    aspect_b = terrain.aspect_deg(dem_b, grid.RES, grid.RES)
    twi_b = terrain.twi(acc_b, slope_b, grid.RES)
    sl = grid.inner_slices(buffered, window)
    fdir = fdir_b[sl]
    ds_index, _ = terrain.downstream_index(fdir)          # edge pixels of the window become outlets
    order, ptr = terrain.topological_levels(ds_index)
    return dict(elevation=dem_b[sl], fdir=fdir, slope=slope_b[sl], aspect=aspect_b[sl],
                accumulation=acc_b[sl], twi=twi_b[sl], ds_index=ds_index, level_order=order, level_ptr=ptr)


def assemble_static(window, layers: LayerSpec, params: Params, soil: dict, terr: dict, bbox=None, key='') -> xr.Dataset:
    """Pure assembly: SLGA arrays on the grid + terrain dict -> static Dataset."""
    L = layers.nlayer
    ny, nx = grid.shape(window)
    sp = params.soil
    clay = slga_to_layers(_as_fraction(soil['Clay'], 'Clay'), layers)
    sand = slga_to_layers(_as_fraction(soil['Sand'], 'Sand'), layers)
    silt = slga_to_layers(_as_fraction(soil['Silt'], 'Silt'), layers)
    bd = slga_to_layers(soil['Bulk_Density'], layers)

    valid = np.isfinite(bd[0]) & np.isfinite(clay[0]) & np.isfinite(sand[0])
    for arr, fill in ((clay, 0.25), (sand, 0.45), (silt, 0.30), (bd, 1.4)):
        med = np.nanmedian(arr, axis=(1, 2))
        for i in range(L):
            np.nan_to_num(arr[i], copy=False, nan=(med[i] if np.isfinite(med[i]) else fill))

    # retention points: SLGA DUL/L15 where available, else Saxton & Rawls from texture
    sr_dul, sr_l15 = ptf.retention_saxton_rawls(sand, clay, sp.organic_matter_pct)
    retention_source = []
    if soil.get('Drained_Upper_Limit') is not None:
        dul = slga_to_layers(_as_fraction(soil['Drained_Upper_Limit'], 'DUL'), layers)
        dul = np.where(np.isfinite(dul), dul, sr_dul); retention_source.append('DUL: SLGA')
    else:
        dul = sr_dul; retention_source.append('DUL: Saxton-Rawls(texture)')
    if soil.get('L15') is not None:
        l15 = slga_to_layers(_as_fraction(soil['L15'], 'L15'), layers)
        l15 = np.where(np.isfinite(l15), l15, sr_l15); retention_source.append('L15: SLGA')
    else:
        l15 = sr_l15; retention_source.append('L15: Saxton-Rawls(texture)')
    l15 = np.minimum(l15, dul - 0.01)
    theta_s = ptf.porosity(bd, sp.rho_particle, sp.theta_s_min, sp.theta_s_max)
    theta_s = np.maximum(theta_s, dul + 0.02)
    b, psi_e = ptf.campbell_from_limits(theta_s, dul, l15, sp.b_min, sp.b_max)
    bad = ~np.isfinite(b) | ~np.isfinite(psi_e)
    if bad.any():
        bc, pc, _, _ = ptf.campbell_cosby(sand * 100, clay * 100)
        b = np.where(bad, bc, b); psi_e = np.where(bad, pc, psi_e)
    b = np.clip(b * sp.b_multiplier, sp.b_min, sp.b_max)
    ksat = ptf.ksat_saxton_rawls(theta_s, dul, b, sp.ksat_min, sp.ksat_max) * sp.ksat_multiplier
    ksat = np.clip(ksat, sp.ksat_min, sp.ksat_max)

    ze_mm = ze_overlap_mm(layers, sp.ze_m)
    zw = ze_mm / ze_mm.sum()
    rew, tew = ptf.rew_tew(np.tensordot(zw, sand, 1), np.tensordot(zw, dul, 1), np.tensordot(zw, l15, 1), sp.ze_m, sp.rew_mm)
    psi_f = ptf.wetting_front_suction(psi_e[0], b[0])
    root = ptf.root_fraction(layers.bounds_m(), params.veg.root_depth_m, params.veg.root_beta)
    root_frac = np.broadcast_to(root[:, None, None], (L, ny, nx)).copy()

    depth_rock = soil.get('Depth_to_Rock')
    depth_soil = soil.get('Depth_of_Soil')
    k_bottom = np.ones((ny, nx))
    if depth_rock is not None:
        dr = np.where(np.isfinite(depth_rock), depth_rock, np.inf)
        k_bottom = np.where(layers.depth() > dr, sp.bedrock_leak_frac, 1.0)

    y, x = grid.coords(window)
    lat, lon = grid.pixel_latlon(window)
    bounds = layers.bounds_m()
    coords = {'layer': np.arange(L), 'y': y, 'x': x,
              'layer_top_m': ('layer', bounds[:, 0]), 'layer_bottom_m': ('layer', bounds[:, 1]),
              'lat': (('y', 'x'), lat), 'lon': (('y', 'x'), lon)}
    L3 = ('layer', 'y', 'x'); P2 = ('y', 'x')
    f32 = lambda a: np.asarray(a, dtype='float32')
    data = {
        'clay': (L3, f32(clay)), 'sand': (L3, f32(sand)), 'silt': (L3, f32(silt)), 'bulk_density': (L3, f32(bd)),
        'theta_s': (L3, f32(theta_s)), 'theta_dul': (L3, f32(dul)), 'theta_l15': (L3, f32(l15)),
        'ksat': (L3, f32(ksat)), 'b': (L3, f32(b)), 'psi_e': (L3, f32(psi_e)), 'root_frac': (L3, f32(root_frac)),
        'psi_f': (P2, f32(psi_f)), 'rew': (P2, f32(rew)), 'tew': (P2, f32(tew)),
        'k_bottom_scale': (P2, f32(k_bottom)), 'valid': (P2, valid),
        'elevation': (P2, f32(terr['elevation'])), 'slope': (P2, f32(terr['slope'])), 'aspect': (P2, f32(terr['aspect'])),
        'accumulation': (P2, f32(terr['accumulation'])), 'twi': (P2, f32(terr['twi'])),
        'fdir': (P2, np.asarray(terr['fdir'], dtype='int16')),
        'ds_index': (P2, np.asarray(terr['ds_index'], dtype='int64').reshape(ny, nx)),
        'level_order': (('pixel',), np.asarray(terr['level_order'], dtype='int64')),
        'level_ptr': (('level_bound',), np.asarray(terr['level_ptr'], dtype='int64')),
        'ze_mm': (('layer',), f32(ze_mm)),
    }
    if depth_rock is not None:
        data['depth_to_rock'] = (P2, f32(depth_rock))
    if depth_soil is not None:
        data['depth_of_soil'] = (P2, f32(depth_soil))
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update({
        'crs': grid.CRS, 'resolution_m': grid.RES, 'window': json.dumps(list(map(int, window))),
        'bbox': json.dumps(bbox) if bbox is not None else '', 'key': key,
        'layers': json.dumps(layers.to_daesim_kwargs()), 'pedotransfer_version': params.pedotransfer_version,
        'units': 'theta m3 m-3; ksat mm h-1; psi_e, psi_f, rew, tew mm; elevation m; slope, aspect deg',
        'built_at': datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z',
        'sources': 'SLGA v2 (TERN) via pyslga; Copernicus GLO-30 via pycopdem',
        'retention_source': '; '.join(retention_source),
    })
    return ds


def to_arrays(ds: xr.Dataset) -> StaticArrays:
    """Engine view of a static dataset (float64 for the physics)."""
    g = lambda n: np.ascontiguousarray(ds[n].values.astype('float64'))
    return StaticArrays(
        dz_mm=(ds['layer_bottom_m'].values - ds['layer_top_m'].values) * 1000.0,
        theta_s=g('theta_s'), theta_dul=g('theta_dul'), theta_l15=g('theta_l15'),
        ksat=g('ksat'), b=g('b'), psi_e=g('psi_e'), psi_f=g('psi_f'), rew=g('rew'), tew=g('tew'),
        ze_mm=g('ze_mm'), root_frac=g('root_frac'), k_bottom_scale=g('k_bottom_scale'),
        valid=ds['valid'].values.astype(bool),
        ds_index=ds['ds_index'].values.astype('int64').ravel(),
        level_order=ds['level_order'].values.astype('int64'), level_ptr=ds['level_ptr'].values.astype('int64'),
        twi=g('twi'), elevation=g('elevation'),
        slope=np.tan(np.radians(g('slope'))) if 'slope' in ds else None,
    )


# -- synthetic inputs (tests, demos) --------------------------------------------------

def synthetic_slga(bbox, attributes=None, depths=SLGA_DEPTHS, api_key=None, res_deg=0.0008333) -> xr.Dataset:
    """A fake ~90 m SLGA window with a west-east clay gradient, in pyslga's layout
    (percent units, dims (lat, lon), lat descending)."""
    attributes = attributes or Params().slga_attributes
    for attr in attributes:
        if attr not in ('Clay', 'Sand', 'Silt', 'Bulk_Density', 'Drained_Upper_Limit', 'L15', 'Depth_of_Soil', 'Depth_to_Rock'):
            raise ValueError(f'unknown attribute {attr}')
    west, south, east, north = bbox
    pad = 0.01
    lat = np.arange(north + pad, south - pad, -res_deg)
    lon = np.arange(west - pad, east + pad, res_deg)
    gx = (lon - lon.min()) / max(lon.max() - lon.min(), 1e-9)
    base = {'Clay': 20 + 20 * gx, 'Sand': 60 - 20 * gx, 'Silt': np.full_like(gx, 20.0),
            'Bulk_Density': 1.3 + 0.2 * gx, 'Drained_Upper_Limit': 25 + 12 * gx, 'L15': 10 + 10 * gx}
    data = {}
    for attr in attributes:
        if attr in ('Depth_of_Soil', 'Depth_to_Rock'):
            data[f'{attr}_{depths[0]}'] = (('lat', 'lon'), np.full((len(lat), len(lon)), 1.5 if attr == 'Depth_to_Rock' else 1.2, dtype='float32'))
            continue
        for k, d in enumerate(depths):
            prof = 1.0 + 0.05 * k if attr in ('Bulk_Density', 'Clay', 'Drained_Upper_Limit', 'L15') else 1.0 - 0.03 * k
            data[f'{attr}_{d}'] = (('lat', 'lon'), np.broadcast_to((base[attr] * prof)[None, :], (len(lat), len(lon))).astype('float32'))
    return xr.Dataset(data, coords={'lat': lat, 'lon': lon}, attrs={'crs': 'EPSG:4326', 'source': 'synthetic'})


def synthetic_dem(bbox, res_deg=1 / 3600, slope_m_per_deg_lat=2000.0) -> xr.Dataset:
    """A fake 30 m DEM: a plane rising to the north with a shallow central valley."""
    west, south, east, north = bbox
    pad = 0.01
    lat = np.arange(north + pad, south - pad, -res_deg)
    lon = np.arange(west - pad, east + pad, res_deg)
    z = 300 + (lat[:, None] - south) * slope_m_per_deg_lat + np.abs(lon[None, :] - (west + east) / 2) * 1500.0
    return xr.Dataset({'elevation': (('lat', 'lon'), z.astype('float32'))}, coords={'lat': lat, 'lon': lon},
                      attrs={'crs': 'EPSG:4326', 'resolution_arcsec': 1})


# -- tests ---------------------------------------------------------------------------

_BBOX = [148.36265, -33.52606, 148.38265, -33.50606]


def test_slga_layer_weighting_matches_hand_calculation():
    per_depth = np.arange(1, 7, dtype=float)[:, None, None] * np.ones((6, 1, 1))
    out = slga_to_layers(per_depth, LayerSpec())
    # default layers coincide with SLGA depths -> identity
    return np.allclose(out[:, 0, 0], np.arange(1, 7))


def _no_hydraulics(bbox, attributes, depths, api_key=None):
    """Mimics TERN today: texture + BD present, DUL/L15/depth layers missing."""
    if any(a in ('Drained_Upper_Limit', 'L15', 'Depth_of_Soil', 'Depth_to_Rock') for a in attributes):
        raise RuntimeError('No SLGA EV COG for ' + attributes[0])
    return synthetic_slga(bbox, attributes, depths, api_key)


def test_build_without_hydraulic_layers_uses_saxton_rawls():
    import tempfile
    tmp = tempfile.mkdtemp(prefix='terrasim_static_test_')
    cfg = Config(out_dir=tmp, tmp_dir=tmp)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds = StaticBuilder(config=cfg, slga_fetch=_no_hydraulics, dem_fetch=lambda bbox: synthetic_dem(bbox)).build(_BBOX)
    return ('Saxton-Rawls' in ds.attrs['retention_source'] and bool((ds['theta_s'] > ds['theta_dul']).all())
            and bool((ds['theta_dul'] > ds['theta_l15']).all()) and 'depth_to_rock' not in ds)


def test_cache_key_changes_with_soil_params():
    from terrasim.params import SoilParams
    a = StaticBuilder(params=Params()).key(_BBOX)
    b = StaticBuilder(params=Params(soil=SoilParams(ksat_multiplier=0.2))).key(_BBOX)
    return a != b and a == StaticBuilder().key(_BBOX)


def test_synthetic_build_has_expected_shape_and_physics_ranges():
    import tempfile
    tmp = tempfile.mkdtemp(prefix='terrasim_static_test_')
    cfg = Config(out_dir=tmp, tmp_dir=tmp)
    sb = StaticBuilder(config=cfg, slga_fetch=synthetic_slga, dem_fetch=lambda bbox: synthetic_dem(bbox))
    ds = sb.build(_BBOX)
    ny, nx = grid.shape(grid.model_window(_BBOX))
    ok_shape = ds['theta_s'].shape == (6, ny, nx) and ds['elevation'].shape == (ny, nx)
    ok_phys = (bool((ds['theta_s'] > ds['theta_dul']).all()) and bool((ds['theta_dul'] > ds['theta_l15']).all())
               and float(ds['ksat'].min()) >= 0.1 and float(ds['ksat'].max()) <= 500)
    # clay rises west -> east, so ksat should fall west -> east
    ok_grad = float(ds['ksat'][0, :, :5].mean()) > float(ds['ksat'][0, :, -5:].mean())
    # cache hit: a second builder with a failing fetch must still return
    sb2 = StaticBuilder(config=cfg, slga_fetch=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('network!')),
                        dem_fetch=lambda *a: (_ for _ in ()).throw(RuntimeError('network!')))
    ds2 = sb2.build(_BBOX)
    arrays = to_arrays(ds2)
    return ok_shape and ok_phys and ok_grad and arrays.ds_index.size == ny * nx and arrays.level_ptr[-1] == ny * nx


def test():
    return all([test_slga_layer_weighting_matches_hand_calculation(), test_cache_key_changes_with_soil_params(),
                test_synthetic_build_has_expected_shape_and_physics_ranges(),
                test_build_without_hydraulic_layers_uses_saxton_rawls()])


if __name__ == '__main__':
    print(test())
