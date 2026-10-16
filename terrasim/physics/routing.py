"""Downslope redistribution of infiltration-excess runoff ("run-on").

Each router takes the local runoff produced at every pixel this step and
an ``absorb(idx, water)`` callback (the engine's infiltration applied to a
subset of pixels, returning what entered the soil), and returns the water
that arrived at each pixel from upslope, the water each pixel passed on,
and the water that left the domain.

- :class:`D8Router` walks the precomputed topological levels (ridges to
  outlets). Run-on reaching a level is offered to its pixels for
  re-infiltration with their own Green-Ampt state; what is not absorbed
  joins their local runoff and is scattered to their downstream pixels.
  One pass per step, fully vectorised per level, mass-conserving.
- :class:`TWIRouter` sends each pixel's runoff to its lower 8-neighbours
  weighted by ``exp(twi)`` and offers it for re-infiltration in one pass
  (no cascade). Kept for sensitivity comparisons.
- :class:`NullRouter`: every pixel's runoff leaves the domain directly.
"""
import numpy as np
from attrs import frozen

from terrasim.terrain import DIRMAP, _DROW, _DCOL


@frozen
class NullRouter:
    def route(s, runoff_local, absorb):
        n = runoff_local.size
        zeros = np.zeros(n)
        return zeros, runoff_local.copy(), runoff_local.copy(), zeros


@frozen
class D8Router:
    """Level-ordered D8 run-on routing.

    Attributes:
        ds_index: Downstream flat index per pixel (-1 = outlet).
        order, level_ptr: Topological levels from :func:`terrasim.terrain.topological_levels`.
    """

    ds_index: np.ndarray
    order: np.ndarray
    level_ptr: np.ndarray

    def route(s, runoff_local, absorb):
        """Returns ``(runon, runoff_pass, runoff_out, absorbed)``, all flat (n,) mm."""
        n = runoff_local.size
        inflow = np.zeros(n)
        passed = np.zeros(n)
        out = np.zeros(n)
        absorbed = np.zeros(n)
        done = np.zeros(n, dtype=bool)
        for k in range(len(s.level_ptr) - 1):
            idx = s.order[s.level_ptr[k]:s.level_ptr[k + 1]]
            done[idx] = True
            w = inflow[idx]
            has_in = w > 0
            if has_in.any():
                sub = idx[has_in]
                got = absorb(sub, w[has_in])
                absorbed[sub] = got
                w = w.copy()
                w[has_in] -= got
            passing = runoff_local[idx] + w
            moving = passing > 0
            if not moving.any():
                continue
            idx_m, passing_m = idx[moving], passing[moving]
            passed[idx_m] = passing_m
            dst = s.ds_index[idx_m]
            inside = dst >= 0
            # a target already processed (only possible with a cyclic field) cannot take water: export it
            inside &= ~done[np.where(inside, dst, 0)]
            if inside.any():
                inflow += np.bincount(dst[inside], weights=passing_m[inside], minlength=n)
            out[idx_m[~inside]] = passing_m[~inside]
        return inflow, passed, out, absorbed


@frozen
class TWIRouter:
    """One-pass redistribution to lower neighbours weighted by exp(TWI)."""

    weights: np.ndarray      # (n, 8) share of a pixel's runoff to each neighbour (rows sum to <= 1)
    targets: np.ndarray      # (n, 8) flat index of each neighbour (-1 if none/higher)

    @classmethod
    def from_terrain(cls, dem: np.ndarray, twi: np.ndarray) -> 'TWIRouter':
        ny, nx = dem.shape
        n = ny * nx
        rows, cols = np.indices((ny, nx))
        pad_dem = np.pad(dem, 1, constant_values=np.inf)
        pad_twi = np.pad(twi, 1, constant_values=-np.inf)
        weights = np.zeros((n, 8)); targets = np.full((n, 8), -1)
        for j, code in enumerate(DIRMAP):
            dr, dc = _DROW[code], _DCOL[code]
            nb_dem = pad_dem[1 + dr:1 + dr + ny, 1 + dc:1 + dc + nx]
            nb_twi = pad_twi[1 + dr:1 + dr + ny, 1 + dc:1 + dc + nx]
            lower = nb_dem < dem
            r2, c2 = rows + dr, cols + dc
            w = np.where(lower, np.exp(np.where(lower, nb_twi, 0.0)), 0.0)
            weights[:, j] = w.ravel()
            targets[:, j] = np.where(lower, r2 * nx + c2, -1).ravel()
        tot = weights.sum(axis=1, keepdims=True)
        weights = np.where(tot > 0, weights / np.where(tot > 0, tot, 1.0), 0.0)
        return cls(weights=weights, targets=targets)

    def route(s, runoff_local, absorb):
        n = runoff_local.size
        sends = runoff_local[:, None] * s.weights
        valid = s.targets >= 0
        inflow = np.bincount(s.targets[valid], weights=sends[valid], minlength=n)
        no_lower = s.weights.sum(axis=1) <= 0
        out = np.where(no_lower, runoff_local, 0.0)
        absorbed = np.zeros(n)
        has_in = inflow > 0
        if has_in.any():
            sub = np.flatnonzero(has_in)
            absorbed[sub] = absorb(sub, inflow[sub])
        out += inflow - absorbed
        passed = runoff_local.copy()
        return inflow, passed, out, absorbed


# -- tests -----------------------------------------------------------------------

def _plane_router(ny=5, nx=4):
    from terrasim import terrain
    dem = np.arange(ny)[:, None] * 1.0 + np.zeros((1, nx))
    fdir = terrain.d8_flowdir_simple(dem, 10.0)
    ds, _ = terrain.downstream_index(fdir)
    order, ptr = terrain.topological_levels(ds)
    return D8Router(ds, order, ptr), dem


def test_d8_impermeable_plane_exports_everything_from_outlet_row():
    router, dem = _plane_router()
    n = dem.size; nx = dem.shape[1]
    runoff = np.ones(n)
    runon, passed, out, absorbed = router.route(runoff, lambda idx, w: np.zeros_like(w))
    return (abs(out.sum() - runoff.sum()) < 1e-12 and np.allclose(out[:nx], dem.shape[0])
            and np.allclose(out[nx:], 0.0) and np.allclose(absorbed, 0.0))


def test_d8_absorbing_receivers_conserve_water():
    router, dem = _plane_router()
    n = dem.size
    runoff = np.ones(n)
    # every receiver swallows half of what arrives
    runon, passed, out, absorbed = router.route(runoff, lambda idx, w: 0.5 * w)
    return abs(runoff.sum() - (out.sum() + absorbed.sum())) < 1e-12 and runon.sum() > 0


def test_twi_router_conserves():
    from terrasim import terrain
    ny, nx = 5, 4
    dem = np.arange(ny)[:, None] * 1.0 + np.zeros((1, nx))
    twi = np.zeros((ny, nx))
    router = TWIRouter.from_terrain(dem, twi)
    runoff = np.ones(ny * nx)
    runon, passed, out, absorbed = router.route(runoff, lambda idx, w: 0.25 * w)
    return abs(runoff.sum() - (out.sum() + absorbed.sum())) < 1e-12


def test():
    return all([test_d8_impermeable_plane_exports_everything_from_outlet_row(),
                test_d8_absorbing_receivers_conserve_water(), test_twi_router_conserves()])


if __name__ == '__main__':
    print(test())
