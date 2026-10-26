"""Lateral subsurface flow along the terrain (throughflow), layer by layer.

Why: a soil column on its own cannot fill a valley floor. In the field,
water above field capacity moves downslope through the soil, slowly on the
slopes and collecting where the landscape converges, so floodplain and
footslope profiles sit near saturation for weeks after rain and shed
saturation-excess runoff from gentle rain that infiltrates everywhere else.
The calibration against SASMAS and MRI pointed at exactly this gap.

Formulation (kinematic throughflow, as in DHSVM / TOPLATS):

    q_i = K(theta_i) * a * tan(beta) * dt * dz_i / cell      [mm over the pixel area]

per layer ``i``: Darcy flux through the layer's downslope face at the land
slope ``beta``, with ``a`` the anisotropy (lateral over vertical
conductivity; macropores and bedding make this 10-100 in the field),
limited to the drainable water ``(theta_i - DUL_i) dz_i``. Each pixel sends
``q_i`` to its D8 downstream neighbour; water arriving in a layer fills it to
saturation, the surplus rises into the layer above, and whatever the whole
profile cannot hold exfiltrates to the surface pond, where it is routed as
runoff on the next step. Outlet pixels export to the domain boundary. The
scheme is explicit: outflow is computed from the state at the start of the
step and water moves one pixel per step (10 m per hour at the hourly step,
faster than real throughflow velocities), so there is no within-step cascade
and the whole grid is one vectorised operation. Mass is conserved to
round-off.

``LateralParams.anisotropy = 0`` (default) switches the process off, so
existing runs are unchanged. Point mode (no routing geometry) ignores it.
"""
import numpy as np
from attrs import frozen

from terrasim.physics import campbell


@frozen
class LateralParams:
    """Throughflow parameters (:class:`terrasim.params.Params`\ ``.lateral``).

    Attributes:
        anisotropy: Lateral / vertical saturated conductivity ratio ``a``; 0 disables.
            Field throughflow typically needs 10-100.
        min_slope: Floor on ``tan(beta)`` so flat valley floors still drain a little.
    """

    anisotropy: float = 0.0
    min_slope: float = 0.002

    def __attrs_post_init__(s):
        if s.anisotropy < 0:
            raise ValueError('anisotropy must be >= 0')
        if s.min_slope < 0:
            raise ValueError('min_slope must be >= 0')

    @property
    def active(s) -> bool:
        return s.anisotropy > 0


def flow(theta: np.ndarray, static, tan_slope: np.ndarray, dt: float, cell_mm: float, p: LateralParams,
         ds_index: np.ndarray, order: np.ndarray = None, level_ptr: np.ndarray = None):
    """One explicit step of throughflow on the D8 network. Mutates ``theta`` (L, n) in place
    and returns ``(lat_in, lat_out, export, exfiltration)``, each ``(n,)`` mm: water received
    from upslope (all layers), water passed downslope (all layers), the part of ``lat_out``
    that left the domain, and water pushed out at the surface by a full profile.
    ``order``/``level_ptr`` are accepted for interface compatibility and unused."""
    L, n = theta.shape
    dz = static.dz_mm; dzc = dz[:, None]
    theta_s = static.theta_s.reshape(L, n); theta_dul = static.theta_dul.reshape(L, n)
    ksat = static.ksat.reshape(L, n); b = static.b.reshape(L, n)
    slope = np.maximum(tan_slope, p.min_slope)
    # 1. send: Darcy throughflow per layer from the start-of-step state, limited to drainable water
    kk = campbell.k(theta, theta_s, ksat, b)                                                   # (L, n)
    q = kk * (slope * (p.anisotropy * dt / cell_mm)) * dzc
    q = np.minimum(q, np.clip((theta - theta_dul) * dzc, 0.0, None))
    theta -= q / dzc
    lat_out = q.sum(axis=0)
    dst = ds_index
    inside = dst >= 0
    export = np.where(inside, 0.0, lat_out)
    inflow = np.zeros((L, n))
    if inside.any():
        di = dst[inside]
        for i in range(L):
            inflow[i] = np.bincount(di, weights=q[i, inside], minlength=n)
    lat_in = inflow.sum(axis=0)
    # 2. receive: fill each layer bottom-up with what arrived in it; surplus rises; the rest exfiltrates
    rising = np.zeros(n)
    for i in range(L - 1, -1, -1):
        a = inflow[i] + rising
        room = np.clip((theta_s[i] - theta[i]) * dz[i], 0.0, None)
        take = np.minimum(a, room)
        theta[i] += take / dz[i]
        rising = a - take
    return lat_in, lat_out, export, rising


# -- tests -----------------------------------------------------------------------

def _plane(ny=6, nx=1, ksat=20.0, dz_top=None):
    from terrasim import terrain
    from terrasim.layers import LayerSpec
    from terrasim.physics.state import StaticArrays
    dem = (np.arange(ny)[::-1])[:, None] * 1.0 + np.zeros((1, nx))        # row 0 highest, flows down the rows
    fdir = terrain.d8_flowdir_simple(dem, 10.0)
    ds, _ = terrain.downstream_index(fdir)
    order, ptr = terrain.topological_levels(ds)
    st = StaticArrays.uniform(LayerSpec(), ny, nx, ksat=ksat, ds_index=ds, level_order=order, level_ptr=ptr)
    return st, ds, order, ptr


def test_off_is_identity_and_conserves_when_on():
    st, ds, order, ptr = _plane()
    L, n = st.nlayer, st.theta_s[0].size
    rng = np.random.default_rng(0)
    theta = (st.theta_dul + rng.uniform(0.3, 1.0, st.theta_s.shape) * (st.theta_s - st.theta_dul)).reshape(L, n)
    slope = np.full(n, 0.1)
    th0 = theta.copy()
    lat_in, lat_out, export, exfil = flow(theta.copy(), st, slope, 1.0, 10_000.0, LateralParams(anisotropy=0.0),
                                          ds, order, ptr)
    off = lat_out.sum() == 0 and export.sum() == 0
    th = th0.copy()
    lat_in, lat_out, export, exfil = flow(th, st, slope, 1.0, 10_000.0, LateralParams(anisotropy=50.0), ds, order, ptr)
    stored0 = np.tensordot(st.dz_mm, th0, axes=(0, 0)).sum(); stored1 = np.tensordot(st.dz_mm, th, axes=(0, 0)).sum()
    conserved = abs(stored0 - stored1 - export.sum() - exfil.sum()) < 1e-9
    per_pixel = np.allclose(np.tensordot(st.dz_mm, th - th0, axes=(0, 0)), lat_in - lat_out - exfil, atol=1e-9)
    bounded = (th <= st.theta_s.reshape(L, n) + 1e-12).all() and (th >= 0).all()
    moved = lat_out.sum() > 0 and lat_in[1:].sum() > 0
    return off and conserved and per_pixel and bounded and moved


def test_valley_fills_and_exfiltrates():
    """On a 6-pixel slope wet above field capacity everywhere except the bottom pixel,
    throughflow wets the bottom pixel from above; a saturated pixel that receives inflow
    returns it to the surface (exfiltration); the outlet exports."""
    st, ds, order, ptr = _plane(ny=6, ksat=50.0)
    L, n = st.nlayer, st.theta_s[0].size
    theta = (0.5 * (st.theta_s + st.theta_dul)).reshape(L, n).copy()          # half-way to saturation on the slope
    theta[:, -1] = st.theta_dul.reshape(L, n)[:, -1]                           # dry bottom pixel
    p = LateralParams(anisotropy=100.0)
    slope = np.array([0.2] * (n - 1) + [0.002])                                # steep slope, flat valley floor
    total_in_bottom = 0.0; export_total = 0.0
    for _ in range(200):
        lat_in, lat_out, export, exfil = flow(theta, st, slope, 1.0, 10_000.0, p, ds, order, ptr)
        total_in_bottom += lat_in[-1]; export_total += export.sum()
    bottom_wetter = theta[:, -1].mean() > st.theta_dul.reshape(L, n)[:, -1].mean() + 0.02
    # a saturated flat receiver cannot take water and barely sends any: it must come out at the surface,
    # while saturated pixels on the slope pass water on and need not exfiltrate
    full = st.theta_s.reshape(L, n).copy(); full[:, 0] = 0.5 * (st.theta_s + st.theta_dul).reshape(L, n)[:, 0]
    _, _, _, exfil = flow(full, st, slope, 1.0, 10_000.0, p, ds, order, ptr)
    returns_flow = exfil[-1] > 0 and exfil[0] == 0
    return bottom_wetter and total_in_bottom > 0 and export_total > 0 and returns_flow


def test_bad_params_raise():
    for kw in ({'anisotropy': -1.0}, {'min_slope': -0.1}):
        try:
            LateralParams(**kw)
            return False
        except ValueError:
            pass
    return LateralParams().active is False and LateralParams(anisotropy=5).active


def test():
    return all([test_off_is_identity_and_conserves_when_on(), test_valley_fills_and_exfiltrates(), test_bad_params_raise()])


if __name__ == '__main__':
    print(test())
