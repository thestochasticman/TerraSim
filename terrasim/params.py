"""Physical parameters and the list of soil inputs the model consumes.

Split by concern so each can be swapped independently: :class:`SoilParams`
(constants and calibration knobs for the soil column), :class:`VegParams`
(how NDVI becomes crop coefficients and roots), and :class:`Params` (the
pair, plus which SLGA layers to fetch). Fields that are not yet used by
the physics (organic carbon, coarse fragments, macroporosity) are declared
now so the schema — and the static cache key — is stable as they come in.
"""
from attrs import frozen

SLGA_DEPTHS = ('0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm')


@frozen
class SoilParams:
    """Soil-column constants and calibration multipliers.

    Attributes:
        rho_particle: Particle density (kg m-3) for porosity = 1 - BD/rho.
        pond_max_mm: Surface detention storage before water runs off (mm).
        bedrock_leak_frac: Fraction of K_sat allowed as deep drainage where the
            bottom layer lies below SLGA Depth_to_Rock.
        ze_m: Depth of the soil-evaporation layer (FAO-56 Ze, m).
        rew_mm: Readily evaporable water (mm); ``None`` derives it from texture.
        ksat_multiplier, b_multiplier: Calibration scalars applied to the
            pedotransfer K_sat and Campbell b.
        theta_s_min, theta_s_max: Clip range for porosity (m3 m-3).
        ksat_min, ksat_max: Clip range for K_sat (mm h-1).
        b_min, b_max: Clip range for Campbell b.
        organic_matter_pct: Organic matter (% by weight) for the Saxton-Rawls
            retention fallback when SLGA DUL/L15 are unavailable.
        organic_carbon_pct, coarse_frag_frac, macroporosity: Reserved for future
            pedotransfer refinements; unused by the physics today.
    """

    rho_particle: float = 2650.0
    pond_max_mm: float = 2.0
    bedrock_leak_frac: float = 0.05
    ze_m: float = 0.10
    rew_mm: float | None = None
    ksat_multiplier: float = 1.0
    b_multiplier: float = 1.0
    theta_s_min: float = 0.30
    theta_s_max: float = 0.70
    ksat_min: float = 0.1
    ksat_max: float = 500.0
    b_min: float = 2.0
    b_max: float = 20.0
    organic_matter_pct: float = 1.5
    organic_carbon_pct: float | None = None
    coarse_frag_frac: float | None = None
    macroporosity: float | None = None


@frozen
class VegParams:
    """NDVI -> FAO-56 crop coefficients, cover fraction and roots.

    Attributes:
        kcb_min, kcb_max: Basal crop coefficient at bare soil / full canopy.
        ndvi_soil, ndvi_veg: NDVI of bare soil / full canopy for the linear map.
        kc_max: Upper limit of Kcb + Ke (FAO-56 eq. 72, ~1.2 for short crops).
        p_depletion: Fraction of available water depletable before stress (FAO-56 p).
        root_depth_m: Maximum rooting depth (m).
        root_beta: Exponential root-density decay rate (per m).
        canopy_store_mm_per_fc: Interception capacity per unit cover fraction (mm).
        h_crop_m: Crop height used in the FAO-56 cover-fraction exponent.
    """

    kcb_min: float = 0.15
    kcb_max: float = 1.10
    ndvi_soil: float = 0.15
    ndvi_veg: float = 0.85
    kc_max: float = 1.20
    p_depletion: float = 0.5
    root_depth_m: float = 1.0
    root_beta: float = 3.0
    canopy_store_mm_per_fc: float = 1.0
    h_crop_m: float = 0.3


@frozen
class Params:
    """All physical parameters plus the SLGA inputs to fetch.

    ``slga_attributes`` are required (texture + bulk density); ``slga_optional``
    are used when the TERN listing has them (DUL/L15 refine the retention
    curve, depth layers restrict deep drainage) and skipped with a warning
    otherwise. ``pedotransfer_version`` is folded into the static-dataset cache key so
    a change in the derivations invalidates cached soils automatically.
    """

    soil: SoilParams = SoilParams()
    veg: VegParams = VegParams()
    slga_attributes: tuple[str, ...] = ('Clay', 'Sand', 'Silt', 'Bulk_Density')
    slga_optional: tuple[str, ...] = ('Drained_Upper_Limit', 'L15', 'Depth_of_Soil', 'Depth_to_Rock')
    slga_depths: tuple[str, ...] = SLGA_DEPTHS
    pedotransfer_version: str = 'v2'


defaultparams = Params()


def test_defaults_construct():
    p = Params()
    return p.soil.rho_particle == 2650.0 and len(p.slga_depths) == 6 and p.veg.kc_max == 1.2


def test():
    return test_defaults_construct()


if __name__ == '__main__':
    print(test())
