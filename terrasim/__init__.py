"""terrasim — process-based 10 m soil moisture for Australian paddocks.

Only lightweight classes are exported here so ``import terrasim`` stays
fast; heavy modules (stores, physics, model) are imported explicitly:

    from terrasim import Terrasim, Params, LayerSpec
    from terrasim.model import Model
"""
from terrasim.terrasim import Terrasim, defaultterrasim
from terrasim.params import Params, SoilParams, VegParams, defaultparams
from terrasim.layers import LayerSpec, DAESIM_DEFAULT

__all__ = ['Terrasim', 'defaultterrasim', 'Params', 'SoilParams', 'VegParams',
           'defaultparams', 'LayerSpec', 'DAESIM_DEFAULT']
