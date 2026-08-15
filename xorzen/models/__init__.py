
"""
xorzen Models Package
Central hub for all model architectures and variants.

Usage:
    # List available models
    >>> from xorzen.models import list_models
    >>> print(list_models())
    ['zero_1m', 'zero_10m', 'zero_50m', 'zero_277m', ...]
    
    # Get model by name
    >>> from xorzen.models import get_model
    >>> ModelClass = get_model('zero_277m')
    >>> model = ModelClass()
    
    # Direct import (recommended)
    >>> from xorzen.models.zero import zero_277M
    >>> model = zero_277M()
"""

# Registry system
from .registry import (
    ModelRegistry,
    list_models,
    get_model,
    create_model,
)

# zero models
from .zero import (
    zeroModel,
    zeroBase,
    zero_1M,
    zero_10M,
    zero_50M,
    zero_277M,
    zero_500M,
    zero_1_3B,
    zero_7B,
    zero277M,  # Legacy alias
)

# IGRIS models (Advanced Recursive)
from .igris.model import IGRISModel
from .igris.config import IGRISConfig
from .igris.variants import IGRIS_Nano, IGRIS_Micro

__all__ = [
    # Registry
    'ModelRegistry',
    'list_models',
    'get_model',
    'create_model',
    
    # zero
    'zeroModel',
    'zeroBase',
    'zero_1M',
    'zero_10M',
    'zero_50M',
    'zero_277M',
    'zero_500M',
    'zero_1_3B',
    'zero_7B',
    'zero277M',
    
    # IGRIS
    'IGRISModel',
    'IGRISConfig',
    'IGRIS_Nano',
    'IGRIS_Micro',
]
