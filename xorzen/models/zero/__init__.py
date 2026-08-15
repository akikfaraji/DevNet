"""
zero Model Package
Provides the core zero architecture and all variants.

Usage:
    # Method 1: Direct variant import (RECOMMENDED)
    >>> from xorzen.models.zero import zero_277M
    >>> model = zero_277M()
    
    # Method 2: Base model with config
    >>> from xorzen.models.zero import zeroModel
    >>> from xorzen.config import ConfigFactory, ModelSize
    >>> config = ConfigFactory.get_config(ModelSize.MINI_277M)
    >>> model = zeroModel(config)
    
    # Method 3: Legacy compatibility
    >>> from xorzen.models.zero import zero277M  # Alias for zero_277M
    >>> model = zero277M()
"""

# Core architecture
from .model import zeroModel

# All variants (explicit parameter counts)
from .variants import (
    zeroBase,
    zero_tiny_23k,
    zero_1M,
    zero_10M,
    zero_50M,
    zero_277M,
    zero_500M,
    zero_1_3B,
    zero_7B,
    # Legacy aliases
    zeroModel277M,
    zero277M,
)

__all__ = [
    # Core
    'zeroModel',
    'zeroBase',
    
    # Variants (by size)
    'zero_1M',
    'zero_10M',
    'zero_50M',
    'zero_277M',
    'zero_500M',
    'zero_1_3B',
    'zero_7B',
    
    # Legacy
    'zeroModel277M',
    'zero277M',
]
