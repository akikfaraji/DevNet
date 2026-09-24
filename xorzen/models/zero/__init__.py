"""
zero Model Package
Provides the core zero architecture and all variants, including agentic sub-variants.

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

    # Method 4: Agentic variants (recursive inference with self-critique)
    >>> from xorzen.models.zero import zero_agentic_nano
    >>> model = zero_agentic_nano()
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

# Agentic sub-variants (recursive inference, self-critique, persistent memory)
from .agentic_config import ZeroAgenticConfig, ZeroAgenticSize
from .agentic_model import ZeroAgenticModel
from .agentic_variants import zero_agentic_nano, zero_agentic_micro

__all__ = [
    # Core
    'zeroModel',
    'zeroBase',

    # Variants (by size)
    'zero_tiny_23k',
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

    # Agentic sub-variants
    'ZeroAgenticConfig',
    'ZeroAgenticSize',
    'ZeroAgenticModel',
    'zero_agentic_nano',
    'zero_agentic_micro',
]
