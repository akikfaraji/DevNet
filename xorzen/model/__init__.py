"""
xorzen Model Package
"""

from .base import BaseModel
from xorzen.config import ModelConfig

def create_model(config: ModelConfig) -> BaseModel:
    """
    Factory function to create a model based on the configuration.
    """
    if config.architecture.value == 'xorzen_zero':
        from xorzen.models.zero import zeroModel
        return zeroModel(config)
    # Add other models here
    # elif config.architecture == 'berudra':
    #     return BerudraModel(config)
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")

__all__ = ['BaseModel', 'create_model']

