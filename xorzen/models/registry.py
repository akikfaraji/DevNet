"""
This module provides a robust `ModelRegistry` system, designed to centralize
the registration, discovery, and management of various model variants within
the `xorzen` framework. It enables developers to easily list, retrieve, and
instantiate models by their registered names or by specific criteria like
parameter count.

The registry ensures a consistent interface for interacting with different
model architectures and sizes, facilitating modularity and extensibility
of the framework.
"""

from typing import Dict, Type, Optional, List, Callable
from pathlib import Path
import warnings

from ..exceptions import ModelNotFoundError, ModelError


class ModelRegistry:
    """
    Serves as the central repository for all available model variants within
    the `xorzen` framework. It facilitates the organized registration, discovery,
    and retrieval of model classes based on their unique names.

    The registry supports:
    - **Dynamic Discovery**: List all registered models.
    - **Retrieval by Name**: Obtain a specific model class by its identifier.
    - **Information Access**: Query detailed metadata for any registered model.
    - **Instantiation**: Create model instances with optional configuration overrides.

    Example Usage:
        ```python
        from xorzen.models import ModelRegistry, zero_277M
        
        # List all currently registered model names
        print(ModelRegistry.list())
        # Expected output: ['zero_1m', 'zero_10m', 'zero_50m', 'zero_277m', ...]
        
        # Retrieve a model class by its name and instantiate it
        model_class = ModelRegistry.get('zero_277m')
        model_instance = model_class()
        
        # Alternatively, instantiate directly if the class is imported
        model_instance_direct = zero_277M()
        ```
    """
    
    _models: Dict[str, Dict] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type,
        param_count: int,
        description: str = "",
        config_factory: Optional[Callable] = None
    ):
        """
        Registers a new model variant with the central registry.
        This makes the model discoverable and instantiable via its `name`.
        If a model with the same `name` is already registered, a warning
        will be issued, and the existing entry will be overwritten.
        
        Args:
            name (`str`): A unique identifier for the model variant (e.g., "zero_277m").
            model_class (`Type`): The Python class of the model (e.g., `zero_277M`).
            param_count (`int`): The total number of parameters in this model variant.
            description (`str`, *optional*): A brief textual description of the model.
                                             Defaults to an empty string.
            config_factory (`Callable`, *optional*): A callable (function or lambda)
                                                     that returns a `ModelConfig`
                                                     instance for this model variant.
                                                     Used for dynamic configuration creation.
                                                     Defaults to `None`.
        """        
        if name in cls._models:
            warnings.warn(f"Model '{name}' already registered. Overwriting.")
        
        cls._models[name] = {
            'class': model_class,
            'param_count': param_count,
            'description': description,
            'config_factory': config_factory
        }
    
    @classmethod
    def get(cls, name: str) -> Type:
        """
        Retrieves the model class associated with the given name from the registry.
        The name matching is case-insensitive.
        
        Args:
            name (`str`): The unique identifier of the model variant to retrieve.
            
        Returns:
            `Type`: The Python class of the requested model.
            
        Raises:
            ModelNotFoundError: If no model with the specified name is found
                                 in the registry.
        """
        name = name.lower()
        if name not in cls._models:
            raise ModelNotFoundError(name, available_models=cls.list())
        
        return cls._models[name]['class']
    
    @classmethod
    def get_info(cls, name: str) -> Dict:
        """
        Retrieves a dictionary containing detailed information about a registered
        model variant, including its class, parameter count, description, and
        configuration factory. The name matching is case-insensitive.
        
        Args:
            name (`str`): The unique identifier of the model variant to retrieve info for.
            
        Returns:
            `Dict`: A copy of the dictionary containing all registered information
                    for the specified model.
            
        Raises:
            ModelNotFoundError: If no model with the specified name is found
                                 in the registry.
        """        
        name = name.lower()
        if name not in cls._models:
            raise ModelNotFoundError(name, available_models=cls.list())
        
        return cls._models[name].copy()
    
    @classmethod
    def list(cls) -> List[str]:
        """
        Returns a sorted list of the names of all model variants currently
        registered in the system.
        
        Returns:
            `List[str]`: A sorted list of model names.
        """        
        return sorted(cls._models.keys())
    
    @classmethod
    def list_detailed(cls) -> Dict[str, Dict]:
        """
        Retrieves a dictionary containing detailed information for all
        registered model variants, keyed by their names. Each entry
        includes the model's parameter count and description.
        
        Returns:
            `Dict[str, Dict]`: A dictionary where keys are model names and values
                               are dictionaries containing 'param_count' and 'description'.
        """        
        return {
            name: {
                'param_count': info['param_count'],
                'description': info['description']
            }
            for name, info in sorted(cls._models.items())
        }
    
    @classmethod
    def find_by_params(
        cls,
        target_params: int,
        tolerance: float = 0.1,
        architecture: Optional[str] = None
    ) -> Optional[str]:
        """
        Searches the registry for the model variant whose parameter count
        is closest to `target_params`, within an optional `tolerance`.
        The search can also be filtered by a specific architectural prefix.
        
        Args:
            target_params (`int`): The desired number of parameters for the model.
            tolerance (`float`, *optional*): The maximum acceptable relative
                                             difference (as a fraction) between
                                             the `target_params` and a candidate
                                             model's parameter count. Defaults to 0.1 (10%).
            architecture (`str`, *optional*): An optional architectural prefix to
                                              filter candidate models (e.g., "zero"
                                              to only consider `zeroModel` variants).
        
        Returns:
            `Optional[str]`: The name of the closest matching model if one is found
                             within the `tolerance`, otherwise `None`.
        """
        if not cls._models:
            return None
        
        candidates = cls._models.items()
        
        # Filter by architecture if specified
        if architecture:
            candidates = [
                (name, info) for name, info in candidates
                if name.startswith(architecture.lower())
            ]
        
        if not candidates:
            return None
        
        # Find closest match
        closest = min(
            candidates,
            key=lambda x: abs(x[1]['param_count'] - target_params)
        )
        
        name, info = closest
        relative_diff = abs(info['param_count'] - target_params) / target_params
        
        if relative_diff <= tolerance:
            return name
        
        return None
    
    @classmethod
    def create(cls, name: str, **kwargs):
        """
        Creates and returns an instantiated model object based on its registered name.
        The model is initialized using its default configuration (as defined by
        its `config_factory`), which can then be overridden by parameters provided
        in `kwargs`.
        
        Args:
            name (`str`): The registered name of the model variant to instantiate.
            **kwargs: Arbitrary keyword arguments that will be passed as overrides
                      to the model's constructor or its underlying `ModelConfig`.
        
        Returns:
            An instantiated object of the requested model class.
            
        Raises:
            ModelNotFoundError: If no model with the specified name is found
                                 in the registry.
            ModelError: For any issues during model instantiation or configuration.
        """        
        info = cls.get_info(name)
        model_class = info['class']
        
        # Create config if factory available
        if info['config_factory'] and not kwargs.get('config'):
            config = info['config_factory']()
            
            # Apply overrides
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            return model_class(config)
        
        return model_class(**kwargs)
    
    @classmethod
    def clear(cls):
        """
        Removes all currently registered model variants from the registry.
        This method is primarily intended for use in testing scenarios
        to ensure a clean state before running new tests.
        """        
        cls._models.clear()


# Convenience functions
def list_models() -> List[str]:
    """
    Convenience function to list the names of all models
    currently registered in the `ModelRegistry`.
    
    Returns:
        `List[str]`: A sorted list of registered model names.
    """    
    return ModelRegistry.list()


def get_model(name: str) -> Type:
    """
    Convenience function to retrieve a specific model class by its
    registered name from the `ModelRegistry`.
    
    Args:
        name (`str`): The unique identifier of the model variant.
        
    Returns:
        `Type`: The Python class of the requested model.
        
    Raises:
            ModelNotFoundError: If the model with the specified name is not found.
    """    
    return ModelRegistry.get(name)


def create_model(name: str, **kwargs):
    """
    Convenience function to create an instantiated model object based on
    its registered name from the `ModelRegistry`. Allows for optional
    configuration overrides.
    
    Args:
        name (`str`): The registered name of the model variant to instantiate.
        **kwargs: Arbitrary keyword arguments to override parameters within
                  the model's configuration during instantiation.
                  
    Returns:
        An instantiated object of the requested model class.
        
    Raises:
        ModelNotFoundError: If the model with the specified name is not found.
        ModelError: For any issues during model instantiation or configuration.
    """    
    return ModelRegistry.create(name, **kwargs)


__all__ = [
    'ModelRegistry',
    'list_models',
    'get_model',
    'create_model',
]
