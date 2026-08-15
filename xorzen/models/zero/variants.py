"""
This module defines the foundational `zeroBase` class, which serves as a common
interface for all `zeroModel` variants within the `xorzen` framework. It provides
convenience methods and standardized initialization logic for model instances.

Additionally, this module instantiates and registers concrete `zeroModel` variants
(e.g., `zero_1M`, `zero_277M`, `zero_7B`) across various parameter scales. Each
variant is pre-configured with specific `ModelConfig` settings, allowing for
easy instantiation and management of different model sizes. Legacy aliases are
also included for backward compatibility.
"""

from typing import Optional
from pathlib import Path

from xorzen.config import ConfigFactory, ModelConfig, ModelSize
from .model import zeroModel
from xorzen.models.registry import ModelRegistry


# === BASE zero CLASS ===

class zeroBase(zeroModel):
    """
    A foundational abstract base class for specific `zeroModel` variants.
    It extends `zeroModel` by providing standardized initialization logic,
    parameter counting, and methods for loading pretrained weights, ensuring
    consistency across all derived model sizes. This class is designed to
    be subclassed by concrete model implementations like `zero_1M` or `zero_277M`.
    
    Attributes:
        MODEL_SIZE (`ModelSize`): A class attribute indicating the predefined
                                  size of the model variant (e.g., `ModelSize.NANO_1M`).
                                  Must be set by subclasses.
        PARAM_COUNT (`int`): A class attribute storing the total parameter
                             count for this specific model variant. Must be set
                             by subclasses.
    """    
    MODEL_SIZE: ModelSize # Placeholder, to be set by variants
    PARAM_COUNT: int # Placeholder, to be set by variants

    def __init__(self, config: Optional[ModelConfig] = None, test_mode: bool = False, **overrides):
        """
        Initializes a `zeroBase` model instance. This constructor dynamically
        retrieves a `ModelConfig` based on the `MODEL_SIZE` attribute of the
        subclass, allowing for parameter overrides via `config` or `overrides`.
        It then passes this configuration to the parent `zeroModel` for construction.
        
        Args:
            config (`ModelConfig`, *optional*): An optional custom `ModelConfig`
                                                instance. If provided, it will be
                                                used instead of generating a default
                                                based on `MODEL_SIZE`.
            test_mode (`bool`, *optional*): If `True`, the underlying `zeroModel`
                                            will be initialized in a simplified
                                            test mode, typically affecting expert
                                            fabric behavior for development or testing.
                                            Defaults to `False`.
            **overrides: Arbitrary keyword arguments to directly override any
                         parameters within the resolved `ModelConfig`. These
                         overrides are applied after the base configuration for
                         the `MODEL_SIZE` has been established.
        """
        if config is None:
            config = ConfigFactory.get_config(
                self.MODEL_SIZE,
                **overrides
            )
        elif overrides:
            config.update(**overrides)
        
        super().__init__(config, test_mode=test_mode)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """
        Loads a pretrained model instance from a specified checkpoint path.
        This class method instantiates the model and then restores its state
        dictionary from the checkpoint file.
        
        Args:
            path (`str`): The file system path to the model checkpoint.
            **kwargs: Additional keyword arguments to be passed to the model's
                      constructor during instantiation (e.g., specific `ModelConfig`
                      overrides that are not part of the checkpoint).
                      
        Returns:
            An instance of the model class (`zeroBase` or its subclass) with
            its weights and configuration loaded from the checkpoint.
        """
        model = cls(**kwargs)
        model.load_checkpoint(path)
        return model
    
    @classmethod
    def param_count(cls) -> int:
        """
        Retrieves the total number of parameters for this specific model variant,
        as defined by the `PARAM_COUNT` class attribute. This provides a static
        estimate of the model's size without requiring instantiation.
        
        Returns:
            An integer representing the total parameter count for the variant.
        """
        return cls.PARAM_COUNT
    
    def __repr__(self) -> str:
        """
        Provides a string representation of the model variant,
        including its class name and parameter count.
        
        Returns:
            A string formatted as "ClassName(params=PARAM_COUNT)".
        """
        return f"{self.__class__.__name__}(params={self.PARAM_COUNT:,})"


# === CONCRETE MODEL VARIANTS ===

# zero_tiny_23k
class zero_tiny_23k(zeroBase):
    """
    Represents the `zero` model variant with approximately 37 thousand parameters.
    This is the functional nano-testbed of XORZENX, used for micro-testing.
    It leverages the `ModelSize.TINY_23K` configuration.
    """
    MODEL_SIZE = ModelSize.TINY_23K
    PARAM_COUNT = 37_824


# zero_1M
class zero_1M(zeroBase):
    """
    Represents the `zero` model variant with approximately 1 million parameters.
    This is a highly compact model, suitable for quick experimentation,
    resource-constrained environments, or as a baseline for larger architectures.
    It leverages the `ModelSize.NANO_1M` configuration.
    """
    MODEL_SIZE = ModelSize.NANO_1M
    PARAM_COUNT = 1_077_503


# zero_10M
class zero_10M(zeroBase):
    """
    Represents the `zero` model variant with approximately 10 million parameters.
    This offers a balance between computational efficiency and increased capacity
    compared to the 1M variant, making it suitable for broader experimental tasks.
    It leverages the `ModelSize.NANO_10M` configuration.
    """
    MODEL_SIZE = ModelSize.NANO_10M
    PARAM_COUNT = 10_970_548


# zero_50M
class zero_50M(zeroBase):
    """
    Represents the `zero` model variant with approximately 50 million parameters.
    This micro-sized model provides a good balance for prototyping and
    developing features where moderate model complexity is required.
    It leverages the `ModelSize.MICRO_50M` configuration.
    """
    MODEL_SIZE = ModelSize.MICRO_50M
    PARAM_COUNT = 50_399_371


# zero_277M
class zero_277M(zeroBase):
    """
    Represents the `zero` model variant with approximately 277 million parameters.
    This is often considered the flagship model in the `zero` family, offering
    a strong balance between performance and computational demands for many
    real-world applications. It leverages the `ModelSize.MINI_277M` configuration.
    """
    MODEL_SIZE = ModelSize.MINI_277M
    PARAM_COUNT = 277_000_335


# zero_500M
class zero_500M(zeroBase):
    """
    Represents the `zero` model variant with approximately 500 million parameters.
    This small-sized model offers increased capacity for more complex tasks
    while maintaining a relatively efficient operational footprint.
    It leverages the `ModelSize.SMALL_500M` configuration.
    """
    MODEL_SIZE = ModelSize.SMALL_500M
    PARAM_COUNT = 500_000_083


# zero_1_3B
class zero_1_3B(zeroBase):
    """
    Represents the `zero` model variant with approximately 1 billion parameters.
    This medium-sized model offers substantial capabilities for demanding
    AI tasks, balancing high performance with manageable computational requirements.
    It leverages the `ModelSize.MEDIUM_1B` configuration.
    """
    MODEL_SIZE = ModelSize.MEDIUM_1B
    PARAM_COUNT = 1_000_000_886


# zero_7B
class zero_7B(zeroBase):
    """
    Represents the `zero` model variant with approximately 7 billion parameters.
    This large-scale model is designed for highly complex AI challenges,
    offering advanced capabilities for state-of-the-art performance in
    resource-intensive applications. It leverages the `ModelSize.XL_7B` configuration.
    """
    MODEL_SIZE = ModelSize.XL_7B
    PARAM_COUNT = 7_000_000_466


# === LEGACY ALIASES (for backward compatibility) ===

class zeroModel277M(zero_277M):
    """
    A legacy alias for the `zero_277M` model variant.
    This class is maintained for backward compatibility with older
    codebases that might reference the model by this name.
    """
    pass

class zero277M(zero_277M):
    """
    Another legacy alias for the `zero_277M` model variant.
    Similar to `zeroModel277M`, this alias ensures backward
    compatibility for existing implementations.
    """
    pass


# Register all models after definition
def _register_zero_models():
    """
    Registers all concrete `zeroModel` variants with the `ModelRegistry`.
    This function ensures that each model variant is made available for
    instantiation throughout the framework, associating it with its name,
    class, parameter count, description, and configuration factory.
    """
    models_to_register = [
        ("zero_tiny_23k", zero_tiny_23k, 37_824, "zero micro-test variant with 37k parameters", lambda: ConfigFactory.get_config(ModelSize.TINY_23K)),
        ("zero_1m", zero_1M, 1_077_503, "zero model with 1 million parameters", lambda: ConfigFactory.get_config(ModelSize.NANO_1M)),
        ("zero_10m", zero_10M, 10_970_548, "zero model with 10 million parameters", lambda: ConfigFactory.get_config(ModelSize.NANO_10M)),
        ("zero_50m", zero_50M, 50_399_371, "zero model with 50 million parameters", lambda: ConfigFactory.get_config(ModelSize.MICRO_50M)),
        ("zero_277m", zero_277M, 277_000_335, "zero model with 277 million parameters (Flagship)", lambda: ConfigFactory.get_config(ModelSize.MINI_277M)),
        ("zero_500m", zero_500M, 500_000_083, "zero model with 500 million parameters", lambda: ConfigFactory.get_config(ModelSize.SMALL_500M)),
        ("zero_1.3b", zero_1_3B, 1_000_000_886, "zero model with 1 billion parameters", lambda: ConfigFactory.get_config(ModelSize.MEDIUM_1B)),
        ("zero_7b", zero_7B, 7_000_000_466, "zero model with 7 billion parameters", lambda: ConfigFactory.get_config(ModelSize.XL_7B)),
    ]
    for name, model_class, param_count, description, config_factory in models_to_register:
        ModelRegistry.register(name, model_class, param_count, description, config_factory)

_register_zero_models()
