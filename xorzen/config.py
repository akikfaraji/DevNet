"""
Provides a robust and comprehensive configuration system for the xorzen-zero framework.
This module defines the structure and behavior for managing model, training, data,
and quantization parameters, supporting a range of model sizes from 1 million (1M)
to 1 billion (1B) parameters.

Key Features:
- **Production-Grade Design**: Engineered for reliability and scalability in production environments.
- **Validation**: Ensures configuration integrity and consistency through strict data checks.
- **Serialization**: Supports seamless saving and loading of configurations to/from various formats (JSON, YAML, Pickle).
- **Optimization**: Includes mechanisms for automatic configuration adjustments based on target hardware and performance goals.
"""

import json
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Type, TypeVar
from dataclasses import dataclass, field, asdict, fields, MISSING
from enum import Enum
import warnings
import copy
import math
from functools import lru_cache

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, some features disabled")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ==================== ENUMS ====================

class ModelSize(Enum):
    """
    Defines a set of predefined model sizes, ranging from very small (NANO_1M)
    to very large (XXXL_70B). Each member represents a specific scale
    of the model, influencing parameters such as hidden dimensions,
    number of layers, and attention heads.
    """
    TINY_23K = "23K"
    NANO_1M = "1M"
    NANO_10M = "10M"
    MICRO_50M = "50M"
    MINI_277M = "277M"
    SMALL_500M = "500M"
    MEDIUM_1B = "1B"
    LARGE_3B = "3B"
    XL_7B = "7B"
    XXL_13B = "13B"
    XXXL_70B = "70B"


class ArchitectureVariant(Enum):
    """
    Specifies the distinct architectural configurations available for models within the framework.
    Each variant represents a fundamental design choice impacting model behavior and performance,
    ranging from standard transformer designs to specialized, innovative architectures like XORZENX_zero.
    """
    STANDARD = "standard"  # Baseline transformer
    HASS = "hass"  # Hybrid Attention-Shard Switch
    MOE = "moe"  # Mixture of Experts
    ADAPTIVE = "adaptive"  # Adaptive compute
    QUANTUM = "quantum"  # Quantum-inspired
    XORZENX_zero = "xorzen_zero"  # Our revolutionary architecture


class RouterType(Enum):
    """
    Defines the available routing mechanisms used within the model, particularly in
    architectures like Mixture-of-Experts (MoE) or adaptive computation.
    These types determine how information is directed or processed through different
    parts of the model, influencing computational efficiency and decision-making.
    """
    NONE = "none"
    DEPTH_ONLY = "depth_only"
    WIDTH_ONLY = "width_only"
    DEPTH_WIDTH = "depth_width"
    ADAPTIVE = "adaptive"
    CERTAINTY = "certainty"


class CoTType(Enum):
    """
    Specifies the variant of Chain-of-Thought (CoT) reasoning employed by the model.
    These types determine how the model generates or leverages intermediate reasoning
    steps to arrive at a final output, impacting interpretability and complex problem-solving capabilities.
    """
    NONE = "none"
    OUTPUT = "output"  # Output CoT steps
    LATENT = "latent"  # Internal latent CoT
    PROVABLE = "provable"  # Provable reasoning
    QUANTUM = "quantum"  # Quantum CoT


class QuantizationScheme(Enum):
    """
    Defines various strategies for model quantization, a process that reduces
    the precision of model weights and activations to decrease memory footprint
    and accelerate inference. Different schemes offer trade-offs between
    model size, speed, and accuracy.
    """
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"
    PROGRESSIVE = "progressive"
    AWARE = "aware"  # Quantization-aware training


# ==================== BASE CONFIGURATION ====================

@dataclass
class BaseConfig:
    """
    Serves as the foundational class for all configuration objects within the xorzen-zero framework.
    It provides core functionalities such as serialization (to enable saving and loading
    configuration states) and a standardized validation interface to ensure data integrity
    across all derived configuration subclasses.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the configuration object into a dictionary representation.
        This method recursively processes dataclass fields, converting Enum members to their
        string values and nested `BaseConfig` instances into their respective dictionaries.
        
        Returns:
            A dictionary representing the configuration state.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.value
            elif isinstance(value, tuple):
                result[f.name] = list(value)
            elif isinstance(value, BaseConfig): # Handle nested BaseConfig subclasses
                result[f.name] = value.to_dict()
            elif isinstance(value, dict): # Handle dictionaries that might contain enums or nested BaseConfig
                processed_dict = {}
                for k, v in value.items():
                    if isinstance(v, Enum):
                        processed_dict[k] = v.value
                    elif isinstance(v, BaseConfig):
                        processed_dict[k] = v.to_dict()
                    elif isinstance(v, tuple):
                        processed_dict[k] = list(v)
                    else:
                        processed_dict[k] = v
                result[f.name] = processed_dict
            else:
                result[f.name] = value
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serializes the configuration object into a JSON formatted string.
        Leverages the `to_dict` method to first convert the configuration
        into a dictionary, then serializes it to JSON.
        
        Args:
            indent: The indentation level for pretty-printing the JSON string.
            
        Returns:
            A JSON string representation of the configuration.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """
        Serializes the configuration object into a YAML formatted string.
        Utilizes the `to_dict` method to convert the configuration into a
        dictionary before dumping it to YAML format.
        
        Returns:
            A YAML string representation of the configuration.
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save(self, path: Union[str, Path]):
        """
        Saves the current configuration object to a specified file path.
        The save format (JSON, YAML, or Pickle) is automatically determined
        by the file extension provided in the `path`. Intermediate directories
        are created if they do not exist.
        
        Args:
            path: The file path (including extension) where the configuration
                  should be saved. Supported extensions are '.json', '.yaml',
                  '.yml', and '.pkl'.
        
        Raises:
            ValueError: If an unsupported file format extension is provided.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
        elif path.suffix == '.pkl':
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseConfig':
        """
        Loads a configuration object from a specified file path.
        The file format is inferred from the extension, and the data is
        deserialized into a new instance of the `BaseConfig` subclass.
        
        Args:
            path: The file path (including extension) from which to load the configuration.
                  Supported extensions are '.json', '.yaml', '.yml', and '.pkl'.
            
        Returns:
            An instance of the loaded configuration.
            
        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
            ValueError: If an unsupported file format extension is provided.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """
        Constructs a configuration object from a dictionary. This class method
        intelligently handles the instantiation of nested dataclasses and the
        correct conversion of string values back into Enum members or other
        appropriate types based on the dataclass field definitions.
        
        Args:
            data: A dictionary containing the configuration parameters.
            
        Returns:
            An instance of the configuration class populated with values from the dictionary.
            
        Raises:
            TypeError: If the input data is not a dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Input data for {cls.__name__}.from_dict must be a dictionary, got {type(data)}")

        # Get the fields defined for this specific dataclass
        field_values = {}
        for f in fields(cls):
            if f.name in data:
                value = data[f.name]
                # Handle nested dataclasses
                if hasattr(f.type, '__dataclass_fields__') and isinstance(value, dict):
                    field_values[f.name] = f.type.from_dict(value)
                # Handle Optional types containing Enums
                elif getattr(f.type, '__origin__', None) is Union:
                    is_optional_enum = False
                    for arg in f.type.__args__:
                        if isinstance(arg, type) and issubclass(arg, Enum):
                            if isinstance(value, str):
                                try:
                                    field_values[f.name] = arg(value)
                                    is_optional_enum = True
                                    break
                                except ValueError:
                                    pass
                    if not is_optional_enum: # If it's an Optional but not Optional[Enum]
                        field_values[f.name] = value
                # Handle direct Enum types
                elif isinstance(f.type, type) and issubclass(f.type, Enum):
                    if isinstance(value, str):
                        field_values[f.name] = f.type(value)
                    else:
                        field_values[f.name] = value # Already correct enum or other type
                # Handle tuples
                elif getattr(f.type, '__origin__', None) is tuple and isinstance(value, list):
                    field_values[f.name] = tuple(value)
                else:
                    field_values[f.name] = value

        return cls(**field_values)
    
    def copy(self) -> 'BaseConfig':
        """
        Creates and returns a deep copy of the current configuration object.
        This ensures that modifications to the copied instance do not affect
        the original object, providing isolation for configuration adjustments.
        
        Returns:
            A new, independent instance of the configuration object with identical values.
        """
        return copy.deepcopy(self)
    
    def update(self, **kwargs):
        """
        Updates specific attributes of the configuration object with new values.
        This method allows for partial modifications to the configuration without
        re-instantiating the entire object. Unknown keys are silently ignored,
        with a warning issued.
        
        Args:
            **kwargs: Arbitrary keyword arguments where the key corresponds to
                      an attribute name of the configuration and the value is
                      the new value for that attribute.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Ignoring unknown config key: {key}")
    
    def validate(self) -> List[str]:
        """
        Performs a basic validation check on the configuration object, primarily
        ensuring that all non-optional fields have a value. This method can be
        extended in subclasses to include more specific validation logic relevant
        to their respective parameters.
        
        Returns:
            A list of strings, where each string describes a validation error.
            The list is empty if the configuration is considered valid.
        """
        errors = []
        
        # Check required fields
        for field_name, field_info in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if value is None:
                # Check if the field is Optional
                is_optional = (
                    hasattr(field_info.type, '__origin__') and 
                    field_info.type.__origin__ is Union and 
                    type(None) in field_info.type.__args__
                )
                if not is_optional:
                    errors.append(f"Field '{field_name}' cannot be None")
        
        return errors
    
    def effective(self) -> Dict[str, Any]:
        """
        Generates a dictionary containing the complete configuration,
        including all explicitly defined parameters and any dynamically
        computed or 'derived' values. This provides a comprehensive view
        of the configuration's operational state.
        
        Returns:
            A dictionary containing all configuration values, including derived ones.
        """
        result = self.to_dict()
        
        # Add derived values
        if hasattr(self, 'compute_derived'):
            result.update(self.compute_derived())
        
        return result


# ==================== MODEL CONFIGURATIONS ====================

@dataclass
class ModelConfig(BaseConfig):
    """
    Defines the comprehensive configuration parameters for an AI model within the xorzen-zero framework.
    This includes architectural specifics, dimensions (e.g., vocab_size, hidden_size),
    attention mechanisms, and settings related to adaptive routing, Chain-of-Thought
    integration, and quantization. It encapsulates all necessary details to define
    and instantiate a particular model variant.
    """
    
    # Model identification
    model_name: str = "xorzen_model"
    model_size: ModelSize = ModelSize.MINI_277M
    architecture: ArchitectureVariant = ArchitectureVariant.XORZENX_zero
    version: str = "1.0.0"
    
    # Core dimensions
    vocab_size: int = 32000
    pad_token_id: int = 0  # token ID used for padding (auto-set per model size)
    context_length: int = 8192
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 32
    
    # Architecture specifics
    router_type: RouterType = RouterType.ADAPTIVE
    cot_type: CoTType = CoTType.LATENT
    quantization: QuantizationScheme = QuantizationScheme.PROGRESSIVE
    
    # Dropout and regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Activation functions
    hidden_act: str = "gelu"
    intermediate_act: str = "gelu"
    
    # Initialization
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    
    # Optimization
    gradient_checkpointing: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = True
    
    # Adaptive routing
    max_depth: int = 24
    min_depth: int = 3
    width_choices: Tuple[int, ...] = (384, 768, 1152, 1536, 2048)
    path_choices: int = 3  # Local, Low-rank, SSM
    # How many of the 3 HASS pathways to actually execute per token.
    # 1 = only the top-1 pathway runs (max sparsity, max savings, risk of
    #     dropped signal). 2 = top-2 pathways run. 3 = all pathways run
    # (no sparsity, equivalent to the old dense-blend behavior).
    # At training time the soft probs are still used for the backward pass
    # (straight-through estimator); at inference the hard top-k runs.
    pathway_top_k: int = 2
    # Global compute budget in [0, 1]. The router conditions its decisions
    # on this scalar so the same trained model can run at different
    # quality/compute tradeoffs. 1.0 = full compute, 0.25 = quarter compute.
    # See ``ComputeController`` in ``compute_controller.py``.
    compute_budget: float = 1.0
    # Whether to enable adaptive halting (early exit per token). When True,
    # the model estimates per-token difficulty after each block and exits
    # tokens that are "solved" up to ``halting_threshold``. Disabled by
    # default to preserve backward compatibility.
    adaptive_halting: bool = False
    halting_threshold: float = 0.9
    
    # Router configuration
    router_hidden_dim: int = 128
    router_num_layers: int = 2
    router_dropout: float = 0.1
    router_temperature: float = 1.0
    router_temperature_annealing: bool = True
    
    # Load balancing
    # ── TUNED: auxiliary losses were drowning the LM signal at small scale.
    # At NANO_1M (hidden=64, 2 experts) load-balance matters much less than
    # just learning the language. Weights scaled down 10–100×.
    load_balancing_weight: float = 0.0001   # was 0.01 — 100× reduction
    expert_capacity_factor: float = 1.25
    
    # Additional auxiliary loss weights
    routing_loss_weight: float = 0.0001    # was 0.01 — uncertainty reg barely needed at nano scale
    load_balance_loss_weight: float = 0.0001  # alias kept for consistency
    
    # CoT configuration
    cot_dim: int = 256
    cot_components: int = 6
    cot_update_method: str = "gru"  # "gru", "lstm", "linear"
    cot_provable: bool = True
    cot_consistency_weight: float = 0.1
    cot_diversity_weight: float = 0.01
    cot_sparsity_weight: float = 0.001
    cot_orthogonality_weight: float = 0.05
    
    # HASS block configuration
    local_window_size: int = 128
    low_rank_dim: int = 96
    ssm_state_dim: int = 16
    ssm_kernel_size: int = 3
    
    # MoE configuration
    expert_count: int = 192
    top_k_experts: int = 2
    expert_hidden_multiplier: float = 4.0
    shard_experts: bool = True
    expert_shard_dir: str = "experts/"
    max_expert_cache: int = 24
    
    # Merger gate
    merger_hidden_multiplier: float = 2.0
    merger_num_layers: int = 2
    merger_type: str = "gated"
    
    # Progressive quantization
    quant_start_step: int = 1000
    quant_end_step: int = 100000
    quant_levels: Tuple[int, ...] = (32, 16, 12, 8, 4)
    quant_method: str = "symmetric"
    
    # Performance
    target_active_ratio: float = 0.1
    max_active_params_per_token: int = 50_000_000
    
    def __post_init__(self):
        """
        Performs post-initialization validation for `ModelConfig` parameters.
        This method ensures that critical architectural parameters are consistent
        (e.g., `hidden_size` is divisible by `num_attention_heads`) and that
        various dimensions and counts (like `cot_dim`, `expert_count`) are valid
        and within expected ranges.
        
        Raises:
            ValueError: If any validation check fails, indicating inconsistent or invalid configuration parameters.
        """
        # Ensure hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Validate routing parameters
        if self.max_depth < self.min_depth:
            raise ValueError(f"max_depth ({self.max_depth}) must be >= min_depth ({self.min_depth})")
        
        if self.max_depth > self.num_layers:
            raise ValueError(
                f"max_depth ({self.max_depth}) must be <= num_layers ({self.num_layers})"
            )
        
        # Validate CoT dimensions
        if self.cot_dim <= 0:
            raise ValueError(f"cot_dim ({self.cot_dim}) must be positive")
        
        if self.cot_components <= 0:
            raise ValueError(f"cot_components ({self.cot_components}) must be positive")
        
        # Validate MoE parameters
        if self.expert_count <= 0:
            raise ValueError(f"expert_count ({self.expert_count}) must be positive")
        
        if self.top_k_experts <= 0 or self.top_k_experts > self.expert_count:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) must be between 1 and expert_count ({self.expert_count})"
            )
    
    def compute_derived(self) -> Dict[str, Any]:
        """
        Calculates and returns a dictionary of derived configuration values
        that are not directly set but are computed from existing parameters.
        This includes metrics like `head_dim`, estimated total parameters,
        memory usage, FLOPs per token, and architecture-specific derived values.
        
        Returns:
            A dictionary containing all computed derived configuration values.
        """
        base_derived = {
            "head_dim": self.hidden_size // self.num_attention_heads,
            "total_parameters": self.estimate_parameters(),
            "memory_bytes": self.estimate_memory(),
            "flops_per_token": self.estimate_flops_per_token(),
        }
        
        # zero-specific derived values
        if self.architecture == ArchitectureVariant.XORZENX_zero:
            zero_derived = {
                "total_cot_dim": self.cot_dim * self.cot_components,
                "expert_hidden_size": int(self.hidden_size * self.expert_hidden_multiplier),
                "merger_hidden_size": int(self.hidden_size * self.merger_hidden_multiplier),
                "router_output_dim": self.max_depth + len(self.width_choices) + self.path_choices + self.expert_count,
                "estimated_active_params": self.estimate_active_parameters(),
                "estimated_efficiency_gain": self.estimate_efficiency_gain(),
                "expert_shard_size_mb": self.estimate_expert_shard_size(),
            }
            base_derived.update(zero_derived)
        
        return base_derived
    
    def estimate_active_parameters(self) -> int:
        """
        Estimates the number of active parameters per token during inference,
        considering sparse activation patterns and various model components
        such as embeddings, attention layers, FFNs, routers, and CoT modules.
        This provides insight into the actual computational load for a given input.
        
        Returns:
            An integer representing the estimated number of active parameters.
        """
        # Base embeddings
        active = self.hidden_size  # Token embedding
        
        # Average active layers
        avg_layers = (self.max_depth + self.min_depth) / 2
        
        # Per active layer
        # Attention (simplified)
        attention_params = 4 * self.hidden_size * self.hidden_size
        
        # FFN (simplified)
        ffn_params = 2 * self.hidden_size * (4 * self.hidden_size)
        
        # Layer norm
        ln_params = 2 * self.hidden_size * 2
        
        layer_params = attention_params + ffn_params + ln_params
        
        # Average width factor
        avg_width = sum(self.width_choices) / len(self.width_choices)
        width_factor = avg_width / self.hidden_size
        
        # Active parameters
        active_params = int(
            active + avg_layers * layer_params * width_factor * self.target_active_ratio
        )
        
        # Add router and CoT
        router_params = self.router_hidden_dim * (self.hidden_size + 1)  # Simplified
        cot_params = self.cot_dim * self.cot_components * 6  # 6 components
        
        active_params += router_params + cot_params
        
        # Add active MoE experts
        expert_params_per = self.hidden_size * int(self.hidden_size * self.expert_hidden_multiplier) * 2
        active_params += self.top_k_experts * expert_params_per
        
        return active_params
    
    def estimate_efficiency_gain(self) -> float:
        """
        Calculates the estimated efficiency gain of the current model configuration
        compared to a hypothetical dense model of equivalent total parameters.
        This metric highlights the benefits of sparse activation and adaptive routing.
        
        Returns:
            A float representing the estimated efficiency gain. A higher value indicates
            greater efficiency relative to a dense model.
        """
        dense_params = self.estimate_parameters()
        active_params = self.estimate_active_parameters()
        
        if active_params == 0:
            return 1.0
        
        return dense_params / active_params
    
    def estimate_expert_shard_size(self, dtype_bytes: int = 2) -> float:
        """
        Estimates the memory size (in MB) required for a single expert shard
        within a Mixture-of-Experts (MoE) architecture. This calculation
        considers the hidden dimensions, expert multiplier, and data type size.
        
        Args:
            dtype_bytes: The number of bytes used to represent each parameter's value
                         (e.g., 2 for bfloat16 or float16, 4 for float32).
            
        Returns:
            A float representing the estimated size of an expert shard in megabytes.
        """
        # Expert parameters
        expert_hidden = int(self.hidden_size * self.expert_hidden_multiplier)
        expert_params = self.hidden_size * expert_hidden * 2  # up and down
        
        # Size in bytes
        size_bytes = expert_params * dtype_bytes
        
        # Convert to MB
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
    
    def estimate_parameters(self) -> int:
        """
        Estimates the total number of trainable parameters in the model.
        This calculation includes parameters from embeddings (token and positional),
        attention layers, feed-forward networks (FFN), and layer normalization,
        providing a theoretical count of the model's capacity.
        
        Returns:
            An integer representing the estimated total number of parameters.
        """
        # Embeddings
        vocab_params = self.vocab_size * self.hidden_size
        position_params = self.context_length * self.hidden_size
        
        # Transformer layers
        # Attention: QKV + output
        attention_params = 4 * self.hidden_size * self.hidden_size
        
        # Feed-forward: up + down
        ff_multiplier = 4  # Standard transformer
        ff_params = 2 * self.hidden_size * (ff_multiplier * self.hidden_size)
        
        # Layer norms
        ln_params = 2 * self.hidden_size * 2  # gamma and beta
        
        # Per layer
        layer_params = attention_params + ff_params + ln_params
        
        # Total
        total = vocab_params + position_params + self.num_layers * layer_params
        
        # Add output layer
        if not self.tie_word_embeddings:
            total += vocab_params
        
        return int(total)
    
    def estimate_memory(self, dtype_bytes: int = 2) -> int:
        """
        Estimates the total memory footprint of the model in bytes,
        considering parameters, optimizer states (for Adam), gradients,
        and activations. This provides a crucial metric for resource planning
        and deployment, especially for large models.
        
        Args:
            dtype_bytes: The number of bytes used to represent each parameter's value
                         (e.g., 2 for bfloat16/float16, 4 for float32).
            
        Returns:
            An integer representing the estimated memory usage in bytes.
        """
        params = self.estimate_parameters()
        
        # Parameters
        param_memory = params * dtype_bytes
        
        # Optimizer states (Adam: 2x params for momentum and variance)
        optimizer_memory = 2 * params * dtype_bytes
        
        # Gradients
        gradient_memory = params * dtype_bytes
        
        # Activations (rough estimate)
        activation_memory = self.context_length * self.hidden_size * self.num_layers * dtype_bytes
        
        # Total
        total = param_memory + optimizer_memory + gradient_memory + activation_memory
        
        return int(total)
    
    def estimate_flops_per_token(self) -> float:
        """
        Estimates the number of floating-point operations (FLOPs) required
        to process a single token through the model. This metric is crucial
        for assessing the computational cost of inference and for hardware
        selection and optimization.
        
        Returns:
            A float representing the estimated FLOPs per token.
        """
        # Attention FLOPs: 2 * n_ctx * n_embd * n_embd
        attention_flops = 2 * self.context_length * self.hidden_size * self.hidden_size
        
        # Feed-forward FLOPs: 2 * n_embd * (4 * n_embd) * 2 (up and down)
        ff_flops = 2 * self.hidden_size * (4 * self.hidden_size) * 2
        
        # Per layer
        layer_flops = attention_flops + ff_flops
        
        # Total
        total = self.num_layers * layer_flops
        
        return float(total)


# ==================== TRAINING CONFIGURATION ====================

@dataclass
class TrainingConfig(BaseConfig):
    """
    Encapsulates all parameters relevant to the training process of a model.
    This includes settings for data loading, optimization algorithms (e.g., learning rate, weight decay),
    learning rate scheduling, gradient handling, mixed precision training, checkpointing,
    evaluation strategies, logging, and distributed training configurations.
    """
    
    # Data
    dataset_path: str = "data/"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    
    # Tokenization
    tokenizer_path: Optional[str] = None
    max_length: int = 8192
    padding: str = "longest"
    truncation: bool = True
    
    # Training loop
    epochs: int = 3
    steps_per_epoch: Optional[int] = None
    total_steps: Optional[int] = None
    
    # Batch size
    batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "linear", "cosine", "constant"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.05
    
    # Gradient handling
    max_grad_norm: float = 1.0
    gradient_clipping: bool = True
    gradient_checkpointing: bool = True
    
    # Mixed precision
    mixed_precision: str = "bf16"  # "fp16", "bf16", "fp32"
    loss_scale: Optional[float] = None
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True
    
    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps", "epoch"
    
    # Logging
    logging_steps: int = 10
    logging_dir: str = "logs/"
    logging_level: str = "INFO"
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Hardware
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Randomness
    seed: int = 42
    deterministic: bool = True
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # Profiling
    profile: bool = False
    profile_steps: int = 100
    
    def __post_init__(self):
        """
        Performs post-initialization validation for `TrainingConfig` parameters.
        This includes checking the validity of the learning rate scheduler,
        mixed precision settings, and evaluation strategy. It also calculates
        the total number of training steps if `steps_per_epoch` is provided.
        
        Raises:
            ValueError: If any validation check fails, indicating an invalid
                        or unsupported configuration value.
        """
        # Validate learning rate scheduler
        valid_schedulers = ["linear", "cosine", "constant", "cosine_with_restarts"]
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(f"Invalid lr_scheduler: {self.lr_scheduler}")
        
        # Validate mixed precision
        valid_precision = ["fp16", "bf16", "fp32"]
        if self.mixed_precision not in valid_precision:
            raise ValueError(f"Invalid mixed_precision: {self.mixed_precision}")
        
        # Validate eval strategy
        valid_strategies = ["steps", "epoch", "no"]
        if self.eval_strategy not in valid_strategies:
            raise ValueError(f"Invalid eval_strategy: {self.eval_strategy}")
        
        # Calculate total steps if needed
        if self.total_steps is None and self.steps_per_epoch is not None:
            self.total_steps = self.epochs * self.steps_per_epoch
    
    def compute_derived(self) -> Dict[str, Any]:
        """
        Calculates and returns a dictionary of derived training-related values
        that are computed from the explicit configuration parameters. This includes
        metrics such as the effective batch size (considering gradient accumulation),
        actual warmup steps, and the total number of training steps.
        
        Returns:
            A dictionary containing all computed derived training values.
        """
        return {
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
            "warmup_steps_actual": int(self.warmup_steps * self.warmup_ratio),
            "total_training_steps": self.total_steps or (self.epochs * (self.steps_per_epoch or 1000)),
        }


# ==================== DATA CONFIGURATION ====================

@dataclass
class DataConfig(BaseConfig):
    """
    Defines parameters related to data handling, preprocessing, and loading for model training and evaluation.
    This includes dataset identification, format specification, text column mapping, Chain-of-Thought (CoT)
    data settings, synthetic data generation ratios, preprocessing pipelines (tokenization, truncation, padding),
    filtering criteria, data augmentation, caching, streaming, shuffling, and batching strategies.
    """
    
    # Dataset
    dataset_name: str = "custom"
    dataset_format: str = "jsonl"  # "jsonl", "parquet", "huggingface"
    text_column: str = "text"
    target_column: str = "target"
    
    # CoT data
    cot_enabled: bool = True
    cot_column: str = "cot"
    answer_column: str = "answer"
    
    # Synthetic data
    synthetic_data_ratio: float = 0.36  # 1T/2.8T
    synthetic_data_path: str = "synthetic_data/"
    
    # Preprocessing
    preprocessing_pipeline: List[str] = field(default_factory=lambda: [
        "tokenize",
        "truncate",
        "pad"
    ])
    
    # Filtering
    min_length: int = 1
    max_length: int = 8192
    filter_toxic: bool = True
    filter_low_quality: bool = True
    
    # Augmentation
    augmentation_enabled: bool = False
    augmentation_methods: List[str] = field(default_factory=lambda: [])
    augmentation_probability: float = 0.1
    
    # Caching
    cache_dir: str = ".cache/"
    use_cache: bool = True
    overwrite_cache: bool = False
    
    # Streaming
    streaming: bool = False
    streaming_buffer_size: int = 1000
    
    # Shuffling
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    shuffle_seed: int = 42
    
    # Batching
    batch_size: int = 1
    drop_last: bool = False
    collate_fn: Optional[str] = None
    
    def __post_init__(self):
        """
        Performs post-initialization validation for `DataConfig` parameters.
        This includes verifying that the specified `dataset_format` is supported,
        checking preprocessing operations, and ensuring that necessary paths
        for synthetic data and caching exist.
        
        Raises:
            ValueError: If an invalid or unsupported `dataset_format` is provided.
            UserWarning: For unknown preprocessing operations.
        """
        # Validate dataset format
        valid_formats = ["jsonl", "parquet", "huggingface", "csv", "text", "bin"]
        if self.dataset_format not in valid_formats:
            raise ValueError(f"Invalid dataset_format: {self.dataset_format}")
        
        # Validate preprocessing pipeline
        valid_operations = ["tokenize", "truncate", "pad", "mask", "chunk"]
        for op in self.preprocessing_pipeline:
            if op not in valid_operations:
                warnings.warn(f"Unknown preprocessing operation: {op}")
        
        # Ensure paths exist
        for path in [self.synthetic_data_path, self.cache_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def compute_derived(self) -> Dict[str, Any]:
        """
        Calculates and returns a dictionary of derived data-related values.
        This includes the full path to the cached processed dataset,
        the estimated number of synthetic samples needed based on the
        `synthetic_data_ratio`, and an estimation of the total dataset size in tokens.
        
        Returns:
            A dictionary containing all computed derived data values.
        """
        return {
            "cache_path": str(Path(self.cache_dir) / f"{self.dataset_name}_processed"),
            "synthetic_samples_needed": int(1_000_000_000 * self.synthetic_data_ratio),
            "estimated_dataset_size": self.estimate_dataset_size(),
        }
    
    def estimate_dataset_size(self) -> int:
        """
        Provides an estimate of the total dataset size in tokens.
        This is typically a rough approximation used for high-level
        resource planning and is not based on a precise file scan.
        
        Returns:
            An integer representing the estimated token count of the dataset.
        """
        # Rough estimate: 1MB ≈ 200,000 tokens
        # This is a placeholder - actual implementation would check files
        return 2_800_000_000_000  # 2.8T tokens default


# ==================== QUANTIZATION CONFIGURATION ====================

@dataclass
class QuantizationConfig(BaseConfig):
    """
    Manages parameters for model quantization, aiming to reduce the computational
    and memory footprint of the model without significant loss in performance.
    This includes settings for enabling/disabling quantization, specifying methods
    (e.g., progressive, static, dynamic), bit widths for weights/activations/embeddings,
    progressive quantization schedules, calibration techniques, quantization-aware training (QAT),
    and layer-wise skipping strategies.
    """
    
    # General
    enabled: bool = True
    method: str = "progressive"  # "static", "dynamic", "progressive", "aware"
    
    # Bit widths
    weight_bits: int = 8
    activation_bits: int = 8
    embedding_bits: int = 8
    
    # Progressive quantization
    progressive_enabled: bool = True
    progressive_start_step: int = 1000
    progressive_end_step: int = 100000
    progressive_levels: Tuple[int, ...] = (32, 16, 12, 8, 4)
    
    # Quantization scheme
    scheme: str = "symmetric"  # "symmetric", "asymmetric"
    per_channel: bool = True
    per_tensor: bool = False
    
    # Calibration
    calibration_samples: int = 100
    calibration_method: str = "minmax"  # "minmax", "histogram", "percentile"
    
    # Quantization-aware training
    qat_enabled: bool = False
    qat_start_step: int = 10000
    qat_num_steps: int = 5000
    
    # Observer
    observer_enabled: bool = True
    observer_momentum: float = 0.1
    observer_averaging_constant: float = 0.01
    
    # Fake quantization
    fake_quant_enabled: bool = True
    fake_quant_mode: str = "training"  # "training", "calibration"
    
    # Layer-wise quantization
    layerwise_quantization: bool = True
    skip_layers: List[str] = field(default_factory=lambda: ["router", "cot"])
    
    # Compression
    compression_enabled: bool = True
    compression_method: str = "gzip"  # "gzip", "lz4", "zstd"
    compression_level: int = 6
    
    def __post_init__(self):
        """
        Performs post-initialization validation for `QuantizationConfig` parameters.
        This method ensures that the specified quantization method, bit widths,
        scheme, and calibration methods are all supported and valid.
        
        Raises:
            ValueError: If any validation check fails, indicating an invalid
                        or unsupported configuration value.
        """
        # Validate quantization method
        valid_methods = ["static", "dynamic", "progressive", "aware"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid quantization method: {self.method}")
        
        # Validate bit widths
        valid_bits = [2, 3, 4, 5, 6, 7, 8, 16, 32]
        for bits in [self.weight_bits, self.activation_bits, self.embedding_bits]:
            if bits not in valid_bits:
                raise ValueError(f"Invalid bit width: {bits}")
        
        # Validate scheme
        if self.scheme not in ["symmetric", "asymmetric"]:
            raise ValueError(f"Invalid quantization scheme: {self.scheme}")
        
        # Validate calibration method
        valid_calibration = ["minmax", "histogram", "percentile"]
        if self.calibration_method not in valid_calibration:
            raise ValueError(f"Invalid calibration method: {self.calibration_method}")
    
    def compute_derived(self) -> Dict[str, Any]:
        """
        Calculates and returns a dictionary of derived quantization-related values.
        This includes estimations for the compression ratio, memory savings
        achieved through quantization, and the potential quantization error
        introduced by the chosen scheme and bit widths.
        
        Returns:
            A dictionary containing all computed derived quantization values.
        """
        return {
            "compression_ratio": self.estimate_compression_ratio(),
            "memory_savings": self.estimate_memory_savings(),
            "quantization_error": self.estimate_quantization_error(),
        }
    
    def estimate_compression_ratio(self) -> float:
        """
        Estimates the overall data compression ratio achieved by applying
        the configured quantization scheme and optional additional compression methods.
        The ratio is based on the reduction from a 32-bit floating-point representation
        to the specified `weight_bits`, further boosted by explicit compression algorithms.
        
        Returns:
            A float representing the estimated compression ratio (e.g., 2.0 for 2x compression).
        """
        # Base ratio from bit reduction
        base_ratio = 32 / self.weight_bits
        
        # Additional compression
        compression_ratios = {
            "gzip": 2.0,
            "lz4": 1.5,
            "zstd": 2.2,
            "none": 1.0,
        }
        
        compression_boost = compression_ratios.get(self.compression_method, 1.0)
        
        return base_ratio * compression_boost
    
    def estimate_memory_savings(self) -> float:
        """
        Estimates the memory savings factor achieved by quantizing model weights
        from a full 32-bit floating-point representation to the specified `weight_bits`.
        
        Returns:
            A float representing the memory savings factor (e.g., 4.0 for a 4x reduction
            when quantizing from 32-bit to 8-bit).
        """
        # From 32-bit to target bits
        return 32.0 / self.weight_bits
    
    def estimate_quantization_error(self) -> float:
        """
        Provides a heuristic estimate of the quantization error introduced by
        the configured quantization scheme and bit widths. This estimation
        takes into account empirical error values based on bit width and
        potential reductions from techniques like per-channel quantization,
        calibration samples, and quantization-aware training (QAT).
        
        Returns:
            A float between 0 and 1 representing the estimated quantization error,
            where 0 indicates no error and 1 indicates maximum theoretical error.
        """
        # Empirical error estimates based on bit width
        error_map = {
            2: 0.3,
            3: 0.2,
            4: 0.1,
            5: 0.07,
            6: 0.05,
            7: 0.03,
            8: 0.02,
            16: 0.001,
            32: 0.0,
        }
        
        base_error = error_map.get(self.weight_bits, 0.1)
        
        # Error reduction from techniques
        if self.per_channel:
            base_error *= 0.7
        
        if self.calibration_samples > 100:
            base_error *= 0.8
        
        if self.qat_enabled:
            base_error *= 0.5
        
        return min(1.0, base_error)


# ==================== COMPLETE CONFIGURATION ====================

@dataclass
class CompleteConfig(BaseConfig):
    """
    Represents the aggregate and complete configuration for the entire xorzen-zero framework.
    It consolidates instances of `ModelConfig`, `TrainingConfig`, `DataConfig`,
    and `QuantizationConfig` into a single, cohesive object. This class handles
    cross-component validation and synchronization, ensuring that all aspects of
    the framework operate with consistent and compatible settings.
    """
    
    # Model configuration
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    
    # Training configuration
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    
    # Data configuration
    data: DataConfig = field(default_factory=lambda: DataConfig())
    
    # Quantization configuration
    quantization: QuantizationConfig = field(default_factory=lambda: QuantizationConfig())
    
    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "json",
        "output_dir": "logs/",
        "wandb": False,
        "tensorboard": True,
    })
    
    # Optimization flags
    optimization: Dict[str, Any] = field(default_factory=lambda: {
        "compile": False,
        "fused_ops": True,
        "memory_efficient": True,
        "cpu_optimized": True,
    })
    
    # Performance targets
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "target_mmlu": 70.0,
        "target_gpqa": 68.0,
        "target_tokens_per_second": 1000,
        "target_active_ratio": 0.1,
    })
    
    def __post_init__(self):
        """
        Performs post-initialization processes for the `CompleteConfig`.
        This method is crucial for:
        1. **Synchronization**: Ensuring consistency across nested configuration
           objects (e.g., syncing `device` between `TrainingConfig` and `ModelConfig`).
        2. **Validation**: Invoking the full validation pipeline for the entire
           composite configuration to catch any inconsistencies or invalid states.
        
        Raises:
            ValueError: If any validation errors are found after synchronization.
        """
        # Synchronize configurations
        self._synchronize_configs()
        
        # Validate
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration errors: {errors}")
    
    def _synchronize_configs(self):
        """
        Ensures consistency and compatibility between various nested configuration
        components within `CompleteConfig`. This involves aligning parameters
        such as the training device, batch sizes, context lengths, and
        quantization settings across `ModelConfig`, `TrainingConfig`, and `DataConfig`.
        """
        # Sync model and training device
        if self.training.device != self.model.__dict__.get('device', 'cpu'):
            self.model.device = self.training.device
        
        # Sync batch sizes
        if self.data.batch_size != self.training.batch_size:
            self.data.batch_size = self.training.batch_size
        
        # Sync context lengths
        if self.data.max_length != self.model.context_length:
            self.data.max_length = self.model.context_length
        
        # Sync quantization settings
        if self.quantization.enabled:
            self.model.quantization = QuantizationScheme.PROGRESSIVE
    
    def validate(self) -> List[str]:
        """
        Performs comprehensive validation across the entire `CompleteConfig` object.
        It aggregates validation results from all nested configuration components
        (`ModelConfig`, `TrainingConfig`, `DataConfig`, `QuantizationConfig`)
        and adds cross-component validation rules to ensure that all integrated
        settings are mutually compatible and logically sound.
        
        Returns:
            A list of strings, each describing a validation error. The list is
            empty if the entire configuration is deemed valid.
        """
        errors = []
        
        # Validate individual configs
        errors.extend(self.model.validate())
        errors.extend(self.training.validate())
        errors.extend(self.data.validate())
        errors.extend(self.quantization.validate())
        
        # Cross-config validation
        if self.model.context_length < self.data.max_length:
            errors.append(
                f"Model context_length ({self.model.context_length}) "
                f"must be >= data.max_length ({self.data.max_length})"
            )
        
        if self.training.batch_size % self.data.batch_size != 0:
            errors.append(
                f"training.batch_size ({self.training.batch_size}) "
                f"must be divisible by data.batch_size ({self.data.batch_size})"
            )
        
        return errors
    
    def compute_derived(self) -> Dict[str, Any]:
        """
        Computes and consolidates all derived values from its constituent
        `ModelConfig`, `TrainingConfig`, `DataConfig`, and `QuantizationConfig`
        objects. Additionally, it computes performance metrics and hardware
        requirements based on the overall configuration. The results are
        flattened into a single dictionary for ease of access and reporting.
        
        Returns:
            A flattened dictionary containing all derived values, performance metrics,
            and estimated hardware requirements from the complete configuration.
        """
        derived = {
            "model": self.model.compute_derived(),
            "training": self.training.compute_derived(),
            "data": self.data.compute_derived(),
            "quantization": self.quantization.compute_derived(),
            "performance": self._compute_performance_metrics(),
            "hardware_requirements": self._compute_hardware_requirements(),
        }
        
        # Flatten for easy access
        flattened = {}
        for category, values in derived.items():
            for key, value in values.items():
                flattened[f"{category}_{key}"] = value
        
        return flattened
    
    def _compute_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculates a set of estimated performance metrics for the configured model.
        This includes predictions for MMLU and GPQA scores (leveraging
        `PerformancePredictor` from `xorzen.utils.math_utils`), estimated
        tokens per second throughput, and an analysis of parameter efficiency.
        
        Returns:
            A dictionary containing various predicted performance metrics.
        """
        from xorzen.utils.math_utils import PerformancePredictor
        
        # Get effective parameters
        effective_params = self.model.estimate_active_parameters() * self.model.estimate_efficiency_gain()
        
        # Predict performance
        mmlu = PerformancePredictor.predict_mmlu_score(
            effective_params,
            self.data.estimate_dataset_size(),
            architecture_efficiency=2.0  # xorzen-zero efficiency
        )
        
        gpqa = PerformancePredictor.predict_gpqa_score(
            mmlu,
            self.data.synthetic_data_ratio,
            reasoning_efficiency=1.5  # CoT efficiency
        )
        
        # Compute throughput
        flops_per_token = self.model.estimate_flops_per_token()
        active_flops = flops_per_token * self.model.target_active_ratio
        
        # Estimate tokens per second (simplified)
        # Assume 1 TFLOPS for CPU, 100 TFLOPS for GPU
        device_flops = 1e12 if "cuda" in self.training.device else 1e12
        tokens_per_second = device_flops / active_flops
        
        return {
            "predicted_mmlu": mmlu,
            "predicted_gpqa": gpqa,
            "predicted_tokens_per_second": tokens_per_second,
            "effective_parameters": effective_params,
            "parameter_efficiency": effective_params / self.model.estimate_parameters(),
            "performance_gap": {
                "mmlu": self.performance["target_mmlu"] - mmlu,
                "gpqa": self.performance["target_gpqa"] - gpqa,
                "throughput": tokens_per_second - self.performance["target_tokens_per_second"],
            }
        }
    
    def _compute_hardware_requirements(self) -> Dict[str, Any]:
        """
        Estimates the minimum and recommended hardware resources necessary
        to effectively run the configured model, considering its memory footprint
        (parameters, gradients, optimizer states, activations), CPU core needs,
        and disk space for checkpoints and datasets.
        
        Returns:
            A dictionary detailing the estimated hardware requirements.
        """
        # Memory requirements
        param_memory = self.model.estimate_memory(dtype_bytes=2)  # bfloat16
        gradient_memory = param_memory
        optimizer_memory = 2 * param_memory  # Adam states
        
        # Activation memory (rough estimate)
        activation_memory = (
            self.model.context_length *
            self.model.hidden_size *
            self.model.num_layers *
            2  # bytes
        )
        
        total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
        
        # Convert to GB
        total_memory_gb = total_memory / (1024 ** 3)
        
        # CPU requirements
        cpu_cores_needed = max(4, self.training.num_workers * 2)
        
        # Disk space
        checkpoint_size = param_memory * 1.2  # 20% overhead
        dataset_size = self.data.estimate_dataset_size() * 4 / (1024 ** 3)  # 4 bytes per token in GB
        
        # Expert shards (if applicable)
        expert_shards_gb = 0
        if hasattr(self.model, 'expert_count') and hasattr(self.model, 'estimate_expert_shard_size'):
            expert_shards_gb = (
                self.model.expert_count *
                self.model.estimate_expert_shard_size() /
                1024  # MB to GB
            )
        
        return {
            "minimum_ram_gb": total_memory_gb * 1.5,  # 50% safety margin
            "recommended_ram_gb": total_memory_gb * 2,
            "cpu_cores": cpu_cores_needed,
            "disk_space_gb": checkpoint_size + dataset_size + expert_shards_gb,
            "gpu_memory_gb": total_memory_gb if "cuda" in self.training.device else 0,
            "checkpoint_size_gb": checkpoint_size / (1024 ** 3),
        }


# ==================== CONFIGURATION MANAGER ====================

class ConfigManager:
    """
    Manages the lifecycle of `CompleteConfig` objects, providing utilities for:
    - **Loading**: Retrieving configurations from various file formats (YAML, JSON).
    - **Saving**: Persisting configurations to disk, including derived values.
    - **Creation**: Instantiating new configurations, potentially with overrides.
    - **Validation**: Ensuring the internal consistency and correctness of a configuration.
    - **Optimization**: Adapting configurations for specific hardware targets (CPU/GPU)
      and memory constraints.
    It also maintains a cache for frequently accessed configurations to improve performance.
    """
    
    def __init__(self, config_dir: str = "configs/"):
        """
        Initializes the `ConfigManager` instance, setting up the directory
        where configuration files will be stored and managed. It ensures
        that the specified configuration directory exists.
        
        Args:
            config_dir: The path to the directory where configuration files
                        are to be stored. Defaults to "configs/".
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for configurations
        self._config_cache: Dict[str, CompleteConfig] = {}
    
    def create_config(
        self,
        model_size: Union[ModelSize, str],
        config_name: Optional[str] = None,
        **overrides
    ) -> CompleteConfig:
        """
        Constructs a new `CompleteConfig` object, starting with a base model
        configuration determined by `model_size` and then applying any specified
        overrides. This method facilitates the dynamic creation of configurations
        tailored to specific requirements.
        
        Args:
            model_size: The desired base model size (e.g., `ModelSize.MINI_277M`).
            config_name: An optional name for the configuration. If provided,
                         it will be used as the model name within the `ModelConfig`.
            **overrides: Arbitrary keyword arguments to modify any parameter
                         within the nested `ModelConfig`, `TrainingConfig`,
                         `DataConfig`, or `QuantizationConfig`. Overrides can be
                         prefixed (e.g., `training_epochs=10`, `data_batch_size=32`).
            
        Returns:
            A newly created and initialized `CompleteConfig` instance.
        """
        # Get model config
        model_config = ConfigFactory.get_config(model_size)
        
        # Apply overrides to model config
        model_overrides = {k: v for k, v in overrides.items() if not k.startswith(('training_', 'data_', 'quantization_'))}
        if model_overrides:
            model_config.update(**model_overrides)
        
        # Create complete config
        config = CompleteConfig(model=model_config)
        
        # Apply other overrides
        for key, value in overrides.items():
            if key.startswith('training_'):
                config.training.update(**{key[9:]: value})
            elif key.startswith('data_'):
                config.data.update(**{key[5:]: value})
            elif key.startswith('quantization_'):
                config.quantization.update(**{key[13:]: value})
            elif key.startswith('logging_'):
                config.logging[key[8:]] = value
            elif key.startswith('optimization_'):
                config.optimization[key[13:]] = value
            elif key.startswith('performance_'):
                config.performance[key[12:]] = value
        
        # Set config name
        if config_name:
            config.model.model_name = config_name
        
        # Cache configuration
        cache_key = config_name or model_config.model_name
        self._config_cache[cache_key] = config
        
        return config
    
    def save_config(self, config: CompleteConfig, name: Optional[str] = None):
        """
        Persists a `CompleteConfig` object to disk in the configured `config_dir`.
        The configuration is saved in YAML format, and a companion JSON file
        containing all derived values is also saved for reference.
        
        Args:
            config: The `CompleteConfig` instance to be saved.
            name: An optional file name for the configuration. If not provided,
                  the `model_name` from the `ModelConfig` will be used.
        """
        name = name or config.model.model_name
        path = self.config_dir / f"{name}.yaml"
        
        config.save(path)
        
        # Also save derived values for reference
        derived_path = self.config_dir / f"{name}_derived.json"
        with open(derived_path, 'w') as f:
            json.dump(config.compute_derived(), f, indent=2)
    
    def load_config(self, name: str) -> CompleteConfig:
        """
        Retrieves a `CompleteConfig` object from a file within the `config_dir`.
        The method attempts to load the configuration from YAML or JSON files.
        A cache is maintained to avoid redundant file I/O for frequently accessed
        configurations.
        
        Args:
            name: The name of the configuration file (without extension) to load.
            
        Returns:
            The loaded `CompleteConfig` instance.
            
        Raises:
            FileNotFoundError: If the specified configuration file cannot be found.
        """
        # Check cache first
        if name in self._config_cache:
            return self._config_cache[name]
        
        # Try different file formats
        for ext in ['.yaml', '.yml', '.json']:
            path = self.config_dir / f"{name}{ext}"
            if path.exists():
                config = CompleteConfig.load(path)
                self._config_cache[name] = config
                return config
        
        raise FileNotFoundError(f"Configuration '{name}' not found in {self.config_dir}")
    
    def list_configs(self) -> List[str]:
        """
        Enumerates all available configuration file names (without extensions)
        present in the `config_dir`. This method identifies configurations
        regardless of their file format (YAML or JSON).
        
        Returns:
            A sorted list of strings, where each string is the name of an
            available configuration.
        """
        configs = []
        
        for ext in ['.yaml', '.yml', '.json']:
            configs.extend([
                path.stem for path in self.config_dir.glob(f"*{ext}")
                if not path.name.endswith('_derived.json')
            ])
        
        return sorted(set(configs))
    
    def get_default_configs(self) -> Dict[str, CompleteConfig]:
        """
        Generates and saves a set of default `CompleteConfig` objects for all
        predefined `ModelSize` variants. These configurations are also cached
        internally for immediate access.
        
        Returns:
            A dictionary where keys are the `ModelSize` string values and
            values are the corresponding default `CompleteConfig` instances.
        """
        configs = {}
        
        for model_size in ModelSize:
            try:
                config = self.create_config(model_size)
                configs[model_size.value] = config
                self.save_config(config, f"default_{model_size.value}")
            except Exception as e:
                warnings.warn(f"Failed to create config for {model_size}: {e}")
        
        return configs
    
    def validate_config(self, config: CompleteConfig) -> Tuple[bool, List[str]]:
        """
        Initiates the comprehensive validation process for a given `CompleteConfig` object.
        This method delegates to the `config` object's internal validation logic,
        which includes checks across all nested configuration components.
        
        Args:
            config: The `CompleteConfig` instance to be validated.
            
        Returns:
            A tuple containing:
            - A boolean indicating `True` if the configuration is valid, `False` otherwise.
            - A list of strings, where each string describes a validation error.
              This list is empty if the configuration is valid.
        """
        errors = config.validate()
        return len(errors) == 0, errors
    
    def optimize_config(
        self,
        config: CompleteConfig,
        target_device: str = "cpu",
        target_memory_gb: Optional[float] = None
    ) -> CompleteConfig:
        """
        Adjusts and optimizes a `CompleteConfig` instance based on specified
        target hardware (CPU or CUDA) and memory constraints. This method
        applies device-specific optimizations (e.g., mixed precision, compilation
        flags) and can scale model dimensions (`hidden_size`, `num_layers`)
        to fit within a target memory budget.
        
        Args:
            config: The `CompleteConfig` instance to be optimized.
            target_device: The intended deployment device. Can be "cpu" or "cuda".
                           Optimizations will be applied accordingly.
            target_memory_gb: An optional target memory constraint in gigabytes.
                              If provided, the model configuration will be scaled
                              down if necessary to attempt to fit within this limit.
            
        Returns:
            A new `CompleteConfig` instance with parameters adjusted for the
            specified target hardware and memory.
        """
        optimized = config.copy()
        
        # Optimize for device
        if target_device == "cpu":
            # CPU optimizations
            optimized.training.device = "cpu"
            optimized.training.mixed_precision = "fp32"  # CPU often benefits from fp32
            optimized.training.num_workers = min(8, optimized.training.num_workers)
            optimized.optimization["cpu_optimized"] = True
            optimized.optimization["compile"] = False  # torch.compile not always better on CPU
            
            # Reduce batch size for CPU
            optimized.training.batch_size = max(1, optimized.training.batch_size // 2)
            optimized.data.batch_size = optimized.training.batch_size
            
        elif target_device == "cuda":
            # GPU optimizations
            optimized.training.device = "cuda"
            optimized.training.mixed_precision = "bf16"
            optimized.optimization["compile"] = True
            optimized.optimization["fused_ops"] = True
        
        # Optimize for memory
        if target_memory_gb is not None:
            current_memory = optimized.model.estimate_memory() / (1024 ** 3)
            
            if current_memory > target_memory_gb:
                # Reduce model size proportionally
                reduction_factor = target_memory_gb / current_memory
                
                # Scale hidden size (maintain aspect ratio)
                new_hidden = int(optimized.model.hidden_size * (reduction_factor ** 0.5))
                new_hidden = max(256, (new_hidden // optimized.model.num_attention_heads) * optimized.model.num_attention_heads)
                
                optimized.model.hidden_size = new_hidden
                
                # Reduce layers if needed
                if reduction_factor < 0.5:
                    optimized.model.num_layers = max(6, int(optimized.model.num_layers * reduction_factor))
                
                # Update model name
                optimized.model.model_name = f"{optimized.model.model_name}_memopt"
        
        # Re-validate
        is_valid, errors = self.validate_config(optimized)
        if not is_valid:
            warnings.warn(f"Optimization created invalid config: {errors}")
        
        return optimized


# ==================== GLOBAL CONFIGURATION ====================

# Global configuration manager
_GLOBAL_CONFIG_MANAGER: Optional[ConfigManager] = None
_CURRENT_CONFIG: Optional[CompleteConfig] = None


def setup_global_config(
    model_size: Union[ModelSize, str] = ModelSize.MINI_277M,
    config_name: Optional[str] = None,
    config_dir: str = "configs/",
    **overrides
) -> CompleteConfig:
    """
    Initializes and sets the global `CompleteConfig` for the application.
    If a `config_name` is provided and a corresponding configuration file
    exists, it will be loaded. Otherwise, a new configuration is created
    based on `model_size` and any provided `overrides`.
    
    Args:
        model_size: The base model size to use if a new configuration is created.
        config_name: An optional name for the configuration. If a file with this
                     name exists in `config_dir`, it will be loaded.
        config_dir: The directory where configuration files are managed.
                    Defaults to "configs/".
        **overrides: Arbitrary keyword arguments to override parameters within
                     the `CompleteConfig`. These are applied after loading
                     or creating the base configuration.
                     
    Returns:
        The active global `CompleteConfig` instance.
    """
    global _GLOBAL_CONFIG_MANAGER, _CURRENT_CONFIG
    
    # Create config manager if needed
    if _GLOBAL_CONFIG_MANAGER is None:
        _GLOBAL_CONFIG_MANAGER = ConfigManager(config_dir)
    
    # Create or load configuration
    if config_name and config_name in _GLOBAL_CONFIG_MANAGER.list_configs():
        _CURRENT_CONFIG = _GLOBAL_CONFIG_MANAGER.load_config(config_name)
    else:
        _CURRENT_CONFIG = _GLOBAL_CONFIG_MANAGER.create_config(
            model_size, config_name, **overrides
        )
    
    return _CURRENT_CONFIG


def get_global_config() -> CompleteConfig:
    """
    Retrieves the currently active global `CompleteConfig` instance.
    If no global configuration has been explicitly set up yet (e.g., via
    `setup_global_config`), a default configuration will be initialized
    and returned.
    
    Returns:
        The active global `CompleteConfig` instance.
    """
    global _CURRENT_CONFIG
    
    if _CURRENT_CONFIG is None:
        # Setup default configuration
        _CURRENT_CONFIG = setup_global_config()
    
    return _CURRENT_CONFIG


def update_global_config(**kwargs):
    """
    Applies updates to the currently active global `CompleteConfig` instance.
    This allows for dynamic modification of configuration parameters after
    the initial setup. Parameters can be updated directly by their name or
    using a dot-separated path for nested attributes (e.g., `model.hidden_size=512`).
    Prefixed arguments (e.g., `training_epochs=10`) are also supported.
    
    Args:
        **kwargs: Arbitrary keyword arguments representing the configuration
                  parameters to be updated.
    """
    global _CURRENT_CONFIG
    
    if _CURRENT_CONFIG is None:
        get_global_config()
    
    config = get_global_config()
    for key, value in kwargs.items():
        if key.startswith('training_'):
            config.training.update(**{key[len('training_'):]: value})
        elif key.startswith('data_'):
            config.data.update(**{key[len('data_'):]: value})
        elif key.startswith('quantization_'):
            config.quantization.update(**{key[len('quantization_'):]: value})
        elif key.startswith('logging_'):
            config.logging[key[8:]] = value
        elif key.startswith('optimization_'):
            config.optimization[key[13:]] = value
        elif key.startswith('performance_'):
            config.performance[key[12:]] = value
        elif hasattr(config, key):
            setattr(config, key, value)
        elif '.' in key:
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            warnings.warn(f"Ignoring unknown config key: {key}")

# ==================== PREDEFINED CONFIGURATIONS ====================

class ConfigFactory:
    """
    A factory class responsible for generating predefined `ModelConfig` instances
    based on specified model sizes and architectural variants. It centralizes
    the creation of standard model configurations, ensuring consistency and
    simplifying model instantiation across the framework.
    """
    
    @staticmethod
    def get_config(
        model_size: Union[ModelSize, str],
        architecture: Union[ArchitectureVariant, str] = ArchitectureVariant.XORZENX_zero,
        **kwargs
    ) -> ModelConfig:
        """
        Retrieves a predefined `ModelConfig` instance based on the specified
        model size and architectural variant.
        """
        # Convert string to enum if needed
        if isinstance(model_size, str):
            model_size = ModelSize(model_size)
        
        if isinstance(architecture, str):
            architecture = ArchitectureVariant(architecture)
        
        config_data = {
            "model_size": model_size,
            "architecture": architecture,
        }
        
        # Size-specific configurations
        if model_size == ModelSize.TINY_23K:
            _h = 8
            config_data.update(
                model_name="xorzen_tiny_23k",
                vocab_size=8,
                context_length=32,
                hidden_size=_h,
                num_layers=1,
                num_attention_heads=2,
                max_depth=1,
                min_depth=1,
                width_choices=(_h,),
                cot_dim=2,
                cot_components=6,
                expert_count=1,
                top_k_experts=1,
                router_hidden_dim=1,
                router_num_layers=1,
                merger_num_layers=1,
                shard_experts=False,
                pad_token_id=0,
            )
        
        elif model_size == ModelSize.NANO_1M:
            _h = 64
            config_data.update(
                model_name="xorzen_nano_1m",
                vocab_size=6765,
                context_length=128,
                hidden_size=_h,
                num_layers=3,
                num_attention_heads=4,
                max_depth=3,
                min_depth=1,
                width_choices=(_h,),
                cot_dim=_h // 4,
                cot_components=6,
                expert_count=2,
                top_k_experts=1,
                router_hidden_dim=_h // 4,
                pad_token_id=0,
                gradient_checkpointing=False,  # no memory pressure at 1M params — 30-40% speedup
            )
        
        elif model_size == ModelSize.NANO_10M:
            _h = 192
            config_data.update(
                model_name="xorzen_nano_10m",
                vocab_size=10000,
                context_length=512,
                hidden_size=_h,
                num_layers=6,
                num_attention_heads=8,
                max_depth=6,
                min_depth=2,
                width_choices=(_h // 2, _h),
                cot_dim=_h // 4,
                cot_components=6,
                expert_count=8,
                top_k_experts=2,
                router_hidden_dim=_h // 4,
                pad_token_id=0,
            )
        
        elif model_size == ModelSize.MICRO_50M:
            _h = 256
            config_data.update(
                model_name="xorzen_micro_50m",
                vocab_size=10000,
                context_length=1024,
                hidden_size=_h,
                num_layers=10,
                num_attention_heads=8,
                max_depth=10,
                min_depth=3,
                width_choices=(_h // 2, _h),
                cot_dim=_h // 4,
                cot_components=6,
                expert_count=43,
                top_k_experts=2,
                router_hidden_dim=_h // 4,
                pad_token_id=0,
            )
        
        elif model_size == ModelSize.MINI_277M:
            _h = 512
            config_data.update(
                model_name="xorzen_zero_277m",
                vocab_size=33898,
                context_length=1024,
                hidden_size=_h,
                num_layers=13,
                num_attention_heads=16,
                max_depth=13,
                min_depth=4,
                width_choices=(_h // 2, _h),
                cot_dim=_h // 8,
                cot_components=6,
                expert_count=64,
                top_k_experts=2,
                router_hidden_dim=_h // 8,
                target_active_ratio=0.1,
                pad_token_id=0,
            )
        
        elif model_size == ModelSize.SMALL_500M:
            _h = 640
            config_data.update(
                model_name="xorzen_small_500m",
                vocab_size=64563,
                context_length=8192,
                hidden_size=_h,
                num_layers=16,
                num_attention_heads=16,
                max_depth=16,
                min_depth=4,
                width_choices=(_h // 2, _h),
                cot_dim=_h // 4,
                cot_components=6,
                expert_count=69,
                top_k_experts=2,
                router_hidden_dim=_h // 4,
                target_active_ratio=0.08,
                pad_token_id=0,
            )
        
        elif model_size == ModelSize.MEDIUM_1B:
            _h = 896
            config_data.update(
                model_name="xorzen_medium_1b",
                vocab_size=68003,
                context_length=8192,
                hidden_size=_h,
                num_layers=24,
                num_attention_heads=16,
                max_depth=24,
                min_depth=4,
                width_choices=(_h // 2, _h),
                cot_dim=_h // 4,
                cot_components=6,
                expert_count=64,
                top_k_experts=2,
                router_hidden_dim=_h // 4,
                target_active_ratio=0.07,
                pad_token_id=0,
            )
        
        elif model_size == ModelSize.LARGE_3B:
            _h = 1280
            config_data.update(
                model_name="xorzen_large_3b",
                vocab_size=94689,
                context_length=8192,
                hidden_size=_h,
                num_layers=32,
                num_attention_heads=32,
                max_depth=32,
                min_depth=5,
                width_choices=(_h // 2, _h),
                cot_dim=_h // 4,
                cot_components=6,
                expert_count=104,
                top_k_experts=2,
                router_hidden_dim=_h // 4,
                target_active_ratio=0.06,
                pad_token_id=0,
            )
        
        elif model_size == ModelSize.XL_7B:
            _h = 1792
            config_data.update(
                model_name="xorzen_xl_7b",
                vocab_size=98425,
                context_length=8192,
                hidden_size=_h,
                num_layers=48,
                num_attention_heads=32,
                max_depth=48,
                min_depth=6,
                width_choices=(_h // 2, _h),
                cot_dim=_h // 4,
                cot_components=6,
                expert_count=116,
                top_k_experts=2,
                router_hidden_dim=_h // 4,
                target_active_ratio=0.05,
                pad_token_id=0,
            )
        
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        config = ModelConfig(**config_data)
        
        # Apply any overrides
        if kwargs:
            config.update(**kwargs)
        
        return config
    
    @staticmethod
    def get_lightweight_config(
        model_name: str = "xorzen_lightweight",
        vocab_size: int = 1000,
        context_length: int = 32,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        expert_count: int = 2,
        top_k_experts: int = 1,
        **kwargs
    ) -> ModelConfig:
        """
        Generates a simplified and lightweight `ModelConfig` specifically designed
        for testing, rapid prototyping, or environments with extremely limited resources.
        """
        config = ModelConfig(
            model_name=model_name,
            vocab_size=vocab_size,
            context_length=context_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            max_depth=num_layers,
            min_depth=1,
            width_choices=(hidden_size,), # Only one width choice for simplicity
            cot_dim=hidden_size // 2,
            cot_components=6, # Changed from 2 to 6 to satisfy zeroModel validation
            expert_count=expert_count,
            top_k_experts=top_k_experts,
            expert_shard_dir="test_experts/", # Use a specific test directory
            max_expert_cache=2, # Small cache size
            target_active_ratio=1.0, # All active for small models
            router_hidden_dim=hidden_size // 4,
            router_num_layers=1,
            router_dropout=0.0,
            # Disable gradient checkpointing for lightweight model
            gradient_checkpointing=False,
            **kwargs
        )
        # Ensure hidden_size is divisible by num_attention_heads
        if config.hidden_size % config.num_attention_heads != 0:
            config.hidden_size = (config.hidden_size // config.num_attention_heads) * config.num_attention_heads
            if config.hidden_size == 0:
                config.hidden_size = config.num_attention_heads # Ensure it's not zero

        config.head_dim = config.hidden_size // config.num_attention_heads
        return config

    @staticmethod
    def get_all_configs() -> Dict[str, ModelConfig]:
        """
        Retrieves a dictionary of all currently predefined `ModelConfig` instances.
        """
        configs = {}
        
        for model_size in ModelSize:
            try:
                config = ConfigFactory.get_config(model_size)
                configs[model_size.value] = config
            except ValueError:
                continue
        
        return configs
    
    @staticmethod
    def get_optimal_config(
        target_params: int,
        target_active_ratio: float = 0.1,
        **kwargs
    ) -> ModelConfig:
        """
        Determines and returns an "optimal" `ModelConfig` instance.
        """
        # Find closest predefined size
        size_mapping = {
            1_000_000: ModelSize.NANO_1M,
            10_000_000: ModelSize.NANO_10M,
            50_000_000: ModelSize.MICRO_50M,
            277_000_000: ModelSize.MINI_277M,
            500_000_000: ModelSize.SMALL_500M,
            1_000_000_000: ModelSize.MEDIUM_1B,
            3_000_000_000: ModelSize.LARGE_3B,
            7_000_000_000: ModelSize.XL_7B,
            13_000_000_000: ModelSize.XXL_13B,
            70_000_000_000: ModelSize.XXXL_70B,
        }
        
        # Find closest size
        closest_size = min(size_mapping.keys(), key=lambda x: abs(x - target_params))
        
        # Get base config
        config = ConfigFactory.get_config(size_mapping[closest_size], **kwargs)
        
        # Adjust to exact target
        if closest_size != target_params:
            # Scale hidden size proportionally
            scale_factor = (target_params / closest_size) ** 0.5  # Square root scaling
            
            config.hidden_size = int(config.hidden_size * scale_factor)
            config.hidden_size = max(256, config.hidden_size)
            
            # Ensure divisibility
            config.hidden_size = (config.hidden_size // config.num_attention_heads) * config.num_attention_heads
            
            # Update derived name
            config.model_name = f"xorzen_custom_{target_params//1_000_000}m"
        
        # Update target active ratio
        config.target_active_ratio = target_active_ratio
        
        return config


__all__ = [
    'ModelSize',
    'ArchitectureVariant',
    'RouterType',
    'CoTType',
    'QuantizationScheme',
    'BaseConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'QuantizationConfig',
    'CompleteConfig',
    'ConfigFactory',
    'ConfigManager',
    'setup_global_config',
    'get_global_config',
    'update_global_config',
]
