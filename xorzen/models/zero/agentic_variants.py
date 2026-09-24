
from .agentic_config import ZeroAgenticConfig, ZeroAgenticSize
from .agentic_model import ZeroAgenticModel
from xorzen.models.registry import ModelRegistry

def zero_agentic_nano(vocab_size=10000, **kwargs):
    """
    zero-Agentic-Nano (~5M parameters).
    Designed to outperform 350M models.
    """
    config = ZeroAgenticConfig(
        hidden_size=256,
        num_layers=6,
        num_attention_heads=8,
        vocab_size=vocab_size,
        recurrence_depth=3,
        memory_slots=32,
        # Fix validation errors by setting depth constraints
        max_depth=6,
        min_depth=2,
        **kwargs
    )
    return ZeroAgenticModel(config)

def zero_agentic_micro(vocab_size=10000, **kwargs):
    """
    zero-Agentic-Micro (~50M parameters).
    Designed to outperform 3B models.
    """
    config = ZeroAgenticConfig(
        hidden_size=512,
        num_layers=12,
        num_attention_heads=8,  # Fixed: was 12 (512/12 not integer), now 8 (512/8=64, valid)
        vocab_size=vocab_size,
        recurrence_depth=4,
        memory_slots=64,
        # Fix validation errors
        max_depth=12,
        min_depth=3,
        **kwargs
    )
    return ZeroAgenticModel(config)


# Register agentic models with ModelRegistry
def _register_zero_agentic_models():
    """
    Registers zero Agentic variants with the ModelRegistry.
    """
    models_to_register = [
        ("zero_agentic_nano", zero_agentic_nano, None,
         "zero Agentic Nano — compact recursive inference (~5M params)",
         lambda: ZeroAgenticConfig(hidden_size=256, num_layers=6, num_attention_heads=8,
                                  vocab_size=10000, recurrence_depth=3, memory_slots=32,
                                  max_depth=6, min_depth=2)),
        ("zero_agentic_micro", zero_agentic_micro, None,
         "zero Agentic Micro — recursive inference (~50M params)",
         lambda: ZeroAgenticConfig(hidden_size=512, num_layers=12, num_attention_heads=8,
                                  vocab_size=10000, recurrence_depth=4, memory_slots=64,
                                  max_depth=12, min_depth=3)),
    ]
    for name, model_fn, param_count, description, config_factory in models_to_register:
        ModelRegistry.register(name, model_fn, param_count or 0, description, config_factory)

_register_zero_agentic_models()
