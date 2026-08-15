
from .config import IGRISConfig, IGRISSize
from .model import IGRISModel

def IGRIS_Nano(vocab_size=10000, **kwargs):
    """
    IGRIS-Nano (~5M parameters).
    Designed to outperform 350M models.
    """
    config = IGRISConfig(
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
    return IGRISModel(config)

def IGRIS_Micro(vocab_size=10000, **kwargs):
    """
    IGRIS-Micro (~50M parameters).
    Designed to outperform 3B models.
    """
    config = IGRISConfig(
        hidden_size=512,
        num_layers=12,
        num_attention_heads=12,
        vocab_size=vocab_size,
        recurrence_depth=4,
        memory_slots=64,
        # Fix validation errors
        max_depth=12,
        min_depth=3,
        **kwargs
    )
    return IGRISModel(config)
