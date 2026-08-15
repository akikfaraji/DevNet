
"""
The `xorzen` package provides a cutting-edge deep learning framework engineered
for building and deploying advanced AI models, particularly focusing on
large-scale language models (LLMs) and achieving Artificial General Intelligence (AGI).
This framework emphasizes efficiency, scalability, and ease of use, offering a
production-ready environment for researchers and developers.

Version 0.2.4 - Major Architectural Upgrade:
This version introduces significant enhancements, including a revamped
configuration system, improved model modularity, and optimized data handling.
"""

__version__ = "0.2.4"
__author__ = "Akik faraji"


# =============================================================================
# MODELS - Explicit model selection
# =============================================================================

from .models.zero import (
    zero_tiny_23k,
    zero_1M,
    zero_10M,
    zero_50M,
    zero_277M,      # Flagship model
    zero_500M,
    zero_1_3B,
    zero_7B,
)

from .models.igris.variants import (
    IGRIS_Nano,
    IGRIS_Micro,
)

from .models import (
    list_models,
    get_model,
    create_model,
    ModelRegistry,
)

# Legacy aliases
zero277M = zero_277M  # Backward compatibility


# =============================================================================
# TOKENIZER - Load and train tokenizers
# =============================================================================

from .tokenizer import (
    # Loading
    load_pretrained,
    load_from_path,
    list_pretrained,
    
    # Training
    train_tokenizer,
    
    # Info
    has_pretrained,
    get_pretrained_path,
)


# =============================================================================
# DATA - Data conversion and loading
# =============================================================================

from .data import (
    # Conversion (txt/json/jsonl -> bin)
    txt_to_bin,
    json_to_bin,
    jsonl_to_bin,
    parquet_to_bin,
    
    # Loading
    load_from_bin,
    load_from_npy,
    load_from_dir,
    load_from_txt,
    load_from_json,
    load_from_jsonl,
    
    # Validation
    validate_data,
    
    # Classes
    DataConverter,
    BinaryDataset,
    DirectoryLoader,
)


# =============================================================================
# TRAINING - Train and continue training
# =============================================================================

from .training import (
    # High-level API
    train,
    continue_train,
    evaluate,
    
    # Classes
    Trainer,
    CheckpointManager,
    TrainingState,
)


# =============================================================================
# EXCEPTIONS - Better error handling
# =============================================================================

from .exceptions import (
    # Base
    XORZENXError,
    
    # Model
    ModelError,
    ModelNotFoundError,
    
    # Data
    DataError,
    DataLoadError,
    DataFormatError,
    
    # Tokenizer
    TokenizerError,
    TokenizerNotFoundError,
    
    # Checkpoint
    CheckpointError,
    CheckpointNotFoundError,
    
    # Training
    TrainingError,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

from .config import (
    ConfigFactory,
    ModelConfig,
    ModelSize,
)

# =============================================================================
# UTILITIES
# =============================================================================

def info():
    """
    Displays comprehensive information about the current `xorzen` library installation.
    """
    print(f"\nXORZENX - Framework v{__version__}")
    print("="*70)
    
    # Models
    models = list_models()
    print(f"\n[MODELS]: {len(models)} variants available")
    print(f"   Flagship: zero_277M (277M params, ~26M active)")
    print(f"   Range: zero_1M -> zero_7B")
    
    # Tokenizers
    tokenizers = list_pretrained()
    if tokenizers:
        print(f"\n[TOKENIZERS]: {len(tokenizers)} pretrained available")
        print(f"   {', '.join(tokenizers)}")
    else:
        print(f"\n[TOKENIZERS]: 0 pretrained (train with xorzen.train_tokenizer)")
    
    # Data formats
    print(f"\n[DATA]:")
    print(f"   Input: txt, json, jsonl, parquet")
    print(f"   Training: .bin (tokenized, memory-mapped)")
    print(f"   Conversion: txt_to_bin(), json_to_bin(), etc.")
    
    # Training
    print(f"\n[TRAIN]:")
    print(f"   Initial: xorzen.train(model, data, epochs=10)")
    print(f"   Continue: xorzen.continue_train(model, data, checkpoint)")
    print(f"   Resume/continue support with checkpoint versioning")
    
    print("\n" + "="*70)
    print("Quick Start: import xorzen; help(xorzen)")
    print("Docs: https://github.com/Akik-Forazi/xorzen.git")
    
    status = {
        'version': __version__,
        'models_available': len(list_models()),
        'tokenizers_available': len(list_pretrained()),
    }
    
    # Check dependencies
    dependencies = {}
    
    try:
        import torch
        dependencies['torch'] = torch.__version__
    except ImportError:
        dependencies['torch'] = None
    
    try:
        import numpy
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        dependencies['numpy'] = None
    
    try:
        import tokenizers
        dependencies['tokenizers'] = tokenizers.__version__
    except ImportError:
        dependencies['tokenizers'] = None
    
    status['dependencies'] = dependencies
    
    # Print report
    print(f"\nXORZENX Installation Check")
    print("="*70)
    print(f"Version: {status['version']}")
    print(f"Models: {status['models_available']} available")
    print(f"Tokenizers: {status['tokenizers_available']} pretrained")
    print("\nDependencies:")
    for name, version in dependencies.items():
        status_icon = "[OK]" if version else "[FAIL]"
        version_str = version if version else "NOT INSTALLED"
        print(f"  {status_icon} {name}: {version_str}")
    
    all_ok = all(v is not None for v in dependencies.values())
    
    if all_ok:
        print("\n[OK] All dependencies installed!")
    else:
        print("\n[WARN] Some dependencies missing. Install with:")
        if not dependencies['torch']:
            print("   pip install torch")
        if not dependencies['numpy']:
            print("   pip install numpy")
        if not dependencies['tokenizers']:
            print("   pip install tokenizers")
    
    print("="*70 + "\n")
    
    return status


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # === MODELS ===
    'zero_tiny_23k',
    'zero_1M',
    'zero_10M',
    'zero_50M',
    'zero_277M',
    'zero_500M',
    'zero_1_3B',
    'zero_7B',
    'zero277M',  # Legacy
    'IGRIS_Nano',
    'IGRIS_Micro',
    'list_models',
    'get_model',
    'create_model',
    'ModelRegistry',
    
    # === TOKENIZER ===
    'load_pretrained',
    'load_from_path',
    'list_pretrained',
    'train_tokenizer',
    'has_pretrained',
    'get_pretrained_path',
    
    # === DATA ===
    # Conversion
    'txt_to_bin',
    'json_to_bin',
    'jsonl_to_bin',
    'parquet_to_bin',
    # Loading
    'load_from_bin',
    'load_from_npy',
    'load_from_dir',
    'load_from_txt',
    'load_from_json',
    'load_from_jsonl',
    # Validation
    'validate_data',
    # Classes
    'DataConverter',
    'BinaryDataset',
    'DirectoryLoader',
    
    # === TRAINING ===
    'train',
    'continue_train',
    'evaluate',
    'Trainer',
    'CheckpointManager',
    'TrainingState',
    
    # === EXCEPTIONS ===
    'XORZENXError',
    'ModelError',
    'ModelNotFoundError',
    'DataError',
    'DataLoadError',
    'DataFormatError',
    'TokenizerError',
    'TokenizerNotFoundError',
    'CheckpointError',
    'CheckpointNotFoundError',
    'TrainingError',
    
    # === CONFIG ===
    'ConfigFactory',
    'ModelSize',
    
    # === UTILITIES ===
    'info',
    'check_installation',
]


# =============================================================================
# INITIALIZATION
# =============================================================================

def _initialize():
    """Initialize xorzen on first import."""
    import os
    
    # Only run once per session
    if os.environ.get('XORZENX_INITIALIZED'):
        return
    
    os.environ['XORZENX_INITIALIZED'] = '1'
    
    # Optional: Print welcome message
    if os.environ.get('XORZENX_VERBOSE'):
        print(f"xorzen v{__version__} loaded")
        print(f"  Models: {len(list_models())} available")
        tokenizers = list_pretrained()
        if tokenizers:
            print(f"  Tokenizers: {len(tokenizers)} pretrained")
        print("  Use xorzen.info() for more details\n")


# Run initialization
_initialize()
