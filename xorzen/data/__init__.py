"""
xorzen Data Processing Package
Comprehensive tools for data processing, conversion, loading, and inspection.

New in v0.2.2:
- Binary format conversion (txt/json/jsonl -> .bin)
- Efficient data loading with memory mapping
- Format handlers for modular processing
- Enhanced directory batch processing

Usage:
    # Convert text to binary
    >>> from xorzen.data import txt_to_bin
    >>> from xorzen.tokenizer import load_pretrained
    >>> tokenizer = load_pretrained('xorzen_32k')
    >>> txt_to_bin('train.txt', 'train.bin', tokenizer, max_length=2048)
    
    # Load binary data
    >>> from xorzen.data import load_from_bin
    >>> data = load_from_bin('train.bin')
    
    # Batch convert directory
    >>> from xorzen.data import DataConverter
    >>> converter = DataConverter(tokenizer=tokenizer)
    >>> stats = converter.convert_directory('raw/', 'processed/', 'txt', 'bin')
"""

# === DATA CONVERSION ===
from .converter import (
    DataConverter,
    # Binary conversions (NEW)
    txt_to_bin,
    json_to_bin,
    jsonl_to_bin,
    parquet_to_bin,
    # Legacy text conversions
    parquet_to_jsonl,
    jsonl_to_json,
    json_to_jsonl,
)

# === DATA LOADING ===
from .loader import (
    # Loaders
    BinaryDataset,
    NumpyDataset,
    DirectoryLoader,
    TorchBinaryDataset,
    # Convenience functions
    load_from_bin,
    load_from_npy,
    load_from_dir,
    load_from_txt,
    load_from_json,
    load_from_jsonl,
    load_from_parquet,
)

# === FORMAT HANDLERS ===
from .formats import (
    BaseFormatHandler,
    TxtFormatHandler,
    JsonFormatHandler,
    JsonlFormatHandler,
    get_handler,
    get_handler_for_file,
    list_supported_formats,
)

# === DATA UTILITIES ===
from .inspector import DataInspector
from .cleaner import DataCleaner
from .pipeline import DataPipeline
from .augmentation import AugmentationPipeline as DataAugmentation
from .sampling import SamplingUtils as DataSampler
from .validation import DataValidator, BatchValidator

# === PYTORCH DATASETS ===
TORCH_DATASETS_AVAILABLE = False
try:
    from .loader import TokenizedDataset, MultiFileDataset
    TORCH_DATASETS_AVAILABLE = True
except (ImportError, AttributeError):
    pass


__all__ = [
    # === Conversion ===
    'DataConverter',
    # Binary
    'txt_to_bin',
    'json_to_bin',
    'jsonl_to_bin',
    'parquet_to_bin',
    # Text
    'parquet_to_jsonl',
    'jsonl_to_json',
    'json_to_jsonl',
    
    # === Loading ===
    # Loaders
    'BinaryDataset',
    'NumpyDataset',
    'DirectoryLoader',
    'TorchBinaryDataset',
    # Functions
    'load_from_bin',
    'load_from_npy',
    'load_from_dir',
    'load_from_txt',
    'load_from_json',
    'load_from_jsonl',
    'load_from_parquet',
    
    # === Format Handlers ===
    'BaseFormatHandler',
    'TxtFormatHandler',
    'JsonFormatHandler',
    'JsonlFormatHandler',
    'get_handler',
    'get_handler_for_file',
    'list_supported_formats',
    
    # === Utilities ===
    'DataInspector',
    'DataCleaner',
    'DataPipeline',
    'DataAugmentation',
    'DataSampler',
]

if TORCH_DATASETS_AVAILABLE:
    __all__.extend(['TokenizedDataset', 'MultiFileDataset'])

__all__.extend([
    'inspect_data',
    'validate_data',
    'DataValidator',
    'BatchValidator',
    'TORCH_DATASETS_AVAILABLE',
])


# === CONVENIENCE FUNCTIONS ===

def inspect_data(path):
    """
    Quick inspection of data file or directory.
    
    Args:
        path: Path to file or directory
        
    Example:
        >>> from xorzen.data import inspect_data
        >>> inspect_data('train.bin')
    """
    from pathlib import Path
    path = Path(path)
    
    if path.is_file():
        if path.suffix == '.bin':
            # Use load_from_bin to get the dataset object, specifying not to use torch
            loader = load_from_bin(path, use_torch=False, sequence_length=2048) # Default sequence_length for inspection
            info = loader.get_stats() # BinaryDataset now has get_stats()
            print(f"\nBinary File: {path.name}")
            print(f"  Size: {info['file_size_mb']:.2f} MB")
            print(f"  Sequences: {info['num_sequences']}")
            print(f"  Sequence Length: {info['sequence_length']}")
            print(f"  Dtype: {info['dtype']}")
            print(f"  Total tokens: {info['total_tokens']:,}")
        elif path.suffix == '.npy':
            # Use load_from_npy to get the dataset object
            loader = load_from_npy(path, sequence_length=2048) # Default sequence_length for inspection
            info = loader.get_stats() # NumpyDataset now has get_stats()
            print(f"\nNumPy File: {path.name}")
            print(f"  Size: {info['file_size_mb']:.2f} MB")
            print(f"  Shape: {info['shape']}")
            print(f"  Dtype: {info['dtype']}")
            print(f"  Total tokens: {info['total_tokens']:,}")
        else:
            print(f"\nFile: {path.name}")
            print(f"  Size: {path.stat().st_size / (1024*1024):.2f} MB")
            print(f"  Format: {path.suffix}")
    
    elif path.is_dir():
        loader = DirectoryLoader(path)
        stats = loader.get_stats()
        print(f"\nDirectory: {path.name}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        print(f"  By format:")
        for fmt, count in stats['files_by_format'].items():
            if count > 0: # Only show formats that actually have files
                print(f"    {fmt}: {count} files")
    
    else:
        print(f"Path not found: {path}")


def validate_data(path):
    """
    Validate data file format.
    
    Args:
        path: Path to file
        
    Returns:
        True if valid, False otherwise
    """
    from pathlib import Path
    path = Path(path)
    
    if not path.exists():
        print(f"File not found: {path}")
        return False
    
    try:
        if path.suffix == '.bin':
            # Use load_from_bin to get the dataset object
            # Provide a default sequence_length for validation to allow BinaryDataset to initialize
            loader = load_from_bin(path, use_torch=False, sequence_length=2048)
            print(f"✓ Valid binary file: {len(loader)} sequences")
            return True
        
        elif path.suffix == '.npy':
            # Use load_from_npy to get the dataset object
            # Provide a default sequence_length for validation
            loader = load_from_npy(path, sequence_length=2048)
            print(f"✓ Valid numpy file: shape {loader.data.shape}")
            return True
        
        elif path.suffix in ['.txt', '.text']:
            data = load_from_txt(path)
            print(f"✓ Valid text file: {len(data)} lines")
            return True
        
        elif path.suffix == '.json':
            data = load_from_json(path)
            print(f"✓ Valid JSON file: {len(data)} records")
            return True
        
        elif path.suffix == '.jsonl':
            data = load_from_jsonl(path)
            print(f"✓ Valid JSONL file: {len(data)} records")
            return True
        
        else:
            print(f"Unsupported format: {path.suffix}")
            return False
    
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

