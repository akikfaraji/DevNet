"""
xorzen Tokenizer Module
Comprehensive tokenization system for xorzen.

New in v0.2.2:
- Clean tokenizer loading: load_pretrained('xorzen_32k')
- Pretrained tokenizer registry
- HuggingFace tokenizers library integration
- Tokenizer metadata system

Usage:
    # Load pretrained tokenizer
    >>> from xorzen.tokenizer import load_pretrained, list_pretrained
    >>> 
    >>> # List available
    >>> print(list_pretrained())
    >>> ['xorzen_32k', 'xorzen_50k', 'xorzen_65k']
    >>> 
    >>> # Load tokenizer
    >>> tokenizer = load_pretrained('xorzen_32k')
    >>> tokens = tokenizer.encode("Hello world!")
    >>> 
    >>> # Use with data conversion
    >>> from xorzen.data import txt_to_bin
    >>> txt_to_bin('train.txt', 'train.bin', tokenizer, max_length=2048)
"""

# === CORE CLASSES ===
from .base import (
    BaseTokenizer,
    TokenizerMetadata,
    TokenizerRegistry,
)

# === TOKENIZER LOADING (NEW - PRIMARY API) ===
from .loader import (
    XORZENXTokenizer,
    load_pretrained,
    load_from_path,
    list_pretrained,
    list_pretrained_detailed,
    list_loadable,
    get_pretrained_path,
    has_pretrained,
    # create_empty_pretrained_metadata,
)

# === TOKENIZER TRAINING ===
from .trainer import train_tokenizer

# === LEGACY COMPONENTS (Backward Compatibility) ===
try:
    from .adapter import xorzenTokenizerAdapter
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False

try:
    from .analysis import TokenizerAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

try:
    from .evaluation import TokenizerEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False


__all__ = [
    # === Core Classes ===
    'BaseTokenizer',
    'TokenizerMetadata',
    'TokenizerRegistry',
    
    # === Primary API (NEW) ===
    'XORZENXTokenizer',
    'load_pretrained',
    'load_from_path',
    'list_pretrained',
    'list_pretrained_detailed',
    'list_loadable',
    'get_pretrained_path',
    'has_pretrained',
    
    # === Training ===
    'train_tokenizer',
    
    # === Utilities ===
    # 'create_empty_pretrained_metadata',
]

# Add legacy components if available
if ADAPTER_AVAILABLE:
    __all__.append('XORZENXTokenizerAdapter')

if ANALYZER_AVAILABLE:
    __all__.append('TokenizerAnalyzer')

if EVALUATOR_AVAILABLE:
    __all__.append('TokenizerEvaluator')


# === CONVENIENCE FUNCTIONS ===

def quick_load(name_or_path: str) -> BaseTokenizer:
    """
    Quickly load a tokenizer by name or path.
    
    Tries to load as pretrained first, then as file path.
    
    Args:
        name_or_path: Tokenizer name or file path
        
    Returns:
        Loaded tokenizer
        
    Example:
        >>> from xorzen.tokenizer import quick_load
        >>> tok = quick_load('xorzen_32k')  # Load pretrained
        >>> tok = quick_load('/path/to/tokenizer.json')  # Load from file
    """
    # Try pretrained first
    if has_pretrained(name_or_path):
        return load_pretrained(name_or_path)
    
    # Try as file path
    from pathlib import Path
    if Path(name_or_path).exists():
        return load_from_path(name_or_path)
    
    # Not found
    from xorzen.exceptions import TokenizerNotFoundError
    raise TokenizerNotFoundError(
        name_or_path,
        available_tokenizers=list_pretrained()
    )


def info(name: str = None):
    """
    Print tokenizer information.
    
    Args:
        name: Tokenizer name (None = list all)
        
    Example:
        >>> from xorzen.tokenizer import info
        >>> info()  # List all
        >>> info('xorzen_32k')  # Show details for xorzen_32k
    """
    if name is None:
        # List all tokenizers
        tokenizers = list_pretrained_detailed()
        
        if not tokenizers:
            print("No pretrained tokenizers found.")
            print("\nTo add tokenizers:")
            print("  1. Train tokenizer: xorzen.tokenizer.train_tokenizer(...)")
            print("  2. Place in: xorzen/tokenizer/pretrained/")
            print("  3. Update metadata.json")
            return
        
        print(f"\nAvailable Tokenizers: {len(tokenizers)}")
        print("=" * 70)
        
        for tok_name, tok_info in tokenizers.items():
            print(f"\n{tok_name}:")
            print(f"  Vocabulary: {tok_info.get('vocab_size', 'unknown'):,} tokens")
            print(f"  Version: {tok_info.get('version', 'unknown')}")
            print(f"  Description: {tok_info.get('description', 'No description')}")
    
    else:
        # Show details for specific tokenizer
        try:
            tokenizer = load_pretrained(name)
            metadata = tokenizer.metadata
            
            print(f"\nTokenizer: {name}")
            print("=" * 70)
            print(f"Vocabulary Size: {tokenizer.get_vocab_size():,} tokens")
            
            if metadata:
                print(f"Version: {metadata.version}")
                print(f"Description: {metadata.description}")
                print(f"Author: {metadata.author}")
                
                if metadata.special_tokens:
                    print("\nSpecial Tokens:")
                    for token_name, token_id in metadata.special_tokens.items():
                        print(f"  {token_name}: {token_id}")
                
                if metadata.training_corpus:
                    print(f"\nTraining Corpus: {metadata.training_corpus}")
                    if metadata.training_corpus_size:
                        print(f"Training Size: {metadata.training_corpus_size:,} tokens")
            
            # Test encoding
            test_text = "Hello world!"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            
            print(f"\nTest Encoding:")
            print(f"  Input: '{test_text}'")
            print(f"  Tokens: {tokens}")
            print(f"  Decoded: '{decoded}'")
        
        except Exception as e:
            print(f"Error loading tokenizer '{name}': {e}")


# Add to __all__
__all__.extend(['quick_load', 'info'])


# === INITIALIZATION MESSAGE ===

def _check_pretrained_availability():
    """Check if pretrained tokenizers are available."""
    available = list_pretrained()
    
    if not available:
        import warnings
        warnings.warn(
            "No pretrained tokenizers found. "
            "To use pretrained tokenizers:\n"
            "  1. Train tokenizers using xorzen.tokenizer.train_tokenizer()\n"
            "  2. Place .json files in xorzen/tokenizer/pretrained/\n"
            "  3. Update metadata.json\n"
            "Or use load_from_path() to load custom tokenizers.",
            UserWarning
        )


# Check on import (only once)
# _check_pretrained_availability() # Removed to prevent circular import/initialization issues.

