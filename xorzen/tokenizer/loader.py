"""
xorzen Tokenizer Loader
Load pretrained tokenizers and manage tokenizer discovery.

This is a CRITICAL module for xorzen - it enables:
- Loading xorzen pretrained tokenizers by name
- Loading custom tokenizers from file
- Discovering available tokenizers
- Integration with HuggingFace tokenizers

Example:
    >>> from xorzen.tokenizer import load_pretrained, list_pretrained
    >>> 
    >>> # List available tokenizers
    >>> print(list_pretrained())
    >>> ['xorzen_32k', 'xorzen_50k', 'xorzen_65k']
    >>> 
    >>> # Load a pretrained tokenizer
    >>> tokenizer = load_pretrained('xorzen_32k')
    >>> tokens = tokenizer.encode("Hello world!")
    >>> print(tokens)
    >>> [2, 3245, 8932, 3]
"""

import os
import json
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings

from .base import BaseTokenizer, TokenizerMetadata, TokenizerRegistry
from xorzen.exceptions import TokenizerNotFoundError, TokenizerLoadError
from xorzen.utils.logger import get_logger

logger = get_logger()

# Try importing tokenizers library
try:
    from tokenizers import Tokenizer as HFTokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    HFTokenizer = None


# =============================================================================
# xorzen TOKENIZER WRAPPER (HuggingFace tokenizers library)
# =============================================================================

class XORZENXTokenizer(BaseTokenizer):
    """
    xorzen tokenizer implementation using HuggingFace tokenizers library.
    
    This wraps the fast tokenizers library for efficient tokenization.
    
    Example:
        >>> tokenizer = XORZENXTokenizer.load('/path/to/tokenizer.json')
        >>> tokens = tokenizer.encode("Hello world!")
        >>> text = tokenizer.decode(tokens)
    """
    
    def __init__(self, tokenizer: 'HFTokenizer', metadata: Optional[TokenizerMetadata] = None):
        """
        Initialize xorzen tokenizer. 
        
        Args:
            tokenizer: HuggingFace Tokenizer instance
            metadata: Tokenizer metadata
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "tokenizers library required for XORZENXTokenizer. "
                "Install with: pip install tokenizers"
            )
        
        super().__init__()
        self._tokenizer = tokenizer
        self._metadata = metadata
        
        # Extract special token IDs
        self._setup_special_tokens()
    
    def _setup_special_tokens(self):
        """
        Setup special token IDs by reading them directly from the tokenizer vocab.
        This is the ground truth — it reflects what the tokenizer file actually
        contains, overriding any class-level defaults.
        """
        try:
            vocab = self._tokenizer.get_vocab()

            # Map of (token_string, fallback_candidates) → attribute names
            token_map = [
                (['<pad>', '[PAD]'],  'pad_token',  'pad_token_id'),
                (['<unk>', '[UNK]'],  'unk_token',  'unk_token_id'),
                (['<s>',   '[BOS]'],  'bos_token',  'bos_token_id'),
                (['</s>',  '[EOS]'],  'eos_token',  'eos_token_id'),
                (['<mask>','[MASK]'], 'mask_token', 'mask_token_id'),
            ]

            for candidates, str_attr, id_attr in token_map:
                for token_str in candidates:
                    if token_str in vocab:
                        setattr(self, str_attr, token_str)
                        setattr(self, id_attr, vocab[token_str])
                        break

        except Exception as e:
            logger.warning("tokenizer.loader",
                           f"Failed to setup special tokens from vocab: {e}. "
                           f"Using class defaults.")
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum length
            padding: Pad to max_length
            truncation: Truncate to max_length
            
        Returns:
            List of token IDs
        """
        try:
            # Prepare encoding options for the underlying tokenizer
            encode_kwargs = {"add_special_tokens": add_special_tokens}

            if max_length is not None:
                encode_kwargs["truncation"] = truncation
                encode_kwargs["max_length"] = max_length
            
            if padding:
                encode_kwargs["padding"] = "max_length" if max_length else True
                encode_kwargs["pad_id"] = self.pad_token_id
            
            # Encode
            encoding = self._tokenizer.encode(text, **encode_kwargs)
            
            return encoding.ids
        
        except Exception as e:
            raise TokenizerLoadError(
                path=str(text[:50]),
                reason=f"Encoding failed: {e}"
            )
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens
            clean_up_tokenization_spaces: Clean up spaces
            
        Returns:
            Decoded text
        """
        try:
            text = self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
            
            if clean_up_tokenization_spaces:
                # Clean up extra spaces
                text = ' '.join(text.split())
            
            return text
        
        except Exception as e:
            raise TokenizerLoadError(
                path=str(token_ids[:10]),
                reason=f"Decoding failed: {e}"
            )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._tokenizer.get_vocab_size()
    
    def get_vocab(self) -> Dict[str, int]:
        """Get full vocabulary."""
        return self._tokenizer.get_vocab()
    
    def save(self, path: Union[str, Path], save_metadata: bool = True):
        """
        Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer.json
            save_metadata: Save metadata file alongside
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        self._tokenizer.save(str(path))
        
        # Save metadata if available
        if save_metadata and self._metadata:
            metadata_path = path.with_suffix('.meta.json')
            self._metadata.save(metadata_path)
        
        logger.info("tokenizer.loader", f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'XORZENXTokenizer':
        """
        Load tokenizer from file.
        
        Args:
            path: Path to tokenizer.json file
            
        Returns:
            Loaded XORZENXTokenizer instance
            
        Raises:
            TokenizerLoadError: If loading fails
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "tokenizers library required. Install with: pip install tokenizers"
            )
        
        path = Path(path)
        
        if not path.exists():
            raise TokenizerLoadError(
                path=str(path),
                reason="File not found"
            )
        
        try:
            # Load tokenizer with error recovery
            try:
                hf_tokenizer = HFTokenizer.from_file(str(path))
            except Exception as tokenizer_error:
                # If tokenizer file is corrupted, try to fix common issues
                error_msg = str(tokenizer_error)
                if "out of vocabulary" in error_msg.lower():
                    logger.warning("tokenizer.loader", 
                                 f"Tokenizer file appears corrupted: {error_msg}")
                    logger.warning("tokenizer.loader",
                                 f"Attempting to rebuild tokenizer from scratch...")
                    # Raise a more informative error
                    raise ValueError(
                        f"Tokenizer file {path.name} is corrupted. "
                        f"The file contains invalid token definitions. "
                        f"Please retrain the tokenizer using the training script.\n"
                        f"Original error: {error_msg}"
                    ) from tokenizer_error
                else:
                    # Re-raise other errors
                    raise
            
            # Load metadata if available
            metadata_path = path.with_suffix('.meta.json')
            metadata = None
            
            if metadata_path.exists():
                try:
                    metadata = TokenizerMetadata.load(metadata_path)
                except Exception as e:
                    logger.warning("tokenizer.loader", 
                                  f"Failed to load metadata: {e}")
            
            logger.info("tokenizer.loader", f"Loaded tokenizer from {path}")
            
            return cls(hf_tokenizer, metadata)
        
        except Exception as e:
            raise TokenizerLoadError(
                path=str(path),
                reason=f"Failed to load tokenizer: {e}"
            )


# =============================================================================
# PRETRAINED TOKENIZER DISCOVERY & REGISTRATION
# =============================================================================

_PRETRAINED_TOKENIZERS_REGISTERED = False

def _get_pretrained_tokenizers_dir() -> Path:
    """Get directory containing pretrained tokenizers."""
    # Try to find the pretrained directory
    tokenizer_module_dir = Path(__file__).parent
    pretrained_dir = tokenizer_module_dir / 'pretrained'
    
    if pretrained_dir.exists():
        return pretrained_dir
    
    # Fallback: check package installation directory
    try:
        import xorzen
        xorzen_dir = Path(xorzen.__file__).parent
        pretrained_dir = xorzen_dir / 'tokenizer' / 'pretrained'
        
        if pretrained_dir.exists():
            return pretrained_dir
    except:
        pass
    
    # Create directory if it doesn't exist
    pretrained_dir = tokenizer_module_dir / 'pretrained'
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    logger.warning("tokenizer.loader", 
                  f"Pretrained tokenizers directory created at {pretrained_dir}. "
                  "No pretrained tokenizers found yet.")
    
    return pretrained_dir

def _load_pretrained_metadata() -> Dict[str, TokenizerMetadata]:
    """
    Load metadata for all pretrained tokenizers from metadata.json.
    
    Returns:
        Dictionary mapping tokenizer names to metadata
    """
    pretrained_dir = _get_pretrained_tokenizers_dir()
    metadata_file = pretrained_dir / 'metadata.json'
    
    if not metadata_file.exists():
        return {}
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {}
        for name, meta_dict in data.items():
            if 'name' not in meta_dict:
                meta_dict['name'] = name
            result[name] = TokenizerMetadata.from_dict(meta_dict)
        
        return result
    
    except Exception as e:
        logger.warning("tokenizer.loader", 
                      f"Failed to load pretrained metadata from {metadata_file}: {e}")
        return {}


def _resolve_tokenizer_path(pretrained_dir: Path, name: str) -> Optional[Path]:
    """
    Resolve a tokenizer file path for a registered name.

    Resolution order:
      1. ``<pretrained_dir>/<name>.json``
      2. Any ``<pretrained_dir>/<name>.json`` ignoring common prefix typos
         (e.g. ``zarx`` vs ``xorzen``). This makes the registry robust to
         historical rename typos in ``metadata.json``.
      3. As a last resort, scan ``pretrained_dir`` for any ``*.json`` file
         whose stem *contains* the trailing suffix of ``name`` (after the
         first underscore), e.g. ``something_agi_tokenizer_65k.json`` still
         resolves for ``xorzen_agi_tokenizer_65k``.

    Returns:
        The resolved path if found, else ``None``.
    """
    # 1. Exact match
    candidate = pretrained_dir / f"{name}.json"
    if candidate.exists():
        return candidate

    # 2. Same stem, ignoring prefix before first underscore
    suffix = name.split("_", 1)[1] if "_" in name else name
    for f in pretrained_dir.glob("*.json"):
        if f.name == "metadata.json":
            continue
        if f.stem.split("_", 1)[-1] == suffix:
            return f

    # 3. Loose contains-match on the suffix
    for f in pretrained_dir.glob("*.json"):
        if f.name == "metadata.json":
            continue
        if suffix in f.stem:
            return f

    return None


def _register_pretrained_tokenizers():
    """
    Register all tokenizers listed in metadata.json.

    This function is metadata-driven. It registers all tokenizers found in
    `metadata.json`, making them available via `list_pretrained()`. If a
    metadata entry's `.json` file cannot be found at the expected path,
    the loader attempts a fuzzy resolution (see `_resolve_tokenizer_path`)
    so a historical typo in the metadata name does not silently break
    `load_pretrained()`.

    Tokenizers whose files genuinely cannot be found are still registered
    (so they appear in `list_pretrained()` and produce a clear error when
    `load_pretrained()` is called), but a warning is logged at registration
    time pointing to the missing file.
    """
    TokenizerRegistry.clear()
    metadata_dict = _load_pretrained_metadata()
    pretrained_dir = _get_pretrained_tokenizers_dir()

    if not metadata_dict:
        logger.warning("tokenizer.loader", "No tokenizer metadata found in metadata.json. No pretrained tokenizers will be registered.")
        return

    # Register each tokenizer from metadata
    for name, metadata in metadata_dict.items():
        path = _resolve_tokenizer_path(pretrained_dir, name)

        if path is None:
            # Fall back to the canonical name-based path so the user gets a
            # clear "file not found" error at load time pointing at the
            # expected location, rather than a silent skip.
            path = pretrained_dir / f"{name}.json"
            logger.warning(
                "tokenizer.loader",
                f"Tokenizer '{name}' is registered in metadata.json but no "
                f"matching .json file was found under {pretrained_dir}. "
                f"Loading it will raise TokenizerLoadError.",
            )
        elif path.stem != name:
            # Found a fuzzy match — register under the metadata name but
            # point at the actual file on disk so loading still works.
            logger.info(
                "tokenizer.loader",
                f"Tokenizer '{name}' resolved to file '{path.name}' (name/file "
                f"stem mismatch). Consider updating metadata.json to match.",
            )

        TokenizerRegistry.register(
            name=name,
            tokenizer_class=XORZENXTokenizer,
            path=str(path),
            metadata=metadata
        )

    if metadata_dict:
        logger.debug("tokenizer.loader",
                    f"Registered {len(metadata_dict)} pretrained tokenizers from metadata.")


def _ensure_pretrained_registered():
    """Ensure that the pretrained tokenizers are registered, but only once."""
    global _PRETRAINED_TOKENIZERS_REGISTERED
    if _PRETRAINED_TOKENIZERS_REGISTERED:
        return
    
    _register_pretrained_tokenizers()
    _PRETRAINED_TOKENIZERS_REGISTERED = True


# Auto-register on module import
_ensure_pretrained_registered()


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def load_pretrained(name: str) -> XORZENXTokenizer:
    """
    Load a pretrained xorzen tokenizer by name.
    
    This is the PRIMARY way to load tokenizers in xorzen.
    
    Args:
        name: Tokenizer name (e.g., 'xorzen_32k', 'xorzen_50k', 'xorzen_65k')
        
    Returns:
        Loaded tokenizer instance
        
    Raises:
        TokenizerNotFoundError: If tokenizer not found in metadata.
        TokenizerLoadError: If loading the tokenizer file fails (e.g., file not found).
        
    Example:
        >>> from xorzen.tokenizer import load_pretrained
        >>> tokenizer = load_pretrained('xorzen_65k') # Assuming xorzen_65k.json exists
        >>> tokens = tokenizer.encode("Hello world!")
        >>> print(tokens)
    """
    _ensure_pretrained_registered()
    
    try:
        tokenizer_class, path, metadata = TokenizerRegistry.get(name)
    except TokenizerNotFoundError:
        available = TokenizerRegistry.list()
        raise TokenizerNotFoundError(name, available_tokenizers=available) from None

    logger.info("tokenizer.loader", f"Loading pretrained tokenizer: {name} from {path}")
    
    # Load tokenizer, which will raise TokenizerLoadError if path doesn't exist
    tokenizer = tokenizer_class.load(path)
    
    # Set metadata if available
    if metadata and not tokenizer.metadata:
        tokenizer.metadata = metadata
    
    return tokenizer


def load_from_path(path: Union[str, Path]) -> XORZENXTokenizer:
    """
    Load a tokenizer from a file path.
    
    Use this for loading custom tokenizers not in the pretrained registry.
    
    Args:
        path: Path to tokenizer.json file
        
    Returns:
        Loaded tokenizer instance
        
    Raises:
        TokenizerLoadError: If loading fails
        
    Example:
        >>> from xorzen.tokenizer import load_from_path
        >>> tokenizer = load_from_path('/path/to/my_tokenizer.json')
        >>> tokens = tokenizer.encode("Hello!")
    """
    logger.info("tokenizer.loader", f"Loading tokenizer from path: {path}")
    return XORZENXTokenizer.load(path)


def list_pretrained() -> List[str]:
    """
    List all available pretrained tokenizers based on metadata.json.
    
    Returns:
        List of tokenizer names
        
    Example:
        >>> from xorzen.tokenizer import list_pretrained
        >>> print(list_pretrained())
        ['xorzen_32k', 'xorzen_50k', 'xorzen_65k', 'xorzen_opmi_65k']
    """
    _ensure_pretrained_registered()
    return TokenizerRegistry.list()


def list_pretrained_detailed() -> Dict[str, Dict[str, Any]]:
    """
    List pretrained tokenizers with detailed information from metadata.json.
    
    Returns:
        Dictionary with tokenizer info
        
    Example:
        >>> from xorzen.tokenizer import list_pretrained_detailed
        >>> info = list_pretrained_detailed()
        >>> print(info['xorzen_65k'])
    """
    _ensure_pretrained_registered()
    return TokenizerRegistry.list_detailed()


def get_pretrained_path(name: str) -> Path:
    """
    Get the configured file path for a pretrained tokenizer.
    
    Args:
        name: Tokenizer name
        
    Returns:
        Path to tokenizer file
        
    Raises:
        TokenizerNotFoundError: If tokenizer not found
        
    Example:
        >>> from xorzen.tokenizer import get_pretrained_path
        >>> path = get_pretrained_path('xorzen_65k')
        >>> print(path)
        /path/to/xorzen/tokenizer/pretrained/xorzen_65k.json
    """
    _ensure_pretrained_registered()
    _, path, _ = TokenizerRegistry.get(name)
    return Path(path)


def has_pretrained(name: str) -> bool:
    """
    Check if a pretrained tokenizer is listed in metadata.json.
    
    Args:
        name: Tokenizer name
        
    Returns:
        True if tokenizer exists in metadata
        
    Example:
        >>> from xorzen.tokenizer import has_pretrained
        >>> if has_pretrained('xorzen_65k'):
        ...     tokenizer = load_pretrained('xorzen_65k')
    """
    _ensure_pretrained_registered()
    return TokenizerRegistry.has(name)


def list_loadable() -> List[str]:
    """
    List tokenizers that are actually loadable (file exists and is valid).
    
    This checks which registered tokenizers have valid files that can be loaded.
    
    Returns:
        List of tokenizer names that can be successfully loaded
        
    Example:
        >>> from xorzen.tokenizer import list_loadable
        >>> print(list_loadable())  # Only shows working tokenizers
        ['xorzen_65k', 'gpt-2_bpe_50k']  # xorzen_bpe_10k excluded if corrupted
    """
    _ensure_pretrained_registered()
    all_names = TokenizerRegistry.list()
    loadable = []
    
    for name in all_names:
        try:
            _, path, _ = TokenizerRegistry.get(name)
            path = Path(path)
            
            # Check if file exists
            if not path.exists():
                logger.debug("tokenizer.loader", f"{name}: file not found")
                continue
                
            # Try to load (will fail if corrupted)
            try:
                HFTokenizer.from_file(str(path))
                loadable.append(name)
            except Exception as e:
                logger.debug("tokenizer.loader", f"{name}: failed to load - {e}")
                
        except Exception as e:
            logger.debug("tokenizer.loader", f"{name}: error checking - {e}")
            continue
    
    return loadable


# This function has served it's purpous of demonstration. So now it is unused and commented.
# def create_empty_pretrained_metadata():
#    """
#    Create an empty metadata.json file for pretrained tokenizers.
   
#    This is a utility function for setting up the pretrained directory.
#    """
#    pretrained_dir = _get_pretrained_tokenizers_dir()
    
#    if not pretrained_dir.exists():
#        logger.warning("tokenizer.loader", 
#                      f"Pretrained tokenizers directory created at {pretrained_dir}. "
#                      "No pretrained tokenizers found yet.")
    
#    metadata_file = pretrained_dir / 'metadata.json'
    
#    if metadata_file.exists():
#        logger.warning("tokenizer.loader", 
#                      f"Metadata file already exists: {metadata_file}")
#        return
    
    # Create template metadata
#    template = {
#        "xorzen_32k": {
#           "name": "xorzen_32k",
#           "vocab_size": 32000,
#            "version": "1.0.0",
#            "description": "xorzen 32K BPE tokenizer trained on diverse corpus",
#            "special_tokens": {
#                "<pad>": 0,
#                "<unk>": 1,
#                "<s>": 2,
#                "</s>": 3
#            },
#            "training_corpus": "mixed_corpus",
#            "training_corpus_size": 100000000,
#            "created_at": "2025-01-01T00:00:00Z",
#            "author": "xorzen Team"
#        },
#        "xorzen_50k": {
#            "name": "xorzen_50k",
#            "vocab_size": 50000,
#            "version": "1.0.0",
#            "description": "xorzen 50K BPE tokenizer with extended vocabulary",
#            "special_tokens": {
#                "<pad>": 0,
#                "<unk>": 1,
#               "<s>": 2,
#                "</s>": 3
#            },
#            "training_corpus": "mixed_corpus",
#            "training_corpus_size": 200000000,
#            "created_at": "2025-01-01T00:00:00Z",
#            "author": "xorzen Team"
#        },
#        "xorzen_65k": {
#            "name": "xorzen_65k",
#            "vocab_size": 65536,
#            "version": "1.0.0",
#            "description": "xorzen 65K BPE tokenizer (standard GPT-2 size)",
#            "special_tokens": {
#                "<pad>": 0,
#                "<unk>": 1,
#                "<s>": 2,
#                "</s>": 3
#            },
#            "training_corpus": "mixed_corpus",
#            "training_corpus_size": 500000000,
#            "created_at": "2025-01-01T00:00:00Z",
#            "author": "xorzen Team"
#        }
#    }
    
#    with open(metadata_file, 'w', encoding='utf-8') as f:
#        json.dump(template, f, indent=2, ensure_ascii=False)
    
#    logger.info("tokenizer.loader", f"Created metadata template: {metadata_file}")
#    print(f"\n✅ Created metadata template: {metadata_file}")
#    print("📝 Edit this file to add your pretrained tokenizers")
#    print(f"📁 Place tokenizer files in: {pretrained_dir}")
#    print("\nExample:")
#    print(f"  1. Train tokenizer and save as: {pretrained_dir}/xorzen_32k.json")
#    print(f"  2. Update metadata in: {metadata_file}")
#    print(f"  3. Use: load_pretrained('xorzen_32k')\n")


# =============================================================================
# INITIALIZATION
# =============================================================================

def _initialize_pretrained_directory():
    """Initialize pretrained directory if needed."""
    pretrained_dir = _get_pretrained_tokenizers_dir()
    
    metadata_file = pretrained_dir / 'metadata.json'
    if not metadata_file.exists() and not any(pretrained_dir.glob('*.json')):
        logger.info("tokenizer.loader", 
                   "Pretrained tokenizers directory is empty. "
                   "Consider running `create_empty_pretrained_metadata()` to set up.")


_initialize_pretrained_directory()


__all__ = [
    'XORZENXTokenizer',
    'load_pretrained',
    'load_from_path',
    'list_pretrained',
    'list_pretrained_detailed',
    'list_loadable',
    'get_pretrained_path',
    'has_pretrained',
    # 'create_empty_pretrained_metadata',
]

