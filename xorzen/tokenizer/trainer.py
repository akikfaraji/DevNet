"""
xorzen Tokenizer Trainer
Enhanced tokenizer training with BPE support.

This module provides comprehensive tokenizer training:
- BPE (Byte-Pair Encoding) tokenizer training
- Support for multiple data formats
- Special token handling
- Metadata generation
- HuggingFace tokenizers library integration

Example:
    >>> from xorzen.tokenizer import train_tokenizer
    >>> 
    >>> # Train from text files
    >>> tokenizer = train_tokenizer(
    ...     files=['train.txt', 'val.txt'],
    ...     vocab_size=32000,
    ...     output_path='my_tokenizer.json'
    ... )
    >>> 
    >>> # Use the trained tokenizer
    >>> tokens = tokenizer.encode("Hello world!")
"""

from typing import List, Optional, Union, Iterator, Dict
from pathlib import Path
import json
import warnings

from xorzen.utils.logger import get_logger
from xorzen.exceptions import TokenizerError

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
    from tokenizers.processors import TemplateProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

logger = get_logger()


# =============================================================================
# TOKENIZER TRAINER
# =============================================================================

def train_tokenizer(
    files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    texts: Optional[Iterator[str]] = None,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    model_type: str = 'bpe',
    show_progress: bool = True,
    save_metadata: bool = True,
    **kwargs
):
    """
    Train a tokenizer from scratch.
    
    This is the PRIMARY tokenizer training function for xorzen.
    Supports training from files or text iterator.
    
    Args:
        files: List of training file paths (txt, json, jsonl)
        texts: Iterator of text strings (alternative to files)
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency
        special_tokens: List of special tokens
        output_path: Where to save trained tokenizer
        model_type: 'bpe' (currently only BPE supported)
        show_progress: Show training progress
        save_metadata: Save tokenizer metadata
        **kwargs: Additional training arguments
        
    Returns:
        Trained tokenizer (XORZENXTokenizer)
        
    Raises:
        TokenizerError: If training fails
        ImportError: If tokenizers library not available
        
    Example:
        >>> from xorzen.tokenizer import train_tokenizer
        >>> 
        >>> # Train from files
        >>> tokenizer = train_tokenizer(
        ...     files=['data/train.txt', 'data/val.txt'],
        ...     vocab_size=32000,
        ...     output_path='tokenizers/my_tok.json',
        ...     show_progress=True
        ... )
        >>> 
        >>> # Test tokenizer
        >>> tokens = tokenizer.encode("Hello world!")
        >>> print(tokens)
        [2, 3245, 8932, 3]
        >>> 
        >>> # Use with data conversion
        >>> from xorzen.data import txt_to_bin
        >>> txt_to_bin('train.txt', 'train.bin', tokenizer, max_length=2048)
    """
    if not TOKENIZERS_AVAILABLE:
        raise ImportError(
            "tokenizers library required for training. "
            "Install with: pip install tokenizers"
        )
    
    if files is None and texts is None:
        raise ValueError("Either 'files' or 'texts' must be provided")
    
    logger.info("tokenizer.trainer", 
               f"Training {model_type.upper()} tokenizer with vocab_size={vocab_size}")
    
    # Prepare files list
    if files is not None:
        if isinstance(files, (str, Path)):
            files = [files]
        
        # Flatten nested lists and convert to strings
        resolved_files = []
        for f_item in files:
            # Handle nested lists
            if isinstance(f_item, (list, tuple)):
                for nested in f_item:
                    if isinstance(nested, Path):
                        resolved_files.append(str(nested.resolve()))
                    else:
                        resolved_files.append(str(Path(nested).resolve()))
            # Handle Path objects
            elif isinstance(f_item, Path):
                resolved_files.append(str(f_item.resolve()))
            # Handle strings
            elif isinstance(f_item, str):
                resolved_files.append(str(Path(f_item).resolve()))
            else:
                raise TypeError(f"Invalid file type: {type(f_item)}. Expected str or Path.")
        
        files = resolved_files
        logger.info("tokenizer.trainer", f"Training on {len(files)} files")
    
    # Default special tokens
    if special_tokens is None:
        special_tokens = [
            "<pad>",  # 0
            "<unk>",  # 1
            "<s>",    # 2 (BOS)
            "</s>",   # 3 (EOS)
            "<mask>",
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
        ]
    
    logger.info("tokenizer.trainer", f"Special tokens: {special_tokens}")
    
    try:
        # Initialize tokenizer based on model type
        if model_type.lower() == 'bpe':
            tokenizer = _train_bpe_tokenizer(
                files=files,
                texts=texts,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
                show_progress=show_progress,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Only 'bpe' is currently supported.")
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            tokenizer.save(str(output_path))
            logger.info("tokenizer.trainer", f"Tokenizer saved to {output_path}")
            
            # Wrap in XORZENXTokenizer first to get proper metadata and special token IDs
            from .loader import XORZENXTokenizer
            from .base import TokenizerMetadata
            
            xorzen_tokenizer = XORZENXTokenizer(tokenizer, None) # Metadata will be added below
            
            # Save metadata
            if save_metadata:
                # Get actual special token map from the XORZENXTokenizer
                actual_special_tokens_map = xorzen_tokenizer.get_special_tokens_map()
                
                _save_tokenizer_metadata(
                    output_path,
                    vocab_size=vocab_size,
                    xorzen_tokenizer_instance=xorzen_tokenizer, # Pass the XORZENXTokenizer instance
                    training_files=files,
                    **kwargs
                )
        
        else: # If no output_path, still wrap in XORZENXTokenizer
            from .loader import XORZENXTokenizer
            from .base import TokenizerMetadata
            xorzen_tokenizer = XORZENXTokenizer(tokenizer, None)

        # Set metadata for the returned tokenizer (even if not saved)
        if xorzen_tokenizer.metadata is None:
            xorzen_tokenizer.metadata = TokenizerMetadata(
                name=output_path.stem if output_path else "custom_tokenizer",
                vocab_size=vocab_size,
                special_tokens=xorzen_tokenizer.get_special_tokens_map() # Use the actual map
            )
        
        logger.info("tokenizer.trainer", "Tokenizer training complete!")
        
        return xorzen_tokenizer
    
    except Exception as e:
        raise TokenizerError(f"Tokenizer training failed: {e}")


def _train_bpe_tokenizer(
    files: Optional[List[str]],
    texts: Optional[Iterator[str]],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
    show_progress: bool,
    add_prefix_space: bool = False,
    **kwargs
) -> Tokenizer:
    """
    Train BPE tokenizer.
    
    Internal function for BPE-specific training.
    """
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(
        unk_token="<unk>",
        byte_fallback=True,  # Handle unknown bytes gracefully
    ))
    
    # Set normalizer (NFKC Unicode normalization)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.StripAccents(),
    ])
    
    # Set pre-tokenizer (ByteLevel handles all bytes, including whitespace)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=add_prefix_space,
        trim_offsets=True,
        use_regex=True,
    )
    
    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=show_progress,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    # Train
    if files:
        tokenizer.train(files, trainer)
        logger.info("tokenizer.trainer", f"Trained on {len(files)} files")
    elif texts:
        tokenizer.train_from_iterator(texts, trainer)
        logger.info("tokenizer.trainer", "Trained from text iterator")
    
    # Set post-processor AFTER training to ensure correct special token IDs
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    if bos_id is not None and eos_id is not None:
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", bos_id),
                ("</s>", eos_id),
            ],
        )
    else:
        logger.warning("tokenizer.trainer", "BOS or EOS tokens not found in trained vocabulary, post-processor not set.")

    return tokenizer


def _save_tokenizer_metadata(
    tokenizer_path: Path,
    vocab_size: int,
    xorzen_tokenizer_instance, # Changed to accept XORZENXTokenizer instance
    training_files: Optional[List[str]] = None,
    **kwargs
):
    """Save tokenizer metadata to JSON."""
    from datetime import datetime
    
    metadata_path = tokenizer_path.with_suffix('.meta.json')
    
    metadata = {
        "name": tokenizer_path.stem,
        "vocab_size": vocab_size,
        "version": "1.0.0",
        "description": f"Custom trained tokenizer with {vocab_size} tokens",
        "special_tokens": xorzen_tokenizer_instance.get_special_tokens_map(), # Get from xorzenTokenizer instance
        "training_files": training_files if training_files else [],
        "created_at": datetime.now().isoformat(),
        "author": "Akik faraji",
    }
    
    # Add any additional kwargs to metadata
    metadata.update(kwargs)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.debug("tokenizer.trainer", f"Metadata saved to {metadata_path}")


# =============================================================================
# BATCH TRAINING
# =============================================================================

def train_multiple_tokenizers(
    configs: List[Dict],
    show_progress: bool = True
) -> List:
    """
    Train multiple tokenizers with different configurations.
    
    Useful for experiments or creating tokenizers with different vocab sizes.
    
    Args:
        configs: List of configuration dictionaries (each passed to train_tokenizer)
        show_progress: Show progress
        
    Returns:
        List of trained tokenizers
        
    Example:
        >>> from xorzen.tokenizer import train_multiple_tokenizers
        >>> 
        >>> configs = [
        ...     {'files': ['train.txt'], 'vocab_size': 16000, 'output_path': 'tok_16k.json'},
        ...     {'files': ['train.txt'], 'vocab_size': 32000, 'output_path': 'tok_32k.json'},
        ...     {'files': ['train.txt'], 'vocab_size': 50000, 'output_path': 'tok_50k.json'},
        ... ]
        >>> 
        >>> tokenizers = train_multiple_tokenizers(configs)
        >>> print(f"Trained {len(tokenizers)} tokenizers")
    """
    tokenizers = []
    
    for i, config in enumerate(configs, 1):
        logger.info("tokenizer.trainer", 
                   f"Training tokenizer {i}/{len(configs)}: vocab_size={config.get('vocab_size')}")
        
        try:
            tokenizer = train_tokenizer(
                show_progress=show_progress,
                **config
            )
            tokenizers.append(tokenizer)
        except Exception as e:
            logger.error("tokenizer.trainer", f"Failed to train tokenizer {i}: {e}")
            tokenizers.append(None)
    
    successful = sum(1 for t in tokenizers if t is not None)
    logger.info("tokenizer.trainer", 
               f"Batch training complete: {successful}/{len(configs)} successful")
    
    return tokenizers


# =============================================================================
# UTILITIES
# =============================================================================

def estimate_vocab_size(
    files: Union[str, Path, List[Union[str, Path]]],
    sample_size: int = 100000
) -> Dict[str, int]:
    """
    Estimate appropriate vocabulary size for training data.
    
    Analyzes a sample of the training data to suggest vocab sizes.
    
    Args:
        files: Training file(s)
        sample_size: Number of characters to sample
        
    Returns:
        Dictionary with statistics and suggestions
        
    Example:
        >>> from xorzen.tokenizer import estimate_vocab_size
        >>> stats = estimate_vocab_size('train.txt')
        >>> print(f"Suggested vocab size: {stats['suggested_vocab_size']}")
    """
    if isinstance(files, (str, Path)):
        files = [files]
    
    # Count unique characters and words in sample
    all_text = ""
    chars_read = 0
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            while chars_read < sample_size:
                chunk = f.read(min(1000, sample_size - chars_read))
                if not chunk:
                    break
                all_text += chunk
                chars_read += len(chunk)
        
        if chars_read >= sample_size:
            break
    
    # Calculate statistics
    unique_chars = len(set(all_text))
    words = all_text.split()
    unique_words = len(set(words))
    
    # Suggest vocab size based on unique words
    if unique_words < 10000:
        suggested = 8000
    elif unique_words < 30000:
        suggested = 16000
    elif unique_words < 100000:
        suggested = 32000
    elif unique_words < 300000:
        suggested = 50000
    else:
        suggested = 65536
    
    stats = {
        'chars_analyzed': chars_read,
        'unique_chars': unique_chars,
        'unique_words': unique_words,
        'suggested_vocab_size': suggested,
        'alternative_sizes': [
            suggested // 2,
            suggested,
            suggested * 2
        ]
    }
    
    logger.info("tokenizer.trainer", 
               f"Analyzed {chars_read:,} characters, found {unique_words:,} unique words")
    logger.info("tokenizer.trainer", f"Suggested vocab size: {suggested:,}")
    
    return stats


__all__ = [
    'train_tokenizer',
    'train_multiple_tokenizers',
    'estimate_vocab_size',
]

