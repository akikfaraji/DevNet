"""
Advanced Tokenizer Module for xorzen Framework

This module provides comprehensive tokenization capabilities including:
- Multiple tokenizer types (BPE, WordPiece, Unigram, SentencePiece)
- Advanced training configurations
- Vocabulary management
- Pre/post-processing pipelines
- Tokenizer analysis and statistics
- Tokenizer merging and ensembling
"""

import os
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Union, Tuple, Iterator, Callable, Any, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from collections import Counter, defaultdict
import warnings

try:
    from tokenizers import (
        Tokenizer,
        ByteLevelBPETokenizer,
        SentencePieceBPETokenizer,
        BertWordPieceTokenizer,
        Regex,
        normalizers,
        pre_tokenizers,
        decoders,
        processors,
        trainers,
        models
    )
    from tokenizers.normalizers import NFD, Lowercase, StripAccents
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Digits
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    warnings.warn("tokenizers library not available. Install with: pip install tokenizers")


# ==================== ENUMS ====================

class TokenizerType(Enum):
    """Supported tokenizer types."""
    BPE = "bpe"
    BYTE_LEVEL_BPE = "byte_level_bpe"
    WORDPIECE = "wordpiece"
    UNIGRAM = "unigram"
    SENTENCEPIECE_BPE = "sentencepiece_bpe"
    SENTENCEPIECE_UNIGRAM = "sentencepiece_unigram"
    CHAR = "char"
    WORD = "word"


class NormalizationType(Enum):
    """Text normalization strategies."""
    NONE = "none"
    NFD = "nfd"
    NFKD = "nfkd"
    NFC = "nfc"
    NFKC = "nfkc"
    LOWERCASE = "lowercase"
    STRIP_ACCENTS = "strip_accents"
    CUSTOM = "custom"


class PreTokenizationType(Enum):
    """Pre-tokenization strategies."""
    NONE = "none"
    WHITESPACE = "whitespace"
    BYTE_LEVEL = "byte_level"
    BERT = "bert"
    METASPACE = "metaspace"
    DIGITS = "digits"
    PUNCTUATION = "punctuation"
    CUSTOM = "custom"


class PaddingStrategy(Enum):
    """Padding strategies."""
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(Enum):
    """Truncation strategies."""
    LONGEST_FIRST = "longest_first"
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    DO_NOT_TRUNCATE = "do_not_truncate"


# ==================== CONFIGURATION ====================

@dataclass
class TokenizerConfig:
    """Comprehensive tokenizer configuration."""
    
    # Basic settings
    tokenizer_type: TokenizerType = TokenizerType.BYTE_LEVEL_BPE
    vocab_size: int = 32000
    min_frequency: int = 2
    
    # Special tokens
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    mask_token: str = "<mask>"
    sep_token: str = "<sep>"
    cls_token: str = "<cls>"
    
    # Additional special tokens
    additional_special_tokens: List[str] = field(default_factory=list)
    
    # Normalization
    normalization: NormalizationType = NormalizationType.NFD
    lowercase: bool = True
    strip_accents: bool = True
    
    # Pre-tokenization
    pre_tokenization: PreTokenizationType = PreTokenizationType.BYTE_LEVEL
    add_prefix_space: bool = True
    
    # Model-specific
    # For BPE
    bpe_dropout: Optional[float] = None
    bpe_continuing_subword_prefix: str = "##"
    bpe_end_of_word_suffix: str = ""
    
    # For WordPiece
    wordpiece_unk_token: str = "[UNK]"
    wordpiece_max_input_chars_per_word: int = 100
    
    # For Unigram
    unigram_shrinking_factor: float = 0.75
    unigram_max_piece_length: int = 16
    unigram_n_sub_iterations: int = 2
    
    # Training
    show_progress: bool = True
    limit_alphabet: Optional[int] = None
    initial_alphabet: Optional[List[str]] = None
    
    # Processing
    max_length: Optional[int] = None
    padding: PaddingStrategy = PaddingStrategy.DO_NOT_PAD
    truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE
    stride: int = 0
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    return_attention_mask: bool = True
    
    # Decoding
    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = True
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Advanced
    split_on_punct: bool = False
    handle_chinese_chars: bool = False
    never_split: Optional[Set[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert enums to strings
        for key, value in d.items():
            if isinstance(value, Enum):
                d[key] = value.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenizerConfig':
        """Create from dictionary."""
        return cls(**data)
    
    
    def save(self, path: Union[str, Path]):
        """Save configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TokenizerConfig':
        """Load configuration."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class TokenizerStatistics:
    """Statistics about a tokenizer."""
    
    vocab_size: int
    num_special_tokens: int
    avg_token_length: float
    max_token_length: int
    min_token_length: int
    token_length_distribution: Dict[int, int]
    most_common_tokens: List[Tuple[str, int]]
    least_common_tokens: List[Tuple[str, int]]
    char_coverage: float
    subword_fertility: float  # Average tokens per word
    compression_ratio: float  # Tokens per character
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"TokenizerStatistics:\n"
            f"  Vocabulary Size: {self.vocab_size:,}\n"
            f"  Special Tokens: {self.num_special_tokens}\n"
            f"  Avg Token Length: {self.avg_token_length:.2f}\n"
            f"  Token Length Range: [{self.min_token_length}, {self.max_token_length}]\n"
            f"  Character Coverage: {self.char_coverage:.2%}\n"
            f"  Subword Fertility: {self.subword_fertility:.2f}\n"
            f"  Compression Ratio: {self.compression_ratio:.3f}"
        )


# ==================== TOKENIZER TRAINER ====================

class TokenizerTrainer:
    """Advanced tokenizer training with multiple algorithms."""
    
    def __init__(self, config: TokenizerConfig):
        """
        Initialize trainer.
        
        Args:
            config: Tokenizer configuration
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library is required for training")
        
        self.config = config
        self.tokenizer: Optional[Tokenizer] = None
    
    def train(
        self,
        files: Optional[List[str]] = None,
        iterator: Optional[Iterator[str]] = None,
        vocab_size: Optional[int] = None,
        show_progress: Optional[bool] = None
    ) -> Tokenizer:
        """
        Train tokenizer.
        
        Args:
            files: List of text files to train on
            iterator: Text iterator to train on
            vocab_size: Vocabulary size (overrides config)
            show_progress: Show progress bar
            
        Returns:
            Trained tokenizer
        """
        if files is None and iterator is None:
            raise ValueError("Must provide either files or iterator")
        
        vocab_size = vocab_size or self.config.vocab_size
        show_progress = show_progress if show_progress is not None else self.config.show_progress
        
        # Create tokenizer based on type
        if self.config.tokenizer_type == TokenizerType.BYTE_LEVEL_BPE:
            tokenizer = self._train_byte_level_bpe(files, iterator, vocab_size, show_progress)
        elif self.config.tokenizer_type == TokenizerType.BPE:
            tokenizer = self._train_bpe(files, iterator, vocab_size, show_progress)
        elif self.config.tokenizer_type == TokenizerType.WORDPIECE:
            tokenizer = self._train_wordpiece(files, iterator, vocab_size, show_progress)
        elif self.config.tokenizer_type == TokenizerType.UNIGRAM:
            tokenizer = self._train_unigram(files, iterator, vocab_size, show_progress)
        elif self.config.tokenizer_type == TokenizerType.SENTENCEPIECE_BPE:
            tokenizer = self._train_sentencepiece_bpe(files, iterator, vocab_size, show_progress)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.config.tokenizer_type}")
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def _get_special_tokens(self) -> List[str]:
        """Get list of special tokens."""
        special_tokens = []
        
        for token in [
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.pad_token,
            self.config.mask_token,
            self.config.sep_token,
            self.config.cls_token
        ]:
            if token:
                special_tokens.append(token)
        
        special_tokens.extend(self.config.additional_special_tokens)
        
        return list(set(special_tokens))  # Remove duplicates
    
    def _setup_normalizer(self, tokenizer: Tokenizer):
        """Setup text normalizer."""
        normalizers_list = []
        
        if self.config.normalization == NormalizationType.NFD:
            normalizers_list.append(NFD())
        elif self.config.normalization == NormalizationType.NFKD:
            normalizers_list.append(normalizers.NFKD())
        elif self.config.normalization == NormalizationType.NFC:
            normalizers_list.append(normalizers.NFC())
        elif self.config.normalization == NormalizationType.NFKC:
            normalizers_list.append(normalizers.NFKC())
        
        if self.config.lowercase:
            normalizers_list.append(Lowercase())
        
        if self.config.strip_accents:
            normalizers_list.append(StripAccents())
        
        if normalizers_list:
            tokenizer.normalizer = normalizers.Sequence(normalizers_list)
    
    def _setup_pre_tokenizer(self, tokenizer: Tokenizer):
        """Setup pre-tokenizer."""
        if self.config.pre_tokenization == PreTokenizationType.BYTE_LEVEL:
            tokenizer.pre_tokenizer = ByteLevel(
                add_prefix_space=self.config.add_prefix_space
            )
        elif self.config.pre_tokenization == PreTokenizationType.WHITESPACE:
            tokenizer.pre_tokenizer = Whitespace()
        elif self.config.pre_tokenization == PreTokenizationType.BERT:
            tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        elif self.config.pre_tokenization == PreTokenizationType.METASPACE:
            tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        elif self.config.pre_tokenization == PreTokenizationType.DIGITS:
            tokenizer.pre_tokenizer = Digits()
    
    def _setup_decoder(self, tokenizer: Tokenizer):
        """Setup decoder."""
        if self.config.pre_tokenization == PreTokenizationType.BYTE_LEVEL:
            tokenizer.decoder = ByteLevelDecoder()
        elif self.config.tokenizer_type == TokenizerType.WORDPIECE:
            tokenizer.decoder = decoders.WordPiece(prefix=self.config.bpe_continuing_subword_prefix)
    
    def _train_byte_level_bpe(
        self,
        files: Optional[List[str]],
        iterator: Optional[Iterator[str]],
        vocab_size: int,
        show_progress: bool
    ) -> Tokenizer:
        """Train Byte-Level BPE tokenizer."""
        tokenizer = ByteLevelBPETokenizer(
            lowercase=self.config.lowercase,
            add_prefix_space=self.config.add_prefix_space
        )
        
        special_tokens = self._get_special_tokens()
        
        if files:
            tokenizer.train(
                files=files,
                vocab_size=vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=special_tokens,
                show_progress=show_progress
            )
        else:
            tokenizer.train_from_iterator(
                iterator=iterator,
                vocab_size=vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=special_tokens,
                show_progress=show_progress
            )
        
        return tokenizer
    
    def _train_bpe(
        self,
        files: Optional[List[str]],
        iterator: Optional[Iterator[str]],
        vocab_size: int,
        show_progress: bool
    ) -> Tokenizer:
        """Train standard BPE tokenizer."""
        # Create BPE model
        model = models.BPE(
            unk_token=self.config.unk_token,
            continuing_subword_prefix=self.config.bpe_continuing_subword_prefix,
            end_of_word_suffix=self.config.bpe_end_of_word_suffix,
            dropout=self.config.bpe_dropout
        )
        
        tokenizer = Tokenizer(model)
        
        # Setup normalizer and pre-tokenizer
        self._setup_normalizer(tokenizer)
        self._setup_pre_tokenizer(tokenizer)
        self._setup_decoder(tokenizer)
        
        # Train
        special_tokens = self._get_special_tokens()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=special_tokens,
            show_progress=show_progress,
            initial_alphabet=self.config.initial_alphabet or [],
            limit_alphabet=self.config.limit_alphabet
        )
        
        if files:
            tokenizer.train(files=files, trainer=trainer)
        else:
            tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
        
        return tokenizer
    
    def _train_wordpiece(
        self,
        files: Optional[List[str]],
        iterator: Optional[Iterator[str]],
        vocab_size: int,
        show_progress: bool
    ) -> Tokenizer:
        """Train WordPiece tokenizer."""
        tokenizer = BertWordPieceTokenizer(
            unk_token=self.config.wordpiece_unk_token,
            lowercase=self.config.lowercase,
            strip_accents=self.config.strip_accents
        )
        
        special_tokens = self._get_special_tokens()
        
        if files:
            tokenizer.train(
                files=files,
                vocab_size=vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=special_tokens,
                show_progress=show_progress,
                wordpieces_prefix=self.config.bpe_continuing_subword_prefix
            )
        else:
            tokenizer.train_from_iterator(
                iterator=iterator,
                vocab_size=vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=special_tokens,
                show_progress=show_progress,
                wordpieces_prefix=self.config.bpe_continuing_subword_prefix
            )
        
        return tokenizer
    
    def _train_unigram(
        self,
        files: Optional[List[str]],
        iterator: Optional[Iterator[str]],
        vocab_size: int,
        show_progress: bool
    ) -> Tokenizer:
        """Train Unigram tokenizer."""
        model = models.Unigram()
        tokenizer = Tokenizer(model)
        
        # Setup normalizer and pre-tokenizer
        self._setup_normalizer(tokenizer)
        self._setup_pre_tokenizer(tokenizer)
        
        # Train
        special_tokens = self._get_special_tokens()
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
            unk_token=self.config.unk_token,
            shrinking_factor=self.config.unigram_shrinking_factor,
            max_piece_length=self.config.unigram_max_piece_length,
            n_sub_iterations=self.config.unigram_n_sub_iterations
        )
        
        if files:
            tokenizer.train(files=files, trainer=trainer)
        else:
            tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
        
        return tokenizer
    
    def _train_sentencepiece_bpe(
        self,
        files: Optional[List[str]],
        iterator: Optional[Iterator[str]],
        vocab_size: int,
        show_progress: bool
    ) -> Tokenizer:
        """Train SentencePiece BPE tokenizer."""
        tokenizer = SentencePieceBPETokenizer()
        
        special_tokens = self._get_special_tokens()
        
        if files:
            tokenizer.train(
                files=files,
                vocab_size=vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=special_tokens,
                show_progress=show_progress
            )
        else:
            tokenizer.train_from_iterator(
                iterator=iterator,
                vocab_size=vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=special_tokens,
                show_progress=show_progress
            )
        
        return tokenizer


# ==================== VOCABULARY MANAGER ====================

class VocabularyManager:
    """Manage tokenizer vocabulary."""
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize vocabulary manager.
        
        Args:
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer
        self._vocab_cache: Optional[Dict[str, int]] = None
        self._reverse_vocab_cache: Optional[Dict[int, str]] = None
    
    @property
    def vocab(self) -> Dict[str, int]:
        """Get vocabulary (token -> id mapping)."""
        if self._vocab_cache is None:
            self._vocab_cache = self.tokenizer.get_vocab()
        return self._vocab_cache
    
    @property
    def reverse_vocab(self) -> Dict[int, str]:
        """Get reverse vocabulary (id -> token mapping)."""
        if self._reverse_vocab_cache is None:
            self._reverse_vocab_cache = {v: k for k, v in self.vocab.items()}
        return self._reverse_vocab_cache
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()
    
    def get_token_id(self, token: str) -> Optional[int]:
        """Get ID for a token."""
        return self.vocab.get(token)
    
    def get_token(self, token_id: int) -> Optional[str]:
        """Get token for an ID."""
        return self.reverse_vocab.get(token_id)
    
    def add_tokens(self, tokens: List[str]) -> int:
        """
        Add new tokens to vocabulary.
        
        Args:
            tokens: List of tokens to add
            
        Returns:
            Number of tokens added
        """
        added = self.tokenizer.add_tokens(tokens)
        self._clear_cache()
        return added
    
    def add_special_tokens(self, special_tokens: List[str]) -> int:
        """
        Add new special tokens to vocabulary.
        
        Args:
            special_tokens: List of special tokens to add
            
        Returns:
            Number of tokens added
        """
        added = self.tokenizer.add_special_tokens(special_tokens)
        self._clear_cache()
        return added
    
    def _clear_cache(self):
        """Clear vocabulary caches."""
        self._vocab_cache = None
        self._reverse_vocab_cache = None
    
    def get_token_frequency(self, token: str, corpus: Iterator[str]) -> int:
        """
        Get frequency of a token in corpus.
        
        Args:
            token: Token to count
            corpus: Text corpus
            
        Returns:
            Token frequency
        """
        count = 0
        token_id = self.get_token_id(token)
        
        if token_id is None:
            return 0
        
        for text in corpus:
            encoding = self.tokenizer.encode(text)
            count += encoding.ids.count(token_id)
        
        return count
    
    def get_most_common_tokens(self, n: int = 100) -> List[Tuple[str, int]]:
        """
        Get most common tokens by ID (lower IDs are typically more common).
        
        Args:
            n: Number of tokens to return
            
        Returns:
            List of (token, id) tuples
        """
        sorted_tokens = sorted(self.vocab.items(), key=lambda x: x[1])
        return sorted_tokens[:n]
    
    def get_least_common_tokens(self, n: int = 100) -> List[Tuple[str, int]]:
        """
        Get least common tokens by ID.
        
        Args:
            n: Number of tokens to return
            
        Returns:
            List of (token, id) tuples
        """
        sorted_tokens = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        return sorted_tokens[:n]
    
    def filter_vocab_by_frequency(
        self,
        corpus: Iterator[str],
        min_frequency: int = 2
    ) -> Set[str]:
        """
        Find tokens below minimum frequency.
        
        Args:
            corpus: Text corpus
            min_frequency: Minimum frequency threshold
            
        Returns:
            Set of low-frequency tokens
        """
        token_counts = Counter()
        
        for text in corpus:
            encoding = self.tokenizer.encode(text)
            for token_id in encoding.ids:
                token = self.get_token(token_id)
                if token:
                    token_counts[token] += 1
        
        low_freq_tokens = {
            token for token, count in token_counts.items()
            if count < min_frequency
        }
        
        return low_freq_tokens
    
    def save_vocab(self, path: Union[str, Path], format: str = "json"):
        """
        Save vocabulary to file.
        
        Args:
            path: Output path
            format: Format ("json" or "txt")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        elif format == "txt":
            with open(path, 'w', encoding='utf-8') as f:
                for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                    f.write(f"{token}\t{idx}\n")
        else:
            raise ValueError(f"Unsupported format: {format}")


# Will continue with more components in the next part...

__all__ = [
    'TokenizerType',
    'NormalizationType',
    'PreTokenizationType',
    'PaddingStrategy',
    'TruncationStrategy',
    'TokenizerConfig',
    'TokenizerStatistics',
    'TokenizerTrainer',
    'VocabularyManager',
]

