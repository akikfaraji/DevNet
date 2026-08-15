"""
xorzen Tokenizer Base Classes
Abstract interfaces for tokenizer implementations.

This module provides the foundation for all tokenizers in xorzen:
- BaseTokenizer: Abstract interface
- TokenizerRegistry: Discovery and loading
- TokenizerMetadata: Tokenizer information

Example:
    >>> from xorzen.tokenizer.base import BaseTokenizer
    >>> class MyTokenizer(BaseTokenizer):
    ...     def encode(self, text):
    ...         return [1, 2, 3]
    ...     def decode(self, ids):
    ...         return "hello"
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import warnings

from xorzen.exceptions import TokenizerError, TokenizerLoadError, TokenizerNotFoundError


class TokenizerMetadata:
    """
    Metadata for tokenizer information.
    
    Stores essential information about a tokenizer:
    - Vocabulary size
    - Special tokens
    - Training corpus info
    - Version and creation date
    
    Example:
        >>> metadata = TokenizerMetadata(
        ...     name='xorzen_32k',
        ...     vocab_size=32000,
        ...     version='1.0.0'
        ... )
        >>> metadata.to_dict()
    """
    
    def __init__(
        self,
        name: str,
        vocab_size: int,
        version: str = "1.0.0",
        description: str = "",
        special_tokens: Optional[Dict[str, int]] = None,
        training_corpus: Optional[str] = None,
        training_corpus_size: Optional[int] = None,
        created_at: Optional[str] = None,
        author: str = "Akik faraji",
        **kwargs
    ):
        """
        Initialize tokenizer metadata.
        
        Args:
            name: Tokenizer name (e.g., 'xorzen_32k')
            vocab_size: Total vocabulary size
            version: Version string
            description: Human-readable description
            special_tokens: Dict of special token names to IDs
            training_corpus: Name of training corpus
            training_corpus_size: Size of training corpus in tokens
            created_at: ISO format creation date
            author: Creator/organization
            **kwargs: Additional custom metadata
        """
        self.name = name
        self.vocab_size = vocab_size
        self.version = version
        self.description = description
        self.special_tokens = special_tokens or {}
        self.training_corpus = training_corpus
        self.training_corpus_size = training_corpus_size
        self.created_at = created_at
        self.author = author
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'name': self.name,
            'vocab_size': self.vocab_size,
            'version': self.version,
            'description': self.description,
            'special_tokens': self.special_tokens,
            'training_corpus': self.training_corpus,
            'training_corpus_size': self.training_corpus_size,
            'created_at': self.created_at,
            'author': self.author,
        }
        data.update(self.extra)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenizerMetadata':
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, path: Union[str, Path]):
        """Save metadata to JSON file."""
        path = Path(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TokenizerMetadata':
        """Load metadata from JSON file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        return f"TokenizerMetadata(name='{self.name}', vocab_size={self.vocab_size}, version='{self.version}')"


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.
    
    All tokenizer implementations must inherit from this class and
    implement the required methods.
    
    Example:
        >>> class MyTokenizer(BaseTokenizer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.vocab = {'hello': 0, 'world': 1}
        ...         self._metadata = TokenizerMetadata(
        ...             name='my_tokenizer',
        ...             vocab_size=len(self.vocab)
        ...         )
        ...     
        ...     def encode(self, text, add_special_tokens=True):
        ...         words = text.split()
        ...         return [self.vocab.get(w, self.unk_token_id) for w in words]
        ...     
        ...     def decode(self, token_ids, skip_special_tokens=True):
        ...         return ' '.join(str(id) for id in token_ids)
        ...     
        ...     def get_vocab_size(self):
        ...         return len(self.vocab)
    """
    
    def __init__(self):
        """Initialize base tokenizer."""
        self._metadata: Optional[TokenizerMetadata] = None
        
        # Standard special token IDs (can be overridden)
        self.pad_token_id: int = 0
        self.unk_token_id: int = 1
        self.bos_token_id: int = 2
        self.eos_token_id: int = 3
        
        # Special token strings
        self.pad_token: str = "<pad>"
        self.unk_token: str = "<unk>"
        self.bos_token: str = "<s>"
        self.eos_token: str = "</s>"
    
    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad to max_length
            truncation: Truncate to max_length
            
        Returns:
            List of token IDs or dict with 'input_ids', 'attention_mask'
            
        Raises:
            TokenizerError: If encoding fails
        """
        pass
    
    @abstractmethod
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
            skip_special_tokens: Skip special tokens in output
            clean_up_tokenization_spaces: Clean up extra spaces
            
        Returns:
            Decoded text string
            
        Raises:
            TokenizerError: If decoding fails
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Total number of tokens in vocabulary
        """
        pass
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get full vocabulary mapping.
        
        Returns:
            Dictionary mapping tokens to IDs
            
        Note:
            This may be expensive for large vocabularies.
            Override for efficient implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_vocab(). "
            "This is optional but recommended for debugging."
        )
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encode multiple texts in batch.
        
        Args:
            texts: List of input texts
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad sequences to same length
            truncation: Truncate to max_length
            return_tensors: 'pt' for PyTorch, 'np' for NumPy
            
        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'
        """
        # Default implementation (can be overridden for efficiency)
        all_input_ids = []
        
        for text in texts:
            ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation
            )
            
            if isinstance(ids, dict):
                ids = ids['input_ids']
            
            all_input_ids.append(ids)
        
        # Pad to same length if requested
        if padding and max_length:
            for i, ids in enumerate(all_input_ids):
                if len(ids) < max_length:
                    all_input_ids[i] = ids + [self.pad_token_id] * (max_length - len(ids))
        
        result = {'input_ids': all_input_ids}
        
        # Create attention mask
        if padding:
            attention_mask = [
                [1 if id != self.pad_token_id else 0 for id in ids]
                for ids in all_input_ids
            ]
            result['attention_mask'] = attention_mask
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            try:
                import torch
                result['input_ids'] = torch.tensor(result['input_ids'])
                if 'attention_mask' in result:
                    result['attention_mask'] = torch.tensor(result['attention_mask'])
            except ImportError:
                warnings.warn("PyTorch not available, returning lists")
        
        elif return_tensors == 'np':
            try:
                import numpy as np
                result['input_ids'] = np.array(result['input_ids'])
                if 'attention_mask' in result:
                    result['attention_mask'] = np.array(result['attention_mask'])
            except ImportError:
                warnings.warn("NumPy not available, returning lists")
        
        return result
    
    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> List[str]:
        """
        Decode multiple sequences in batch.
        
        Args:
            sequences: List of token ID sequences
            skip_special_tokens: Skip special tokens
            clean_up_tokenization_spaces: Clean up spaces
            
        Returns:
            List of decoded strings
        """
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            for seq in sequences
        ]
    
    def save(self, path: Union[str, Path]):
        """
        Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement save(). "
            "Implement this method to enable saving."
        )
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseTokenizer':
        """
        Load tokenizer from file.
        
        Args:
            path: Path to tokenizer file
            
        Returns:
            Loaded tokenizer instance
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement load(). "
            "Implement this method to enable loading."
        )
    
    @property
    def metadata(self) -> Optional[TokenizerMetadata]:
        """Get tokenizer metadata."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: TokenizerMetadata):
        """Set tokenizer metadata."""
        self._metadata = value
    
    def get_special_tokens_map(self) -> Dict[str, Union[str, int]]:
        """Get mapping of special token names to tokens/IDs."""
        return {
            'pad_token': self.pad_token,
            'pad_token_id': self.pad_token_id,
            'unk_token': self.unk_token,
            'unk_token_id': self.unk_token_id,
            'bos_token': self.bos_token,
            'bos_token_id': self.bos_token_id,
            'eos_token': self.eos_token,
            'eos_token_id': self.eos_token_id,
        }
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.get_vocab_size()
    
    def __repr__(self) -> str:
        vocab_size = self.get_vocab_size()
        name = self._metadata.name if self._metadata else self.__class__.__name__
        return f"{name}(vocab_size={vocab_size})"
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        **kwargs
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Make tokenizer callable for convenient encoding.
        
        Args:
            text: Single text or list of texts
            add_special_tokens: Add special tokens
            **kwargs: Additional encoding arguments
            
        Returns:
            Encoded result (single sequence or batch)
        """
        if isinstance(text, str):
            return self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        else:
            return self.batch_encode(text, add_special_tokens=add_special_tokens, **kwargs)


class TokenizerRegistry:
    """
    Registry for managing available tokenizers.
    
    Keeps track of installed tokenizers and provides discovery.
    
    Example:
        >>> TokenizerRegistry.register('my_tokenizer', MyTokenizer, '/path/to/tokenizer')
        >>> tokenizer_class, path = TokenizerRegistry.get('my_tokenizer')
        >>> tokenizer = tokenizer_class.load(path)
    """
    
    _tokenizers: Dict[str, Tuple[type, str]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        tokenizer_class: type,
        path: str,
        metadata: Optional[TokenizerMetadata] = None
    ):
        """
        Register a tokenizer.
        
        Args:
            name: Tokenizer identifier
            tokenizer_class: Tokenizer class
            path: Path to tokenizer file
            metadata: Optional metadata
        """
        if not issubclass(tokenizer_class, BaseTokenizer):
            raise ValueError(
                f"tokenizer_class must be subclass of BaseTokenizer, "
                f"got {tokenizer_class}"
            )
        
        cls._tokenizers[name] = (tokenizer_class, path, metadata)
    
    @classmethod
    def get(cls, name: str) -> Tuple[type, str, Optional[TokenizerMetadata]]:
        """
        Get tokenizer class and path.
        
        Args:
            name: Tokenizer identifier
            
        Returns:
            Tuple of (tokenizer_class, path, metadata)
            
        Raises:
            TokenizerNotFoundError: If tokenizer not found
        """
        if name not in cls._tokenizers:
            available = cls.list()
            raise TokenizerNotFoundError(name, available_tokenizers=available)
        
        return cls._tokenizers[name]
    
    @classmethod
    def list(cls) -> List[str]:
        """List all registered tokenizers."""
        return sorted(cls._tokenizers.keys())
    
    @classmethod
    def list_detailed(cls) -> Dict[str, Dict[str, Any]]:
        """List tokenizers with detailed information."""
        result = {}
        
        for name, (tokenizer_class, path, metadata) in cls._tokenizers.items():
            info = {
                'class': tokenizer_class.__name__,
                'path': path,
            }
            
            if metadata:
                info.update({
                    'vocab_size': metadata.vocab_size,
                    'version': metadata.version,
                    'description': metadata.description,
                })
            
            result[name] = info
        
        return result
    
    @classmethod
    def has(cls, name: str) -> bool:
        """Check if tokenizer is registered."""
        return name in cls._tokenizers
    
    @classmethod
    def unregister(cls, name: str):
        """Remove tokenizer from registry."""
        if name in cls._tokenizers:
            del cls._tokenizers[name]
    
    @classmethod
    def clear(cls):
        """Clear all registered tokenizers."""
        cls._tokenizers.clear()


__all__ = [
    'BaseTokenizer',
    'TokenizerMetadata',
    'TokenizerRegistry',
]

