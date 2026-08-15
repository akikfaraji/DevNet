"""
Tokenizer Utilities

Provides utility functions for tokenizer manipulation including:
- Tokenizer merging
- Tokenizer ensembling
- Text preprocessing
- Special handling for different languages
- Tokenizer conversion between formats
"""

import re
import unicodedata
from typing import List, Optional, Dict, Set, Tuple, Callable, Union
from pathlib import Path
import json

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


# ==================== TEXT PREPROCESSING ====================

class TextPreprocessor:
    """Comprehensive text preprocessing utilities."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.unicode_categories = set()
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFKC') -> str:
        """
        Normalize Unicode characters.
        
        Args:
            text: Input text
            form: Normalization form (NFC, NFKC, NFD, NFKD)
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize(form, text)
    
    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Remove control characters."""
        return ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
    
    @staticmethod
    def remove_accents(text: str) -> str:
        """Remove accents from characters."""
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    @staticmethod
    def lowercase(text: str) -> str:
        """Convert to lowercase."""
        return text.lower()
    
    @staticmethod
    def remove_punctuation(text: str, keep: Optional[Set[str]] = None) -> str:
        """
        Remove punctuation.
        
        Args:
            text: Input text
            keep: Set of punctuation to keep
            
        Returns:
            Text without punctuation
        """
        if keep is None:
            keep = set()
        
        result = []
        for char in text:
            if char in keep or not unicodedata.category(char).startswith('P'):
                result.append(char)
        
        return ''.join(result)
    
    @staticmethod
    def remove_digits(text: str) -> str:
        """Remove all digits."""
        return ''.join(char for char in text if not char.isdigit())
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs."""
        # Simple URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    @staticmethod
    def expand_contractions(text: str, contractions_dict: Optional[Dict[str, str]] = None) -> str:
        """
        Expand contractions.
        
        Args:
            text: Input text
            contractions_dict: Dictionary of contractions to expansions
            
        Returns:
            Text with expanded contractions
        """
        if contractions_dict is None:
            # Default English contractions
            contractions_dict = {
                "ain't": "am not",
                "aren't": "are not",
                "can't": "cannot",
                "can't've": "cannot have",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he would",
                "he'll": "he will",
                "he's": "he is",
                "how'd": "how did",
                "how'll": "how will",
                "how's": "how is",
                "i'd": "i would",
                "i'll": "i will",
                "i'm": "i am",
                "i've": "i have",
                "isn't": "is not",
                "it'd": "it would",
                "it'll": "it will",
                "it's": "it is",
                "let's": "let us",
                "mustn't": "must not",
                "shan't": "shall not",
                "she'd": "she would",
                "she'll": "she will",
                "she's": "she is",
                "shouldn't": "should not",
                "that's": "that is",
                "there's": "there is",
                "they'd": "they would",
                "they'll": "they will",
                "they're": "they are",
                "they've": "they have",
                "wasn't": "was not",
                "we'd": "we would",
                "we'll": "we will",
                "we're": "we are",
                "we've": "we have",
                "weren't": "were not",
                "what'll": "what will",
                "what're": "what are",
                "what's": "what is",
                "what've": "what have",
                "where's": "where is",
                "who'd": "who would",
                "who'll": "who will",
                "who's": "who is",
                "won't": "will not",
                "wouldn't": "would not",
                "you'd": "you would",
                "you'll": "you will",
                "you're": "you are",
                "you've": "you have"
            }
        
        # Create pattern
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_dict.keys()) + r')\b', re.IGNORECASE)
        
        def replace(match):
            return contractions_dict[match.group(0).lower()]
        
        return pattern.sub(replace, text)
    
    def preprocess(
        self,
        text: str,
        lowercase: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_html: bool = False,
        remove_punctuation: bool = False,
        remove_digits: bool = False,
        remove_accents: bool = False,
        expand_contractions: bool = False,
        normalize_unicode: bool = False,
        normalize_whitespace: bool = True,
        remove_control_chars: bool = True,
        custom_filters: Optional[List[Callable[[str], str]]] = None
    ) -> str:
        """
        Apply multiple preprocessing steps.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_html: Remove HTML tags
            remove_punctuation: Remove punctuation
            remove_digits: Remove digits
            remove_accents: Remove accents
            expand_contractions: Expand contractions
            normalize_unicode: Normalize Unicode
            normalize_whitespace: Normalize whitespace
            remove_control_chars: Remove control characters
            custom_filters: List of custom filter functions
            
        Returns:
            Preprocessed text
        """
        if remove_html:
            text = self.remove_html_tags(text)
        
        if remove_urls:
            text = self.remove_urls(text)
        
        if remove_emails:
            text = self.remove_emails(text)
        
        if remove_control_chars:
            text = self.remove_control_characters(text)
        
        if normalize_unicode:
            text = self.normalize_unicode(text)
        
        if remove_accents:
            text = self.remove_accents(text)
        
        if expand_contractions:
            text = self.expand_contractions(text)
        
        if lowercase:
            text = self.lowercase(text)
        
        if remove_punctuation:
            text = self.remove_punctuation(text)
        
        if remove_digits:
            text = self.remove_digits(text)
        
        if normalize_whitespace:
            text = self.normalize_whitespace(text)
        
        # Apply custom filters
        if custom_filters:
            for filter_func in custom_filters:
                text = filter_func(text)
        
        return text


# ==================== TOKENIZER MERGER ====================

class TokenizerMerger:
    """Merge multiple tokenizers into one."""
    
    def __init__(self):
        """Initialize merger."""
        pass
    
    @staticmethod
    def merge_vocabularies(
        tokenizers: List[Tokenizer],
        strategy: str = 'union',
        max_vocab_size: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Merge vocabularies from multiple tokenizers.
        
        Args:
            tokenizers: List of tokenizers
            strategy: Merge strategy ('union', 'intersection', 'weighted')
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            Merged vocabulary
        """
        if strategy == 'union':
            # Union of all vocabularies
            merged_vocab = {}
            current_id = 0
            
            for tokenizer in tokenizers:
                vocab = tokenizer.get_vocab()
                for token in vocab.keys():
                    if token not in merged_vocab:
                        merged_vocab[token] = current_id
                        current_id += 1
        
        elif strategy == 'intersection':
            # Intersection of all vocabularies
            vocabs = [set(tok.get_vocab().keys()) for tok in tokenizers]
            common_tokens = set.intersection(*vocabs)
            
            merged_vocab = {token: idx for idx, token in enumerate(sorted(common_tokens))}
        
        elif strategy == 'weighted':
            # Weighted merge based on frequency
            token_counts = Counter()
            
            for tokenizer in tokenizers:
                vocab = tokenizer.get_vocab()
                # Use inverse ID as proxy for frequency (lower ID = more frequent)
                for token, token_id in vocab.items():
                    # Weight by inverse of ID (normalized)
                    weight = 1.0 / (token_id + 1)
                    token_counts[token] += weight
            
            # Take top tokens
            if max_vocab_size:
                top_tokens = [token for token, _ in token_counts.most_common(max_vocab_size)]
            else:
                top_tokens = list(token_counts.keys())
            
            merged_vocab = {token: idx for idx, token in enumerate(top_tokens)}
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Limit vocabulary size
        if max_vocab_size and len(merged_vocab) > max_vocab_size:
            # Keep most common tokens
            sorted_tokens = sorted(merged_vocab.items(), key=lambda x: x[1])[:max_vocab_size]
            merged_vocab = dict(sorted_tokens)
        
        return merged_vocab


# ==================== TOKENIZER ENSEMBLE ====================

class TokenizerEnsemble:
    """Ensemble multiple tokenizers."""
    
    def __init__(self, tokenizers: List[Tuple[str, Tokenizer, float]]):
        """
        Initialize ensemble.
        
        Args:
            tokenizers: List of (name, tokenizer, weight) tuples
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library is required")
        
        self.tokenizers = tokenizers
        
        # Normalize weights
        total_weight = sum(weight for _, _, weight in tokenizers)
        self.tokenizers = [
            (name, tok, weight / total_weight)
            for name, tok, weight in tokenizers
        ]
    
    def encode(self, text: str, strategy: str = 'vote') -> List[int]:
        """
        Encode text using ensemble.
        
        Args:
            text: Input text
            strategy: Ensemble strategy ('vote', 'weighted', 'longest', 'shortest')
            
        Returns:
            List of token IDs
        """
        # Get encodings from all tokenizers
        encodings = []
        for name, tokenizer, weight in self.tokenizers:
            encoding = tokenizer.encode(text)
            encodings.append((name, encoding.ids, weight))
        
        if strategy == 'vote':
            # Majority voting on each position
            max_len = max(len(enc) for _, enc, _ in encodings)
            result = []
            
            for pos in range(max_len):
                votes = Counter()
                for _, enc, weight in encodings:
                    if pos < len(enc):
                        votes[enc[pos]] += weight
                
                # Take most voted token
                if votes:
                    most_common = votes.most_common(1)[0][0]
                    result.append(most_common)
        
        elif strategy == 'weighted':
            # Use weighted average of token IDs (not always meaningful)
            max_len = max(len(enc) for _, enc, _ in encodings)
            result = []
            
            for pos in range(max_len):
                weighted_sum = 0
                total_weight = 0
                
                for _, enc, weight in encodings:
                    if pos < len(enc):
                        weighted_sum += enc[pos] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    result.append(int(round(weighted_sum / total_weight)))
        
        elif strategy == 'longest':
            # Use longest encoding
            longest_enc = max(encodings, key=lambda x: len(x[1]))
            result = longest_enc[1]
        
        elif strategy == 'shortest':
            # Use shortest encoding
            shortest_enc = min(encodings, key=lambda x: len(x[1]))
            result = shortest_enc[1]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return result
    
    def decode(self, token_ids: List[int], tokenizer_name: Optional[str] = None) -> str:
        """
        Decode token IDs.
        
        Args:
            token_ids: List of token IDs
            tokenizer_name: Name of specific tokenizer to use (None for first)
            
        Returns:
            Decoded text
        """
        if tokenizer_name:
            for name, tokenizer, _ in self.tokenizers:
                if name == tokenizer_name:
                    return tokenizer.decode(token_ids)
            raise ValueError(f"Tokenizer '{tokenizer_name}' not found")
        
        # Use first tokenizer by default
        return self.tokenizers[0][1].decode(token_ids)


# ==================== FORMAT CONVERSION ====================

class TokenizerConverter:
    """Convert tokenizers between different formats."""
    
    @staticmethod
    def to_huggingface_format(
        tokenizer: Tokenizer,
        output_dir: Union[str, Path],
        model_type: str = "bpe"
    ):
        """
        Convert to Hugging Face format.
        
        Args:
            tokenizer: Source tokenizer
            output_dir: Output directory
            model_type: Model type (bpe, wordpiece, unigram)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        tokenizer.save(str(output_dir / "tokenizer.json"))
        
        # Create tokenizer_config.json
        config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_type": model_type,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
        }
        
        with open(output_dir / "tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def from_vocab_file(
        vocab_file: Union[str, Path],
        tokenizer_type: str = "bpe",
        **kwargs
    ) -> Tokenizer:
        """
        Create tokenizer from vocabulary file.
        
        Args:
            vocab_file: Path to vocabulary file
            tokenizer_type: Type of tokenizer
            **kwargs: Additional arguments
            
        Returns:
            Tokenizer instance
        """
        # Implementation depends on vocab file format
        # This is a placeholder
        raise NotImplementedError("Conversion from vocab file not yet implemented")

__all__ = [
    'TextPreprocessor',
    'TokenizerMerger',
    'TokenizerEnsemble',
    'TokenizerConverter',
]


# Continuation needed for more utilities...
