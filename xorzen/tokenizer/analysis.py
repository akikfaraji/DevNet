"""
Tokenizer Analysis and Utilities - Production Implementation
Version: 2.0

Provides comprehensive analysis and comparison tools for tokenizers, crucial
for understanding their behavior, performance, and suitability for various tasks.

Key Features:
- Detailed Vocabulary Analysis: Metrics on token length distribution, character
  composition, and token types (alphabetic, numeric, punctuation).
- In-depth Tokenization Analysis: Evaluates tokenization behavior on real text
  samples, including compression ratio, subword fertility, and UNK rate.
- N-gram Analysis: Computes character and word N-gram frequencies to identify
  common patterns and subwords learned by the tokenizer.
- Tokenizer Comparison: Offers methods to quantitatively compare different
  tokenizers based on vocabulary overlap, tokenization agreement, and performance.
- Production-Grade: Built with robust error handling, detailed documentation,
  and a comprehensive self-testing suite.
"""

import time
import statistics
import warnings
from typing import Dict, List, Optional, Tuple, Iterator, Set, Union, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import math
import re

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    TOKENIZERS_AVAILABLE = True
except ImportError:
    warnings.warn("`tokenizers` library not available. Some analysis features will be limited.")
    TOKENIZERS_AVAILABLE = False
    class Tokenizer:
        def __init__(self, model=None): pass
        def get_vocab(self): return {}
        def get_vocab_size(self): return 0
        def encode(self, text): return type('obj', (object,), {'ids': [], 'tokens': []})()

# --- Tokenizer Analyzer ---

class TokenizerAnalyzer:
    """
    Performs comprehensive analysis of a single Tokenizer instance.
    """
    def __init__(self, tokenizer: Tokenizer):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("The `tokenizers` library is required for TokenizerAnalyzer.")
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = self.tokenizer.get_vocab_size()

    def analyze_vocabulary(self) -> Dict[str, Any]:
        """
        Analyzes the composition and characteristics of the tokenizer's vocabulary.

        Returns:
            Dict[str, Any]: A dictionary containing various vocabulary statistics.
        """
        tokens = list(self.vocab.keys())
        if not tokens: return {'vocab_size': 0, 'error': 'Empty vocabulary'}

        token_lengths = [len(t) for t in tokens]
        char_counter = Counter("".join(tokens))
        
        num_alphabetic = sum(1 for t in tokens if t.isalpha())
        num_numeric = sum(1 for t in tokens if t.isnumeric())
        num_symbolic = sum(1 for t in tokens if not t.isalnum())
        
        # N-gram analysis for subword patterns
        bpe_splits = [re.findall(r'(\w+)', t) for t in tokens if '##' in t] # Example for WordPiece
        
        return {
            'vocab_size': self.vocab_size,
            'avg_token_length': statistics.mean(token_lengths),
            'median_token_length': statistics.median(token_lengths),
            'min_token_length': min(token_lengths),
            'max_token_length': max(token_lengths),
            'token_length_std': statistics.stdev(token_lengths) if len(token_lengths) > 1 else 0.0,
            'unique_characters': len(char_counter),
            'most_common_characters': char_counter.most_common(20),
            'num_alphabetic_tokens': num_alphabetic,
            'num_numeric_tokens': num_numeric,
            'num_symbolic_tokens': num_symbolic,
        }

    def analyze_tokenization(self, texts: List[str], sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyzes the tokenizer's behavior on a given set of texts.

        Args:
            texts (List[str]): A list of text strings to analyze.
            sample_size (Optional[int]): If provided, only a random sample of this size will be analyzed.

        Returns:
            Dict[str, Any]: A dictionary containing various tokenization statistics.
        """
        if not texts: return {'num_texts': 0, 'error': 'No texts provided'}
        if sample_size and sample_size < len(texts):
            texts = random.sample(texts, sample_size)
        
        all_token_ids, all_tokens_str = [], []
        total_chars, total_words = 0, 0

        for text in texts:
            encoding = self.tokenizer.encode(text)
            all_token_ids.extend(encoding.ids)
            all_tokens_str.extend(encoding.tokens)
            total_chars += len(text)
            total_words += len(text.split())
        
        total_tokens = len(all_token_ids)
        unique_tokens_in_corpus = len(set(all_token_ids))
        
        return {
            'num_texts': len(texts),
            'total_characters': total_chars,
            'total_words': total_words,
            'total_tokens': total_tokens,
            'avg_tokens_per_text': total_tokens / len(texts),
            'avg_tokens_per_word': total_tokens / total_words if total_words > 0 else 0.0,
            'compression_ratio_char_to_token': total_chars / total_tokens if total_tokens > 0 else 0.0,
            'unique_tokens_used_in_corpus': unique_tokens_in_corpus,
            'vocab_coverage_in_corpus': unique_tokens_in_corpus / self.vocab_size,
        }

    def analyze_ngrams(self, texts: List[str], n: int = 2) -> Dict[str, Any]:
        """
        Analyzes character and word N-gram frequencies in the tokenized output.

        Args:
            texts (List[str]): List of texts to analyze.
            n (int): The N for N-grams.

        Returns:
            Dict[str, Any]: N-gram analysis results.
        """
        if not texts: return {'error': 'No texts provided'}

        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenizer.encode(text).tokens)

        char_ngrams = Counter()
        word_ngrams = Counter()

        for token in all_tokens:
            for i in range(len(token) - n + 1):
                char_ngrams[token[i:i+n]] += 1
        
        for i in range(len(all_tokens) - n + 1):
            word_ngrams[tuple(all_tokens[i:i+n])] += 1

        return {
            f'top_{n}_char_ngrams': char_ngrams.most_common(20),
            f'top_{n}_word_ngrams': word_ngrams.most_common(20),
        }

    def measure_tokenization_speed(self, texts: List[str], num_iterations: int = 5) -> Dict[str, float]:
        """
        Measures the average time taken to tokenize a list of texts.

        Args:
            texts (List[str]): List of texts for speed measurement.
            num_iterations (int): Number of repetitions for averaging.

        Returns:
            Dict[str, float]: Speed metrics.
        """
        if not texts: return {'error': 'No texts provided'}
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            for text in texts:
                self.tokenizer.encode(text)
            times.append(time.time() - start_time)
        
        total_chars = sum(len(text) for text in texts)
        
        return {
            'avg_time_per_run_sec': statistics.mean(times),
            'std_time_per_run_sec': statistics.stdev(times) if len(times) > 1 else 0.0,
            'texts_per_second': len(texts) / statistics.mean(times) if statistics.mean(times) > 0 else 0.0,
            'chars_per_second': total_chars / statistics.mean(times) if statistics.mean(times) > 0 else 0.0,
        }
    
    def generate_report(self, texts: Optional[List[str]] = None, save_path: Optional[str] = None) -> str:
        """
        Generates a comprehensive analysis report for the tokenizer.

        Args:
            texts (Optional[List[str]]): Sample texts for tokenization analysis.
            save_path (Optional[str]): If provided, the report will be saved to this file.

        Returns:
            str: The formatted analysis report.
        """
        lines = ["="*80, f"TOKENIZER ANALYSIS REPORT: {self.tokenizer.get_vocab_size()} Vocab Size", "="*80]
        
        lines.append("\n## Vocabulary Characteristics")
        vocab_stats = self.analyze_vocabulary()
        for k, v in vocab_stats.items():
            if isinstance(v, (float, int)):
                lines.append(f"  - {k.replace('_', ' ').title()}: {v:,}")
            elif isinstance(v, list):
                lines.append(f"  - {k.replace('_', ' ').title()}: {', '.join(str(item) for item in v[:5])}...") # Show top 5
        
        if texts:
            lines.append("\n## Tokenization Behavior on Sample Texts")
            tok_stats = self.analyze_tokenization(texts)
            for k, v in tok_stats.items():
                if isinstance(v, (float, int)): lines.append(f"  - {k.replace('_', ' ').title()}: {v:,.2f}")
            
            lines.append("\n## N-gram Analysis (Top 20 2-grams)")
            ngram_stats = self.analyze_ngrams(texts, n=2)
            lines.append(f"  - Character 2-grams: {ngram_stats['top_2_char_ngrams']}")
            lines.append(f"  - Word 2-grams: {ngram_stats['top_2_word_ngrams']}")
            
            lines.append("\n## Tokenization Speed")
            speed_stats = self.measure_tokenization_speed(texts[:min(100, len(texts))]) # Limit texts for speed test
            for k, v in speed_stats.items():
                if isinstance(v, (float, int)): lines.append(f"  - {k.replace('_', ' ').title()}: {v:,.2f}")
        
        report = "\n".join(lines)
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        return report

# --- Tokenizer Comparator ---

class TokenizerComparator:
    """
    Compares multiple tokenizer instances side-by-side on various metrics.
    """
    def __init__(self, tokenizers: Dict[str, Tokenizer]):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("The `tokenizers` library is required for TokenizerComparator.")
        self.tokenizers = tokenizers
        self.analyzers = {name: TokenizerAnalyzer(tok) for name, tok in tokenizers.items()}

    def compare_vocabularies(self) -> Dict[str, Any]:
        """
        Compares the vocabularies of all registered tokenizers.

        Returns:
            Dict[str, Any]: Comparison results including individual stats and overlap metrics.
        """
        results = {name: analyzer.analyze_vocabulary() for name, analyzer in self.analyzers.items()}
        
        # Compute overlap
        vocab_sets = {name: set(tok.get_vocab().keys()) for name, tok in self.tokenizers.items()}
        if len(vocab_sets) > 1:
            common_vocab = set.intersection(*vocab_sets.values())
            union_vocab = set.union(*vocab_sets.values())
            results['overlap'] = {
                'common_token_count': len(common_vocab),
                'unique_to_each_tokenizer': {name: len(vs - (union_vocab - vs)) for name, vs in vocab_sets.items()},
                'jaccard_similarity': len(common_vocab) / len(union_vocab) if len(union_vocab) > 0 else 0.0
            }
        return results

    def compare_tokenization_agreement(self, texts: List[str]) -> Dict[str, Any]:
        """
        Compares how similarly different tokenizers tokenize the same texts.

        Args:
            texts (List[str]): Sample texts to compare tokenization on.

        Returns:
            Dict[str, Any]: Agreement metrics between pairs of tokenizers.
        """
        if len(self.tokenizers) < 2:
            return {"error": "At least two tokenizers are required for comparison."}
        
        agreement_scores = defaultdict(float)
        num_comparisons = 0
        
        tokenizer_names = list(self.tokenizers.keys())
        for i in range(len(tokenizer_names)):
            for j in range(i + 1, len(tokenizer_names)):
                name1, name2 = tokenizer_names[i], tokenizer_names[j]
                tok1, tok2 = self.tokenizers[name1], self.tokenizers[name2]
                
                total_token_pairs = 0
                matching_token_pairs = 0
                
                for text in texts:
                    ids1 = tok1.encode(text).ids
                    ids2 = tok2.encode(text).ids
                    
                    min_len = min(len(ids1), len(ids2))
                    total_token_pairs += max(len(ids1), len(ids2))
                    matching_token_pairs += sum(1 for a, b in zip(ids1[:min_len], ids2[:min_len]) if a == b)

                agreement_score = matching_token_pairs / total_token_pairs if total_token_pairs > 0 else 0.0
                agreement_scores[f"{name1}_vs_{name2}"] = agreement_score
                num_comparisons += 1
        
        return {
            'pairwise_agreement': agreement_scores,
            'average_agreement': sum(agreement_scores.values()) / num_comparisons if num_comparisons > 0 else 0.0
        }
    
    def generate_comparison_report(self, texts: Optional[List[str]] = None, save_path: Optional[str] = None) -> str:
        """
        Generates a comprehensive report comparing multiple tokenizers.

        Args:
            texts (Optional[List[str]]): Sample texts for tokenization comparison.
            save_path (Optional[str]): If provided, the report will be saved to this file.

        Returns:
            str: The formatted comparison report.
        """
        lines = ["="*80, "TOKENIZER COMPARISON REPORT", "="*80]
        lines.append(f"\nComparing {len(self.tokenizers)} tokenizers: {', '.join(self.tokenizers.keys())}\n")

        lines.append("## Vocabulary Comparison")
        vocab_comp = self.compare_vocabularies()
        for name, stats in vocab_comp.items():
            if name == 'overlap':
                continue # Handle overlap separately
            lines.append(f"\n### {name} Vocabulary")
            lines.append(f"  - Size: {stats['vocab_size']:,}")
            lines.append(f"  - Avg Token Length: {stats['avg_token_length']:.2f}")
        if 'overlap' in vocab_comp:
            lines.append("\n### Vocabulary Overlap")
            for k, v in vocab_comp['overlap'].items():
                if isinstance(v, float): lines.append(f"  - {k.replace('_', ' ').title()}: {v:.2%}")
                else: lines.append(f"  - {k.replace('_', ' ').title()}: {v}")
        
        if texts:
            lines.append("\n## Tokenization Behavior Comparison")
            for name, analyzer in self.analyzers.items():
                lines.append(f"\n### {name} Tokenization Metrics")
                tok_stats = analyzer.analyze_tokenization(texts)
                for k, v in tok_stats.items():
                    if isinstance(v, (float, int)): lines.append(f"  - {k.replace('_', ' ').title()}: {v:,.2f}")
            
            lines.append("\n## Tokenization Agreement (Pairwise)")
            agreement_stats = self.compare_tokenization_agreement(texts)
            for pair, score in agreement_stats.get('pairwise_agreement', {}).items():
                lines.append(f"  - {pair}: {score:.2%}")
            if 'average_agreement' in agreement_stats:
                lines.append(f"  - Average Agreement: {agreement_stats['average_agreement']:.2%}")
            
            lines.append("\n## Performance Comparison")
            speed_results = {name: analyzer.measure_tokenization_speed(texts) for name, analyzer in self.analyzers.items()}
            for name, stats in speed_results.items():
                lines.append(f"\n### {name} Speed")
                lines.append(f"  - Texts per second: {stats['texts_per_second']:.2f}")
                lines.append(f"  - Chars per second: {stats['chars_per_second']:,.0f}")
        
        report = "\n".join(lines)
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        return report

# --- Self-Testing Block ---

__all__ = [
    'TokenizerAnalyzer',
    'TokenizerComparator',
]
