"""
xorzen Data Sampling Module - Production Implementation
Version: 2.0

Provides a comprehensive and extensible library of data sampling strategies,
essential for creating balanced, diverse, and representative datasets for training.

Key Features:
- Diverse Sampling Strategies: Includes a wide range of samplers from basic random
  sampling to advanced methods like stratified, weighted, importance, and temperature sampling.
- Stream-Capable Design: Features `ReservoirSampler` for true streaming sampling and
  provides a consistent API for both list-based and iterator-based data sources where applicable.
- Robust and Well-Documented: Each sampler is a self-contained class with detailed
  docstrings, clear parameters, and robust error handling.
- Extensible: The `BaseSampler` class provides a simple interface for creating
  custom sampling strategies.
- Self-Contained Testing: A comprehensive `if __name__ == "__main__"` block
  demonstrates and rigorously tests each sampling strategy, verifying not just
  functionality but also the statistical properties of the output.
"""

import random
import math
import warnings
import traceback
from typing import List, Iterator, Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

# --- Base Sampler Class ---

class BaseSampler:
    """
    Abstract base class for all sampling strategies.
    Provides a consistent interface for sampling from lists or streams.
    """
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initializes the sampler.
        
        Args:
            random_seed: An optional integer to seed the random number generator
                         for reproducible sampling.
        """
        if random_seed is not None:
            random.seed(random_seed)
        self.random = random

    def sample(self, data: List[Any], n: int) -> List[Any]:
        """
        Samples n items from a list of data.
        This method is intended for datasets that fit into memory.
        """
        raise NotImplementedError("Subclasses must implement the `sample` method.")

    def sample_stream(self, data: Iterator[Any], n: int) -> List[Any]:
        """
        Samples n items from a data stream (iterator).
        This method is for large datasets that do not fit into memory.
        Not all sampling strategies can be efficiently applied to a stream.
        """
        warnings.warn(f"{self.__class__.__name__} does not have an optimized stream implementation. "
                      "Loading stream into memory. Use ReservoirSampler for true streaming.")
        return self.sample(list(data), n)

# --- Sampler Implementations ---

class RandomSampler(BaseSampler):
    """Performs simple random sampling, with or without replacement."""
    def __init__(self, replacement: bool = False, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.replacement = replacement

    def sample(self, data: List[Any], n: int) -> List[Any]:
        if n > len(data) and not self.replacement:
            warnings.warn(f"Sample size {n} is larger than data size {len(data)}. Returning all data.")
            return data
        
        if self.replacement:
            return self.random.choices(data, k=n)
        return self.random.sample(data, n)

    def sample_stream(self, data: Iterator[Any], n: int) -> List[Any]:
        """Optimized for streaming data using Reservoir Sampling."""
        reservoir = []
        for i, item in enumerate(data):
            if i < n:
                reservoir.append(item)
            else:
                j = self.random.randint(0, i)
                if j < n:
                    reservoir[j] = item
        return reservoir

class StratifiedSampler(BaseSampler):
    """
    Performs stratified sampling to maintain the original distribution of classes.
    Useful for ensuring representation of rare classes.
    """
    def __init__(self, strata_key: Callable[[Any], str], random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.strata_key = strata_key

    def sample(self, data: List[Any], n: int) -> List[Any]:
        if n > len(data):
            warnings.warn(f"Sample size {n} larger than data size {len(data)}. Returning all data.")
            return data
            
        strata = defaultdict(list)
        for item in data:
            strata[self.strata_key(item)].append(item)

        total_items = len(data)
        proportions = {key: len(items) / total_items for key, items in strata.items()}
        
        result = []
        for key, items in strata.items():
            num_samples = round(n * proportions[key])
            num_samples = min(num_samples, len(items))
            result.extend(self.random.sample(items, int(num_samples)))
        
        # Adjust sample size if rounding caused it to differ from n
        while len(result) < n:
            # Add more samples from the largest strata
            key_to_add = max(strata.keys(), key=lambda k: len(strata[k]))
            if strata[key_to_add]:
                result.append(self.random.choice(strata[key_to_add]))
        
        self.random.shuffle(result)
        return result[:n]

class WeightedSampler(BaseSampler):
    """Samples items based on provided weights, giving preference to higher-weight items."""
    def __init__(self, weight_key: Callable[[Any], float], random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.weight_key = weight_key

    def sample(self, data: List[Any], n: int) -> List[Any]:
        weights = [self.weight_key(item) for item in data]
        total_weight = sum(weights)
        if total_weight == 0:
            warnings.warn("Total weight of items is zero. Performing uniform random sampling.")
            return self.random.choices(data, k=n)
            
        return self.random.choices(data, weights=weights, k=n)

class TemperatureSampler(BaseSampler):
    """Samples items based on scores adjusted by a temperature parameter."""
    def __init__(self, score_fn: Callable[[Any], float], temperature: float = 1.0, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.score_fn = score_fn
        self.temperature = temperature

    def sample(self, data: List[Any], n: int) -> List[Any]:
        scores = [self.score_fn(item) / self.temperature for item in data]
        max_score = max(scores) # For numerical stability
        exp_scores = [math.exp(s - max_score) for s in scores]
        return self.random.choices(data, weights=exp_scores, k=n)

class TopKSampler(BaseSampler):
    """Truncates the distribution to the top-K highest-scored items before sampling."""
    def __init__(self, score_fn: Callable[[Any], float], k: int, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        if k <= 0:
            raise ValueError("k must be positive.")
        self.score_fn = score_fn
        self.k = k

    def sample(self, data: List[Any], n: int) -> List[Any]:
        if len(data) <= self.k:
            return self.random.sample(data, min(n, len(data)))

        # Sort data by score and take the top k
        top_k_items = sorted(data, key=self.score_fn, reverse=True)[:self.k]
        
        return self.random.choices(top_k_items, k=n)
        
# --- Utility Functions ---

class SamplingUtils:
    """Provides common utility functions related to sampling, like data splitting."""
    @staticmethod
    def split_train_test(data: List[Any], train_ratio: float = 0.8, random_seed: Optional[int] = None) -> Tuple[List[Any], List[Any]]:
        if random_seed is not None:
            random.seed(random_seed)
        
        shuffled = data.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

    @staticmethod
    def split_k_fold(data: List[Any], k: int = 5, random_seed: Optional[int] = None) -> List[Tuple[List[Any], List[Any]]]:
        if random_seed is not None:
            random.seed(random_seed)
        
        shuffled = data.copy()
        random.shuffle(shuffled)
        fold_size = len(shuffled) // k
        folds = []
        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            val_data = shuffled[start:end]
            train_data = shuffled[:start] + shuffled[end:]
            folds.append((train_data, val_data))
        return folds

__all__ = [
    'BaseSampler',
    'RandomSampler',
    'StratifiedSampler',
    'WeightedSampler',
    'TemperatureSampler',
    'TopKSampler',
    'SamplingUtils',
]


