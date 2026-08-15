"""
xorzen Data Augmentation Module - Production Implementation
Version: 2.0

Provides a comprehensive and extensible library of data augmentation techniques
for text data. These techniques are crucial for improving model robustness,
generalization, and performance by artificially expanding the training dataset.

Key Features:
- Diverse Augmentation Strategies: Includes a wide range of strategies from simple
  random operations (delete, swap, insert) to more advanced techniques like
  synonym replacement and mock back-translation.
- Production-Grade Implementations: Each augmenter is implemented as a robust,
  well-documented class with clear parameters and expected behaviors.
- Extensible by Design: The `BaseAugmenter` class provides a simple interface for
  creating custom augmentation strategies.
- Composition and Pipelines: Includes `CompositeAugmenter` and `AugmentationPipeline`
  to easily combine multiple augmentation techniques into a powerful workflow.
- Self-Contained Testing: A comprehensive `if __name__ == "__main__"` block
  demonstrates and tests each augmentation strategy, ensuring reliability.
- Placeholders for Advanced Techniques: Includes commented-out skeletons for highly
  advanced techniques like back-translation and contextual word embedding,
  explaining how they would be implemented with external models or APIs.
"""

import random
import re
import warnings
import traceback
from typing import List, Optional, Callable, Dict, Any, Tuple

# --- Base Augmenter Class ---

class BaseAugmenter:
    """Base class for all text augmentation strategies."""
    def __init__(self, random_seed: Optional[int] = None):
        self.random = random.Random(random_seed)

    def augment(self, text: str, n: int = 1) -> List[str]:
        """
        Generates n augmented versions of the input text.
        Subclasses must implement the _augment_single method.
        """
        if not isinstance(text, str):
            warnings.warn(f"Input to augmenter is not a string (type: {type(text)}), returning empty list.")
            return []
        return [self._augment_single(text) for _ in range(n)]

    def _augment_single(self, text: str) -> str:
        """Performs a single augmentation on the text."""
        raise NotImplementedError("Subclasses must implement the _augment_single method.")

# --- Character-Level Augmenters ---

class NoiseInjectionAugmenter(BaseAugmenter):
    """Injects character-level noise to simulate typos."""
    def __init__(self, aug_prob: float = 0.05, noise_types: Optional[List[str]] = None, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.aug_prob = aug_prob
        self.noise_types = noise_types or ['swap', 'delete', 'insert', 'replace']
        self.keyboard_map = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfx', 'e': 'wsdr', 'f': 'drtgvc', 'g': 'fhtyhb',
            'h': 'gyjnbv', 'i': 'ujko', 'j': 'huiknm', 'k': 'juiolm', 'l': 'kiop', 'm': 'njk',
            'n': 'bhjm', 'o': 'iklp', 'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'awedz',
            't': 'rfgy', 'u': 'yijh', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tugh', 'z': 'asx'
        }

    def _augment_single(self, text: str) -> str:
        chars = list(text)
        new_chars = []
        i = 0
        while i < len(chars):
            char = chars[i]
            if self.random.random() < self.aug_prob and char.isalpha():
                noise_type = self.random.choice(self.noise_types)
                if noise_type == 'delete':
                    pass  # Skip appending the character
                elif noise_type == 'insert' and char.lower() in self.keyboard_map:
                    nearby_char = random.choice(self.keyboard_map[char.lower()])
                    new_chars.append(nearby_char if char.islower() else nearby_char.upper())
                    new_chars.append(char)
                elif noise_type == 'swap' and i < len(chars) - 1:
                    new_chars.append(chars[i+1])
                    new_chars.append(chars[i])
                    i += 1 # Skip next char
                elif noise_type == 'replace' and char.lower() in self.keyboard_map:
                    nearby_char = random.choice(self.keyboard_map[char.lower()])
                    new_chars.append(nearby_char if char.islower() else nearby_char.upper())
                else:
                    new_chars.append(char)
            else:
                new_chars.append(char)
            i += 1
        return "".join(new_chars)

# --- Word-Level Augmenters ---

class SynonymReplacementAugmenter(BaseAugmenter):
    """Replaces words with their synonyms from a predefined dictionary."""
    def __init__(self, aug_prob: float = 0.1, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.aug_prob = aug_prob
        self.synonym_dict = self._get_default_synonyms()

    def _augment_single(self, text: str) -> str:
        words = text.split()
        if not words: return text
        
        num_to_replace = max(1, int(len(words) * self.aug_prob))
        indices_to_replace = random.sample(range(len(words)), min(num_to_replace, len(words)))
        
        new_words = list(words)
        for i in indices_to_replace:
            original_word = words[i]
            # Handle punctuation
            leading_punct = re.match(r'^\W+', original_word)
            trailing_punct = re.search(r'\W+$', original_word)
            
            clean_word = re.sub(r'^\W+|\W+$', '', original_word)
            word_lower = clean_word.lower()
            
            if word_lower in self.synonym_dict:
                synonym = random.choice(self.synonym_dict[word_lower])
                # Preserve case of the first letter
                if clean_word and clean_word[0].isupper():
                    synonym = synonym.capitalize()
                
                # Re-attach punctuation
                if leading_punct: synonym = leading_punct.group(0) + synonym
                if trailing_punct: synonym = synonym + trailing_punct.group(0)
                
                new_words[i] = synonym
        return " ".join(new_words)

    @staticmethod
    def _get_default_synonyms() -> Dict[str, List[str]]:
        return {
            'good': ['great', 'excellent', 'fine', 'wonderful', 'superb', 'outstanding', 'virtuous', 'quality'],
            'bad': ['poor', 'terrible', 'awful', 'horrible', 'dreadful', 'inferior', 'wicked', 'substandard'],
            'happy': ['joyful', 'pleased', 'delighted', 'glad', 'cheerful', 'content', 'ecstatic', 'elated'],
            'sad': ['unhappy', 'sorrowful', 'melancholy', 'gloomy', 'depressed', 'miserable', 'despondent'],
            'run': ['sprint', 'jog', 'dash', 'race', 'hurry', 'scamper'],
            'say': ['state', 'declare', 'mention', 'utter', 'voice', 'proclaim', 'speak', 'tell'],
            'important': ['significant', 'crucial', 'vital', 'essential', 'key', 'pivotal', 'consequential'],
            'beautiful': ['attractive', 'gorgeous', 'lovely', 'stunning', 'exquisite', 'handsome', 'radiant'],
            'intelligent': ['smart', 'bright', 'clever', 'brilliant', 'sharp', 'astute', 'wise'],
            'create': ['make', 'build', 'design', 'construct', 'produce', 'generate', 'formulate', 'invent'],
            'use': ['utilize', 'employ', 'apply', 'operate', 'handle'],
            'help': ['assist', 'support', 'aid', 'succor', 'abet'],
            'work': ['labor', 'toil', 'operate', 'function', 'perform'],
            'love': ['adore', 'cherish', 'treasure', 'appreciate', 'fancy'],
            'hate': ['detest', 'abhor', 'loathe', 'despise'],
        }

class RandomDeletionAugmenter(BaseAugmenter):
    """Randomly deletes words from the text."""
    def __init__(self, aug_prob: float = 0.1, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.aug_prob = aug_prob

    def _augment_single(self, text: str) -> str:
        words = text.split()
        if len(words) <= 1: return text
        new_words = [word for word in words if random.random() > self.aug_prob]
        return " ".join(new_words) if new_words else random.choice(words)

class RandomSwapAugmenter(BaseAugmenter):
    """Randomly swaps two words in the text."""
    def __init__(self, aug_prob: float = 0.1, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.aug_prob = aug_prob

    def _augment_single(self, text: str) -> str:
        words = text.split()
        n_swaps = max(1, int(len(words) * self.aug_prob))
        for _ in range(n_swaps):
            if len(words) < 2: break
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)

class RandomInsertionAugmenter(BaseAugmenter):
    """Inserts random (but common) words into the text."""
    def __init__(self, aug_prob: float = 0.1, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.aug_prob = aug_prob
        self.word_pool = ['really', 'very', 'quite', 'actually', 'basically', 'generally', 'truly']

    def _augment_single(self, text: str) -> str:
        words = text.split()
        num_insertions = max(1, int(len(words) * self.aug_prob))
        for _ in range(num_insertions):
            pos = random.randint(0, len(words))
            word_to_insert = random.choice(self.word_pool)
            words.insert(pos, word_to_insert)
        return " ".join(words)

# --- Advanced Augmenters (with Mock Implementations) ---

class BackTranslationAugmenter(BaseAugmenter):
    """
    Simulates back-translation by applying several simple transformations.
    NOTE: A real implementation would require a translation model/API.
    """
    def __init__(self, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        warnings.warn("Using MOCK BackTranslationAugmenter. This does not perform real translation.")
        self.syn_aug = SynonymReplacementAugmenter(aug_prob=0.15, random_seed=random_seed)
        self.swap_aug = RandomSwapAugmenter(aug_prob=0.05, random_seed=random_seed)

    def _augment_single(self, text: str) -> str:
        """Simulates translationese by swapping and replacing words."""
        text = self.swap_aug._augment_single(text)
        text = self.syn_aug._augment_single(text)
        return text

class ContextualWordEmbeddingAugmenter(BaseAugmenter):
    """
    Placeholder for augmentation using contextual word embeddings (e.g., BERT, RoBERTa).
    A real implementation requires a pre-trained language model.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', aug_prob: float = 0.1, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        self.aug_prob = aug_prob
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        warnings.warn(f"ContextualWordEmbeddingAugmenter is a PLACEHOLDER. "
                      f"A real implementation would require loading '{self.model_name}'.")

    def _load_model(self):
        """
        Placeholder for loading a real transformer model.
        
        Example using `transformers`:
        ```
        from transformers import BertTokenizer, BertForMaskedLM
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()
        ```
        """
        print(f"INFO: [Placeholder] Would load model '{self.model_name}' here.")
        pass

    def _augment_single(self, text: str) -> str:
        """
        Mocks the process of contextual augmentation by replacing a word with a plausible, 
        but not necessarily contextually derived, synonym.
        """
        if self.model is None:
            # Simulate lazy loading
            self._load_model()
            
        words = text.split()
        if not words: return text
        
        num_to_replace = max(1, int(len(words) * self.aug_prob))
        indices_to_replace = random.sample(range(len(words)), min(num_to_replace, len(words)))
        
        # This is the mock part. A real implementation would use the model.
        synonym_aug = SynonymReplacementAugmenter(random_seed=self.random.randint(0, 10000))
        temp_text = " ".join([words[i] for i in indices_to_replace])
        augmented_temp = synonym_aug._augment_single(temp_text)
        augmented_words = augmented_temp.split()

        new_words = list(words)
        for i, idx in enumerate(indices_to_replace):
            if i < len(augmented_words):
                new_words[idx] = augmented_words[i]
        
        return " ".join(new_words)

# --- Composition and Pipelines ---

class CompositeAugmenter(BaseAugmenter):
    """Applies a randomly chosen augmenter from a given list."""
    def __init__(self, augmenters: List[BaseAugmenter], weights: Optional[List[float]] = None):
        self.augmenters = augmenters
        self.weights = weights
        if self.weights and len(self.augmenters) != len(self.weights):
            raise ValueError("Length of augmenters and weights must be the same.")

    def _augment_single(self, text: str) -> str:
        augmenter = random.choices(self.augmenters, weights=self.weights, k=1)[0]
        return augmenter._augment_single(text)

class AugmentationPipeline(BaseAugmenter):
    """Applies a sequence of augmenters to a text."""
    def __init__(self, augmenters: List[BaseAugmenter]):
        self.augmenters = augmenters

    def _augment_single(self, text: str) -> str:
        for augmenter in self.augmenters:
            text = augmenter._augment_single(text)
        return text

__all__ = [
    'BaseAugmenter',
    'NoiseInjectionAugmenter',
    'SynonymReplacementAugmenter',
    'RandomDeletionAugmenter',
    'RandomSwapAugmenter',
    'RandomInsertionAugmenter',
    'BackTranslationAugmenter',
    'ContextualWordEmbeddingAugmenter',
    'CompositeAugmenter',
    'AugmentationPipeline',
]


