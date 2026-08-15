"""
xorzen Tokenizer Adapter and Manager - Production Implementation
Version: 2.0

Encapsulates all interactions with the `tokenizers` library and manages multiple
tokenizer instances within the xorzen framework.

Key Features:
- Multi-Tokenizer Management: Can train, load, and manage multiple tokenizer
  configurations (e.g., a "legacy" BPE and a new "fast" BPE) under different names.
- Active Tokenizer Switching: Allows setting an "active" tokenizer for encoding and
  decoding operations.
- Integrated Training Methods: Provides methods to launch different training
  scripts, including the user-provided "fast BPE" trainer, using a clean
  subprocess-based approach.
- Built-in Analysis: Integrates the `TokenizerAnalyzer` to provide detailed
  reports and comparisons between managed tokenizers.
- Production-Grade: Includes robust error handling, clear documentation, and a
  comprehensive self-testing suite that demonstrates all core functionalities.
"""

import os
import json
import warnings
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any

try:
    from tokenizers import Tokenizer, ByteLevelBPETokenizer
    from .analysis import TokenizerAnalyzer
    from .trainer import train_tokenizer as xorzen_train_tokenizer # CORRECTED IMPORT LOCATION
except (ImportError, ModuleNotFoundError):
    warnings.warn("Could not import from local framework. Using dummy classes for standalone testing.")
    # Define dummy classes if running standalone
    class Tokenizer:
        def __init__(self, model=None): pass
        def train_from_iterator(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
        def encode(self, *args, **kwargs): return self
        @property
        def ids(self): return []
        @staticmethod
        def from_file(path): return Tokenizer()
        def get_vocab_size(self): return 0
    class ByteLevelBPETokenizer(Tokenizer): pass
    class TokenizerAnalyzer:
        def __init__(self, tokenizer): pass
        def generate_report(self, texts): return "Dummy Analysis Report"


class XORZENXTokenizerAdapter:
    """
    Manages the lifecycle of multiple tokenizers, including training, loading,
    switching, and analysis.
    """
    
    SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]

    def __init__(self, default_tokenizer_path: Optional[str] = None):
        """
        Initializes the tokenizer adapter.

        Args:
            default_tokenizer_path (Optional[str]): Path to a tokenizer to load
                                                     and set as active by default.
        """
        self.tokenizers: Dict[str, Tokenizer] = {}
        self.active_tokenizer_name: Optional[str] = None
        
        if default_tokenizer_path:
            try:
                name = Path(default_tokenizer_path).stem
                self.load_tokenizer(name, default_tokenizer_path)
                self.activate_tokenizer(name)
            except Exception as e:
                warnings.warn(f"Failed to load default tokenizer from '{default_tokenizer_path}': {e}")

    def load_tokenizer(self, name: str, path: str) -> None:
        """
        Loads a tokenizer from a file and registers it under a given name.

        Args:
            name (str): The name to assign to the loaded tokenizer.
            path (str): The file path to the tokenizer.json file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        try:
            tokenizer = Tokenizer.from_file(path)
            # Store the path for workers in parallel processing scenarios
            setattr(tokenizer, 'path', path) 
            self.tokenizers[name] = tokenizer
            print(f"Successfully loaded tokenizer '{name}' from {path}")
            if self.active_tokenizer_name is None:
                self.activate_tokenizer(name)
        except Exception as e:
            raise IOError(f"Failed to load or parse tokenizer file at {path}: {e}")

    def activate_tokenizer(self, name: str) -> None:
        """Sets a loaded tokenizer as the active one for encoding/decoding."""
        if name not in self.tokenizers:
            raise ValueError(f"Tokenizer '{name}' not found. Available tokenizers: {self.list_tokenizers()}")
        self.active_tokenizer_name = name
        print(f"Tokenizer '{name}' is now active.")

    def list_tokenizers(self) -> List[str]:
        """Returns a list of all loaded tokenizer names."""
        return list(self.tokenizers.keys())

    @property
    def tokenizer(self) -> Tokenizer:
        """Returns the currently active tokenizer instance."""
        if self.active_tokenizer_name is None or self.active_tokenizer_name not in self.tokenizers:
            raise RuntimeError("No active tokenizer set. Please load and activate a tokenizer first.")
        return self.tokenizers[self.active_tokenizer_name]

    def train_legacy(
        self,
        name: str,
        files: List[str], # Changed from corpus_iterator
        vocab_size: int = 5000,
        min_frequency: int = 2,
        save_path: Optional[str] = None
    ) -> None:
        """
        Trains a legacy ByteLevelBPE tokenizer.

        Args:
            name (str): The name to assign to the loaded tokenizer.
            files (List[str]): A list of file paths for training.
            vocab_size (int): The target vocabulary size.
            min_frequency (int): The minimum frequency for a token to be included.
            save_path (Optional[str]): Path to save the trained tokenizer file.
        """
        print(f"Training legacy BPE tokenizer '{name}' with vocab_size={vocab_size}...")
        bpe_tokenizer = ByteLevelBPETokenizer(lowercase=True, add_prefix_space=True)
        bpe_tokenizer.train( # Changed to train (not train_from_iterator)
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.SPECIAL_TOKENS
        )
        
        
        # This returns a different type, so we need to get the underlying Tokenizer
        # and attach its path for consistency if we want to use it in workers.
        # This is a bit hacky, but the tokenizers library returns different objects
        # depending on whether it's loaded from file or created.
        temp_path = Path(save_path or (Path(os.getcwd()) / f"{name}_temp.json"))
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        bpe_tokenizer.save(str(temp_path))
        tokenizer = Tokenizer.from_file(str(temp_path))
        if not save_path: temp_path.unlink() # Clean up temp file
        
        setattr(tokenizer, 'path', str(save_path) if save_path else None)
        self.tokenizers[name] = tokenizer
        
        if save_path:
            self.save(name, save_path)
        
        print(f"Tokenizer '{name}' trained. Vocab size: {self.get_vocab_size(name)}")

    def train_fast_bpe(
        self,
        name: str,
        output_dir: str,
        data_dirs: List[str],
        vocab_size: int = 65536
    ) -> None:
        """
        Trains a new 'fast' BPE tokenizer by directly calling the modern training function.

        Args:
            name (str): Name to register for the new tokenizer.
            output_dir (str): Directory to save the new tokenizer files.
            data_dirs (List[str]): List of directories containing training data.
            vocab_size (int): The target vocabulary size.
        """
        print(f"--- Starting Fast BPE Tokenizer Training for '{name}' ---")
        
        try:
            # Directly call the modern tokenizer training function
            trained_tokenizer = xorzen_train_tokenizer(
                data_paths=data_dirs,
                output_dir=output_dir,
                vocab_size=vocab_size,
                min_frequency=2, # Using default min_frequency
                special_tokens=self.SPECIAL_TOKENS # Using adapter's default special tokens
            )
            
            print("--- Tokenizer Training Finished ---")
            
            # Load the newly trained tokenizer
            tokenizer_path = Path(output_dir) / "tokenizer.json"
            self.load_tokenizer(name, str(tokenizer_path))
            self.activate_tokenizer(name)
            
        except Exception as e:
            print(f"An error occurred during fast BPE training: {e}")
            raise

    def save(self, name: str, path: str):
        """Saves a specified tokenizer to a file."""
        if name not in self.tokenizers:
            raise ValueError(f"Tokenizer '{name}' not found.")
        
        tokenizer_to_save = self.tokenizers[name]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tokenizer_to_save.save(path)
        print(f"Tokenizer '{name}' saved to {path}")

    def analyze(self, name: str, sample_texts: List[str]) -> str:
        """Analyzes a tokenizer and returns a report."""
        if name not in self.tokenizers:
            raise ValueError(f"Tokenizer '{name}' not found.")
        
        analyzer = TokenizerAnalyzer(self.tokenizers[name])
        return analyzer.generate_report(sample_texts)

    # --- Passthrough methods for the active tokenizer ---
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self, name: Optional[str] = None) -> int:
        if name:
            if name not in self.tokenizers:
                raise ValueError(f"Tokenizer '{name}' not found.")
            return self.tokenizers[name].get_vocab_size()
        return self.tokenizer.get_vocab_size()

# --- Self-Testing Block ---

if __name__ == '__main__':
    print("="*80, "\nXORZENX Tokenizer Adapter: Self-Test and Demonstration\n", "="*80)
    
    # --- 1. Setup Test Environment ---
    TEST_ROOT = Path("./_test_tokenizer_adapter")
    if TEST_ROOT.exists():
        import shutil
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir()
    
    CORPUS_DIR = TEST_ROOT / "corpus"
    CORPUS_DIR.mkdir()
    LEGACY_TOKENIZER_PATH = str(TEST_ROOT / "legacy_tokenizer.json")
    FAST_TOKENIZER_OUTPUT_DIR = str(TEST_ROOT / "fast_bpe_65k")
    Path(FAST_TOKENIZER_OUTPUT_DIR).mkdir(exist_ok=True)

    print(f"✓ Created test directory: {TEST_ROOT.resolve()}")

    # --- 2. Create Dummy Corpus ---
    dummy_corpus_content = "This is a sample corpus for testing tokenizers. " * 100
    for i in range(5):
        (CORPUS_DIR / f"corpus_{i}.txt").write_text(dummy_corpus_content.replace("sample", f"sample{i}"))
    print("✓ Created dummy corpus files.")
    
    # --- 3. Initialize Adapter ---
    adapter = XORZENXTokenizerAdapter()
    print("\n--- Testing Legacy BPE Training ---")
    
    try:
        adapter.train_legacy(
            name="legacy_bpe",
            files=[str(f) for f in CORPUS_DIR.glob("*.txt")], # Pass list of file paths
            vocab_size=1000,
            save_path=LEGACY_TOKENIZER_PATH
        )
        assert "legacy_bpe" in adapter.list_tokenizers()
        assert Path(LEGACY_TOKENIZER_PATH).exists()
        print("✓ Legacy training successful.")
    except Exception as e:
        print("✗ FAILED: Legacy training failed.")
        traceback.print_exc()
        exit(1)

    # --- 4. Prepare and Test Fast BPE Trainer Integration ---
    # Temporarily modify fast_bpe_trainer.py for test runner compatibility
    fast_trainer_script_path = Path(__file__).parent / "fast_bpe_trainer.py"
    original_fast_trainer_content = fast_trainer_script_path.read_text()
    
    # Define the necessary functions for the temporary modification
    fast_bpe_trainer_mod_header = """
def train_fast_bpe_modified(data_dirs: List[str], output_dir: str, vocab_size: int):
    # This is a modified version of the original train_fast_bpe to accept arguments
    # and not use global variables for paths.
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    from tokenizers.normalizers import NFKC, Lowercase, Sequence
    from pathlib import Path
    import os
    import json
    import random # for dummy corpus

    vocab_size = vocab_size
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|user|>", "<|assistant|>", "<|system|>"
        ],
        show_progress=False # Disable progress for tests
    )

    files = []
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.is_dir():
            files.extend([str(f) for f in path.rglob("*.txt")]) # Simplified for test
        elif path.is_file(): # For direct file paths
            files.append(str(path))
    
    if not files:
        # Generate a dummy file if no files found for the test runner's context
        dummy_file = output_dir_path / "dummy_corpus_for_test.txt"
        dummy_file.write_text("dummy content " * 100)
        files = [str(dummy_file)]

    tokenizer.train(files, trainer)

    # Post-processor (simplified for test)
    try:
        im_start_id = tokenizer.token_to_id("<|im_start|>")
        im_end_id = tokenizer.token_to_id("<|im_end|>")
        if im_start_id is not None and im_end_id is not None:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="<|im_start|>user\n$0<|im_end|>\n<|im_start|>assistant\n",
                pair="<|im_start|>user\n$0<|im_end|>\n<|im_start|>assistant\n$1<|im_end|>\n",
                special_tokens=[("<|im_start|>", im_start_id), ("<|im_end|>", im_end_id)],
            )
    except Exception: pass # Ignore if special tokens not in vocab
    
    tokenizer.save(str(output_dir_path / "tokenizer.json"))
    tokenizer.model.save(str(output_dir_path)) # This saves the model.json and vocab.json
    
    # Clean up dummy file if created by test runner
    if 'dummy_file' in locals() and dummy_file.exists():
        dummy_file.unlink()

def main_cli_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    train_fast_bpe_modified(
        data_dirs=config_data['data_dirs'],
        output_dir=config_data['output_dir'],
        vocab_size=config_data['vocab_size']
    )

if __name__ == '__main__':
    main_cli_entry()
"""
    
    # Overwrite the trainer script for the test run
    fast_trainer_script_path.write_text(fast_bpe_trainer_mod_header)

    print("\n--- Testing Fast BPE Training Integration ---")
    try:
        FAST_TOKENIZER_OUTPUT_DIR = TEST_ROOT / "fast_bpe"
        FAST_TOKENIZER_OUTPUT_DIR.mkdir(exist_ok=True)
        
        adapter.train_fast_bpe(
            name="fast_bpe",
            output_dir=str(FAST_TOKENIZER_OUTPUT_DIR),
            data_dirs=[str(CORPUS_DIR)],
            vocab_size=1200 # Use a different vocab size
        )
        assert "fast_bpe" in adapter.list_tokenizers()
        assert (FAST_TOKENIZER_OUTPUT_DIR / "tokenizer.json").exists()
        print("✓ Fast BPE training integration successful.")
    except Exception as e:
        print(f"✗ FAILED: Fast BPE training integration failed.")
        traceback.print_exc()
        all_tests_passed = False
    finally:
        # Restore original script content
        fast_trainer_script_path.write_text(original_fast_trainer_content)
        # Clean up temp config file generated by adapter.train_fast_bpe
        temp_config_file = Path(FAST_TOKENIZER_OUTPUT_DIR) / "fast_bpe_trainer_config.json"
        if temp_config_file.exists():
            temp_config_file.unlink() # Clean up temp config file


    # --- 5. Test Multi-Tokenizer Management ---
    print("\n--- Testing Multi-Tokenizer Management ---")
    assert len(adapter.list_tokenizers()) == 2
    
    # Encode with legacy
    adapter.activate_tokenizer("legacy_bpe")
    encoded_legacy = adapter.encode("This is a sample for testing.")
    
    # Encode with fast
    adapter.activate_tokenizer("fast_bpe")
    encoded_fast = adapter.encode("This is a sample for testing.")
    
    print(f"  - Legacy BPE encoded: {encoded_legacy}")
    print(f"  - Fast BPE encoded:   {encoded_fast}")
    assert encoded_legacy != encoded_fast # They should be different due to different training
    print("✓ Successfully encoded with different active tokenizers.")

    # --- 6. Test Analysis Integration ---
    print("\n--- Testing Analysis Integration ---")
    try:
        report = adapter.analyze("legacy_bpe", sample_texts=[dummy_corpus_content])
        assert "Tokenizer Analysis Report" in report
        assert "Vocabulary Size: 1000" in report
        print("✓ Analysis report generated successfully.")
        # print(report) # Optionally print the full report
    except Exception as e:
        print("✗ FAILED: Analysis integration test failed.")
        traceback.print_exc()
        all_tests_passed = False

    # --- Final Cleanup ---
    print("\n--- Cleaning up test directory... ---")
    try:
        import shutil
        shutil.rmtree(TEST_ROOT)
        print("✓ Test directory removed.")
    except Exception as e:
        print(f"Warning: Failed to clean up test directory: {e}")

    print("\n" + "="*80)
    print("FINAL STATUS:", "✓ ALL TOKENIZER ADAPTER TESTS PASSED" if all_tests_passed else "✗ SOME TOKENIZER ADAPTER TESTS FAILED")
    print("="*80)

__all__ = ['XORZENXTokenizerAdapter']

