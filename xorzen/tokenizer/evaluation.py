import os
import re
import math
import statistics
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from collections import Counter

from xorzen.utils.logger import get_logger
from xorzen.tokenizer.adapter import xorzenTokenizerAdapter
from xorzen.tokenizer.analysis import TokenizerAnalyzer
from xorzen.data.processor import _read_txt_file, _read_jsonl_data, _read_json_data

logger = get_logger()

class TokenizerEvaluator:
    """
    Evaluates a tokenizer's performance and suitability on a given dataset,
    providing a comprehensive report and a 10-mark score.
    """

    def __init__(self, tokenizer_adapter: XORZENXTokenizerAdapter, tokenizer_name: str, dataset_paths: List[Union[str, Path]], text_keys: Union[str, List[str]] = 'text', sample_size: Optional[int] = None, chunk_size_evaluation: int = 1000):
        """
        Initializes the TokenizerEvaluator.

        Args:
            tokenizer_adapter: An instance of XORZENXTokenizerAdapter managing the tokenizers.
            tokenizer_name: The name of the tokenizer to evaluate (must be loaded in adapter).
            dataset_paths: A list of file paths or directory paths containing the evaluation data.
            text_keys: The key(s) in JSON/JSONL records that contains the text to tokenize. Can be a string or a list of strings.
            sample_size: Number of texts to sample from the dataset for evaluation.
                         If None, the entire dataset will be used.
            chunk_size_evaluation: Number of texts to process in each chunk during evaluation.
        """
        self.tokenizer_adapter = tokenizer_adapter
        self.tokenizer_name = tokenizer_name
        self.dataset_paths = [Path(p) for p in dataset_paths]
        
        if isinstance(text_keys, str):
            self.text_keys = [text_keys]
        else:
            self.text_keys = text_keys
        
        self.sample_size = sample_size
        self.chunk_size_evaluation = chunk_size_evaluation
        self.tokenizer = self._get_tokenizer()
        self.analyzer = TokenizerAnalyzer(self.tokenizer)
        # self.corpus_texts will no longer be stored in memory

        logger.info("tokenizer.evaluation", f"Initializing evaluator for tokenizer '{tokenizer_name}' with dataset paths: {dataset_paths}")
        # self._load_corpus() no longer called here, data streamed in evaluate()

    def _get_tokenizer(self):
        """Retrieves and activates the specified tokenizer."""
        try:
            self.tokenizer_adapter.activate_tokenizer(self.tokenizer_name)
            return self.tokenizer_adapter.tokenizer
        except Exception as e:
            logger.error("tokenizer.evaluation", f"Failed to activate or retrieve tokenizer '{self.tokenizer_name}': {e}", exception=e)
            raise

    def _get_text_iterator(self):
        """
        Generates texts from the dataset, respecting sample_size.
        """
        logger.info("tokenizer.evaluation", "Preparing text iterator for evaluation...")
        
        files_to_process = []
        supported_file_extensions = {'.txt', '.json', '.jsonl'}

        for input_path_item in self.dataset_paths:
            if input_path_item.is_file():
                if input_path_item.suffix.lower() in supported_file_extensions:
                    files_to_process.append(input_path_item)
                else:
                    logger.warning("tokenizer.evaluation", f"Unsupported file type for evaluation: {input_path_item.suffix}. Skipping {input_path_item}.")
            elif input_path_item.is_dir():
                for ext in supported_file_extensions:
                    files_to_process.extend(input_path_item.rglob(f'*{ext}'))
            else:
                logger.warning("tokenizer.evaluation", f"Input path '{input_path_item}' not found or is not a supported file/directory type. Skipping.")
        
        if not files_to_process:
            logger.error("tokenizer.evaluation", "No supported input files found for evaluation.")
            return

        texts_yielded = 0
        for file_path in files_to_process:
            file_reader = None
            if file_path.suffix.lower() == '.txt':
                file_reader = _read_txt_file(file_path)
            elif file_path.suffix.lower() == '.jsonl':
                file_reader = _read_jsonl_data(file_path, text_keys=self.text_keys)
            elif file_path.suffix.lower() == '.json':
                file_reader = _read_json_data(file_path, text_keys=self.text_keys)
            
            if file_reader:
                for text_content in file_reader:
                    if text_content:
                        yield text_content
                        texts_yielded += 1
                        if self.sample_size is not None and texts_yielded >= self.sample_size:
                            logger.info("tokenizer.evaluation", f"Yielded {texts_yielded} texts up to sample_size limit.")
                            return
        logger.info("tokenizer.evaluation", f"Finished yielding {texts_yielded} texts from all files.")

    def evaluate(self) -> Dict[str, Any]:
        """
        Performs a detailed evaluation of the tokenizer on the loaded corpus using chunking and streaming.

        Returns:
            A dictionary containing evaluation results, including the 10-mark score.
        """
        logger.info("tokenizer.evaluation", "Starting tokenizer evaluation...")
        
        # Accumulators for metrics (streaming approach)
        unique_token_ids_in_corpus = set()
        total_tokens = 0
        sum_token_lengths = 0
        max_token_length_seen = 0
        total_original_words = 0
        total_original_chars = 0
        total_texts_processed = 0
        unk_token_count = 0
        special_token_count = 0
        
        # Prepare for UNK and special token identification
        unk_token_id = self.tokenizer.token_to_id(XORZENXTokenizerAdapter.SPECIAL_TOKENS[0])
        special_token_ids = {self.tokenizer.token_to_id(t) for t in XORZENXTokenizerAdapter.SPECIAL_TOKENS if self.tokenizer.token_to_id(t) is not None}
        if unk_token_id is not None:
            special_token_ids.discard(unk_token_id)

        # Small sample for speed analysis
        speed_analysis_sample = []
        speed_analysis_sample_limit = 100 # Always take 100 samples for speed, if available
        
        # Iterate through texts in chunks
        current_text_chunk = []
        text_iterator = self._get_text_iterator()

        # Wrap iterator with tqdm for progress tracking, if available
        try:
            from tqdm.auto import tqdm
            if self.sample_size is not None:
                pbar = tqdm(total=self.sample_size, unit="texts", desc="Evaluating Tokenizer")
            else:
                pbar = tqdm(unit="texts", desc="Evaluating Tokenizer (full dataset)")
        except ImportError:
            pbar = None # No progress bar if tqdm not available
            logger.warning("tokenizer.evaluation", "tqdm not installed, progress bar disabled.")

        for text in text_iterator:
            current_text_chunk.append(text)
            
            # Populate speed analysis sample
            if len(speed_analysis_sample) < speed_analysis_sample_limit:
                speed_analysis_sample.append(text)

            if len(current_text_chunk) >= self.chunk_size_evaluation:
                # Process the chunk
                for text_in_chunk in current_text_chunk:
                    encoded = self.tokenizer.encode(text_in_chunk)
                    ids = encoded.ids
                    
                    # Accumulate metrics
                    unique_token_ids_in_corpus.update(ids)
                    total_tokens += len(ids)
                    total_original_chars += len(text_in_chunk)
                    total_original_words += len(text_in_chunk.split())
                    total_texts_processed += 1

                    for token_id in ids:
                        if token_id == unk_token_id:
                            unk_token_count += 1
                        elif token_id in special_token_ids:
                            special_token_count += 1
                    
                    if encoded.tokens: # Ensure tokens list is not empty
                        sum_token_lengths += sum(len(t) for t in encoded.tokens)
                        max_token_length_seen = max(max_token_length_seen, max(len(t) for t in encoded.tokens))
                
                if pbar: pbar.update(len(current_text_chunk))
                current_text_chunk = [] # Reset chunk
                logger.info("tokenizer.evaluation", f"Processed {total_texts_processed} texts...")
        
        # Process any remaining texts in the last chunk
        if current_text_chunk:
            for text_in_chunk in current_text_chunk:
                encoded = self.tokenizer.encode(text_in_chunk)
                ids = encoded.ids
                
                # Accumulate metrics
                unique_token_ids_in_corpus.update(ids)
                total_tokens += len(ids)
                total_original_chars += len(text_in_chunk)
                total_original_words += len(text_in_chunk.split())
                total_texts_processed += 1

                for token_id in ids:
                    if token_id == unk_token_id:
                        unk_token_count += 1
                    elif token_id in special_token_ids:
                        special_token_count += 1
                
                if encoded.tokens:
                    sum_token_lengths += sum(len(t) for t in encoded.tokens)
                    max_token_length_seen = max(max_token_length_seen, max(len(t) for t in encoded.tokens))
            
            if pbar: pbar.update(len(current_text_chunk))
            logger.info("tokenizer.evaluation", f"Processed final {len(current_text_chunk)} texts.")

        if pbar: pbar.close()

        if total_texts_processed == 0:
            logger.warning("tokenizer.evaluation", "No corpus texts processed. Cannot perform evaluation.")
            return {"error": "No corpus data available for evaluation."}

        results = {}

        # Vocab analysis is static
        results["vocab_analysis"] = self.analyzer.analyze_vocabulary() 
        
        # Aggregate tokenization analysis
        avg_tokens_per_text = total_tokens / total_texts_processed if total_texts_processed > 0 else 0
        avg_tokens_per_word = total_tokens / total_original_words if total_original_words > 0 else 0
        compression_ratio_char_to_token = total_original_chars / total_tokens if total_tokens > 0 else 0
        avg_token_length = sum_token_lengths / total_tokens if total_tokens > 0 else 0
        vocab_coverage_in_corpus = len(unique_token_ids_in_corpus) / self.tokenizer.get_vocab_size() if self.tokenizer.get_vocab_size() > 0 else 0

        results["tokenization_analysis"] = {
            'num_texts': total_texts_processed,
            'total_characters': total_original_chars,
            'total_words': total_original_words,
            'total_tokens': total_tokens,
            'avg_tokens_per_text': avg_tokens_per_text,
            'avg_tokens_per_word': avg_tokens_per_word,
            'compression_ratio_char_to_token': compression_ratio_char_to_token,
            'unique_tokens_used_in_corpus': len(unique_token_ids_in_corpus),
            'vocab_coverage_in_corpus': vocab_coverage_in_corpus,
        }
        
        # Speed analysis on a small sample
        results["speed_analysis"] = self.analyzer.measure_tokenization_speed(speed_analysis_sample)

        results["custom_metrics"] = {
            "unk_token_count": unk_token_count,
            "unk_token_ratio": (unk_token_count / total_tokens) * 100 if total_tokens > 0 else 0,
            "special_token_count": special_token_count,
            "special_token_ratio": (special_token_count / total_tokens) * 100 if total_tokens > 0 else 0,
            "avg_token_length": avg_token_length,
            "max_token_length": max_token_length_seen,
            "compression_ratio_char_to_token": compression_ratio_char_to_token,
            "avg_tokens_per_word": avg_tokens_per_word,
        }

        # Calculate 10-mark score
        score = self._calculate_score(results)
        results["evaluation_score"] = score
        
        logger.info("tokenizer.evaluation", f"Tokenizer evaluation completed for '{self.tokenizer_name}'. Score: {score:.2f}/10")
        return results

    def _calculate_score(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Calculates a 10-mark score based on tokenizer performance metrics.
        Higher is better.

        Scoring criteria (example weights, adjustable):
        - UNK Token Ratio: (Lower is better) Max 3 points (0% UNK = 3, 10% UNK = 0)
        - Compression Ratio (char to token): (Higher is better) Max 2 points (e.g., 4:1 = 2, 1:1 = 0)
        - Average Tokens per Word: (Closer to 1 is better) Max 2 points (e.g., 1.1 = 2, 2.0 = 0)
        - Special Token Ratio: (Lower is better, excluding UNK) Max 1 point (0% = 1, 5% = 0)
        - Max Token Length: (Reasonable is better) Max 1 point (e.g., <50 = 1, >100 = 0)
        - Overall Vocabulary Coverage in Corpus: (Higher is better) Max 1 point (100% = 1, <50% = 0)
        """
        score = 0.0
        metrics = evaluation_results["custom_metrics"]
        tok_analysis = evaluation_results["tokenization_analysis"]

        # 1. UNK Token Ratio (Max 3 points)
        unk_ratio = metrics["unk_token_ratio"]
        if unk_ratio <= 0.5: # <0.5% UNK is excellent
            score += 3.0
        elif unk_ratio <= 1.0: # <1% UNK is very good
            score += 2.5
        elif unk_ratio <= 2.0: # <2% UNK is good
            score += 2.0
        elif unk_ratio <= 5.0: # <5% UNK is acceptable
            score += 1.0
        elif unk_ratio <= 10.0: # <10% UNK is poor
            score += 0.5
        # else: 0 points

        # 2. Compression Ratio (char to token) (Max 2 points)
        # Ideal compression ratios depend on language and desired token granularity.
        # For English BPE, 3-5 chars/token is often good.
        comp_ratio = metrics["compression_ratio_char_to_token"]
        if comp_ratio >= 3.5: # Very good compression
            score += 2.0
        elif comp_ratio >= 2.5: # Good compression
            score += 1.5
        elif comp_ratio >= 1.5: # Acceptable compression
            score += 0.5
        # else: 0 points

        # 3. Average Tokens per Word (Max 2 points)
        # Closer to 1 is better for semantic clarity, but >1 is expected for subword tokenizers.
        avg_tokens_per_word = metrics["avg_tokens_per_word"]
        if 1.0 < avg_tokens_per_word <= 1.2: # Near optimal subword granularity
            score += 2.0
        elif 1.2 < avg_tokens_per_word <= 1.5: # Good
            score += 1.5
        elif 1.5 < avg_tokens_per_word <= 2.0: # Acceptable
            score += 0.5
        # else: 0 points

        # 4. Special Token Ratio (excluding UNK) (Max 1 point)
        special_ratio = metrics["special_token_ratio"]
        if special_ratio <= 0.1: # Minimal special token interference
            score += 1.0
        elif special_ratio <= 0.5:
            score += 0.5
        # else: 0 points

        # 5. Max Token Length (Max 1 point)
        max_token_len = metrics["max_token_length"]
        if max_token_len < 50: # Very reasonable max token length
            score += 1.0
        elif max_token_len < 100: # Acceptable
            score += 0.5
        # else: 0 points

        # 6. Overall Vocabulary Coverage in Corpus (Max 1 point)
        # How much of the full vocabulary is actually used in the sample corpus.
        vocab_coverage = tok_analysis["vocab_coverage_in_corpus"]
        if vocab_coverage >= 0.9: # Excellent coverage
            score += 1.0
        elif vocab_coverage >= 0.7: # Good coverage
            score += 0.5
        # else: 0 points

        return min(10.0, score) # Cap at 10

    def _generate_report(self, evaluation_results: Dict[str, Any], score: float, total_texts_processed: int) -> str:
        """
        Generates a human-readable report summarizing the evaluation results.
        """
        report_lines = [
            "=" * 80,
            f"Tokenizer Evaluation Report for '{self.tokenizer_name}'",
            "=" * 80,
            f"Date: {Path(__file__).stat().st_mtime}", # Using file modification time as a proxy for 'today'
            f"Dataset Sample Size: {total_texts_processed} texts",
            f"Tokenizer Vocab Size: {self.tokenizer.get_vocab_size()}",
            "\n" + "=" * 30 + " Final Score " + "=" * 30,
            f"  Overall Tokenizer Score: {score:.2f} / 10.0",
            "=" * 80,
            "\n## Key Performance Indicators (KPIs)",
        ]

        cm = evaluation_results["custom_metrics"]
        ta = evaluation_results["tokenization_analysis"]
        va = evaluation_results["vocab_analysis"]
        
        report_lines.extend([
            f"- **UNK Token Ratio**: {cm['unk_token_ratio']:.2f}% (Target: <1%)",
            f"- **Compression Ratio (Chars/Token)**: {cm['compression_ratio_char_to_token']:.2f} (Target: 2.5 - 4.0)",
            f"- **Average Tokens per Word**: {cm['avg_tokens_per_word']:.2f} (Target: 1.0 - 1.5)",
            f"- **Special Token Ratio (excluding UNK)**: {cm['special_token_ratio']:.2f}% (Target: <0.5%)",
            f"- **Maximum Token Length**: {cm['max_token_length']} (Target: <100)",
            f"- **Vocabulary Coverage in Corpus**: {ta['vocab_coverage_in_corpus']:.2f} (Target: >0.8)",
        ])

        report_lines.append("\n## Detailed Analysis")
        report_lines.append("\n### Vocabulary Analysis")
        for k, v in va.items():
            if isinstance(v, (float, int)):
                report_lines.append(f"  - {k.replace('_', ' ').title()}: {v:,}")
            elif isinstance(v, list):
                report_lines.append(f"  - {k.replace('_', ' ').title()}: {', '.join(str(item) for item in v[:5])}...")

        report_lines.append("\n### Tokenization Analysis")
        for k, v in ta.items():
            if isinstance(v, (float, int)):
                report_lines.append(f"  - {k.replace('_', ' ').title()}: {v:,.2f}")
        
        report_lines.append(f"  - Total UNK Tokens: {cm['unk_token_count']:,}")
        report_lines.append(f"  - Total Other Special Tokens: {cm['special_token_count']:,}")

        report_lines.append("\n### Potential Issues & Recommendations")
        issues = []
        if cm['unk_token_ratio'] > 5.0:
            issues.append(f"- High UNK token ratio ({cm['unk_token_ratio']:.2f}%). Consider retraining with a larger or more diverse corpus, or increasing vocabulary size.")
        if cm['compression_ratio_char_to_token'] < 1.5:
            issues.append(f"- Low compression ratio ({cm['compression_ratio_char_to_token']:.2f}). This might lead to longer sequences and higher computational cost. Review pre-tokenizer settings or consider different tokenizer types (e.g., SentencePiece).")
        if cm['avg_tokens_per_word'] > 2.0:
            issues.append(f"- High average tokens per word ({cm['avg_tokens_per_word']:.2f}). The tokenizer might be splitting words too aggressively, potentially losing semantic coherence. Adjust BPE merge rules or use a different pre-tokenizer.")
        if cm['special_token_ratio'] > 2.0:
            issues.append(f"- High usage of other special tokens ({cm['special_token_ratio']:.2f}%). Ensure these tokens are correctly used and not inadvertently tokenizing regular text.")
        if cm['max_token_length'] > 100:
            issues.append(f"- Very long maximum token length ({cm['max_token_length']}). This could indicate issues with certain patterns not being split, or very specific long words.")
        if ta['vocab_coverage_in_corpus'] < 0.5:
            issues.append(f"- Low vocabulary coverage in corpus ({ta['vocab_coverage_in_corpus']:.2f}). Many words in the corpus are not covered by the vocabulary, contributing to high UNK rates. Ensure corpus is representative.")
        
        if issues:
            report_lines.extend(issues)
        else:
            report_lines.append("- No significant issues identified based on current metrics and thresholds. Well done!")
        
        report_lines.append("\n## Tokenization Speed")
        speed_analysis = evaluation_results["speed_analysis"]
        report_lines.extend([
            f"  - Texts per second: {speed_analysis['texts_per_second']:.2f}",
            f"  - Characters per second: {speed_analysis['chars_per_second']:.2f}",
            f"  - Average time per run: {speed_analysis['avg_time_per_run_sec']:.4f} seconds",
        ])

        report_lines.append("\n" + "=" * 80)
        return "\n".join(report_lines)

    def generate_detailed_report(self) -> str:
        """
        Runs the evaluation and generates a detailed report string.
        """
        results = self.evaluate()
        score = results.get("evaluation_score", 0.0)
        total_texts_processed = results["tokenization_analysis"]["num_texts"]
        return self._generate_report(results, score, total_texts_processed)

# Example Usage (for testing within this file)
if __name__ == '__main__':
    # Assume a dummy tokenizer and corpus for testing
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    import tempfile
    import shutil

    # Setup dummy logger
    class DummyLogger:
        def info(self, *args, **kwargs): print(f"INFO: {args}")
        def debug(self, *args, **kwargs): print(f"DEBUG: {args}")
        def warning(self, *args, **kwargs): print(f"WARNING: {args}")
        def error(self, *args, **kwargs): print(f"ERROR: {args}")
    get_logger = lambda: DummyLogger() # Override get_logger for local test

    TEST_DIR = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {TEST_DIR}")

    # Create dummy corpus
    corpus_data = [
        "This is a sample text for tokenizer evaluation. It contains some unique words like xorzen-zero and complex_math_expression.",
        "Another sentence with common words and also some <unk> tokens if the vocabulary is small. We need good compression.",
        "The quick brown fox jumps over the lazy dog. A very standard english sentence. Some long_token_here_for_testing_max_length.",
        "This is an <unk> example with <special_token> in it. The tokenizer should handle it properly. <bos> <eos>",
        "Short text.",
        "A really really really really really really really really really really long word that might break the tokenizer if it is not handled correctly. Akik Faraji, FRAZIYM AI. XORZENX_zero. If two lines $l$ and $m$ have equations $y = -x + 6$, and $y = -4x + 6$, what is the probability that a point randomly selected in the 1st quadrant and below $l$ will fall between $l$ and $m$? Express your answer as a decimal to the nearest hundredth.\n\n[asy]\nimport cse5; import olympiad;\nsize(150);\nadd(grid(8,8));\ndraw((0,0)--(8,0),linewidth(1.2));\ndraw((0,",
        "Pneumonoultramicroscopicsilicovolcanoconiosis is a very long word.",
        "Some numeric data 12345 67890.",
        "A mix of cases: TensorFlow Pytorch HuggingFace.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s t u v w x y z."
    ]
    corpus_file_path = TEST_DIR / "corpus.txt"
    corpus_file_path.write_text("\n".join(corpus_data))

    # --- Train a dummy tokenizer ---
    tokenizer_path = TEST_DIR / "dummy_tokenizer.json"
    
    # Simple BPE tokenizer
    tokenizer_obj = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer_obj.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=100, # Small vocab to generate UNKs
        min_frequency=1,
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>", "<mask>", "<special_token>"]
    )
    tokenizer_obj.train([str(corpus_file_path)], trainer)
    tokenizer_obj.save(str(tokenizer_path))
    
    # Create a dummy XORZENXTokenizerAdapter
    adapter = XORZENXTokenizerAdapter()
    adapter.load_tokenizer("test_tokenizer", str(tokenizer_path))

    # --- Evaluate ---
    evaluator = TokenizerEvaluator(
        tokenizer_adapter=adapter,
        tokenizer_name="test_tokenizer",
        dataset_paths=[corpus_file_path],
        sample_size=None # Use entire small corpus for testing
    )
    report = evaluator.generate_detailed_report()
    print(report)

    # Clean up
    shutil.rmtree(TEST_DIR)
    print(f"\nCleaned up temporary directory: {TEST_DIR}")

__all__ = ['TokenizerEvaluator']

