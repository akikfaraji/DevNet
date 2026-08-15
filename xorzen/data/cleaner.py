"""
Data Cleaner Module for xorzen
Provides comprehensive data cleaning and quality improvement tools.
"""

import os
import sys
import json
import argparse
import re
from typing import Union, List, Dict, Any, Optional, Callable, Set
from pathlib import Path
from collections import Counter
import itertools # Moved from _is_repetitive

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from xorzen.utils.logger import get_logger, setup_global_logger, LogLevel


class DataCleaner:
    """Comprehensive data cleaning and quality improvement tool."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger()
        self.config = config or self._default_config()
        self.stats = {
            "total_processed": 0,
            "total_removed": 0,
            "total_modified": 0,
            "issues_found": Counter()
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default cleaning configuration."""
        return {
            # Length filters
            "min_length": 10,  # Minimum character length
            "max_length": 100000,  # Maximum character length
            "min_words": 3,  # Minimum word count
            
            # Quality filters
            "remove_duplicates": True,
            "remove_empty": True,
            "remove_urls": False,
            "remove_emails": False,
            "normalize_whitespace": True,
            "remove_special_chars": False,
            
            # Language filters
            "min_alpha_ratio": 0.5,  # Minimum alphabetic character ratio
            "max_digit_ratio": 0.5,  # Maximum digit ratio
            "max_special_ratio": 0.3,  # Maximum special character ratio
            
            # Content filters
            "filter_profanity": False,
            "filter_urls_threshold": 3,  # Remove if more than N URLs
            "filter_repetition": True,  # Remove highly repetitive text
            "repetition_threshold": 0.7,  # Character-level repetition threshold
            
            # Encoding
            "fix_encoding": True,
            "normalize_unicode": True,
            
            # Custom filters
            "custom_filters": []
        }
    
    def clean_text(self, text: str) -> Optional[str]:
        """
        Clean a single text string according to configuration.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text or None if text should be removed
        """
        if not text or not isinstance(text, str):
            self.stats["issues_found"]["empty_or_invalid"] += 1
            return None
        
        original_text = text
        
        # Basic cleaning
        if self.config["normalize_whitespace"]:
            text = ' '.join(text.split())
        
        if self.config["fix_encoding"]:
            text = self._fix_encoding(text)
        
        if self.config["normalize_unicode"]:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
        
        # Length filters
        if len(text) < self.config["min_length"]:
            self.stats["issues_found"]["too_short"] += 1
            return None
        
        if len(text) > self.config["max_length"]:
            self.stats["issues_found"]["too_long"] += 1
            return None
        
        words = text.split()
        if len(words) < self.config["min_words"]:
            self.stats["issues_found"]["too_few_words"] += 1
            return None
        
        # Quality filters
        if not self._check_quality(text):
            return None
        
        # Content filters
        if self.config["remove_urls"]:
            text = self._remove_urls(text)
        
        if self.config["remove_emails"]:
            text = self._remove_emails(text)
        
        if self.config["remove_special_chars"]:
            text = self._remove_special_chars(text)
        
        # Check if text was significantly modified
        if text != original_text:
            self.stats["total_modified"] += 1
        
        return text if text.strip() else None
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Common encoding fixes
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€"': '--',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã¢': 'â',
            'Ã´': 'ô',
            'Ã»': 'û',
            'Ã§': 'ç',
            'Ã±': 'ñ'
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _check_quality(self, text: str) -> bool:
        """Check if text meets quality thresholds."""
        if not text:
            return False
        
        # Alpha ratio
        alpha_count = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_count / len(text)
        
        if alpha_ratio < self.config["min_alpha_ratio"]:
            self.stats["issues_found"]["low_alpha_ratio"] += 1
            return False
        
        # Digit ratio
        digit_count = sum(c.isdigit() for c in text)
        digit_ratio = digit_count / len(text)
        
        if digit_ratio > self.config["max_digit_ratio"]:
            self.stats["issues_found"]["high_digit_ratio"] += 1
            return False
        
        # Special character ratio
        special_count = sum(not c.isalnum() and not c.isspace() for c in text)
        special_ratio = special_count / len(text)
        
        if special_ratio > self.config["max_special_ratio"]:
            self.stats["issues_found"]["high_special_ratio"] += 1
            return False
        
        # Repetition check
        if self.config["filter_repetition"]:
            if self._is_repetitive(text):
                self.stats["issues_found"]["repetitive"] += 1
                return False
        
        # URL threshold
        if self.config["filter_urls_threshold"] > 0:
            url_count = len(re.findall(r'http[s]?://\S+', text))
            if url_count > self.config["filter_urls_threshold"]:
                self.stats["issues_found"]["too_many_urls"] += 1
                return False
        
        return True
    
    def _is_repetitive(self, text: str) -> bool:
        """Check if text is highly repetitive."""
        if len(text) < 100:
            return False
        
        # Check for repeated characters
        max_char_repeat = max((len(list(group)) for char, group in 
                              __import__('itertools').groupby(text)), default=0)
        
        if max_char_repeat > 20:
            return True
        
        # Check for repeated n-grams
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check 3-gram repetition
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if trigrams:
            most_common = Counter(trigrams).most_common(1)[0]
            if most_common[1] / len(trigrams) > self.config["repetition_threshold"]:
                return True
        
        return False
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and basic punctuation."""
        return re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', text)
    
    def clean_file(self, input_path: Union[str, Path], 
                   output_path: Union[str, Path],
                   text_keys: Union[str, List[str]] = 'text',
                   preserve_format: bool = True) -> Dict[str, Any]:
        """
        Clean a data file and save the result.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            text_keys: Key(s) for text extraction in JSON/JSONL
            preserve_format: Preserve original file format
            
        Returns:
            Dictionary with cleaning statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_ext = input_path.suffix.lower()
        
        self.logger.info("data.cleaner", f"Cleaning file: {input_path}")
        
        if file_ext == '.txt':
            return self._clean_txt(input_path, output_path)
        elif file_ext == '.jsonl':
            return self._clean_jsonl(input_path, output_path, text_keys)
        elif file_ext == '.json':
            return self._clean_json(input_path, output_path, text_keys)
        elif file_ext == '.parquet':
            return self._clean_parquet(input_path, output_path, text_keys)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _clean_txt(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Clean text file."""
        stats = {"input_file": str(input_path), "output_file": str(output_path)}
        
        cleaned_lines = []
        seen = set() if self.config["remove_duplicates"] else None
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stats["total_processed"] += 1
                line = line.strip()
                
                if not line and self.config["remove_empty"]:
                    self.stats["total_removed"] += 1
                    continue
                
                cleaned = self.clean_text(line)
                
                if cleaned is None:
                    self.stats["total_removed"] += 1
                    continue
                
                # Deduplication
                if seen is not None:
                    if cleaned in seen:
                        self.stats["issues_found"]["duplicate"] += 1
                        self.stats["total_removed"] += 1
                        continue
                    seen.add(cleaned)
                
                cleaned_lines.append(cleaned)
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(line + '\n')
        
        stats.update(self.stats.copy())
        stats["output_lines"] = len(cleaned_lines)
        
        self.logger.info("data.cleaner", 
                        f"Cleaned {self.stats['total_processed']} lines, "
                        f"removed {self.stats['total_removed']}, "
                        f"kept {len(cleaned_lines)}")
        
        return stats
    
    def _clean_jsonl(self, input_path: Path, output_path: Path, 
                     text_keys: Union[str, List[str]]) -> Dict[str, Any]:
        """Clean JSONL file."""
        if isinstance(text_keys, str):
            text_keys = [text_keys]
        
        stats = {"input_file": str(input_path), "output_file": str(output_path)}
        
        cleaned_records = []
        seen = set() if self.config["remove_duplicates"] else None
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stats["total_processed"] += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # Find and clean text
                    text_found = False
                    for key in text_keys:
                        if key in record:
                            original_text = str(record[key])
                            cleaned_text = self.clean_text(original_text)
                            
                            if cleaned_text is None:
                                self.stats["total_removed"] += 1
                                text_found = False
                                break
                            
                            record[key] = cleaned_text
                            text_found = True
                            break
                    
                    if not text_found:
                        self.stats["total_removed"] += 1
                        continue
                    
                    # Deduplication
                    if seen is not None:
                        text_to_check = record.get(text_keys[0], "")
                        if text_to_check in seen:
                            self.stats["issues_found"]["duplicate"] += 1
                            self.stats["total_removed"] += 1
                            continue
                        seen.add(text_to_check)
                    
                    cleaned_records.append(record)
                    
                except json.JSONDecodeError:
                    self.stats["issues_found"]["json_decode_error"] += 1
                    self.stats["total_removed"] += 1
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        stats.update(self.stats.copy())
        stats["output_records"] = len(cleaned_records)
        
        self.logger.info("data.cleaner", 
                        f"Cleaned {self.stats['total_processed']} records, "
                        f"removed {self.stats['total_removed']}, "
                        f"kept {len(cleaned_records)}")
        
        return stats
    
    def _clean_json(self, input_path: Path, output_path: Path, 
                    text_keys: Union[str, List[str]]) -> Dict[str, Any]:
        """Clean JSON file."""
        if isinstance(text_keys, str):
            text_keys = [text_keys]
        
        stats = {"input_file": str(input_path), "output_file": str(output_path)}
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Input JSON must be an array")
        
        cleaned_records = []
        seen = set() if self.config["remove_duplicates"] else None
        
        for record in data:
            self.stats["total_processed"] += 1
            
            # Find and clean text
            text_found = False
            for key in text_keys:
                if key in record:
                    original_text = str(record[key])
                    cleaned_text = self.clean_text(original_text)
                    
                    if cleaned_text is None:
                        self.stats["total_removed"] += 1
                        text_found = False
                        break
                    
                    record[key] = cleaned_text
                    text_found = True
                    break
            
            if not text_found:
                self.stats["total_removed"] += 1
                continue
            
            # Deduplication
            if seen is not None:
                text_to_check = record.get(text_keys[0], "")
                if text_to_check in seen:
                    self.stats["issues_found"]["duplicate"] += 1
                    self.stats["total_removed"] += 1
                    continue
                seen.add(text_to_check)
            
            cleaned_records.append(record)
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
        
        stats.update(self.stats.copy())
        stats["output_records"] = len(cleaned_records)
        
        self.logger.info("data.cleaner", 
                        f"Cleaned {self.stats['total_processed']} records, "
                        f"removed {self.stats['total_removed']}, "
                        f"kept {len(cleaned_records)}")
        
        return stats
    
    def _clean_parquet(self, input_path: Path, output_path: Path, 
                       text_keys: Union[str, List[str]]) -> Dict[str, Any]:
        """Clean Parquet file."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas/pyarrow required for Parquet cleaning")
        
        if isinstance(text_keys, str):
            text_keys = [text_keys]
        
        stats = {"input_file": str(input_path), "output_file": str(output_path)}
        
        df = pd.read_parquet(input_path)
        original_len = len(df)
        self.stats["total_processed"] = original_len
        
        # Find text column
        text_col = None
        for key in text_keys:
            if key in df.columns:
                text_col = key
                break
        
        if text_col is None:
            raise ValueError(f"None of the text keys {text_keys} found in Parquet columns")
        
        # Clean text column
        df[text_col] = df[text_col].astype(str).apply(self.clean_text)
        
        # Remove rows where cleaning returned None
        df = df[df[text_col].notna()]
        
        # Deduplication
        if self.config["remove_duplicates"]:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=[text_col])
            duplicates_removed = before_dedup - len(df)
            self.stats["issues_found"]["duplicate"] = duplicates_removed
        
        self.stats["total_removed"] = original_len - len(df)
        
        # Write output
        df.to_parquet(output_path, index=False)
        
        stats.update(self.stats.copy())
        stats["output_records"] = len(df)
        
        self.logger.info("data.cleaner", 
                        f"Cleaned {original_len} records, "
                        f"removed {self.stats['total_removed']}, "
                        f"kept {len(df)}")
        
        return stats
    
    def clean_directory(self, input_dir: Union[str, Path], 
                       output_dir: Union[str, Path],
                       text_keys: Union[str, List[str]] = 'text',
                       recursive: bool = True) -> Dict[str, Any]:
        """
        Clean all data files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            text_keys: Key(s) for text extraction
            recursive: Whether to process recursively
            
        Returns:
            Dictionary with overall statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("data.cleaner", f"Cleaning directory: {input_dir}")
        
        # Find all supported files
        supported_extensions = {'.txt', '.json', '.jsonl', '.parquet'}
        files = []
        
        if recursive:
            for ext in supported_extensions:
                files.extend(input_dir.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                files.extend(input_dir.glob(f"*{ext}"))
        
        self.logger.info("data.cleaner", f"Found {len(files)} files to clean")
        
        # Clean each file
        file_stats = []
        overall_stats = {
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "file_details": []
        }
        
        for filepath in files:
            try:
                # Determine output path
                relative_path = filepath.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Clean file
                stats = self.clean_file(filepath, output_path, text_keys)
                file_stats.append(stats)
                overall_stats["successful"] += 1
                overall_stats["file_details"].append({
                    "input": str(filepath),
                    "output": str(output_path),
                    "status": "success",
                    "processed": stats.get("total_processed", 0),
                    "removed": stats.get("total_removed", 0)
                })
                
            except Exception as e:
                self.logger.error("data.cleaner", f"Error cleaning {filepath}: {e}")
                overall_stats["failed"] += 1
                overall_stats["file_details"].append({
                    "input": str(filepath),
                    "status": "failed",
                    "error": str(e)
                })
        
        overall_stats["detailed_stats"] = file_stats
        
        self.logger.info("data.cleaner", 
                        f"Cleaning complete: {overall_stats['successful']} successful, "
                        f"{overall_stats['failed']} failed")
        
        return overall_stats


def main():
    """CLI entry point for data cleaner."""
    parser = argparse.ArgumentParser(description="Clean and improve data quality for xorzen.")
    parser.add_argument("input", type=str, help="Input file or directory path")
    parser.add_argument("output", type=str, help="Output file or directory path")
    parser.add_argument("--text_keys", nargs='+', default=['text'],
                       help="Key(s) for text extraction in JSON/JSONL")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--min_length", type=int, help="Minimum text length")
    parser.add_argument("--max_length", type=int, help="Maximum text length")
    parser.add_argument("--min_words", type=int, help="Minimum word count")
    parser.add_argument("--remove_duplicates", action="store_true", help="Remove duplicates")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--report", type=str, help="Save cleaning report to file")
    
    args = parser.parse_args()
    
    # Setup logger
    setup_global_logger(name="xorzen-data-cleaner", log_dir="logs_cleaner",
                       level=LogLevel.INFO, enable_async=False)
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
        if args.min_length:
            config["min_length"] = args.min_length
        if args.max_length:
            config["max_length"] = args.max_length
        if args.min_words:
            config["min_words"] = args.min_words
        if args.remove_duplicates:
            config["remove_duplicates"] = True
    
    cleaner = DataCleaner(config)
    
    try:
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if input_path.is_file():
            stats = cleaner.clean_file(input_path, output_path, args.text_keys)
        elif input_path.is_dir():
            stats = cleaner.clean_directory(input_path, output_path, args.text_keys, args.recursive)
        else:
            raise ValueError(f"Invalid input path: {args.input}")
        
        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            get_logger().info("data.cleaner", f"Report saved to: {report_path}")
        
        print("\nCleaning Summary:")
        print(f"Total processed: {stats.get('total_processed', 0)}")
        print(f"Total removed: {stats.get('total_removed', 0)}")
        print(f"Total modified: {stats.get('total_modified', 0)}")
        print(f"Issues found: {dict(stats.get('issues_found', {}))}")
        
    except Exception as e:
        get_logger().critical("data.cleaner", f"Cleaning failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        get_logger().cleanup()


if __name__ == '__main__':
    main()

__all__ = ['DataCleaner']

