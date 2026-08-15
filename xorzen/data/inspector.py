"""
Data Inspector Module for xorzen
Provides comprehensive data inspection, quality analysis, and statistics generation.
"""

import os
import sys
import json
import argparse
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import re

from xorzen.utils.logger import get_logger, setup_global_logger, LogLevel

try:
    import pandas as pd
    import pyarrow.parquet as pq
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    get_logger().warning("data.inspector", "pandas and/or pyarrow not available. Parquet inspection will not be supported.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class DataInspector:
    """Comprehensive data inspection and quality analysis tool."""
    
    def __init__(self):
        self.logger = get_logger()
        self.stats = {}
        
    def inspect_file(self, filepath: Union[str, Path], 
                     text_keys: Union[str, List[str]] = 'text',
                     sample_size: int = 100) -> Dict[str, Any]:
        """
        Inspect a single data file and return comprehensive statistics.
        
        Args:
            filepath: Path to the data file
            text_keys: Key(s) to extract text from JSON/JSONL files
            sample_size: Number of samples to analyze for detailed stats
            
        Returns:
            Dictionary containing file statistics
        """
        filepath = Path(filepath)
        if not filepath.exists():
            self.logger.error("data.inspector", f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        file_ext = filepath.suffix.lower()
        
        if file_ext == '.txt':
            return self._inspect_txt(filepath, sample_size)
        elif file_ext == '.jsonl':
            return self._inspect_jsonl(filepath, text_keys, sample_size)
        elif file_ext == '.json':
            return self._inspect_json(filepath, text_keys, sample_size)
        elif file_ext == '.parquet':
            return self._inspect_parquet(filepath, text_keys, sample_size)
        elif file_ext in ['.npy', '.bin', '.pt']:
            return self._inspect_binary(filepath, file_ext)
        else:
            self.logger.warning("data.inspector", f"Unsupported file type: {file_ext}")
            return {"error": f"Unsupported file type: {file_ext}"}
    
    def _inspect_txt(self, filepath: Path, sample_size: int) -> Dict[str, Any]:
        """Inspect text file."""
        stats = {
            "file_type": "txt",
            "filepath": str(filepath),
            "file_size_bytes": filepath.stat().st_size,
            "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2)
        }
        
        lines = []
        total_chars = 0
        line_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        lines.append(line)
                        total_chars += len(line)
                        line_count += 1
                        if i >= sample_size:
                            break
            
            if lines:
                stats.update(self._analyze_text_samples(lines))
                stats["total_lines_sampled"] = line_count
                stats["avg_chars_per_line"] = round(total_chars / len(lines), 2) if lines else 0
            
            # Count total lines
            with open(filepath, 'r', encoding='utf-8') as f:
                stats["total_lines"] = sum(1 for _ in f)
                
        except Exception as e:
            self.logger.error("data.inspector", f"Error inspecting TXT file: {e}", exc_info=True)
            stats["error"] = str(e)
        
        return stats
    
    def _inspect_jsonl(self, filepath: Path, text_keys: Union[str, List[str]], 
                       sample_size: int) -> Dict[str, Any]:
        """Inspect JSONL file."""
        if isinstance(text_keys, str):
            text_keys = [text_keys]
        
        stats = {
            "file_type": "jsonl",
            "filepath": str(filepath),
            "file_size_bytes": filepath.stat().st_size,
            "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2)
        }
        
        records = []
        text_samples = []
        total_records = 0
        malformed_count = 0
        missing_text_key_count = 0
        field_counter = Counter()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    total_records += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        records.append(record)
                        
                        # Count fields
                        for key in record.keys():
                            field_counter[key] += 1
                        
                        # Extract text
                        text_found = False
                        for key in text_keys:
                            if key in record:
                                text_samples.append(str(record[key]))
                                text_found = True
                                break
                        
                        if not text_found:
                            missing_text_key_count += 1
                            
                    except json.JSONDecodeError:
                        malformed_count += 1
                    
                    if len(records) >= sample_size:
                        break
            
            stats["total_records"] = total_records
            stats["malformed_records"] = malformed_count
            stats["missing_text_key_records"] = missing_text_key_count
            stats["fields_found"] = dict(field_counter.most_common())
            stats["sampled_records"] = len(records)
            
            if text_samples:
                stats.update(self._analyze_text_samples(text_samples))
            
            if records:
                stats["sample_records"] = records[:5]  # First 5 records as examples
                
        except Exception as e:
            self.logger.error("data.inspector", f"Error inspecting JSONL file: {e}", exc_info=True)
            stats["error"] = str(e)
        
        return stats
    
    def _inspect_json(self, filepath: Path, text_keys: Union[str, List[str]], 
                      sample_size: int) -> Dict[str, Any]:
        """Inspect JSON file (array format)."""
        if isinstance(text_keys, str):
            text_keys = [text_keys]
        
        stats = {
            "file_type": "json",
            "filepath": str(filepath),
            "file_size_bytes": filepath.stat().st_size,
            "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2)
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                stats["error"] = "JSON file is not an array"
                return stats
            
            stats["total_records"] = len(data)
            
            text_samples = []
            field_counter = Counter()
            missing_text_key_count = 0
            
            for i, record in enumerate(data[:sample_size]):
                # Count fields
                if isinstance(record, dict):
                    for key in record.keys():
                        field_counter[key] += 1
                    
                    # Extract text
                    text_found = False
                    for key in text_keys:
                        if key in record:
                            text_samples.append(str(record[key]))
                            text_found = True
                            break
                    
                    if not text_found:
                        missing_text_key_count += 1
            
            stats["fields_found"] = dict(field_counter.most_common())
            stats["missing_text_key_records"] = missing_text_key_count
            stats["sampled_records"] = min(sample_size, len(data))
            
            if text_samples:
                stats.update(self._analyze_text_samples(text_samples))
            
            if data:
                stats["sample_records"] = data[:5]  # First 5 records
                
        except Exception as e:
            self.logger.error("data.inspector", f"Error inspecting JSON file: {e}", exc_info=True)
            stats["error"] = str(e)
        
        return stats
    
    def _inspect_parquet(self, filepath: Path, text_keys: Union[str, List[str]], 
                         sample_size: int) -> Dict[str, Any]:
        """Inspect Parquet file."""
        if not PANDAS_AVAILABLE:
            return {"error": "pandas/pyarrow not available"}
        
        if isinstance(text_keys, str):
            text_keys = [text_keys]
        
        stats = {
            "file_type": "parquet",
            "filepath": str(filepath),
            "file_size_bytes": filepath.stat().st_size,
            "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2)
        }
        
        try:
            df = pd.read_parquet(filepath)
            stats["total_records"] = len(df)
            stats["columns"] = list(df.columns)
            stats["column_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Memory usage
            stats["memory_usage_mb"] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            
            # Missing values
            stats["missing_values"] = df.isnull().sum().to_dict()
            
            # Sample data
            sample_df = df.head(sample_size)
            
            # Extract text samples
            text_samples = []
            text_col_found = False
            for key in text_keys:
                if key in df.columns:
                    text_samples = sample_df[key].astype(str).tolist()
                    text_col_found = True
                    break
            
            if not text_col_found:
                stats["warning"] = f"None of the text keys {text_keys} found in columns"
            
            if text_samples:
                stats.update(self._analyze_text_samples(text_samples))
            
            # Sample records
            stats["sample_records"] = sample_df.head(5).to_dict('records')
            
        except Exception as e:
            self.logger.error("data.inspector", f"Error inspecting Parquet file: {e}", exc_info=True)
            stats["error"] = str(e)
        
        return stats
    
    def _inspect_binary(self, filepath: Path, file_ext: str) -> Dict[str, Any]:
        """Inspect binary format files (npy, bin, pt)."""
        stats = {
            "file_type": file_ext.replace('.', ''),
            "filepath": str(filepath),
            "file_size_bytes": filepath.stat().st_size,
            "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2)
        }
        
        try:
            if file_ext == '.npy' and NUMPY_AVAILABLE:
                data = np.load(filepath)
                stats["shape"] = data.shape
                stats["dtype"] = str(data.dtype)
                stats["total_tokens"] = data.size
                stats["unique_tokens"] = len(np.unique(data))
                stats["min_token_id"] = int(data.min())
                stats["max_token_id"] = int(data.max())
                stats["sample_tokens"] = data[:100].tolist()
                
            elif file_ext == '.pt':
                try:
                    import torch
                    data = torch.load(filepath)
                    if isinstance(data, torch.Tensor):
                        stats["shape"] = list(data.shape)
                        stats["dtype"] = str(data.dtype)
                        stats["total_tokens"] = data.numel()
                        stats["unique_tokens"] = len(torch.unique(data))
                        stats["min_token_id"] = int(data.min())
                        stats["max_token_id"] = int(data.max())
                        stats["sample_tokens"] = data[:100].tolist()
                    else:
                        stats["data_type"] = str(type(data))
                except ImportError:
                    stats["error"] = "PyTorch not available"
                    
            elif file_ext == '.bin':
                # Try to load metadata for dtype
                metadata_path = filepath.with_suffix('.meta.json')
                bin_dtype = np.uint16 # Default assumption
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as meta_f:
                            metadata = json.load(meta_f)
                            if 'dtype' in metadata:
                                bin_dtype = np.dtype(metadata['dtype'])
                            if 'total_tokens' in metadata:
                                stats["total_tokens"] = metadata['total_tokens']
                            if 'shape' in metadata:
                                stats["shape"] = metadata['shape']
                    except Exception as meta_e:
                        self.logger.warning("data.inspector", f"Could not load metadata for {filepath}: {meta_e}")

                if "total_tokens" not in stats: # Fallback if not in metadata
                    file_size = filepath.stat().st_size
                    stats["total_tokens"] = file_size // bin_dtype.itemsize # Use determined dtype itemsize
                
                with open(filepath, 'rb') as f:
                    # Read enough bytes for up to 100 tokens of the determined dtype
                    bytes_to_read = min(filepath.stat().st_size, 100 * bin_dtype.itemsize)
                    sample_bytes = f.read(bytes_to_read)
                    
                    if sample_bytes:
                        sample_tokens_np = np.frombuffer(sample_bytes, dtype=bin_dtype)
                        sample_tokens = sample_tokens_np.tolist()
                        stats["sample_tokens"] = sample_tokens
                        stats["min_token_id"] = int(sample_tokens_np.min())
                        stats["max_token_id"] = int(sample_tokens_np.max())
                    else:
                        stats["sample_tokens"] = []
                        stats["min_token_id"] = 0
                        stats["max_token_id"] = 0
                stats["dtype"] = str(bin_dtype) # Add actual dtype used
                    
                    
        except Exception as e:
            self.logger.error("data.inspector", f"Error inspecting binary file: {e}", exc_info=True)
            stats["error"] = str(e)
        
        return stats
    
    def _analyze_text_samples(self, text_samples: List[str]) -> Dict[str, Any]:
        """Analyze text samples for quality metrics."""
        if not text_samples:
            return {}
        
        stats = {}
        
        # Length statistics
        lengths = [len(text) for text in text_samples]
        stats["text_length_stats"] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
            "median": sorted(lengths)[len(lengths) // 2]
        }
        
        # Character and word counts
        total_chars = sum(lengths)
        total_words = sum(len(text.split()) for text in text_samples)
        
        stats["avg_chars_per_sample"] = round(total_chars / len(text_samples), 2)
        stats["avg_words_per_sample"] = round(total_words / len(text_samples), 2)
        stats["total_samples_analyzed"] = len(text_samples)
        
        # Language detection (simple heuristic)
        ascii_count = sum(1 for text in text_samples if text.isascii())
        stats["ascii_ratio"] = round(ascii_count / len(text_samples), 3)
        
        # Detect potential issues
        empty_count = sum(1 for text in text_samples if not text.strip())
        stats["empty_samples"] = empty_count
        
        # Repetition detection (simple)
        unique_samples = len(set(text_samples))
        stats["unique_samples"] = unique_samples
        stats["duplicate_ratio"] = round(1 - (unique_samples / len(text_samples)), 3)
        
        # Special character analysis
        special_char_pattern = re.compile(r'[^\w\s]')
        special_char_counts = [len(special_char_pattern.findall(text)) for text in text_samples]
        stats["avg_special_chars"] = round(sum(special_char_counts) / len(text_samples), 2)
        
        # Number analysis
        number_pattern = re.compile(r'\d+')
        number_counts = [len(number_pattern.findall(text)) for text in text_samples]
        stats["avg_numbers"] = round(sum(number_counts) / len(text_samples), 2)
        
        # URL detection
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        urls_found = sum(1 for text in text_samples if url_pattern.search(text))
        stats["samples_with_urls"] = urls_found
        
        # Email detection
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        emails_found = sum(1 for text in text_samples if email_pattern.search(text))
        stats["samples_with_emails"] = emails_found
        
        return stats
    
    def inspect_directory(self, directory: Union[str, Path], 
                         text_keys: Union[str, List[str]] = 'text',
                         recursive: bool = True,
                         sample_size: int = 100) -> Dict[str, Any]:
        """
        Inspect all data files in a directory.
        
        Args:
            directory: Path to directory
            text_keys: Key(s) for text extraction
            recursive: Whether to search recursively
            sample_size: Sample size for each file
            
        Returns:
            Dictionary with comprehensive statistics
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        self.logger.info("data.inspector", f"Inspecting directory: {directory}")
        
        # Collect all supported files
        supported_extensions = {'.txt', '.json', '.jsonl', '.parquet', '.npy', '.bin', '.pt'}
        files = []
        
        if recursive:
            for ext in supported_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        self.logger.info("data.inspector", f"Found {len(files)} files to inspect")
        
        # Inspect each file
        file_stats = []
        summary = {
            "directory": str(directory),
            "total_files": len(files),
            "files_by_type": Counter(),
            "total_size_mb": 0,
            "total_records": 0,
            "file_details": []
        }
        
        for filepath in files:
            try:
                stats = self.inspect_file(filepath, text_keys, sample_size)
                file_stats.append(stats)
                
                # Update summary
                summary["files_by_type"][stats.get("file_type", "unknown")] += 1
                summary["total_size_mb"] += stats.get("file_size_mb", 0)
                summary["total_records"] += stats.get("total_records", 0)
                summary["file_details"].append({
                    "filename": filepath.name,
                    "type": stats.get("file_type"),
                    "size_mb": stats.get("file_size_mb"),
                    "records": stats.get("total_records", 0)
                })
                
            except Exception as e:
                self.logger.error("data.inspector", f"Error inspecting {filepath}: {e}")
        
        summary["files_by_type"] = dict(summary["files_by_type"])
        summary["total_size_mb"] = round(summary["total_size_mb"], 2)
        summary["detailed_stats"] = file_stats
        
        return summary
    
    def generate_report(self, stats: Dict[str, Any], output_path: Optional[Union[str, Path]] = None):
        """Generate a formatted inspection report."""
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info("data.inspector", f"Report saved to: {output_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("DATA INSPECTION REPORT")
        print("="*80 + "\n")
        
        if "directory" in stats:
            print(f"Directory: {stats['directory']}")
            print(f"Total Files: {stats['total_files']}")
            print(f"Total Size: {stats['total_size_mb']} MB")
            print(f"Total Records: {stats['total_records']}")
            print(f"\nFiles by Type: {stats['files_by_type']}")
        else:
            print(f"File: {stats.get('filepath', 'Unknown')}")
            print(f"Type: {stats.get('file_type', 'Unknown')}")
            print(f"Size: {stats.get('file_size_mb', 0)} MB")
            print(f"Records: {stats.get('total_records', 0)}")
        
        print("\n" + "="*80 + "\n")


def main():
    """CLI entry point for data inspector."""
    parser = argparse.ArgumentParser(description="Inspect and analyze data files for xorzen.")
    parser.add_argument("path", type=str, help="Path to file or directory to inspect")
    parser.add_argument("--text_keys", nargs='+', default=['text'], 
                       help="Key(s) to extract text from JSON/JSONL files")
    parser.add_argument("--sample_size", type=int, default=100,
                       help="Number of samples to analyze per file")
    parser.add_argument("--output", type=str, help="Output path for JSON report")
    parser.add_argument("--recursive", action="store_true", 
                       help="Recursively inspect directories")
    
    args = parser.parse_args()
    
    # Setup logger
    setup_global_logger(name="xorzen-data-inspector", log_dir="logs_inspector", 
                       level=LogLevel.INFO, enable_async=False)
    
    inspector = DataInspector()
    
    try:
        path = Path(args.path)
        
        if path.is_file():
            stats = inspector.inspect_file(path, args.text_keys, args.sample_size)
        elif path.is_dir():
            stats = inspector.inspect_directory(path, args.text_keys, args.recursive, args.sample_size)
        else:
            raise ValueError(f"Invalid path: {args.path}")
        
        inspector.generate_report(stats, args.output)
        
    except Exception as e:
        get_logger().critical("data.inspector", f"Inspection failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        get_logger().cleanup()


if __name__ == '__main__':
    main()

__all__ = ['DataInspector']

