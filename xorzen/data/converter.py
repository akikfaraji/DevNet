"""
Enhanced Data Converter Module for xorzen
Provides comprehensive format conversion with batch processing support and binary tokenized formats.

This module extends the existing DataConverter with:
- Binary format conversion (txt/json/jsonl/parquet -> .bin)
- Tokenization integration
- Efficient streaming processing
- Metadata generation
- Progress tracking
"""

import json
import os
import sys
import argparse
import struct
from typing import Union, Iterator, List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from xorzen.utils.logger import get_logger, setup_global_logger, LogLevel
from xorzen.exceptions import DataConversionError, DataFormatError

try:
    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow as pa
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    get_logger().warning("data.converter", "pandas and/or pyarrow not found. Parquet conversion will not be supported.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    get_logger().warning("data.converter", "numpy not found. Binary conversion will not be supported.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DataConverter:
    """
    Enhanced data format converter with batch processing and binary support.
    
    Supports conversions between:
    - Text formats: txt, json, jsonl, parquet
    - Binary formats: .bin (tokenized)
    - Numpy formats: .npy
    
    Example:
        >>> converter = DataConverter(tokenizer=my_tokenizer)
        >>> stats = converter.txt_to_bin('data.txt', 'data.bin')
        >>> print(f"Converted {stats['tokens_written']} tokens")
    """
    
    def __init__(self, max_workers: int = 4, tokenizer=None):
        self.logger = get_logger()
        self.max_workers = max_workers
        self.tokenizer = tokenizer
    
    # =========================================================================
    # TEXT FORMAT CONVERSIONS (EXISTING)
    # =========================================================================
    
    def convert_file(self, input_path: Union[str, Path], 
                    output_path: Union[str, Path],
                    input_format: Optional[str] = None,
                    output_format: Optional[str] = None,
                    text_column: str = 'text',
                    chunk_size: int = 10000) -> Dict[str, Any]:
        """
        Convert a file from one format to another.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            input_format: Input format (auto-detected if None)
            output_format: Output format (auto-detected if None)
            text_column: Column name for text data
            chunk_size: Chunk size for large file processing
            
        Returns:
            Conversion statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Auto-detect formats
        if input_format is None:
            input_format = input_path.suffix.lower().replace('.', '')
        if output_format is None:
            output_format = output_path.suffix.lower().replace('.', '')
        
        self.logger.info("data.converter", 
                        f"Converting {input_path} ({input_format}) to {output_path} ({output_format})")
        
        # Route to appropriate converter
        converter_key = f"{input_format}_to_{output_format}"
        
        converters = {
            # Text to text conversions
            "parquet_to_jsonl": self._parquet_to_jsonl,
            "parquet_to_json": self._parquet_to_json,
            "parquet_to_txt": self._parquet_to_txt,
            "jsonl_to_json": self._jsonl_to_json,
            "jsonl_to_parquet": self._jsonl_to_parquet,
            "jsonl_to_txt": self._jsonl_to_txt,
            "json_to_jsonl": self._json_to_jsonl,
            "json_to_parquet": self._json_to_parquet,
            "json_to_txt": self._json_to_txt,
            "txt_to_jsonl": self._txt_to_jsonl,
            "txt_to_json": self._txt_to_json,
            "txt_to_parquet": self._txt_to_parquet,
            # Binary conversions
            "txt_to_bin": self.txt_to_bin,
            "json_to_bin": self.json_to_bin,
            "jsonl_to_bin": self.jsonl_to_bin,
            "parquet_to_bin": self.parquet_to_bin,
            "txt_to_npy": self.txt_to_npy,
            "json_to_npy": self.json_to_npy,
            "jsonl_to_npy": self.jsonl_to_npy,
            "parquet_to_npy": self.parquet_to_npy,
        }
        
        if converter_key not in converters:
            raise DataConversionError(
                from_format=input_format,
                to_format=output_format,
                reason=f"Conversion not supported. Available: {list(converters.keys())}"
            )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return converters[converter_key](input_path, output_path, text_column, chunk_size)
    
    # =========================================================================
    # PARQUET CONVERSIONS
    # =========================================================================
    
    def _parquet_to_jsonl(self, input_path: Path, output_path: Path, 
                         text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert Parquet to JSONL."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas/pyarrow required for Parquet conversion")
        
        stats = {"input": str(input_path), "output": str(output_path), "records": 0}
        
        parquet_file = pq.ParquetFile(input_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
                    stats["records"] += 1
        
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    def _parquet_to_json(self, input_path: Path, output_path: Path, 
                        text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert Parquet to JSON array."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas/pyarrow required for Parquet conversion")
        
        df = pd.read_parquet(input_path)
        data = df.to_dict('records')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        stats = {"input": str(input_path), "output": str(output_path), "records": len(data)}
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    def _parquet_to_txt(self, input_path: Path, output_path: Path, 
                       text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert Parquet to text file."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas/pyarrow required for Parquet conversion")
        
        stats = {"input": str(input_path), "output": str(output_path), "lines": 0}
        
        parquet_file = pq.ParquetFile(input_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                if text_column in df.columns:
                    for text in df[text_column]:
                        f.write(str(text) + '\n')
                        stats["lines"] += 1
                else:
                    raise ValueError(f"Column '{text_column}' not found in Parquet file")
        
        self.logger.info("data.converter", f"Converted {stats['lines']} lines")
        return stats
    
    # =========================================================================
    # JSONL CONVERSIONS
    # =========================================================================
    
    def _jsonl_to_json(self, input_path: Path, output_path: Path, 
                      text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert JSONL to JSON array."""
        data = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        stats = {"input": str(input_path), "output": str(output_path), "records": len(data)}
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    def _jsonl_to_parquet(self, input_path: Path, output_path: Path, 
                         text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert JSONL to Parquet."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas/pyarrow required for Parquet conversion")
        
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        
        stats = {"input": str(input_path), "output": str(output_path), "records": len(data)}
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    def _jsonl_to_txt(self, input_path: Path, output_path: Path, 
                     text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert JSONL to text file."""
        stats = {"input": str(input_path), "output": str(output_path), "lines": 0}
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if text_column in record:
                            fout.write(str(record[text_column]) + '\n')
                            stats["lines"] += 1
                    except json.JSONDecodeError:
                        continue
        
        self.logger.info("data.converter", f"Converted {stats['lines']} lines")
        return stats
    
    # =========================================================================
    # JSON CONVERSIONS
    # =========================================================================
    
    def _json_to_jsonl(self, input_path: Path, output_path: Path, 
                      text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert JSON array to JSONL."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Input JSON must be an array")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        stats = {"input": str(input_path), "output": str(output_path), "records": len(data)}
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    def _json_to_parquet(self, input_path: Path, output_path: Path, 
                        text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert JSON array to Parquet."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas/pyarrow required for Parquet conversion")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Input JSON must be an array")
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        
        stats = {"input": str(input_path), "output": str(output_path), "records": len(data)}
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    def _json_to_txt(self, input_path: Path, output_path: Path, 
                    text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert JSON array to text file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Input JSON must be an array")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in data:
                if text_column in record:
                    f.write(str(record[text_column]) + '\n')
        
        stats = {"input": str(input_path), "output": str(output_path), "records": len(data)}
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    # =========================================================================
    # TEXT FILE CONVERSIONS
    # =========================================================================
    
    def _txt_to_jsonl(self, input_path: Path, output_path: Path, 
                     text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert text file to JSONL."""
        stats = {"input": str(input_path), "output": str(output_path), "records": 0}
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if line:
                    record = {text_column: line}
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats["records"] += 1
        
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    def _txt_to_json(self, input_path: Path, output_path: Path, 
                    text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert text file to JSON array."""
        data = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append({text_column: line})
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        stats = {"input": str(input_path), "output": str(output_path), "records": len(data)}
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    def _txt_to_parquet(self, input_path: Path, output_path: Path, 
                       text_column: str, chunk_size: int) -> Dict[str, Any]:
        """Convert text file to Parquet."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas/pyarrow required for Parquet conversion")
        
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append({text_column: line})
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        
        stats = {"input": str(input_path), "output": str(output_path), "records": len(data)}
        self.logger.info("data.converter", f"Converted {stats['records']} records")
        return stats
    
    # =========================================================================
    # BINARY CONVERSION METHODS (NEW - HIGH PRIORITY FOR xorzen)
    # =========================================================================
    
    def txt_to_bin(self, input_path: Union[str, Path], output_path: Union[str, Path],
                   text_column: str = 'text', chunk_size: int = 10000,
                   tokenizer=None, max_length: Optional[int] = None,
                   stride: Optional[int] = None, add_special_tokens: bool = True,
                   show_progress: bool = True) -> Dict[str, Any]:
        """
        Convert text file to binary tokenized format (.bin).
        
        This is a critical method for xorzen's data pipeline - converts raw text
        into tokenized binary format for fast training data loading.
        
        Args:
            input_path: Input text file
            output_path: Output .bin file
            text_column: Unused for txt (kept for API consistency)
            chunk_size: Unused for txt (kept for API consistency)
            tokenizer: Tokenizer instance (uses self.tokenizer if None)
            max_length: Maximum sequence length (None = continuous stream)
            stride: Stride for overlapping sequences
            add_special_tokens: Add BOS/EOS tokens
            show_progress: Show progress bar
            
        Returns:
            Conversion statistics including tokens_written, sequences_created
            
        Example:
            >>> converter = DataConverter(tokenizer=my_tok)
            >>> stats = converter.txt_to_bin('train.txt', 'train.bin', max_length=2048)
            >>> print(f"Created {stats['sequences_created']} sequences")
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for binary conversion. Install: pip install numpy")
        
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError(
                "Tokenizer required for binary conversion. "
                "Pass tokenizer argument or set converter.tokenizer"
            )
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("data.converter", 
                        f"Converting {input_path} to binary format (tokenized)")
        
        stats = {
            "input": str(input_path),
            "output": str(output_path),
            "lines_read": 0,
            "tokens_written": 0,
            "sequences_created": 0,
            "file_size_bytes": 0
        }

        # --- Streaming tokenization: batch lines -> write chunks, never hold all tokens in RAM ---
        BATCH_SIZE = 512   # lines per tokenization batch
        FLUSH_TOKENS = 1_000_000  # write to disk every ~1M tokens (~2MB at uint16)

        token_dtype = np.uint16
        token_buffer: List[int] = []
        line_batch: List[str] = []

        def _flush(buf: List[int], f_out) -> int:
            """Write buffer to the raw binary stream; return number of tokens written."""
            if not buf:
                return 0
            chunk = np.array(buf, dtype=token_dtype)
            chunk.tofile(f_out)
            return len(buf)

        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
                 open(output_path, 'wb') as f_out:

                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue

                    line_batch.append(line)
                    stats["lines_read"] += 1

                    if len(line_batch) >= BATCH_SIZE:
                        batch_tokens = self._tokenize_batch(line_batch, tokenizer, add_special_tokens)
                        for toks in batch_tokens:
                            token_buffer.extend(toks)
                        line_batch = []

                    # Flush to disk when buffer is large enough
                    if len(token_buffer) >= FLUSH_TOKENS:
                        stats["tokens_written"] += _flush(token_buffer, f_out)
                        token_buffer = []

                        if stats["lines_read"] % 10000 == 0:
                            self.logger.debug("data.converter",
                                             f"{stats['lines_read']} lines, "
                                             f"{stats['tokens_written']:,} tokens written")

                # Flush remaining lines
                if line_batch:
                    batch_tokens = self._tokenize_batch(line_batch, tokenizer, add_special_tokens)
                    for toks in batch_tokens:
                        token_buffer.extend(toks)

                # Final flush
                stats["tokens_written"] += _flush(token_buffer, f_out)

        except Exception as e:
            raise DataConversionError(
                from_format='txt',
                to_format='bin',
                reason=f"Failed to read/tokenize file: {e}"
            )

        # --- Reshape into sequences if max_length is requested ---
        # Re-read the flat file we just wrote and reshape in-place (no double RAM).
        if max_length and stats["tokens_written"] > 0:
            stride = stride or max_length
            flat = np.fromfile(str(output_path), dtype=token_dtype)
            sequences = self._create_sequences(flat, max_length, stride)
            if sequences:
                token_array = np.stack(sequences)
                token_array.tofile(str(output_path))
                stats["sequences_created"] = len(sequences)
                stats["tokens_written"] = int(token_array.size)
            else:
                token_array = flat.reshape(1, -1)  # single continuous shard
                stats["sequences_created"] = 1
        else:
            token_array = np.fromfile(str(output_path), dtype=token_dtype)
            stats["sequences_created"] = 1

        stats["file_size_bytes"] = output_path.stat().st_size
        self._save_metadata(output_path, token_array, stats, max_length, stride, input_path)

        self.logger.info("data.converter",
                        f"Converted {stats['lines_read']} lines to "
                        f"{stats['tokens_written']:,} tokens "
                        f"({stats['sequences_created']} sequences)")

        return stats
    
    def json_to_bin(self, input_path: Union[str, Path], output_path: Union[str, Path],
                    text_column: str = 'text', chunk_size: int = 10000,
                    tokenizer=None, max_length: Optional[int] = None,
                    stride: Optional[int] = None, add_special_tokens: bool = True,
                    show_progress: bool = True) -> Dict[str, Any]:
        """
        Convert JSON file to binary tokenized format (.bin).
        
        Args:
            input_path: Input JSON file (must be array or have array field)
            output_path: Output .bin file
            text_column: Field containing text data
            chunk_size: Unused (kept for API consistency)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
            add_special_tokens: Add BOS/EOS tokens
            show_progress: Show progress bar
            
        Returns:
            Conversion statistics
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for binary conversion")
        
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer required for binary conversion")
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("data.converter", 
                        f"Converting {input_path} (JSON) to binary format")
        
        stats = {
            "input": str(input_path),
            "output": str(output_path),
            "records_read": 0,
            "tokens_written": 0,
            "sequences_created": 0,
            "file_size_bytes": 0
        }
        
        # Load JSON
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise DataConversionError(
                from_format='json',
                to_format='bin',
                reason=f"Failed to load JSON: {e}"
            )
        
        # Handle different JSON structures
        if not isinstance(data, list):
            # Try to find array field
            found_array = False
            for key, value in data.items():
                if isinstance(value, list):
                    data = value
                    found_array = True
                    break
            
            if not found_array:
                if isinstance(data, dict) and (text_column in data or 'text' in data):
                    # Single record
                    data = [data]
                else:
                    raise DataFormatError(
                        format_type='json',
                        reason="JSON must be an array or contain an array field",
                        supported_formats=['json array', 'json with data field']
                    )
        
        all_tokens = []
        
        # Process records
        for record in data:
            text = self._extract_text_from_record(record, text_column)
            if not text:
                continue
            
            tokens = self._tokenize_text(text, tokenizer, add_special_tokens)
            all_tokens.extend(tokens)
            stats["records_read"] += 1
        
        # Convert and save
        token_array = np.array(all_tokens, dtype=np.uint16)
        
        if max_length:
            stride = stride or max_length
            sequences = self._create_sequences(token_array, max_length, stride)
            if sequences:
                token_array = np.stack(sequences)
                stats["sequences_created"] = len(sequences)
        else:
            stats["sequences_created"] = 1
        
        token_array.tofile(str(output_path))
        stats["tokens_written"] = int(token_array.size)
        stats["file_size_bytes"] = output_path.stat().st_size
        
        self._save_metadata(output_path, token_array, stats, max_length, stride, input_path)
        
        self.logger.info("data.converter", 
                        f"Converted {stats['records_read']} records to "
                        f"{stats['tokens_written']} tokens")
        
        return stats
    
    def jsonl_to_bin(self, input_path: Union[str, Path], output_path: Union[str, Path],
                     text_column: str = 'text', chunk_size: int = 10000,
                     tokenizer=None, max_length: Optional[int] = None,
                     stride: Optional[int] = None, add_special_tokens: bool = True,
                     show_progress: bool = True) -> Dict[str, Any]:
        """
        Convert JSONL file to binary tokenized format (.bin).
        
        This is a critical method for xorzen - handles large JSONL datasets efficiently
        with streaming processing.
        
        Args:
            input_path: Input JSONL file
            output_path: Output .bin file
            text_column: Field containing text data
            chunk_size: Process in chunks for memory efficiency
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
            add_special_tokens: Add BOS/EOS tokens
            show_progress: Show progress bar
            
        Returns:
            Conversion statistics
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for binary conversion")
        
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer required for binary conversion")
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("data.converter", 
                        f"Converting {input_path} (JSONL) to binary format")
        
        stats = {
            "input": str(input_path),
            "output": str(output_path),
            "lines_read": 0,
            "valid_records": 0,
            "tokens_written": 0,
            "sequences_created": 0,
            "file_size_bytes": 0
        }
        
        BATCH_SIZE = 512
        FLUSH_TOKENS = 1_000_000
        token_dtype = np.uint16
        token_buffer: List[int] = []
        text_batch: List[str] = []

        def _flush(buf: List[int], f_out) -> int:
            if not buf:
                return 0
            np.array(buf, dtype=token_dtype).tofile(f_out)
            return len(buf)

        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
                 open(output_path, 'wb') as f_out:

                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    stats["lines_read"] += 1

                    try:
                        record = json.loads(line)
                        text = self._extract_text_from_record(record, text_column)
                        if not text:
                            continue
                        text_batch.append(text)
                        stats["valid_records"] += 1
                    except json.JSONDecodeError:
                        continue

                    if len(text_batch) >= BATCH_SIZE:
                        for toks in self._tokenize_batch(text_batch, tokenizer, add_special_tokens):
                            token_buffer.extend(toks)
                        text_batch = []

                    if len(token_buffer) >= FLUSH_TOKENS:
                        stats["tokens_written"] += _flush(token_buffer, f_out)
                        token_buffer = []
                        if stats["valid_records"] % 10000 == 0:
                            self.logger.debug("data.converter",
                                             f"{stats['valid_records']} records, "
                                             f"{stats['tokens_written']:,} tokens written")

                if text_batch:
                    for toks in self._tokenize_batch(text_batch, tokenizer, add_special_tokens):
                        token_buffer.extend(toks)
                stats["tokens_written"] += _flush(token_buffer, f_out)

        except Exception as e:
            raise DataConversionError(
                from_format='jsonl',
                to_format='bin',
                reason=f"Failed to process JSONL: {e}"
            )

        if max_length and stats["tokens_written"] > 0:
            stride = stride or max_length
            flat = np.fromfile(str(output_path), dtype=token_dtype)
            sequences = self._create_sequences(flat, max_length, stride)
            if sequences:
                token_array = np.stack(sequences)
                token_array.tofile(str(output_path))
                stats["sequences_created"] = len(sequences)
                stats["tokens_written"] = int(token_array.size)
            else:
                token_array = flat
                stats["sequences_created"] = 1
        else:
            token_array = np.fromfile(str(output_path), dtype=token_dtype)
            stats["sequences_created"] = 1

        stats["file_size_bytes"] = output_path.stat().st_size
        self._save_metadata(output_path, token_array, stats, max_length, stride, input_path)

        self.logger.info("data.converter",
                        f"Converted {stats['valid_records']}/{stats['lines_read']} records to "
                        f"{stats['tokens_written']:,} tokens "
                        f"({stats['sequences_created']} sequences)")

        return stats
    
    def parquet_to_bin(self, input_path: Union[str, Path], output_path: Union[str, Path],
                       text_column: str = 'text', chunk_size: int = 10000,
                       tokenizer=None, max_length: Optional[int] = None,
                       stride: Optional[int] = None, add_special_tokens: bool = True,
                       show_progress: bool = True) -> Dict[str, Any]:
        """
        Convert Parquet file to binary tokenized format (.bin).
        
        Efficiently handles large Parquet datasets with batch processing.
        
        Args:
            input_path: Input Parquet file
            output_path: Output .bin file
            text_column: Column containing text data
            chunk_size: Batch size for processing
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
            add_special_tokens: Add BOS/EOS tokens
            show_progress: Show progress bar
            
        Returns:
            Conversion statistics
        """
        if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
            raise ImportError("pandas and numpy required for Parquet to binary conversion")
        
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer required for binary conversion")
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("data.converter", 
                        f"Converting {input_path} (Parquet) to binary format")
        
        stats = {
            "input": str(input_path),
            "output": str(output_path),
            "records_read": 0,
            "tokens_written": 0,
            "sequences_created": 0,
            "file_size_bytes": 0
        }
        
        all_tokens = []
        
        # Read Parquet in chunks (efficient for large files)
        try:
            parquet_file = pq.ParquetFile(input_path)
            
            for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
                df = batch.to_pandas()
                
                if text_column not in df.columns:
                    raise DataFormatError(
                        format_type='parquet',
                        reason=f"Column '{text_column}' not found. Available: {list(df.columns)}",
                        supported_formats=['parquet with text column']
                    )
                
                for text in df[text_column]:
                    if pd.isna(text) or not text:
                        continue
                    
                    text = str(text)
                    tokens = self._tokenize_text(text, tokenizer, add_special_tokens)
                    all_tokens.extend(tokens)
                    stats["records_read"] += 1
                
                # Periodic progress log
                if (batch_idx + 1) % 10 == 0:
                    self.logger.debug("data.converter", 
                                    f"Processed {batch_idx + 1} batches, "
                                    f"{stats['records_read']} records, "
                                    f"{len(all_tokens)} tokens")
        
        except Exception as e:
            raise DataConversionError(
                from_format='parquet',
                to_format='bin',
                reason=f"Failed to process Parquet: {e}"
            )
        
        # Convert and save
        token_array = np.array(all_tokens, dtype=np.uint16)
        
        if max_length:
            stride = stride or max_length
            sequences = self._create_sequences(token_array, max_length, stride)
            if sequences:
                token_array = np.stack(sequences)
                stats["sequences_created"] = len(sequences)
        else:
            stats["sequences_created"] = 1
        
        token_array.tofile(str(output_path))
        stats["tokens_written"] = int(token_array.size)
        stats["file_size_bytes"] = output_path.stat().st_size
        
        self._save_metadata(output_path, token_array, stats, max_length, stride, input_path)
        
        self.logger.info("data.converter", 
                        f"Converted {stats['records_read']} records to "
                        f"{stats['tokens_written']} tokens")
        
        return stats
    
    # =========================================================================
    # NUMPY FORMAT CONVERSIONS (.npy)
    # =========================================================================
    
    def txt_to_npy(self, input_path: Union[str, Path], output_path: Union[str, Path],
                   text_column: str = 'text', chunk_size: int = 10000,
                   **kwargs) -> Dict[str, Any]:
        """Convert text to numpy array format (.npy)."""
        # Use txt_to_bin then convert
        temp_bin = output_path.with_suffix('.temp.bin')
        stats = self.txt_to_bin(input_path, temp_bin, text_column, chunk_size, **kwargs)
        
        # Load bin and save as npy
        metadata_path = temp_bin.with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        shape = metadata['shape']
        dtype = metadata['dtype']
        
        token_array = np.fromfile(str(temp_bin), dtype=dtype)
        # Reshape to the original shape recorded in metadata
        token_array = token_array.reshape(shape)
        np.save(str(output_path), token_array)
        
        # Cleanup temp files
        temp_bin.unlink()
        metadata_path.unlink()
        
        stats['output'] = str(output_path)
        return stats
    
    def json_to_npy(self, input_path: Union[str, Path], output_path: Union[str, Path],
                    text_column: str = 'text', chunk_size: int = 10000,
                    **kwargs) -> Dict[str, Any]:
        """Convert JSON to numpy array format (.npy)."""
        temp_bin = output_path.with_suffix('.temp.bin')
        stats = self.json_to_bin(input_path, temp_bin, text_column, chunk_size, **kwargs)
        
        metadata_path = temp_bin.with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        shape = metadata['shape']
        dtype = metadata['dtype']
        
        token_array = np.fromfile(str(temp_bin), dtype=dtype)
        # Reshape to the original shape recorded in metadata
        token_array = token_array.reshape(shape)
        np.save(str(output_path), token_array)
        
        temp_bin.unlink()
        metadata_path.unlink()
        
        stats['output'] = str(output_path)
        return stats
    
    def jsonl_to_npy(self, input_path: Union[str, Path], output_path: Union[str, Path],
                     text_column: str = 'text', chunk_size: int = 10000,
                     **kwargs) -> Dict[str, Any]:
        """Convert JSONL to numpy array format (.npy)."""
        temp_bin = output_path.with_suffix('.temp.bin')
        stats = self.jsonl_to_bin(input_path, temp_bin, text_column, chunk_size, **kwargs)
        
        metadata_path = temp_bin.with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        shape = metadata['shape']
        dtype = metadata['dtype']
        
        token_array = np.fromfile(str(temp_bin), dtype=dtype)
        # Reshape to the original shape recorded in metadata
        token_array = token_array.reshape(shape)
        np.save(str(output_path), token_array)
        
        temp_bin.unlink()
        metadata_path.unlink()
        
        stats['output'] = str(output_path)
        return stats
    
    def parquet_to_npy(self, input_path: Union[str, Path], output_path: Union[str, Path],
                       text_column: str = 'text', chunk_size: int = 10000,
                       **kwargs) -> Dict[str, Any]:
        """Convert Parquet to numpy array format (.npy)."""
        temp_bin = output_path.with_suffix('.temp.bin')
        stats = self.parquet_to_bin(input_path, temp_bin, text_column, chunk_size, **kwargs)
        
        metadata_path = temp_bin.with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        shape = metadata['shape']
        dtype = metadata['dtype']
        
        token_array = np.fromfile(str(temp_bin), dtype=dtype)
        # Reshape to the original shape recorded in metadata
        token_array = token_array.reshape(shape)
        np.save(str(output_path), token_array)
        
        temp_bin.unlink()
        metadata_path.unlink()
        
        stats['output'] = str(output_path)
        return stats
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _tokenize_text(self, text: str, tokenizer, add_special_tokens: bool) -> List[int]:
        """
        Tokenize a single text string.
        Prefer _tokenize_batch() for bulk work — this is a single-item fallback.
        """
        results = self._tokenize_batch([text], tokenizer, add_special_tokens)
        return results[0] if results else []

    def _tokenize_batch(self, texts: List[str], tokenizer, add_special_tokens: bool) -> List[List[int]]:
        """
        Tokenize a batch of texts efficiently.

        Uses encode_batch() when available (tokenizers / HuggingFace fast tokenizers),
        which is 10-100x faster than calling encode() in a Python loop.
        Falls back to a loop for slow/custom tokenizers.
        """
        try:
            # --- Fast path: batch encode (tokenizers library / HF fast tokenizers) ---
            if hasattr(tokenizer, 'encode_batch'):
                results = tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
                # tokenizers library returns Encoding objects with .ids
                if results and hasattr(results[0], 'ids'):
                    return [r.ids for r in results]
                return [list(r) for r in results]

            # --- HuggingFace slow tokenizer with batch support ---
            if hasattr(tokenizer, '__call__'):
                out = tokenizer(texts, add_special_tokens=add_special_tokens)
                if hasattr(out, 'input_ids'):
                    return [list(ids) for ids in out['input_ids']]

            # --- Fallback: loop over encode() ---
            output = []
            for text in texts:
                if hasattr(tokenizer, 'encode'):
                    result = tokenizer.encode(text, add_special_tokens=add_special_tokens)
                    if hasattr(result, 'ids'):
                        output.append(result.ids)
                    else:
                        output.append(list(result))
                elif hasattr(tokenizer, 'tokenize'):
                    output.append(tokenizer.tokenize(text))
                else:
                    raise ValueError(
                        "Tokenizer must have encode_batch(), encode(), or tokenize(). "
                        "Supported: HuggingFace (fast/slow), tokenizers library, custom."
                    )
            return output

        except Exception as e:
            self.logger.error("data.converter", f"Batch tokenization failed: {e}")
            return [[] for _ in texts]
    
    def _extract_text_from_record(self, record: Union[str, Dict], text_field: str) -> str:
        """Extract text from a record (handles different formats)."""
        if isinstance(record, str):
            return record
        
        if isinstance(record, dict):
            # Try specified field
            text = record.get(text_field, '')
            if text:
                return str(text)
            
            # Try alternative common fields
            for field in ['text', 'content', 'body', 'message', 'prompt']:
                text = record.get(field, '')
                if text:
                    return str(text)
        
        return ''
    
    def _create_sequences(self, token_array: np.ndarray, max_length: int, 
                         stride: int) -> List[np.ndarray]:
        """
        Create fixed-length sequences from token array with optional overlap.
        
        Args:
            token_array: 1D array of tokens
            max_length: Maximum sequence length
            stride: Step size (stride < max_length creates overlap)
            
        Returns:
            List of sequences
        """
        sequences = []
        
        for i in range(0, len(token_array), stride):
            seq = token_array[i:i + max_length]
            
            if len(seq) == 0:
                continue
            
            # Pad if necessary (only for last sequence)
            if len(seq) < max_length:
                if i + stride >= len(token_array):  # Last sequence
                    pad_token = 0  # Padding token ID
                    seq = np.pad(seq, (0, max_length - len(seq)), 
                               constant_values=pad_token)
                else:
                    # Skip incomplete sequences that aren't the last one
                    continue
            
            sequences.append(seq)
        
        return sequences
    
    def _save_metadata(self, output_path: Path, token_array: np.ndarray, 
                      stats: Dict, max_length: Optional[int], 
                      stride: Optional[int], source_file: Path):
        """Save metadata JSON alongside binary file."""
        metadata_path = output_path.with_suffix('.meta.json')
        
        metadata = {
            "dtype": str(token_array.dtype),
            "shape": list(token_array.shape),
            "total_tokens": int(stats["tokens_written"]),
            "sequences": int(stats["sequences_created"]),
            "max_length": max_length,
            "stride": stride,
            "source_file": str(source_file),
            "output_file": str(output_path),
            "file_size_bytes": int(stats["file_size_bytes"]),
            "records_processed": stats.get("records_read", stats.get("lines_read", 0)),
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.debug("data.converter", f"Metadata saved to {metadata_path}")
    
    # =========================================================================
    # DIRECTORY BATCH CONVERSION
    # =========================================================================
    
    def convert_directory(self, input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         input_format: str,
                         output_format: str,
                         text_column: str = 'text',
                         recursive: bool = True,
                         parallel: bool = True,
                         **kwargs) -> Dict[str, Any]:
        """
        Convert all files in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            input_format: Input file format (e.g., 'txt', 'json', 'jsonl')
            output_format: Output file format (e.g., 'bin', 'npy', 'jsonl')
            text_column: Text column/field name
            recursive: Process subdirectories recursively
            parallel: Use parallel processing (faster for many files)
            **kwargs: Additional arguments passed to converters (e.g., tokenizer, max_length)
            
        Returns:
            Conversion statistics with per-file details
            
        Example:
            >>> converter = DataConverter(tokenizer=my_tok)
            >>> stats = converter.convert_directory(
            ...     'raw_data/',
            ...     'processed/',
            ...     'txt',
            ...     'bin',
            ...     max_length=2048,
            ...     parallel=True
            ... )
            >>> print(f"Converted {stats['successful']}/{stats['total_files']} files")
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all input files
        pattern = f"*.{input_format}"
        if recursive:
            files = list(input_dir.rglob(pattern))
        else:
            files = list(input_dir.glob(pattern))
        
        self.logger.info("data.converter", 
                        f"Found {len(files)} {input_format} files to convert")
        
        if len(files) == 0:
            self.logger.warning("data.converter", f"No {input_format} files found in {input_dir}")
            return {
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "file_details": []
            }
        
        stats = {
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "file_details": [],
            "total_tokens": 0,
            "total_sequences": 0
        }
        
        # Process files
        if parallel and len(files) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                for filepath in files:
                    relative_path = filepath.relative_to(input_dir)
                    output_path = output_dir / relative_path.with_suffix(f'.{output_format}')
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    future = executor.submit(
                        self.convert_file,
                        filepath, output_path,
                        input_format, output_format,
                        text_column, kwargs.get('chunk_size', 10000)
                    )
                    futures[future] = (filepath, output_path)
                
                # Collect results
                for future in as_completed(futures):
                    filepath, output_path = futures[future]
                    try:
                        file_stats = future.result()
                        stats["successful"] += 1
                        stats["total_tokens"] += file_stats.get("tokens_written", 0)
                        stats["total_sequences"] += file_stats.get("sequences_created", 0)
                        
                        stats["file_details"].append({
                            "input": str(filepath),
                            "output": str(output_path),
                            "status": "success",
                            **file_stats
                        })
                    except Exception as e:
                        self.logger.error("data.converter", 
                                        f"Failed to convert {filepath}: {e}")
                        stats["failed"] += 1
                        stats["file_details"].append({
                            "input": str(filepath),
                            "status": "failed",
                            "error": str(e)
                        })
        
        else:
            # Sequential processing
            for filepath in files:
                try:
                    relative_path = filepath.relative_to(input_dir)
                    output_path = output_dir / relative_path.with_suffix(f'.{output_format}')
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    file_stats = self.convert_file(
                        filepath, output_path,
                        input_format, output_format,
                        text_column, kwargs.get('chunk_size', 10000)
                    )
                    
                    stats["successful"] += 1
                    stats["total_tokens"] += file_stats.get("tokens_written", 0)
                    stats["total_sequences"] += file_stats.get("sequences_created", 0)
                    
                    stats["file_details"].append({
                        "input": str(filepath),
                        "output": str(output_path),
                        "status": "success",
                        **file_stats
                    })
                
                except Exception as e:
                    self.logger.error("data.converter", 
                                    f"Failed to convert {filepath}: {e}")
                    stats["failed"] += 1
                    stats["file_details"].append({
                        "input": str(filepath),
                        "status": "failed",
                        "error": str(e)
                    })
        
        self.logger.info("data.converter", 
                        f"Batch conversion complete: {stats['successful']} successful, "
                        f"{stats['failed']} failed. Total tokens: {stats['total_tokens']:,}")
        
        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS (For backward compatibility and ease of use)
# =============================================================================

def txt_to_bin(input_path: Union[str, Path], output_path: Union[str, Path],
               tokenizer, **kwargs) -> Dict[str, Any]:
    """
    Convenience function: Convert text file to binary format.
    
    Example:
        >>> from xorzen.data import txt_to_bin
        >>> from xorzen.tokenizer import load_pretrained
        >>> tok = load_pretrained('xorzen_32k')
        >>> stats = txt_to_bin('train.txt', 'train.bin', tok, max_length=2048)
    """
    converter = DataConverter(tokenizer=tokenizer)
    return converter.txt_to_bin(input_path, output_path, **kwargs)


def json_to_bin(input_path: Union[str, Path], output_path: Union[str, Path],
                tokenizer, text_field: str = 'text', **kwargs) -> Dict[str, Any]:
    """Convenience function: Convert JSON to binary format."""
    converter = DataConverter(tokenizer=tokenizer)
    return converter.json_to_bin(input_path, output_path, text_column=text_field, **kwargs)


def jsonl_to_bin(input_path: Union[str, Path], output_path: Union[str, Path],
                 tokenizer, text_field: str = 'text', **kwargs) -> Dict[str, Any]:
    """Convenience function: Convert JSONL to binary format."""
    converter = DataConverter(tokenizer=tokenizer)
    return converter.jsonl_to_bin(input_path, output_path, text_column=text_field, **kwargs)


def parquet_to_bin(input_path: Union[str, Path], output_path: Union[str, Path],
                   tokenizer, text_column: str = 'text', **kwargs) -> Dict[str, Any]:
    """Convenience function: Convert Parquet to binary format."""
    converter = DataConverter(tokenizer=tokenizer)
    return converter.parquet_to_bin(input_path, output_path, text_column=text_column, **kwargs)


# Legacy function wrappers (maintain backward compatibility)
def parquet_to_jsonl(input_filepath: Union[str, Path], output_filepath: Union[str, Path], 
                    text_column: str = 'text'):
    """Legacy wrapper for parquet to jsonl conversion."""
    converter = DataConverter()
    return converter.convert_file(input_filepath, output_filepath, 'parquet', 'jsonl', text_column)


def jsonl_to_json(input_filepath: Union[str, Path], output_filepath: Union[str, Path]):
    """Legacy wrapper for jsonl to json conversion."""
    converter = DataConverter()
    return converter.convert_file(input_filepath, output_filepath, 'jsonl', 'json')


def json_to_jsonl(input_filepath: Union[str, Path], output_filepath: Union[str, Path]):
    """Legacy wrapper for json to jsonl conversion."""
    converter = DataConverter()
    return converter.convert_file(input_filepath, output_filepath, 'json', 'jsonl')


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for data converter."""
    parser = argparse.ArgumentParser(
        description="xorzen Data Converter - Convert between text and binary formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert text to binary
  python -m xorzen.data.converter data.txt data.bin --output_format bin --tokenizer xorzen_32k
  
  # Convert directory of JSONL files to binary
  python -m xorzen.data.converter raw/ processed/ --input_format jsonl --output_format bin --recursive
  
  # Convert with specific sequence length
  python -m xorzen.data.converter train.txt train.bin --output_format bin --max_length 2048
        """
    )
    
    parser.add_argument("input", type=str, help="Input file or directory path")
    parser.add_argument("output", type=str, help="Output file or directory path")
    
    parser.add_argument("--input_format", type=str, 
                       help="Input format (txt/json/jsonl/parquet, auto-detect if not specified)")
    parser.add_argument("--output_format", type=str, 
                       help="Output format (bin/npy/txt/json/jsonl/parquet)")
    
    parser.add_argument("--text_column", type=str, default="text",
                       help="Column/field name for text content (default: text)")
    parser.add_argument("--tokenizer", type=str,
                       help="Tokenizer name or path (required for binary conversion)")
    
    parser.add_argument("--max_length", type=int,
                       help="Maximum sequence length for binary conversion")
    parser.add_argument("--stride", type=int,
                       help="Stride for overlapping sequences (default: max_length)")
    parser.add_argument("--no_special_tokens", action="store_true",
                       help="Don't add special tokens (BOS/EOS)")
    
    parser.add_argument("--chunk_size", type=int, default=10000,
                       help="Chunk size for processing large files (default: 10000)")
    parser.add_argument("--recursive", action="store_true",
                       help="Process directories recursively")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel processing for directories")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    
    parser.add_argument("--report", type=str, 
                       help="Save conversion report to JSON file")
    parser.add_argument("--log_dir", type=str, default="logs_converter",
                       help="Log directory (default: logs_converter)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logger
    log_level = LogLevel.DEBUG if args.verbose else LogLevel.INFO
    setup_global_logger(
        name="xorzen-data-converter", 
        log_dir=args.log_dir,
        level=log_level, 
        enable_async=False
    )
    
    # Load tokenizer if needed
    tokenizer = None
    if args.tokenizer and (args.output_format in ['bin', 'npy']):
        try:
            from xorzen.tokenizer import load_pretrained, load_from_path
            
            # Try loading as pretrained first
            try:
                tokenizer = load_pretrained(args.tokenizer)
                get_logger().info("data.converter", 
                                f"Loaded pretrained tokenizer: {args.tokenizer}")
            except:
                # Try loading from path
                tokenizer = load_from_path(args.tokenizer)
                get_logger().info("data.converter", 
                                f"Loaded tokenizer from: {args.tokenizer}")
        except Exception as e:
            get_logger().error("data.converter", f"Failed to load tokenizer: {e}")
            sys.exit(1)
    elif args.output_format in ['bin', 'npy'] and not args.tokenizer:
        get_logger().error("data.converter", 
                          "Tokenizer required for binary conversion. Use --tokenizer argument")
        sys.exit(1)
    
    # Create converter
    converter = DataConverter(max_workers=args.workers, tokenizer=tokenizer)
    
    try:
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        # Prepare kwargs for binary conversion
        binary_kwargs = {}
        if args.max_length:
            binary_kwargs['max_length'] = args.max_length
        if args.stride:
            binary_kwargs['stride'] = args.stride
        binary_kwargs['add_special_tokens'] = not args.no_special_tokens
        
        if input_path.is_file():
            # Single file conversion
            stats = converter.convert_file(
                input_path, output_path, 
                args.input_format, args.output_format,
                args.text_column, args.chunk_size
            )
        
        elif input_path.is_dir():
            # Directory conversion
            if not args.input_format or not args.output_format:
                raise ValueError(
                    "--input_format and --output_format required for directory conversion"
                )
            
            stats = converter.convert_directory(
                input_path, output_path,
                args.input_format, args.output_format,
                args.text_column, args.recursive, args.parallel,
                **binary_kwargs
            )
        
        else:
            raise ValueError(f"Invalid input path: {args.input}")
        
        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            get_logger().info("data.converter", f"Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("Conversion Summary")
        print("="*70)
        
        if "total_files" in stats:
            print(f"Total files: {stats['total_files']}")
            print(f"Successful: {stats['successful']}")
            print(f"Failed: {stats['failed']}")
            if 'total_tokens' in stats:
                print(f"Total tokens: {stats['total_tokens']:,}")
            if 'total_sequences' in stats:
                print(f"Total sequences: {stats['total_sequences']:,}")
        else:
            print(f"Input: {stats.get('input')}")
            print(f"Output: {stats.get('output')}")
            if 'tokens_written' in stats:
                print(f"Tokens: {stats['tokens_written']:,}")
            if 'sequences_created' in stats:
                print(f"Sequences: {stats['sequences_created']:,}")
            if 'records_read' in stats:
                print(f"Records: {stats.get('records_read', stats.get('lines_read', 0)):,}")
        
        print("="*70 + "\n")
        
    except Exception as e:
        get_logger().critical("data.converter", f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        get_logger().cleanup()


if __name__ == '__main__':
    main()

