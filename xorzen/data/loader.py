"""
xorzen Data Loader
Comprehensive data loading utilities for all supported formats.

This module provides the public API for loading data:
- load_from_bin() - Load pre-tokenized binary data
- load_from_npy() - Load numpy format data
- load_from_txt() - Load raw text files
- load_from_json() - Load JSON files
- load_from_jsonl() - Load JSONL files
- load_from_parquet() - Load Parquet files
- load_from_dir() - Auto-detect and load from directory

Example:
    >>> from xorzen.data import load_from_bin
    >>> dataset = load_from_bin('train.bin', batch_size=32)
    >>> for batch in dataset:
    ...     # Training loop
    ...     pass
"""

import json
import struct
from typing import Union, List, Dict, Any, Optional, Iterator, Tuple
from pathlib import Path
import warnings

from xorzen.exceptions import DataLoadError, DataFormatError
from xorzen.utils.logger import get_logger

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available - binary data loading disabled")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("torch not available - PyTorch dataset functionality disabled")

logger = get_logger()


# =============================================================================
# BINARY DATA LOADERS (.bin format)
# =============================================================================

class BinaryDataset:
    """
    Fast dataset for pre-tokenized binary data (.bin format).
    
    This is a critical component for xorzen's efficient training pipeline.
    Binary files are memory-mapped for zero-copy access.
    
    Features:
    - Memory-mapped file access (no RAM overhead)
    - Fast random access
    - Automatic batching
    - Sequence length validation
    - Metadata loading
    
    Example:
        >>> dataset = BinaryDataset('train.bin', sequence_length=2048)
        >>> print(f"Dataset size: {len(dataset)} sequences")
        >>> batch = dataset[0]  # Get first sequence
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        sequence_length: Optional[int] = None,
        dtype: str = 'uint16',
        mmap_mode: str = 'r',
        validate: bool = True
    ):
        """
        Initialize binary dataset.
        
        Args:
            file_path: Path to .bin file
            sequence_length: Fixed sequence length (None for auto-detect)
            dtype: Data type of tokens (uint16, uint32, int32)
            mmap_mode: Memory map mode ('r' for read-only, 'r+' for read-write)
            validate: Validate metadata and file consistency
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for binary data loading")
        
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise DataLoadError(
                f"Binary file not found: {file_path}. "
                "Did you convert your data to binary format?"
            )
        
        self.dtype = np.dtype(dtype)
        self.mmap_mode = mmap_mode
        self.sequence_length = sequence_length
        
        # Load metadata if available
        self.metadata = self._load_metadata()
        
        # Memory-map the binary file
        try:
            self.data = np.memmap(
                str(self.file_path),
                dtype=self.dtype,
                mode=self.mmap_mode
            )
        except Exception as e:
            raise DataLoadError(
                f"Failed to memory-map binary file {file_path}: {e}. "
                "The file may be corrupted."
            )
        
        # Determine sequence length and shape
        self._determine_shape()
        
        # Validate if requested
        if validate:
            self._validate()
        
        logger.info("data.loader", 
                   f"Loaded binary dataset: {len(self)} sequences of length {self.sequence_length}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata JSON if available."""
        metadata_path = self.file_path.with_suffix('.meta.json')
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.debug("data.loader", f"Loaded metadata from {metadata_path}")
                return metadata
            except Exception as e:
                logger.warning("data.loader", f"Failed to load metadata: {e}")
                return {}
        
        return {}
    
    def _determine_shape(self):
        """Determine dataset shape from metadata or data."""
        total_tokens_in_file = len(self.data)

        if self.metadata:
            shape_from_meta = self.metadata.get('shape')
            if shape_from_meta and len(shape_from_meta) == 2:
                self.num_sequences = shape_from_meta[0]
                self.sequence_length = shape_from_meta[1]
                # Ensure memmap is reshaped to the shape from metadata
                self.data = self.data.reshape(self.num_sequences, self.sequence_length)
                logger.debug("data.loader", f"Shape from metadata: {self.num_sequences} sequences, {self.sequence_length} length")
                return
            elif shape_from_meta and len(shape_from_meta) == 1:
                total_tokens_in_file = shape_from_meta[0]
                logger.debug("data.loader", f"1D shape from metadata: {total_tokens_in_file} tokens")
            
            if self.metadata.get('sequence_length') is not None:
                self.sequence_length = self.metadata['sequence_length']
                logger.debug("data.loader", f"Sequence length from metadata: {self.sequence_length}")

        # If sequence_length is still not set, use a default for continuous streams
        if self.sequence_length is None:
            self.sequence_length = 2048  # Default sequence length for processing continuous streams
            logger.warning("data.loader", f"Sequence length not specified, defaulting to {self.sequence_length} for continuous stream processing.")

        # Calculate num_sequences based on determined/default sequence_length
        if self.sequence_length <= 0:
            raise ValueError("Sequence length must be positive.")
        
        self.num_sequences = total_tokens_in_file // self.sequence_length
        
        # If the data is a continuous stream, ensure data can be viewed as sequences
        if self.num_sequences > 0:
            # Trim the end if it doesn't form a full sequence
            effective_total_tokens = self.num_sequences * self.sequence_length
            if len(self.data) != effective_total_tokens:
                self.data = self.data[:effective_total_tokens]
            
            try:
                self.data = self.data.reshape(self.num_sequences, self.sequence_length)
            except Exception as e:
                logger.warning("data.loader", f"Could not reshape 1D memmap to 2D view: {e}. Data will be accessed in 1D and sliced.")
                # Fallback: keep as 1D and slice in __getitem__ (but current __getitem__ assumes 2D)
                # For now, let's assume reshape works if num_sequences is correctly calculated.
                # If it still fails here, it implies a fundamental data/length mismatch.
                raise DataFormatError(
                    format_type='bin',
                    reason=f"Failed to reshape data into {self.num_sequences} sequences of length {self.sequence_length}: {e}",
                    supported_formats=['correctly sized 1D array or 2D array']
                )
    
    def _validate(self):
        """Validate dataset consistency."""
        # Check file size consistency
        expected_size = self.num_sequences * self.sequence_length * self.dtype.itemsize
        actual_size = self.file_path.stat().st_size
        
        if abs(actual_size - expected_size) > 1000:  # Allow small tolerance
            logger.warning("data.loader", 
                          f"File size mismatch: expected ~{expected_size} bytes, "
                          f"got {actual_size} bytes")
        
        # Check for invalid token IDs (optional)
        if self.metadata.get('vocab_size'):
            vocab_size = self.metadata['vocab_size']
            max_token = int(self.data.max())
            if max_token >= vocab_size:
                logger.warning("data.loader", 
                              f"Found token ID {max_token} >= vocab_size {vocab_size}")
    
    def __len__(self) -> int:
        """Get number of sequences."""
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get sequence by index."""
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.num_sequences})")
        
        return self.data[idx]
    
    def get_batch(self, indices: List[int]) -> np.ndarray:
        """Get multiple sequences as a batch."""
        return np.stack([self[idx] for idx in indices])
    
    def iter_batches(self, batch_size: int, shuffle: bool = False) -> Iterator[np.ndarray]:
        """
        Iterate over dataset in batches.
        
        Args:
            batch_size: Number of sequences per batch
            shuffle: Shuffle data before batching
            
        Yields:
            Batches of shape (batch_size, sequence_length)
        """
        indices = np.arange(self.num_sequences)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.num_sequences, batch_size):
            end_idx = min(start_idx + batch_size, self.num_sequences)
            batch_indices = indices[start_idx:end_idx]
            yield self.get_batch(batch_indices)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'file_path': str(self.file_path),
            'num_sequences': self.num_sequences,
            'sequence_length': self.sequence_length,
            'total_tokens': self.num_sequences * self.sequence_length,
            'dtype': str(self.dtype),
            'file_size_mb': self.file_path.stat().st_size / (1024 ** 2),
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return (f"BinaryDataset(sequences={self.num_sequences}, "
                f"length={self.sequence_length}, dtype={self.dtype})")


class TorchBinaryDataset(Dataset):
    """
    PyTorch Dataset wrapper for binary data.
    
    Integrates with PyTorch DataLoader for training.
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = TorchBinaryDataset('train.bin')
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     # batch is a torch.Tensor
        ...     pass
    """
    
    def __init__(self, file_path: Union[str, Path], **kwargs):
        """
        Initialize PyTorch dataset.
        
        Args:
            file_path: Path to .bin file
            **kwargs: Arguments passed to BinaryDataset
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch required for TorchBinaryDataset")
        
        self.dataset = BinaryDataset(file_path, **kwargs)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single training sample for language modeling.
        Returns a dictionary containing 'input_ids' and 'labels' tensors.
        """
        raw_sequence_np = self.dataset[idx] # This is a numpy.ndarray of shape [sequence_length]
        
        # Ensure the sequence is long enough for input and label
        if len(raw_sequence_np) < 2:
            # If the sequence is too short, handle by padding or skipping.
            # For simplicity, we'll raise an error as this indicates a data issue
            # or an insufficient sequence_length in BinaryDataset.
            raise ValueError(
                f"Sequence at index {idx} has length {len(raw_sequence_np)}, "
                "but at least 2 tokens are required for next-token prediction."
            )
        
        # For next-token prediction, input_ids are tokens[:-1] and labels are tokens[1:]
        input_ids_tensor = torch.from_numpy(raw_sequence_np[:-1]).long()
        labels_tensor = torch.from_numpy(raw_sequence_np[1:]).long()
        
        return {'input_ids': input_ids_tensor, 'labels': labels_tensor}
    
    def get_stats(self) -> Dict[str, Any]:
        return self.dataset.get_stats()


# =============================================================================
# NUMPY DATA LOADERS (.npy format)
# =============================================================================

class NumpyDataset:
    """
    Dataset for numpy array files (.npy).
    
    Similar to BinaryDataset but loads entire file into memory.
    Use for smaller datasets that fit in RAM.
    
    Example:
        >>> dataset = NumpyDataset('train.npy')
        >>> batch = dataset[0:32]  # Get first 32 sequences
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        sequence_length: Optional[int] = None,
        mmap_mode: Optional[str] = None  # None loads to memory, 'r' for mmap
    ):
        """
        Initialize numpy dataset.
        
        Args:
            file_path: Path to .npy file
            sequence_length: Expected sequence length (for validation)
            mmap_mode: Memory map mode (None loads to RAM)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for numpy data loading")
        
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise DataLoadError(f"Numpy file not found: {file_path}")
        
        # Load data
        try:
            self.data = np.load(str(self.file_path), mmap_mode=mmap_mode)
        except Exception as e:
            raise DataLoadError(f"Failed to load numpy file {file_path}: {e}")
        
        # Validate shape
        if self.data.ndim not in [1, 2]:
            raise DataFormatError(
                format_type='npy',
                reason=f"Expected 1D or 2D array, got {self.data.ndim}D",
                supported_formats=['1D array', '2D array (sequences, length)']
            )
        
        # Determine dimensions
        if self.data.ndim == 2:
            self.num_sequences, self.sequence_length = self.data.shape
        else: # 1D array
            if sequence_length is not None:
                self.sequence_length = sequence_length
            else:
                self.sequence_length = 2048 # Default for 1D continuous stream
                logger.warning("data.loader", f"1D numpy array and sequence_length not specified, defaulting to {self.sequence_length}.")
            
            if self.sequence_length <= 0:
                raise ValueError("Sequence length must be positive.")
            
            self.num_sequences = len(self.data) // self.sequence_length
            
            # Trim data if it doesn't perfectly fit into sequences
            effective_total_tokens = self.num_sequences * self.sequence_length
            if len(self.data) != effective_total_tokens:
                logger.warning("data.loader", f"1D numpy array length ({len(self.data)}) not divisible by sequence_length ({self.sequence_length}). Trimming to {effective_total_tokens} tokens.")
                self.data = self.data[:effective_total_tokens]
        
        logger.info("data.loader", 
                   f"Loaded numpy dataset: {self.num_sequences} sequences of length {self.sequence_length}")
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        """Get sequence(s) by index or slice."""
        if isinstance(idx, slice):
            if self.data.ndim == 2:
                return self.data[idx]
            else: # 1D array - slice multiple sequences
                start_seq = idx.start if idx.start is not None else 0
                stop_seq = idx.stop if idx.stop is not None else self.num_sequences
                step_seq = idx.step if idx.step is not None else 1
                
                # Calculate start and end indices in the 1D array
                start_idx = start_seq * self.sequence_length
                end_idx = stop_seq * self.sequence_length
                
                # Extract and reshape
                sliced_data = self.data[start_idx:end_idx]
                return sliced_data.reshape(-1, self.sequence_length)
        
        elif isinstance(idx, int):
            if self.data.ndim == 2:
                return self.data[idx]
            else: # 1D array - slice a single sequence
                start = idx * self.sequence_length
                end = start + self.sequence_length
                return self.data[start:end]
        else:
            raise TypeError("Index must be an integer or a slice.")
    
    
    def get_batch(self, indices: List[int]) -> np.ndarray:
        """Get multiple sequences as a batch."""
        return np.stack([self[idx] for idx in indices])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'file_path': str(self.file_path),
            'num_sequences': self.num_sequences,
            'sequence_length': self.sequence_length,
            'total_tokens': self.num_sequences * self.sequence_length,
            'dtype': str(self.data.dtype),
            'shape': self.data.shape,
            'file_size_mb': self.file_path.stat().st_size / (1024 ** 2),
        }


# =============================================================================
# TEXT FORMAT LOADERS
# =============================================================================

def load_text_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> List[str]:
    """
    Load text file as list of lines.
    
    Args:
        file_path: Path to text file
        encoding: Text encoding
        
    Returns:
        List of text lines
        
    Example:
        >>> lines = load_text_file('data.txt')
        >>> print(f"Loaded {len(lines)} lines")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise DataLoadError(f"Text file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        logger.info("data.loader", f"Loaded {len(lines)} lines from {file_path}")
        return lines
    
    except Exception as e:
        raise DataLoadError(f"Failed to load text file {file_path}: {e}")


def load_json_file(file_path: Union[str, Path], text_field: str = 'text') -> List[Dict[str, Any]]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        text_field: Field containing text (for extraction)
        
    Returns:
        List of records (or single record wrapped in list)
        
    Example:
        >>> records = load_json_file('data.json')
        >>> texts = [r['text'] for r in records if 'text' in r]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise DataLoadError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Try to find array field
            found_array = False
            for value in data.values():
                if isinstance(value, list):
                    records = value
                    found_array = True
                    break
            
            if not found_array:
                # Single record
                records = [data]
        else:
            raise DataFormatError(
                format_type='json',
                reason="JSON must be object or array",
                supported_formats=['json object', 'json array']
            )
        
        logger.info("data.loader", f"Loaded {len(records)} records from {file_path}")
        return records
    
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise DataLoadError(f"Failed to load JSON file {file_path}: {e}")


def load_jsonl_file(file_path: Union[str, Path], text_field: str = 'text') -> List[Dict[str, Any]]:
    """
    Load JSONL file.
    
    Args:
        file_path: Path to JSONL file
        text_field: Field containing text
        
    Returns:
        List of records
        
    Example:
        >>> records = load_jsonl_file('data.jsonl')
        >>> texts = [r['text'] for r in records if 'text' in r]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise DataLoadError(f"JSONL file not found: {file_path}")
    
    records = []
    errors = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    errors += 1
                    if errors <= 10:  # Log first 10 errors
                        logger.warning("data.loader", 
                                     f"Invalid JSON at line {line_num} in {file_path}")
        
        if errors > 10:
            logger.warning("data.loader", 
                          f"Total {errors} invalid JSON lines in {file_path}")
        
        logger.info("data.loader", 
                   f"Loaded {len(records)} records from {file_path} ({errors} errors)")
        return records
    
    except Exception as e:
        raise DataLoadError(f"Failed to load JSONL file {file_path}: {e}")


def load_parquet_file(file_path: Union[str, Path], text_column: str = 'text') -> List[Dict[str, Any]]:
    """
    Load Parquet file.
    
    Args:
        file_path: Path to Parquet file
        text_column: Column containing text
        
    Returns:
        List of records
        
    Example:
        >>> records = load_parquet_file('data.parquet')
        >>> texts = [r['text'] for r in records if 'text' in r]
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for Parquet loading")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise DataLoadError(f"Parquet file not found: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        records = df.to_dict('records')
        
        logger.info("data.loader", f"Loaded {len(records)} records from {file_path}")
        return records
    
    except Exception as e:
        raise DataLoadError(f"Failed to load Parquet file {file_path}: {e}")


# =============================================================================
# DIRECTORY LOADERS
# =============================================================================

class DirectoryLoader:
    """
    Load data from a directory of files.
    
    Automatically detects file formats and loads all compatible files.
    
    Example:
        >>> loader = DirectoryLoader('data/')
        >>> files = loader.scan()
        >>> print(f"Found {len(files['txt'])} text files")
        >>> all_data = loader.load_all()
    """
    
    SUPPORTED_FORMATS = {
        'bin': ['.bin'],
        'npy': ['.npy'],
        'txt': ['.txt', '.text'],
        'json': ['.json'],
        'jsonl': ['.jsonl', '.ndjson'],
        'parquet': ['.parquet', '.pq']
    }
    
    def __init__(self, directory: Union[str, Path], recursive: bool = True):
        """
        Initialize directory loader.
        
        Args:
            directory: Path to directory
            recursive: Scan subdirectories recursively
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            raise DataLoadError(f"Directory not found: {directory}")
        
        if not self.directory.is_dir():
            raise DataLoadError(f"Not a directory: {directory}")
        
        self.recursive = recursive
        self.files_by_format = {}
    
    def scan(self) -> Dict[str, List[Path]]:
        """
        Scan directory and categorize files by format.
        
        Returns:
            Dictionary mapping format names to lists of file paths
            
        Example:
            >>> loader = DirectoryLoader('data/')
            >>> files = loader.scan()
            >>> print(files)
            {'txt': [Path('data/file1.txt'), ...], 'bin': [...]}
        """
        self.files_by_format = {fmt: [] for fmt in self.SUPPORTED_FORMATS}
        
        # Scan files
        if self.recursive:
            all_files = self.directory.rglob('*')
        else:
            all_files = self.directory.glob('*')
        
        for file_path in all_files:
            if not file_path.is_file():
                continue
            
            # Check format
            ext = file_path.suffix.lower()
            for fmt, extensions in self.SUPPORTED_FORMATS.items():
                if ext in extensions:
                    self.files_by_format[fmt].append(file_path)
                    break
        
        # Log summary
        total_files = sum(len(files) for files in self.files_by_format.values())
        logger.info("data.loader", 
                   f"Scanned {self.directory}: found {total_files} files")
        
        for fmt, files in self.files_by_format.items():
            if files:
                logger.debug("data.loader", f"  {fmt}: {len(files)} files")
        
        return self.files_by_format
    
    def load_all(
        self,
        formats: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load all files from directory.
        
        Args:
            formats: List of formats to load (None = all)
            **kwargs: Format-specific loading arguments
            
        Returns:
            Dictionary with loaded data by format
            
        Example:
            >>> loader = DirectoryLoader('data/')
            >>> data = loader.load_all(formats=['txt', 'jsonl'])
            >>> print(f"Loaded {len(data['txt'])} text files")
        """
        if not self.files_by_format:
            self.scan()
        
        if formats is None:
            formats = list(self.SUPPORTED_FORMATS.keys())
        
        loaded_data = {}
        
        for fmt in formats:
            files = self.files_by_format.get(fmt, [])
            if not files:
                continue
            
            logger.info("data.loader", f"Loading {len(files)} {fmt} files...")
            
            # Load based on format
            if fmt == 'bin':
                loaded_data['bin'] = [
                    BinaryDataset(f, **kwargs) for f in files
                ]
            elif fmt == 'npy':
                loaded_data['npy'] = [
                    NumpyDataset(f, **kwargs) for f in files
                ]
            elif fmt == 'txt':
                loaded_data['txt'] = [
                    load_text_file(f) for f in files
                ]
            elif fmt == 'json':
                loaded_data['json'] = [
                    load_json_file(f, **kwargs) for f in files
                ]
            elif fmt == 'jsonl':
                loaded_data['jsonl'] = [
                    load_jsonl_file(f, **kwargs) for f in files
                ]
            elif fmt == 'parquet':
                loaded_data['parquet'] = [
                    load_parquet_file(f, **kwargs) for f in files
                ]
        
        return loaded_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get directory statistics."""
        if not self.files_by_format:
            self.scan()
        
        stats = {
            'directory': str(self.directory),
            'recursive': self.recursive,
            'total_files': sum(len(files) for files in self.files_by_format.values()),
            'files_by_format': {
                fmt: len(files) for fmt, files in self.files_by_format.items()
            }
        }
        
        # Estimate total size
        total_size = 0
        for files in self.files_by_format.values():
            for file_path in files:
                try:
                    total_size += file_path.stat().st_size
                except:
                    pass
        
        stats['total_size_mb'] = total_size / (1024 ** 2)
        
        return stats


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def load_from_bin(
    file_path: Union[str, Path],
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    shuffle: bool = False,
    use_torch: bool = True,
    **kwargs
) -> Union[BinaryDataset, TorchBinaryDataset, DataLoader]:
    """
    Load pre-tokenized binary data.
    
    This is the primary data loading function for xorzen training.
    
    Args:
        file_path: Path to .bin file
        batch_size: Batch size (returns DataLoader if specified)
        sequence_length: Sequence length (auto-detect if None)
        shuffle: Shuffle data
        use_torch: Use PyTorch dataset/loader
        **kwargs: Additional arguments
        
    Returns:
        Dataset or DataLoader depending on arguments
        
    Example:
        >>> # Simple loading
        >>> dataset = load_from_bin('train.bin')
        >>> 
        >>> # With batching
        >>> loader = load_from_bin('train.bin', batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     # Training loop
        ...     pass
    """
    if use_torch and TORCH_AVAILABLE:
        dataset = TorchBinaryDataset(file_path, sequence_length=sequence_length, **kwargs)
        
        if batch_size:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=kwargs.get('num_workers', 0),
                pin_memory=kwargs.get('pin_memory', False)
            )
            return loader
        else:
            return dataset
    else:
        dataset = BinaryDataset(file_path, sequence_length=sequence_length, **kwargs)
        return dataset


def load_from_npy(file_path: Union[str, Path], **kwargs) -> NumpyDataset:
    """
    Load numpy format data.
    
    Args:
        file_path: Path to .npy file
        **kwargs: Additional arguments
        
    Returns:
        NumpyDataset instance
        
    Example:
        >>> dataset = load_from_npy('train.npy')
        >>> batch = dataset[0:32]
    """
    return NumpyDataset(file_path, **kwargs)


def load_from_txt(file_path: Union[str, Path], **kwargs) -> List[str]:
    """
    Load text file.
    
    Args:
        file_path: Path to .txt file
        **kwargs: Additional arguments (encoding, etc.)
        
    Returns:
        List of text lines
        
    Example:
        >>> lines = load_from_txt('data.txt')
        >>> print(f"Loaded {len(lines)} lines")
    """
    return load_text_file(file_path, **kwargs)


def load_from_json(file_path: Union[str, Path], **kwargs) -> List[Dict]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to .json file
        **kwargs: Additional arguments
        
    Returns:
        List of records
        
    Example:
        >>> records = load_from_json('data.json')
    """
    return load_json_file(file_path, **kwargs)


def load_from_jsonl(file_path: Union[str, Path], **kwargs) -> List[Dict]:
    """
    Load JSONL file.
    
    Args:
        file_path: Path to .jsonl file
        **kwargs: Additional arguments
        
    Returns:
        List of records
        
    Example:
        >>> records = load_from_jsonl('data.jsonl')
    """
    return load_jsonl_file(file_path, **kwargs)


def load_from_parquet(file_path: Union[str, Path], **kwargs) -> List[Dict]:
    """
    Load Parquet file.
    
    Args:
        file_path: Path to .parquet file
        **kwargs: Additional arguments
        
    Returns:
        List of records
        
    Example:
        >>> records = load_from_parquet('data.parquet')
    """
    return load_parquet_file(file_path, **kwargs)


def load_from_dir(directory: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """
    Load all data files from a directory.
    
    Automatically detects and loads all supported formats.
    
    Args:
        directory: Path to directory
        **kwargs: Additional arguments (recursive, formats, etc.)
        
    Returns:
        Dictionary with loaded data by format
        
    Example:
        >>> data = load_from_dir('data/', recursive=True)
        >>> print(f"Loaded {len(data.get('bin', []))} binary files")
        >>> print(f"Loaded {len(data.get('txt', []))} text files")
    """
    loader = DirectoryLoader(directory, recursive=kwargs.get('recursive', True))
    return loader.load_all(formats=kwargs.get('formats'), **kwargs)


__all__ = [
    # Classes
    'BinaryDataset',
    'TorchBinaryDataset',
    'NumpyDataset',
    'DirectoryLoader',
    
    # Functions
    'load_from_bin',
    'load_from_npy',
    'load_from_txt',
    'load_from_json',
    'load_from_jsonl',
    'load_from_parquet',
    'load_from_dir',
]

