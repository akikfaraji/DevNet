"""
xorzen Data Validation System
Comprehensive validation for datasets and data quality checks.

This module provides tools to validate:
- Data format correctness
- Token ID validity
- Sequence length consistency
- Data quality metrics
- Duplicate detection
- Statistical analysis

Example:
    >>> from xorzen.data import validate_data
    >>> validator = DataValidator('train.bin')
    >>> report = validator.validate()
    >>> if report['is_valid']:
    ...     print("Data is valid!")
    ... else:
    ...     print(f"Found {len(report['errors'])} issues")
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from collections import Counter, defaultdict
import warnings

from xorzen.exceptions import DataValidationError, DataFormatError
from xorzen.utils.logger import get_logger

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = get_logger()


class DataValidator:
    """
    Comprehensive data validator for xorzen datasets.
    
    Validates data quality, format correctness, and statistical properties.
    
    Example:
        >>> validator = DataValidator('train.bin', vocab_size=32000)
        >>> report = validator.validate(
        ...     check_duplicates=True,
        ...     check_statistics=True
        ... )
        >>> print(report['summary'])
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        vocab_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        metadata_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize data validator.
        
        Args:
            data_path: Path to data file (.bin, .npy, etc.)
            vocab_size: Expected vocabulary size (for token validation)
            sequence_length: Expected sequence length
            metadata_path: Optional path to metadata file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.metadata_path = metadata_path or self.data_path.with_suffix('.meta.json')
        
        # Load metadata if available
        self.metadata = self._load_metadata()
        
        # Auto-detect vocab_size and sequence_length from metadata
        if self.metadata:
            if not vocab_size and 'vocab_size' in self.metadata:
                self.vocab_size = self.metadata['vocab_size']
            if not sequence_length and 'max_length' in self.metadata:
                self.sequence_length = self.metadata['max_length']
        
        # Validation results
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata file if available."""
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("data.validation", f"Failed to load metadata: {e}")
        return {}
    
    def validate(
        self,
        check_format: bool = True,
        check_tokens: bool = True,
        check_sequences: bool = True,
        check_duplicates: bool = False,
        check_statistics: bool = True,
        check_quality: bool = True,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation.
        
        Args:
            check_format: Validate file format and structure
            check_tokens: Validate token IDs
            check_sequences: Validate sequence lengths
            check_duplicates: Check for duplicate sequences (expensive)
            check_statistics: Compute statistical metrics
            check_quality: Check data quality indicators
            sample_size: Sample size for expensive checks (None = all data)
            
        Returns:
            Validation report dictionary
            
        Example:
            >>> validator = DataValidator('train.bin', vocab_size=32000)
            >>> report = validator.validate()
            >>> if not report['is_valid']:
            ...     for error in report['errors']:
            ...         print(f"ERROR: {error}")
        """
        logger.info("data.validation", f"Starting validation of {self.data_path}")
        
        # Reset results
        self.errors = []
        self.warnings = []
        self.info = {}
        
        # Run validation checks
        if check_format:
            self._validate_format()
        
        if check_tokens:
            self._validate_tokens(sample_size)
        
        if check_sequences:
            self._validate_sequences()
        
        if check_duplicates:
            self._check_duplicates(sample_size)
        
        if check_statistics:
            self._compute_statistics(sample_size)
        
        if check_quality:
            self._check_quality(sample_size)
        
        # Compile report
        report = {
            'file': str(self.data_path),
            'is_valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'summary': self._generate_summary()
        }
        
        logger.info("data.validation", 
                   f"Validation complete: {len(self.errors)} errors, "
                   f"{len(self.warnings)} warnings")
        
        return report
    
    def _validate_format(self):
        """Validate file format and structure."""
        logger.debug("data.validation", "Validating format...")
        
        # Check file extension
        ext = self.data_path.suffix.lower()
        supported_exts = ['.bin', '.npy', '.npz']
        
        if ext not in supported_exts:
            self.warnings.append(
                f"Unusual file extension '{ext}'. "
                f"Supported: {supported_exts}"
            )
        
        # Check file size
        file_size = self.data_path.stat().st_size
        self.info['file_size_bytes'] = file_size
        self.info['file_size_mb'] = file_size / (1024 ** 2)
        
        if file_size == 0:
            self.errors.append("File is empty (0 bytes)")
            return
        
        if file_size < 1000:
            self.warnings.append(f"File is very small ({file_size} bytes)")
        
        # Check metadata consistency
        if self.metadata:
            expected_size = self.metadata.get('file_size_bytes')
            if expected_size and abs(expected_size - file_size) > 1000:
                self.warnings.append(
                    f"File size mismatch: metadata says {expected_size} bytes, "
                    f"actual is {file_size} bytes"
                )
        
        # Try loading data
        try:
            if ext == '.bin':
                dtype_str = self.metadata.get('dtype', 'uint16')
                try:
                    data_dtype = np.dtype(dtype_str)
                except TypeError: # Handle cases where dtype_str is not a valid numpy dtype string
                    logger.warning("data.validation", f"Invalid dtype '{dtype_str}' in metadata for {self.data_path}. Defaulting to uint16.")
                    data_dtype = np.uint16
                
                # Update self.info with the dtype actually used
                self.info['dtype'] = str(data_dtype)

                if file_size % data_dtype.itemsize != 0:
                    self.errors.append(f"File size ({file_size} bytes) is not a multiple of expected dtype itemsize ({data_dtype.itemsize} bytes). File may be corrupted or dtype incorrect.")
                
                self.info['total_tokens'] = file_size // data_dtype.itemsize
                # No actual data load needed here, just size check
            elif ext == '.npy':
                data = np.load(str(self.data_path), mmap_mode='r')
                self.info['shape'] = data.shape
                self.info['dtype'] = str(data.dtype)
                self.info['total_tokens'] = data.size
        except Exception as e:
            self.errors.append(f"Failed to load data file: {e}")
            return
    
    def _validate_tokens(self, sample_size: Optional[int] = None):
        """Validate token IDs."""
        if not NUMPY_AVAILABLE:
            self.warnings.append("numpy not available - skipping token validation")
            return
        
        try:
            # Load data
            ext = self.data_path.suffix.lower()
            if ext == '.bin':
                dtype_str = self.metadata.get('dtype', 'uint16')
                try:
                    data_dtype = np.dtype(dtype_str)
                except TypeError:
                    logger.warning("data.validation", f"Invalid dtype '{dtype_str}' in metadata for {self.data_path}. Defaulting to uint16.")
                    data_dtype = np.uint16
                data = np.fromfile(str(self.data_path), dtype=data_dtype)
            elif ext == '.npy':
                data = np.load(str(self.data_path), mmap_mode='r').flatten()
            else:
                return
            
            # Sample if needed
            if sample_size and len(data) > sample_size:
                indices = np.random.choice(len(data), sample_size, replace=False)
                data = data[indices]
            
            # Check token ID range
            min_token = int(data.min())
            max_token = int(data.max())
            
            self.info['min_token_id'] = min_token
            self.info['max_token_id'] = max_token
            
            if min_token < 0:
                self.errors.append(f"Found negative token ID: {min_token}")
            
            if self.vocab_size:
                if max_token >= self.vocab_size:
                    self.errors.append(
                        f"Found token ID {max_token} >= vocab_size {self.vocab_size}. "
                        f"Valid range is [0, {self.vocab_size - 1}]"
                    )
            else:
                self.warnings.append(
                    f"vocab_size not specified. Cannot validate max token ID {max_token}"
                )
            
            # Check for special tokens (typically low IDs)
            special_token_count = np.sum(data < 10)
            special_token_ratio = special_token_count / len(data)
            
            self.info['special_token_ratio'] = float(special_token_ratio)
            
            if special_token_ratio > 0.5:
                self.warnings.append(
                    f"High ratio of special tokens ({special_token_ratio:.2%}). "
                    "This might indicate padding or formatting issues"
                )
            
            # Check for zeros (often padding)
            zero_count = np.sum(data == 0)
            zero_ratio = zero_count / len(data)
            
            self.info['zero_token_ratio'] = float(zero_ratio)
            
            if zero_ratio > 0.3:
                self.warnings.append(
                    f"High ratio of zero tokens ({zero_ratio:.2%}). "
                    "This might indicate excessive padding"
                )
        
        except Exception as e:
            self.errors.append(f"Token validation failed: {e}")
    
    def _validate_sequences(self):
        """Validate sequence lengths."""
        if not NUMPY_AVAILABLE:
            return
        
        logger.debug("data.validation", "Validating sequences...")
        
        try:
            # Load data
            ext = self.data_path.suffix.lower()
            if ext == '.bin':
                dtype_str = self.metadata.get('dtype', 'uint16')
                try:
                    data_dtype = np.dtype(dtype_str)
                except TypeError:
                    logger.warning("data.validation", f"Invalid dtype '{dtype_str}' in metadata for {self.data_path}. Defaulting to uint16.")
                    data_dtype = np.uint16
                data = np.fromfile(str(self.data_path), dtype=data_dtype)
            elif ext == '.npy':
                data = np.load(str(self.data_path), mmap_mode='r')
            else:
                return
            
            # Check if data is 2D (sequences)
            if data.ndim == 2:
                num_sequences, seq_length = data.shape
                
                self.info['num_sequences'] = num_sequences
                self.info['sequence_length'] = seq_length
                
                # Validate against expected sequence length
                if self.sequence_length and seq_length != self.sequence_length:
                    self.warnings.append(
                        f"Sequence length mismatch: expected {self.sequence_length}, "
                        f"found {seq_length}"
                    )
            
            elif data.ndim == 1:
                # 1D array - continuous token stream
                total_tokens = len(data)
                self.info['total_tokens'] = total_tokens
                
                if self.sequence_length:
                    num_complete_sequences = total_tokens // self.sequence_length
                    remainder = total_tokens % self.sequence_length
                    
                    self.info['num_complete_sequences'] = num_complete_sequences
                    self.info['remainder_tokens'] = remainder
                    
                    if remainder > 0:
                        self.warnings.append(
                            f"Found {remainder} remainder tokens that don't form "
                            f"a complete sequence of length {self.sequence_length}"
                        )
        
        except Exception as e:
            self.errors.append(f"Sequence validation failed: {e}")
    
    def _check_duplicates(self, sample_size: Optional[int] = None):
        """Check for duplicate sequences."""
        if not NUMPY_AVAILABLE:
            return
        
        logger.debug("data.validation", "Checking for duplicates...")
        
        try:
            # Load data
            ext = self.data_path.suffix.lower()
            if ext == '.bin':
                dtype_str = self.metadata.get('dtype', 'uint16')
                try:
                    data_dtype = np.dtype(dtype_str)
                except TypeError:
                    logger.warning("data.validation", f"Invalid dtype '{dtype_str}' in metadata for {self.data_path}. Defaulting to uint16.")
                    data_dtype = np.uint16
                data = np.fromfile(str(self.data_path), dtype=data_dtype)
            elif ext == '.npy':
                data = np.load(str(self.data_path), mmap_mode='r')
            else:
                return
            
            # Only check 2D data (sequences)
            if data.ndim != 2:
                logger.debug("data.validation", "Skipping duplicate check for 1D data")
                return
            
            num_sequences = data.shape[0]
            
            # Sample if needed
            if sample_size and num_sequences > sample_size:
                indices = np.random.choice(num_sequences, sample_size, replace=False)
                data = data[indices]
                num_sequences = sample_size
            
            # Compute hashes
            hashes = set()
            duplicates = 0
            
            for seq in data:
                seq_hash = hashlib.md5(seq.tobytes()).hexdigest()
                if seq_hash in hashes:
                    duplicates += 1
                else:
                    hashes.add(seq_hash)
            
            duplicate_ratio = duplicates / num_sequences
            
            self.info['duplicates_found'] = duplicates
            self.info['duplicate_ratio'] = float(duplicate_ratio)
            
            if duplicate_ratio > 0.01:  # >1% duplicates
                self.warnings.append(
                    f"Found {duplicates} duplicate sequences ({duplicate_ratio:.2%}). "
                    "This might indicate data quality issues"
                )
        
        except Exception as e:
            self.errors.append(f"Duplicate check failed: {e}")
    
    def _compute_statistics(self, sample_size: Optional[int] = None):
        """Compute statistical metrics."""
        if not NUMPY_AVAILABLE:
            return
        
        logger.debug("data.validation", "Computing statistics...")
        
        try:
            # Load data
            ext = self.data_path.suffix.lower()
            if ext == '.bin':
                dtype_str = self.metadata.get('dtype', 'uint16')
                try:
                    data_dtype = np.dtype(dtype_str)
                except TypeError:
                    logger.warning("data.validation", f"Invalid dtype '{dtype_str}' in metadata for {self.data_path}. Defaulting to uint16.")
                    data_dtype = np.uint16
                data = np.fromfile(str(self.data_path), dtype=data_dtype)
            elif ext == '.npy':
                data = np.load(str(self.data_path), mmap_mode='r').flatten()
            else:
                return
            
            # Sample if needed
            if sample_size and len(data) > sample_size:
                indices = np.random.choice(len(data), sample_size, replace=False)
                data = data[indices]
            
            # Token distribution
            token_counts = Counter(data)
            most_common = token_counts.most_common(10)
            
            self.info['unique_tokens'] = len(token_counts)
            self.info['most_common_tokens'] = [(int(tok), int(count)) for tok, count in most_common]
            
            # Statistical measures
            self.info['mean_token_id'] = float(np.mean(data))
            self.info['std_token_id'] = float(np.std(data))
            self.info['median_token_id'] = float(np.median(data))
            
            # Check for distribution issues
            if len(token_counts) < 100 and self.vocab_size and self.vocab_size > 1000:
                self.warnings.append(
                    f"Very few unique tokens ({len(token_counts)}) compared to "
                    f"vocab size ({self.vocab_size}). Data might be low diversity"
                )
            
            # Check for skewed distribution
            if most_common and most_common[0][1] / len(data) > 0.5:
                self.warnings.append(
                    f"Token distribution is highly skewed: most common token "
                    f"appears in {most_common[0][1] / len(data):.1%} of positions"
                )
        
        except Exception as e:
            self.errors.append(f"Statistics computation failed: {e}")
    
    def _check_quality(self, sample_size: Optional[int] = None):
        """Check data quality indicators."""
        if not NUMPY_AVAILABLE:
            return
        
        logger.debug("data.validation", "Checking data quality...")
        
        try:
            # Load data
            ext = self.data_path.suffix.lower()
            if ext == '.bin':
                dtype_str = self.metadata.get('dtype', 'uint16')
                try:
                    data_dtype = np.dtype(dtype_str)
                except TypeError:
                    logger.warning("data.validation", f"Invalid dtype '{dtype_str}' in metadata for {self.data_path}. Defaulting to uint16.")
                    data_dtype = np.uint16
                data = np.fromfile(str(self.data_path), dtype=data_dtype)
            elif ext == '.npy':
                data = np.load(str(self.data_path), mmap_mode='r')
            else:
                return
            
            # Sample if needed for 2D data
            if data.ndim == 2 and sample_size and data.shape[0] > sample_size:
                indices = np.random.choice(data.shape[0], sample_size, replace=False)
                data = data[indices]
            
            # Flatten for analysis
            data_flat = data.flatten()
            
            # Check for repetitive patterns
            if len(data_flat) > 100:
                # Look for repeated subsequences
                window = 10
                repetitions = 0
                
                for i in range(len(data_flat) - 2 * window):
                    if np.array_equal(data_flat[i:i+window], data_flat[i+window:i+2*window]):
                        repetitions += 1
                
                repetition_ratio = repetitions / (len(data_flat) - 2 * window)
                
                self.info['repetition_ratio'] = float(repetition_ratio)
                
                if repetition_ratio > 0.1:
                    self.warnings.append(
                        f"High repetition ratio ({repetition_ratio:.2%}). "
                        "Data might contain repetitive patterns"
                    )
            
            # Check for constant sequences (all same token)
            if data.ndim == 2:
                constant_sequences = 0
                
                for seq in data:
                    if len(set(seq)) == 1:
                        constant_sequences += 1
                
                if constant_sequences > 0:
                    self.warnings.append(
                        f"Found {constant_sequences} constant sequences "
                        "(all tokens are the same)"
                    )
        
        except Exception as e:
            self.errors.append(f"Quality check failed: {e}")
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary."""
        if len(self.errors) > 0:
            status = f"❌ INVALID ({len(self.errors)} errors)"
        elif len(self.warnings) > 0:
            status = f"⚠️  VALID WITH WARNINGS ({len(self.warnings)} warnings)"
        else:
            status = "✅ VALID"
        
        summary_lines = [
            f"Validation Status: {status}",
            f"File: {self.data_path}",
            ""
        ]
        
        if self.info:
            summary_lines.append("Data Info:")
            for key, value in self.info.items():
                if isinstance(value, float):
                    summary_lines.append(f"  {key}: {value:.4f}")
                elif isinstance(value, list) and len(value) > 5:
                    summary_lines.append(f"  {key}: [truncated, {len(value)} items]")
                else:
                    summary_lines.append(f"  {key}: {value}")
            summary_lines.append("")
        
        if self.errors:
            summary_lines.append("Errors:")
            for error in self.errors:
                summary_lines.append(f"  ❌ {error}")
            summary_lines.append("")
        
        if self.warnings:
            summary_lines.append("Warnings:")
            for warning in self.warnings:
                summary_lines.append(f"  ⚠️  {warning}")
        
        return "\n".join(summary_lines)


class BatchValidator:
    """
    Validate multiple data files in batch.
    
    Example:
        >>> validator = BatchValidator(['train.bin', 'val.bin', 'test.bin'])
        >>> results = validator.validate_all()
        >>> for file, report in results.items():
        ...     print(f"{file}: {report['summary']}")
    """
    
    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        vocab_size: Optional[int] = None,
        sequence_length: Optional[int] = None
    ):
        """
        Initialize batch validator.
        
        Args:
            file_paths: List of data file paths
            vocab_size: Expected vocabulary size
            sequence_length: Expected sequence length
        """
        self.file_paths = [Path(p) for p in file_paths]
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
    
    def validate_all(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Validate all files.
        
        Args:
            **kwargs: Arguments passed to DataValidator.validate()
            
        Returns:
            Dictionary mapping file paths to validation reports
        """
        results = {}
        
        for file_path in self.file_paths:
            logger.info("data.validation", f"Validating {file_path}...")
            
            try:
                validator = DataValidator(
                    file_path,
                    vocab_size=self.vocab_size,
                    sequence_length=self.sequence_length,
                    **kwargs
                )
                report = validator.validate(**kwargs)
                results[str(file_path)] = report
            
            except Exception as e:
                logger.error("data.validation", f"Validation failed for {file_path}: {e}")
                results[str(file_path)] = {
                    'file': str(file_path),
                    'is_valid': False,
                    'errors': [f"Validation crashed: {e}"],
                    'warnings': [],
                    'info': {},
                    'summary': f"Validation crashed: {e}"
                }
        
        return results
    
    def get_summary(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate summary for all validations."""
        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r['is_valid'])
        invalid_files = total_files - valid_files
        
        summary = [
            "="*70,
            "Batch Validation Summary",
            "="*70,
            f"Total files: {total_files}",
            f"Valid: {valid_files}",
            f"Invalid: {invalid_files}",
            ""
        ]
        
        for file_path, report in results.items():
            status = "✅" if report['is_valid'] else "❌"
            error_count = len(report.get('errors', []))
            warning_count = len(report.get('warnings', []))
            
            summary.append(f"{status} {file_path}")
            if error_count > 0:
                summary.append(f"    {error_count} errors")
            if warning_count > 0:
                summary.append(f"    {warning_count} warnings")
        
        return "\n".join(summary)


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def validate_data(
    data_path: Union[str, Path],
    vocab_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate a data file.
    
    Convenience function for quick validation.
    
    Args:
        data_path: Path to data file
        vocab_size: Expected vocabulary size
        sequence_length: Expected sequence length
        **kwargs: Additional validation arguments
        
    Returns:
        Validation report
        
    Example:
        >>> from xorzen.data import validate_data
        >>> report = validate_data('train.bin', vocab_size=32000)
        >>> if report['is_valid']:
        ...     print("Data is valid!")
        ... else:
        ...     print("Issues found:")
        ...     for error in report['errors']:
        ...         print(f"  - {error}")
    """
    validator = DataValidator(data_path, vocab_size, sequence_length)
    return validator.validate(**kwargs)


def validate_batch(
    file_paths: List[Union[str, Path]],
    vocab_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Validate multiple data files.
    
    Args:
        file_paths: List of file paths
        vocab_size: Expected vocabulary size
        sequence_length: Expected sequence length
        **kwargs: Additional validation arguments
        
    Returns:
        Dictionary mapping file paths to validation reports
        
    Example:
        >>> from xorzen.data import validate_batch
        >>> results = validate_batch(['train.bin', 'val.bin'])
        >>> for file, report in results.items():
        ...     print(f"{file}: {'valid' if report['is_valid'] else 'invalid'}")
    """
    validator = BatchValidator(file_paths, vocab_size, sequence_length)
    return validator.validate_all(**kwargs)


__all__ = [
    'DataValidator',
    'BatchValidator',
    'validate_data',
    'validate_batch',
]

