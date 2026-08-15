"""
Base Format Handler
Abstract interface for data format handlers.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Dict, Any, Optional
from pathlib import Path

from xorzen.exceptions import DataFormatError


class BaseFormatHandler(ABC):
    """
    Abstract base class for data format handlers.
    
    Each format (txt, json, jsonl, parquet, etc.) should implement this interface.
    """
    
    FORMAT_NAME: str = "base"  # Override in subclasses
    FILE_EXTENSIONS: List[str] = []  # e.g., ['.txt', '.text']
    
    def __init__(self):
        """Initialize format handler."""
        self.stats = {
            'files_processed': 0,
            'records_read': 0,
            'bytes_processed': 0,
            'errors': 0,
        }
    
    @abstractmethod
    def read(self, file_path: Path, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Read data from file and yield records.
        
        Args:
            file_path: Path to input file
            **kwargs: Format-specific options
            
        Yields:
            Dictionary records with at least a 'text' field
            
        Raises:
            DataFormatError: If format is invalid
        """
        pass
    
    @abstractmethod
    def write(self, file_path: Path, records: Iterator[Dict[str, Any]], **kwargs):
        """
        Write records to file.
        
        Args:
            file_path: Path to output file
            records: Iterator of dictionary records
            **kwargs: Format-specific options
            
        Raises:
            DataFormatError: If write fails
        """
        pass
    
    @abstractmethod
    def validate(self, file_path: Path) -> bool:
        """
        Validate that file is in correct format.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def is_compatible(self, file_path: Path) -> bool:
        """
        Check if file extension is compatible with this handler.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if extension matches, False otherwise
        """
        return file_path.suffix.lower() in self.FILE_EXTENSIONS
    
    def estimate_size(self, file_path: Path) -> int:
        """
        Estimate number of records in file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Estimated record count
        """
        if not file_path.exists():
            return 0
        # Default: use file size as rough estimate
        # Subclasses can provide better estimates
        return file_path.stat().st_size // 100  # Assume ~100 bytes per record
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'files_processed': 0,
            'records_read': 0,
            'bytes_processed': 0,
            'errors': 0,
        }


__all__ = ['BaseFormatHandler']
