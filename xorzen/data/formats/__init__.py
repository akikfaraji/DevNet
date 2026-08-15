"""
Data Formats Package
Provides format handlers for different file types.

Usage:
    >>> from xorzen.data.formats import get_handler
    >>> handler = get_handler('txt')
    >>> for record in handler.read(Path('data.txt')):
    ...     print(record['text'])
"""

from pathlib import Path
from typing import Optional

from .base import BaseFormatHandler
from .txt import TxtFormatHandler
from .json import JsonFormatHandler
from .jsonl import JsonlFormatHandler

# Registry of format handlers
_HANDLERS = {
    'txt': TxtFormatHandler,
    'text': TxtFormatHandler,
    'json': JsonFormatHandler,
    'jsonl': JsonlFormatHandler,
    'ndjson': JsonlFormatHandler,
}


def get_handler(format_name: str) -> BaseFormatHandler:
    """
    Get handler for specified format.
    
    Args:
        format_name: Format name (e.g., 'txt', 'json', 'jsonl')
        
    Returns:
        Format handler instance
        
    Raises:
        ValueError: If format not supported
    """
    format_name = format_name.lower()
    
    if format_name not in _HANDLERS:
        raise ValueError(
            f"Unsupported format: {format_name}. "
            f"Supported formats: {list(_HANDLERS.keys())}"
        )
    
    return _HANDLERS[format_name]()


def get_handler_for_file(file_path: Path) -> Optional[BaseFormatHandler]:
    """
    Get appropriate handler for a file based on extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        Format handler instance or None if no handler found
    """
    ext = file_path.suffix.lower().lstrip('.')
    
    if ext in _HANDLERS:
        return _HANDLERS[ext]()
    
    return None


def list_supported_formats() -> list:
    """List all supported format names."""
    return sorted(set(_HANDLERS.keys()))


__all__ = [
    'BaseFormatHandler',
    'TxtFormatHandler',
    'JsonFormatHandler',
    'JsonlFormatHandler',
    'get_handler',
    'get_handler_for_file',
    'list_supported_formats',
]
