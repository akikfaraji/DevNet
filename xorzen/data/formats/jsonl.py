"""
JSONL Format Handler
Handles JSON Lines format (.jsonl).
"""

import json
from typing import Iterator, Dict, Any
from pathlib import Path

from .base import BaseFormatHandler
from xorzen.exceptions import DataFormatError


class JsonlFormatHandler(BaseFormatHandler):
    """
    Handler for JSON Lines format.
    
    Each line is a separate JSON object.
    """
    
    FORMAT_NAME = "jsonl"
    FILE_EXTENSIONS = ['.jsonl', '.ndjson']
    
    def read(
        self,
        file_path: Path,
        text_field: str = 'text',
        encoding: str = 'utf-8',
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Read JSONL file.
        
        Args:
            file_path: Path to JSONL file
            text_field: Field containing text data
            encoding: Text encoding
            
        Yields:
            Dictionary records
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        
                        # Ensure 'text' field exists
                        if text_field != 'text' and text_field in record:
                            record['text'] = record[text_field]
                        
                        if 'text' not in record:
                            # Try to find any text-like field
                            for key in ['content', 'body', 'message']:
                                if key in record:
                                    record['text'] = record[key]
                                    break
                        
                        record['source'] = str(file_path)
                        record['line_number'] = idx
                        
                        yield record
                        self.stats['records_read'] += 1
                    
                    except json.JSONDecodeError as e:
                        self.stats['errors'] += 1
                        # Skip invalid lines but continue
                        continue
            
            self.stats['files_processed'] += 1
            self.stats['bytes_processed'] += file_path.stat().st_size
        
        except Exception as e:
            self.stats['errors'] += 1
            raise DataFormatError(
                format_type='jsonl',
                reason=f"Failed to read file: {e}",
                supported_formats=['jsonl', 'ndjson']
            )
    
    def write(
        self,
        file_path: Path,
        records: Iterator[Dict[str, Any]],
        encoding: str = 'utf-8',
        ensure_ascii: bool = False,
        **kwargs
    ):
        """
        Write records to JSONL file.
        
        Args:
            file_path: Output path
            records: Iterator of records
            encoding: Text encoding
            ensure_ascii: Whether to escape non-ASCII characters
        """
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                for record in records:
                    json_str = json.dumps(record, ensure_ascii=ensure_ascii)
                    f.write(json_str + '\n')
                    self.stats['records_read'] += 1
            
            self.stats['files_processed'] += 1
        
        except Exception as e:
            self.stats['errors'] += 1
            raise DataFormatError(
                format_type='jsonl',
                reason=f"Failed to write file: {e}",
                supported_formats=['jsonl']
            )
    
    def validate(self, file_path: Path) -> bool:
        """Validate JSONL file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try parsing first 10 lines
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    line = line.strip()
                    if line:
                        json.loads(line)
            return True
        except:
            return False
    
    def estimate_size(self, file_path: Path) -> int:
        """Estimate number of records by counting lines."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except:
            return super().estimate_size(file_path)


__all__ = ['JsonlFormatHandler']
