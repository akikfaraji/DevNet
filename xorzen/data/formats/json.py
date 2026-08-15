"""
JSON Format Handler
Handles JSON format (.json).
"""

import json
from typing import Iterator, Dict, Any, List
from pathlib import Path

from .base import BaseFormatHandler
from xorzen.exceptions import DataFormatError


class JsonFormatHandler(BaseFormatHandler):
    """
    Handler for JSON format.
    
    Can handle both:
    - Single JSON object with array of records
    - Multiple top-level JSON objects
    """
    
    FORMAT_NAME = "json"
    FILE_EXTENSIONS = ['.json']
    
    def read(
        self,
        file_path: Path,
        text_field: str = 'text',
        array_key: str = None,  # If data is in {"data": [...]}
        encoding: str = 'utf-8',
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Read JSON file.
        
        Args:
            file_path: Path to JSON file
            text_field: Field containing text data
            array_key: Key containing array of records (if nested)
            encoding: Text encoding
            
        Yields:
            Dictionary records
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Handle different JSON structures
            records: List[Dict] = []
            
            if isinstance(data, list):
                # Direct array of objects
                records = data
            elif isinstance(data, dict):
                if array_key and array_key in data:
                    # Nested array: {"data": [...]}
                    records = data[array_key]
                elif text_field in data or 'text' in data:
                    # Single record
                    records = [data]
                else:
                    # Try to find array field
                    for key, value in data.items():
                        if isinstance(value, list):
                            records = value
                            break
            
            # Yield records
            for idx, record in enumerate(records):
                if not isinstance(record, dict):
                    # Handle list of strings
                    if isinstance(record, str):
                        record = {'text': record}
                    else:
                        continue
                
                # Ensure 'text' field exists
                if text_field != 'text' and text_field in record:
                    record['text'] = record[text_field]
                
                if 'text' not in record:
                    # Try common field names
                    for key in ['content', 'body', 'message']:
                        if key in record:
                            record['text'] = record[key]
                            break
                
                record['source'] = str(file_path)
                record['index'] = idx
                
                yield record
                self.stats['records_read'] += 1
            
            self.stats['files_processed'] += 1
            self.stats['bytes_processed'] += file_path.stat().st_size
        
        except json.JSONDecodeError as e:
            self.stats['errors'] += 1
            raise DataFormatError(
                format_type='json',
                reason=f"Invalid JSON: {e}",
                supported_formats=['json']
            )
        except Exception as e:
            self.stats['errors'] += 1
            raise DataFormatError(
                format_type='json',
                reason=f"Failed to read file: {e}",
                supported_formats=['json']
            )
    
    def write(
        self,
        file_path: Path,
        records: Iterator[Dict[str, Any]],
        encoding: str = 'utf-8',
        ensure_ascii: bool = False,
        indent: int = 2,
        wrap_in_array: bool = True,
        **kwargs
    ):
        """
        Write records to JSON file.
        
        Args:
            file_path: Output path
            records: Iterator of records
            encoding: Text encoding
            ensure_ascii: Whether to escape non-ASCII characters
            indent: JSON indentation
            wrap_in_array: Wrap records in array (vs writing as-is)
        """
        try:
            # Collect all records (JSON requires full structure)
            records_list = list(records)
            self.stats['records_read'] = len(records_list)
            
            # Determine output structure
            if wrap_in_array:
                output_data = records_list
            elif len(records_list) == 1:
                output_data = records_list[0]
            else:
                output_data = {'data': records_list}
            
            # Write JSON
            with open(file_path, 'w', encoding=encoding) as f:
                json.dump(
                    output_data,
                    f,
                    ensure_ascii=ensure_ascii,
                    indent=indent
                )
            
            self.stats['files_processed'] += 1
        
        except Exception as e:
            self.stats['errors'] += 1
            raise DataFormatError(
                format_type='json',
                reason=f"Failed to write file: {e}",
                supported_formats=['json']
            )
    
    def validate(self, file_path: Path) -> bool:
        """Validate JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except:
            return False
    
    def estimate_size(self, file_path: Path) -> int:
        """Estimate number of records."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict):
                # Look for array fields
                for value in data.values():
                    if isinstance(value, list):
                        return len(value)
            
            return 1  # Single record
        except:
            return super().estimate_size(file_path)


__all__ = ['JsonFormatHandler']
