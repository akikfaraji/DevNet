"""
Text Format Handler
Handles plain text files (.txt).
"""

from typing import Iterator, Dict, Any
from pathlib import Path

from .base import BaseFormatHandler
from xorzen.exceptions import DataFormatError


class TxtFormatHandler(BaseFormatHandler):
    """
    Handler for plain text files.
    
    Each line or paragraph is treated as a separate record.
    """
    
    FORMAT_NAME = "txt"
    FILE_EXTENSIONS = ['.txt', '.text']
    
    def read(
        self,
        file_path: Path,
        mode: str = 'line',  # 'line' or 'paragraph'
        encoding: str = 'utf-8',
        errors: str = 'ignore',
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Read text file.
        
        Args:
            file_path: Path to text file
            mode: 'line' (one record per line) or 'paragraph' (blank line separated)
            encoding: Text encoding
            errors: Error handling strategy
            
        Yields:
            Records with 'text' field
        """
        try:
            with open(file_path, 'r', encoding=encoding, errors=errors) as f:
                if mode == 'line':
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if line:  # Skip empty lines
                            yield {
                                'text': line,
                                'source': str(file_path),
                                'line_number': idx + 1
                            }
                            self.stats['records_read'] += 1
                
                elif mode == 'paragraph':
                    paragraph = []
                    para_num = 1
                    
                    for line in f:
                        line = line.strip()
                        if line:
                            paragraph.append(line)
                        elif paragraph:  # Empty line and we have content
                            yield {
                                'text': ' '.join(paragraph),
                                'source': str(file_path),
                                'paragraph_number': para_num
                            }
                            self.stats['records_read'] += 1
                            paragraph = []
                            para_num += 1
                    
                    # Don't forget last paragraph
                    if paragraph:
                        yield {
                            'text': ' '.join(paragraph),
                            'source': str(file_path),
                            'paragraph_number': para_num
                        }
                        self.stats['records_read'] += 1
            
            self.stats['files_processed'] += 1
            self.stats['bytes_processed'] += file_path.stat().st_size
        
        except Exception as e:
            self.stats['errors'] += 1
            raise DataFormatError(
                format_type='txt',
                reason=f"Failed to read file: {e}",
                supported_formats=['txt', 'text']
            )
    
    def write(
        self,
        file_path: Path,
        records: Iterator[Dict[str, Any]],
        mode: str = 'line',
        encoding: str = 'utf-8',
        **kwargs
    ):
        """
        Write records to text file.
        
        Args:
            file_path: Output path
            records: Iterator of records
            mode: 'line' or 'paragraph'
            encoding: Text encoding
        """
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                for record in records:
                    text = record.get('text', '')
                    if not text:
                        continue
                    
                    f.write(text)
                    
                    if mode == 'line':
                        f.write('\n')
                    elif mode == 'paragraph':
                        f.write('\n\n')
                    
                    self.stats['records_read'] += 1
            
            self.stats['files_processed'] += 1
        
        except Exception as e:
            self.stats['errors'] += 1
            raise DataFormatError(
                format_type='txt',
                reason=f"Failed to write file: {e}",
                supported_formats=['txt']
            )
    
    def validate(self, file_path: Path) -> bool:
        """Validate text file (just check if readable)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Try reading first 1KB
                f.read(1024)
            return True
        except:
            return False


__all__ = ['TxtFormatHandler']
