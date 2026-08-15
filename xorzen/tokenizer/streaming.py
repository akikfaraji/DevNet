"""
Streaming Tokenizer Module

Provides efficient streaming tokenization for large-scale data processing:
- Streaming tokenization with chunking
- Memory-efficient batch processing
- Parallel tokenization
- Progress tracking and monitoring
- Token streaming for real-time applications
"""

import os
import time
import queue
import threading
from typing import Iterator, List, Optional, Dict, Any, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    warnings.warn("tokenizers library not available")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ==================== DATA STRUCTURES ====================

@dataclass
class TokenizationChunk:
    """A chunk of tokenized data."""
    chunk_id: int
    token_ids: List[int]
    tokens: List[str]
    offsets: Optional[List[Tuple[int, int]]] = None
    attention_mask: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Get number of tokens."""
        return len(self.token_ids)


@dataclass
class StreamingStats:
    """Statistics for streaming tokenization."""
    total_texts: int = 0
    total_tokens: int = 0
    total_chars: int = 0
    elapsed_time: float = 0.0
    throughput_texts_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    throughput_chars_per_sec: float = 0.0
    avg_tokens_per_text: float = 0.0
    
    def update(self, num_texts: int, num_tokens: int, num_chars: int, elapsed: float):
        """Update statistics."""
        self.total_texts += num_texts
        self.total_tokens += num_tokens
        self.total_chars += num_chars
        self.elapsed_time += elapsed
        
        if self.elapsed_time > 0:
            self.throughput_texts_per_sec = self.total_texts / self.elapsed_time
            self.throughput_tokens_per_sec = self.total_tokens / self.elapsed_time
            self.throughput_chars_per_sec = self.total_chars / self.elapsed_time
        
        if self.total_texts > 0:
            self.avg_tokens_per_text = self.total_tokens / self.total_texts
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"StreamingStats("
            f"texts={self.total_texts:,}, "
            f"tokens={self.total_tokens:,}, "
            f"throughput={self.throughput_texts_per_sec:.1f} texts/s, "
            f"{self.throughput_tokens_per_sec:,.0f} tokens/s)"
        )


# ==================== STREAMING TOKENIZER ====================

class StreamingTokenizer:
    """
    Streaming tokenizer for efficient large-scale processing.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int = 1000,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_offsets: bool = False,
        return_attention_mask: bool = True,
        num_workers: int = 1,
        buffer_size: int = 100,
        show_progress: bool = True
    ):
        """
        Initialize streaming tokenizer.
        
        Args:
            tokenizer: Base tokenizer
            chunk_size: Number of texts per chunk
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks
            return_offsets: Return character offsets
            return_attention_mask: Return attention masks
            num_workers: Number of parallel workers
            buffer_size: Size of processing buffer
            show_progress: Show progress bar
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library is required")
        
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.stride = stride
        self.return_offsets = return_offsets
        self.return_attention_mask = return_attention_mask
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.show_progress = show_progress
        
        self.stats = StreamingStats()
    
    def stream_tokenize(
        self,
        texts: Iterator[str],
        total: Optional[int] = None
    ) -> Iterator[TokenizationChunk]:
        """
        Stream tokenize texts.
        
        Args:
            texts: Iterator of texts
            total: Total number of texts (for progress bar)
            
        Yields:
            TokenizationChunk instances
        """
        start_time = time.time()
        chunk_id = 0
        buffer = []
        
        # Setup progress bar
        pbar = None
        if self.show_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=total, desc="Tokenizing", unit="texts")
        
        try:
            for text in texts:
                buffer.append(text)
                
                if len(buffer) >= self.chunk_size:
                    # Process chunk
                    chunk = self._tokenize_batch(buffer, chunk_id)
                    chunk_id += 1
                    
                    # Update stats
                    elapsed = time.time() - start_time
                    self.stats.update(
                        len(buffer),
                        len(chunk.token_ids),
                        sum(len(t) for t in buffer),
                        elapsed
                    )
                    
                    # Update progress
                    if pbar:
                        pbar.update(len(buffer))
                    
                    buffer = []
                    yield chunk
            
            # Process remaining
            if buffer:
                chunk = self._tokenize_batch(buffer, chunk_id)
                elapsed = time.time() - start_time
                self.stats.update(
                    len(buffer),
                    len(chunk.token_ids),
                    sum(len(t) for t in buffer),
                    elapsed
                )
                
                if pbar:
                    pbar.update(len(buffer))
                
                yield chunk
        
        finally:
            if pbar:
                pbar.close()
    
    def stream_tokenize_parallel(
        self,
        texts: Iterator[str],
        total: Optional[int] = None
    ) -> Iterator[TokenizationChunk]:
        """
        Stream tokenize texts with parallel processing.
        
        Args:
            texts: Iterator of texts
            total: Total number of texts
            
        Yields:
            TokenizationChunk instances
        """
        if self.num_workers <= 1:
            yield from self.stream_tokenize(texts, total)
            return
        
        start_time = time.time()
        
        # Setup progress bar
        pbar = None
        if self.show_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=total, desc="Tokenizing (parallel)", unit="texts")
        
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                chunk_id = 0
                buffer = []
                
                for text in texts:
                    buffer.append(text)
                    
                    if len(buffer) >= self.chunk_size:
                        # Submit chunk for processing
                        future = executor.submit(self._tokenize_batch, buffer.copy(), chunk_id)
                        futures[future] = (chunk_id, len(buffer), sum(len(t) for t in buffer))
                        chunk_id += 1
                        buffer = []
                    
                    # Check completed futures
                    if len(futures) >= self.buffer_size:
                        for future in as_completed(futures):
                            chunk_id_done, num_texts, num_chars = futures.pop(future)
                            chunk = future.result()
                            
                            elapsed = time.time() - start_time
                            self.stats.update(num_texts, len(chunk.token_ids), num_chars, elapsed)
                            
                            if pbar:
                                pbar.update(num_texts)
                            
                            yield chunk
                
                # Submit remaining
                if buffer:
                    future = executor.submit(self._tokenize_batch, buffer, chunk_id)
                    futures[future] = (chunk_id, len(buffer), sum(len(t) for t in buffer))
                
                # Wait for remaining futures
                for future in as_completed(futures):
                    chunk_id_done, num_texts, num_chars = futures[future]
                    chunk = future.result()
                    
                    elapsed = time.time() - start_time
                    self.stats.update(num_texts, len(chunk.token_ids), num_chars, elapsed)
                    
                    if pbar:
                        pbar.update(num_texts)
                    
                    yield chunk
        
        finally:
            if pbar:
                pbar.close()
    
    def _tokenize_batch(self, texts: List[str], chunk_id: int) -> TokenizationChunk:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of texts
            chunk_id: Chunk identifier
            
        Returns:
            TokenizationChunk
        """
        # Prepare encoding options
        encode_kwargs = {
            "add_special_tokens": True, # Assuming special tokens are always added in streaming
            "max_length": self.max_length,
            "padding": "max_length" if self.max_length else False, # Pad to max_length if max_length is set
            "truncation": self.truncation,
        }

        # Tokenize all texts
        encodings = self.tokenizer.encode_batch(texts, **encode_kwargs)
        
        # Flatten token IDs
        all_token_ids = []
        all_tokens = []
        all_offsets = [] if self.return_offsets else None
        all_attention_mask = [] if self.return_attention_mask else None
        
        for encoding in encodings:
            all_token_ids.extend(encoding.ids)
            all_tokens.extend(encoding.tokens)
            
            if self.return_offsets:
                all_offsets.extend(encoding.offsets)
            
            if self.return_attention_mask:
                all_attention_mask.extend(encoding.attention_mask)
        
        return TokenizationChunk(
            chunk_id=chunk_id,
            token_ids=all_token_ids,
            tokens=all_tokens,
            offsets=all_offsets,
            attention_mask=all_attention_mask,
            metadata={'num_texts': len(texts)}
        )
    
    def get_stats(self) -> StreamingStats:
        """Get streaming statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = StreamingStats()


# ==================== TOKEN STREAM ====================

class TokenStream:
    """
    Stream individual tokens for real-time processing.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        buffer_size: int = 1000,
        prefetch: int = 10
    ):
        """
        Initialize token stream.
        
        Args:
            tokenizer: Base tokenizer
            buffer_size: Size of token buffer
            prefetch: Number of texts to prefetch
        """
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        
        self.token_queue = deque(maxlen=buffer_size)
        self.text_queue = deque(maxlen=prefetch)
        self.finished = False
    
    def stream_tokens(self, texts: Iterator[str]) -> Iterator[int]:
        """
        Stream individual tokens.
        
        Args:
            texts: Iterator of texts
            
        Yields:
            Token IDs
        """
        for text in texts:
            encoding = self.tokenizer.encode(text)
            for token_id in encoding.ids:
                yield token_id
    
    def stream_tokens_with_metadata(
        self,
        texts: Iterator[str]
    ) -> Iterator[Tuple[int, str, Tuple[int, int]]]:
        """
        Stream tokens with metadata.
        
        Args:
            texts: Iterator of texts
            
        Yields:
            (token_id, token_str, offset) tuples
        """
        for text in texts:
            encoding = self.tokenizer.encode(text)
            
            for token_id, token_str, offset in zip(
                encoding.ids,
                encoding.tokens,
                encoding.offsets
            ):
                yield token_id, token_str, offset


# ==================== CHUNKED FILE TOKENIZER ====================

class ChunkedFileTokenizer:
    """
    Tokenize large files in chunks with memory efficiency.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size_bytes: int = 10_000_000,  # 10 MB chunks
        overlap_bytes: int = 1000,
        encoding: str = 'utf-8',
        errors: str = 'ignore'
    ):
        """
        Initialize chunked file tokenizer.
        
        Args:
            tokenizer: Base tokenizer
            chunk_size_bytes: Size of file chunks in bytes
            overlap_bytes: Overlap between chunks
            encoding: Text encoding
            errors: Error handling
        """
        self.tokenizer = tokenizer
        self.chunk_size_bytes = chunk_size_bytes
        self.overlap_bytes = overlap_bytes
        self.encoding = encoding
        self.errors = errors
    
    def tokenize_file(
        self,
        file_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None
    ) -> Iterator[TokenizationChunk]:
        """
        Tokenize file in chunks.
        
        Args:
            file_path: Path to input file
            save_path: Optional path to save tokens
            
        Yields:
            TokenizationChunk instances
        """
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        
        chunk_id = 0
        position = 0
        overlap_text = ""
        
        with open(file_path, 'r', encoding=self.encoding, errors=self.errors) as f:
            while position < file_size:
                # Read chunk
                f.seek(position)
                chunk_data = f.read(self.chunk_size_bytes)
                
                if not chunk_data:
                    break
                
                # Add overlap from previous chunk
                full_chunk = overlap_text + chunk_data
                
                # Tokenize
                encoding = self.tokenizer.encode(full_chunk)
                
                # Create chunk
                chunk = TokenizationChunk(
                    chunk_id=chunk_id,
                    token_ids=encoding.ids,
                    tokens=encoding.tokens,
                    metadata={
                        'file': str(file_path),
                        'position': position,
                        'size': len(chunk_data)
                    }
                )
                
                # Save if requested
                if save_path:
                    self._save_chunk(chunk, save_path)
                
                yield chunk
                
                # Update for next iteration
                position += len(chunk_data.encode(self.encoding))
                overlap_text = chunk_data[-self.overlap_bytes:] if len(chunk_data) > self.overlap_bytes else chunk_data
                chunk_id += 1
    
    def _save_chunk(self, chunk: TokenizationChunk, base_path: Union[str, Path]):
        """Save chunk to disk."""
        import numpy as np
        
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        chunk_file = base_path / f"chunk_{chunk.chunk_id:06d}.npz"
        
        np.savez_compressed(
            chunk_file,
            token_ids=np.array(chunk.token_ids, dtype=np.int32),
            chunk_id=chunk.chunk_id,
            **chunk.metadata
        )


# ==================== BATCH TOKENIZER ====================

class BatchTokenizer:
    """
    Efficient batch tokenization with automatic batching.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ):
        """
        Initialize batch tokenizer.
        
        Args:
            tokenizer: Base tokenizer
            batch_size: Batch size
            max_length: Maximum length
            padding: Enable padding
            truncation: Enable truncation
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Configure tokenizer (moved to tokenize_batched)
    
    def tokenize_batched(
        self,
        texts: List[str],
        return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tokenize texts in batches.
        
        Args:
            texts: List of texts
            return_tensors: Return format ('pt', 'tf', 'np', None)
            
        Returns:
            Dictionary with tokenization results
        """
        all_input_ids = []
        all_attention_masks = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Prepare encoding options
            encode_kwargs = {
                "add_special_tokens": True, # Assuming special tokens are always added
                "max_length": self.max_length,
                "padding": "max_length" if self.padding and self.max_length else False,
                "truncation": self.truncation,
            }
            
            encodings = self.tokenizer.encode_batch(batch, **encode_kwargs)
            
            for encoding in encodings:
                all_input_ids.append(encoding.ids)
                all_attention_masks.append(encoding.attention_mask)
        
        result = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks
        }
        
        # Convert to requested tensor format
        if return_tensors == 'np':
            import numpy as np
            result = {k: np.array(v) for k, v in result.items()}
        elif return_tensors == 'pt':
            try:
                import torch
                result = {k: torch.tensor(v) for k, v in result.items()}
            except ImportError:
                warnings.warn("PyTorch not available, returning lists")
        elif return_tensors == 'tf':
            try:
                import tensorflow as tf
                result = {k: tf.constant(v) for k, v in result.items()}
            except ImportError:
                warnings.warn("TensorFlow not available, returning lists")
        
        return result
    
    def tokenize_iterator(
        self,
        texts: Iterator[str]
    ) -> Iterator[Dict[str, List[int]]]:
        """
        Tokenize iterator of texts.
        
        Args:
            texts: Iterator of texts
            
        Yields:
            Dictionaries with tokenization results
        """
        batch = []
        
        for text in texts:
            batch.append(text)
            
            if len(batch) >= self.batch_size:
                result = self.tokenize_batched(batch)
                
                for i in range(len(batch)):
                    yield {
                        'input_ids': result['input_ids'][i],
                        'attention_mask': result['attention_mask'][i]
                    }
                
                batch = []
        
        # Process remaining
        if batch:
            result = self.tokenize_batched(batch)
            
            for i in range(len(batch)):
                yield {
                    'input_ids': result['input_ids'][i],
                    'attention_mask': result['attention_mask'][i]
                }


# ==================== MEMORY-MAPPED TOKENIZER ====================

class MemoryMappedTokenizer:
    """
    Tokenize and store results in memory-mapped files for large datasets.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        output_dir: Union[str, Path],
        dtype: str = 'int32'
    ):
        """
        Initialize memory-mapped tokenizer.
        
        Args:
            tokenizer: Base tokenizer
            output_dir: Output directory
            dtype: NumPy dtype
        """
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
    
    def tokenize_and_map(
        self,
        texts: List[str],
        mmap_name: str = "tokens"
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Tokenize texts and create memory-mapped array.
        
        Args:
            texts: List of texts
            mmap_name: Name for memory-mapped file
            
        Returns:
            (mmap_array, metadata)
        """
        import numpy as np
        
        # Tokenize all texts
        all_token_ids = []
        offsets = [0]
        
        for text in texts:
            encoding = self.tokenizer.encode(text)
            all_token_ids.extend(encoding.ids)
            offsets.append(len(all_token_ids))
        
        # Create memory-mapped file
        mmap_path = self.output_dir / f"{mmap_name}.mmap"
        mmap_array = np.memmap(
            mmap_path,
            dtype=self.dtype,
            mode='w+',
            shape=(len(all_token_ids),)
        )
        
        # Write tokens
        mmap_array[:] = all_token_ids
        mmap_array.flush()
        
        # Save metadata
        metadata = {
            'num_texts': len(texts),
            'total_tokens': len(all_token_ids),
            'offsets': offsets,
            'dtype': self.dtype
        }
        
        metadata_path = self.output_dir / f"{mmap_name}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return mmap_array, metadata
    
    def load_mmap(self, mmap_name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load memory-mapped tokens.
        
        Args:
            mmap_name: Name of memory-mapped file
            
        Returns:
            (mmap_array, metadata)
        """
        import numpy as np
        import json
        
        # Load metadata
        metadata_path = self.output_dir / f"{mmap_name}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load memory-mapped file
        mmap_path = self.output_dir / f"{mmap_name}.mmap"
        mmap_array = np.memmap(
            mmap_path,
            dtype=metadata['dtype'],
            mode='r',
            shape=(metadata['total_tokens'],)
        )
        
        return mmap_array, metadata


__all__ = [
    'TokenizationChunk',
    'StreamingStats',
    'StreamingTokenizer',
    'TokenStream',
    'ChunkedFileTokenizer',
    'BatchTokenizer',
    'MemoryMappedTokenizer',
]

