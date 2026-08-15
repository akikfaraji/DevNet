"""
Sharded Binary Dataset Module for xorzen
Version: 1.0

Provides a PyTorch Dataset implementation for reading data that has been
tokenized and saved into multiple sharded raw binary files (.bin).

This class is designed to work with the output of `xorzen.data.loader.XORZENXDataLoader`
when `shard_format='bin'`. It efficiently loads data for training by memory-mapping
the shards, which keeps RAM usage low even for very large datasets.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional

class BinDataset(Dataset):
    """
    A PyTorch Dataset for reading sharded, tokenized data from .bin files.

    This dataset handles large tokenized datasets by loading them from multiple
    smaller binary files (shards). It uses memory-mapping for efficiency.

    Args:
        data_dir (Path): The directory containing the .bin shard files.
        context_length (int): The sequence length for each sample.
        dtype (np.dtype): The numpy dtype of the tokens in the binary files.
    """
    def __init__(self, data_dir: Path, context_length: int, dtype: np.dtype = np.uint16):
        self.data_dir = data_dir
        self.context_length = context_length
        self.dtype = dtype
        self.shard_paths = sorted(list(self.data_dir.glob("*.bin")))

        if not self.shard_paths:
            raise FileNotFoundError(f"No .bin shard files found in directory: {self.data_dir}")

        # Memory-map the shards for efficiency
        self.shards = [np.memmap(p, dtype=self.dtype, mode='r') for p in self.shard_paths]
        self.shard_lengths = [len(s) for s in self.shards]
        self.total_length = sum(self.shard_lengths)
        
        # Cumulative lengths for quick index-to-shard mapping
        self.cumulative_lengths = np.cumsum([0] + self.shard_lengths)

        if self.total_length <= self.context_length:
            raise ValueError("Total dataset length is smaller than the context length.")

    def __len__(self):
        """
        Returns the total number of possible sequences of `context_length`
        that can be extracted from the dataset.
        """
        return self.total_length - self.context_length

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single training sample.

        Args:
            idx (int): The index of the sample.

        Returns:
            A dictionary containing 'input_ids' and 'labels' tensors.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        start_token_global_idx = idx
        shard_idx = np.searchsorted(self.cumulative_lengths, start_token_global_idx, side='right') - 1
        pos_in_shard = start_token_global_idx - self.cumulative_lengths[shard_idx]

        tokens_buffer = np.zeros(self.context_length + 1, dtype=np.int64)
        
        tokens_copied = 0
        current_shard_idx = shard_idx
        current_pos_in_shard = pos_in_shard

        while tokens_copied < self.context_length + 1:
            if current_shard_idx >= len(self.shards):
                break
            
            shard = self.shards[current_shard_idx]
            remaining_in_shard = self.shard_lengths[current_shard_idx] - current_pos_in_shard
            needed_for_sequence = (self.context_length + 1) - tokens_copied
            num_to_copy = min(remaining_in_shard, needed_for_sequence)
            
            if num_to_copy > 0:
                end_pos_in_shard = current_pos_in_shard + num_to_copy
                tokens_buffer[tokens_copied : tokens_copied + num_to_copy] = shard[current_pos_in_shard : end_pos_in_shard]
                tokens_copied += num_to_copy
            
            if tokens_copied < self.context_length + 1:
                current_shard_idx += 1
                current_pos_in_shard = 0

        x = torch.from_numpy(tokens_buffer[:-1])
        y = torch.from_numpy(tokens_buffer[1:])
        
        return {'input_ids': x, 'labels': y}

    def get_stats(self) -> dict:
        """
        Returns a dictionary containing statistics about the dataset.
        """
        # Calculate file size in MB
        file_size_mb = 0
        for p in self.shard_paths:
            if p.exists():
                file_size_mb += p.stat().st_size
        file_size_mb /= (1024 * 1024)

        return {
            "file_size_mb": file_size_mb,
            "shape": (len(self), self.context_length), # Represent as (num_sequences, context_length)
            "dtype": str(self.dtype),
            "num_sequences": len(self),
            "sequence_length": self.context_length,
            "total_tokens": self.total_length,
            "num_shards": len(self.shards),
            "shard_lengths": self.shard_lengths
        }

__all__ = ['BinDataset']

