"""
Sharded Dataset Module for xorzen
Version: 1.0

Provides a PyTorch Dataset implementation for reading data that has been
tokenized and saved into multiple sharded files (e.g., .npy, .bin).

This class is designed to work with the output of `xorzen.data.loader.XORZENXDataLoader`.
It efficiently loads data for training by memory-mapping the shards, which keeps
the RAM usage low even for very large datasets.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class ShardedDataset(Dataset):
    """
    A PyTorch Dataset for reading sharded, tokenized data from disk.
    
    This dataset is designed to handle large tokenized datasets by loading them
    from multiple smaller files (shards). It uses memory-mapping for efficiency,
    allowing it to handle datasets much larger than the available RAM.

    Args:
        data_dir (Path): The directory containing the .npy shard files.
        context_length (int): The sequence length for each sample.
    """
    def __init__(self, data_dir: Path, context_length: int):
        self.data_dir = data_dir
        self.context_length = context_length
        self.shard_paths = sorted(list(self.data_dir.glob("*.npy")))

        if not self.shard_paths:
            raise FileNotFoundError(f"No .npy shard files found in directory: {self.data_dir}")

        # Load shards using memory-mapping for efficiency
        self.shards = [np.load(p, mmap_mode='r') for p in self.shard_paths]
        self.shard_lengths = [len(s) for s in self.shards]
        self.total_length = sum(self.shard_lengths)
        
        # Cumulative lengths for quick index-to-shard mapping
        self.cumulative_lengths = np.cumsum([0] + self.shard_lengths)

        if self.total_length <= self.context_length:
            raise ValueError("Total dataset length is smaller than the context length. Not enough data to create even one sample.")

    def __len__(self):
        """
        Returns the total number of possible sequences of `context_length`
        that can be extracted from the dataset.
        """
        # Each sample requires context_length + 1 tokens (for x and y)
        # We can create one sample for each token in the dataset, minus the last `context_length` ones
        return self.total_length - self.context_length

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single training sample (input_ids and labels).

        Args:
            idx (int): The index of the sample.

        Returns:
            A dictionary containing 'input_ids' and 'labels' tensors.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        # Calculate the global start position for the sequence of (context_length + 1)
        start_token_global_idx = idx

        # Find which shard the sequence starts in using binary search
        # This is much faster than iterating through all shards
        shard_idx = np.searchsorted(self.cumulative_lengths, start_token_global_idx, side='right') - 1
        
        # Calculate the starting position within that specific shard
        pos_in_shard = start_token_global_idx - self.cumulative_lengths[shard_idx]

        # Initialize a buffer to hold the tokens for one sample
        tokens_buffer = np.zeros(self.context_length + 1, dtype=np.int64)
        
        tokens_copied = 0
        current_shard_idx = shard_idx
        current_pos_in_shard = pos_in_shard

        # Efficiently copy tokens from one or more shards into the buffer
        while tokens_copied < self.context_length + 1:
            if current_shard_idx >= len(self.shards):
                # This should not happen with a correct __len__ implementation,
                # but as a safeguard, pad with zeros.
                break
            
            shard = self.shards[current_shard_idx]
            
            # Number of tokens available in the current shard from the current position
            remaining_in_shard = self.shard_lengths[current_shard_idx] - current_pos_in_shard
            
            # Number of tokens we still need to complete the sequence
            needed_for_sequence = (self.context_length + 1) - tokens_copied
            
            # Number of tokens to copy in this iteration (the minimum of the two)
            num_to_copy = min(remaining_in_shard, needed_for_sequence)
            
            if num_to_copy > 0:
                # Copy the slice of the shard into our buffer
                end_pos_in_shard = current_pos_in_shard + num_to_copy
                tokens_buffer[tokens_copied : tokens_copied + num_to_copy] = shard[current_pos_in_shard : end_pos_in_shard]
                tokens_copied += num_to_copy
            
            # If we've exhausted the current shard, move to the next one
            if tokens_copied < self.context_length + 1:
                current_shard_idx += 1
                current_pos_in_shard = 0 # Start from the beginning of the new shard

        # Create input (x) and labels (y) for language modeling
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
            "dtype": str(self.shards[0].dtype) if self.shards else "unknown",
            "num_sequences": len(self),
            "sequence_length": self.context_length,
            "total_tokens": self.total_length,
            "num_shards": len(self.shards),
            "shard_lengths": self.shard_lengths
        }


__all__ = ['ShardedDataset']

