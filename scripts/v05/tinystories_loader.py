"""
TinyStories loader for the v0.5 real-data validation experiment.

Downloads a subset of TinyStories (HuggingFace: roneneldan/TinyStories),
tokenizes it with the zero_bpe_10k tokenizer (vocab=10000, matching the
zero_10M model config), and caches the tokenized stream as a uint16 numpy
array under /home/z/my-project/xorzen_dev/data/.

This is storage-efficient:
  - Raw text: ~50k stories ≈ 50MB
  - Tokenized uint16: ~15-25MB (vocab=10000 fits in uint16)
  - Train/val split is reproducible from a seed

We do NOT keep raw text on disk — only the cached token array.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

PROJ = Path("/home/z/my-project/xorzen_dev")
sys.path.insert(0, str(PROJ))

from xorzen.tokenizer.loader import load_pretrained

CACHE_DIR = PROJ / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_CACHE = CACHE_DIR / "tinystories_tokens_uint16.npy"
META_CACHE = CACHE_DIR / "tinystories_meta.json"

# Use the 10M Xorzen config's vocab_size for a perfect match
TOKENIZER_NAME = "zero_bpe_10k"
NUM_STORIES_TRAIN = 45_000   # ~ 8-9M tokens of training data
NUM_STORIES_VAL = 2_000      # ~ 350-400k tokens of validation data
SEED = 42


def tokenize_and_cache() -> Tuple[np.ndarray, dict]:
    """Download TinyStories, tokenize, cache as uint16 numpy array."""
    if TOKEN_CACHE.exists() and META_CACHE.exists():
        import json
        with open(META_CACHE) as f:
            meta = json.load(f)
        # We saved the array with tofile()/fromfile (raw uint16, no pickle),
        # so load via np.fromfile, not np.load (which expects .npy header).
        tokens = np.fromfile(TOKEN_CACHE, dtype=np.uint16)
        print(f"[tinystories] loaded cache: {tokens.shape} tokens, vocab={meta['vocab_size']}")
        return tokens, meta

    print(f"[tinystories] downloading TinyStories (train+val) from HuggingFace...")
    from datasets import load_dataset

    tok = load_pretrained(TOKENIZER_NAME)
    vocab_size = tok.get_vocab_size()
    assert vocab_size == 10000, f"expected 10k vocab, got {vocab_size}"
    assert vocab_size < 65536, "vocab must fit in uint16"

    t0 = time.time()
    ds_train = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ds_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    all_tokens = []
    n_train_kept = 0
    n_val_kept = 0
    boundary = 0  # index where validation starts
    rng = np.random.default_rng(SEED)

    # Collect train
    for i, ex in enumerate(ds_train):
        if n_train_kept >= NUM_STORIES_TRAIN:
            break
        text = ex["text"]
        ids = tok.encode(text)
        # Add EOS between stories (use the largest token id < vocab as a story-end marker)
        # The zero_bpe_10k tokenizer may not have a dedicated EOS; use vocab_size-1
        eos_id = vocab_size - 1
        ids = list(ids) + [eos_id]
        all_tokens.append(np.asarray(ids, dtype=np.uint16))
        n_train_kept += 1
        if n_train_kept % 5000 == 0:
            print(f"  train: {n_train_kept}/{NUM_STORIES_TRAIN} stories, "
                  f"{sum(len(a) for a in all_tokens):,} tokens, {time.time()-t0:.0f}s elapsed")

    boundary = sum(len(a) for a in all_tokens)

    # Collect val
    for i, ex in enumerate(ds_val):
        if n_val_kept >= NUM_STORIES_VAL:
            break
        text = ex["text"]
        ids = tok.encode(text)
        eos_id = vocab_size - 1
        ids = list(ids) + [eos_id]
        all_tokens.append(np.asarray(ids, dtype=np.uint16))
        n_val_kept += 1

    tokens = np.concatenate(all_tokens)
    del all_tokens
    print(f"[tinystories] total tokens: {len(tokens):,}  "
          f"(train: {boundary:,}, val: {len(tokens)-boundary:,})  "
          f"download+tokenize time: {time.time()-t0:.0f}s")

    tokens.tofile(TOKEN_CACHE)  # save raw uint16 array (no overhead)
    import json
    meta = {
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": vocab_size,
        "total_tokens": int(len(tokens)),
        "train_tokens": int(boundary),
        "val_tokens": int(len(tokens) - boundary),
        "train_stories": n_train_kept,
        "val_stories": n_val_kept,
        "seed": SEED,
        "eos_token_id": vocab_size - 1,
    }
    with open(META_CACHE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[tinystories] cached to {TOKEN_CACHE} ({TOKEN_CACHE.stat().st_size/1e6:.1f} MB)")
    return tokens, meta


def make_split(tokens: np.ndarray, meta: dict, seq_len: int, batch_size: int,
               split: str = "train", drop_last: bool = True, shuffle: bool = True,
               seed: int = SEED):
    """Slice the cached token stream into (batch_size, seq_len) chunks.

    Returns an iterable of numpy uint16 arrays of shape (batch_size, seq_len).
    The model handles the next-token shift internally (logits[:, :-1] vs labels[:, 1:]).
    """
    if split == "train":
        n = meta["train_tokens"]
        offset = 0
    elif split == "val":
        n = meta["val_tokens"]
        offset = meta["train_tokens"]
    else:
        raise ValueError(split)

    seq = tokens[offset:offset + n]
    # Number of full non-overlapping windows of length seq_len
    n_windows = len(seq) // seq_len
    if n_windows == 0:
        raise RuntimeError(f"too few tokens for seq_len={seq_len} in split={split}: {len(seq)}")
    usable = n_windows * seq_len
    seq = seq[:usable]
    windows = seq.reshape(n_windows, seq_len).astype(np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(windows)

    n_batches = n_windows // batch_size
    if drop_last and n_batches * batch_size < n_windows:
        windows = windows[:n_batches * batch_size]

    print(f"[{split}] {n_windows} windows × {seq_len} tokens = "
          f"{n_windows * seq_len:,} tokens, {n_batches} batches × {batch_size}")

    return windows, n_batches


if __name__ == "__main__":
    tokens, meta = tokenize_and_cache()
    print("\nMeta:", meta)
    windows, n_batches = make_split(tokens, meta, seq_len=128, batch_size=8, split="train")
    print("First batch shape:", windows[:8].shape, "dtype:", windows[:8].dtype)
    print("First window:", windows[0][:15])
