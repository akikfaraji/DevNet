#!/usr/bin/env python3
"""
XORZEN Colab Training Script
=============================
Trains a zero model on Project Gutenberg text data with GPU support.

Usage (Google Colab):
    !pip install git+https://github.com/akikfaraji/DevNet.git
    !python train_colab.py --model zero_10M --epochs 3

Usage (local):
    python train_colab.py --model zero_10M --epochs 3 --data-dir data/gutenberg/txt
"""

import os
import sys
import time
import argparse
import math
from pathlib import Path

def setup_colab():
    """Install dependencies for Google Colab."""
    import subprocess
    print("=" * 60)
    print("XORZEN Colab Setup")
    print("=" * 60)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")
        else:
            print("WARNING: No GPU detected. Go to Runtime > Change runtime type > GPU")
    except ImportError:
        print("Installing PyTorch with CUDA support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)

    # Install xorzen if not already installed
    try:
        import xorzen
        print(f"XORZEN version: {xorzen.__version__}")
    except ImportError:
        print("Installing XORZEN from GitHub...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/akikfaraji/DevNet.git"
        ], check=True)

    # Install Gutenberg dataset tools
    try:
        import requests
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)

    print("Setup complete!")


def download_gutenberg_sample(output_dir="data/gutenberg/txt", num_books=20):
    """Download a sample of Project Gutenberg books."""
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classic books from Project Gutenberg (public domain)
    book_ids = [
        1342,   # Pride and Prejudice
        11,     # Alice in Wonderland
        1661,   # Sherlock Holmes
        84,     # Frankenstein
        98,     # Tale of Two Cities
        174,    # Dorian Gray
        16,     # Peter Pan
        1260,   # Jane Eyre
        46,     # Christmas Carol
        74,     # Tom Sawyer
        5200,   # Metamorphosis
        25344,  # Scarlet Letter
        1080,   # Modest Proposal
        2701,   # Moby Dick
        35,     # Time Machine
        158,    # Emma
        1232,   # Prince
        1250,   # Republic
        2554,   # Crime and Punishment (English)
        1952,   # Yellow Wallpaper
    ][:num_books]

    downloaded = 0
    for book_id in book_ids:
        outfile = output_dir / f"book_{book_id}.txt"
        if outfile.exists():
            downloaded += 1
            continue

        try:
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                outfile.write_text(resp.text, encoding='utf-8')
                downloaded += 1
                print(f"  Downloaded book {book_id}")
            else:
                # Try alternate URL
                url = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    outfile.write_text(resp.text, encoding='utf-8')
                    downloaded += 1
                    print(f"  Downloaded book {book_id} (alt)")
        except Exception as e:
            print(f"  Failed to download book {book_id}: {e}")

    print(f"Downloaded {downloaded}/{num_books} books to {output_dir}")
    return output_dir


def estimate_training_time(model, dataset_size_tokens, batch_size, seq_length, device="cuda"):
    """
    Estimate training time based on model size and hardware.

    Uses empirical FLOP estimates:
    - Forward pass: ~2 * num_params * seq_length FLOPs per token
    - Backward pass: ~4 * num_params * seq_length FLOPs per token
    - Total per step: ~6 * num_params * seq_length * batch_size FLOPs

    GPU throughput estimates (TFLOPS):
    - T4: ~8 TFLOPS (FP16)
    - V100: ~14 TFLOPS (FP16)
    - A100: ~40 TFLOPS (FP16)
    - L4: ~15 TFLOPS (FP16)
    """
    import torch

    num_params = sum(p.numel() for p in model.parameters())

    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
6       # Estimate TFLOPS based on GPU
        gpu_tflops = 8  # default (T4)
        if "A100" in gpu_name:
            gpu_tflops = 40
        elif "V100" in gpu_name:
            gpu_tflops = 14
        elif "L4" in gpu_name:
            gpu_tflops = 15
        elif "T4" in gpu_name:
            gpu_tflops = 8
        elif "P100" in gpu_name:
            gpu_tflops = 5
        elif "A10G" in gpu_name:
            gpu_tflops = 15
    else:
        gpu_name = "CPU"
        gpu_tflops = 0.1  # ~100 GFLOPS for CPU

    # FLOPs per training step
    flops_per_step = 6 * num_params * seq_length * batch_size

    # Steps per second (accounting for ~40% utilization)
    utilization = 0.35
    steps_per_second = (gpu_tflops * 1e12 * utilization) / flops_per_step

    # Total steps for 1 epoch
    steps_per_epoch = dataset_size_tokens // (batch_size * seq_length)

    # Time per epoch
    seconds_per_epoch = steps_per_epoch / steps_per_second

    return {
        "model_params": num_params,
        "gpu_name": gpu_name,
        "gpu_tflops": gpu_tflops,
        "flops_per_step": flops_per_step,
        "steps_per_second": steps_per_second,
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_epoch": seconds_per_epoch,
        "minutes_per_epoch": seconds_per_epoch / 60,
        "hours_per_epoch": seconds_per_epoch / 3600,
    }


def train(args):
    """Main training function."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    # Import xorzen
    from xorzen.models.zero import zero_1M, zero_10M, zero_50M, zero_277M
    from xorzen.models.zero.agentic_variants import zero_agentic_nano
    from xorzen.tokenizer import load_pretrained

    # Select model
    model_map = {
        "zero_1M": zero_1M,
        "zero_10M": zero_10M,
        "zero_50M": zero_50M,
        "zero_277M": zero_277M,
        "zero_agentic_nano": zero_agentic_nano,
    }

    if args.model not in model_map:
        print(f"Unknown model: {args.model}. Choose from: {list(model_map.keys())}")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   9   print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")

    # Load tokenizer
    try:
        tokenizer = load_pretrained('zero_bpe_10k')
        vocab_size = 10000
        print(f"Tokenizer: zero_bpe_10k (vocab={vocab_size})")
    except Exception as e:
        print(f"Warning: Could not load pretrained tokenizer ({e}). Using basic tokenizer.")
        vocab_size = 10000

    # Create model
    print(f"\nCreating model: {args.model}")
    model = model_map[args.model]()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    model = model.to(device)

    # Load text data
    data_dir = Path(args.data_dir)
    text_files = list(data_dir.glob("*.txt"))
    print(f"\nData files: {len(text_files)} text files in {data_dir}")

    if not text_files:
        print("No text files found. Downloading Gutenberg sample...")
        data_dir = download_gutenberg_sample(num_books=20)
        text_files = list(data_dir.glob("*.txt"))

    # Simple text dataset
    class TextDataset(Dataset):
        def __init__(self, text, tokenizer, seq_length):
            self.seq_length = seq_length
            # Simple character/word level tokenization if no tokenizer
            if tokenizer is not None:
                try:
                    self.tokens = tokenizer.encode(text).ids
                except:
                    self.tokens = self._simple_tokenize(text)
            else:
                self.tokens = self._simple_tokenize(text)

        def _simple_tokenize(self, text):
            """Simple word-level tokenization."""
            words = text.lower().split()
            vocab = {}
            tokens = []
            for w in words:
                if w not in vocab:
                    vocab[w] = len(vocab) + 1
                tokens.append(vocab[w])
            return tokens

        def __len__(self):
            return max(0, len(self.tokens) - self.seq_length - 1)

        def __getitem__(self, idx):
            x = torch.tensor(self.tokens[idx:idx+self.seq_length], dtype=torch.long)
            y = torch.tensor(self.tokens[idx+1:idx+self.seq_length+1], dtype=torch.long)
            return x, y

    # Load and combine all text
    print("Loading text data...")
    all_text = ""
    total_chars = 0
    for f in text_files:
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
            all_text += text + "\n"
            total_chars += len(text)
        except Exception as e:
            print(f"  Warning: Could not read {f}: {e}")

    print(f"Total text: {total_chars:,} characters")

    # Estimate tokens (rough: 1 token ≈ 4 chars for BPE)
    estimated_tokens = total_chars // 4
    print(f"Estimated tokens: ~{estimated_tokens:,}")

    # Create dataset
    dataset = TextDataset(all_text, tokenizer, args.seq_length)
    print(f"Training examples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Colab compatible
        pin_memory=device.type == "cuda",
    )

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler (cosine)
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda" and args.fp16))

    # Estimate training time
    est = estimate_training_time(
        model, estimated_tokens, args.batch_size, args.seq_length,
        device="cuda" if device.type == "cuda" else "cpu"
    )

    print("\n" + "=" * 60)
    print("TRAINING TIME ESTIMATE")
    print("=" * 60)
    print(f"Model: {args.model} ({est['model_params']:,} params)")
    print(f"GPU: {est['gpu_name']} (~{est['gpu_tflops']} TFLOPS)")
    print(f"Dataset: ~{estimated_tokens:,} tokens")
    print(f"Batch size: {args.batch_size}, Seq length: {args.seq_length}")
    print(f"Steps/epoch: {est['steps_per_epoch']:,}")
    print(f"Est. speed: {est['steps_per_second']:.2f} steps/sec")
    print(f"Time per epoch: {est['hours_per_epoch']:.2f} hours ({est['minutes_per_epoch']:.1f} min)")
    print(f"Total ({args.epochs} epochs): {est['hours_per_epoch'] * args.epochs:.2f} hours")
    print("=" * 60)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=args.fp16):
                output = model(x, labels=y)
                loss = output.loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if batch_idx % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                tokens_per_sec = (batch_idx + 1) * args.batch_size * args.seq_length / max(elapsed, 1e-6)
                print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {tokens_per_sec:,.0f} tok/s"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} complete: avg_loss={avg_loss:.4f}, time={epoch_time/60:.1f}min")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = f"checkpoints/{args.model}_epoch{epoch+1}_loss{avg_loss:.4f}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'global_step': global_step,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")

    # Final save
    final_path = f"checkpoints/{args.model}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs,
        'best_loss': best_loss,
    }, final_path)
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XORZEN Colab Training")
    parser.add_argument("--model", type=str, default="zero_10M",
                        choices=["zero_1M", "zero_10M", "zero_50M", "zero_277M", "zero_agentic_nano"],
                        help="Model variant to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 mixed precision")
    parser.add_argument("--data-dir", type=str, default="data/gutenberg/txt",
                        help="Directory with text files")
    parser.add_argument("--setup-only", action="store_true", help="Only run Colab setup")
    parser.add_argument("--download-only", action="store_true", help="Only download Gutenberg data")
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate training time")

    args = parser.parse_args()

    # Always run setup in Colab
    if 'google.colab' in str(getattr(sys, 'modules', {})):
        setup_colab()

    if args.setup_only:
        setup_colab()
        sys.exit(0)

    if args.download_only:
        download_gutenberg_sample(num_books=20)
        sys.exit(0)

    train(args)
