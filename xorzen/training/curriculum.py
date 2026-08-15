"""
CurriculumTrainer
==================
Trains a xorzen model by reading one file at a time, testing on it, then
moving on — like a student working through a reading list.

Flow for each book:
    1. Tokenize the file text
    2. Split into train / eval chunks (90% / 10%)
    3. Run N gradient steps on the train chunk (next-token prediction)
    4. Evaluate loss on the eval chunk
    5. Decide whether to revisit (if loss is still high) or move on
    6. Log everything to a JSON session log
    7. Save a checkpoint after every book

On CPU at ~5M parameters (IGRIS-Nano) this runs comfortably.
Estimated time: 30s - 3min per book depending on length.

Usage:
    from xorzen.training.curriculum import CurriculumTrainer, CurriculumConfig
    from xorzen.models.zero import zero_10M
    from xorzen.tokenizer import load_pretrained

    tokenizer = load_pretrained('zero_bpe_10k')   # vocab_size=10000, trained on Gutenberg
    model     = zero_10M()                         # vocab_size=10000, matches tokenizer

    trainer = CurriculumTrainer(
        model=model,
        tokenizer=tokenizer,
        data_dir='data/gutenberg/txt',
        output_dir='checkpoints/gutenberg_10m_run1',
    )
    trainer.train()
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from xorzen.utils.logger import get_logger


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurriculumConfig:
    """All tunable knobs for CurriculumTrainer."""

    # --- Data ---
    file_glob: str = "*.txt"          # Which files to pick up from data_dir
    shuffle_files: bool = True        # Randomise book order each run
    train_split: float = 0.90         # Fraction of each book used for training
    max_seq_len: int = 512            # Tokens per training sequence
    max_tokens_per_book: int = 50_000 # Cap tokens per book (keeps CPU time sane)

    # --- Training per book ---
    steps_per_book: int = 64          # Gradient steps on each book
    batch_size: int = 4               # Sequences per gradient step
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 4             # LR warmup steps at start of each book

    # --- Revisit logic ---
    # If eval loss after a book is still above this threshold,
    # the book is added back to a "revisit queue" for later.
    revisit_loss_threshold: float = 3.5
    max_revisits: int = 2             # Max times we revisit any single book

    # --- Checkpointing ---
    save_every_n_books: int = 10      # Save full model checkpoint every N books
    resume: bool = True               # Auto-resume from last checkpoint if found

    # --- Logging ---
    log_every_n_steps: int = 16       # Print training loss every N steps


# ─────────────────────────────────────────────────────────────────────────────
# Simple in-memory dataset for one book
# ─────────────────────────────────────────────────────────────────────────────

class BookDataset:
    """
    Tokenises one text file and slices it into fixed-length sequences.
    Keeps everything as plain Python lists — no numpy/torch until batching.
    """

    def __init__(
        self,
        token_ids: List[int],
        seq_len: int,
        max_tokens: int,
    ):
        # Hard cap so no single giant book dominates training time
        token_ids = token_ids[:max_tokens]

        # Slice into non-overlapping windows of length seq_len+1
        # Input  = ids[i : i+seq_len]
        # Target = ids[i+1 : i+seq_len+1]
        self.sequences: List[Tuple[List[int], List[int]]] = []
        for i in range(0, len(token_ids) - seq_len, seq_len):
            src = token_ids[i : i + seq_len]
            tgt = token_ids[i + 1 : i + seq_len + 1]
            if len(src) == seq_len and len(tgt) == seq_len:
                self.sequences.append((src, tgt))

    def __len__(self) -> int:
        return len(self.sequences)

    def split(self, train_frac: float):
        """Return (train_dataset, eval_dataset) by splitting sequences."""
        n = max(1, int(len(self.sequences) * train_frac))
        train_ds = _SequenceSlice(self.sequences[:n])
        eval_ds  = _SequenceSlice(self.sequences[n:])
        return train_ds, eval_ds


class _SequenceSlice:
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def get_batch(self, indices: List[int]) -> Tuple["torch.Tensor", "torch.Tensor"]:
        srcs = [self.sequences[i][0] for i in indices]
        tgts = [self.sequences[i][1] for i in indices]
        return (
            torch.tensor(srcs, dtype=torch.long),
            torch.tensor(tgts, dtype=torch.long),
        )

    def random_batch(self, batch_size: int):
        indices = random.sample(
            range(len(self.sequences)),
            min(batch_size, len(self.sequences))
        )
        return self.get_batch(indices)


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumTrainer:
    """
    Train a xorzen model one book at a time, with per-book evaluation
    and optional revisiting of books the model struggled with.
    """

    def __init__(
        self,
        model: "nn.Module",
        tokenizer,
        data_dir: str,
        output_dir: str = "checkpoints/curriculum",
        config: Optional[CurriculumConfig] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CurriculumTrainer.")

        self.model     = model
        self.tokenizer = tokenizer
        self.data_dir  = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cfg       = config or CurriculumConfig()
        self.logger    = get_logger()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Session log — one JSON object per book
        self.log_path = self.output_dir / "curriculum_log.jsonl"

        # Optimiser — created once, shared across all books so momentum
        # carries forward (the model doesn't forget the optimiser state)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # State that survives between books
        self.state: Dict[str, Any] = {
            "books_completed": 0,
            "total_steps": 0,
            "revisit_queue": [],   # list of {"path": ..., "revisits_left": N}
            "book_results": [],    # summary per book for final report
        }

        self._state_path = self.output_dir / "curriculum_state.json"

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def train(self):
        """Run the full curriculum."""
        files = sorted(self.data_dir.glob(self.cfg.file_glob))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.cfg.file_glob}' in {self.data_dir}"
            )

        self.logger.info("curriculum", f"Found {len(files)} books in {self.data_dir}")

        # Resume from checkpoint if available
        start_index = 0
        if self.cfg.resume:
            start_index = self._try_resume()

        if self.cfg.shuffle_files:
            random.shuffle(files)

        # ── Main loop ──────────────────────────────────────────────────
        for book_idx, file_path in enumerate(files):

            if book_idx < start_index:
                continue   # Already trained on this book

            self.logger.info(
                "curriculum",
                f"[{book_idx + 1}/{len(files)}] {file_path.name}"
            )

            result = self._train_one_book(file_path, book_idx)
            self._log_result(result)

            # Revisit queue check
            if (
                result["eval_loss"] > self.cfg.revisit_loss_threshold
                and result["revisits_done"] < self.cfg.max_revisits
            ):
                self.state["revisit_queue"].append({
                    "path": str(file_path),
                    "revisits_left": self.cfg.max_revisits - result["revisits_done"],
                })
                self.logger.info(
                    "curriculum",
                    f"  → Added to revisit queue "
                    f"(eval_loss={result['eval_loss']:.3f})"
                )

            self.state["books_completed"] += 1

            # Periodic checkpoint
            if (book_idx + 1) % self.cfg.save_every_n_books == 0:
                self._save_checkpoint(book_idx + 1)

        # ── Revisit pass ───────────────────────────────────────────────
        if self.state["revisit_queue"]:
            self.logger.info(
                "curriculum",
                f"Revisit pass: {len(self.state['revisit_queue'])} books"
            )
            queue = list(self.state["revisit_queue"])
            self.state["revisit_queue"] = []
            for entry in queue:
                path = Path(entry["path"])
                if path.exists():
                    result = self._train_one_book(
                        path,
                        book_idx=None,
                        revisit_number=self.cfg.max_revisits - entry["revisits_left"] + 1
                    )
                    self._log_result(result)

        # ── Final checkpoint ───────────────────────────────────────────
        self._save_checkpoint("final")
        self._write_summary()
        self.logger.info("curriculum", "Training complete.")

    # ─────────────────────────────────────────────────────────────────────
    # Core: train on one book
    # ─────────────────────────────────────────────────────────────────────

    def _train_one_book(
        self,
        file_path: Path,
        book_idx,
        revisit_number: int = 0,
    ) -> Dict[str, Any]:
        """
        Tokenise, split, train, evaluate one file.
        Returns a result dict for logging.
        """
        t0 = time.time()

        # 1. Read & tokenize
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as e:
            self.logger.warning("curriculum", f"  Could not read {file_path.name}: {e}")
            return self._empty_result(file_path, revisit_number)

        # Extract author/title from filename ("Author___Title.txt")
        stem = file_path.stem
        if "___" in stem:
            author, _, title = stem.partition("___")
        else:
            author, title = "Unknown", stem

        token_ids = self._tokenize(text)
        if len(token_ids) < self.cfg.max_seq_len * 2:
            self.logger.info(
                "curriculum",
                f"  Skipping '{file_path.name}' — too short "
                f"({len(token_ids)} tokens)"
            )
            return self._empty_result(file_path, revisit_number)

        # 2. Build dataset & split
        dataset  = BookDataset(token_ids, self.cfg.max_seq_len, self.cfg.max_tokens_per_book)
        train_ds, eval_ds = dataset.split(self.cfg.train_split)

        if len(train_ds) == 0:
            return self._empty_result(file_path, revisit_number)

        # 3. Train
        train_loss = self._run_train_steps(train_ds)

        # 4. Evaluate
        eval_loss = self._run_eval(eval_ds)

        elapsed = time.time() - t0
        self.logger.info(
            "curriculum",
            f"  train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}  "
            f"seqs={len(dataset)}  time={elapsed:.1f}s"
        )

        return {
            "file": file_path.name,
            "author": author,
            "title": title,
            "train_loss": round(train_loss, 4),
            "eval_loss": round(eval_loss, 4),
            "n_sequences": len(dataset),
            "n_tokens": len(token_ids),
            "elapsed_s": round(elapsed, 1),
            "revisits_done": revisit_number,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Training steps
    # ─────────────────────────────────────────────────────────────────────

    def _run_train_steps(self, train_ds: "_SequenceSlice") -> float:
        """Run cfg.steps_per_book gradient steps. Returns mean training loss."""
        self.model.train()
        total_loss = 0.0
        cfg = self.cfg

        for step in range(cfg.steps_per_book):

            # LR warmup (tiny — just for the first few steps of a new book)
            if step < cfg.warmup_steps:
                scale = (step + 1) / cfg.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = cfg.learning_rate * scale
            else:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = cfg.learning_rate

            src, tgt = train_ds.random_batch(cfg.batch_size)
            # src, tgt: (batch, seq_len)

            self.optimizer.zero_grad()
            loss = self._forward_loss(src, tgt)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            self.state["total_steps"] += 1

            if (step + 1) % cfg.log_every_n_steps == 0:
                self.logger.debug(
                    "curriculum",
                    f"    step {step + 1}/{cfg.steps_per_book}  "
                    f"loss={total_loss / (step + 1):.4f}"
                )

        return total_loss / cfg.steps_per_book

    @torch.no_grad()
    def _run_eval(self, eval_ds: "_SequenceSlice") -> float:
        """Evaluate on all eval sequences. Returns mean cross-entropy loss."""
        if len(eval_ds) == 0:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        n_batches  = 0
        batch_size = self.cfg.batch_size * 2  # Larger batch — no gradients needed

        indices = list(range(len(eval_ds)))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            src, tgt  = eval_ds.get_batch(batch_idx)
            total_loss += self._forward_loss(src, tgt).item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _forward_loss(self, src: "torch.Tensor", tgt: "torch.Tensor") -> "torch.Tensor":
        """
        Run the model forward and compute cross-entropy loss.
        Handles all xorzen output formats: XORZENXModelOutput dataclass, dict,
        tuple, or plain tensor.
        """
        output = self.model(src)

        # --- unwrap logits from whatever the model returns ---
        logits = None

        # XORZENXModelOutput dataclass (has .logits attribute)
        if hasattr(output, 'logits') and output.logits is not None:
            logits = output.logits

        # Plain dict
        elif isinstance(output, dict):
            logits = output.get('logits') or output.get('output')
            if logits is None:
                for v in output.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 3:
                        logits = v
                        break

        # Tuple / list — first element is logits by convention
        elif isinstance(output, (tuple, list)):
            logits = output[0]

        # Plain tensor
        elif isinstance(output, torch.Tensor):
            logits = output

        if logits is None or not isinstance(logits, torch.Tensor):
            raise ValueError(
                f"Could not extract logits from model output (type={type(output)}). "
                "Expected XORZENXModelOutput with .logits, dict, tuple, or Tensor."
            )

        # logits: (batch, seq_len, vocab_size)  →  flatten for cross-entropy
        batch, seq_len, vocab = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(batch * seq_len, vocab),
            tgt.reshape(batch * seq_len),
            ignore_index=-100,
        )
        return loss

    # ─────────────────────────────────────────────────────────────────────
    # Tokenization
    # ─────────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using the xorzen tokenizer.
        Wraps the sequence with BOS and EOS tokens so the model learns
        proper document boundaries, which matters for generation quality.
        """
        tok = self.tokenizer
        try:
            if hasattr(tok, "encode"):
                result = tok.encode(text, add_special_tokens=False)
                ids = result.ids if hasattr(result, "ids") else list(result)
            elif hasattr(tok, "tokenize"):
                ids = list(tok.tokenize(text))
            else:
                return []

            # Wrap with BOS / EOS using the actual IDs from the loaded tokenizer
            bos = getattr(tok, 'bos_token_id', 3)   # <s>  = 3 for zero_bpe_10k
            eos = getattr(tok, 'eos_token_id', 4)   # </s> = 4 for zero_bpe_10k
            return [bos] + ids + [eos]

        except Exception as e:
            self.logger.warning("curriculum", f"Tokenization error: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────
    # Checkpointing & logging
    # ─────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag):
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "curriculum_state":     self.state,
        }, path)
        # Also save plain state JSON for easy inspection
        with open(self._state_path, "w") as f:
            json.dump(self.state, f, indent=2)
        self.logger.info("curriculum", f"  Checkpoint saved → {path.name}")

    def _try_resume(self) -> int:
        """Load the latest checkpoint if available. Returns book index to start from."""
        checkpoints = sorted(self.output_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return 0
        latest = checkpoints[-1]
        self.logger.info("curriculum", f"Resuming from {latest.name}")
        ckpt = torch.load(latest, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.state = ckpt.get("curriculum_state", self.state)
        return self.state["books_completed"]

    def _log_result(self, result: Dict[str, Any]):
        """Append one result line to the JSONL log."""
        self.state["book_results"].append(result)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def _write_summary(self):
        """Write a human-readable summary at the end."""
        results = self.state["book_results"]
        if not results:
            return

        valid = [r for r in results if r.get("eval_loss") and not math.isnan(r["eval_loss"])]
        if not valid:
            return

        avg_eval = sum(r["eval_loss"] for r in valid) / len(valid)
        best     = min(valid, key=lambda r: r["eval_loss"])
        worst    = max(valid, key=lambda r: r["eval_loss"])
        total_s  = sum(r.get("elapsed_s", 0) for r in valid)

        summary = {
            "books_trained": len(valid),
            "total_steps": self.state["total_steps"],
            "avg_eval_loss": round(avg_eval, 4),
            "best_book": best["file"],
            "best_eval_loss": best["eval_loss"],
            "worst_book": worst["file"],
            "worst_eval_loss": worst["eval_loss"],
            "total_time_minutes": round(total_s / 60, 1),
        }

        summary_path = self.output_dir / "curriculum_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("curriculum", "── Summary ──────────────────────────────")
        for k, v in summary.items():
            self.logger.info("curriculum", f"  {k}: {v}")

    @staticmethod
    def _empty_result(file_path: Path, revisit_number: int) -> Dict[str, Any]:
        return {
            "file": file_path.name,
            "author": "Unknown",
            "title": file_path.stem,
            "train_loss": float("nan"),
            "eval_loss": float("nan"),
            "n_sequences": 0,
            "n_tokens": 0,
            "elapsed_s": 0,
            "revisits_done": revisit_number,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


__all__ = ["CurriculumTrainer", "CurriculumConfig"]
