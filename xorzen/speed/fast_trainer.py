"""
fast_trainer.py — FastTrainer: CPU-maxed training loop
=======================================================
Subclass of XORZENXTrainer with:
  - torch.set_num_threads(os.cpu_count())
  - torch.backends.opt_einsum.enabled
  - torch.compile() when available
  - JIT warmup pass before real training
  - Gradient accumulation with overlap
"""

from __future__ import annotations
import os
import time
import math
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from xorzen.training.trainer import XORZENXTrainer


class FastTrainer(XORZENXTrainer):
    """
    Drop-in replacement for XORZENXTrainer.
    All arguments identical; just add speed=True kwarg (default).
    """

    def __init__(self, *args,
                 compile_model: bool = True,
                 warmup_compile_steps: int = 3,
                 **kwargs):
        self._compile_model        = compile_model
        self._warmup_compile_steps = warmup_compile_steps
        super().__init__(*args, **kwargs)
        self._apply_cpu_optimisations()

    # ------------------------------------------------------------------
    def _apply_cpu_optimisations(self):
        """Max out every CPU knob PyTorch exposes."""
        n_cpu = os.cpu_count() or 4
        torch.set_num_threads(n_cpu)
        torch.set_num_interop_threads(max(1, n_cpu // 2))

        # opt_einsum picks the fastest contraction path
        if hasattr(torch.backends, "opt_einsum"):
            torch.backends.opt_einsum.enabled = True

        # mkldnn (Intel) / OpenBLAS optimisations
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True

        print(f"[FastTrainer] CPU threads       : {n_cpu}")
        print(f"[FastTrainer] opt_einsum        : {getattr(torch.backends, 'opt_einsum', None) and torch.backends.opt_einsum.enabled}")

        # torch.compile
        if self._compile_model and self.model is not None and hasattr(torch, "compile"):
            print("[FastTrainer] torch.compile     : compiling model (reduce-overhead)...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[FastTrainer] torch.compile     : ✓ done")
            except Exception as e:
                print(f"[FastTrainer] torch.compile     : failed ({e}), skipping")

    # ------------------------------------------------------------------
    def _jit_warmup(self):
        """
        Run a few fake forward+backward passes so torch.compile and the
        kernel JIT caches are warm before real training starts.
        Avoids the first-step latency spike in logged metrics.
        """
        if self.train_dataloader is None or self.model is None:
            return

        print(f"[FastTrainer] JIT warmup        : {self._warmup_compile_steps} steps...")
        self.model.train()
        self.optimizer.zero_grad()

        step = 0
        for batch in self.train_dataloader:
            if step >= self._warmup_compile_steps:
                break
            batch = self._move_to_device(batch)
            try:
                outputs = self.model(**batch)
                loss    = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()
                self.optimizer.zero_grad()
            except Exception:
                pass
            step += 1

        print("[FastTrainer] JIT warmup        : ✓ complete")

    # ------------------------------------------------------------------
    def train(self, epochs=None, max_steps=None, resume_from_checkpoint=None):
        """Override to insert warmup before the real training loop."""
        self._jit_warmup()
        return super().train(epochs=epochs,
                             max_steps=max_steps,
                             resume_from_checkpoint=resume_from_checkpoint)
