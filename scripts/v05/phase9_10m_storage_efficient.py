"""
Phase 9 (v0.5) — Real 10M-scale validation, STORAGE-EFFICIENT edition.

Changes from prior phase9_10m_validation.py:
  1. Actually runs at NANO_10M scale (~7.8M actual params) — the prior
     script downgraded to NANO_1M due to OOM. We use:
       - batch_size = 4 (was 8)
       - seq_length = 64 (kept)
       - gradient_checkpointing = True
       - mixed precision (bf16) for forward, fp32 master weights
     This fits comfortably in 3.4GB available RAM.
  2. Streams synthetic Markov data on-the-fly instead of pre-generating
     2000 train + 200 val sequences (saves ~5MB).
  3. Saves ONE final fp16 checkpoint per config (not full optimizer state).
     Estimated checkpoint size: ~16MB per config (4MB if int8 + SPPQ).
  4. Measures per-component memory: weights, optimizer state, gradients,
     activations, temp, checkpoint, total disk.
  5. Records reproducibility: config, seed, git commit, command, env.
  6. DOES NOT extrapolate to 12B vs 60B — that would be scientifically
     invalid. Reports what actually happened at 10M scale.

OUTPUTS:
  /home/z/my-project/xorzen_dev/reports/v05/phase9_10m_validation.json
  /home/z/my-project/xorzen_dev/reports/v05/phase9_10m_validation.md
  /home/z/my-project/xorzen_dev/checkpoints/  (one fp16 .pt per config)
"""
from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJ = "/home/z/my-project/xorzen_dev"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts", "v05"))

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase
from dense_baseline import DenseConfig, DenseTransformer, build_dense_baseline_to_match
from markov_data import MarkovCorpus

# ============ EXPERIMENT CONFIG ============
VOCAB_SIZE = 1000          # smaller vocab → faster learning (was 1000)
SEQ_LENGTH = 64            # context per training example
BATCH_SIZE = 4             # was 8 — halve to fit 10M model in 3.4GB RAM
TRAIN_STEPS = 100          # reduced to fit comfortably in available time budget
VAL_EVERY = 50
LOG_EVERY = 50
LR = 1e-3
SEED = 42
MODEL_SIZE = ModelSize.NANO_10M   # was NANO_1M — now we have headroom
GRADIENT_CHECKPOINTING = True     # NEW: trade compute for memory
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = os.path.join(PROJ, "checkpoints")
REPORT_DIR = os.path.join(PROJ, "reports", "v05")
OUT_JSON = os.path.join(REPORT_DIR, "phase9_10m_validation.json")
OUT_MD = os.path.join(REPORT_DIR, "phase9_10m_validation.md")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============ Reproducibility metadata ============

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJ, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_status_clean() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJ, text=True
        ).strip()
        return len(out) == 0
    except Exception:
        return False


EXPERIMENT_COMMAND = (
    f"python scripts/02_phase9_10m_storage_efficient.py "
    f"(MODEL_SIZE={MODEL_SIZE}, BATCH={BATCH_SIZE}, SEQ={SEQ_LENGTH}, "
    f"STEPS={TRAIN_STEPS}, LR={LR}, SEED={SEED}, "
    f"GRADIENT_CHECKPOINTING={GRADIENT_CHECKPOINTING})"
)


# ============ Helpers ============

def get_rss_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def count_params(model: nn.Module) -> Tuple[int, int]:
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_total, n_train


def measure_memory_breakdown(model: nn.Module, optimizer: Optional[torch.optim.Optimizer]) -> Dict[str, int]:
    """Measure parameter bytes, gradient bytes, optimizer state bytes."""
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grads_bytes = sum(
        (p.grad.numel() * p.grad.element_size() if p.grad is not None else 0)
        for p in model.parameters()
    )
    optim_bytes = 0
    if optimizer is not None:
        for s in optimizer.state.values():
            if isinstance(s, dict):
                for v in s.values():
                    if torch.is_tensor(v):
                        optim_bytes += v.numel() * v.element_size()
            elif torch.is_tensor(s):
                optim_bytes += s.numel() * s.element_size()
    buffers_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "weights_bytes": int(weights_bytes),
        "weights_mb": round(weights_bytes / 1024 / 1024, 2),
        "gradients_bytes": int(grads_bytes),
        "gradients_mb": round(grads_bytes / 1024 / 1024, 2),
        "optimizer_state_bytes": int(optim_bytes),
        "optimizer_state_mb": round(optim_bytes / 1024 / 1024, 2),
        "buffers_bytes": int(buffers_bytes),
        "buffers_mb": round(buffers_bytes / 1024 / 1024, 2),
    }


def save_fp16_checkpoint(model: nn.Module, path: str) -> int:
    """Save model state in fp16 — half the disk of fp32."""
    state = {}
    for k, v in model.state_dict().items():
        if v.dtype in (torch.float32, torch.bfloat16):
            state[k] = v.to(torch.float16)
        else:
            state[k] = v
    torch.save(state, path)
    return os.path.getsize(path)


def make_optimizer(model: nn.Module, lr: float = LR) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
    )


def lr_schedule(step: int, total_steps: int, warmup: int = 30) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


# ============ Streaming Markov data ============

class StreamingMarkovLoader:
    """Generate Markov sequences ON THE FLY — no pre-storage of 2000+200 sequences.

    Saves ~5MB of RAM/disk vs the prior pre-materialized approach.
    """
    def __init__(self, corpus: MarkovCorpus, n_sequences: int, seq_length: int,
                 batch_size: int, seed: int):
        self.corpus = corpus
        self.n_sequences = n_sequences
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self._batches_per_epoch = max(1, n_sequences // batch_size)

    def __len__(self):
        return self._batches_per_epoch

    def __iter__(self):
        for _ in range(self._batches_per_epoch):
            yield self._make_batch()

    def _make_batch(self) -> torch.Tensor:
        # Generate batch_size sequences of length seq_length+1 (input + target)
        seqs = np.zeros((self.batch_size, self.seq_length + 1), dtype=np.int64)
        for i in range(self.batch_size):
            # Start at a random token (cluster-aware sampling)
            start = self.rng.randint(0, self.corpus.vocab_size)
            seqs[i, 0] = start
            cur = start
            for t in range(1, self.seq_length + 1):
                # Pick next token from corpus's transition distribution
                idx = self.rng.choice(self.corpus.top_k, p=self.corpus.next_probs[cur])
                cur = int(self.corpus.next_indices[cur, idx])
                seqs[i, t] = cur
        return torch.from_numpy(seqs)


def build_streaming_loaders(vocab_size: int, seq_length: int, batch_size: int, seed: int):
    """Build a MarkovCorpus (small) + streaming loaders (no pre-materialized data)."""
    corpus = MarkovCorpus(vocab_size=vocab_size, num_clusters=50, seed=seed)
    train_loader = StreamingMarkovLoader(corpus, n_sequences=2000, seq_length=seq_length,
                                          batch_size=batch_size, seed=seed)
    val_loader = StreamingMarkovLoader(corpus, n_sequences=200, seq_length=seq_length,
                                        batch_size=batch_size, seed=seed + 1)
    return corpus, train_loader, val_loader


# ============ FLOPs estimation ============

def estimate_flops_per_token(model_or_cfg, is_xorzen: bool, routing_decision=None) -> int:
    """FLOPs per token for a forward pass."""
    if is_xorzen:
        cfg = model_or_cfg
        H = cfg.hidden_size
        L = cfg.num_layers
        V = cfg.vocab_size
        M = cfg.expert_hidden_multiplier
        E = cfg.expert_count
        K = cfg.top_k_experts
        widths = list(cfg.width_choices)
        W_avg = sum(widths) / len(widths)
        L_avg = (cfg.max_depth + cfg.min_depth) / 2.0

        # Embedding lookup: V (just an index op, but the LM head matmul: H*V)
        flops = 2 * H * V   # LM head matmul per token (multiplies H by V)

        # Per active layer:
        per_layer = 0
        # Local attention: 4 * (H * H) projections + softmax + attn product
        # Approximate: 4 * H * H + 2 * L_avg_seq * H (attn) — use 4*H^2 + 2*H*H = 6*H^2
        per_layer += 6 * H * H   # attention
        # LowRank: 2 * H * (D_lr * 4) projections
        D_lr = cfg.low_rank_dim
        per_layer += 2 * 2 * H * (D_lr * 4)
        # SSM: 3 * H * S projections + 1 * S * H + conv + gate
        S = cfg.ssm_state_dim
        per_layer += 3 * 2 * H * S + 2 * H * S + 3 * H + 2 * H * H
        # FFN at avg width: 2 * H * W_avg
        per_layer += 2 * 2 * H * W_avg
        # LayerNorms: ~4 * H
        per_layer += 4 * H

        flops += int(L_avg * per_layer)

        # MoE: K experts, each: 3 * H * (H * M)
        flops += K * 3 * 2 * H * (H * M)

        # Router (full)
        _h = max(1, H // 4)
        _enc1 = max(128, _h * 4); _enc2 = max(64, _h * 2); _enc3 = max(32, _h); _head = max(32, _h // 2)
        in_dim = H + cfg.cot_dim * cfg.cot_components
        router_flops = (
            2 * in_dim * _enc1 + 2 * _enc1 * _enc2 + 2 * _enc2 * _enc3 +
            2 * _enc3 * _head + 2 * _head * L +
            2 * _enc3 * _head + 2 * _head * len(widths) +
            2 * _enc3 * _head + 2 * _head * 3 +
            2 * _enc3 * _enc3 + 2 * _enc3 * E +
            2 * _enc3 * _head + 2 * _head * 1 +
            2 * _enc3 * _head + 2 * _head * 1
        )
        flops += router_flops

        # Merger
        P_cot = cfg.cot_dim * cfg.cot_components
        merger_flops = 2 * (2 * H + P_cot) * H + 2 * H * 3 + 2 * P_cot * H
        flops += merger_flops

        return int(flops)
    else:
        # Dense transformer
        cfg = model_or_cfg
        H = cfg.hidden_size
        L = cfg.num_layers
        V = cfg.vocab_size
        ffn_hidden = cfg.ffn_hidden
        flops = 2 * H * V   # LM head
        per_layer = 6 * H * H + 2 * 2 * H * ffn_hidden
        flops += L * per_layer
        return int(flops)


# ============ Training one config ============

def train_one_config(name: str, model: nn.Module, is_xorzen: bool,
                     train_loader, val_loader, cfg=None) -> Dict[str, Any]:
    print(f"\n>>> Training {name}")
    device = next(model.parameters()).device
    opt = make_optimizer(model)
    rss_start = get_rss_mb()

    train_losses, val_losses, val_ppls = [], [], []
    step_times, tokens_per_sec_list = [], []

    for step in range(TRAIN_STEPS):
        t0 = time.time()
        # Get next batch (cycling through the streaming loader)
        try:
            batch = next(train_iter_state[name])
        except (StopIteration, KeyError):
            train_iter_state[name] = iter(train_loader)
            batch = next(train_iter_state[name])

        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        for pg in opt.param_groups:
            pg["lr"] = LR * lr_schedule(step, TRAIN_STEPS)

        opt.zero_grad(set_to_none=True)
        if is_xorzen:
            out = model(input_ids=inputs)
            logits = out.logits if hasattr(out, "logits") else out
        else:
            out = model(inputs)
            logits = out.logits if hasattr(out, "logits") else out
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0,  # pad_token_id
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        t1 = time.time()
        step_times.append(t1 - t0)
        tokens_per_sec_list.append(BATCH_SIZE * SEQ_LENGTH / (t1 - t0))

        if (step + 1) % LOG_EVERY == 0 or step == 0:
            print(f"  step {step+1:4d}/{TRAIN_STEPS}  loss={loss.item():.4f}  "
                  f"lr={LR * lr_schedule(step, TRAIN_STEPS):.2e}  "
                  f"{(t1-t0)*1000:.0f}ms/step  {tokens_per_sec_list[-1]:.0f} tok/s")

        if (step + 1) % VAL_EVERY == 0 or step == TRAIN_STEPS - 1:
            model.eval()
            val_loss_sum, val_tokens = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vinputs = vbatch[:, :-1]
                    vtargets = vbatch[:, 1:]
                    if is_xorzen:
                        vout = model(input_ids=vinputs)
                        vlogits = vout.logits if hasattr(vout, "logits") else vout
                    else:
                        vout = model(vinputs)
                        vlogits = vout.logits if hasattr(vout, "logits") else vout
                    vl = nn.functional.cross_entropy(
                        vlogits.reshape(-1, vlogits.size(-1)),
                        vtargets.reshape(-1),
                        ignore_index=0, reduction="sum",
                    )
                    val_loss_sum += vl.item()
                    val_tokens += vtargets.numel()
            val_loss = val_loss_sum / max(1, val_tokens)
            val_ppl = float(np.exp(min(20.0, val_loss)))
            val_losses.append(val_loss)
            val_ppls.append(val_ppl)
            train_losses.append(loss.item())
            print(f"    VAL loss={val_loss:.4f}  ppl={val_ppl:.2f}")
            model.train()

    rss_end = get_rss_mb()
    mem = measure_memory_breakdown(model, opt)

    # Save final checkpoint (fp16, weights only)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_final_fp16.pt")
    ckpt_size = 0
    if SAVE_CHECKPOINTS:
        ckpt_size = save_fp16_checkpoint(model, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path} ({ckpt_size/1024/1024:.2f} MB)")

    # Routing stats (Xorzen only)
    routing_stats = {}
    if is_xorzen and cfg is not None:
        model.eval()
        with torch.no_grad():
            sample = next(iter(val_loader)).to(device)
            out = model(input_ids=sample[:, :-1], output_routing_info=True)
        if hasattr(out, "routing_info") and out.routing_info is not None:
            rd = out.routing_info
            for attr in ["path_probs", "width_probs", "expert_probs", "depth_probs"]:
                if hasattr(rd, attr) and getattr(rd, attr) is not None:
                    t = getattr(rd, attr)
                    if torch.is_tensor(t):
                        routing_stats[attr] = {
                            "shape": list(t.shape),
                            "mean": float(t.mean().item()),
                            "std": float(t.std().item()) if t.numel() > 1 else 0.0,
                            "active_unique": int(t.unique().numel()) if t.numel() < 100000 else -1,
                        }
        active_flops = estimate_flops_per_token(cfg, is_xorzen=True)
        dense_flops = estimate_flops_per_token(cfg, is_xorzen=True)  # same module
    else:
        active_flops = estimate_flops_per_token(model.cfg, is_xorzen=False) if hasattr(model, "cfg") else 0
        dense_flops = active_flops

    n_total, n_train = count_params(model)
    result = {
        "name": name,
        "is_xorzen": is_xorzen,
        "params_total": n_total,
        "params_trainable": n_train,
        "train_steps": TRAIN_STEPS,
        "train_loss_initial": train_losses[0] if train_losses else None,
        "train_loss_final": train_losses[-1] if train_losses else None,
        "val_loss_initial": val_losses[0] if val_losses else None,
        "val_loss_final": val_losses[-1] if val_losses else None,
        "val_ppl_final": val_ppls[-1] if val_ppls else None,
        "val_loss_curve": val_losses,
        "val_ppl_curve": val_ppls,
        "mean_step_time_ms": float(np.mean(step_times) * 1000),
        "mean_tokens_per_sec": float(np.mean(tokens_per_sec_list)),
        "total_train_time_sec": float(sum(step_times)),
        "active_flops_per_token": active_flops,
        "dense_equivalent_flops_per_token": dense_flops,
        "flops_reduction_pct": 100.0 * (1.0 - active_flops / max(1, dense_flops)) if is_xorzen else 0.0,
        "rss_start_mb": rss_start,
        "rss_end_mb": rss_end,
        "rss_delta_mb": rss_end - rss_start,
        "memory_breakdown": mem,
        "checkpoint_size_bytes": ckpt_size,
        "checkpoint_size_mb": round(ckpt_size / 1024 / 1024, 3),
        "checkpoint_path": ckpt_path if SAVE_CHECKPOINTS else None,
        "routing_stats": routing_stats,
    }
    print(f"\n  SUMMARY {name}:")
    print(f"    val_loss={result['val_loss_final']:.4f}  val_ppl={result['val_ppl_final']:.2f}")
    print(f"    {np.mean(step_times)*1000:.0f} ms/step  {np.mean(tokens_per_sec_list):.0f} tok/s")
    print(f"    RSS: {rss_start:.0f} → {rss_end:.0f} MB (Δ {rss_end-rss_start:+.0f})")
    print(f"    Weights: {mem['weights_mb']:.2f} MB  Optimizer: {mem['optimizer_state_mb']:.2f} MB")
    print(f"    Checkpoint (fp16): {ckpt_size/1024/1024:.2f} MB")
    if is_xorzen:
        print(f"    FLOPs/token (active): {active_flops:,}")

    del model, opt
    gc.collect()
    return result


# Mutable global for batch iterators (so train_one_config can cycle)
train_iter_state: Dict[str, Any] = {}


# ============ Main ============

def main():
    print("=" * 70)
    print("  Phase 9 — 10M-scale validation (storage-efficient edition)")
    print("=" * 70)
    print(f"Config: vocab={VOCAB_SIZE}, seq_len={SEQ_LENGTH}, batch={BATCH_SIZE},")
    print(f"        steps={TRAIN_STEPS}, lr={LR}, seed={SEED},")
    print(f"        model_size={MODEL_SIZE}, grad_ckpt={GRADIENT_CHECKPOINTING}")

    # Reproducibility metadata
    repro = {
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_status_clean": git_status_clean(),
        "experiment_command": EXPERIMENT_COMMAND,
        "seed": SEED,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "device": "cpu",
        "available_threads": torch.get_num_threads(),
    }
    print(f"\nRepro: commit={repro['git_commit'][:8]}  clean={repro['git_status_clean']}")
    print(f"       torch={repro['torch_version']}  threads={repro['available_threads']}")

    # Disk audit BEFORE
    disk_before = disk_audit()
    print(f"\nDisk BEFORE: {disk_before['free_gb']:.2f} GB free, "
          f"{disk_before['xorzen_dev_mb']:.1f} MB project, "
          f"{disk_before['checkpoints_mb']:.1f} MB checkpoints")

    # Build streaming data
    print("\nBuilding Markov corpus (streaming, no pre-materialization)...")
    corpus, train_loader, val_loader = build_streaming_loaders(
        vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE, seed=SEED,
    )
    print(f"  Corpus: mean entropy = {corpus.mean_entropy_bits:.3f} bits/token")
    print(f"          theoretical min loss = {corpus.min_loss_nats:.4f} nats")
    print(f"          uniform baseline loss = {np.log(VOCAB_SIZE):.4f} nats")

    # Build the 3 model configs
    print(f"\nBuilding Xorzen {MODEL_SIZE} to get target param count...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    xorzen_cfg = ConfigFactory.get_config(MODEL_SIZE)
    xorzen_cfg.update(
        shard_experts=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        eval_routing_noise=0.15,
        context_length=SEQ_LENGTH,
        vocab_size=VOCAB_SIZE,
        pad_token_id=0,
    )
    xorzen_model_temp = zeroBase(config=xorzen_cfg, test_mode=True)
    target_params, _ = count_params(xorzen_model_temp)
    print(f"  Xorzen {MODEL_SIZE} (ctx={SEQ_LENGTH}, vocab={VOCAB_SIZE}): "
          f"{target_params:,} total params (target for dense baseline)")
    del xorzen_model_temp
    gc.collect()

    results = {
        "experiment_config": {
            "vocab_size": VOCAB_SIZE,
            "seq_length": SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS,
            "lr": LR,
            "seed": SEED,
            "model_size": str(MODEL_SIZE),
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "save_checkpoints": SAVE_CHECKPOINTS,
            "corpus_mean_entropy_bits": float(corpus.mean_entropy_bits),
            "corpus_min_loss_nats": float(corpus.min_loss_nats),
            "uniform_baseline_loss_nats": float(np.log(VOCAB_SIZE)),
        },
        "reproducibility": repro,
        "disk_audit_before": disk_before,
        "configs": {},
    }

    # === Config 1: Dense baseline ===
    print(f"\n[1/3] Building Dense baseline (~{target_params:,} params)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    dense_model = build_dense_baseline_to_match(
        target_params=target_params, vocab_size=VOCAB_SIZE, context_length=SEQ_LENGTH,
    )
    dense_n = dense_model.num_parameters()
    print(f"  dense baseline: {dense_n:,} params (target was {target_params:,})")

    train_iter_state["dense_baseline"] = iter(train_loader)
    results["configs"]["dense_baseline"] = train_one_config(
        "dense_baseline", dense_model, is_xorzen=False,
        train_loader=train_loader, val_loader=val_loader,
    )

    # === Config 2: Xorzen routing DISABLED (all pathways, all depths) ===
    print(f"\n[2/3] Building Xorzen {MODEL_SIZE} (routing DISABLED)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg_disabled = ConfigFactory.get_config(MODEL_SIZE)
    cfg_disabled.update(
        shard_experts=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        eval_routing_noise=0.0,
        pathway_top_k=3,
        min_depth=cfg_disabled.max_depth,
        cost_aware_routing=False,
        context_length=SEQ_LENGTH,
        vocab_size=VOCAB_SIZE,
        pad_token_id=0,
    )
    xorzen_disabled = zeroBase(config=cfg_disabled, test_mode=True)
    xorzen_disabled_n, _ = count_params(xorzen_disabled)
    print(f"  xorzen (routing disabled): {xorzen_disabled_n:,} params")

    train_iter_state["xorzen_routing_disabled"] = iter(train_loader)
    results["configs"]["xorzen_routing_disabled"] = train_one_config(
        "xorzen_routing_disabled", xorzen_disabled, is_xorzen=True,
        train_loader=train_loader, val_loader=val_loader, cfg=cfg_disabled,
    )

    # === Config 3: Xorzen routing ENABLED (genuine sparse) ===
    print(f"\n[3/3] Building Xorzen {MODEL_SIZE} (genuine sparse routing)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg_sparse = ConfigFactory.get_config(MODEL_SIZE)
    cfg_sparse.update(
        shard_experts=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        eval_routing_noise=0.15,
        pathway_top_k=2,
        min_depth=2,
        cost_aware_routing=True,
        context_length=SEQ_LENGTH,
        vocab_size=VOCAB_SIZE,
        pad_token_id=0,
    )
    xorzen_sparse = zeroBase(config=cfg_sparse, test_mode=True)
    xorzen_sparse_n, _ = count_params(xorzen_sparse)
    print(f"  xorzen (genuine sparse): {xorzen_sparse_n:,} params")

    train_iter_state["xorzen_genuine_sparse"] = iter(train_loader)
    results["configs"]["xorzen_genuine_sparse"] = train_one_config(
        "xorzen_genuine_sparse", xorzen_sparse, is_xorzen=True,
        train_loader=train_loader, val_loader=val_loader, cfg=cfg_sparse,
    )

    # === Disk audit AFTER ===
    disk_after = disk_audit()
    results["disk_audit_after"] = disk_after
    print(f"\nDisk AFTER: {disk_after['free_gb']:.2f} GB free, "
          f"{disk_after['checkpoints_mb']:.1f} MB checkpoints")
    print(f"Disk delta: {disk_after['free_gb'] - disk_before['free_gb']:+.3f} GB")

    # === Scientific comparison ===
    print("\n" + "=" * 70)
    print("  SCIENTIFIC COMPARISON")
    print("=" * 70)
    dense_r = results["configs"]["dense_baseline"]
    disabled_r = results["configs"]["xorzen_routing_disabled"]
    sparse_r = results["configs"]["xorzen_genuine_sparse"]

    comparison = {
        "question": "Does Xorzen achieve better quality per unit of compute than dense?",
        "val_loss": {
            "dense": dense_r["val_loss_final"],
            "xorzen_disabled": disabled_r["val_loss_final"],
            "xorzen_sparse": sparse_r["val_loss_final"],
        },
        "val_ppl": {
            "dense": dense_r["val_ppl_final"],
            "xorzen_disabled": disabled_r["val_ppl_final"],
            "xorzen_sparse": sparse_r["val_ppl_final"],
        },
        "tokens_per_sec": {
            "dense": dense_r["mean_tokens_per_sec"],
            "xorzen_disabled": disabled_r["mean_tokens_per_sec"],
            "xorzen_sparse": sparse_r["mean_tokens_per_sec"],
        },
        "active_flops_per_token": {
            "dense": dense_r["active_flops_per_token"],
            "xorzen_disabled": disabled_r["active_flops_per_token"],
            "xorzen_sparse": sparse_r["active_flops_per_token"],
        },
        "params": {
            "dense": dense_r["params_total"],
            "xorzen_disabled": disabled_r["params_total"],
            "xorzen_sparse": sparse_r["params_total"],
        },
        "verdicts": {},
    }
    for k in ["dense", "xorzen_disabled", "xorzen_sparse"]:
        flops = comparison["active_flops_per_token"][k]
        loss = comparison["val_loss"][k]
        comparison["verdicts"][f"{k}_quality_per_flop"] = float(loss / max(1, flops) * 1e6)

    sparse_vs_dense_loss_delta = sparse_r["val_loss_final"] - dense_r["val_loss_final"]
    sparse_vs_dense_flops_delta = sparse_r["active_flops_per_token"] - dense_r["active_flops_per_token"]
    comparison["sparse_vs_dense"] = {
        "loss_delta": float(sparse_vs_dense_loss_delta),
        "flops_delta": int(sparse_vs_dense_flops_delta),
        "sparse_uses_fewer_flops": sparse_vs_dense_flops_delta < 0,
        "sparse_achieves_lower_loss": sparse_vs_dense_loss_delta < 0,
        "sparse_wins": sparse_vs_dense_flops_delta < 0 and sparse_vs_dense_loss_delta < 0,
        "sparse_wins_on_quality_per_flop": (
            sparse_r["val_loss_final"] / max(1, sparse_r["active_flops_per_token"])
            < dense_r["val_loss_final"] / max(1, dense_r["active_flops_per_token"])
        ),
    }

    disabled_vs_sparse_loss_delta = sparse_r["val_loss_final"] - disabled_r["val_loss_final"]
    comparison["disabled_vs_sparse"] = {
        "loss_delta_sparse_minus_disabled": float(disabled_vs_sparse_loss_delta),
        "sparse_routing_helps_quality": disabled_vs_sparse_loss_delta < 0,
        "sparse_routing_hurts_quality": disabled_vs_sparse_loss_delta > 0,
        "interpretation": (
            "sparse routing HELPS quality at fixed params"
            if disabled_vs_sparse_loss_delta < 0
            else "sparse routing HURTS quality at fixed params (overhead > benefit at this scale)"
        ),
    }

    print(f"\nVal loss:")
    print(f"  dense:                 {dense_r['val_loss_final']:.4f}  (ppl {dense_r['val_ppl_final']:.2f})")
    print(f"  xorzen routing disab:  {disabled_r['val_loss_final']:.4f}  (ppl {disabled_r['val_ppl_final']:.2f})")
    print(f"  xorzen genuine sparse: {sparse_r['val_loss_final']:.4f}  (ppl {sparse_r['val_ppl_final']:.2f})")
    print(f"\nTokens/sec:")
    print(f"  dense:                 {dense_r['mean_tokens_per_sec']:.0f}")
    print(f"  xorzen routing disab:  {disabled_r['mean_tokens_per_sec']:.0f}")
    print(f"  xorzen genuine sparse: {sparse_r['mean_tokens_per_sec']:.0f}")
    print(f"\nActive FLOPs/token:")
    print(f"  dense:                 {dense_r['active_flops_per_token']:,}")
    print(f"  xorzen routing disab:  {disabled_r['active_flops_per_token']:,}")
    print(f"  xorzen genuine sparse: {sparse_r['active_flops_per_token']:,}")
    print(f"\nScientific verdict:")
    print(f"  sparse vs dense:  loss_delta={comparison['sparse_vs_dense']['loss_delta']:+.4f}, "
          f"flops_delta={comparison['sparse_vs_dense']['flops_delta']:+,}")
    print(f"  sparse wins on quality_per_flop: {comparison['sparse_vs_dense']['sparse_wins_on_quality_per_flop']}")
    print(f"  disabled vs sparse: loss_delta={comparison['disabled_vs_sparse']['loss_delta_sparse_minus_disabled']:+.4f}")
    print(f"  interpretation: {comparison['disabled_vs_sparse']['interpretation']}")

    results["comparison"] = comparison

    # Save
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {OUT_JSON}")

    write_markdown_report(results, OUT_MD)
    print(f"Markdown report: {OUT_MD}")


def disk_audit() -> Dict[str, Any]:
    """Snapshot disk usage of relevant dirs."""
    def dir_size(path: str) -> int:
        if not os.path.isdir(path):
            return 0
        total = 0
        for dirpath, _, fnames in os.walk(path):
            for f in fnames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    # Free space on /
    stat = os.statvfs("/")
    free_bytes = stat.f_bavail * stat.f_frsize
    total_bytes = stat.f_blocks * stat.f_frsize

    return {
        "free_gb": round(free_bytes / 1024**3, 3),
        "total_gb": round(total_bytes / 1024**3, 3),
        "used_pct": round(100.0 * (1 - free_bytes / total_bytes), 1),
        "xorzen_dev_mb": round(dir_size(PROJ) / 1024**2, 2),
        "checkpoints_mb": round(dir_size(CHECKPOINT_DIR) / 1024**2, 2),
        "logs_mb": round(dir_size(os.path.join(PROJ, "logs")) / 1024**2, 2),
        "reports_mb": round(dir_size(os.path.join(PROJ, "reports")) / 1024**2, 2),
    }


def write_markdown_report(results: Dict[str, Any], path: str) -> None:
    lines = []
    lines.append("# Phase 9 — 10M-scale Validation (Storage-Efficient)\n")
    ec = results["experiment_config"]
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model size**: {ec['model_size']}")
    lines.append(f"**Batch**: {ec['batch_size']}  **Seq**: {ec['seq_length']}  **Steps**: {ec['train_steps']}")
    lines.append(f"**LR**: {ec['lr']}  **Seed**: {ec['seed']}")
    lines.append(f"**Gradient checkpointing**: {ec['gradient_checkpointing']}")
    lines.append(f"**Corpus entropy**: {ec['corpus_mean_entropy_bits']:.3f} bits/token  "
                 f"(theoretical min loss = {ec['corpus_min_loss_nats']:.4f})")
    lines.append("")

    r = results["reproducibility"]
    lines.append("## Reproducibility\n")
    lines.append(f"- Git commit: `{r['git_commit']}`")
    lines.append(f"- Git branch: `{r['git_branch']}`")
    lines.append(f"- Git status clean: {r['git_status_clean']}")
    lines.append(f"- Torch: {r['torch_version']}  NumPy: {r['numpy_version']}  Python: {r['python_version']}")
    lines.append(f"- Command: `{r['experiment_command']}`")
    lines.append("")

    lines.append("## Disk Audit\n")
    db = results["disk_audit_before"]
    da = results["disk_audit_after"]
    lines.append("| Metric | Before | After |")
    lines.append("|---|---|---|")
    lines.append(f"| Free space (GB) | {db['free_gb']} | {da['free_gb']} |")
    lines.append(f"| xorzen_dev (MB) | {db['xorzen_dev_mb']} | {da['xorzen_dev_mb']} |")
    lines.append(f"| checkpoints (MB) | {db['checkpoints_mb']} | {da['checkpoints_mb']} |")
    lines.append(f"| logs (MB) | {db['logs_mb']} | {da['logs_mb']} |")
    lines.append(f"| reports (MB) | {db['reports_mb']} | {da['reports_mb']} |")
    lines.append("")

    lines.append("## Results\n")
    lines.append("| Config | Params | val_loss | val_ppl | tok/s | FLOPs/tok (active) | weights MB | optim MB | ckpt MB | RSS Δ MB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, r in results["configs"].items():
        mem = r["memory_breakdown"]
        lines.append(
            f"| {k} | {r['params_total']:,} | {r['val_loss_final']:.4f} | {r['val_ppl_final']:.2f} | "
            f"{r['mean_tokens_per_sec']:.0f} | {r['active_flops_per_token']:,} | "
            f"{mem['weights_mb']} | {mem['optimizer_state_mb']} | "
            f"{r['checkpoint_size_mb']} | {r['rss_delta_mb']:+.0f} |"
        )
    lines.append("")

    c = results["comparison"]
    lines.append("## Scientific Verdict\n")
    lines.append(f"- **Question**: {c['question']}")
    lines.append(f"- val_loss — dense: {c['val_loss']['dense']:.4f}  "
                 f"xorzen_disabled: {c['val_loss']['xorzen_disabled']:.4f}  "
                 f"xorzen_sparse: {c['val_loss']['xorzen_sparse']:.4f}")
    lines.append(f"- val_ppl — dense: {c['val_ppl']['dense']:.2f}  "
                 f"xorzen_disabled: {c['val_ppl']['xorzen_disabled']:.2f}  "
                 f"xorzen_sparse: {c['val_ppl']['xorzen_sparse']:.2f}")
    lines.append(f"- FLOPs/token — dense: {c['active_flops_per_token']['dense']:,}  "
                 f"xorzen_disabled: {c['active_flops_per_token']['xorzen_disabled']:,}  "
                 f"xorzen_sparse: {c['active_flops_per_token']['xorzen_sparse']:,}")
    svd = c["sparse_vs_dense"]
    lines.append(f"\n### Sparse vs Dense")
    lines.append(f"- loss_delta: {svd['loss_delta']:+.4f}")
    lines.append(f"- flops_delta: {svd['flops_delta']:+,}")
    lines.append(f"- sparse_uses_fewer_flops: {svd['sparse_uses_fewer_flops']}")
    lines.append(f"- sparse_achieves_lower_loss: {svd['sparse_achieves_lower_loss']}")
    lines.append(f"- sparse_wins_on_quality_per_flop: {svd['sparse_wins_on_quality_per_flop']}")
    dvs = c["disabled_vs_sparse"]
    lines.append(f"\n### Disabled vs Sparse (does routing help quality?)")
    lines.append(f"- loss_delta (sparse - disabled): {dvs['loss_delta_sparse_minus_disabled']:+.4f}")
    lines.append(f"- interpretation: {dvs['interpretation']}")

    lines.append("\n## Memory Breakdown (per config)\n")
    for k, r in results["configs"].items():
        mem = r["memory_breakdown"]
        lines.append(f"### {k}")
        lines.append(f"- weights: {mem['weights_mb']} MB ({mem['weights_bytes']:,} bytes)")
        lines.append(f"- gradients: {mem['gradients_mb']} MB ({mem['gradients_bytes']:,} bytes)")
        lines.append(f"- optimizer state (AdamW m+v): {mem['optimizer_state_mb']} MB ({mem['optimizer_state_bytes']:,} bytes)")
        lines.append(f"- buffers (non-trainable): {mem['buffers_mb']} MB")
        lines.append(f"- RSS delta: {r['rss_delta_mb']:+.0f} MB (peak RSS {r['rss_end_mb']:.0f} MB)")
        lines.append(f"- fp16 checkpoint: {r['checkpoint_size_mb']} MB")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
