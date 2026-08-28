#!/usr/bin/env python3
"""
Phase 3: Train -> Observe -> Diagnose -> Fix -> Re-test
Focused on conditional compute quality.

All results saved to /home/z/my-project/DevNet/reports/v04/phase3_results.json
"""

import sys, os, json, time, math, copy, traceback
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

PROJECT_ROOT = Path("/home/z/my-project/DevNet")
sys.path.insert(0, str(PROJECT_ROOT))

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.model import zeroModel
from xorzen.model.base import ModelOutput
from xorzen.model.components.routing import RoutingDecision, AdaptiveRouter
from xorzen.utils.math_utils import count_parameters

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
TRAINING_STEPS = 300
BATCH_SIZE = 2
SEQ_LENGTH = 64
LEARNING_RATE = 1e-3
LOG_INTERVAL = 50

OUTPUT_DIR = PROJECT_ROOT / "reports" / "v04"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "phase3_results.json"

# =============================================================================
# SYNTHETIC DATASET WITH ADVERSARIAL FIXTURES
# =============================================================================

class SyntheticLMDataset(Dataset):
    """Synthetic LM data with 5 distinct fixture types."""
    FIXTURE_NAMES = [
        "uniform_random", "repetitive", "structured",
        "bimodal", "increasing"
    ]

    def __init__(self, vocab_size, seq_length, num_samples, seed=42):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)
        self.fixture_names = self.FIXTURE_NAMES
        self.data = self._generate_all()

    def _generate_all(self):
        data = []
        per_fixture = self.num_samples // len(self.fixture_names)
        for fixture in self.fixture_names:
            for _ in range(per_fixture):
                data.append(self._gen(fixture))
        while len(data) < self.num_samples:
            data.append(self._gen("uniform_random"))
        return data

    def _gen(self, ft):
        vs, sl = self.vocab_size, self.seq_length
        if ft == "uniform_random":
            return self.rng.randint(1, vs, size=sl).astype(np.int64)
        elif ft == "repetitive":
            return np.full(sl, self.rng.randint(1, vs), dtype=np.int64)
        elif ft == "structured":
            return (np.arange(sl) % vs + 1).astype(np.int64)
        elif ft == "bimodal":
            t = np.empty(sl, dtype=np.int64)
            for i in range(sl):
                if i % 2 == 0:
                    t[i] = self.rng.randint(1, max(2, vs // 4))
                else:
                    t[i] = self.rng.randint(vs // 2, vs)
            return t
        elif ft == "increasing":
            base = self.rng.randint(1, max(1, vs - sl))
            return (np.arange(base, min(base + sl, vs)) % vs + 1).astype(np.int64)
        return self.rng.randint(1, vs, size=sl).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.data[idx]).long()
        return {"input_ids": tokens, "labels": tokens.clone()}


def create_adversarial_batch(vocab_size, seq_length, device):
    """Create batch with 5 deliberately different input characteristics."""
    fixtures = {
        "uniform_random": torch.randint(1, vocab_size, (1, seq_length)),
        "repetitive": torch.full((1, seq_length), 42, dtype=torch.long),
        "structured": (torch.arange(seq_length) % vocab_size + 1).unsqueeze(0),
        "bimodal": torch.cat([
            torch.randint(1, max(2, vocab_size // 4), (1, seq_length // 2)),
            torch.randint(vocab_size // 2, vocab_size, (1, seq_length - seq_length // 2)),
        ], dim=1),
        "increasing": (torch.arange(100, 100 + seq_length) % vocab_size).unsqueeze(0).clamp(min=1),
    }
    input_ids = torch.cat(list(fixtures.values()), dim=0).to(device)
    return input_ids, list(fixtures.keys())


# =============================================================================
# ROUTING STATISTICS COLLECTOR
# =============================================================================

@dataclass
class RoutingStats:
    depth_mask_sum: float = 0.0
    depth_mask_per_layer: List[float] = field(default_factory=list)
    depth_entropy: float = 0.0
    depth_unique_patterns: int = 0
    width_idx_distribution: Dict[int, int] = field(default_factory=dict)
    width_entropy: float = 0.0
    width_mean_multiplier: float = 0.0
    path_probs_mean: List[float] = field(default_factory=list)
    path_entropy: float = 0.0
    path_top1_distribution: Dict[int, int] = field(default_factory=dict)
    expert_probs_mean: List[float] = field(default_factory=list)
    expert_entropy: float = 0.0
    expert_top1_distribution: Dict[int, int] = field(default_factory=dict)
    complexity_mean: float = 0.0
    complexity_std: float = 0.0
    uncertainty_mean: float = 0.0
    estimated_active_params: int = 0
    heuristic_active_params: int = 0
    per_token_compute_mean: float = 0.0
    per_token_compute_std: float = 0.0
    per_token_compute_cv: float = 0.0
    lm_loss: float = 0.0
    routing_loss: float = 0.0
    total_loss: float = 0.0
    forward_time_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


def collect_routing_stats(model, input_ids, fixture_names=None, tag=""):
    """Run one forward pass and collect comprehensive routing statistics."""
    stats = RoutingStats()
    device = input_ids.device
    labels = input_ids.clone()

    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(input_ids=input_ids, labels=labels, output_routing_info=True, return_dict=True)
        t1 = time.perf_counter()

    stats.forward_time_ms = (t1 - t0) * 1000
    rd = outputs.routing_info
    shape = input_ids.shape
    B = shape[0]
    T = shape[1]

    # Depth
    depth_mask = rd.depth_mask  # [B, T, max_depth]
    stats.depth_mask_sum = depth_mask.sum(dim=-1).float().mean().item()
    for li in range(depth_mask.shape[-1]):
        stats.depth_mask_per_layer.append(depth_mask[:, :, li].float().mean().item())
    flat_mask = depth_mask.reshape(B * T, -1).round().int()
    stats.depth_unique_patterns = len(set(tuple(r.tolist()) for r in flat_mask))
    avg_la = depth_mask.float().mean(dim=[0, 1])
    avg_la = avg_la / (avg_la.sum() + 1e-10)
    stats.depth_entropy = -((avg_la + 1e-10) * torch.log(avg_la + 1e-10)).sum().item()

    # Width
    width_idx = rd.width_idx  # [B, T]
    for w in width_idx.flatten().unique():
        stats.width_idx_distribution[int(w.item())] = int((width_idx == w).sum().item())
    stats.width_mean_multiplier = rd.width_multiplier.float().mean().item()
    wp = rd.width_probs.float().mean(dim=[0, 1])
    wp = wp / (wp.sum() + 1e-10)
    stats.width_entropy = -((wp + 1e-10) * torch.log(wp + 1e-10)).sum().item()

    # Pathway
    pp = rd.path_probs.float().mean(dim=[0, 1])
    stats.path_probs_mean = pp.tolist()
    pp_n = pp / (pp.sum() + 1e-10)
    stats.path_entropy = -((pp_n + 1e-10) * torch.log(pp_n + 1e-10)).sum().item()
    pt1 = rd.path_probs.argmax(dim=-1)
    for p in pt1.flatten().unique():
        stats.path_top1_distribution[int(p.item())] = int((pt1 == p).sum().item())

    # Expert
    ep = rd.expert_probs.float().mean(dim=[0, 1])
    stats.expert_probs_mean = ep.tolist()
    ep_n = ep / (ep.sum() + 1e-10)
    stats.expert_entropy = -((ep_n + 1e-10) * torch.log(ep_n + 1e-10)).sum().item()
    et1 = rd.expert_probs.argmax(dim=-1)
    for e in et1.flatten().unique():
        stats.expert_top1_distribution[int(e.item())] = int((et1 == e).sum().item())

    # Complexity
    c = rd.complexity
    stats.complexity_mean = c.float().mean().item()
    stats.complexity_std = c.float().std().item()
    stats.uncertainty_mean = rd.uncertainty.float().mean().item()

    # Active params
    stats.estimated_active_params = outputs.active_params
    stats.heuristic_active_params = int(model.config.estimate_active_parameters())

    # Per-token compute distribution (proxy)
    df = depth_mask.float().sum(dim=-1)
    wm = rd.width_multiplier.float().squeeze(-1)
    nap = (rd.path_probs > (1.0 / 3.0)).float().sum(dim=-1)
    ptc = df * wm * (1.0 + nap / 3.0)
    stats.per_token_compute_mean = ptc.mean().item()
    stats.per_token_compute_std = ptc.std().item()
    stats.per_token_compute_cv = stats.per_token_compute_std / (stats.per_token_compute_mean + 1e-10)

    # Losses
    stats.lm_loss = outputs.lm_loss.item() if outputs.lm_loss is not None else 0.0
    stats.routing_loss = outputs.routing_loss.item() if outputs.routing_loss is not None else 0.0
    stats.total_loss = outputs.loss.item() if outputs.loss is not None else 0.0

    return stats


def collect_per_fixture_stats(model, input_ids, fixture_names, device):
    """Collect routing stats PER FIXTURE."""
    results = {}
    model.eval()
    for i, fname in enumerate(fixture_names):
        single = input_ids[i:i+1]
        labels = single.clone()
        with torch.no_grad():
            out = model(input_ids=single, labels=labels, output_routing_info=True, return_dict=True)
        rd = out.routing_info
        results[fname] = {
            "fixture": fname,
            "complexity_mean": rd.complexity.float().mean().item(),
            "complexity_std": rd.complexity.float().std().item(),
            "depth_mask_sum": rd.depth_mask.float().sum(dim=-1).mean().item(),
            "width_mean_multiplier": rd.width_multiplier.float().mean().item(),
            "path_probs": rd.path_probs.float().mean(dim=1).squeeze(0).tolist(),
            "expert_top1": rd.expert_indices[:, :, 0].squeeze(0).tolist(),
        }
    return results


# =============================================================================
# DIAGNOSTIC: CORRECT ACTIVE PARAMETER ACCOUNTING
# =============================================================================

def compute_correct_active_params(model, routing_decision):
    """Compute actively executed parameters from the forward graph."""
    c = model.config
    rd = routing_decision
    B = rd.depth_mask.shape[0]
    T = rd.depth_mask.shape[1]
    r = {}

    # Always-on components
    r["embeddings"] = c.vocab_size * c.hidden_size + c.context_length * c.hidden_size
    r["router"] = count_parameters(model.router)
    r["cot"] = count_parameters(model.cot)
    r["merger"] = count_parameters(model.merger)
    r["lm_head"] = c.hidden_size * c.vocab_size

    # HASS blocks
    total_hass = sum(count_parameters(b) for b in model.blocks)
    r["hass_total"] = total_hass

    # Expert params
    inter_dim = int(c.hidden_size * c.expert_hidden_multiplier)
    params_per_expert = c.hidden_size * inter_dim * 2
    r["params_per_expert"] = params_per_expert
    r["active_expert_params"] = c.top_k_experts * params_per_expert
    r["total_expert_params"] = c.expert_count * params_per_expert

    # Training: ALL HASS blocks execute (STE blend)
    training_active = (
        r["embeddings"] + r["router"] + r["cot"] + r["merger"] +
        total_hass + c.top_k_experts * params_per_expert + r["lm_head"]
    )
    r["training_active_params"] = training_active

    # Inference: only layers with active tokens
    layer_activity = rd.depth_mask.float().mean().item()
    pp = rd.path_probs
    path_usage = [(pp.argmax(dim=-1) == p).float().mean().item() for p in range(3)]
    pathway_sparsity = 1.0 - max(path_usage) * 0.3
    inference_active = (
        r["embeddings"] + r["router"] + r["cot"] + r["merger"] +
        total_hass * layer_activity * pathway_sparsity +
        c.top_k_experts * params_per_expert + r["lm_head"]
    )
    r["inference_active_params_estimate"] = int(inference_active)
    r["inference_layer_activity"] = layer_activity
    r["pathway_usage"] = path_usage

    # Heuristic vs model runtime
    r["heuristic_estimate"] = int(c.estimate_active_parameters())
    r["model_runtime_estimate"] = model._estimate_active_params(rd)

    total_trainable = count_parameters(model, only_trainable=True)
    total_all = count_parameters(model)
    r["total_trainable"] = total_trainable
    r["total_all"] = total_all
    r["training_active_pct"] = 100.0 * training_active / max(1, total_trainable)
    r["heuristic_pct"] = 100.0 * r["heuristic_estimate"] / max(1, total_trainable)
    r["model_estimate_pct"] = 100.0 * r["model_runtime_estimate"] / max(1, total_trainable)

    return r


# =============================================================================
# DIAGNOSTIC: COMPLEXITY BIAS ANALYSIS
# =============================================================================

def analyze_complexity_bias(model, input_ids):
    """Measure complexity bias magnitude vs learned routing logits."""
    model.eval()
    device = input_ids.device
    B, T = input_ids.shape[0], input_ids.shape[1]
    cot_dim = model.config.cot_dim * model.config.cot_components

    with torch.no_grad():
        cot_f = torch.zeros(B, T, cot_dim, device=device)
        ri = torch.cat([model.token_embedding(input_ids), cot_f], dim=-1)
        feats = model.router.feature_encoder(ri.view(B * T, -1)).view(B, T, -1)
        dl_raw = model.router.depth_router(feats)
        wl_raw = model.router.width_router(feats)
        pl_raw = model.router.path_router(feats)
        el_raw = model.router.expert_router(feats)
        complexity = model.router.complexity_estimator(feats)

    L = model.config.max_depth
    W = model.router.num_widths
    P = 3

    d_bias = complexity.squeeze(-1).unsqueeze(-1) * torch.linspace(0, 1, L, device=device).unsqueeze(0).unsqueeze(0) * 2.0
    w_bias = complexity.squeeze(-1).unsqueeze(-1) * torch.linspace(-1, 1, W, device=device).unsqueeze(0).unsqueeze(0) * 3.0

    ca = bool(getattr(model.config, 'cost_aware_routing', True))
    ca_bias = {}
    if ca:
        budget = float(getattr(model.config, 'compute_budget', 1.0))
        sp = 1.0 - max(0.05, min(1.0, budget))
        dlb = torch.linspace(0, -sp * 3.0, L, device=device)
        ds = -sp * 4.0 * (1.0 - complexity.squeeze(-1))
        ca_bias["depth_layer_bias_max"] = dlb.abs().max().item()
        ca_bias["depth_shift_mean"] = ds.mean().item()
        ca_bias["depth_shift_std"] = ds.std().item()
        ca_bias["width_bias_axis"] = torch.linspace(sp * 2.0, -sp * 2.0, W, device=device).tolist()
        ca_bias["path_bias_axis"] = torch.linspace(sp * 1.5, -sp * 0.5, P, device=device).tolist()

    return {
        "complexity_mean": complexity.float().mean().item(),
        "complexity_std": complexity.float().std().item(),
        "complexity_min": complexity.float().min().item(),
        "complexity_max": complexity.float().max().item(),
        "depth_logits_raw_mean": dl_raw.float().mean().item(),
        "depth_logits_raw_std": dl_raw.float().std().item(),
        "depth_logits_raw_absmax": dl_raw.float().abs().max().item(),
        "depth_bias_magnitude_mean": d_bias.abs().mean().item(),
        "depth_bias_magnitude_max": d_bias.abs().max().item(),
        "depth_bias_to_logit_ratio": d_bias.abs().mean().item() / (dl_raw.float().abs().mean().item() + 1e-10),
        "width_logits_raw_mean": wl_raw.float().mean().item(),
        "width_logits_raw_std": wl_raw.float().std().item(),
        "width_logits_raw_absmax": wl_raw.float().abs().max().item(),
        "width_bias_magnitude_mean": w_bias.abs().mean().item(),
        "width_bias_magnitude_max": w_bias.abs().max().item(),
        "width_bias_to_logit_ratio": w_bias.abs().mean().item() / (wl_raw.float().abs().mean().item() + 1e-10),
        "path_logits_raw_mean": pl_raw.float().mean().item(),
        "expert_logits_raw_mean": el_raw.float().mean().item(),
        "cost_aware_routing": ca,
        "cost_aware_bias": ca_bias,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    print("=" * 80)
    print("Phase 3: Train -> Observe -> Diagnose -> Fix -> Re-test")
    print("Focused on Conditional Compute Quality")
    print("=" * 80)

    all_results = {
        "seed": SEED, "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE, "seq_length": SEQ_LENGTH,
        "learning_rate": LEARNING_RATE,
    }

    set_seed(SEED)
    device = torch.device("cpu")

    # =========================================================================
    # STEP 1: Create model (NANO_10M base, modified for CPU + all 4 routing axes)
    # =========================================================================
    print("\n[1/8] Creating model...")
    c = ConfigFactory.get_config(ModelSize.NANO_10M)
    c.gradient_checkpointing = False
    c.expert_count = 6  # reduced from 8 for CPU
    c.top_k_experts = 2
    c.shard_experts = False
    c.router_dropout = 0.0
    c.dropout = 0.0
    c.model_name = "xorzen_phase3_test"

    model = zeroModel(c, test_mode=False)
    model = model.to(device)
    tp = count_parameters(model)
    trp = count_parameters(model, only_trainable=True)
    print(f"  Model: {c.model_name}")
    print(f"  Total params: {tp:,}, Trainable: {trp:,}")
    print(f"  Hidden: {c.hidden_size}, Layers: {c.num_layers}")
    print(f"  Width choices: {c.width_choices}")
    print(f"  Experts: {c.expert_count}, Top-K: {c.top_k_experts}")
    print(f"  Depth: {c.min_depth}-{c.max_depth}")
    all_results["model_config"] = {
        "hidden_size": c.hidden_size, "num_layers": c.num_layers,
        "width_choices": list(c.width_choices),
        "expert_count": c.expert_count, "top_k_experts": c.top_k_experts,
        "max_depth": c.max_depth, "min_depth": c.min_depth,
        "vocab_size": c.vocab_size, "total_params": tp, "trainable_params": trp,
    }

    # =========================================================================
    # STEP 2: Create adversarial batch + dataset
    # =========================================================================
    print("\n[2/8] Creating adversarial fixtures and dataset...")
    adv_ids, fnames = create_adversarial_batch(c.vocab_size, SEQ_LENGTH, device)
    print(f"  Adversarial batch: {adv_ids.shape}, fixtures: {fnames}")

    dataset = SyntheticLMDataset(c.vocab_size, SEQ_LENGTH, TRAINING_STEPS * BATCH_SIZE + 100, seed=SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"  Dataset: {len(dataset)} samples, {len(loader)} batches")

    # =========================================================================
    # STEP 3: PRE-TRAINING BASELINE
    # =========================================================================
    print("\n[3/8] Pre-training baseline...")
    set_seed(SEED)
    bl = collect_routing_stats(model, adv_ids, fnames, tag="pre")
    print(f"  depth={bl.depth_mask_sum:.3f}, width={bl.width_mean_multiplier:.3f}")
    print(f"  path_probs={[f'{p:.3f}' for p in bl.path_probs_mean]}")
    print(f"  expert_ent={bl.expert_entropy:.3f}, compute_CV={bl.per_token_compute_cv:.4f}")
    print(f"  depth_unique_patterns={bl.depth_unique_patterns}")
    all_results["baseline"] = bl.to_dict()

    bl_pf = collect_per_fixture_stats(model, adv_ids, fnames, device)
    print(f"  Per-fixture complexity:")
    for fn, fs in bl_pf.items():
        print(f"    {fn}: c={fs['complexity_mean']:.3f}, d={fs['depth_mask_sum']:.2f}, w={fs['width_mean_multiplier']:.3f}")
    all_results["baseline_per_fixture"] = bl_pf

    bl_bias = analyze_complexity_bias(model, adv_ids)
    print(f"  Bias/logit ratio: depth={bl_bias['depth_bias_to_logit_ratio']:.3f}, width={bl_bias['width_bias_to_logit_ratio']:.3f}")
    all_results["pre_bias"] = bl_bias

    # Active params diagnosis
    set_seed(SEED)
    model.eval()
    with torch.no_grad():
        cd = model.config.cot_dim * model.config.cot_components
        n_adv = adv_ids.shape[0]
        sl = adv_ids.shape[1]
        cf = torch.zeros(n_adv, sl, cd, device=device)
        rd = model.router(model.token_embedding(adv_ids), cf, training=False)
    apd = compute_correct_active_params(model, rd)
    print(f"  Active params: heuristic={apd['heuristic_estimate']:,} ({apd['heuristic_pct']:.1f}%)")
    print(f"    model_runtime={apd['model_runtime_estimate']:,} ({apd['model_estimate_pct']:.1f}%)")
    print(f"    training_actual={apd['training_active_params']:,} ({apd['training_active_pct']:.1f}%)")
    all_results["active_params_pre"] = apd

    # =========================================================================
    # STEP 4: TRAINING
    # =========================================================================
    print(f"\n[4/8] Training for {TRAINING_STEPS} steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAINING_STEPS)

    model.train()
    loss_history = []
    snapshots = []
    data_iter = iter(loader)
    t_start = time.time()
    step = 0

    while step < TRAINING_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        inp = batch["input_ids"].to(device)
        lab = batch["labels"].to(device)

        outputs = model(input_ids=inp, labels=lab, output_routing_info=True, return_dict=True)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append({
            "step": step, "total_loss": loss.item(),
            "lm_loss": (outputs.lm_loss.item() if outputs.lm_loss is not None else 0.0),
            "routing_loss": (outputs.routing_loss.item() if outputs.routing_loss is not None else 0.0),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if (step + 1) % LOG_INTERVAL == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                ss = collect_routing_stats(model, adv_ids, fnames, tag=f"step_{step}")
            pf = collect_per_fixture_stats(model, adv_ids, fnames, device)
            with torch.no_grad():
                sb = analyze_complexity_bias(model, adv_ids)
            model.train()

            snapshots.append({"step": step, "stats": ss.to_dict(), "per_fixture": pf, "bias": sb})
            el = time.time() - t_start
            print(f"  Step {step:4d} | loss={loss.item():.4f} | d={ss.depth_mask_sum:.3f} w={ss.width_mean_multiplier:.3f} "
                  f"pe={ss.path_entropy:.3f} ee={ss.expert_entropy:.3f} cv={ss.per_token_compute_cv:.4f} | {el:.1f}s")

        step += 1

    train_time = time.time() - t_start
    print(f"  Training done in {train_time:.1f}s")
    all_results["training_time_s"] = train_time
    all_results["loss_history_last10"] = loss_history[-10:]

    # =========================================================================
    # STEP 5: POST-TRAINING ANALYSIS
    # =========================================================================
    print("\n[5/8] Post-training analysis...")
    set_seed(SEED)
    post = collect_routing_stats(model, adv_ids, fnames, tag="post")
    print(f"  depth: {bl.depth_mask_sum:.3f} -> {post.depth_mask_sum:.3f}")
    print(f"  width: {bl.width_mean_multiplier:.3f} -> {post.width_mean_multiplier:.3f}")
    print(f"  path: {[f'{p:.3f}' for p in bl.path_probs_mean]} -> {[f'{p:.3f}' for p in post.path_probs_mean]}")
    print(f"  expert_ent: {bl.expert_entropy:.3f} -> {post.expert_entropy:.3f}")
    print(f"  compute_CV: {bl.per_token_compute_cv:.4f} -> {post.per_token_compute_cv:.4f}")
    print(f"  depth_unique: {bl.depth_unique_patterns} -> {post.depth_unique_patterns}")
    all_results["post_train"] = post.to_dict()

    post_pf = collect_per_fixture_stats(model, adv_ids, fnames, device)
    print(f"\n  Per-fixture comparison:")
    print(f"  {'Fixture':<18} {'Comp':>6} {'Depth':>8} {'Width':>8}")
    print(f"  {'-'*45}")
    for fn in fnames:
        pre = bl_pf[fn]; po = post_pf[fn]
        print(f"  {fn:<18} {pre['complexity_mean']:>6.3f} {pre['depth_mask_sum']:>8.2f}->{po['depth_mask_sum']:>5.2f} {pre['width_mean_multiplier']:>8.3f}->{po['width_mean_multiplier']:>5.3f}")
    all_results["post_per_fixture"] = post_pf

    post_bias = analyze_complexity_bias(model, adv_ids)
    all_results["post_bias"] = post_bias
    print(f"\n  Bias/logit ratios: depth {bl_bias['depth_bias_to_logit_ratio']:.3f}->{post_bias['depth_bias_to_logit_ratio']:.3f}, "
          f"width {bl_bias['width_bias_to_logit_ratio']:.3f}->{post_bias['width_bias_to_logit_ratio']:.3f}")

    # =========================================================================
    # STEP 6: ACTIVE PARAMS DIAGNOSIS
    # =========================================================================
    print("\n[6/8] Active parameter accounting...")
    set_seed(SEED)
    model.eval()
    with torch.no_grad():
        cd = model.config.cot_dim * model.config.cot_components
        n_adv2 = adv_ids.shape[0]
        sl2 = adv_ids.shape[1]
        cf = torch.zeros(n_adv2, sl2, cd, device=device)
        rd2 = model.router(model.token_embedding(adv_ids), cf, training=False)
    apd2 = compute_correct_active_params(model, rd2)
    all_results["active_params_post"] = apd2

    print(f"  Post-training:")
    print(f"    heuristic={apd2['heuristic_estimate']:,} ({apd2['heuristic_pct']:.1f}%)")
    print(f"    model_runtime={apd2['model_runtime_estimate']:,} ({apd2['model_estimate_pct']:.1f}%)")
    print(f"    training_actual={apd2['training_active_params']:,} ({apd2['training_active_pct']:.1f}%)")
    print(f"    inference_estimate={apd2['inference_active_params_estimate']:,}")
    print(f"    layer_activity={apd2['inference_layer_activity']:.3f}")
    print(f"    pathway_usage={apd2['pathway_usage']}")

    # =========================================================================
    # STEP 7: DIVERSITY DIAGNOSIS
    # =========================================================================
    print("\n[7/8] Routing diversity diagnosis...")
    fd = {fn: post_pf[fn]['depth_mask_sum'] for fn in fnames}
    fw = {fn: post_pf[fn]['width_mean_multiplier'] for fn in fnames}
    fp = {fn: post_pf[fn]['path_probs'] for fn in fnames}

    depth_var = float(np.var(list(fd.values())))
    width_var = float(np.var(list(fw.values())))

    path_divs = {}
    for i, f1 in enumerate(fnames):
        for j, f2 in enumerate(fnames):
            if i < j:
                p1 = np.array(fp[f1]) + 1e-10
                p2 = np.array(fp[f2]) + 1e-10
                p1 = p1 / p1.sum(); p2 = p2 / p2.sum()
                path_divs[f"{f1}_vs_{f2}"] = float(np.sum(p1 * np.log(p1 / p2)))

    max_kl = max(path_divs.values()) if path_divs else 0.0
    mean_kl = float(np.mean(list(path_divs.values()))) if path_divs else 0.0

    cv_improved = post.per_token_compute_cv > bl.per_token_compute_cv
    path_meaningful = mean_kl > 0.01
    depth_meaningful = depth_var > 0.01

    classification = (
        "INPUT_DEPENDENT" if (path_meaningful and depth_meaningful)
        else "PARTIAL" if (path_meaningful or depth_meaningful)
        else "UNIFORM"
    )

    div_diag = {
        "depth_variance_across_fixtures": depth_var,
        "width_variance_across_fixtures": width_var,
        "path_kl_divergences": path_divs,
        "max_path_kl": max_kl, "mean_path_kl": mean_kl,
        "compute_cv_pre": bl.per_token_compute_cv,
        "compute_cv_post": post.per_token_compute_cv,
        "depth_unique_pre": bl.depth_unique_patterns,
        "depth_unique_post": post.depth_unique_patterns,
        "path_entropy_pre": bl.path_entropy,
        "path_entropy_post": post.path_entropy,
        "width_entropy_pre": bl.width_entropy,
        "width_entropy_post": post.width_entropy,
        "expert_entropy_pre": bl.expert_entropy,
        "expert_entropy_post": post.expert_entropy,
        "input_dependent_routing": path_meaningful or depth_meaningful,
        "classification": classification,
    }
    all_results["diversity_diagnosis"] = div_diag
    print(f"  depth_var={depth_var:.4f}, width_var={width_var:.4f}")
    print(f"  mean_path_kl={mean_kl:.4f}, max_path_kl={max_kl:.4f}")
    print(f"  compute_CV: {bl.per_token_compute_cv:.4f} -> {post.per_token_compute_cv:.4f}")
    print(f"  Classification: {classification}")

    # =========================================================================
    # STEP 8: REGRESSION TESTS
    # =========================================================================
    print("\n[8/8] Running existing test suite...")
    test_results = {}
    for tf in ["test_fix_p5_load_balance.py", "test_fix_sppq_schedule.py",
               "test_fix_tokenizer_roundtrip.py", "test_fixes.py",
               "test_phase2_regression.py", "test_phase2_sparsity.py",
               "test_phase4_v04.py", "test_v05_fixes.py",
               "test_phase1_correctness.py"]:
        tfp = PROJECT_ROOT / "tests" / tf
        if not tfp.exists():
            test_results[tf] = "FILE_NOT_FOUND"; continue
        print(f"  Running {tf}...")
        t0 = time.time()
        rc = os.system(f"cd {PROJECT_ROOT} && /usr/bin/python3 -m pytest tests/{tf} -x -q 2>&1 | tail -3")
        test_results[tf] = f"exit={rc >> 8}, time={time.time()-t0:.1f}s"
    all_results["regression_tests"] = test_results

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3 FINAL REPORT")
    print("=" * 80)

    report = []

    # Entry 1: Active Parameter Accounting
    report.append({
        "experiment": "Active parameter estimation accuracy",
        "observation": f"Heuristic={apd2['heuristic_estimate']:,} ({apd2['heuristic_pct']:.1f}%), model_runtime={apd2['model_runtime_estimate']:,} ({apd2['model_estimate_pct']:.1f}%), training_actual={apd2['training_active_params']:,} ({apd2['training_active_pct']:.1f}%)",
        "root_cause": "_estimate_active_params uses soft depth_mask (STE probabilities). During training, ALL layers execute (STE blend computes everything then masks). The formula measures routing INTENT, not actual EXECUTION. The config.estimate_active_parameters() heuristic additionally multiplies by target_active_ratio (0.1 default), making it even more inaccurate.",
        "fix": "Add mode parameter: mode='intent' (current, inference planning) or mode='execution' (training reality). In 'execution' mode during training, report 100% of HASS params. At inference, use hard mask to count only executed layers. Document both modes.",
        "evidence": f"heuristic={apd2['heuristic_estimate']:,}, model_runtime={apd2['model_runtime_estimate']:,}, training_actual={apd2['training_active_params']:,}, inference_est={apd2['inference_active_params_estimate']:,}, layer_activity={apd2['inference_layer_activity']:.3f}",
        "classification": "PARTIAL",
    })

    # Entry 2: Per-token Compute Distribution
    report.append({
        "experiment": "Per-token compute distribution diversity",
        "observation": f"Pre-train CV={bl.per_token_compute_cv:.4f}, post-train CV={post.per_token_compute_cv:.4f}. Depth unique patterns: {bl.depth_unique_patterns} -> {post.depth_unique_patterns}.",
        "root_cause": f"Complexity bias/logit ratio: depth {bl_bias['depth_bias_to_logit_ratio']:.3f}->{post_bias['depth_bias_to_logit_ratio']:.3f}, width {bl_bias['width_bias_to_logit_ratio']:.3f}->{post_bias['width_bias_to_logit_ratio']:.3f}. Complexity std: {post_bias['complexity_std']:.3f}.",
        "fix": "None yet" if cv_improved else "Monitor at scale; if CV remains low after substantial training, consider reducing bias multipliers or adding bias warmup.",
        "evidence": f"CV: {bl.per_token_compute_cv:.4f}->{post.per_token_compute_cv:.4f}, patterns: {bl.depth_unique_patterns}->{post.depth_unique_patterns}, path_ent: {bl.path_entropy:.3f}->{post.path_entropy:.3f}",
        "classification": classification,
    })

    # Entry 3: Input-Dependent Routing
    report.append({
        "experiment": "Input-dependent routing (adversarial fixtures)",
        "observation": f"Path KL: mean={mean_kl:.4f}, max={max_kl:.4f}. Depth var={depth_var:.4f}. Classification={classification}.",
        "root_cause": "Router receives token embeddings as input. Whether routing becomes input-dependent depends on whether the feature encoder learns input-discriminative representations.",
        "fix": "None yet" if div_diag["input_dependent_routing"] else "If routing remains uniform after longer training, feature encoder may need stronger learning rate or contrastive loss.",
        "evidence": json.dumps({k: round(v, 4) for k, v in fd.items()}),
        "classification": classification,
    })

    # Entry 4: Complexity Bias Magnitude
    report.append({
        "experiment": "Complexity bias vs learned logits",
        "observation": f"Depth ratio: {bl_bias['depth_bias_to_logit_ratio']:.3f}->{post_bias['depth_bias_to_logit_ratio']:.3f}. Width ratio: {bl_bias['width_bias_to_logit_ratio']:.3f}->{post_bias['width_bias_to_logit_ratio']:.3f}.",
        "root_cause": f"{'Learned logits grew to dominate bias.' if post_bias['depth_bias_to_logit_ratio'] < 1.0 else 'Bias still dominates learned logits.'} Complexity output: mean={post_bias['complexity_mean']:.3f}, std={post_bias['complexity_std']:.3f}.",
        "fix": "No fix needed if ratio decreases during training (learned logits growing). If ratio remains > 1.0 after substantial training, reduce bias multipliers (currently 2.0 depth, 3.0 width) or anneal them.",
        "evidence": f"pre_d={bl_bias['depth_bias_to_logit_ratio']:.3f}, post_d={post_bias['depth_bias_to_logit_ratio']:.3f}, pre_w={bl_bias['width_bias_to_logit_ratio']:.3f}, post_w={post_bias['width_bias_to_logit_ratio']:.3f}",
        "classification": "PROVEN_DESIGN" if post_bias['depth_bias_to_logit_ratio'] < 1.0 else "BIAS_DOMINATED",
    })

    for i, e in enumerate(report, 1):
        print(f"\n--- Entry {i} ---")
        for k, v in e.items():
            print(f"  {k}: {v}")

    all_results["report_entries"] = report
    all_results["periodic_snapshots"] = snapshots

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results: {RESULTS_FILE}")
    return all_results


if __name__ == "__main__":
    main()
