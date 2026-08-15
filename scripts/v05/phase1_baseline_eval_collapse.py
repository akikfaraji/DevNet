"""
Phase 1 (v0.5) — Measure eval routing collapse BEFORE the fix.

Trains a tiny Xorzen model for ~200 steps, then evaluates the routing
diversity in EVAL mode. We expect:
  - path_probs heavily collapsed onto 1-2 pathways
  - width_idx collapsed onto 1 width
  - depth_mask: most layers always active or always inactive

This script produces JSON output under reports/v05/phase1_baseline_eval.json.
"""
import sys, os, json, time
sys.path.insert(0, "/home/z/my-project/xorzen_dev")
import torch
import numpy as np

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase

OUT_DIR = "/home/z/my-project/xorzen_dev/reports/v05"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

H = 64
cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
cfg.update(
    vocab_size=128, context_length=64, hidden_size=H,
    num_layers=4, num_attention_heads=4, max_depth=4, min_depth=1,
    width_choices=(H // 2, H), expert_count=4, top_k_experts=2,
    router_hidden_dim=H // 4, dropout=0.0, pad_token_id=0,
    load_balancing_weight=0.001, shard_experts=False,
    pathway_top_k=2, gradient_checkpointing=False,
    use_sliced_ffn=True, width_div_weight=0.1,
    path_div_weight=0.2, unify_load_balance=True, cost_aware_routing=True,
)

model = zeroBase(config=cfg, test_mode=True)
model.train()

# Synthetic deterministic sequences (varying across batch)
rng = np.random.RandomState(42)
sequences = []
for i in range(8):
    offset = rng.randint(1, cfg.vocab_size)
    start = rng.randint(0, cfg.vocab_size)
    seq = [(start + offset * t) % cfg.vocab_size for t in range(cfg.context_length)]
    sequences.append(seq)
sequences = torch.tensor(sequences, dtype=torch.long)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01
)

# Train briefly to give the router a chance to collapse
print("Training 200 steps to induce routing collapse...")
losses = []
for step in range(200):
    optimizer.zero_grad()
    out = model(input_ids=sequences, labels=sequences, output_routing_info=True)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 50 == 0:
        print(f"  step {step}: lm_loss={out.lm_loss.item():.4f}")
    losses.append(float(out.lm_loss.item()))

# Now evaluate routing diversity in EVAL mode
model.eval()
with torch.no_grad():
    out = model(input_ids=sequences, output_routing_info=True)
rd = out.routing_info

# Pathway diversity
path_probs = rd.path_probs  # [B, T, 3]
path_argmax = path_probs.argmax(dim=-1)  # [B, T]
unique_paths = torch.unique(path_argmax).numel()
path_distrib = torch.bincount(path_argmax.flatten(), minlength=3).float()
path_distrib = path_distrib / path_distrib.sum()

# Width diversity
width_idx = rd.width_idx  # [B, T]
unique_widths = torch.unique(width_idx).numel()
width_distrib = torch.bincount(width_idx.flatten(), minlength=len(cfg.width_choices)).float()
width_distrib = width_distrib / width_distrib.sum()

# Depth diversity
depth_mask = rd.depth_mask  # [B, T, max_depth]
depth_active_per_layer = depth_mask.float().mean(dim=(0, 1))  # [max_depth]
unique_depth_patterns = torch.unique(depth_mask.view(-1, cfg.max_depth), dim=0).shape[0]

# Expert diversity
expert_indices = rd.expert_indices  # [B, T, top_k]
unique_experts = torch.unique(expert_indices).numel()

results = {
    "model_size": "TINY_23K, H=64, 4 layers",
    "train_steps": 200,
    "train_loss_start": losses[0],
    "train_loss_end": losses[-1],
    "eval_pathway": {
        "unique_paths_active": int(unique_paths),
        "pathway_distribution": path_distrib.tolist(),
        "n_pathways_configured": 3,
        "collapsed": bool(unique_paths < 3),
    },
    "eval_width": {
        "unique_widths_active": int(unique_widths),
        "width_distribution": width_distrib.tolist(),
        "n_widths_configured": len(cfg.width_choices),
        "collapsed": bool(unique_widths < len(cfg.width_choices)),
    },
    "eval_depth": {
        "active_per_layer": depth_active_per_layer.tolist(),
        "unique_depth_patterns": int(unique_depth_patterns),
        "max_patterns_possible": 2 ** cfg.max_depth,
    },
    "eval_expert": {
        "unique_experts_active": int(unique_experts),
        "n_experts_configured": cfg.expert_count,
        "top_k": cfg.top_k_experts,
    },
}

out_path = os.path.join(OUT_DIR, "phase1_baseline_eval.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(json.dumps(results, indent=2))
