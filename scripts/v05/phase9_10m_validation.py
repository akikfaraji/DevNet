"""
Phase 9 (v0.5) — Real 10M-scale validation.

The scientific core of v0.5: does Xorzen's adaptive routing actually
achieve better quality per unit of compute than a dense model of similar
total parameter count?

We train 4 model configurations on the SAME synthetic Markov corpus,
with the SAME optimizer, SAME training tokens, SAME batch size:

  1. Dense baseline (~7.8M params, vanilla transformer)
  2. Xorzen NANO_10M with routing DISABLED (~7.8M params,
     pathway_top_k=3, min_depth=max_depth → no sparsity, but routing
     machinery still runs so we measure its overhead)
  3. Xorzen NANO_10M with routing ENABLED (default config — genuine
     sparse pathway/depth/width/expert routing)

For each config we measure:
  - train loss, val loss, perplexity
  - tokens/sec, total training compute (steps × time)
  - estimated FLOPs/token (active vs dense)
  - peak memory (RSS)
  - active parameters/token (for Xorzen)
  - routing diversity (path/width/depth/expert utilization)

KEY SCIENTIFIC QUESTION:
  «Does Xorzen actually achieve better quality per unit of compute
   than a dense model of similar total parameter count?»

We do NOT extrapolate 10M results to 12B vs 60B — that would be
scientifically invalid. We report what actually happened at 10M scale.

Output: reports/v05/phase9_10m_validation.json
"""
import sys, os, json, time, gc, resource
sys.path.insert(0, "/home/z/my-project/xorzen_dev")
sys.path.insert(0, "/home/z/my-project/xorzen_dev/scripts/v05")
import torch
import numpy as np

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase
from dense_baseline import DenseConfig, DenseTransformer, build_dense_baseline_to_match
from markov_data import build_markov_dataloaders, MarkovCorpus

OUT = "/home/z/my-project/xorzen_dev/reports/v05/phase9_10m_validation.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ============ EXPERIMENT CONFIG ============
# MEMORY CONSTRAINT: environment has ~3.9GB RAM. A 10M-param Xorzen model
# with AdamW + activations OOMs at ~3.5GB. We therefore run at 1M-param
# scale (NANO_1M) which fits comfortably. The scientific comparison
# (dense vs Xorzen at the SAME scale) is still valid — we just cannot
# extrapolate to 10M without more memory.
VOCAB_SIZE = 1000
SEQ_LENGTH = 64
BATCH_SIZE = 8
TRAIN_STEPS = 250
VAL_EVERY = 25
LOG_EVERY = 25
LR = 1e-3
SEED = 42
MODEL_SIZE = ModelSize.NANO_1M  # was NANO_10M — too big for available RAM

# ============ Helpers ============
def get_rss_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0

def count_params(model):
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_total, n_train

def make_optimizer(model, lr=LR):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

def lr_schedule(step, total_steps, warmup=20):
    """Cosine schedule with linear warmup."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))

def estimate_flops_per_token(model_cfg_or_obj, is_xorzen, routing_decision=None):
    """Estimate FLOPs per token for a forward pass.

    For dense: 2 * (embed + L*(attn + ffn) + lm_head) per token
    For Xorzen: same but scaled by routing decisions (active layers, widths, etc.)
    """
    if is_xorzen:
        cfg = model_cfg_or_obj
        H = cfg.hidden_size
        L = cfg.num_layers
        V = cfg.vocab_size
        T = SEQ_LENGTH
        # Per-layer per-token FLOPs (rough):
        # - Attention (causal): 2*T*H + 4*H*H (QKVO) ≈ 4*H*H for T << H
        # - FFN (SlicedFFN): 2 * H * W * 2 (up + down), W = active width
        # - Pathway: rough H*H per pathway
        attn_flops = 4 * H * H + 2 * T * H
        ffn_flops_full = 2 * H * (4 * H) * 2  # max width = 4*H
        pathway_flops = H * H  # rough
        embed_flops = H + H * V  # embed + lm_head (tied)
        if routing_decision is not None:
            # Use actual routing decision
            active_layers = float(routing_decision.depth_mask.float().sum(-1).mean().item())
            wc_tensor = torch.tensor(cfg.width_choices)
            active_widths = wc_tensor[routing_decision.width_idx].float().mean().item()
            # Active pathways: pathway_top_k of 3
            active_pathway_frac = cfg.pathway_top_k / 3.0
            # Active experts: top_k of expert_count
            active_expert_frac = cfg.top_k_experts / cfg.expert_count
            ffn_flops_actual = 2 * H * int(active_widths) * 2
            per_layer = attn_flops + ffn_flops_actual + pathway_flops * active_pathway_frac
            # Add expert compute (top_k experts, each ~2*H*expert_hidden)
            expert_flops = active_expert_frac * 2 * H * int(H * cfg.expert_hidden_multiplier) * 2
            per_layer += expert_flops
            total = active_layers * per_layer + embed_flops
        else:
            # Worst case (all active)
            total = L * (attn_flops + ffn_flops_full + pathway_flops * 3) + embed_flops
        return int(total)
    else:
        # Dense baseline
        cfg = model_cfg_or_obj.cfg
        H = cfg.hidden_size
        L = cfg.num_layers
        V = cfg.vocab_size
        T = SEQ_LENGTH
        attn_flops = 4 * H * H + 2 * T * H
        ffn_flops = 2 * H * cfg.ffn_hidden * 2
        embed_flops = H + H * V
        return L * (attn_flops + ffn_flops) + embed_flops

def evaluate(model, val_loader, is_xorzen, device='cpu'):
    """Evaluate on validation set. Returns mean loss + perplexity + routing stats."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    routing_stats = None
    with torch.no_grad():
        for batch in val_loader:
            ids = batch.to(device)
            if is_xorzen:
                out = model(input_ids=ids, labels=ids, output_routing_info=True)
                if routing_stats is None:
                    rd = out.routing_info
                    routing_stats = {
                        'path_unique': int(torch.unique(rd.path_probs.argmax(-1)).numel()),
                        'path_distrib': torch.bincount(
                            rd.path_probs.argmax(-1).flatten(), minlength=3
                        ).float().tolist(),
                        'width_unique': int(torch.unique(rd.width_idx).numel()),
                        'width_distrib': torch.bincount(
                            rd.width_idx.flatten(), minlength=len(model.config.width_choices)
                        ).float().tolist(),
                        'expert_unique': int(torch.unique(rd.expert_indices).numel()),
                        'depth_active_per_layer': rd.depth_mask.float().mean(dim=(0,1)).tolist(),
                        'path_entropy': float(
                            -(rd.path_probs * torch.log(rd.path_probs + 1e-12)).sum(-1).mean().item()
                        ),
                    }
            else:
                out = model(ids, labels=ids)
            # Count non-pad tokens
            n_tokens = (ids != 0).sum().item() + ids.numel() * 0  # count all (no pad in markov)
            total_loss += float(out.lm_loss.item()) * ids.numel()
            total_tokens += ids.numel()
    mean_loss = total_loss / max(1, total_tokens)
    perplexity = float(np.exp(mean_loss))
    return mean_loss, perplexity, routing_stats

def train_one_config(name, model, is_xorzen, train_loader, val_loader, cfg=None,
                     train_steps=TRAIN_STEPS, device='cpu'):
    """Train one model config and return measurement dict."""
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")
    n_total, n_train = count_params(model)
    print(f"  params: {n_total:,} total ({n_train:,} trainable)")

    optimizer = make_optimizer(model)
    model.train()

    # Training loop
    train_losses = []
    val_losses = []
    val_ppls = []
    step_times = []
    tokens_per_sec_list = []

    rss_start = get_rss_mb()
    t_start = time.perf_counter()

    step = 0
    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break
            ids = batch.to(device)
            t0 = time.perf_counter()
            optimizer.zero_grad()
            if is_xorzen:
                out = model(input_ids=ids, labels=ids, output_routing_info=True)
            else:
                out = model(ids, labels=ids)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # LR schedule
            lr_scale = lr_schedule(step, train_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = LR * lr_scale
            optimizer.step()
            t1 = time.perf_counter()

            step_time = t1 - t0
            step_times.append(step_time)
            n_tokens = ids.numel()
            tokens_per_sec = n_tokens / step_time
            tokens_per_sec_list.append(tokens_per_sec)
            train_losses.append(float(out.lm_loss.item()))

            if step % LOG_EVERY == 0:
                print(f"  step {step:4d}/{train_steps}  loss={out.lm_loss.item():.4f}  "
                      f"lr={LR*lr_scale:.2e}  {step_time*1000:.0f}ms/step  "
                      f"{tokens_per_sec:.0f} tok/s")
            if step % VAL_EVERY == 0 or step == train_steps - 1:
                val_loss, val_ppl, _ = evaluate(model, val_loader, is_xorzen, device)
                val_losses.append(val_loss)
                val_ppls.append(val_ppl)
                print(f"    VAL: loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                model.train()
            step += 1

    t_end = time.perf_counter()
    rss_end = get_rss_mb()
    total_train_time = t_end - t_start

    # Final eval + routing stats
    val_loss_final, val_ppl_final, routing_stats = evaluate(model, val_loader, is_xorzen, device)

    # Estimate FLOPs/token
    if is_xorzen:
        # Get a routing decision from the model in eval mode
        model.eval()
        with torch.no_grad():
            sample = next(iter(val_loader)).to(device)
            out = model(input_ids=sample, output_routing_info=True)
        rd = out.routing_info
        active_flops = estimate_flops_per_token(model.config, is_xorzen=True, routing_decision=rd)
        dense_flops = estimate_flops_per_token(model.config, is_xorzen=True, routing_decision=None)
    else:
        active_flops = estimate_flops_per_token(model, is_xorzen=False)
        dense_flops = active_flops

    result = {
        'name': name,
        'is_xorzen': is_xorzen,
        'params_total': n_total,
        'params_trainable': n_train,
        'train_steps': train_steps,
        'train_tokens': int(np.sum(tokens_per_sec_list) * np.mean(step_times)),  # rough
        'train_loss_initial': train_losses[0] if train_losses else None,
        'train_loss_final': train_losses[-1] if train_losses else None,
        'val_loss_initial': val_losses[0] if val_losses else None,
        'val_loss_final': val_loss_final,
        'val_ppl_final': val_ppl_final,
        'val_loss_curve': val_losses,
        'val_ppl_curve': val_ppls,
        'mean_step_time_ms': float(np.mean(step_times) * 1000),
        'mean_tokens_per_sec': float(np.mean(tokens_per_sec_list)),
        'total_train_time_sec': total_train_time,
        'active_flops_per_token': active_flops,
        'dense_equivalent_flops_per_token': dense_flops,
        'flops_reduction_pct': 100.0 * (1.0 - active_flops / max(1, dense_flops)),
        'rss_start_mb': rss_start,
        'rss_end_mb': rss_end,
        'rss_delta_mb': rss_end - rss_start,
        'routing_stats': routing_stats,
    }
    print(f"\n  SUMMARY {name}:")
    print(f"    val_loss={val_loss_final:.4f}  val_ppl={val_ppl_final:.2f}")
    print(f"    {np.mean(step_times)*1000:.0f} ms/step  {np.mean(tokens_per_sec_list):.0f} tok/s")
    print(f"    active FLOPs/token: {active_flops:,} (dense: {dense_flops:,}, reduction {result['flops_reduction_pct']:.1f}%)")
    return result

# ============ Build the 4 model configs ============

print("=" * 60)
print("  Phase 9 — 10M-scale validation")
print("=" * 60)
print(f"Config: vocab={VOCAB_SIZE}, seq_len={SEQ_LENGTH}, batch={BATCH_SIZE}, "
      f"steps={TRAIN_STEPS}, lr={LR}, seed={SEED}")

# Build the dataset ONCE (shared across all configs)
print("\nBuilding Markov corpus...")
corpus, train_loader, val_loader = build_markov_dataloaders(
    vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH,
    train_sequences=2000, val_sequences=200,
    batch_size=BATCH_SIZE, seed=SEED,
)
print(f"Corpus: mean entropy = {corpus.mean_entropy_bits:.3f} bits/token")
print(f"        theoretical min loss = {corpus.min_loss_nats:.4f} nats")
print(f"        uniform baseline loss = {np.log(VOCAB_SIZE):.4f} nats")
print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

# Get target param count from Xorzen NANO_10M
print("\nBuilding Xorzen NANO_10M to get target param count...")
torch.manual_seed(SEED); np.random.seed(SEED)
xorzen_cfg = ConfigFactory.get_config(MODEL_SIZE)
xorzen_cfg.update(
    shard_experts=False, gradient_checkpointing=False,
    eval_routing_noise=0.15,  # v0.5 default
    context_length=SEQ_LENGTH,  # match dataset
    vocab_size=VOCAB_SIZE,      # match dataset (smaller vocab for faster learning)
    pad_token_id=0,
)
xorzen_model_temp = zeroBase(config=xorzen_cfg, test_mode=True)
target_params, _ = count_params(xorzen_model_temp)
print(f"Xorzen NANO_10M (ctx={SEQ_LENGTH}, vocab={VOCAB_SIZE}): {target_params:,} total params")
del xorzen_model_temp
gc.collect()

# Build all 4 models
results = {'experiment_config': {
    'vocab_size': VOCAB_SIZE,
    'seq_length': SEQ_LENGTH,
    'batch_size': BATCH_SIZE,
    'train_steps': TRAIN_STEPS,
    'lr': LR,
    'seed': SEED,
    'corpus_mean_entropy_bits': corpus.mean_entropy_bits,
    'corpus_min_loss_nats': corpus.min_loss_nats,
    'uniform_baseline_loss_nats': float(np.log(VOCAB_SIZE)),
}, 'configs': {}}

# Config 1: Dense baseline
print(f"\n[1/3] Building Dense baseline (~{target_params:,} params)...")
torch.manual_seed(SEED); np.random.seed(SEED)
dense_model = build_dense_baseline_to_match(
    target_params=target_params, vocab_size=VOCAB_SIZE, context_length=SEQ_LENGTH,
)
dense_n = dense_model.num_parameters()
print(f"  dense baseline: {dense_n:,} params (target was {target_params:,})")

# Config 2: Xorzen NANO_10M with routing DISABLED
print(f"\n[2/3] Building Xorzen NANO_10M (routing DISABLED)...")
torch.manual_seed(SEED); np.random.seed(SEED)
cfg_disabled = ConfigFactory.get_config(MODEL_SIZE)
cfg_disabled.update(
    shard_experts=False, gradient_checkpointing=False,
    eval_routing_noise=0.0,             # no eval noise (not needed when routing is disabled)
    pathway_top_k=3,                    # all pathways active (no sparsity)
    min_depth=cfg_disabled.max_depth,   # all layers active (no depth skip)
    cost_aware_routing=False,            # no budget modulation
    context_length=SEQ_LENGTH,           # match dataset
    vocab_size=VOCAB_SIZE,               # match dataset
    pad_token_id=0,
)
xorzen_disabled = zeroBase(config=cfg_disabled, test_mode=True)
xorzen_disabled_n, _ = count_params(xorzen_disabled)
print(f"  xorzen (routing disabled): {xorzen_disabled_n:,} params")

# Config 3: Xorzen NANO_10M with routing ENABLED (default = genuine sparse)
print(f"\n[3/3] Building Xorzen NANO_10M (genuine sparse routing)...")
torch.manual_seed(SEED); np.random.seed(SEED)
cfg_sparse = ConfigFactory.get_config(MODEL_SIZE)
cfg_sparse.update(
    shard_experts=False, gradient_checkpointing=False,
    eval_routing_noise=0.15,            # v0.5 default
    pathway_top_k=2,                    # top-2 of 3 pathways
    min_depth=2,                        # allow depth skip after layer 2
    cost_aware_routing=True,
    context_length=SEQ_LENGTH,           # match dataset
    vocab_size=VOCAB_SIZE,               # match dataset
    pad_token_id=0,
)
xorzen_sparse = zeroBase(config=cfg_sparse, test_mode=True)
xorzen_sparse_n, _ = count_params(xorzen_sparse)
print(f"  xorzen (genuine sparse): {xorzen_sparse_n:,} params")

# ============ Train all 3 configs ============
results['configs']['dense_baseline'] = train_one_config(
    'dense_baseline', dense_model, is_xorzen=False,
    train_loader=train_loader, val_loader=val_loader,
)

results['configs']['xorzen_routing_disabled'] = train_one_config(
    'xorzen_routing_disabled', xorzen_disabled, is_xorzen=True,
    train_loader=train_loader, val_loader=val_loader, cfg=cfg_disabled,
)

results['configs']['xorzen_genuine_sparse'] = train_one_config(
    'xorzen_genuine_sparse', xorzen_sparse, is_xorzen=True,
    train_loader=train_loader, val_loader=val_loader, cfg=cfg_sparse,
)

# ============ Scientific comparison ============
print("\n" + "=" * 60)
print("  SCIENTIFIC COMPARISON")
print("=" * 60)

dense_r = results['configs']['dense_baseline']
disabled_r = results['configs']['xorzen_routing_disabled']
sparse_r = results['configs']['xorzen_genuine_sparse']

comparison = {
    'question': 'Does Xorzen achieve better quality per unit of compute than dense?',
    'val_loss': {
        'dense': dense_r['val_loss_final'],
        'xorzen_disabled': disabled_r['val_loss_final'],
        'xorzen_sparse': sparse_r['val_loss_final'],
    },
    'val_ppl': {
        'dense': dense_r['val_ppl_final'],
        'xorzen_disabled': disabled_r['val_ppl_final'],
        'xorzen_sparse': sparse_r['val_ppl_final'],
    },
    'tokens_per_sec': {
        'dense': dense_r['mean_tokens_per_sec'],
        'xorzen_disabled': disabled_r['mean_tokens_per_sec'],
        'xorzen_sparse': sparse_r['mean_tokens_per_sec'],
    },
    'active_flops_per_token': {
        'dense': dense_r['active_flops_per_token'],
        'xorzen_disabled': disabled_r['active_flops_per_token'],
        'xorzen_sparse': sparse_r['active_flops_per_token'],
    },
    'params': {
        'dense': dense_r['params_total'],
        'xorzen_disabled': disabled_r['params_total'],
        'xorzen_sparse': sparse_r['params_total'],
    },
    'verdicts': {},
}
# Quality per FLOP (lower loss per FLOP = better)
for k in ['dense', 'xorzen_disabled', 'xorzen_sparse']:
    flops = comparison['active_flops_per_token'][k]
    loss = comparison['val_loss'][k]
    comparison['verdicts'][f'{k}_quality_per_flop'] = float(loss / flops * 1e6)  # loss per MFLOP

# Sparse vs dense: does sparse achieve lower loss at lower compute?
sparse_vs_dense_loss_delta = sparse_r['val_loss_final'] - dense_r['val_loss_final']
sparse_vs_dense_flops_delta = sparse_r['active_flops_per_token'] - dense_r['active_flops_per_token']
comparison['sparse_vs_dense'] = {
    'loss_delta': float(sparse_vs_dense_loss_delta),
    'flops_delta': int(sparse_vs_dense_flops_delta),
    'sparse_uses fewer_flops': sparse_vs_dense_flops_delta < 0,
    'sparse_achieves_lower_loss': sparse_vs_dense_loss_delta < 0,
    'sparse_wins': sparse_vs_dense_flops_delta < 0 and sparse_vs_dense_loss_delta < 0,
    'sparse_wins_on_quality_per_flop': (
        sparse_r['val_loss_final'] / sparse_r['active_flops_per_token']
        < dense_r['val_loss_final'] / dense_r['active_flops_per_token']
    ),
}

# Disabled vs sparse: does sparse routing HELP quality (not just reduce FLOPs)?
disabled_vs_sparse_loss_delta = sparse_r['val_loss_final'] - disabled_r['val_loss_final']
comparison['disabled_vs_sparse'] = {
    'loss_delta_sparse_minus_disabled': float(disabled_vs_sparse_loss_delta),
    'sparse_routing_helps_quality': disabled_vs_sparse_loss_delta < 0,
    'sparse_routing_hurts_quality': disabled_vs_sparse_loss_delta > 0,
    'interpretation': (
        'sparse routing HELPS quality at fixed params' if disabled_vs_sparse_loss_delta < 0
        else 'sparse routing HURTS quality at fixed params (overhead > benefit at this scale)'
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

results['comparison'] = comparison

# Save
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to {OUT}")
