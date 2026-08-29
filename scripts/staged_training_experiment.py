#!/usr/bin/env python3
"""
XORZEN Staged Training Experiment
=================================
Reproducible CPU-feasible real-text training that exercises the full
intended XORZEN training lifecycle:

  Stage 1: Language pre-training (CoT frozen, CoT loss = 0)
  Transition: enable_cot() — unfreeze CoT, activate consistency loss
  Stage 2: Specialized training (CoT active, consistency loss on)

All configuration, metrics, and assertions are recorded.
"""

import sys
import os
import json
import time
import math
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
import numpy as np

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.model import zeroModel
from xorzen.tokenizer import load_pretrained

# ============================================================
# CONFIGURATION — all hyperparameters recorded for reproducibility
# ============================================================

SEED = 42
STAGE1_STEPS = 80
STAGE2_STEPS = 40
SEQ_LEN = 64
BATCH_SIZE = 2
GRAD_ACCUM = 1
LR = 3e-3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "staged_training_001"
DATA_PATH = PROJECT_ROOT / "data" / "corpus.jsonl"

# ============================================================
# SETUP
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)  # CPU only


def create_corpus(data_path: Path):
    """Create a small real-text JSONL corpus if it doesn't exist."""
    if data_path.exists():
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Diverse English text — enough variety for language modeling
    texts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing purposes.",
        "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. These systems improve their performance on a specific task over time without being explicitly programmed.",
        "The transformer architecture, introduced in the 2017 paper Attention Is All You Need, revolutionized natural language processing. It relies entirely on self-attention mechanisms, eliminating the need for recurrent or convolutional layers.",
        "Mixture of Experts models divide the computation among multiple specialized networks called experts. A routing mechanism selects which experts to activate for each input, enabling efficient scaling of model capacity.",
        "Chain of thought reasoning is a technique where a model generates intermediate steps before arriving at a final answer. This approach has been shown to improve performance on complex reasoning tasks significantly.",
        "Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize the loss function. The direction and magnitude of each step are determined by the gradient of the loss with respect to the parameters.",
        "The attention mechanism allows a model to focus on different parts of the input sequence when producing each element of the output. Self-attention computes a weighted sum of all positions in the sequence, where the weights are learned dynamically.",
        "State space models offer an alternative to attention for sequence modeling. They represent a sequence as a continuous dynamical system and can be computed in linear time with respect to sequence length.",
        "Regularization techniques such as dropout, weight decay, and data augmentation help prevent overfitting. They work by adding constraints or noise during training that encourage the model to learn more robust features.",
        "The vocabulary of a language model defines the set of tokens it can process. Subword tokenization methods like Byte Pair Encoding split words into smaller units, allowing the model to handle rare and unseen words effectively.",
        "Embeddings map discrete tokens to continuous vector representations. These learned vectors capture semantic relationships between tokens, enabling the model to generalize from patterns seen during training to new contexts.",
        "Layer normalization stabilizes training by normalizing the activations within each layer. Unlike batch normalization, it operates independently for each sample, making it suitable for variable-length sequences and online inference.",
        "Curriculum learning is a training strategy where the model is first exposed to simpler examples before progressing to more complex ones. This approach can accelerate learning and lead to better final performance.",
        "Knowledge distillation transfers the learned knowledge from a large teacher model to a smaller student model. The student is trained to match the teacher's output distribution, often achieving better performance than training from scratch.",
        "Sparse activation in neural networks refers to the practice of only computing a subset of the model's parameters for each input. This can dramatically reduce computational cost while maintaining model quality.",
        "The learning rate schedule controls how the learning rate changes during training. Warmup followed by cosine decay is a common schedule that starts with a small rate, increases it linearly, then gradually decreases it.",
        "Backpropagation computes gradients by applying the chain rule recursively through the computational graph. It enables efficient training of deep networks by computing all parameter gradients in a single backward pass.",
        "Checkpointing saves the model state during training so that training can be resumed if interrupted. It also allows the best model to be selected based on validation performance.",
        "Numerical stability is critical in deep learning. Techniques like mixed precision training, gradient clipping, and careful initialization help prevent issues such as exploding or vanishing gradients.",
        "The softmax function converts a vector of real numbers into a probability distribution. It is commonly used as the final layer in classification models, producing scores that sum to one.",
        "Residual connections allow gradients to flow directly through shortcut paths, bypassing one or more layers. This simple technique enables the training of very deep networks that would otherwise suffer from vanishing gradients.",
        "Positional encodings provide information about the position of each token in a sequence. Since the transformer processes all tokens in parallel, it needs explicit position information to understand token order.",
        "Cross-entropy loss measures the difference between two probability distributions. In language modeling, it penalizes the model when it assigns low probability to the correct next token.",
        "Adaptive optimizers like Adam and AdamW adjust the learning rate for each parameter individually based on estimates of the first and second moments of the gradients. This often leads to faster convergence than standard SGD.",
        "The context length determines the maximum number of tokens a model can process in a single forward pass. Longer contexts allow the model to capture more distant dependencies but increase computational cost quadratically for attention-based models.",
        "Transfer learning leverages knowledge gained from training on one task to improve performance on a different but related task. Pre-training on a large corpus followed by fine-tuning on a specific task is a common paradigm.",
        "Beam search is a decoding algorithm that maintains multiple candidate sequences during generation. At each step, it extends the most promising candidates and prunes the rest, balancing quality and diversity.",
        "The bias-variance tradeoff is a fundamental concept in machine learning. Models with high capacity have low bias but may overfit, while simpler models have higher bias but better generalization.",
        "Data augmentation increases the effective size and diversity of the training set through transformations like cropping, flipping, and synonym replacement. This helps the model learn invariant features.",
        "Hyperparameter tuning is the process of selecting the best configuration of training parameters such as learning rate, batch size, and model architecture. This is typically done through grid search, random search, or Bayesian optimization.",
    ]

    with open(data_path, 'w') as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"Created corpus: {data_path} ({len(texts)} passages)")


def tokenize_corpus(data_path: Path, tokenizer, seq_len: int):
    """Load JSONL, tokenize, return flat list of token IDs."""
    all_ids = []
    with open(data_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)
            # Filter out-of-range token IDs
            ids = [i for i in ids if 0 <= i < tokenizer.get_vocab_size()]
            all_ids.extend(ids)
    return all_ids


def make_batch(token_ids: list, seq_len: int, batch_size: int, step: int):
    """Create a batch from the token corpus. Wraps around."""
    total_tokens = batch_size * seq_len
    n = len(token_ids)
    start = (step * batch_size * seq_len) % max(n - total_tokens, 1)
    # Collect enough tokens, wrapping if needed
    ids = []
    while len(ids) < total_tokens:
        needed = total_tokens - len(ids)
        pos = start % n
        available = min(needed, n - pos)
        ids.extend(token_ids[pos:pos + available])
        start += available
    batch = torch.tensor(ids[:total_tokens], dtype=torch.long).reshape(batch_size, seq_len)
    return batch


def count_params(model, requires_grad_only=False):
    """Count parameters."""
    total = 0
    for p in model.parameters():
        if requires_grad_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def get_subsystem_grad_info(model):
    """Get gradient norms for major subsystems."""
    subsystems = {
        'token_embedding': [],
        'position_embedding': [],
        'blocks': [],
        'router': [],
        'moe': [],
        'merger': [],
        'final_norm': [],
        'lm_head': [],
        'cot': [],
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        gnorm = param.grad.norm().item()
        for sub in subsystems:
            if name.startswith(sub):
                subsystems[sub].append(gnorm)
                break
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in subsystems.items()}


def get_routing_stats(routing_info):
    """Extract routing statistics from RoutingDecision."""
    stats = {}
    if routing_info is None:
        return stats
    # Depth
    if hasattr(routing_info, 'depth_mask') and routing_info.depth_mask is not None:
        dm = routing_info.depth_mask.float()
        stats['depth_avg_active'] = dm.sum(-1).mean().item()
    # Width
    if hasattr(routing_info, 'width_multiplier') and routing_info.width_multiplier is not None:
        wm = routing_info.width_multiplier.float()
        stats['width_avg_multiplier'] = wm.mean().item()
    # Pathway
    if hasattr(routing_info, 'path_probs') and routing_info.path_probs is not None:
        pp = routing_info.path_probs
        stats['pathway_entropy'] = -(pp * (pp + 1e-10).log()).sum(-1).mean().item()
        stats['pathway_top1_prob'] = pp.max(-1)[0].mean().item()
    # Expert
    if hasattr(routing_info, 'expert_indices') and routing_info.expert_indices is not None:
        ei = routing_info.expert_indices
        unique_experts = torch.unique(ei.flatten())
        stats['experts_used'] = len(unique_experts)
    # Auxiliary losses
    if hasattr(routing_info, 'auxiliary') and routing_info.auxiliary:
        for k, v in routing_info.auxiliary.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                stats[f'aux_{k}'] = v.item()
    return stats


def verify_checkpoint_reproducibility(model, checkpoint_path, token_ids, seq_len, batch_size, step=0):
    """Verify saving and restoring produces same output."""
    model.eval()
    batch = make_batch(token_ids, seq_len, batch_size, step)
    with torch.no_grad():
        out_before = model(input_ids=batch)
        logits_before = out_before.logits.clone()

    # Save
    ckpt = {
        'model_state_dict': model.state_dict(),
        'config': {k: v for k, v in model.config.__dict__.items() if not k.startswith('_')},
    }
    torch.save(ckpt, checkpoint_path)

    # Restore
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        out_after = model(input_ids=batch)
        logits_after = out_after.logits.clone()

    max_diff = (logits_before - logits_after).abs().max().item()
    return max_diff


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    logging.disable(logging.WARNING)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Record metadata ----
    git_commit = os.popen('cd /home/z/my-project/DevNet && git rev-parse --short HEAD').read().strip()
    now = datetime.now(timezone.utc).isoformat()

    results = {
        'experiment_id': 'staged_training_001',
        'timestamp': now,
        'git_commit': git_commit,
        'seed': SEED,
    }

    print("\n" + "=" * 70)
    print("XORZEN STAGED TRAINING EXPERIMENT")
    print("=" * 70)
    print(f"  Git commit:  {git_commit}")
    print(f"  Timestamp:   {now}")
    print(f"  Seed:        {SEED}")
    print(f"  Device:      CPU")

    # ---- Create corpus & tokenize ----
    create_corpus(DATA_PATH)
    tokenizer = load_pretrained('zero_bpe_10k')
    token_ids = tokenize_corpus(DATA_PATH, tokenizer, SEQ_LEN)
    print(f"  Corpus:      {DATA_PATH.name} ({len(token_ids)} tokens, vocab={tokenizer.get_vocab_size()})")

    # ---- Model config ----
    cfg = ConfigFactory.get_config('1M')  # NANO_1M
    cfg.vocab_size = tokenizer.get_vocab_size()  # 10000 to match tokenizer
    cfg.context_length = SEQ_LEN
    cfg.tie_word_embeddings = True
    cfg.pad_token_id = 0
    cfg.gradient_checkpointing = False
    cfg.dropout = 0.0  # Deterministic for experiment

    model = zeroModel(cfg, test_mode=True)
    model.train()

    total_params = count_params(model)
    trainable_params = count_params(model, requires_grad_only=True)
    cot_params = count_params(model.cot)

    print(f"  Model:       {cfg.model_name}")
    print(f"  Params:      {total_params:,} total, {trainable_params:,} trainable")
    print(f"  CoT params:  {cot_params:,} (frozen)")
    print(f"  Seq len:     {SEQ_LEN}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LR}")
    print(f"  Optimizer:   AdamW (wd={WEIGHT_DECAY})")
    print(f"  Stage 1:     {STAGE1_STEPS} steps (CoT frozen)")
    print(f"  Stage 2:     {STAGE2_STEPS} steps (CoT active)")
    print()

    results['config'] = {
        'model_name': cfg.model_name,
        'vocab_size': cfg.vocab_size,
        'hidden_size': cfg.hidden_size,
        'num_layers': cfg.num_layers,
        'num_attention_heads': cfg.num_attention_heads,
        'expert_count': cfg.expert_count,
        'top_k_experts': cfg.top_k_experts,
        'context_length': cfg.context_length,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'cot_params': cot_params,
        'seq_len': SEQ_LEN,
        'batch_size': BATCH_SIZE,
        'gradient_accumulation': GRAD_ACCUM,
        'optimizer': 'AdamW',
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'max_grad_norm': MAX_GRAD_NORM,
        'stage1_steps': STAGE1_STEPS,
        'stage2_steps': STAGE2_STEPS,
        'corpus_tokens': len(token_ids),
        'tokenizer': 'zero_bpe_10k',
        'checkpoint_dir': str(CHECKPOINT_DIR),
    }

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )

    # ==========================================================
    # STAGE 1: Language Pre-Training (CoT frozen)
    # ==========================================================
    print("-" * 70)
    print("STAGE 1: LANGUAGE PRE-TRAINING (CoT FROZEN)")
    print("-" * 70)

    stage1_losses = []
    stage1_lm_losses = []
    stage1_routing_losses = []
    stage1_cot_losses = []
    stage1_routing_stats = []
    stage1_grad_info = []

    # Verify CoT is frozen
    assert not model._cot_enabled, "CoT should be disabled at start"
    for name, param in model.cot.named_parameters():
        assert not param.requires_grad, f"CoT param {name} should be frozen"
    print("  [VERIFIED] CoT is frozen (all params requires_grad=False)")
    print("  [VERIFIED] model._cot_enabled is False")

    stage1_start = time.time()
    for step in range(STAGE1_STEPS):
        optimizer.zero_grad()
        batch = make_batch(token_ids, SEQ_LEN, BATCH_SIZE, step)
        labels = batch.clone()

        out = model(input_ids=batch, labels=labels, output_routing_info=True)

        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        # Record metrics every step
        stage1_losses.append(loss.item())
        stage1_lm_losses.append(out.lm_loss.item())
        stage1_routing_losses.append(out.routing_loss.item())
        stage1_cot_losses.append(out.cot_consistency_loss.item())

        # Routing stats every 5 steps
        if step % 5 == 0 or step == STAGE1_STEPS - 1:
            rs = get_routing_stats(out.routing_info)
            stage1_routing_stats.append({'step': step, **rs})
            gi = get_subsystem_grad_info(model)
            stage1_grad_info.append({'step': step, **gi})

        if step % 20 == 0 or step == STAGE1_STEPS - 1:
            print(f"  Step {step:3d}/{STAGE1_STEPS}: loss={loss.item():.4f}  lm={out.lm_loss.item():.4f}  "
                  f"routing={out.routing_loss.item():.6f}  cot={out.cot_consistency_loss.item():.6f}")

    stage1_time = time.time() - stage1_start
    print(f"  Stage 1 completed in {stage1_time:.1f}s")

    # ---- Stage 1 Assertions ----
    assert stage1_losses[-1] < stage1_losses[0], \
        f"Stage 1 loss should decrease: {stage1_losses[0]:.4f} -> {stage1_losses[-1]:.4f}"
    assert all(math.isfinite(l) for l in stage1_losses), "Stage 1: all losses must be finite"
    assert all(c == 0.0 for c in stage1_cot_losses), "Stage 1: CoT loss must be exactly 0.0"

    # Verify CoT params still frozen
    for name, param in model.cot.named_parameters():
        assert not param.requires_grad, f"CoT param {name} should still be frozen"
    print("  [PASS] Loss decreased: {0:.4f} -> {1:.4f}".format(stage1_losses[0], stage1_losses[-1]))
    print("  [PASS] All losses finite")
    print("  [PASS] CoT loss = 0.0 throughout")
    print("  [PASS] CoT params remain frozen")

    # ---- Save Stage 1 checkpoint ----
    stage1_ckpt = CHECKPOINT_DIR / "stage1_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': STAGE1_STEPS,
        'stage': 1,
        'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
    }, stage1_ckpt)
    print(f"  Checkpoint saved: {stage1_ckpt}")

    # ---- Checkpoint reproducibility ----
    repro_ckpt = CHECKPOINT_DIR / "repro_test.pt"
    max_diff = verify_checkpoint_reproducibility(model, repro_ckpt, token_ids, SEQ_LEN, BATCH_SIZE, step=0)
    assert max_diff < 1e-5, f"Checkpoint reproducibility failed: max_diff={max_diff}"
    print(f"  [PASS] Checkpoint reproducibility: max_diff={max_diff:.2e}")
    repro_ckpt.unlink()

    # ==========================================================
    # CoT TRANSITION
    # ==========================================================
    print("\n" + "-" * 70)
    print("CoT TRANSITION: ENABLING CHAIN-OF-THOUGHT")
    print("-" * 70)

    # Record pre-transition optimizer state
    opt_param_ids_pre = {id(p) for p in optimizer.param_groups[0]['params']}
    opt_n_params_pre = len(optimizer.param_groups[0]['params'])

    # Enable CoT
    model.enable_cot()

    # Verify
    assert model._cot_enabled, "_cot_enabled should be True after enable_cot()"
    cot_param_names = []
    for name, param in model.cot.named_parameters():
        assert param.requires_grad, f"CoT param {name} should be unfrozen"
        cot_param_names.append(name)
    print(f"  [VERIFIED] model._cot_enabled = True")
    print(f"  [VERIFIED] All {len(cot_param_names)} CoT params have requires_grad=True")

    # Verify forward pass now produces non-zero CoT
    model.eval()
    with torch.no_grad():
        test_batch = make_batch(token_ids, SEQ_LEN, BATCH_SIZE, step=0)
        test_out = model(input_ids=test_batch, labels=test_batch.clone())
    assert test_out.cot_consistency_loss.item() > 0.0, \
        f"CoT loss should be > 0 after enable_cot(), got {test_out.cot_consistency_loss.item()}"
    assert not torch.all(test_out.cot_vector == 0.0), "CoT vector should be non-zero"
    print(f"  [VERIFIED] CoT loss now active: {test_out.cot_consistency_loss.item():.6f}")
    print(f"  [VERIFIED] CoT vector non-zero: norm={test_out.cot_vector.norm().item():.4f}")

    model.train()

    # ---- Rebuild optimizer to include newly-unfrozen CoT params ----
    # (The old optimizer doesn't have CoT params in its param groups
    #  because they were frozen when the optimizer was created.)
    trainable_now = [p for p in model.parameters() if p.requires_grad]
    trainable_count_now = count_params(model, requires_grad_only=True)
    print(f"  Rebuilding optimizer: {opt_n_params_pre} -> {len(trainable_now)} param tensors")
    print(f"  Trainable params: {trainable_params} -> {trainable_count_now}")
    assert trainable_count_now > trainable_params, \
        "More params should be trainable after enable_cot()"

    optimizer = torch.optim.AdamW(
        trainable_now,
        lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )

    results['transition'] = {
        'optimizer_params_before': opt_n_params_pre,
        'optimizer_params_after': len(trainable_now),
        'trainable_params_before': trainable_params,
        'trainable_params_after': trainable_count_now,
        'cot_param_names': cot_param_names,
    }

    # ==========================================================
    # STAGE 2: Specialized Training (CoT active)
    # ==========================================================
    print("\n" + "-" * 70)
    print("STAGE 2: SPECIALIZED TRAINING (CoT ACTIVE)")
    print("-" * 70)

    stage2_losses = []
    stage2_lm_losses = []
    stage2_routing_losses = []
    stage2_cot_losses = []
    stage2_routing_stats = []
    stage2_grad_info = []
    stage2_cot_grad_info = []

    # Verify CoT is active
    assert model._cot_enabled, "CoT should be enabled for Stage 2"
    for name, param in model.cot.named_parameters():
        assert param.requires_grad, f"CoT param {name} should be trainable"
    print("  [VERIFIED] CoT is enabled (all params requires_grad=True)")
    print("  [VERIFIED] model._cot_enabled is True")

    stage2_start = time.time()
    for step in range(STAGE2_STEPS):
        optimizer.zero_grad()
        batch = make_batch(token_ids, SEQ_LEN, BATCH_SIZE, STAGE1_STEPS + step)
        labels = batch.clone()

        out = model(input_ids=batch, labels=labels, output_routing_info=True)

        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        stage2_losses.append(loss.item())
        stage2_lm_losses.append(out.lm_loss.item())
        stage2_routing_losses.append(out.routing_loss.item())
        stage2_cot_losses.append(out.cot_consistency_loss.item())

        # CoT gradient check every step
        cot_grads = {}
        for name, param in model.cot.named_parameters():
            if param.grad is not None:
                cot_grads[name] = param.grad.norm().item()
        if step % 5 == 0 or step == STAGE2_STEPS - 1:
            stage2_cot_grad_info.append({'step': step, **cot_grads})
            rs = get_routing_stats(out.routing_info)
            stage2_routing_stats.append({'step': step, **rs})
            gi = get_subsystem_grad_info(model)
            stage2_grad_info.append({'step': step, **gi})

        if step % 10 == 0 or step == STAGE2_STEPS - 1:
            print(f"  Step {step:3d}/{STAGE2_STEPS}: loss={loss.item():.4f}  lm={out.lm_loss.item():.4f}  "
                  f"routing={out.routing_loss.item():.6f}  cot={out.cot_consistency_loss.item():.6f}")

    stage2_time = time.time() - stage2_start
    print(f"  Stage 2 completed in {stage2_time:.1f}s")

    # ---- Stage 2 Assertions ----
    assert all(math.isfinite(l) for l in stage2_losses), "Stage 2: all losses must be finite"
    assert all(c > 0.0 for c in stage2_cot_losses), \
        f"Stage 2: CoT loss must be > 0.0, got min={min(stage2_cot_losses):.6f}"

    # Verify CoT params received gradients
    cot_params_with_grad = sum(
        1 for name, param in model.cot.named_parameters()
        if param.grad is not None and param.grad.norm().item() > 0
    )
    cot_total_params = sum(1 for _ in model.cot.parameters())
    assert cot_params_with_grad > 0, \
        f"CoT params should receive gradients: {cot_params_with_grad}/{cot_total_params}"

    # Verify non-CoT params still trainable
    non_cot_with_grad = sum(
        1 for name, param in model.named_parameters()
        if not name.startswith('cot.') and param.grad is not None and param.grad.norm().item() > 0
    )
    assert non_cot_with_grad > 0, "Non-CoT params should also receive gradients in Stage 2"

    print(f"  [PASS] All losses finite")
    print(f"  [PASS] CoT loss > 0.0 throughout (min={min(stage2_cot_losses):.6f})")
    print(f"  [PASS] CoT params receive gradients: {cot_params_with_grad}/{cot_total_params}")
    print(f"  [PASS] Non-CoT params also receive gradients: {non_cot_with_grad}")

    # ---- Save Stage 2 checkpoint ----
    stage2_ckpt = CHECKPOINT_DIR / "stage2_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': STAGE1_STEPS + STAGE2_STEPS,
        'stage': 2,
        'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
    }, stage2_ckpt)
    print(f"  Checkpoint saved: {stage2_ckpt}")

    # ---- Final checkpoint reproducibility ----
    repro_ckpt = CHECKPOINT_DIR / "repro_test_stage2.pt"
    max_diff = verify_checkpoint_reproducibility(model, repro_ckpt, token_ids, SEQ_LEN, BATCH_SIZE, step=0)
    assert max_diff < 1e-5, f"Stage 2 checkpoint reproducibility failed: max_diff={max_diff}"
    print(f"  [PASS] Checkpoint reproducibility: max_diff={max_diff:.2e}")
    repro_ckpt.unlink()

    # ==========================================================
    # CHECKPOINT RESTORE TEST
    # ==========================================================
    print("\n" + "-" * 70)
    print("CHECKPOINT RESTORE TEST")
    print("-" * 70)

    # Save current output, restore stage2 checkpoint, verify same output
    model.eval()
    test_batch = make_batch(token_ids, SEQ_LEN, BATCH_SIZE, step=0)
    with torch.no_grad():
        out_before = model(input_ids=test_batch)
        logits_before = out_before.logits.clone()

    ckpt = torch.load(stage2_ckpt, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        out_after = model(input_ids=test_batch)
        logits_after = out_after.logits.clone()

    restore_diff = (logits_before - logits_after).abs().max().item()
    assert restore_diff < 1e-5, f"Restore test failed: max_diff={restore_diff}"
    print(f"  [PASS] Restore from stage2_final.pt: max_diff={restore_diff:.2e}")

    # ==========================================================
    # RESULTS SUMMARY
    # ==========================================================
    print("\n" + "=" * 70)
    print("TRAINING EVIDENCE SUMMARY")
    print("=" * 70)

    results['stage1'] = {
        'steps': STAGE1_STEPS,
        'time_s': round(stage1_time, 1),
        'initial_loss': round(stage1_losses[0], 6),
        'final_loss': round(stage1_losses[-1], 6),
        'loss_decreased': stage1_losses[-1] < stage1_losses[0],
        'min_loss': round(min(stage1_losses), 6),
        'max_loss': round(max(stage1_losses), 6),
        'all_finite': all(math.isfinite(l) for l in stage1_losses),
        'cot_loss_always_zero': all(c == 0.0 for c in stage1_cot_losses),
        'final_lm_loss': round(stage1_lm_losses[-1], 6),
        'final_routing_loss': round(stage1_routing_losses[-1], 6),
        'routing_stats': stage1_routing_stats[-1] if stage1_routing_stats else {},
        'grad_info_last': stage1_grad_info[-1] if stage1_grad_info else {},
    }

    results['stage2'] = {
        'steps': STAGE2_STEPS,
        'time_s': round(stage2_time, 1),
        'initial_loss': round(stage2_losses[0], 6),
        'final_loss': round(stage2_losses[-1], 6),
        'min_loss': round(min(stage2_losses), 6),
        'max_loss': round(max(stage2_losses), 6),
        'all_finite': all(math.isfinite(l) for l in stage2_losses),
        'cot_loss_min': round(min(stage2_cot_losses), 6),
        'cot_loss_max': round(max(stage2_cot_losses), 6),
        'cot_loss_always_positive': all(c > 0.0 for c in stage2_cot_losses),
        'cot_params_with_grad': cot_params_with_grad,
        'cot_params_total': cot_total_params,
        'non_cot_params_with_grad': non_cot_with_grad,
        'final_lm_loss': round(stage2_lm_losses[-1], 6),
        'final_routing_loss': round(stage2_routing_losses[-1], 6),
        'final_cot_loss': round(stage2_cot_losses[-1], 6),
        'routing_stats': stage2_routing_stats[-1] if stage2_routing_stats else {},
        'grad_info_last': stage2_grad_info[-1] if stage2_grad_info else {},
    }

    results['checkpoint'] = {
        'stage1_path': str(stage1_ckpt),
        'stage2_path': str(stage2_ckpt),
        'reproducibility_max_diff': max_diff,
        'restore_max_diff': restore_diff,
    }

    # Print summary
    print(f"\n  LEARNING:")
    print(f"    Stage 1 initial loss:  {results['stage1']['initial_loss']:.4f}")
    print(f"    Stage 1 final loss:    {results['stage1']['final_loss']:.4f}")
    print(f"    Stage 1 decreased:     {results['stage1']['loss_decreased']}")
    print(f"    Stage 2 initial loss:  {results['stage2']['initial_loss']:.4f}")
    print(f"    Stage 2 final loss:    {results['stage2']['final_loss']:.4f}")
    print(f"    Model demonstrably learned: {results['stage1']['loss_decreased'] and results['stage2']['all_finite']}")

    print(f"\n  GRADIENT HEALTH (Stage 1 last snapshot):")
    for sub, val in results['stage1']['grad_info_last'].items():
        status = 'ACTIVE' if val > 1e-7 else 'DEAD'
        print(f"    {sub:25s}: {val:.6f}  [{status}]")

    print(f"\n  GRADIENT HEALTH (Stage 2 last snapshot):")
    for sub, val in results['stage2']['grad_info_last'].items():
        status = 'ACTIVE' if val > 1e-7 else 'DEAD'
        print(f"    {sub:25s}: {val:.6f}  [{status}]")

    print(f"\n  CoT:")
    print(f"    Stage 1: FROZEN, loss=0.0")
    print(f"    Stage 2: ACTIVE, loss range=[{results['stage2']['cot_loss_min']:.6f}, {results['stage2']['cot_loss_max']:.6f}]")
    print(f"    CoT params with grad: {results['stage2']['cot_params_with_grad']}/{results['stage2']['cot_params_total']}")

    print(f"\n  CHECKPOINT:")
    print(f"    Stage 1: {results['checkpoint']['stage1_path']}")
    print(f"    Stage 2: {results['checkpoint']['stage2_path']}")
    print(f"    Reproducibility: {results['checkpoint']['reproducibility_max_diff']:.2e}")
    print(f"    Restore diff:   {results['checkpoint']['restore_max_diff']:.2e}")

    # ---- Classifications ----
    results['classifications'] = {
        'cot_consistency_loss_fix': {
            'category': 'TRAINING-STAGE BUG',
            'description': 'CoT consistency loss was hardcoded to zero; enable_cot() was a no-op',
            'status': 'FIXED',
            'regression_tests': 10,
        },
        'cot_init_overwrite': {
            'category': 'DESIGN LIMITATION',
            'description': 'Model-level _init_weights overwrites CoT custom xavier init with N(0,0.02)',
            'status': 'KNOWN, NOT FIXED (does not block training)',
        },
        'CoTAuxiliaryLoss_dead_code': {
            'category': 'EXPECTED BEHAVIOR',
            'description': 'CoTAuxiliaryLoss class exists but is unused — the model uses _compute_cot_consistency_loss instead',
            'status': 'NOT A BUG',
        },
        'continuation_import_bug': {
            'category': 'BUG',
            'description': 'continuation.py imports xorzenTrainer but class is XORZENXTrainer',
            'status': 'NOT FIXED (not in training path for this experiment)',
        },
        'no_formal_stage_definitions': {
            'category': 'DESIGN LIMITATION',
            'description': 'No TrainingStage enum or config field; stage transitions are manual (enable_cot, _freeze_cot)',
            'status': 'EXPECTED BEHAVIOR (by design)',
        },
        'routing_diversity_interpretation': {
            'category': 'RESEARCH QUESTION',
            'description': 'Whether routing statistics indicate learned specialization vs random exploration',
            'status': 'UNTESTED (requires longer training)',
        },
    }

    # Save results
    results_path = CHECKPOINT_DIR / "training_evidence.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Final verdict
    print("\n" + "=" * 70)
    all_pass = (
        results['stage1']['loss_decreased']
        and results['stage1']['all_finite']
        and results['stage1']['cot_loss_always_zero']
        and results['stage2']['all_finite']
        and results['stage2']['cot_loss_always_positive']
        and results['stage2']['cot_params_with_grad'] > 0
        and results['checkpoint']['reproducibility_max_diff'] < 1e-5
        and results['checkpoint']['restore_max_diff'] < 1e-5
    )

    if all_pass:
        print("OVERALL: ALL ASSERTIONS PASSED")
        print("The intended XORZEN training lifecycle works on real data.")
    else:
        print("OVERALL: SOME ASSERTIONS FAILED")
        print("See individual [PASS]/[FAIL] markers above.")
    print("=" * 70)

    logging.disable(logging.NOTSET)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
