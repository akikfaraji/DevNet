# Phase 9 — 10M-scale Validation (Storage-Efficient)

**Date**: 2026-08-15 08:45:49
**Model size**: ModelSize.NANO_10M
**Batch**: 4  **Seq**: 64  **Steps**: 100
**LR**: 0.001  **Seed**: 42
**Gradient checkpointing**: True
**Corpus entropy**: 4.041 bits/token  (theoretical min loss = 2.8012)

## Reproducibility

- Git commit: `56b1d1014105d01c432456991e299a6850790ed1`
- Git branch: `main`
- Git status clean: False
- Torch: 2.13.0+cpu  NumPy: 2.1.3  Python: 3.12.13
- Command: `python scripts/02_phase9_10m_storage_efficient.py (MODEL_SIZE=ModelSize.NANO_10M, BATCH=4, SEQ=64, STEPS=100, LR=0.001, SEED=42, GRADIENT_CHECKPOINTING=True)`

## Disk Audit

| Metric | Before | After |
|---|---|---|
| Free space (GB) | 1.389 | 1.354 |
| xorzen_dev (MB) | 10.09 | 45.5 |
| checkpoints (MB) | 0.0 | 35.41 |
| logs (MB) | 0.07 | 0.07 |
| reports (MB) | 0.45 | 0.45 |

## Results

| Config | Params | val_loss | val_ppl | tok/s | FLOPs/tok (active) | weights MB | optim MB | ckpt MB | RSS Δ MB |
|---|---|---|---|---|---|---|---|---|---|
| dense_baseline | 5,882,016 | 6.7580 | 860.88 | 1987 | 10,984,960 | 22.44 | 44.88 | 11.682 | +125 |
| xorzen_routing_disabled | 5,962,297 | 6.6375 | 763.21 | 864 | 7,133,248 | 22.74 | 39.55 | 11.865 | +53 |
| xorzen_genuine_sparse | 5,962,297 | 6.7192 | 828.12 | 887 | 5,680,576 | 22.74 | 39.55 | 11.865 | +55 |

## Scientific Verdict

- **Question**: Does Xorzen achieve better quality per unit of compute than dense?
- val_loss — dense: 6.7580  xorzen_disabled: 6.6375  xorzen_sparse: 6.7192
- val_ppl — dense: 860.88  xorzen_disabled: 763.21  xorzen_sparse: 828.12
- FLOPs/token — dense: 10,984,960  xorzen_disabled: 7,133,248  xorzen_sparse: 5,680,576

### Sparse vs Dense
- loss_delta: -0.0388
- flops_delta: -5,304,384
- sparse_uses_fewer_flops: True
- sparse_achieves_lower_loss: True
- sparse_wins_on_quality_per_flop: False

### Disabled vs Sparse (does routing help quality?)
- loss_delta (sparse - disabled): +0.0816
- interpretation: sparse routing HURTS quality at fixed params (overhead > benefit at this scale)

## Memory Breakdown (per config)

### dense_baseline
- weights: 22.44 MB (23,528,064 bytes)
- gradients: 22.44 MB (23,528,064 bytes)
- optimizer state (AdamW m+v): 44.88 MB (47,056,480 bytes)
- buffers (non-trainable): 0.02 MB
- RSS delta: +125 MB (peak RSS 505 MB)
- fp16 checkpoint: 11.682 MB

### xorzen_routing_disabled
- weights: 22.74 MB (23,849,188 bytes)
- gradients: 19.77 MB (20,734,676 bytes)
- optimizer state (AdamW m+v): 39.55 MB (41,470,704 bytes)
- buffers (non-trainable): 0.05 MB
- RSS delta: +53 MB (peak RSS 558 MB)
- fp16 checkpoint: 11.865 MB

### xorzen_genuine_sparse
- weights: 22.74 MB (23,849,188 bytes)
- gradients: 19.77 MB (20,734,676 bytes)
- optimizer state (AdamW m+v): 39.55 MB (41,470,704 bytes)
- buffers (non-trainable): 0.01 MB
- RSS delta: +55 MB (peak RSS 614 MB)
- fp16 checkpoint: 11.865 MB
