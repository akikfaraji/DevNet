# XORZEN zero_277M Google Colab Training Notebook

## Overview

This notebook (`XORZEN_zero_277M_Colab.ipynb`) performs a **real pre-training run** of the
XORZEN zero_277M model (277M parameters, ~26M active per token) using the actual XORZEN
framework installed from GitHub.

**This is NOT a demo.** Every cell performs real computation. The model, tokenizer,
optimizer, and training loop all come from the XORZEN codebase.

## How to Use

### 1. Open in Google Colab

Upload `XORZEN_zero_277M_Colab.ipynb` to [Google Colab](https://colab.research.google.com/),
or open it directly from GitHub.

### 2. Select GPU Runtime

Go to **Runtime → Change runtime type** and select:
- **T4 GPU** (16 GB VRAM) — minimum, training will be slow
- **A100 GPU** (40 GB VRAM) — recommended for practical training speed
- **V100 GPU** (16 GB VRAM) — acceptable

### 3. Edit Configuration (Cell 1)

The first code cell contains all configurable parameters:
- `BATCH_SIZE` — per-GPU batch size (default: 4)
- `MAX_TRAINING_STEPS` — total optimizer steps (default: 10,000)
- `MIXED_PRECISION` — "bf16", "fp16", or "fp32" (default: "bf16")
- `MOUNT_GOOGLE_DRIVE` — enable persistent checkpoint storage (default: True)

### 4. Run All Cells

Execute cells sequentially. The notebook will:
1. Detect hardware and report any limitations
2. Install XORZEN from GitHub
3. Download TinyStories dataset (CC-BY-4.0)
4. Load the XORZEN AGI tokenizer (65K vocab BPE)
5. Tokenize and pack the dataset
6. Instantiate the actual zero_277M model
7. Run pre-flight sanity tests
8. Benchmark throughput
9. Train with real gradient updates
10. Validate, checkpoint, and generate text
11. Export a complete release artifact

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8 GB   | 16+ GB      |
| System RAM| 12 GB  | 25+ GB      |
| Disk      | 10 GB  | 20+ GB      |

If VRAM is insufficient, the notebook will report the limitation and suggest
using gradient accumulation (already default: `GRADIENT_ACCUMULATION_STEPS=8`).

**The notebook will NOT silently reduce the model size.**

## XORZEN Installation

The notebook installs XORZEN from GitHub:
```
pip install git+https://github.com/akikfaraji/DevNet.git
```

If you have a pre-built wheel, set `XORZEN_WHEEL_PATH` in the configuration cell.

## Dataset

**TinyStories** (`roneneldan/TinyStories`) — a public dataset of ~2M short stories
for small language model training. License: CC-BY-4.0 (suitable for research;
not reviewed for commercial redistribution).

## Tokenizer

Uses the **xorzen_agi_tokenizer_65k** — a 65,000-token BPE tokenizer shipped
with XORZEN, trained on Project Gutenberg text. Includes special tokens for
reasoning (`<|think|>`), tools, agentic markers, and multilingual support.

## Checkpointing

- **Google Drive**: If mounted, checkpoints save to `MyDrive/XORZEN/zero_277M/checkpoints/`
- **Local**: Fallback to `/content/xorzen_checkpoints/`
- **Resume**: Re-running the notebook auto-detects the latest checkpoint and resumes
- **Frequency**: Every 1,000 steps (configurable via `CHECKPOINT_INTERVAL`)

Checkpoint contents: model state dict, optimizer state, scheduler state,
global step, best validation loss, training metrics.

## How to Resume Training

1. Re-open the notebook in Colab
2. Mount Google Drive (same account)
3. Run all cells — the training cell will detect the latest checkpoint and resume

## Release Artifacts

After training, the notebook exports to `FINAL_DIR`:
- `model.pt` — model weights
- `config.json` — model architecture configuration
- `training_config.json` — training hyperparameters
- `training_metrics.json` — loss/LR/grad_norm per step
- `validation_metrics.json` — validation results
- `generation_samples.json` — text generation outputs
- `environment.txt` — runtime environment info
- `checksums.json` — SHA-256 hashes of key files
- `reproducibility_report.json` — full reproducibility info

## Model Architecture (zero_277M)

| Parameter         | Value          |
|-------------------|----------------|
| Total parameters  | 277,000,335    |
| Active per token  | ~26M (10%)     |
| Hidden size       | 512            |
| Layers            | 13             |
| Attention heads   | 16             |
| Context length    | 1024           |
| MoE experts       | 64 (disk-sharded) |
| Top-K experts     | 2              |
| Width choices     | (256, 512)     |
| Depth range       | 4–13 (adaptive)|
| CoT dim           | 64 × 6 components |

## Engineering Guarantees

The notebook does NOT:
- Invent XORZEN APIs (all calls verified against source)
- Replace zero_277M with a fake model
- Silently reduce model size
- Use CPU when GPU was requested
- Skip failed tests
- Fabricate benchmark numbers
- Hide CUDA OOM errors
- Use synthetic data for training
- Claim training succeeded if it didn't complete

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `BATCH_SIZE` to 1–2, increase `GRADIENT_ACCUMULATION_STEPS` |
| Slow training | Use A100 GPU, enable `MIXED_PRECISION="bf16"` |
| Drive mount fails | Set `MOUNT_GOOGLE_DRIVE=False`, use local checkpoints |
| Tokenizer not found | Verify XORZEN installed correctly; check `xorzen.list_pretrained()` |
| NaN loss | Reduce `LEARNING_RATE` (try 1e-4), check data quality |

## License

XORZEN is under the XORZENX Proprietary License. This notebook is provided
for research and experimental purposes.
