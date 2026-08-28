# v0.5 Phase 11 — Real-Data 10M-Scale Validation

Generated: 2026-08-15 10:44:53

## Dataset & Tokenizer

- **Dataset**: TinyStories (HuggingFace: roneneldan/TinyStories)
- **Tokenizer**: zero_bpe_10k (vocab=10000)
- **Vocab size**: 10,000
- **Sequence length**: 128
- **Train tokens available**: 10,989,344
- **Val tokens available**: 427,320
- **Tokens trained per model**: 614,400 (600 steps × 8 batch × 128 seq)

## Exact Model Configs

### Dense baseline
- H=192, L=6, heads=8, ffn_hidden=768 (4×H), tied LM head, no MoE

### Xorzen NANO_10M (routing disabled)
- H=192, L=6, E=8, K=2 (forced all-active), pathway_top_k=3, min_depth=max_depth=6
- All structural pathways + 1 dummy expert materialized (test_mode)

### Xorzen NANO_10M (genuine sparse)
- H=192, L=6, E=8, K=2, pathway_top_k=2, min_depth=2, max_depth=6, width_choices=[96,192]
- Same architecture as above; only routing behavior differs

## Training Configuration

- Optimizer: AdamW(betas=(0.9,0.95))
- LR: 0.0003 (cosine_warmup, warmup=60)
- Weight decay: 0.01
- Grad clip: 1.0
- Seed: 42

## Results Table (equal training tokens)

| Metric | dense_baseline | xorzen_routing_disabled | xorzen_genuine_sparse |
|---|---:|---:|---:|
| P_full | 4,614,144 | 10,799,161 | 10,799,161 |
| P_resident (state_dict) | 4,614,144 | 7,702,585 | 7,702,585 |
| P_active/token | 4,614,144 | 6,540,364 | 5,141,288 |
| Active ratio (P_active/P_full) | 100.00% | 60.56% | 47.61% |
| Tokens trained | 614,400 | 614,400 | 614,400 |
| Train loss (initial) | 9.2451 | 9.2146 | 9.2465 |
| Train loss (final) | 4.2069 | 2.7791 | 3.3126 |
| **Val loss (final)** | **4.2651** | **3.3183** | **3.3202** |
| **Val perplexity** | **71.17** | **27.61** | **27.67** |
| Mean step time | 383ms | 624ms | 745ms |
| Tokens/sec | 2692 | 1647 | 1380 |
| Total training time | 258.0s | 419.2s | 502.6s |
| Active FLOPs/token | 9,738,624 | 21,682,560 | 9,775,488 |
| Total training FLOPs (fwd+bwd) | 17,950,231,756,800 | 39,965,294,592,000 | 18,018,179,481,600 |
| Peak RSS (MB) | 898 | 1128 | 1201 |
| Grad norm (mean/max) | 1.38/3.19 | 4.90/19.47 | 4.08/25.61 |
| Health: NaN/Inf steps | 0 | 0 | 0 |

## Routing Statistics (Xorzen genuine sparse, final eval)

- Active experts: 8 / 8 declared (100% utilization)
- Expert load entropy: 2.0781 bits
- Max expert load share: 0.1377 (lower = more balanced)
- Expert load balance CV: 0.0515
- Pathway entropy: -0.0000 bits (max=ln(3)=1.0986)
- Pathway distribution: [0.0, 0.0, 1.0]

## Equal-FLOPs Comparison (analytical)

- Dense model total training FLOPs: 17,950,231,756,800
- Tokens each model COULD have trained on this budget:
  - dense: 614,400 tokens
  - xorzen_routing_disabled: 275,954 tokens
  - xorzen_genuine_sparse: 612,083 tokens
- Note: in this run all three models trained on the SAME number of tokens. The equal-FLOPs comparison shows that sparse Xorzen could have trained on ~1.0× more tokens for the same compute budget.

## Verdicts

### Sparse vs Dense (equal tokens)
- Val loss delta: -0.9448 (sparse wins)
- FLOPs/token delta: +36,864 (sparse uses more compute)
- Quality per FLOP: sparse_routing_helps_quality_per_flop

### Sparse vs Disabled (does routing itself help quality?)
- Val loss delta: +0.0020
- Interpretation: sparse_routing_HURTS_quality_at_fixed_params_at_10M_scale

### Routing collapse check: **partial_collapse**
- Active experts: 8 / 8 (100% utilization)
- Path entropy: -0.0000 (max possible: 1.0986)
- Max expert load share: 0.1377

### Silent fallback check: **genuinely_sparse**
- FLOPs reduction vs dense-equivalent: 54.9%

## Scaling-Law Trend on Real Data

- Question: Does the sparse-vs-dense trend from the scaling-law table survive real data?
- Sparse P_full: 10,799,161
- Sparse P_active: 5,141,288 (47.6% of P_full)
- Dense P_full: 4,614,144 (100% active)
- Sparse compute efficiency: 1.00× fewer FLOPs/token
- Sparse achieves lower loss at lower compute: False
- **Trend survives real data: False**

## Remaining Problems

- Routing showed partial_collapse — only 8 of 8 experts active at end.
- Sparse-vs-dense scaling-law trend does NOT survive real data at 10M scale.
- Sparse routing HURTS quality vs routing-disabled by 0.0020 — adaptive routing overhead > benefit at 10M scale.
- This experiment does NOT validate the 12B-vs-60B hypothesis — it only tests the weaker claim that sparse routing helps quality per FLOP at 10M scale.
- Equal-FLOPs comparison is analytical only; would require re-running each model on a different number of tokens to give empirical equal-FLOPs curves.

## Reproducibility

- Seed: 42
- Total training time: 1179.7s (19.7min)
- Checkpoints: /home/z/my-project/xorzen_dev/checkpoints/phase11
- Tokenized TinyStories cache: /home/z/my-project/xorzen_dev/data/tinystories_tokens_uint16.npy
- Full results JSON: /home/z/my-project/xorzen_dev/reports/v05/phase11_real_data_validation.json
