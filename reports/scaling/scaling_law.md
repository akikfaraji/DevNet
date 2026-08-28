# Xorzen Scaling Law — v0.5 (Recovered from Codebase)

Generated: 2026-08-15 08:30:39

## Methodology

Xorzen scaling law — derived from real parameter counts of instantiated variants.

**Key finding:** PARAM_COUNT class-attribute labels are aspirational, not actual.
  At 1.3B label, actual is 413M — labels overstate by 58.6%.

## Scaling-Law Table

| Variant | H | L | E | K | P_total (label) | P_total (actual) | Label acc % | P_active/token | Active % | FLOPs/token (active) | Compute eff x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_tiny_23k | 8 | 1 | 1 | 1 | 37,824 | 35,880 | 94.86% | 32,011 | 89.22% | 192,066 | 1.121x |
| zero_1M | 64 | 3 | 2 | 1 | 1,077,503 | 1,008,015 | 93.55% | 766,936 | 76.08% | 4,601,616 | 1.314x |
| zero_10M | 192 | 6 | 8 | 2 | 10,970,548 | 7,776,313 | 70.88% | 5,141,288 | 66.11% | 30,847,728 | 1.513x |
| zero_50M | 256 | 10 | 43 | 2 | 50,399,371 | 17,139,264 | 34.01% | 9,324,797 | 54.41% | 55,948,782 | 1.838x |
| zero_277M | 512 | 13 | 64 | 2 | 277,000,335 | 78,324,760 | 28.28% | 41,877,684 | 53.47% | 251,266,104 | 1.87x |
| zero_500M | 640 | 16 | 69 | 2 | 500,000,083 | 165,015,584 | 33.0% | 88,799,800 | 53.81% | 532,798,800 | 1.858x |
| zero_1_3B | 896 | 24 | 64 | 2 | 1,000,000,886 | 391,447,491 | 39.14% | 171,506,868 | 43.81% | 1,029,041,208 | 2.282x |

## Formulas

- **P_total**: `V*H + L*(4H^2 + 2H*D_lr*4 + 3H*D_ssm + 2H*(H*M) + 2H*4) + Router + CoT + E*3*H*(H*M) + Merger + 2H`
- **P_active_per_token**: `V*H/L + L_avg*[k_path*(per_block_active) + K*3*H*(H*M)]  where L_avg=(max_depth+min_depth)/2, k_path=pathway_top_k`
- **FLOPs_per_token_active**: `6 * P_active_per_token  (forward only, dense-equivalent)`
- **FLOPs_total_training**: `18 * P_active_per_token * num_tokens  (forward + backward ≈ 3x forward)`
- **Chinchilla_equivalent**: `L(C) = E + A/N^alpha + B/D^beta  where N=P_active_per_token, D=tokens, C=6*N*D`

**Note:** P_total scales as O(L*H^2 + E*H^2*M + V*H).  P_active scales as O(L_avg*k_path*H^2 + K*H^2*M + V*H/L).

## Hypothesis: 12B Xorzen > 60B Dense

Average active ratio at scale (≥10M variants): **51.37%**

Projected 12B Xorzen: P_total=12,000,000,000, P_active/token=6,164,977,835, active_ratio=51.37%

60B Dense Chinchilla-optimal: tokens=1,200,000,000,000, compute=432,000,000,000,000,000,000,000 FLOPs


### Equal-compute scenario:
- 12B Xorzen trained on 11,678,874,105,797 tokens
- Predicted L(60B dense) = 2.4186
- Predicted L(12B Xorzen) = 2.2633
- Xorzen wins: **True**

### Conditions for 12B Xorzen > 60B Dense:

- 1. Xorzen must achieve ≥6.5% active ratio at 12B scale (matches ≥50M variants)
- 2. Router must make GOOD routing decisions (not collapse) — quality of sparse activation matters
- 3. Training data D must be ≫ Chinchilla-optimal for 60B (since P_active_12B < N_60B,
-    we need D_12B ≥ D_60B * (N_60B / P_active_12B)^(alpha/beta) to compensate for the
-    smaller active-param count, OR rely on routing quality to extract more from each param)
- 4. The HASS pathway diversity (Local+LowRank+SSM) must contribute genuine complementary
-    signal — Phase 14 ablation found LowRank was harmful at tiny scale; this might reverse
-    at 12B scale where LowRank's global context becomes valuable
- 5. The MoE expert specialization must be real (experts learn different things). At tiny
-    scale, ShardedExpertFabric in test_mode uses a single dummy expert — production mode
-    with E=64+ experts needs to be validated for genuine specialization

**Caveat:** The 12B > 60B claim is a HYPOTHESIS, not a proven result. Validation requires actually training at 12B scale (out of scope for this environment). The 10M-scale validation in this run tests the WEAKER claim: 'does sparse routing help quality per FLOP at small scale?'


## Per-Variant Component Breakdown

### zero_tiny_23k
- Actual total: 35,880
- Predicted (test mode): 35,424.0  (error -1.3%)
- Predicted (full experts): 35,424.0  (error -1.3%)
- Component breakdown:
  - router: 20,456
  - hass_low_rank: 7,704
  - cot: 4,197
  - hass_ssm: 808
  - moe_fabric: 768
  - merger: 635
  - hass_ffn: 632
  - embeddings: 320
  - hass_local: 320
  - hass_layernorm: 32
  - final_norm: 8

### zero_1M
- Actual total: 1,008,015
- Predicted (test mode): 985,860.0  (error -2.2%)
- Predicted (full experts): 1,035,012.0  (error +2.7%)
- Component breakdown:
  - embeddings: 441,152
  - hass_low_rank: 152,640
  - hass_ffn: 101,184
  - cot: 99,873
  - hass_local: 50,304
  - moe_fabric: 49,152
  - hass_ssm: 38,880
  - router: 38,475
  - merger: 35,523
  - hass_layernorm: 768
  - final_norm: 64

### zero_10M
- Actual total: 7,776,313
- Predicted (test mode): 7,549,732.0  (error -2.9%)
- Predicted (full experts): 10,646,308.0  (error +36.9%)
- Component breakdown:
  - embeddings: 2,018,304
  - hass_ffn: 1,786,752
  - hass_low_rank: 897,408
  - hass_local: 890,496
  - cot: 766,049
  - hass_ssm: 527,040
  - moe_fabric: 442,368
  - merger: 315,459
  - router: 127,637
  - hass_layernorm: 4,608
  - final_norm: 192

### zero_50M
- Actual total: 17,139,264
- Predicted (test mode): 16,647,940.0  (error -2.9%)
- Predicted (full experts): 49,678,084.0  (error +189.8%)
- Component breakdown:
  - hass_ffn: 5,281,280
  - embeddings: 2,822,144
  - hass_local: 2,634,240
  - hass_low_rank: 1,989,120
  - hass_ssm: 1,498,560
  - cot: 1,332,609
  - moe_fabric: 786,432
  - merger: 559,875
  - router: 224,508
  - hass_layernorm: 10,240
  - final_norm: 256

### zero_277M
- Actual total: 78,324,760
- Predicted (test mode): 77,480,612.0  (error -1.1%)
- Predicted (full experts): 275,661,476.0  (error +251.9%)
- Component breakdown:
  - hass_ffn: 27,362,816
  - embeddings: 17,880,064
  - hass_local: 13,661,440
  - hass_ssm: 7,302,880
  - hass_low_rank: 5,151,744
  - moe_fabric: 3,145,728
  - cot: 1,857,409
  - merger: 1,644,035
  - router: 291,508
  - hass_layernorm: 26,624
  - final_norm: 512

### zero_500M
- Actual total: 165,015,584
- Predicted (test mode): 158,339,780.0  (error -4.0%)
- Predicted (full experts): 492,573,380.0  (error +198.5%)
- Component breakdown:
  - hass_ffn: 52,582,400
  - embeddings: 46,563,200
  - hass_local: 26,260,480
  - hass_ssm: 13,856,256
  - cot: 8,000,577
  - hass_low_rank: 7,919,616
  - moe_fabric: 4,915,200
  - merger: 3,488,643
  - router: 1,387,612
  - hass_layernorm: 40,960
  - final_norm: 640

### zero_1_3B
- Actual total: 391,447,491
- Predicted (test mode): 381,295,204.0  (error -2.6%)
- Predicted (full experts): 988,224,100.0  (error +152.4%)
- Component breakdown:
  - hass_ffn: 154,463,232
  - hass_local: 77,167,104
  - embeddings: 68,270,720
  - hass_ssm: 40,107,264
  - hass_low_rank: 16,616,448
  - cot: 15,558,849
  - moe_fabric: 9,633,792
  - merger: 6,833,795
  - router: 2,709,375
  - hass_layernorm: 86,016
  - final_norm: 896
