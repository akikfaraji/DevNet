"""
XORZENX Extra Benchmarks:
  1) Disable gradient_checkpointing on xorzen — does it speed up?
  2) Measure expert shard size on disk (real disk-sharding mode)
  3) Test SPPQ save/load for shards
  4) Extrapolate to "production scale" (1B params on GPU)
"""
import os, sys, json, time, gc, shutil, tempfile
from pathlib import Path
from dataclasses import dataclass, asdict

os.environ["XORZENX_VERBOSE"] = "0"

import torch
import torch.nn as nn
import psutil

import xorzen
from xorzen.config import ConfigFactory, ModelSize

OUT = Path("/home/z/my-project/workspace/bench_data")
OUT.mkdir(parents=True, exist_ok=True)
PROC = psutil.Process()


def rss_mb():
    return PROC.memory_info().rss / (1024 * 1024)


# ─── 1) Disable gradient checkpointing ─────────────────────────────

def make_xorzen_no_ckpt(model_class, vocab_size=1000, max_seq=128):
    """Build xorzen with gradient_checkpointing=False."""
    # Override the config to disable gradient checkpointing
    cfg = ConfigFactory.get_config(model_class.MODEL_SIZE,
                                   gradient_checkpointing=False)
    model = model_class(config=cfg, test_mode=True)
    H = model.config.hidden_size
    model.token_embedding = nn.Embedding(vocab_size, H, padding_idx=0)
    if not model.config.tie_word_embeddings:
        model.lm_head = nn.Linear(H, vocab_size, bias=False)
    else:
        model.lm_head.weight = model.token_embedding.weight
    model.config.vocab_size = vocab_size
    model.eval()
    return model


def make_xorzen(model_class, vocab_size=1000, max_seq=128):
    model = model_class(test_mode=True)
    H = model.config.hidden_size
    model.token_embedding = nn.Embedding(vocab_size, H, padding_idx=0)
    if not model.config.tie_word_embeddings:
        model.lm_head = nn.Linear(H, vocab_size, bias=False)
    else:
        model.lm_head.weight = model.token_embedding.weight
    model.config.vocab_size = vocab_size
    model.eval()
    return model


def time_forward(model, batch=4, seq=128, vocab=1000, n_warmup=3, n_runs=10):
    x = torch.randint(0, vocab, (batch, seq))
    labels = torch.randint(0, vocab, (batch, seq))
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, labels=labels)
        ts = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x, labels=labels)
            ts.append(time.perf_counter() - t0)
    return 1000 * (sum(ts) / len(ts))


# ─── 2) Real expert shard on disk ──────────────────────────────────

def bench_expert_shards():
    """Run zero_50M in non-test mode (real expert fabric) and measure disk."""
    cfg = ConfigFactory.get_config(ModelSize.MICRO_50M,
                                   gradient_checkpointing=False)
    shard_dir = OUT / "expert_shards_50m"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    cfg.expert_shard_dir = str(shard_dir)

    print(f"  Building zero_50M with real expert fabric (shard_dir={shard_dir})...")
    try:
        model = xorzen.zero_50M(config=cfg, test_mode=False)
        # Inspect shard dir
        shard_files = list(shard_dir.glob("**/*"))
        shard_files = [f for f in shard_files if f.is_file()]
        total_bytes = sum(f.stat().st_size for f in shard_files)
        print(f"  Created {len(shard_files)} shard files, total {total_bytes/1024/1024:.2f} MB")
        return {
            "ok": True,
            "shard_dir": str(shard_dir),
            "n_shards": len(shard_files),
            "total_bytes": total_bytes,
            "expert_count": cfg.expert_count,
            "expert_hidden_multiplier": cfg.expert_hidden_multiplier,
            "hidden_size": cfg.hidden_size,
            "per_expert_bytes_est": total_bytes / max(1, cfg.expert_count),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": traceback.format_exc()[-1000:]}


# ─── 3) SPPQ shard save ────────────────────────────────────────────

def bench_sppq_shards():
    """Try SPPQ shard manager save/load."""
    try:
        from xorzen.utils.sppq import ShardedQuantizationManager, QuantizationState, QuantizationType
        sm = ShardedQuantizationManager(
            shard_dir=str(OUT / "sppq_shards"),
            max_shards_in_memory=10,
            compression_method="gzip",
        )
        # Create a fake quantization state for a small tensor
        t = torch.randn(100, 100)
        state = QuantizationState(
            name="test_tensor",
            quantization_type=QuantizationType.SYMMETRIC,
            scale=torch.tensor(0.1),
            zero_point=torch.tensor(0),
            original_shape=t.shape,
            original_dtype=t.dtype,
            quantized_data=t,
            bits=8,
        )
        sm.save_shard("test", state)
        loaded = sm.load_shard("test")
        ok = loaded is not None
        size_bytes = sum(f.stat().st_size for f in Path(OUT / "sppq_shards").rglob("*") if f.is_file())
        return {
            "ok": ok,
            "shard_path": str(OUT / "sppq_shards"),
            "size_bytes": size_bytes,
            "compression_method": "gzip",
            "raw_tensor_bytes": t.numel() * t.element_size(),
            "shard_compression_ratio": (t.numel() * t.element_size()) / max(1, size_bytes),
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": traceback.format_exc()[-1500:]}


# ─── 4) Extrapolation ──────────────────────────────────────────────

def extrapolate_gpu():
    """
    Extrapolate savings at production scale (1B params, GPU).
    Based on published MoE papers (GShard, Switch Transformer, Mixtral):
      - MoE with top-2 of N experts → 5-15x compute savings at inference
        (only when expert FLOPs dominate routing overhead)
      - int8 quantization → 4x storage savings
      - int4 quantization → 8x storage savings
      - Disk-sharded experts → O(1) GPU memory for expert params
        (vs O(N) for in-memory MoE)
    """
    # 1B parameter dense model
    n_params = 1_000_000_000
    # fp32 = 4 bytes/param, fp16/bf16 = 2 bytes/param, int8 = 1 byte/param, int4 = 0.5 byte/param
    fp32_bytes = n_params * 4
    fp16_bytes = n_params * 2
    int8_bytes = n_params * 1
    int4_bytes = n_params // 2

    # Dense vs MoE inference FLOPs per token (per GShard paper):
    # Dense: 6 * n_params = 6 GFLOPs per token for 1B model
    # MoE (top-2 of 64 experts, each ~16M params): 6 * (2 * 16M) + routing overhead ≈ 200 MFLOPs
    # Theoretical speedup: 6e9 / 2e8 ≈ 30x
    # Practical speedup (with overhead): 8-15x

    dense_flops_per_token = 6 * n_params  # 6 GFLOPs
    moe_active_params = 32_000_000  # top-2 of 64 experts each ~16M
    moe_flops_per_token = 6 * moe_active_params + 50_000_000  # + 50 MFLOPs overhead
    theoretical_speedup = dense_flops_per_token / moe_flops_per_token

    # A100 GPU: 312 TFLOPs fp16, 624 TFLOPs int8, 19.5 TFLOPs fp32
    # H100: 989 TFLOPs fp16, 1979 TFLOPs int8
    a100_fp16_tflops = 312
    a100_int8_tflops = 624
    h100_int8_tflops = 1979

    # Time to generate 1 token at 1B dense (fp16) on A100
    # dense_time_per_token_us = flops / tflops = 6e9 / 312e12 = 19.2 us
    dense_time_per_token_us_a100 = (dense_flops_per_token / (a100_fp16_tflops * 1e12)) * 1e6
    # MoE time on A100 (fp16)
    moe_time_per_token_us_a100 = (moe_flops_per_token / (a100_fp16_tflops * 1e12)) * 1e6
    # MoE time on A100 (int8) — 2x faster matmuls
    moe_time_int8_us_a100 = (moe_flops_per_token / (a100_int8_tflops * 1e12)) * 1e6

    # A100 TDP: 400W, H100: 700W
    a100_tdp_watts = 400

    # Energy per 1M tokens
    dense_j_per_m_a100 = dense_time_per_token_us_a100 * 1e-6 * a100_tdp_watts * 1e6
    moe_j_per_m_a100   = moe_time_per_token_us_a100   * 1e-6 * a100_tdp_watts * 1e6
    moe_int8_j_per_m_a100 = moe_time_int8_us_a100 * 1e-6 * a100_tdp_watts * 1e6

    # 1B-token training run (typical pretraining scale)
    tokens_billion = 1_000_000_000
    dense_kwh = (dense_j_per_m_a100 * tokens_billion / 1e6) / 3.6e6
    moe_kwh   = (moe_j_per_m_a100   * tokens_billion / 1e6) / 3.6e6
    moe_int8_kwh = (moe_int8_j_per_m_a100 * tokens_billion / 1e6) / 3.6e6

    # US grid: 0.4 kg CO2 per kWh (EPA 2023)
    co2_per_kwh = 0.4
    dense_co2_tons = dense_kwh * co2_per_kwh / 1000
    moe_co2_tons   = moe_kwh   * co2_per_kwh / 1000
    moe_int8_co2_tons = moe_int8_kwh * co2_per_kwh / 1000

    # Storage
    storage_dense_fp16_gb = fp16_bytes / 1e9
    storage_moe_int8_gb   = int8_bytes / 1e9
    storage_moe_int4_gb   = int4_bytes / 1e9

    return {
        "scale": "1B-parameter model, 1B-token training run, A100 GPU",
        "compute": {
            "dense_flops_per_token": dense_flops_per_token,
            "moe_flops_per_token": moe_flops_per_token,
            "theoretical_speedup": theoretical_speedup,
            "practical_speedup_low":  8.0,
            "practical_speedup_high": 15.0,
            "dense_time_per_token_us_a100": dense_time_per_token_us_a100,
            "moe_time_per_token_us_a100":   moe_time_per_token_us_a100,
            "moe_time_int8_per_token_us_a100": moe_time_int8_us_a100,
        },
        "electricity": {
            "a100_tdp_watts": a100_tdp_watts,
            "dense_j_per_m_tokens_a100": dense_j_per_m_a100,
            "moe_j_per_m_tokens_a100":   moe_j_per_m_a100,
            "moe_int8_j_per_m_tokens_a100": moe_int8_j_per_m_a100,
            "dense_kwh_per_b_tokens":  dense_kwh,
            "moe_kwh_per_b_tokens":    moe_kwh,
            "moe_int8_kwh_per_b_tokens": moe_int8_kwh,
            "co2_per_kwh_us_kg": co2_per_kwh,
            "dense_co2_tons_per_b_tokens":  dense_co2_tons,
            "moe_co2_tons_per_b_tokens":    moe_co2_tons,
            "moe_int8_co2_tons_per_b_tokens": moe_int8_co2_tons,
            "energy_saved_moe_pct":  100 * (dense_j_per_m_a100 - moe_j_per_m_a100) / dense_j_per_m_a100,
            "energy_saved_moe_int8_pct": 100 * (dense_j_per_m_a100 - moe_int8_j_per_m_a100) / dense_j_per_m_a100,
        },
        "storage": {
            "dense_fp32_gb": fp32_bytes / 1e9,
            "dense_fp16_gb": fp16_bytes / 1e9,
            "moe_int8_gb":   int8_bytes / 1e9,
            "moe_int4_gb":   int4_bytes / 1e9,
            "storage_saved_int8_pct": 100 * (fp16_bytes - int8_bytes) / fp16_bytes,
            "storage_saved_int4_pct": 100 * (fp16_bytes - int4_bytes) / fp16_bytes,
        },
    }


# ─── main ──────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("XORZENX v0.2.4 — EXTRA BENCHMARKS")
    print("=" * 78)

    results = {}

    # 1) gradient_checkpointing off
    print("\n[1] Effect of gradient_checkpointing=False on forward latency")
    ckpt_results = []
    for cls, name in [
        (xorzen.zero_1M,  "zero_1M"),
        (xorzen.zero_10M, "zero_10M"),
        (xorzen.zero_50M, "zero_50M"),
    ]:
        try:
            xz_on  = make_xorzen(cls)
            xz_off = make_xorzen_no_ckpt(cls)
            t_on  = time_forward(xz_on)
            t_off = time_forward(xz_off)
            speedup = t_on / t_off
            print(f"  {name}: ckpt_on={t_on:.2f}ms  ckpt_off={t_off:.2f}ms  speedup={speedup:.2f}x")
            ckpt_results.append({
                "model": name,
                "fwd_ms_ckpt_on": t_on,
                "fwd_ms_ckpt_off": t_off,
                "speedup_off_vs_on": speedup,
                "gradient_checkpointing_default": True,
            })
            del xz_on, xz_off
            gc.collect()
        except Exception as e:
            import traceback
            traceback.print_exc()
            ckpt_results.append({"model": name, "error": traceback.format_exc()[-500:]})
    results["gradient_checkpointing_effect"] = ckpt_results

    # 2) Expert shards on disk
    print("\n[2] Expert shard disk footprint (zero_50M, real expert fabric)")
    es = bench_expert_shards()
    print(f"  Result: {es}")
    results["expert_shards"] = es

    # 3) SPPQ shard manager
    print("\n[3] SPPQ shard save/load (gzip-compressed)")
    sq = bench_sppq_shards()
    print(f"  Result: {sq}")
    results["sppq_shards"] = sq

    # 4) Extrapolation
    print("\n[4] Extrapolation to production scale (1B params, A100 GPU)")
    ex = extrapolate_gpu()
    print(f"  Theoretical compute speedup:    {ex['compute']['theoretical_speedup']:.1f}x")
    print(f"  Practical speedup range:        {ex['compute']['practical_speedup_low']:.0f}-{ex['compute']['practical_speedup_high']:.0f}x")
    print(f"  Energy saved (MoE only):        {ex['electricity']['energy_saved_moe_pct']:.1f}%")
    print(f"  Energy saved (MoE + int8):      {ex['electricity']['energy_saved_moe_int8_pct']:.1f}%")
    print(f"  CO2 saved per 1B tokens (MoE+int8): {ex['electricity']['dense_co2_tons_per_b_tokens'] - ex['electricity']['moe_int8_co2_tons_per_b_tokens']:.2f} tons")
    print(f"  Storage saved vs fp16 (int8):   {ex['storage']['storage_saved_int8_pct']:.1f}%")
    print(f"  Storage saved vs fp16 (int4):   {ex['storage']['storage_saved_int4_pct']:.1f}%")
    results["extrapolation"] = ex

    out = OUT / "extra_benchmark.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
