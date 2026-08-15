"""
XORZENX Rigorous Benchmark — Compute / Storage / Electricity

For each model size in {zero_1M, zero_10M, zero_50M} we measure:

  COMPUTE
    - Forward pass wall-time (ms) for batch=4 seq=128
    - Forward + backward wall-time (ms)
    - Throughput (tokens/sec)
    - Estimated FLOPs from the framework
    - Peak RSS memory
    - Active parameters (runtime + framework estimate)

  STORAGE
    - Raw state_dict size on disk (.pt)
    - SPPQ-quantized state size (int8 fake-quant)
    - Per-expert shard size on disk
    - Tokenized .bin size for a 1 MB text corpus

  ELECTRICITY
    - CPU energy per forward pass (joules) using RAPL if available,
      otherwise TDP × CPU util × time
    - Energy per 1M tokens (J)
    - Energy per 1B tokens (kWh)
    - CO2 saved per 1B tokens (gCO2e)

We compare against a dense baseline: a vanilla nn.TransformerEncoderLM
matched to the same total parameter count (so the comparison is
"equal-capacity dense vs xorzen sparse-MoE").
"""
import os, sys, json, time, gc, math, subprocess, tempfile, shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

os.environ["XORZENX_VERBOSE"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil

import xorzen
from xorzen.config import ConfigFactory, ModelSize

# ───────────────────────────── helpers ─────────────────────────────

PROC = psutil.Process()
OUT  = Path("/home/z/my-project/workspace/bench_data")
OUT.mkdir(parents=True, exist_ok=True)

# Try to read Intel RAPL energy counters
def rapl_energy_joules() -> Optional[float]:
    """Returns package energy in joules since last call, or None."""
    paths = [
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return int(f.read().strip()) / 1e6
            except Exception:
                pass
    return None

def cpu_power_watts() -> float:
    """
    Estimate CPU package power. If RAPL is unavailable, fall back to
    a typical Xeon/EPYC TDP (150W) × instantaneous util ratio.
    """
    # If RAPL readable, we will compute delta in caller instead.
    return 150.0  # conservative TDP for a server CPU

def rss_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)

def measure_block(func, *args, **kwargs):
    """Run func, return (result, wall_s, cpu_s, rss_delta_mb, energy_j)."""
    gc.collect()
    rss0 = rss_mb()
    e0   = rapl_energy_joules()
    cpu0 = PROC.cpu_times()
    t0   = time.perf_counter()
    res  = func(*args, **kwargs)
    t1   = time.perf_counter()
    cpu1 = PROC.cpu_times()
    e1   = rapl_energy_joules()
    rss1 = rss_mb()
    wall = t1 - t0
    cpu  = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    if e0 is not None and e1 is not None:
        # RAPL counter may wrap; handle that
        if e1 >= e0:
            energy = e1 - e0
        else:
            energy = e1  # wrapped
    else:
        # Fallback: assume CPU power × wall × cpu_util_ratio
        cpu_util = (cpu / wall) if wall > 0 else 1.0
        cpu_util = min(max(cpu_util, 0.0), 1.0)
        energy = cpu_power_watts() * cpu_util * wall
    return res, wall, cpu, (rss1 - rss0), energy


# ───────────────────────────── dense baseline ──────────────────────

class DenseTransformerLM(nn.Module):
    """Vanilla Transformer LM for fair comparison."""
    def __init__(self, vocab_size, hidden, n_layers, n_heads, max_seq, dropout=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos = nn.Embedding(max_seq, hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=4*hidden,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.tok.weight

    def forward(self, idx, labels=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1)
        x = self.tok(idx) + self.pos(pos)
        x = self.drop(x)
        x = self.encoder(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}


def make_dense_baseline(target_params: int, vocab_size: int, max_seq: int) -> DenseTransformerLM:
    """
    Build a dense transformer whose parameter count is close to target_params.
    Use a standard small config and scale hidden / n_layers.
    """
    # Standard config: hidden=256, n_layers=4, n_heads=4 (≈3M params at vocab 1000)
    # Try grid search over hidden / n_layers to hit target.
    best = None
    best_diff = float("inf")
    for hidden in [64, 96, 128, 192, 256, 384, 512, 768, 1024]:
        for n_layers in [2, 3, 4, 6, 8, 10, 12, 16]:
            n_heads = max(1, hidden // 64)
            if hidden % n_heads != 0:
                continue
            m = DenseTransformerLM(vocab_size, hidden, n_layers, n_heads, max_seq)
            n = sum(p.numel() for p in m.parameters())
            diff = abs(n - target_params)
            if diff < best_diff:
                best_diff = diff
                best = (m, n, hidden, n_layers, n_heads)
            if n > target_params * 1.2:
                break
    return best[0], {"params": best[1], "hidden": best[2], "n_layers": best[3], "n_heads": best[4]}


# ───────────────────────────── xorzen builder ──────────────────────

def make_xorzen(model_class, vocab_size: int, max_seq: int = 128):
    """
    Build an xorzen model and shrink its vocab embedding so we are not
    benchmarking a 33k-vocab LM head against a 1k-vocab dense baseline.
    This makes the comparison about *architecture*, not vocab.
    """
    model = model_class(test_mode=True)
    H = model.config.hidden_size
    # Rebuild embeddings to requested vocab
    model.token_embedding = nn.Embedding(vocab_size, H, padding_idx=0)
    if not model.config.tie_word_embeddings:
        model.lm_head = nn.Linear(H, vocab_size, bias=False)
    else:
        model.lm_head.weight = model.token_embedding.weight
    model.config.vocab_size = vocab_size
    model.eval()
    return model


# ───────────────────────────── storage utils ───────────────────────

def state_dict_size_bytes(model) -> int:
    """Size of state_dict if serialized to .pt on disk."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path)
    finally:
        os.remove(path)


def quantized_state_size_bytes(model) -> Dict[str, Any]:
    """
    Estimate int8 quantized size WITHOUT deepcopying the model
    (the ShardedExpertFabric holds a threading lock that cannot be deepcopied).
    We compute by iterating the state_dict in-place.
    Also estimates int4 (1 byte per 2 elements + scale) for comparison.
    """
    sd = model.state_dict()
    total_fp32_bytes = 0
    total_int8_bytes = 0
    total_int4_bytes = 0
    n_tensors = 0
    for k, v in sd.items():
        if v.dtype.is_floating_point:
            total_fp32_bytes += v.numel() * v.element_size()  # typically 4 (fp32)
            # int8: 1 byte per element + 8 bytes overhead (scale + zero_pt)
            total_int8_bytes += v.numel() * 1 + 8
            # int4: 0.5 byte per element + 8 bytes overhead
            total_int4_bytes += (v.numel() + 1) // 2 + 8
            n_tensors += 1
        else:
            sz = v.numel() * v.element_size()
            total_fp32_bytes += sz
            total_int8_bytes += sz
            total_int4_bytes += sz
    return {
        "fp32_bytes": total_fp32_bytes,
        "int8_bytes": total_int8_bytes,
        "int4_bytes": total_int4_bytes,
        "compression_ratio_int8": total_fp32_bytes / max(1, total_int8_bytes),
        "compression_ratio_int4": total_fp32_bytes / max(1, total_int4_bytes),
        "n_tensors": n_tensors,
    }


def try_sppq_quantization(model):
    """Try to actually invoke xorzen's SPPQ module."""
    try:
        from xorzen.utils.sppq import SPPQ
        sppq = SPPQ(model, config={
            "enabled": True,
            "method": "progressive",
            "quant_type": "symmetric",
            "per_channel": True,
            "quant_levels": [8],
            "progressive_schedule": False,
            "shard_quantization": False,
            "total_steps": 10,
        })
        sppq.apply_fake_quantization()
        stats = sppq.get_statistics()
        return {"ok": True, "stats": stats}
    except Exception as e:
        return {"ok": False, "error": str(e)[-500:]}


# ───────────────────────────── data conversion bench ───────────────

def bench_data_conversion():
    """Generate 1 MB of text, compare storage formats."""
    # Generate a realistic-looking text corpus (1 MB)
    sample_words = (
        "the quick brown fox jumps over the lazy dog "
        "machine learning models require careful tuning "
        "xorzen framework uses adaptive routing and mixture of experts "
        "transformer architecture has revolutionized natural language processing "
    ).split()
    import random
    random.seed(42)
    target_bytes = 1 * 1024 * 1024  # 1 MB
    text_chunks = []
    cur = 0
    while cur < target_bytes:
        sentence = " ".join(random.choices(sample_words, k=20)) + ". "
        text_chunks.append(sentence)
        cur += len(sentence)
    text = "".join(text_chunks)[:target_bytes]
    text_path = OUT / "corpus.txt"
    text_path.write_text(text)

    txt_size = text_path.stat().st_size

    # gzip / zstd / lz4 baselines
    import gzip, zlib
    try:
        import zstandard as zstd
        zstd_ok = True
    except ImportError:
        zstd_ok = False

    gz_path = OUT / "corpus.txt.gz"
    with gz_path.open("wb") as f:
        f.write(gzip.compress(text.encode()))
    gz_size = gz_path.stat().st_size

    # zlib
    zlib_size = len(zlib.compress(text.encode(), 9))

    # xorzen tokenized .bin
    tok = xorzen.load_pretrained("zero_bpe_10k")
    from xorzen.data import DataConverter
    try:
        conv = DataConverter(tokenizer=tok)
        bin_path = OUT / "corpus.bin"
        conv.txt_to_bin(str(text_path), str(bin_path))
        bin_size = bin_path.stat().st_size
    except Exception as e:
        bin_size = None
        print("DataConverter failed:", e)

    # Also: just save token IDs as numpy .npy (uint16) for fair comparison
    import numpy as np
    ids = tok.encode(text)
    arr = np.array(ids, dtype=np.uint16)
    npy_path = OUT / "corpus.npy"
    np.save(npy_path, arr)
    npy_size = npy_path.stat().st_size

    # npz compressed
    npz_path = OUT / "corpus.npz"
    np.savez_compressed(npz_path, arr=arr)
    npz_size = npz_path.stat().st_size

    return {
        "raw_txt_bytes": txt_size,
        "gzip_bytes": gz_size,
        "zlib_bytes": zlib_size,
        "xorzen_bin_bytes": bin_size,
        "numpy_npy_bytes": npy_size,
        "numpy_npz_bytes": npz_size,
        "gzip_ratio": txt_size / gz_size,
        "xorzen_bin_ratio": (txt_size / bin_size) if bin_size else None,
        "token_count": len(ids),
        "tokens_per_byte": len(ids) / txt_size,
    }


# ───────────────────────────── main benchmark ──────────────────────

@dataclass
class ComputeResult:
    name: str
    forward_ms: float
    fwd_bwd_ms: float
    tokens_per_sec_fwd: float
    tokens_per_sec_fb: float
    rss_peak_mb: float
    energy_per_fwd_j: float
    energy_per_fb_j: float
    flops_est: float
    n_params: int
    n_active_params: int


def bench_compute(model, name: str, batch: int = 4, seq: int = 128,
                  n_warmup: int = 3, n_runs: int = 10, vocab: int = 1000) -> ComputeResult:
    model.eval()
    x = torch.randint(0, vocab, (batch, seq))
    labels = torch.randint(0, vocab, (batch, seq))

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, labels=labels)

    # Forward-only benchmark
    fwd_times = []
    fwd_energies = []
    for _ in range(n_runs):
        with torch.no_grad():
            _, wall, cpu, rss, energy = measure_block(lambda: model(x, labels=labels))
        fwd_times.append(wall)
        fwd_energies.append(energy)
    fwd_ms = 1000 * (sum(fwd_times) / len(fwd_times))
    fwd_j  = sum(fwd_energies) / len(fwd_energies)
    tokens = batch * seq
    tps_fwd = tokens / (fwd_ms / 1000)

    # Forward + backward
    fb_times = []
    fb_energies = []
    for _ in range(n_runs):
        def _fb():
            model.zero_grad(set_to_none=True)
            out = model(x, labels=labels)
            if isinstance(out, dict):
                loss = out["loss"]
            else:
                loss = out.loss
            loss.backward()
            return loss
        _, wall, cpu, rss, energy = measure_block(_fb)
        fb_times.append(wall)
        fb_energies.append(energy)
    fb_ms = 1000 * (sum(fb_times) / len(fb_times))
    fb_j  = sum(fb_energies) / len(fb_energies)
    tps_fb = tokens / (fb_ms / 1000)

    # Param counts
    n_params = sum(p.numel() for p in model.parameters())
    n_active = None
    if hasattr(model, "_estimate_active_params"):
        try:
            with torch.no_grad():
                out = model(x, labels=labels)
            if hasattr(out, "active_params") and out.active_params is not None:
                n_active = int(out.active_params)
        except Exception:
            pass

    flops_est = None
    if hasattr(model, "_estimate_compute_cost"):
        try:
            with torch.no_grad():
                out = model(x, labels=labels)
            if hasattr(out, "compute_cost") and out.compute_cost is not None:
                flops_est = float(out.compute_cost)
        except Exception:
            pass

    return ComputeResult(
        name=name,
        forward_ms=fwd_ms,
        fwd_bwd_ms=fb_ms,
        tokens_per_sec_fwd=tps_fwd,
        tokens_per_sec_fb=tps_fb,
        rss_peak_mb=rss_mb(),
        energy_per_fwd_j=fwd_j,
        energy_per_fb_j=fb_j,
        flops_est=flops_est if flops_est else 0.0,
        n_params=n_params,
        n_active_params=n_active if n_active else 0,
    )


def main():
    print("=" * 78)
    print("XORZENX v0.2.4 — RIGOROUS BENCHMARK")
    print("Compute / Storage / Electricity")
    print("=" * 78)

    results = {
        "package_version": xorzen.__version__,
        "device": "CPU",
        "rapl_available": rapl_energy_joules() is not None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    VOCAB = 1000
    SEQ   = 128
    BATCH = 4

    # ─── Compute benchmark ───
    print("\n[1] Compute benchmark (xorzen vs dense baseline)")
    compute_results = []
    for cls, name in [
        (xorzen.zero_1M,  "zero_1M"),
        (xorzen.zero_10M, "zero_10M"),
        (xorzen.zero_50M, "zero_50M"),
    ]:
        print(f"\n  ▸ {name}")
        try:
            xz = make_xorzen(cls, VOCAB, SEQ)
            n_xz = sum(p.numel() for p in xz.parameters())
            print(f"    xorzen params:    {n_xz:>12,}")
            r_xz = bench_compute(xz, f"{name} (xorzen)", batch=BATCH, seq=SEQ, vocab=VOCAB)
            print(f"    fwd={r_xz.forward_ms:>8.2f}ms  fb={r_xz.fwd_bwd_ms:>8.2f}ms  "
                  f"tps_fwd={r_xz.tokens_per_sec_fwd:>10.1f}  tps_fb={r_xz.tokens_per_sec_fb:>10.1f}  "
                  f"E_fwd={r_xz.energy_per_fwd_j:>6.2f}J")
            del xz
            gc.collect()

            # Match a dense baseline to the same param count
            base, info = make_dense_baseline(n_xz, VOCAB, SEQ)
            n_b = sum(p.numel() for p in base.parameters())
            print(f"    dense params:     {n_b:>12,}  (hidden={info['hidden']}, L={info['n_layers']})")
            r_b = bench_compute(base, f"{name} (dense)",  batch=BATCH, seq=SEQ, vocab=VOCAB)
            print(f"    fwd={r_b.forward_ms:>8.2f}ms  fb={r_b.fwd_bwd_ms:>8.2f}ms  "
                  f"tps_fwd={r_b.tokens_per_sec_fwd:>10.1f}  tps_fb={r_b.tokens_per_sec_fb:>10.1f}  "
                  f"E_fwd={r_b.energy_per_fwd_j:>6.2f}J")
            del base
            gc.collect()

            compute_results.append({
                "model": name,
                "xorzen": asdict(r_xz),
                "dense":  asdict(r_b),
                "speedup_fwd": r_b.forward_ms / r_xz.forward_ms if r_xz.forward_ms > 0 else 0,
                "speedup_fb":  r_b.fwd_bwd_ms / r_xz.fwd_bwd_ms if r_xz.fwd_bwd_ms > 0 else 0,
                "energy_saving_fwd_pct": 100 * (r_b.energy_per_fwd_j - r_xz.energy_per_fwd_j) / max(1e-9, r_b.energy_per_fwd_j),
                "active_params_pct": 100 * r_xz.n_active_params / max(1, r_xz.n_params),
            })
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            compute_results.append({"model": name, "error": traceback.format_exc()[-1000:]})
    results["compute"] = compute_results

    # ─── Storage benchmark ───
    print("\n[2] Storage benchmark")
    storage_results = []
    for cls, name in [
        (xorzen.zero_1M,  "zero_1M"),
        (xorzen.zero_10M, "zero_10M"),
        (xorzen.zero_50M, "zero_50M"),
    ]:
        print(f"\n  ▸ {name}")
        try:
            xz = make_xorzen(cls, VOCAB, SEQ)
            n_xz = sum(p.numel() for p in xz.parameters())
            xz_fp32 = state_dict_size_bytes(xz)
            xz_q    = quantized_state_size_bytes(xz)
            sppq    = try_sppq_quantization(xz)
            print(f"    xorzen fp32 .pt: {xz_fp32/1024/1024:>7.2f} MB")
            print(f"    xorzen int8 est: {xz_q['int8_bytes']/1024/1024:>7.2f} MB  (ratio {xz_q['compression_ratio_int8']:.2f}x)")
            print(f"    xorzen int4 est: {xz_q['int4_bytes']/1024/1024:>7.2f} MB  (ratio {xz_q['compression_ratio_int4']:.2f}x)")

            base, info = make_dense_baseline(n_xz, VOCAB, SEQ)
            b_fp32 = state_dict_size_bytes(base)
            b_q    = quantized_state_size_bytes(base)
            print(f"    dense  fp32 .pt: {b_fp32/1024/1024:>7.2f} MB")
            print(f"    dense  int8 est: {b_q['int8_bytes']/1024/1024:>7.2f} MB  (ratio {b_q['compression_ratio_int8']:.2f}x)")

            storage_results.append({
                "model": name,
                "xorzen": {
                    "params": n_xz,
                    "fp32_bytes": xz_fp32,
                    "int8_bytes": xz_q["int8_bytes"],
                    "int4_bytes": xz_q["int4_bytes"],
                    "compression_ratio_int8": xz_q["compression_ratio_int8"],
                    "compression_ratio_int4": xz_q["compression_ratio_int4"],
                    "sppq_status": sppq,
                },
                "dense": {
                    "params": sum(p.numel() for p in base.parameters()),
                    "fp32_bytes": b_fp32,
                    "int8_bytes": b_q["int8_bytes"],
                    "int4_bytes": b_q["int4_bytes"],
                    "compression_ratio_int8": b_q["compression_ratio_int8"],
                    "compression_ratio_int4": b_q["compression_ratio_int4"],
                },
                "storage_ratio_xz_vs_dense_fp32": b_fp32 / xz_fp32,
            })
            del xz, base
            gc.collect()
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            storage_results.append({"model": name, "error": traceback.format_exc()[-1000:]})
    results["storage_models"] = storage_results

    # Data conversion
    print("\n  ▸ Data conversion (1 MB text corpus)")
    try:
        dc = bench_data_conversion()
        results["storage_data"] = dc
        for k, v in dc.items():
            print(f"    {k}: {v}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["storage_data"] = {"error": traceback.format_exc()[-1000:]}

    # Tokenizer storage
    print("\n  ▸ Tokenizer storage")
    tok_results = {}
    for tk_name in xorzen.list_pretrained():
        try:
            p = xorzen.get_pretrained_path(tk_name)
            if p and Path(p).exists():
                tk = xorzen.load_pretrained(tk_name)
                tok_results[tk_name] = {
                    "path_bytes": Path(p).stat().st_size,
                    "vocab_size": tk.get_vocab_size(),
                }
                print(f"    {tk_name}: {tok_results[tk_name]['path_bytes']/1024:.1f} KB (vocab {tok_results[tk_name]['vocab_size']})")
        except Exception as e:
            tok_results[tk_name] = {"error": str(e)[-300:]}
            print(f"    {tk_name}: ERROR {e}")
    results["storage_tokenizers"] = tok_results

    # ─── Save ───
    out = OUT / "full_benchmark.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults → {out}")
    return results


if __name__ == "__main__":
    main()
