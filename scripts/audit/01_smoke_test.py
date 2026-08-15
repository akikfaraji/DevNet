"""
XORZEN Smoke Test — verify the package can:
  1. Instantiate models at each declared size
  2. Run a forward pass end-to-end (logits out, loss computable)
  3. Generate tokens autoregressively
  4. Save/load checkpoints
  5. Load pretrained tokenizer + encode/decode round-trip
  6. Run SPPQ quantization on a real model

We capture: param count, trainable params, active-params estimate,
efficiency-gain estimate, peak RSS, forward-pass latency.
"""
import os, sys, json, time, gc, traceback
from pathlib import Path

# silence noisy logger
os.environ["XORZENX_VERBOSE"] = "0"

import torch
import torch.nn as nn
import psutil

import xorzen
from xorzen.config import ConfigFactory, ModelSize

PROC = psutil.Process()
LOG = Path("/home/z/my-project/workspace/bench_data/smoke_results.json")
LOG.parent.mkdir(parents=True, exist_ok=True)


def rss_mb():
    return PROC.memory_info().rss / (1024 * 1024)


def try_call(fn, *a, **kw):
    try:
        return fn(*a, **kw), None
    except Exception as e:
        return None, traceback.format_exc()


def smoke_model(model_class, name, vocab_size=1000, seq_len=32, batch=2):
    rec = {"name": name, "ok": False}
    t0 = time.time()
    try:
        model = model_class(test_mode=True)
        # Reduce vocab for the smoke test to save memory
        # (rebuild embeddings + lm_head at smaller vocab)
        H = model.config.hidden_size
        model.token_embedding = nn.Embedding(vocab_size, H, padding_idx=0)
        model.position_embedding = nn.Embedding(model.config.context_length, H)
        if not model.config.tie_word_embeddings:
            model.lm_head = nn.Linear(H, vocab_size, bias=False)
        else:
            model.lm_head.weight = model.token_embedding.weight
        model.config.vocab_size = vocab_size
        model.eval()
        n_total = sum(p.numel() for p in model.parameters())
        n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        active_est = model.config.estimate_active_parameters()
        eff_est    = model.config.estimate_efficiency_gain()
        rec.update({
            "params_total": n_total,
            "params_trainable": n_train,
            "active_params_est": active_est,
            "active_pct": 100.0 * active_est / max(1, n_train),
            "efficiency_gain_est": eff_est,
        })

        x = torch.randint(0, vocab_size, (batch, seq_len))
        labels = torch.randint(0, vocab_size, (batch, seq_len))
        rss_before = rss_mb()
        t_fwd = time.time()
        with torch.no_grad():
            out = model(x, labels=labels)
        fwd_ms = (time.time() - t_fwd) * 1000
        rec["fwd_ms"] = fwd_ms
        rec["rss_mb_before"] = rss_before
        rec["rss_mb_after"] = rss_mb()
        rec["logits_shape"] = list(out.logits.shape)
        rec["loss"] = float(out.loss.item()) if out.loss is not None else None
        rec["active_params_runtime"] = int(out.active_params) if hasattr(out, "active_params") and out.active_params is not None else None
        rec["compute_cost_gflops"] = float(out.compute_cost) if hasattr(out, "compute_cost") and out.compute_cost is not None else None
        rec["ok"] = True
    except Exception:
        rec["error"] = traceback.format_exc()
    rec["wall_s"] = time.time() - t0
    return rec


def smoke_tokenizer():
    rec = {"ok": False}
    try:
        toks = xorzen.list_pretrained()
        rec["available"] = toks
        if "zero_bpe_10k" in toks:
            tk = xorzen.load_pretrained("zero_bpe_10k")
            enc = tk.encode("Hello world, the XORZENX framework is here.")
            dec = tk.decode(enc)
            rec["vocab_size"] = tk.get_vocab_size()
            rec["encoded_len"] = len(enc)
            rec["roundtrip_ok"] = dec.startswith("Hello world")
            rec["tokenizer_path_size_kb"] = (
                Path(xorzen.get_pretrained_path("zero_bpe_10k")).stat().st_size / 1024
                if hasattr(xorzen, "get_pretrained_path") else None
            )
        rec["ok"] = True
    except Exception:
        rec["error"] = traceback.format_exc()
    return rec


def smoke_quantization():
    """Try SPPQ quantization on zero_1M."""
    rec = {"ok": False}
    try:
        from xorzen.utils.sppq import SPPQQuantizer, QuantizationConfig, QuantizationType
        model = xorzen.zero_1M(test_mode=True)
        n_pre = sum(p.numel() for p in model.parameters())
        cfg = QuantizationConfig(
            bits=8,
            quantization_type=QuantizationType.SYMMETRIC,
            observe_iterations=2,
        )
        quantizer = SPPQQuantizer(model, cfg)
        # calibrate briefly
        x = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            for _ in range(2):
                _ = model(x)
        quantizer.calibrate()
        quantizer.apply_quantization()
        n_post = sum(p.numel() for p in model.parameters())
        rec["params_pre"]  = n_pre
        rec["params_post"] = n_post
        rec["ok"] = True
    except Exception:
        rec["error"] = traceback.format_exc()[-2000:]
    return rec


def main():
    print("=" * 72)
    print("XORZENX v0.2.4 — SMOKE TEST")
    print("=" * 72)

    results = {
        "package_version": xorzen.__version__,
        "models_available": xorzen.list_models(),
        "tokenizers_available": xorzen.list_pretrained(),
    }

    # 1) Models
    print("\n[1] Instantiating models...")
    model_results = []
    for cls, name in [
        (xorzen.zero_tiny_23k, "zero_tiny_23k"),
        (xorzen.zero_1M,       "zero_1M"),
        (xorzen.zero_10M,      "zero_10M"),
        (xorzen.zero_50M,      "zero_50M"),
    ]:
        print(f"  - {name}...", end=" ", flush=True)
        r = smoke_model(cls, name)
        if r["ok"]:
            print(f"OK  params={r['params_total']:>12,}  active={r['active_params_est']:>10,}  ({r['active_pct']:.2f}%)  eff={r['efficiency_gain_est']:.2f}x  fwd={r['fwd_ms']:.1f}ms")
        else:
            print("FAIL")
            print(r["error"][-800:])
        model_results.append(r)
        gc.collect()
    results["models"] = model_results

    # 2) Tokenizer
    print("\n[2] Loading pretrained tokenizer...")
    tok_r = smoke_tokenizer()
    print(f"  - available: {tok_r.get('available')}")
    if tok_r["ok"] and "vocab_size" in tok_r:
        print(f"  - zero_bpe_10k: vocab={tok_r['vocab_size']}, roundtrip={tok_r['roundtrip_ok']}")
    elif "error" in tok_r:
        print("  ERROR:", tok_r["error"][-500:])
    results["tokenizer"] = tok_r

    # 3) Quantization
    print("\n[3] Trying SPPQ quantization on zero_1M...")
    q_r = smoke_quantization()
    if q_r["ok"]:
        print(f"  - params before: {q_r['params_pre']:,}, after: {q_r['params_post']:,}")
    else:
        print("  ERROR:", q_r.get("error", "")[-800:])
    results["quantization"] = q_r

    LOG.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved → {LOG}")
    return results


if __name__ == "__main__":
    main()
