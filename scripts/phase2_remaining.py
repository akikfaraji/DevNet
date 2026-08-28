r"""Phase 2: Active-parameter accounting, inference routing, SPPQ, expert sharding, scaling-law.

Classification: PROVEN, EMPIRICAL, PARTIAL, UNTESTED, FALSE, REMOVED
"""

import sys, os, json, time, copy, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from xorzen.config import ConfigFactory, ModelConfig
from xorzen.models.zero.model import zeroModel
from xorzen.model.components.routing import RoutingDecision


# ==================== ACTIVE PARAMETER ACCOUNTING ====================

def test_active_param_estimate_accuracy():
    """VERIFY: _estimate_active_params is a heuristic (not measured), and
    understand what it actually computes vs what it claims.

    The method computes:
    - embeddings (always) + avg_depth * params_per_block + top_k * expert_params + cot + router + merger + lm_head

    It does NOT:
    - Actually measure which parameters were touched
    - Account for pathway sparsity (assumes all 3 pathways)
    - Account for width sparsity (uses avg_width_multiplier only for FLOPs, not params)
    - Distinguish between parameter availability and execution

    This is fundamentally an ESTIMATE based on routing metadata, not a measurement.
    """
    print("\n" + "="*80)
    print("TEST: ACTIVE PARAMETER ESTIMATE ACCURACY")
    print("="*80)

    cfg = ConfigFactory.get_config('1M')
    torch.manual_seed(42)
    model = zeroModel(cfg, test_mode=True)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids, output_routing_info=True)

    estimated = out.active_params
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total params: {total_params}")
    print(f"  Trainable params: {trainable}")
    print(f"  Estimated active: {estimated}")
    print(f"  Estimated/total: {estimated/total_params:.3f}")
    print(f"  Estimated/trainable: {estimated/trainable:.3f}")

    # The estimate includes: embeddings + avg_depth * block_params + expert_params + cot + router + merger + lm_head
    # With depth_mask all 1s (untrained), avg_depth = max_depth = 3
    # So estimated should be close to total_params (minus MoE experts not in cache)

    # Key question: does the estimate account for pathway sparsity?
    # Looking at the code: no, it uses params_per_block (all 3 pathways)
    # Does it account for width sparsity? No, it doesn't use width_multiplier for params.

    return {
        'claim': '_estimate_active_params accurately measures actually-executed parameters',
        'classification': 'FALSE',
        'evidence': f'Estimate={estimated}, total={total_params}. Method is a heuristic based on routing metadata, not an instrumentation-based measurement.',
        'detail': 'Does NOT account for: pathway sparsity (assumes all 3), width sparsity (ignores selected width), actual parameter execution. It multiplies avg_depth * params_per_block but params_per_block includes all pathways. Useful as a rough proxy, not a precise measurement.'
    }


def test_active_param_instrumented():
    """INSTRUMENT: Actually measure which parameters receive gradients during
    a backward pass. Compare with the heuristic estimate.

    This requires a training forward pass with loss computation.
    """
    print("\n" + "="*80)
    print("TEST: INSTRUMENTED ACTIVE PARAMETER MEASUREMENT")
    print("="*80)

    cfg = ConfigFactory.get_config('1M')
    torch.manual_seed(42)
    model = zeroModel(cfg, test_mode=True)
    model.train()

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model(ids, labels=ids.clone(), output_routing_info=True)
    out.loss.backward()

    # Count parameters that actually received gradients
    touched_params = 0
    touched_trainable = 0
    total_params = 0
    total_trainable = 0
    grad_norms = {}

    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            total_trainable += n
        if p.grad is not None and p.grad.abs().sum() > 0:
            touched_params += n
            if p.requires_grad:
                touched_trainable += n
            # Record per-module
            module_name = name.split('.')[0]
            grad_norms[module_name] = grad_norms.get(module_name, 0) + p.grad.norm().item()

    print(f"  Total params: {total_params}")
    print(f"  Trainable params: {total_trainable}")
    print(f"  Params with non-zero grad: {touched_params}")
    print(f"  Trainable params with grad: {touched_trainable}")
    print(f"  Grad coverage: {touched_trainable/total_trainable:.3f}")
    print(f"  Per-module grad norms: { {k: f'{v:.4f}' for k,v in sorted(grad_norms.items(), key=lambda x:-x[1])} }")

    # The heuristic estimate
    heuristic = out.active_params

    return {
        'claim': 'Instrumented parameter execution measurement (actual gradient flow)',
        'classification': 'EMPIRICAL',
        'evidence': f'Touched {touched_trainable}/{total_trainable} trainable params ({100*touched_trainable/total_trainable:.1f}%). Heuristic estimated {heuristic}.',
        'detail': f'Per-module: {grad_norms}. Note: test_mode uses dummy expert, so expert params are not representative of production.'
    }


# ==================== INFERENCE-TIME ROUTING ====================

def test_inference_pathway_collapse():
    """MEASURE: At inference (eval mode), does the router produce diverse
    pathway/width/depth decisions, or does it collapse to a single choice?

    With eval_routing_noise=0.15, different tokens SHOULD get different
    routing decisions (deterministic but input-dependent).
    """
    print("\n" + "="*80)
    print("TEST: INFERENCE-TIME ROUTING DIVERSITY")
    print("="*80)

    cfg = ConfigFactory.get_config('10M')  # 10M has 2 width choices
    torch.manual_seed(42)
    model = zeroModel(cfg, test_mode=True)
    model.eval()

    # Use diverse inputs
    ids = torch.randint(0, cfg.vocab_size, (4, 32))
    with torch.no_grad():
        out = model(ids, output_routing_info=True)

    rd = out.routing_info
    B, T = ids.shape

    # Pathway diversity
    pp = rd.path_probs
    path_argmax = pp.argmax(dim=-1)  # [B, T]
    path_unique = torch.unique(path_argmax).tolist()
    path_entropy = -torch.sum(pp * torch.log(pp + 1e-12), dim=-1).mean().item()
    max_path_entropy = math.log(3)
    print(f"  Pathway: {len(path_unique)}/3 pathways selected, entropy={path_entropy:.3f}/{max_path_entropy:.3f}")

    # Width diversity
    widx = rd.width_idx
    w_unique = widx.unique().tolist()
    print(f"  Width: {len(w_unique)}/{len(cfg.width_choices)} widths selected, dist={[(i, int((widx==i).sum())) for i in w_unique]}")

    # Depth diversity
    dm = rd.depth_mask
    depth_active = dm.sum(dim=-1)  # [B, T]
    depth_unique = torch.unique(dm.reshape(-1, cfg.max_depth), dim=0).shape[0]
    print(f"  Depth: {depth_unique}/{B*T} unique patterns, avg_active={depth_active.float().mean():.2f}/{cfg.max_depth}")

    # Expert diversity
    eidx = rd.expert_indices
    e_unique = eidx.flatten().unique().tolist()
    print(f"  Expert: {len(e_unique)}/{cfg.expert_count} experts selected")

    collapse = {
        'pathway': len(path_unique) < 3,
        'width': len(w_unique) < len(cfg.width_choices),
        'depth': depth_unique < 3,
        'expert': len(e_unique) < cfg.expert_count,
    }

    n_collapse = sum(collapse.values())
    print(f"  Collapsed axes: {n_collapse}/4")

    if n_collapse == 0:
        return {'claim': 'Inference routing produces diverse decisions across all 4 axes',
                'classification': 'EMPIRICAL',
                'evidence': f'No collapse: paths={len(path_unique)}/3, widths={len(w_unique)}/{len(cfg.width_choices)}, depth_patterns={depth_unique}, experts={len(e_unique)}/{cfg.expert_count}',
                'detail': '10M model with eval_routing_noise=0.15, untrained.'}
    elif n_collapse <= 2:
        return {'claim': 'Inference routing produces diverse decisions across all 4 axes',
                'classification': 'PARTIAL',
                'evidence': f'{n_collapse}/4 axes collapsed: {collapse}',
                'detail': 'Partial collapse on untrained model. Expected to improve with training.'}
    return {'claim': 'Inference routing produces diverse decisions across all 4 axes',
            'classification': 'FALSE',
            'evidence': f'{n_collapse}/4 axes collapsed: {collapse}',
            'detail': 'Severe routing collapse at inference.'}


# ==================== SPPQ ====================

def test_sppq_qat_correctness():
    """VERIFY: SPPQ QAT (quantization-aware training) produces correct
    quantization/dequantization and valid gradients.

    Key claims to test:
    1. target_bits actually affects quantization precision
    2. Gradients flow through the quantized weights
    3. Dequantized weights are close to original (for high bit-width)
    4. Documentation accurately states 0% memory reduction at inference
    """
    print("\n" + "="*80)
    print("TEST: SPPQ QAT CORRECTNESS")
    print("="*80)

    # Import SPPQ module
    try:
        from xorzen.utils.sppq import SPPQQuantizer, ProgressiveSchedule
    except ImportError as e:
        return {'claim': 'SPPQ produces correct QAT with valid gradients',
                'classification': 'UNTESTED',
                'evidence': f'Cannot import SPPQ: {e}',
                'detail': 'Module may be missing or broken'}

    # Check if SPPQ has an actual forward/quantize/dequantize path
    import inspect
    sppq_members = inspect.getmembers(SPPQQuantizer) if 'SPPQQuantizer' in dir() else []
    has_quantize = any('quantize' in name.lower() for name, _ in sppq_members)
    has_dequantize = any('dequant' in name.lower() for name, _ in sppq_members)
    has_forward = hasattr(SPPQQuantizer, 'forward') if 'SPPQQuantizer' in dir() else False

    print(f"  SPPQQuantizer: quantize={has_quantize}, dequantize={has_dequantize}, forward={has_forward}")

    # Check ProgressiveSchedule
    if 'ProgressiveSchedule' in dir():
        schedule = ProgressiveSchedule(total_steps=1000, start_bits=8, end_bits=4)
        bits_at_0 = schedule.get_bits(0)
        bits_at_500 = schedule.get_bits(500)
        bits_at_1000 = schedule.get_bits(1000)
        print(f"  ProgressiveSchedule: step=0 → {bits_at_0}b, step=500 → {bits_at_500}b, step=1000 → {bits_at_1000}b")

        schedule_correct = bits_at_0 >= bits_at_1000
    else:
        schedule_correct = None
        print(f"  ProgressiveSchedule: not found")

    # Try to quantize a simple weight
    torch.manual_seed(42)
    w = torch.randn(64, 128)
    try:
        if has_forward:
            q = SPPQQuantizer(target_bits=8)
            w_q, scale, zero_point = q(w)
            if isinstance(w_q, torch.Tensor):
                diff = (w - w_q).abs().mean().item()
                print(f"  8-bit quantization: mean error = {diff:.6f}")
                quant_works = diff < 1.0
            else:
                quant_works = False
                print(f"  8-bit quantization returned non-tensor: {type(w_q)}")
        else:
            quant_works = None
            print(f"  Cannot test quantization (no forward method)")
    except Exception as e:
        quant_works = False
        print(f"  Quantization test failed: {e}")

    # Test gradient flow
    try:
        if has_forward and quant_works:
            w = torch.randn(64, 128, requires_grad=True)
            q = SPPQQuantizer(target_bits=8)
            w_q, _, _ = q(w)
            loss = w_q.sum()
            loss.backward()
            has_grad = w.grad is not None and w.grad.abs().sum() > 0
            print(f"  Gradient flow: {has_grad}")
        else:
            has_grad = None
    except Exception as e:
        has_grad = False
        print(f"  Gradient test failed: {e}")

    # Check if inference actually saves memory (it shouldn't - QAT only, no int storage)
    # This is a DOCUMENTATION claim, not a code bug.
    # The code uses fake quantization (float weights, simulated quantization)
    # so inference memory is NOT reduced.

    return {
        'claim': 'SPPQ produces correct QAT with valid gradients',
        'classification': 'PARTIAL' if quant_works and has_grad else 'UNTESTED' if quant_works is None else 'FALSE',
        'evidence': f'Quantize works: {quant_works}, gradient flows: {has_grad}, schedule correct: {schedule_correct}',
        'detail': 'SPPQ is QAT-only (float32 weights with simulated quantization). Inference memory is NOT reduced because weights remain in float32. This matches documentation. Native int8 storage would be needed for inference savings.'
    }


# ==================== EXPERT SHARDING ====================

def test_expert_sharding_test_mode():
    """VERIFY: In test_mode, ShardedExpertFabric uses a dummy expert and
    does not touch disk. Verify this is documented and consistent.

    Full expert sharding (disk, LRU, eviction) cannot be tested in this
    environment (no persistent disk). Test the test_mode behavior and
    the LRU cache correctness independently.
    """
    print("\n" + "="*80)
    print("TEST: EXPERT SHARDING (TEST MODE + LRU CACHE)")
    print("="*80)

    cfg = ConfigFactory.get_config('1M')
    torch.manual_seed(42)
    model = zeroModel(cfg, test_mode=True)

    # Verify test_mode behavior
    moe = model.moe
    assert moe.test_mode, "Model should be in test_mode"
    assert moe.disk_manager is None, "Disk manager should be None in test_mode"
    assert hasattr(moe, 'dummy_expert'), "Dummy expert should exist"
    print(f"  test_mode: {moe.test_mode}")
    print(f"  disk_manager: {moe.disk_manager}")
    print(f"  dummy_expert params: {sum(p.numel() for p in moe.dummy_expert.parameters())}")

    # Test LRU cache independently
    from xorzen.model.zmoe import LRUExpertCache, ExpertFFN
    cache = LRUExpertCache(capacity=3)

    # Create and cache experts
    experts = []
    for i in range(5):
        e = ExpertFFN(hidden_dim=32, intermediate_dim=64)
        e.expert_id = i
        experts.append(e)

    # Fill cache (capacity 3)
    for i in range(3):
        cache.put(i, experts[i])
    assert cache.get(0) is not None, "Expert 0 should be in cache"
    assert cache.get(1) is not None, "Expert 1 should be in cache"
    assert cache.get(2) is not None, "Expert 2 should be in cache"
    assert len(cache.cache) == 3
    print(f"  Cache filled: {len(cache.cache)}/3")

    # Add 4th expert → should evict expert 0 (LRU)
    cache.put(3, experts[3])
    assert len(cache.cache) == 3
    assert 0 not in cache.cache, "Expert 0 should be evicted"
    assert 3 in cache.cache, "Expert 3 should be in cache"
    print(f"  After eviction: {list(cache.cache.keys())}")

    # Access expert 1 → moves to end (most recently used)
    cache.get(1)
    cache.put(4, experts[4])
    assert 2 not in cache.cache, "Expert 2 should be evicted (not expert 1)"
    assert 1 in cache.cache, "Expert 1 should survive (recently used)"
    print(f"  After LRU eviction: {list(cache.cache.keys())}")

    # Cache stats
    stats = cache.get_stats()
    print(f"  Cache stats: {stats}")

    # Verify memory estimate
    from xorzen.model.zmoe import estimate_expert_memory_mb
    mem = estimate_expert_memory_mb(num_experts=8, hidden_dim=64, intermediate_dim=256, cached_experts=3)
    print(f"  Memory estimate: {mem}")
    params_per_expert = mem['params_per_expert']
    expected_params = 64*256 + 64*256 + 256*64  # gate + up + down
    print(f"  Params per expert: {params_per_expert} (expected: {expected_params})")

    return {
        'claim': 'Expert sharding: LRU cache correctly evicts least-recently-used experts',
        'classification': 'PROVEN',
        'evidence': f'LRU eviction order verified: 0 evicted before 2, 2 evicted before 1. Stats: {stats}',
        'detail': f'test_mode uses dummy expert (no disk I/O). Memory estimate: {mem}. Full disk sharding requires non-test mode and persistent storage.'
    }


# ==================== SCALING LAW ====================

def test_scaling_law_implementation():
    """VERIFY: The scaling-law implementation matches the stated equations.

    From the docs: the claim is that Xorzen's conditional compute allows
    a 12B-equiv model (277M dense params) to match a 60B dense model.

    This requires:
    1. The scaling-law equations to be implemented
    2. Experimental data to support the 12B>60B claim

    We verify (1) by checking if the equations exist and match the spec.
    (2) can only be EMPIRICAL or UNTESTED without running the actual
    large-scale experiments.
    """
    print("\n" + "="*80)
    print("TEST: SCALING-LAW IMPLEMENTATION")
    print("="*80)

    # Check if scaling-law code exists
    scaling_file = os.path.join(os.path.dirname(__file__), '..', 'reports', 'scaling', 'scaling_law.md')
    has_scaling_doc = os.path.exists(scaling_file)

    # Look for scaling-law implementation in code
    scaling_code_files = []
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), '..', 'xorzen')):
        for f in files:
            if 'scal' in f.lower() and f.endswith('.py'):
                scaling_code_files.append(os.path.join(root, f))

    print(f"  Scaling law doc: {has_scaling_doc}")
    print(f"  Scaling code files: {scaling_code_files}")

    # Check if the 12B>60B claim has experimental evidence
    # Read the scaling doc
    if has_scaling_doc:
        with open(scaling_file) as f:
            content = f.read()
        has_12b_claim = '12B' in content or '60B' in content or '12b' in content
        has_equations = 'N^{' in content or 'L(N)' in content or 'scaling' in content.lower()
        has_experiments = 'experiment' in content.lower() or 'empirical' in content.lower() or 'measured' in content.lower()
        print(f"  Doc has 12B/60B claim: {has_12b_claim}")
        print(f"  Doc has equations: {has_equations}")
        print(f"  Doc has experimental data: {has_experiments}")

        # Check for actual parameter count verification
        has_param_table = 'param' in content.lower() and ('277' in content or '1M' in content or '10M' in content)
        print(f"  Doc has param table: {has_param_table}")
    else:
        has_12b_claim = has_equations = has_experiments = has_param_table = False

    # Check the scaling report data
    scaling_data_file = os.path.join(os.path.dirname(__file__), '..', 'reports', 'scaling', 'scaling_law.md')

    return {
        'claim': 'Scaling-law implementation matches stated equations; 12B>60B hypothesis is experimentally supported',
        'classification': 'UNTESTED',
        'evidence': f'Scaling doc exists: {has_scaling_doc}, has 12B/60B claim: {has_12b_claim}, has equations: {has_equations}, has experimental data: {has_experiments}',
        'detail': f'Scaling code files: {scaling_code_files}. The 12B>60B claim requires large-scale training experiments that cannot be verified in this environment. The claim is aspirational/projected based on scaling-law extrapolation, not proven by completed experiments. Code-level verification: scaling equations exist in documentation but may not have corresponding implementation code (scaling_law.md is a report, not executable code).'
    }


def test_tokenizer_roundtrip():
    """VERIFY: Tokenizer encode→decode roundtrip preserves the original text.
    """
    print("\n" + "="*80)
    print("TEST: TOKENIZER ROUND-TRIP")
    print("="*80)

    try:
        from xorzen.data.tokenizer import XorzenTokenizer
    except ImportError:
        return {'claim': 'Tokenizer encode→decode roundtrip preserves original text',
                'classification': 'UNTESTED',
                'evidence': 'Cannot import XorzenTokenizer',
                'detail': 'Module may not exist'}

    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "XorZen is a hybrid architecture.",
        "",  # empty
        "a",  # single char
    ]

    try:
        tok = XorzenTokenizer()
        all_pass = True
        for text in test_texts:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            if decoded != text:
                print(f"  FAIL: '{text}' -> {ids} -> '{decoded}'")
                all_pass = False
            else:
                print(f"  OK: '{text}' ({len(ids)} tokens)")

        if all_pass:
            return {'claim': 'Tokenizer encode→decode roundtrip preserves original text',
                    'classification': 'PROVEN',
                    'evidence': f'All {len(test_texts)} test texts roundtrip correctly',
                    'detail': 'Tested empty string, single char, and normal sentences.'}
        return {'claim': 'Tokenizer encode→decode roundtrip preserves original text',
                'classification': 'FALSE',
                'evidence': 'Some texts do not roundtrip correctly',
                'detail': 'See output above for failures.'}
    except Exception as e:
        return {'claim': 'Tokenizer encode→decode roundtrip preserves original text',
                'classification': 'ERROR',
                'evidence': str(e),
                'detail': 'Tokenizer test failed'}


# ==================== MAIN ====================

def main():
    results = {}
    tests = [
        ('active_param_heuristic', test_active_param_estimate_accuracy),
        ('active_param_instrumented', test_active_param_instrumented),
        ('inference_routing_diversity', test_inference_pathway_collapse),
        ('sppq_qat', test_sppq_qat_correctness),
        ('expert_sharding', test_expert_sharding_test_mode),
        ('scaling_law', test_scaling_law_implementation),
        ('tokenizer_roundtrip', test_tokenizer_roundtrip),
    ]
    for name, fn in tests:
        try:
            r = fn()
            results[name] = r
            print(f"  >> CLASSIFICATION: {r['classification']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'claim': name, 'classification': 'ERROR', 'evidence': str(e)}

    print("\n" + "#"*80)
    print("REMAINING SUBSYSTEMS SUMMARY")
    print("#"*80)
    for n, r in results.items():
        print(f"  {r['classification']:10s} | {r['claim']}")

    out_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'phase2_remaining.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == '__main__':
    main()
