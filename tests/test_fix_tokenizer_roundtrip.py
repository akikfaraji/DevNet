R"""
Regression test for tokenizer round-trip accuracy.

P12 claimed: decode(encode(x)).startswith(x).  This uses startswith
because BPE tokenizers may prepend a space.  The stronger claim is
that decode(encode(x)) == x for common ASCII text, which is only
guaranteed when the input was seen during training.  We test the
weaker claim (startswith) for correctness and document the limitation.
"""

import sys
sys.path.insert(0, "/home/z/my-project/DevNet")

import pytest
import xorzen


def test_65k_roundtrip_common_ascii():
    """Common ASCII text should round-trip exactly (no leading space)."""
    tk = xorzen.load_pretrained("xorzen_agi_tokenizer_65k")
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tk.encode(text)
    decoded = tk.decode(tokens)
    assert decoded == text, (
        f"Exact round-trip failed for common ASCII text.\n"
        f"  Input:  {text!r}\n"
        f"  Tokens: {tokens[:10]}...\n"
        f"  Output: {decoded!r}\n"
    )


def test_65k_roundtrip_starts_with():
    """Even when BPE adds a leading space, output must start with input."""
    tk = xorzen.load_pretrained("xorzen_agi_tokenizer_65k")
    # Use a single short word that is almost certainly in vocabulary
    text = "Hello"
    tokens = tk.encode(text)
    decoded = tk.decode(tokens)
    assert decoded.startswith(text), (
        f"startswith round-trip failed.\n"
        f"  Input:  {text!r}\n"
        f"  Output: {decoded!r}\n"
    )


def test_65k_roundtrip_math_expression():
    """Math expressions should round-trip closely."""
    tk = xorzen.load_pretrained("xorzen_agi_tokenizer_65k")
    text = "2 + 2 = 4"
    tokens = tk.encode(text)
    decoded = tk.decode(tokens)
    assert decoded == text or decoded.startswith(text), (
        f"Round-trip unexpected for math text.\n"
        f"  Input:  {text!r}\n"
        f"  Output: {decoded!r}\n"
    )


def test_10k_roundtrip_exists():
    """10k tokenizer should load and encode/decode."""
    tk = xorzen.load_pretrained("zero_bpe_10k")
    text = "test"
    tokens = tk.encode(text)
    decoded = tk.decode(tokens)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
    assert text in decoded or decoded.startswith(text), (
        f"10k round-trip failed. Input: {text!r}, Output: {decoded!r}"
    )
