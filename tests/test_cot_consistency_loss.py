"""
CoT Consistency Loss Regression Tests — TRAINING-STAGE BUG fix

Bug: The CoT consistency loss was hardcoded to torch.tensor(0.0) and the CoT
module was never called in the forward pass, making enable_cot() a no-op.

Root cause: Three cascading defects:
1. self.cot never called — zeros tensor used unconditionally
2. cot_consistency_loss = torch.tensor(0.0) — unconditional
3. cot_consistency_loss not added to loss accumulation

Minimal fix:
- Added self._cot_enabled flag (default False)
- _freeze_cot() sets _cot_enabled = False
- enable_cot() sets _cot_enabled = True
- Forward pass calls self.cot() when _cot_enabled
- Loss computation uses _compute_cot_consistency_loss() when _cot_enabled
- cot_consistency_loss added to total loss accumulation

Regression tests proving:
1. Consistency loss is zero/inactive when CoT is supposed to be inactive
2. It becomes genuinely non-zero when Stage 2 enables it
3. Gradients reach intended CoT-related parameters
4. Frozen parameters remain frozen where required
5. The loss cannot silently remain hardcoded to zero
"""

import sys
import os
import logging
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.model import zeroModel


# ---------- helpers ----------

def _make_config(**overrides):
    """Create a tiny model config for fast CPU tests."""
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.pad_token_id = 0
    cfg.tie_word_embeddings = True
    cfg.gradient_checkpointing = False
    cfg.context_length = 64
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_model(cfg):
    """Create model in test mode (CPU, no disk shards)."""
    logging.disable(logging.CRITICAL)
    try:
        return zeroModel(cfg, test_mode=True)
    finally:
        logging.disable(logging.NOTSET)


# ---------- tests ----------

class TestCoTConsistencyLossInactive:
    """CoT consistency loss must be exactly zero during pre-training (Stage 1)."""

    def test_cot_loss_zero_when_disabled(self):
        """Assertion 1: consistency loss is zero when _cot_enabled is False."""
        cfg = _make_config()
        model = _make_model(cfg)
        model.train()

        assert model._cot_enabled is False, "CoT should be disabled after init"

        input_ids = torch.randint(1, cfg.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)

        assert out.cot_consistency_loss is not None
        assert out.cot_consistency_loss.item() == 0.0, \
            f"CoT loss should be 0.0 when disabled, got {out.cot_consistency_loss.item()}"

    def test_cot_vector_is_zeros_when_disabled(self):
        """CoT vector should be all zeros when disabled."""
        cfg = _make_config()
        model = _make_model(cfg)
        model.eval()

        input_ids = torch.randint(1, cfg.vocab_size, (2, 16))
        out = model(input_ids=input_ids, output_cot_vector=True)

        assert out.cot_vector is not None
        assert torch.all(out.cot_vector == 0.0), \
            "CoT vector should be all zeros when disabled"

    def test_cot_params_frozen_when_disabled(self):
        """Assertion 4: all CoT params have requires_grad=False before enable_cot()."""
        cfg = _make_config()
        model = _make_model(cfg)

        for name, param in model.cot.named_parameters():
            assert not param.requires_grad, \
                f"CoT param {name} should be frozen before enable_cot()"


class TestCoTConsistencyLossActive:
    """CoT consistency loss must become genuinely non-zero after enable_cot()."""

    def test_cot_loss_nonzero_when_enabled(self):
        """Assertion 2: consistency loss is non-zero when CoT is enabled."""
        cfg = _make_config()
        model = _make_model(cfg)
        model.eval()

        # Enable CoT
        model.enable_cot()
        assert model._cot_enabled is True, "CoT should be enabled after enable_cot()"

        # Need seq_length >= 2 for _compute_cot_consistency_loss to be non-trivial
        input_ids = torch.randint(1, cfg.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)

        assert out.cot_consistency_loss is not None
        # The CoT vector is non-zero now, so consistency loss should be > 0
        # (it measures L2 diff between consecutive token CoT vectors)
        assert out.cot_consistency_loss.item() > 0.0, \
            f"CoT loss should be > 0.0 when enabled, got {out.cot_consistency_loss.item()}"

    def test_cot_loss_included_in_total_loss(self):
        """Assertion 5: cot_consistency_loss is part of the backward loss."""
        cfg = _make_config()
        model = _make_model(cfg)
        model.eval()

        model.enable_cot()

        input_ids = torch.randint(1, cfg.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)

        # The returned loss must differ from lm_loss by at least the CoT loss
        assert out.loss is not None
        assert out.lm_loss is not None
        assert out.cot_consistency_loss is not None
        # loss = lm_loss + routing_loss + load_balance_loss + cot_consistency_loss
        # So loss - lm_loss should be >= cot_consistency_loss (both routing and
        # load_balance are >= 0)
        diff = (out.loss - out.lm_loss).item()
        cot_val = out.cot_consistency_loss.item()
        assert diff >= cot_val - 1e-6, \
            f"loss - lm_loss ({diff}) should be >= cot_consistency_loss ({cot_val})"

    def test_cot_vector_nonzero_when_enabled(self):
        """CoT vector should be non-zero when CoT is enabled."""
        cfg = _make_config()
        model = _make_model(cfg)
        model.eval()

        model.enable_cot()

        input_ids = torch.randint(1, cfg.vocab_size, (2, 16))
        out = model(input_ids=input_ids, output_cot_vector=True)

        assert out.cot_vector is not None
        assert not torch.all(out.cot_vector == 0.0), \
            "CoT vector should be non-zero when enabled"


class TestCoTGradients:
    """Gradients must reach CoT parameters after enable_cot()."""

    def test_gradients_reach_cot_params(self):
        """Assertion 3: CoT params receive gradients after enable_cot()."""
        cfg = _make_config()
        model = _make_model(cfg)
        model.train()

        model.enable_cot()

        # Verify params are unfrozen
        cot_param_count = 0
        for name, param in model.cot.named_parameters():
            assert param.requires_grad, \
                f"CoT param {name} should have requires_grad=True after enable_cot()"
            cot_param_count += 1
        assert cot_param_count > 0, "CoT module should have parameters"

        input_ids = torch.randint(1, cfg.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)

        out.loss.backward()

        # At least some CoT params should have non-zero gradients
        params_with_grad = 0
        for name, param in model.cot.named_parameters():
            if param.grad is not None and param.grad.norm().item() > 0:
                params_with_grad += 1

        assert params_with_grad > 0, \
            f"Expected CoT params to receive gradients, but {params_with_grad}/{cot_param_count} have non-zero grad"

    def test_no_gradients_when_frozen(self):
        """Assertion 4 (supplement): CoT params get NO gradients when frozen."""
        cfg = _make_config()
        model = _make_model(cfg)
        model.train()

        # CoT is frozen by default
        assert not model._cot_enabled

        input_ids = torch.randint(1, cfg.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()

        for name, param in model.cot.named_parameters():
            # Frozen params should either have no grad or zero grad
            if param.grad is not None:
                assert param.grad.norm().item() == 0.0, \
                    f"Frozen CoT param {name} should have zero gradient"


class TestCoTEnableDisable:
    """enable_cot / _freeze_cot lifecycle."""

    def test_enable_then_freeze_roundtrip(self):
        """After enable_cot() + _freeze_cot(), CoT should be disabled again."""
        cfg = _make_config()
        model = _make_model(cfg)

        model.enable_cot()
        assert model._cot_enabled is True

        model._freeze_cot()
        assert model._cot_enabled is False

        for name, param in model.cot.named_parameters():
            assert not param.requires_grad, \
                f"CoT param {name} should be frozen after _freeze_cot()"

    def test_cot_loss_zero_after_refreeze(self):
        """After enable_cot() then _freeze_cot(), loss should be zero again."""
        cfg = _make_config()
        model = _make_model(cfg)
        model.eval()

        model.enable_cot()
        model._freeze_cot()

        input_ids = torch.randint(1, cfg.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)

        assert out.cot_consistency_loss.item() == 0.0, \
            "CoT loss should be 0.0 after refreezing"
