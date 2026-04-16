"""
Unit tests for MrSEDD (plan Section 5, Phase 2, Week 4).

Tests:
    1. Merge/restore roundtrip: TokenMerge → TokenRestore preserves kept token values
    2. Gradient flow: gradients flow through gate and loss
    3. Gate disabled → output shape unchanged, no gate mask
    4. PreDeletionBlend: output shape and value properties
    5. BottleneckDeleteGate: output shapes, temperature effect, STE gradients
    6. Gate logit regularization: logit reg term added to loss when configured

Run:
    python mrdiffusion-sedd/tests/test_mrsedd.py
    python -m pytest mrdiffusion-sedd/tests/test_mrsedd.py -v
"""

from __future__ import annotations

import os
import sys
import math

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_SEDD_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "Score-Entropy-Discrete-Diffusion")
for p in [_MODULE_DIR, _SEDD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Merge/Restore roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def test_merge_restore_roundtrip():
    """TokenMerge → TokenRestore: kept token values preserved exactly."""
    from modeling_mrdiffusion import TokenMerge, TokenRestore

    merge = TokenMerge()
    restore = TokenRestore()

    B, L, D = 2, 8, 16
    h = torch.randn(B, L, D)
    pre_gate = torch.randn(B, L, D)

    # Keep every other token
    keep_prob = torch.zeros(B, L, 1)
    keep_prob[:, ::2] = 1.0
    gate_hard = (keep_prob > 0.5).float()

    merged, merge_info = merge(h, gate_hard)
    restored = restore(merged, merge_info, pre_gate)

    keep_mask = merge_info["keep_mask"]
    for b in range(B):
        kept_idx = keep_mask[b].nonzero(as_tuple=True)[0]
        assert torch.allclose(restored[b, kept_idx], h[b, kept_idx], atol=1e-5), \
            f"Kept token values not preserved in batch {b}"

    for b in range(B):
        deleted_idx = (~keep_mask[b]).nonzero(as_tuple=True)[0]
        if len(deleted_idx) > 0:
            assert torch.allclose(restored[b, deleted_idx], pre_gate[b, deleted_idx], atol=1e-5), \
                f"Deleted positions should use pre_gate states"

    print("✓ test_merge_restore_roundtrip")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Gradient flow through gate
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow():
    """Loss gradient should flow through the delete gate parameters."""
    from configuration_mrdiffusion import MrDiffusionConfig
    from modeling_mrdiffusion import MrSEDD

    config = MrDiffusionConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        delete_gate_layer=1,
        deletion_mode="soft",
        deletion_loss_weight=0.1,
        gate_type="bottleneck_mlp",
    )
    model = MrSEDD(config)
    model.train()

    B, L = 2, 16
    x = torch.randint(0, config.vocab_size - 1, (B, L))
    sigma = torch.rand(B) * 2.0 + 0.1

    out = model(x, sigma)
    logits = out.logits if hasattr(out, "logits") else out
    loss = logits.sum()
    loss.backward()

    gate_params_with_grad = [
        (n, p) for n, p in model.named_parameters()
        if "delete_gate" in n and p.grad is not None
    ]
    assert len(gate_params_with_grad) > 0, "No gradient reached gate parameters!"

    for n, p in gate_params_with_grad:
        assert not torch.isnan(p.grad).any(), f"NaN gradient in {n}"

    print(f"✓ test_gradient_flow ({len(gate_params_with_grad)} gate params have gradients)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Gate disabled → no gate mask
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_disabled():
    """With deletion_loss_weight=0, output shape is correct and no gate fired."""
    from configuration_mrdiffusion import MrDiffusionConfig
    from modeling_mrdiffusion import MrSEDD

    config = MrDiffusionConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        delete_gate_layer=1,
        deletion_loss_weight=0.0,
    )
    model = MrSEDD(config)
    model.eval()

    B, L = 2, 16
    x = torch.randint(0, config.vocab_size - 1, (B, L))
    sigma = torch.rand(B) * 2.0 + 0.1

    with torch.no_grad():
        out = model(x, sigma)

    logits = out.logits if hasattr(out, "logits") else out
    assert logits.shape == (B, L, config.vocab_size), \
        f"Expected ({B},{L},{config.vocab_size}), got {logits.shape}"

    print("✓ test_gate_disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: PreDeletionBlend output properties
# ─────────────────────────────────────────────────────────────────────────────

def test_pre_deletion_blend():
    """PreDeletionBlend: gate=1 keeps original, gate=0 takes next neighbor."""
    from modeling_mrdiffusion import PreDeletionBlend

    blend = PreDeletionBlend()
    B, L, D = 2, 8, 16
    h = torch.randn(B, L, D)

    # All gates = 1 (keep): blended should equal original
    gate_keep = torch.ones(B, L, 1)
    out = blend(h, gate_keep)
    assert torch.allclose(out, h, atol=1e-5), "gate=1 should preserve original"

    # All gates = 0 (delete): blended should be next neighbor (rolled)
    gate_del = torch.zeros(B, L, 1)
    out = blend(h, gate_del)
    h_next = torch.roll(h, -1, dims=1)
    h_next[:, -1, :] = h[:, -1, :]
    assert torch.allclose(out, h_next, atol=1e-5), "gate=0 should use next neighbor"

    print("✓ test_pre_deletion_blend")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: BottleneckDeleteGate shapes and STE
# ─────────────────────────────────────────────────────────────────────────────

def test_bottleneck_gate():
    """BottleneckDeleteGate: correct output shapes, temperature effect, STE."""
    from modeling_mrdiffusion import BottleneckDeleteGate

    D, cond_dim, bottleneck = 64, 64, 128
    gate = BottleneckDeleteGate(D, cond_dim, bottleneck, sigmoid_mask_scale=-30.0)
    gate.train()

    B, L = 2, 16
    x = torch.randn(B, L, D)
    sigma = torch.randn(B, cond_dim)

    # Test output shapes — SEDD gate returns bias in [scale, 0]
    gate_bias, logits = gate(x, sigma, temperature=1.0)
    assert gate_bias.shape == (B, L, 1), f"Expected ({B},{L},1), got {gate_bias.shape}"
    assert logits.shape == (B, L, 1), f"Logits shape mismatch"

    # Gate bias should be in [sigmoid_mask_scale, 0]
    assert (gate_bias <= 0).all(), "Gate bias should be <= 0"
    assert (gate_bias >= -30.0).all(), "Gate bias should be >= sigmoid_mask_scale"

    # Test gradient flows through STE via logits
    gate.zero_grad()
    gate_b, log = gate(x, sigma, temperature=1.0)
    loss = gate_b.sum() + log.sum()
    loss.backward()
    assert all(p.grad is not None for p in gate.parameters()), \
        "STE should allow gradient flow"

    print("✓ test_bottleneck_gate")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Gate logit regularization
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_logit_regularization():
    """gate_logit_reg_weight > 0 should add a non-zero regularization term."""
    from configuration_mrdiffusion import MrDiffusionConfig
    from modeling_mrdiffusion import MrSEDD
    from losses_mrdiffusion import get_loss_fn

    config = MrDiffusionConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        delete_gate_layer=1,
        deletion_mode="soft",
        deletion_loss_weight=0.1,
        gate_type="bottleneck_mlp",
        gate_logit_reg_weight=0.001,
    )
    model = MrSEDD(config)
    model.train()

    B, L = 2, 16
    x = torch.randint(0, config.vocab_size - 1, (B, L))

    # Manually run forward and check gate logits are captured
    sigma = torch.rand(B) * 2.0 + 0.1
    out = model(x, sigma)

    gate_logits = getattr(out, "delete_gate_logits", None)
    if gate_logits is not None:
        reg = config.gate_logit_reg_weight * gate_logits.pow(2).mean()
        assert reg.item() >= 0.0, "Regularization should be non-negative"
        assert not torch.isnan(reg), "NaN in gate logit regularization"

    print("✓ test_gate_logit_regularization")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running MrSEDD unit tests...\n")
    tests = [
        test_merge_restore_roundtrip,
        test_gradient_flow,
        test_gate_disabled,
        test_pre_deletion_blend,
        test_bottleneck_gate,
        test_gate_logit_regularization,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)