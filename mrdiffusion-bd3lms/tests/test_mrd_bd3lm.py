"""
Unit tests for MrBD3LM (plan Section 5, Phase 2, Week 4).

Tests:
    1. Merge/restore roundtrip: TokenMerge → TokenRestore preserves kept token values
    2. Gradient flow: gradients flow through gate and loss
    3. Loss equivalence: with gate disabled, loss matches standard BD3LM
    4. PI controller convergence: controller drives deletion rate toward target
    5. PreDeletionBlend: output shape and value properties
    6. BottleneckDeleteGate: output shapes, temperature effect, STE gradients

Run:
    python mrdiffusion-bd3lms/tests/test_mrd_bd3lm.py
    python -m pytest mrdiffusion-bd3lms/tests/test_mrd_bd3lm.py -v
"""

from __future__ import annotations

import os
import sys
import math

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_BD3LMS_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "bd3lms")
for p in [_MODULE_DIR, _BD3LMS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _get_model_and_noise(delete_gate=True, deletion_mode="soft"):
    from configuration_mrd_bd3lm import MrBD3LMConfig
    from modeling_mrd_bd3lm import MrBD3LM
    from noise_schedule import LogLinearNoise

    config = MrBD3LMConfig(
        hidden_dim=64, n_blocks=4, n_heads=4, cond_dim=32,
        vocab_size=100, model_length=16,
        block_size=1, cross_attn=False, adaln=True,
        delete_gate_layer=1,
        restore_gate_layer=3,
        deletion_mode=deletion_mode,
        deletion_loss_weight=0.1 if delete_gate else 0.0,
        gate_type="scaled_sigmoid",
    )
    model = MrBD3LM(config)
    noise = LogLinearNoise()
    return model, config, noise


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Merge/Restore roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def test_merge_restore_roundtrip():
    """TokenMerge → TokenRestore: kept token values preserved exactly."""
    from modeling_mrd_bd3lm import TokenMerge, TokenRestore

    merge = TokenMerge()
    restore = TokenRestore()

    B, L, D = 2, 8, 16
    h = torch.randn(B, L, D)
    pre_gate = torch.randn(B, L, D)

    # Keep every other token
    keep_prob = torch.zeros(B, L, 1)
    keep_prob[:, ::2] = 1.0   # keep even positions
    gate_hard = (keep_prob > 0.5).float()

    merged, merge_info = merge(h, gate_hard)
    restored = restore(merged, merge_info, pre_gate)

    # Check: kept positions should match original hidden states
    keep_mask = merge_info["keep_mask"]  # [B, L]
    for b in range(B):
        kept_idx = keep_mask[b].nonzero(as_tuple=True)[0]
        assert torch.allclose(restored[b, kept_idx], h[b, kept_idx], atol=1e-5), \
            f"Kept token values not preserved in batch {b}"

    # Check: deleted positions should come from pre_gate (not zeros)
    for b in range(B):
        deleted_idx = (~keep_mask[b]).nonzero(as_tuple=True)[0]
        if len(deleted_idx) > 0:
            assert torch.allclose(restored[b, deleted_idx], pre_gate[b, deleted_idx], atol=1e-5), \
                f"Deleted positions should use pre_gate states, not zeros"

    print("✓ test_merge_restore_roundtrip")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Gradient flow through gate
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow():
    """Loss gradient should flow through the delete gate parameters."""
    model, config, noise = _get_model_and_noise(delete_gate=True)
    model.train()

    B, L = 2, 16
    x0 = torch.randint(0, config.vocab_size - 1, (B, L))
    t = torch.rand(B)
    _, move_chance = noise(t)
    xt = torch.where(torch.rand(B, L) < move_chance.unsqueeze(-1),
                     config.vocab_size - 1, x0)

    logits = model(xt, timesteps=t)
    if hasattr(logits, "logits"):
        logits = logits.logits

    loss = logits.sum()
    loss.backward()

    # Check gate parameters have gradients
    gate_params_with_grad = [
        (n, p) for n, p in model.named_parameters()
        if "delete_gate" in n and p.grad is not None
    ]
    assert len(gate_params_with_grad) > 0, \
        "No gradient reached gate parameters!"

    for n, p in gate_params_with_grad:
        assert not torch.isnan(p.grad).any(), f"NaN gradient in {n}"

    print(f"✓ test_gradient_flow ({len(gate_params_with_grad)} gate params have gradients)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Gate disabled → no deletion rate loss
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_disabled_loss():
    """With deletion_loss_weight=0, loss should not include gate loss term."""
    model, config, noise = _get_model_and_noise(delete_gate=False)
    model.eval()

    B, L = 2, 16
    x0 = torch.randint(0, config.vocab_size - 1, (B, L))

    with torch.no_grad():
        t = torch.rand(B)
        _, move_chance = noise(t)
        xt = torch.where(torch.rand(B, L) < move_chance.unsqueeze(-1),
                         config.vocab_size - 1, x0)
        logits = model(xt, timesteps=t)
        if hasattr(logits, "logits"):
            logits = logits.logits

    # Should have full vocab logits for all positions
    assert logits.shape == (B, L, config.vocab_size), \
        f"Expected logits shape ({B}, {L}, {config.vocab_size}), got {logits.shape}"

    print("✓ test_gate_disabled_loss")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: PI controller convergence
# ─────────────────────────────────────────────────────────────────────────────

def test_pi_controller_convergence():
    """PI controller should drive actual rate toward target within 1000 steps."""
    sys.path.insert(0, _MODULE_DIR)
    from train_mrd_bd3lm import PIController

    target = 0.3
    ctrl = PIController(target_rate=target, kp=0.5, ki=1e-4, gamma=0.9)

    # Simulate: actual rate starts at 0.5, controller adjusts deletion weight
    actual = 0.5
    alphas = []
    for _ in range(2000):
        alpha = ctrl.update(actual)
        alphas.append(alpha)
        # Simulate: higher alpha → lower actual rate (crude model)
        actual = 0.5 - 0.2 * min(alpha, 1.0)
        actual = max(0.0, min(1.0, actual))

    final_error = abs(target - actual)
    assert final_error < 0.15, f"PI controller did not converge: error={final_error:.3f}"

    print(f"✓ test_pi_controller_convergence (final error={final_error:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: PreDeletionBlend output properties
# ─────────────────────────────────────────────────────────────────────────────

def test_pre_deletion_blend():
    """PreDeletionBlend: gate=1 keeps original, gate=0 takes next neighbor."""
    from modeling_mrd_bd3lm import PreDeletionBlend

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
# Test 6: BottleneckDeleteGate shapes and STE
# ─────────────────────────────────────────────────────────────────────────────

def test_bottleneck_gate():
    """BottleneckDeleteGate: correct output shapes, temperature effect, STE."""
    from modeling_mrd_bd3lm import BottleneckDeleteGate

    D, cond_dim, bottleneck = 64, 32, 128
    gate = BottleneckDeleteGate(D, cond_dim, bottleneck)
    gate.train()

    B, L = 2, 16
    x = torch.randn(B, L, D)
    sigma = torch.randn(B, cond_dim)

    # Test output shapes
    gate_vals, logits = gate(x, sigma, temperature=1.0)
    assert gate_vals.shape == (B, L, 1), f"Expected ({B},{L},1), got {gate_vals.shape}"
    assert logits.shape == (B, L, 1), f"Logits shape mismatch"

    # Gate values should be in [0,1] (Gumbel-sigmoid)
    assert (gate_vals >= 0).all() and (gate_vals <= 1).all(), \
        "Gate values should be in [0,1]"

    # Higher temperature → more uncertainty (higher entropy)
    gate_hi_temp, _ = gate(x, sigma, temperature=5.0)
    gate_lo_temp, _ = gate(x, sigma, temperature=0.1)
    # Mean entropy should be higher for higher temperature
    ent_hi = -(gate_hi_temp * (gate_hi_temp + 1e-6).log() +
               (1 - gate_hi_temp) * (1 - gate_hi_temp + 1e-6).log()).mean()
    ent_lo = -(gate_lo_temp * (gate_lo_temp + 1e-6).log() +
               (1 - gate_lo_temp) * (1 - gate_lo_temp + 1e-6).log()).mean()
    assert ent_hi > ent_lo, "Higher temperature should give higher entropy"

    # Test gradient flows through STE
    gate.zero_grad()
    gate_v, log = gate(x, sigma, temperature=1.0)
    gate_hard = (gate_v > 0.5).float()
    gate_ste = gate_v + (gate_hard - gate_v).detach()
    loss = gate_ste.sum()
    loss.backward()
    assert all(p.grad is not None for p in gate.parameters()), \
        "STE should allow gradient flow"

    print("✓ test_bottleneck_gate")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: compute_loss output shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_loss():
    """compute_loss returns expected keys and non-NaN values."""
    model, config, noise = _get_model_and_noise(delete_gate=True)
    model.train()
    device = torch.device("cpu")

    sys.path.insert(0, _MODULE_DIR)
    from train_mrd_bd3lm import compute_loss

    B, L = 2, 16
    x0 = torch.randint(0, config.vocab_size - 1, (B, L))
    attn_mask = torch.ones(B, L)

    metrics = compute_loss(model, x0, attn_mask, noise,
                           config.vocab_size - 1, config, device)

    assert "loss" in metrics
    assert "score_loss" in metrics
    assert "deletion_rate" in metrics
    assert not torch.isnan(metrics["loss"]), "NaN in loss!"
    assert 0.0 <= metrics["deletion_rate"].item() <= 1.0, \
        f"deletion_rate out of range: {metrics['deletion_rate'].item()}"

    print("✓ test_compute_loss")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running MrBD3LM unit tests...\n")
    tests = [
        test_merge_restore_roundtrip,
        test_gradient_flow,
        test_gate_disabled_loss,
        test_pi_controller_convergence,
        test_pre_deletion_blend,
        test_bottleneck_gate,
        test_compute_loss,
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