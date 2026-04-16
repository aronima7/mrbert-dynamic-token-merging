"""
FLOPs analysis for MrBD3LM (plan Section 4.1).

Computes analytical FLOPs:
    FLOPs_base  = N · L² · d  (N layers, full sequence L, hidden dim d)
    FLOPs_mr    = k · L² · d + (N-k) · L'² · d + gate_overhead
    Speedup     = FLOPs_base / FLOPs_mr

Also profiles wall-clock time per forward pass over 100 steps.

Usage
-----
python mrdiffusion-bd3lms/evaluation/eval_flops.py \\
    --checkpoint ./mrd_bd3lm_output/final/checkpoint.pt \\
    --batch_size 8 --seq_len 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_BD3LMS_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "bd3lms")
for p in [_MODULE_DIR, _BD3LMS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configuration_mrd_bd3lm import MrBD3LMConfig
from modeling_mrd_bd3lm import MrBD3LM


def analytical_flops(n_blocks, seq_len, hidden_dim, delete_gate_layer, deletion_rate, n_heads):
    """
    Analytical FLOPs for one forward pass (attention only, leading term).

    For each DDiTBlock: 4 * L² * D  (QK, AV, QKV proj, out proj ~ 4L²D).
    """
    L = seq_len
    d = hidden_dim
    attn_flops_per_block = 4 * L * L * d  # approximate

    L_prime = int(L * (1 - deletion_rate))
    gate_block = delete_gate_layer

    flops_base = n_blocks * attn_flops_per_block
    flops_phase1 = (gate_block + 1) * attn_flops_per_block
    flops_phase2 = (n_blocks - gate_block - 1) * 4 * L_prime * L_prime * d
    gate_overhead = 2 * L * d  # gate MLP: two linear layers
    flops_mr = flops_phase1 + flops_phase2 + gate_overhead

    return flops_base, flops_mr, flops_base / max(flops_mr, 1)


def profile_wall_clock(model, batch_size, seq_len, n_steps=100, device="cuda"):
    """Time per forward pass over n_steps warm-up + n_steps measured."""
    model.eval()
    x = torch.randint(0, model.config.vocab_size - 1,
                      (batch_size, seq_len * 2 if model.config.cross_attn else seq_len),
                      device=device)
    t = torch.rand(batch_size, device=device)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            model(x, timesteps=t)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        with torch.no_grad():
            model(x, timesteps=t)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / n_steps * 1000  # ms per step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--model_size", default="small", choices=["tiny", "small", "medium"])
    p.add_argument("--deletion_rate", type=float, default=0.3,
                   help="Assumed actual deletion rate for FLOPs computation")
    p.add_argument("--no_profile", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    presets = {
        "tiny":   dict(hidden_dim=384, n_blocks=6,  n_heads=6),
        "small":  dict(hidden_dim=768, n_blocks=12, n_heads=12),
        "medium": dict(hidden_dim=1024, n_blocks=24, n_heads=16),
    }
    preset = presets[args.model_size]

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg_dict = ckpt.get("config", {})
        config = MrBD3LMConfig(**{k: v for k, v in cfg_dict.items()
                                  if hasattr(MrBD3LMConfig, k) or True})
    else:
        config = MrBD3LMConfig(
            model_length=args.seq_len,
            **preset,
        )

    model = MrBD3LM(config).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)

    n_params = sum(p.numel() for p in model.parameters())

    # Analytical FLOPs
    flops_base, flops_mr, speedup = analytical_flops(
        config.n_blocks, args.seq_len, config.hidden_dim,
        config.delete_gate_layer, args.deletion_rate, config.n_heads)

    print(f"\n{'='*60}")
    print(f"Model: {args.model_size}  ({n_params/1e6:.1f}M params)")
    print(f"Seq len: {args.seq_len}  delete_gate_layer: {config.delete_gate_layer}")
    print(f"Assumed deletion rate: {args.deletion_rate:.1%}")
    print(f"{'='*60}")
    print(f"FLOPs (baseline):  {flops_base/1e9:.2f}G")
    print(f"FLOPs (MrBD3LM):   {flops_mr/1e9:.2f}G")
    print(f"FLOPs speedup:     {speedup:.2f}x")

    if not args.no_profile and torch.cuda.is_available():
        ms_per_step = profile_wall_clock(model, args.batch_size, args.seq_len,
                                         device=device)
        print(f"\nWall-clock (MrBD3LM): {ms_per_step:.1f} ms/step")
        print(f"  Batch size: {args.batch_size}, Device: {device}")

        # Compare with baseline (no gate): build config with deletion_loss_weight=0
        cfg_base = MrBD3LMConfig(**{**vars(config), "deletion_loss_weight": 0.0})
        model_base = MrBD3LM(cfg_base).to(device)
        ms_base = profile_wall_clock(model_base, args.batch_size, args.seq_len,
                                      device=device)
        print(f"Wall-clock (baseline): {ms_base:.1f} ms/step")
        print(f"Wall-clock speedup:    {ms_base/ms_per_step:.2f}x")
        del model_base
    print(f"{'='*60}")


if __name__ == "__main__":
    main()