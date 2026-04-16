"""
FLOPs analysis for MrSEDD (plan Section 4.3, Experiment 3).

Computes:
  - Analytical FLOPs for baseline SEDD vs MrSEDD at a given deletion rate
  - Wall-clock profiling via 100 forward passes

Usage
-----
python mrdiffusion-sedd/evaluation/eval_flops.py \
    --checkpoint ./mrsedd_output/final/checkpoint.pt \
    --deletion_rate 0.3 --seq_len 128 --batch_size 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_SEDD_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "Score-Entropy-Discrete-Diffusion")
for p in [_MODULE_DIR, _SEDD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configuration_mrdiffusion import MrDiffusionConfig
from modeling_mrdiffusion import MrSEDD


# ---------------------------------------------------------------------------
# Analytical FLOPs
# ---------------------------------------------------------------------------

def analytical_flops(n_blocks: int, seq_len: int, hidden_dim: int,
                     gate_layer: int, deletion_rate: float,
                     n_heads: int = 16) -> tuple[float, float, float]:
    """Return (baseline_flops, mr_flops, speedup) for one forward pass.

    Approximation: each DDiTBlock ≈ 4·L·d² FLOPs (2 attn projections + FFN).
    After gate_layer, MrSEDD processes L' = L·(1-deletion_rate) tokens.
    """
    L = seq_len
    d = hidden_dim
    L_prime = int(L * (1.0 - deletion_rate))

    # FLOPs per block: attention O(L²d + Ld²) + FFN O(4Ld²) ≈ 5Ld²
    flops_per_block_full = 5 * L * d * d
    flops_per_block_compressed = 5 * L_prime * d * d

    # Gate overhead ≈ L·d (linear projection + MLP)
    gate_flops = 2 * L * d * d  # bottleneck MLP overhead

    baseline = n_blocks * flops_per_block_full
    mr = (gate_layer * flops_per_block_full
          + gate_flops
          + (n_blocks - gate_layer) * flops_per_block_compressed)

    speedup = baseline / max(mr, 1)
    return baseline, mr, speedup


# ---------------------------------------------------------------------------
# Wall-clock profiling
# ---------------------------------------------------------------------------

def profile_wall_clock(model, config, device, batch_size=8, seq_len=128,
                        n_warmup=10, n_runs=100):
    """Time n_runs forward passes and report mean ms/step."""
    model.eval()
    vocab = config.vocab_size
    dummy = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    t_dummy = torch.rand(batch_size, device=device) * 0.99 + 0.005
    sigma_dummy = -torch.log1p(-t_dummy)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy, sigma_dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy, sigma_dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_step = elapsed / n_runs * 1000
    return ms_per_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--deletion_rate", type=float, default=0.3)
    p.add_argument("--n_blocks", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--gate_layer", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--profile", action="store_true",
                   help="Run wall-clock profiling (requires --checkpoint)")
    p.add_argument("--deletion_mode", default="hard", choices=["soft", "hard"],
                   help="Deletion mode for wall-clock profiling. 'hard' (default) physically "
                        "removes tokens — this is what produces actual speedup at inference.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Analytical FLOPs
    baseline_f, mr_f, speedup = analytical_flops(
        args.n_blocks, args.seq_len, args.hidden_dim,
        args.gate_layer, args.deletion_rate, args.n_heads)

    print(f"\nAnalytical FLOPs (seq_len={args.seq_len}, deletion_rate={args.deletion_rate})")
    print(f"  Baseline SEDD : {baseline_f / 1e9:.2f} G FLOPs")
    print(f"  MrSEDD        : {mr_f / 1e9:.2f} G FLOPs")
    print(f"  Speedup       : {speedup:.2f}x")

    # Wall-clock profiling
    if args.profile:
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for profiling")
            return

        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg_dict = ckpt.get("config", {})
        try:
            from omegaconf import OmegaConf
            config = OmegaConf.structured(MrDiffusionConfig(**cfg_dict))
        except Exception:
            config = MrDiffusionConfig()

        model = MrSEDD(config).to(device)
        state = ckpt.get("model", ckpt.get("ema", ckpt))
        model.load_state_dict(state, strict=False)
        model.mr_config.deletion_mode = args.deletion_mode
        print(f"[inference] deletion_mode={args.deletion_mode}")

        ms = profile_wall_clock(model, config, device,
                                  batch_size=args.batch_size,
                                  seq_len=args.seq_len)
        print(f"\nWall-clock profiling ({args.batch_size} samples × {args.seq_len} tokens):")
        print(f"  {ms:.1f} ms/step")


if __name__ == "__main__":
    main()