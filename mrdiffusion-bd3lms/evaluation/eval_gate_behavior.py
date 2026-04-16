"""
Gate behavior analysis for MrBD3LM (plan Section 4.4).

Analyses:
    1. Deletion rate vs noise level t ∈ {0.0, 0.1, ..., 1.0}
       - Overall rate r(t)
       - Rate for [MASK] tokens r_mask(t)
       - Rate for clean tokens r_clean(t)

    2. Gate value distribution histograms at t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}

    3. Token-level heatmap (5 example sequences)

    4. PI controller convergence: |r_target(t) - r_actual(t)| vs training step
       (requires a W&B run name or a log file)

Usage
-----
python mrdiffusion-bd3lms/evaluation/eval_gate_behavior.py \\
    --checkpoint ./mrd_bd3lm_output/final/checkpoint.pt \\
    --dataset wikitext-2-v1 --seq_len 128 --batch_size 8 \\
    --output_dir ./gate_analysis
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_BD3LMS_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "bd3lms")
for p in [_MODULE_DIR, _BD3LMS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configuration_mrd_bd3lm import MrBD3LMConfig
from modeling_mrd_bd3lm import MrBD3LM
from noise_schedule import LogLinearNoise


@torch.no_grad()
def analyze_deletion_vs_noise(model, dataset_loader, noise, mask_index,
                               config, device, n_noise_levels=11, n_batches=50):
    """Compute deletion rates for clean vs masked tokens across noise levels."""
    noise_levels = np.linspace(0.0, 1.0, n_noise_levels)
    results = {t: {"overall": [], "mask": [], "clean": []} for t in noise_levels}

    model.eval()
    for i, batch in enumerate(dataset_loader):
        if i >= n_batches:
            break
        x0 = batch["input_ids"].to(device)
        B, L = x0.shape

        for t_val in noise_levels:
            t = torch.full((B,), t_val, device=device).clamp(1e-3, 1 - 1e-3)
            _, move_chance = noise(t)
            p = move_chance.unsqueeze(-1)
            move_idx = torch.rand(B, L, device=device) < p
            xt = torch.where(move_idx, mask_index, x0)

            if config.cross_attn:
                x_input = torch.cat([xt, x0], dim=-1)
            else:
                x_input = xt

            model(x_input, timesteps=t)
            gate = model.backbone._last_gate_output
            if gate is None:
                continue

            gate_del = (gate.squeeze(-1) <= config.deletion_threshold).float()  # [B, L]

            # Overall
            results[t_val]["overall"].append(gate_del.mean().item())
            # Mask tokens
            mask_pos = move_idx.float()
            if mask_pos.sum() > 0:
                results[t_val]["mask"].append(
                    (gate_del * mask_pos).sum().item() / mask_pos.sum().item())
            # Clean tokens
            clean_pos = (~move_idx).float()
            if clean_pos.sum() > 0:
                results[t_val]["clean"].append(
                    (gate_del * clean_pos).sum().item() / clean_pos.sum().item())

    # Average
    summary = {}
    for t_val in noise_levels:
        summary[t_val] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in results[t_val].items()
        }
    model.train()
    return summary


@torch.no_grad()
def collect_gate_logits_at_noise(model, dataset_loader, noise, mask_index,
                                  config, device, t_levels=(0.1, 0.3, 0.5, 0.7, 0.9),
                                  n_batches=20):
    """Collect gate logit distributions at specified noise levels."""
    logit_hist = {t: [] for t in t_levels}
    model.eval()

    for i, batch in enumerate(dataset_loader):
        if i >= n_batches:
            break
        x0 = batch["input_ids"].to(device)
        B, L = x0.shape

        for t_val in t_levels:
            t = torch.full((B,), t_val, device=device).clamp(1e-3, 1 - 1e-3)
            _, move_chance = noise(t)
            p = move_chance.unsqueeze(-1)
            xt = torch.where(torch.rand(B, L, device=device) < p, mask_index, x0)

            x_input = torch.cat([xt, x0], dim=-1) if config.cross_attn else xt
            model(x_input, timesteps=t)
            gate_logits = getattr(model.backbone, "_last_gate_logits", None)
            if gate_logits is not None:
                logit_hist[t_val].extend(
                    gate_logits.cpu().float().flatten().tolist())

    model.train()
    return logit_hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="wikitext-2-v1")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_dir", default="./gate_analysis")
    p.add_argument("--n_batches", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt.get("config", {})
    try:
        config = MrBD3LMConfig(**cfg_dict)
    except Exception:
        config = MrBD3LMConfig(model_length=args.seq_len)

    model = MrBD3LM(config).to(device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    noise = LogLinearNoise().to(device)
    mask_index = config.vocab_size - 1

    # Data
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
    from train_mrd_bd3lm import load_dataset
    _, val_loader, _ = load_dataset(
        args.dataset, seq_len=args.seq_len,
        batch_size=args.batch_size, eval_batch_size=args.batch_size)

    # Analysis 1: deletion rate vs noise level
    print("[Analysis 1] Deletion rate vs noise level...")
    del_vs_noise = analyze_deletion_vs_noise(
        model, val_loader, noise, mask_index, config, device,
        n_batches=args.n_batches)

    print(f"\n{'t':>5} | {'overall':>8} | {'mask':>8} | {'clean':>8}")
    print("-" * 40)
    for t_val, stats in sorted(del_vs_noise.items()):
        print(f"{t_val:>5.2f} | {stats['overall']:>8.3f} | "
              f"{stats['mask']:>8.3f} | {stats['clean']:>8.3f}")

    # Save
    import json
    with open(os.path.join(args.output_dir, "deletion_vs_noise.json"), "w") as f:
        json.dump(del_vs_noise, f, indent=2)

    # Analysis 2: gate logit distributions
    print("\n[Analysis 2] Gate logit distributions...")
    logit_hist = collect_gate_logits_at_noise(
        model, val_loader, noise, mask_index, config, device)

    for t_val, logits in logit_hist.items():
        if logits:
            arr = np.array(logits)
            print(f"  t={t_val:.1f}: mean={arr.mean():.2f}, std={arr.std():.2f}, "
                  f"p10={np.percentile(arr,10):.2f}, p90={np.percentile(arr,90):.2f}")
            np.save(os.path.join(args.output_dir, f"logits_t{t_val:.1f}.npy"), arr)

    print(f"\n[done] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()