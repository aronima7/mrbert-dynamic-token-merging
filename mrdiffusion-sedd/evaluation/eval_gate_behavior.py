"""
Gate behavior analysis for MrSEDD (plan Section 4.4).

Analyses:
    1. Deletion rate vs noise level sigma ∈ {0.01, 0.1, 0.5, 1.0, 2.0, 4.0}
       - Overall rate r(sigma)
       - Rate for [MASK] tokens r_mask(sigma)
       - Rate for clean tokens r_clean(sigma)

    2. Gate value distribution histograms at multiple noise levels

    3. Entropy of gate distribution (bimodality measure)

Usage
-----
python mrdiffusion-sedd/evaluation/eval_gate_behavior.py \
    --checkpoint ./mrsedd_output/final/checkpoint.pt \
    --dataset wikitext-2 --seq_len 128 --batch_size 8 \
    --output_dir ./gate_analysis_sedd
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_SEDD_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "Score-Entropy-Discrete-Diffusion")
for p in [_MODULE_DIR, _SEDD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configuration_mrdiffusion import MrDiffusionConfig
from modeling_mrdiffusion import MrSEDD


@torch.no_grad()
def analyze_deletion_vs_noise(model, dataset_loader, mask_index, config,
                               device, n_batches=50):
    """Compute deletion rates for clean vs masked tokens across noise levels."""
    # SEDD uses sigma = -log(1-t); sample representative sigma levels
    sigma_levels = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
    results = {s: {"overall": [], "mask": [], "clean": []} for s in sigma_levels}

    model.eval()
    deletion_threshold = getattr(config, "deletion_threshold", 0.5)

    for i, batch in enumerate(dataset_loader):
        if i >= n_batches:
            break
        x0 = batch["input_ids"].to(device)
        B, L = x0.shape

        for sigma_val in sigma_levels:
            sigma = torch.full((B,), sigma_val, device=device)
            t = 1.0 - torch.exp(-sigma)
            p = t.unsqueeze(-1)
            move_idx = torch.rand(B, L, device=device) < p
            xt = torch.where(move_idx, torch.tensor(mask_index, device=device), x0)

            out = model(xt, sigma)
            gate = getattr(out, "delete_gate_output", None)
            if gate is None:
                gate = model._last_gate_output if hasattr(model, "_last_gate_output") else None
            if gate is None:
                continue

            # gate in [0,1]; deletion = gate < threshold
            gate_del = (gate.squeeze(-1) < deletion_threshold).float()

            results[sigma_val]["overall"].append(gate_del.mean().item())

            mask_pos = move_idx.float()
            if mask_pos.sum() > 0:
                results[sigma_val]["mask"].append(
                    (gate_del * mask_pos).sum().item() / mask_pos.sum().item())

            clean_pos = (~move_idx).float()
            if clean_pos.sum() > 0:
                results[sigma_val]["clean"].append(
                    (gate_del * clean_pos).sum().item() / clean_pos.sum().item())

    summary = {}
    for s_val in sigma_levels:
        summary[s_val] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in results[s_val].items()
        }
    model.train()
    return summary


@torch.no_grad()
def collect_gate_values_at_noise(model, dataset_loader, mask_index, config,
                                  device, sigma_levels=(0.1, 0.5, 1.0, 2.0),
                                  n_batches=20):
    """Collect gate value distributions at specified noise levels."""
    gate_hist = {s: [] for s in sigma_levels}
    model.eval()

    for i, batch in enumerate(dataset_loader):
        if i >= n_batches:
            break
        x0 = batch["input_ids"].to(device)
        B, L = x0.shape

        for sigma_val in sigma_levels:
            sigma = torch.full((B,), sigma_val, device=device)
            t = 1.0 - torch.exp(-sigma)
            xt = torch.where(torch.rand(B, L, device=device) < t.unsqueeze(-1),
                             torch.tensor(mask_index, device=device), x0)

            out = model(xt, sigma)
            gate = getattr(out, "delete_gate_output", None)
            if gate is not None:
                gate_hist[sigma_val].extend(
                    gate.cpu().float().flatten().tolist())

    model.train()
    return gate_hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="wikitext-2")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_dir", default="./gate_analysis_sedd")
    p.add_argument("--n_batches", type=int, default=50)
    p.add_argument("--deletion_mode", default="soft", choices=["soft", "hard"],
                   help="Deletion mode for gate analysis. 'soft' (default) is recommended here "
                        "to observe what the gate would delete without changing the forward pass; "
                        "use 'hard' to measure gate behavior under actual inference conditions.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt.get("config", {})
    try:
        config = MrDiffusionConfig(**cfg_dict)
    except Exception:
        config = MrDiffusionConfig()

    model = MrSEDD(config).to(device)
    state = ckpt.get("model", ckpt.get("ema", ckpt))
    model.load_state_dict(state, strict=False)
    model.mr_config.deletion_mode = args.deletion_mode
    print(f"[inference] deletion_mode={args.deletion_mode}")

    mask_index = getattr(config, "mask_index", config.vocab_size - 1)

    # Data
    from datasets import load_dataset as hf_load
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.dataset.startswith("wikitext"):
        ds = hf_load("wikitext", args.dataset + "-raw-v1" if "-raw" not in args.dataset else args.dataset,
                     split="validation")
    else:
        ds = hf_load(args.dataset, split="validation")

    text_col = "text" if "text" in ds.column_names else ds.column_names[0]

    def tok(batch):
        return tokenizer(batch[text_col], truncation=True,
                         max_length=args.seq_len, padding="max_length")

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Analysis 1: deletion rate vs noise level
    print("[Analysis 1] Deletion rate vs noise level (sigma)...")
    del_vs_noise = analyze_deletion_vs_noise(
        model, loader, mask_index, config, device, n_batches=args.n_batches)

    print(f"\n{'sigma':>6} | {'overall':>8} | {'mask':>8} | {'clean':>8}")
    print("-" * 42)
    for s_val, stats in sorted(del_vs_noise.items()):
        print(f"{s_val:>6.3f} | {stats['overall']:>8.3f} | "
              f"{stats['mask']:>8.3f} | {stats['clean']:>8.3f}")

    with open(os.path.join(args.output_dir, "deletion_vs_noise.json"), "w") as f:
        json.dump({str(k): v for k, v in del_vs_noise.items()}, f, indent=2)

    # Analysis 2: gate value distributions
    print("\n[Analysis 2] Gate value distributions at various sigma levels...")
    gate_hist = collect_gate_values_at_noise(
        model, loader, mask_index, config, device)

    for s_val, vals in gate_hist.items():
        if vals:
            arr = np.array(vals)
            print(f"  sigma={s_val:.2f}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
                  f"p10={np.percentile(arr,10):.3f}, p90={np.percentile(arr,90):.3f}")
            np.save(os.path.join(args.output_dir, f"gate_vals_sigma{s_val:.2f}.npy"), arr)

    print(f"\n[done] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()