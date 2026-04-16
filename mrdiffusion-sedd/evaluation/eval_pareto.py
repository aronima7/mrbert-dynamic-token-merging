"""
Pareto frontier evaluation for MrSEDD (plan Section 4.2, Experiment 6).

Plots generative perplexity vs total FLOPs for:
  - Baseline SEDD at 32, 64, 128, 256, 512 diffusion steps
  - MrSEDD at same step counts

Usage
-----
python mrdiffusion-sedd/evaluation/eval_pareto.py \
    --mr_checkpoint ./mrsedd_output/final/checkpoint.pt \
    --baseline_checkpoint ./sedd_baseline/checkpoint.pt \
    --seq_len 128 --batch_size 8 \
    --output_dir ./pareto_results_sedd
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
from eval_flops import analytical_flops


def eval_gen_ppl(model, tokenizer, config, device,
                 n_samples=64, num_steps=128, seq_len=128):
    """Compute generative perplexity at a given number of diffusion steps."""
    sys.path.insert(0, _MODULE_DIR)
    from train_mrdiffusion import compute_generative_perplexity
    return compute_generative_perplexity(
        model, tokenizer, config, device,
        n_samples=n_samples, num_steps=num_steps, seq_len=seq_len)


def compute_pareto_curve(model, config, tokenizer, device,
                          step_counts, n_samples, deletion_rate):
    """Return list of (total_flops_G, gen_ppl) for each step count."""
    _, mr_flops, _ = analytical_flops(
        config.n_blocks, config.model_length,
        config.hidden_dim, config.delete_gate_layer,
        deletion_rate, config.n_heads)

    results = []
    for steps in step_counts:
        print(f"  Evaluating at {steps} diffusion steps...")
        ppl = eval_gen_ppl(model, tokenizer, config, device,
                            n_samples=n_samples, num_steps=steps)
        total_flops = steps * mr_flops / 1e9
        results.append({"steps": steps, "gen_ppl": ppl, "total_flops_G": total_flops})
        print(f"    steps={steps}, gen_ppl={ppl:.2f}, flops={total_flops:.1f}G")
    return results


def _load_model(checkpoint_path, device, fallback_seq_len, deletion_mode="hard"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("config", {})
    try:
        config = MrDiffusionConfig(**cfg_dict)
    except Exception:
        config = MrDiffusionConfig()
    model = MrSEDD(config).to(device)
    state = ckpt.get("model", ckpt.get("ema", ckpt))
    model.load_state_dict(state, strict=False)
    model.mr_config.deletion_mode = deletion_mode
    print(f"[inference] deletion_mode={deletion_mode}")
    return model, config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mr_checkpoint", required=True)
    p.add_argument("--baseline_checkpoint", default=None)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--deletion_rate", type=float, default=0.3)
    p.add_argument("--output_dir", default="./pareto_results_sedd")
    p.add_argument("--step_counts", nargs="+", type=int,
                   default=[32, 64, 128, 256, 512])
    p.add_argument("--deletion_mode", default="hard", choices=["soft", "hard"],
                   help="Deletion mode at inference. 'hard' (default) physically removes tokens "
                        "as per guide: soft during training, hard at inference.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # MrSEDD
    model, config = _load_model(args.mr_checkpoint, device, args.seq_len,
                                 deletion_mode=args.deletion_mode)
    print(f"\nEvaluating MrSEDD Pareto curve...")
    all_results["mrsedd"] = compute_pareto_curve(
        model, config, tokenizer, device,
        args.step_counts, args.n_samples, args.deletion_rate)

    # Baseline SEDD (if provided)
    if args.baseline_checkpoint:
        del model
        model_base, config_base = _load_model(args.baseline_checkpoint, device, args.seq_len,
                                               deletion_mode=args.deletion_mode)
        baseline_flops, _, _ = analytical_flops(
            config_base.n_blocks, config_base.model_length,
            config_base.hidden_dim, config_base.delete_gate_layer,
            0.0, config_base.n_heads)

        print(f"\nEvaluating Baseline SEDD Pareto curve...")
        base_results = []
        for steps in args.step_counts:
            ppl = eval_gen_ppl(model_base, tokenizer, config_base, device,
                                n_samples=args.n_samples, num_steps=steps,
                                seq_len=args.seq_len)
            total_flops = steps * baseline_flops / 1e9
            base_results.append({"steps": steps, "gen_ppl": ppl,
                                   "total_flops_G": total_flops})
        all_results["baseline"] = base_results

    # Save
    out_path = os.path.join(args.output_dir, "pareto_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Pareto results to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"{'Steps':>8} | {'Gen PPL':>8} | {'FLOPs (G)':>10}")
        print("-" * 35)
        for r in results:
            print(f"{r['steps']:>8} | {r['gen_ppl']:>8.2f} | {r['total_flops_G']:>10.1f}")


if __name__ == "__main__":
    main()