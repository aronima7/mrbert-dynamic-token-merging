"""
Pareto frontier evaluation for MrBD3LM (plan Section 4.2, Experiment 6).

Plots generative perplexity vs total FLOPs for:
  - Baseline BD3-LM at 32, 64, 128, 256, 512, 1024 diffusion steps
  - MrBD3LM at same step counts

Usage
-----
python mrdiffusion-bd3lms/evaluation/eval_pareto.py \\
    --baseline_checkpoint ./baseline/final/checkpoint.pt \\
    --mr_checkpoint ./mrd_output/final/checkpoint.pt \\
    --seq_len 128 --batch_size 8 \\
    --output_dir ./pareto_results
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
_BD3LMS_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "bd3lms")
for p in [_MODULE_DIR, _BD3LMS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configuration_mrd_bd3lm import MrBD3LMConfig
from modeling_mrd_bd3lm import MrBD3LM
from noise_schedule import LogLinearNoise
from eval_flops import analytical_flops


def eval_gen_ppl(model, tokenizer, noise, mask_index, config, device,
                 n_samples=64, num_steps=128):
    """Compute generative perplexity at a given number of diffusion steps."""
    sys.path.insert(0, _MODULE_DIR)
    from train_mrd_bd3lm import compute_generative_perplexity
    return compute_generative_perplexity(
        model, tokenizer, noise, mask_index, config, device,
        n_samples=n_samples)


def compute_pareto_curve(model, config, tokenizer, noise, mask_index, device,
                          step_counts, n_samples, deletion_rate):
    """Return list of (total_flops_G, gen_ppl) for each step count."""
    flops_base, flops_mr, _ = analytical_flops(
        config.n_blocks, config.model_length, config.hidden_dim,
        config.delete_gate_layer, deletion_rate, config.n_heads)

    results = []
    for steps in step_counts:
        print(f"  Evaluating at {steps} diffusion steps...")
        ppl = eval_gen_ppl(model, tokenizer, noise, mask_index, config, device,
                            n_samples=n_samples, num_steps=steps)
        total_flops = steps * flops_mr / 1e9
        results.append({"steps": steps, "gen_ppl": ppl, "total_flops_G": total_flops})
        print(f"    steps={steps}, gen_ppl={ppl:.2f}, flops={total_flops:.1f}G")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_checkpoint", default=None)
    p.add_argument("--mr_checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--deletion_rate", type=float, default=0.3)
    p.add_argument("--output_dir", default="./pareto_results")
    p.add_argument("--step_counts", nargs="+", type=int,
                   default=[32, 64, 128, 256, 512])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise = LogLinearNoise().to(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # MrBD3LM
    ckpt = torch.load(args.mr_checkpoint, map_location=device)
    cfg_dict = ckpt.get("config", {})
    try:
        config = MrBD3LMConfig(**cfg_dict)
    except Exception:
        config = MrBD3LMConfig(model_length=args.seq_len)
    model = MrBD3LM(config).to(device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    mask_index = config.vocab_size - 1

    print(f"\nEvaluating MrBD3LM Pareto curve...")
    all_results["mrd"] = compute_pareto_curve(
        model, config, tokenizer, noise, mask_index, device,
        args.step_counts, args.n_samples, args.deletion_rate)

    # Baseline (if provided)
    if args.baseline_checkpoint:
        ckpt_base = torch.load(args.baseline_checkpoint, map_location=device)
        cfg_base = ckpt_base.get("config", {})
        try:
            config_base = MrBD3LMConfig(**cfg_base)
        except Exception:
            config_base = MrBD3LMConfig(model_length=args.seq_len,
                                         deletion_loss_weight=0.0)
        model_base = MrBD3LM(config_base).to(device)
        model_base.load_state_dict(ckpt_base.get("model", ckpt_base), strict=False)

        flops_base, _, _ = analytical_flops(
            config_base.n_blocks, config_base.model_length, config_base.hidden_dim,
            config_base.delete_gate_layer, 0.0, config_base.n_heads)

        print(f"\nEvaluating Baseline Pareto curve...")
        base_results = []
        for steps in args.step_counts:
            ppl = eval_gen_ppl(model_base, tokenizer, noise,
                                config_base.vocab_size - 1, config_base,
                                device, n_samples=args.n_samples, num_steps=steps)
            total_flops = steps * flops_base / 1e9
            base_results.append({"steps": steps, "gen_ppl": ppl,
                                   "total_flops_G": total_flops})
        all_results["baseline"] = base_results
        del model_base

    # Save
    out_path = os.path.join(args.output_dir, "pareto_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Pareto results to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"{'Steps':>8} | {'Gen PPL':>8} | {'FLOPs (G)':>10}")
        print("-" * 35)
        for r in results:
            print(f"{r['steps']:>8} | {r['gen_ppl']:>8.2f} | {r['total_flops_G']:>10.1f}")


if __name__ == "__main__":
    main()