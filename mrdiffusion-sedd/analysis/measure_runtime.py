"""
measure_runtime.py

Measure per-step inference runtime (ms/step) for SEDD and MrSEDD checkpoints
with hard deletion at inference time.

For each model provided, runs N warmup passes then N timed forward passes and reports:
  - mean runtime (ms/step)
  - runtime decrease vs baseline SEDD (%)
  - actual mean deletion rate across sigma levels
  - effective sequence length after deletion
  - theoretical speedup (analytical FLOPs ratio)

Crucially, this measures a SINGLE denoising step — multiply by num_sampling_steps
for total sampling time. The savings compound: if each step is 1.3× faster,
sampling with 128 steps saves ~23% wall-clock time.

Usage (from mrdiffusion-sedd/ directory):

  # Compare baseline SEDD vs MrSEDD checkpoint (hard deletion at inference)
  python analysis/measure_runtime.py \
      --models "SEDD,./runs/baseline/checkpoints/best" \
              "MrSEDD-30%,./runs/soft-gate-fixes-v2/checkpoints/best" \
      --deletion_mode hard

  # Quick test on CPU
  python analysis/measure_runtime.py \
      --models "SEDD,./runs/baseline/final/checkpoint.pth" \
              "MrSEDD-30%,./runs/soft/final/checkpoint.pth" \
      --n_warmup 2 --n_timed 10 --batch_size 4

  # Multi-step sampling profile (measure full generation, not just per-step)
  python analysis/measure_runtime.py \
      --models "SEDD,./runs/baseline/final/checkpoint.pth" \
              "MrSEDD-30%,./runs/soft/final/checkpoint.pth" \
      --measure_sampling --num_steps 128

Output:
  Prints a runtime table to stdout.
  Saves analysis/figures/runtime_table.csv and runtime_comparison.pdf.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
_SEDD_ROOT = os.path.join(_PROJECT_ROOT, "diffusion", "Score-Entropy-Discrete-Diffusion")

for p in [_MODULE_DIR, _SEDD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configuration_mrdiffusion import MrDiffusionConfig
from modeling_mrdiffusion import MrSEDD


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str, deletion_mode: str = "hard"):
    """
    Load a SEDD or MrSEDD checkpoint.

    Expects SEDD-style checkpoint dict with keys: model, config, mr_config (optional).
    Falls back to loading state_dict directly if structure is different.
    """
    from omegaconf import OmegaConf
    from model import SEDD

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect checkpoint format
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        sedd_config = ckpt.get("config")
        mr_config_dict = ckpt.get("mr_config")
    else:
        state_dict = ckpt
        sedd_config = None
        mr_config_dict = None

    # Load SEDD config
    if sedd_config is None:
        config_path = os.path.join(_SEDD_ROOT, "configs", "config.yaml")
        model_cfg_path = os.path.join(_SEDD_ROOT, "configs", "model", "small.yaml")
        base_cfg = OmegaConf.load(config_path)
        model_cfg = OmegaConf.load(model_cfg_path)
        sedd_config = OmegaConf.merge(base_cfg, OmegaConf.create({"model": model_cfg}))

    # Determine if this is MrSEDD or vanilla SEDD
    has_gate_keys = any("delete_gate" in k for k in state_dict.keys())

    if has_gate_keys:
        mr_config = MrDiffusionConfig.from_dict(mr_config_dict) if mr_config_dict else MrDiffusionConfig()
        mr_config.deletion_mode = deletion_mode
        model = MrSEDD(sedd_config, mr_config).to(device)
    else:
        model = SEDD(sedd_config).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, has_gate_keys


# ---------------------------------------------------------------------------
# Per-step runtime measurement
# ---------------------------------------------------------------------------

def measure_per_step_runtime(
    model,
    is_mrsedd: bool,
    device: str,
    seq_len: int = 128,
    batch_size: int = 8,
    n_warmup: int = 10,
    n_timed: int = 100,
    vocab_size: int = 50257,
) -> dict:
    """
    Time individual forward passes at various sigma levels.

    Returns dict with:
      - mean_ms_per_step: float
      - std_ms_per_step: float
      - mean_deletion_rate: float (average across sigma levels; 0.0 for baseline)
      - deletion_rates_by_sigma: list of (sigma, deletion_rate) pairs
    """
    model.eval()

    # Generate inputs at varied sigma levels (simulating actual sampling trajectory)
    sigma_levels = torch.linspace(0.01, 19.0, n_timed, device=device)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        sigma_warmup = torch.full((batch_size,), 5.0, device=device)
        for _ in range(n_warmup):
            _ = model(dummy_input, sigma_warmup)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed passes
    step_times_ms = []
    deletion_rates = []
    sigma_values = []

    with torch.no_grad():
        for i in range(n_timed):
            sigma = sigma_levels[i].expand(batch_size)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            output = model(dummy_input, sigma)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            step_times_ms.append((t1 - t0) * 1000.0)

            # Extract deletion rate from gate output
            if is_mrsedd and hasattr(output, "delete_gate_output") and output.delete_gate_output is not None:
                gate_out = output.delete_gate_output
                sigmoid_mask_scale = model.mr_config.sigmoid_mask_scale
                rate = (gate_out / sigmoid_mask_scale).squeeze(-1).mean().item()
                deletion_rates.append(rate)
                sigma_values.append(sigma_levels[i].item())

    mean_ms = np.mean(step_times_ms)
    std_ms = np.std(step_times_ms)
    mean_del_rate = np.mean(deletion_rates) if deletion_rates else 0.0

    return {
        "mean_ms_per_step": mean_ms,
        "std_ms_per_step": std_ms,
        "mean_deletion_rate": mean_del_rate,
        "deletion_rates_by_sigma": list(zip(sigma_values, deletion_rates)),
    }


# ---------------------------------------------------------------------------
# Multi-step sampling measurement
# ---------------------------------------------------------------------------

def measure_sampling_runtime(
    model,
    is_mrsedd: bool,
    device: str,
    seq_len: int = 128,
    batch_size: int = 8,
    num_steps: int = 128,
    n_runs: int = 5,
    vocab_size: int = 50257,
) -> dict:
    """
    Time full sampling (num_steps denoising steps).

    Returns dict with:
      - mean_ms_total: float (total ms for num_steps)
      - std_ms_total: float
      - ms_per_step: float
    """
    model.eval()

    # Simulate sampling trajectory: sigma decreases from ~sigma_max to ~0
    sigma_schedule = torch.linspace(19.0, 0.01, num_steps, device=device)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup (one full trajectory)
    with torch.no_grad():
        for sigma_val in sigma_schedule[:min(10, num_steps)]:
            sigma = sigma_val.expand(batch_size)
            _ = model(dummy_input, sigma)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    run_times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            for sigma_val in sigma_schedule:
                sigma = sigma_val.expand(batch_size)
                _ = model(dummy_input, sigma)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            run_times.append((t1 - t0) * 1000.0)

    mean_total = np.mean(run_times)
    std_total = np.std(run_times)

    return {
        "mean_ms_total": mean_total,
        "std_ms_total": std_total,
        "ms_per_step": mean_total / num_steps,
    }


# ---------------------------------------------------------------------------
# Output: table + plots
# ---------------------------------------------------------------------------

def print_and_save_table(results: list, baseline_ms: float, output_dir: str):
    """Print runtime comparison table and save CSV."""
    print()
    print("=" * 95)
    print(f"  {'Model':<22}  {'ms/step':>9}  {'±':>6}  {'vs Baseline':>12}  {'Del Rate':>9}  {'Eff Seq':>8}  {'Speedup':>8}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*6}  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*8}")

    rows = []
    for r in results:
        ms = r["mean_ms"]
        std = r["std_ms"]
        pct_dec = (1.0 - ms / baseline_ms) * 100.0 if baseline_ms > 0 else 0.0
        del_rate = r["deletion_rate"]
        eff_seq = r["effective_seq_len"]
        speedup = baseline_ms / ms if ms > 0 else float("inf")

        pct_str = f"{pct_dec:+.1f}%" if r["label"] != results[0]["label"] else "baseline"
        del_str = f"{del_rate:.1%}" if del_rate > 0 else "n/a"
        seq_str = f"{eff_seq:.0f}" if eff_seq else "128"

        print(f"  {r['label']:<22}  {ms:>9.2f}  {std:>6.2f}  {pct_str:>12}  {del_str:>9}  {seq_str:>8}  {speedup:>7.2f}×")
        rows.append(r)

    print("=" * 95)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "runtime_table.csv")
    with open(csv_path, "w") as f:
        f.write("label,ms_per_step,std_ms,pct_decrease,deletion_rate,effective_seq_len,speedup\n")
        for r in rows:
            speedup = baseline_ms / r["mean_ms"] if r["mean_ms"] > 0 else 0
            pct = (1.0 - r["mean_ms"] / baseline_ms) * 100.0 if baseline_ms > 0 else 0
            f.write(f"{r['label']},{r['mean_ms']:.3f},{r['std_ms']:.3f},{pct:.2f},"
                    f"{r['deletion_rate']:.4f},{r['effective_seq_len']:.1f},{speedup:.3f}\n")
    print(f"\nSaved → {csv_path}")


def plot_runtime_comparison(results: list, baseline_ms: float, output_dir: str):
    """Bar chart: ms/step for each model."""
    labels = [r["label"] for r in results]
    means = [r["mean_ms"] for r in results]
    stds = [r["std_ms"] for r in results]
    colors = ["#377eb8" if r["deletion_rate"] == 0 else "#e41a1c" for r in results]

    fig, ax = plt.subplots(figsize=(max(5, len(results) * 1.5), 4))
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, width=0.6, zorder=3)

    ax.axhline(y=baseline_ms, color="#377eb8", linestyle="--", linewidth=1.2,
               label=f"Baseline ({baseline_ms:.1f} ms/step)", zorder=2)

    for i, (bar, ms) in enumerate(zip(bars, means)):
        if ms < baseline_ms and baseline_ms > 0:
            pct = (1.0 - ms / baseline_ms) * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stds[i] + 0.1,
                    f"−{pct:.1f}%", ha="center", va="bottom", fontsize=8, color="#e41a1c")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Inference time (ms / denoising step)")
    ax.set_title("Per-step runtime: SEDD vs MrSEDD (hard deletion)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(means) * 1.3)

    path = os.path.join(output_dir, "runtime_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_deletion_rate_vs_sigma(all_sigma_data: dict, output_dir: str):
    """Plot deletion rate vs sigma for each MrSEDD model."""
    if not all_sigma_data:
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3"]

    for i, (label, data) in enumerate(all_sigma_data.items()):
        sigmas = [d[0] for d in data]
        rates = [d[1] for d in data]
        color = colors[i % len(colors)]
        ax.scatter(sigmas, rates, color=color, s=15, alpha=0.6, label=label)
        # Smoothed line
        if len(sigmas) > 5:
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(sigmas, rates, 2)
            x_smooth = np.linspace(min(sigmas), max(sigmas), 50)
            y_smooth = P.polyval(x_smooth, coeffs)
            ax.plot(x_smooth, y_smooth, color=color, linewidth=2)

    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("Deletion rate")
    ax.set_title("Gate deletion rate vs noise level (inference)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.8)

    path = os.path.join(output_dir, "deletion_rate_vs_sigma.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Measure MrSEDD vs SEDD inference runtime.")
    p.add_argument("--models", nargs="+", metavar="LABEL,PATH", required=True,
                   help="Models to benchmark. First entry is baseline. Format: 'label,/path/to/checkpoint.pth'")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_warmup", type=int, default=10)
    p.add_argument("--n_timed", type=int, default=100)
    p.add_argument("--deletion_mode", default="hard", choices=["soft", "hard"],
                   help="Deletion mode at inference. 'hard' gives real speedup.")
    p.add_argument("--measure_sampling", action="store_true",
                   help="Also measure full sampling (num_steps forward passes).")
    p.add_argument("--num_steps", type=int, default=128,
                   help="Number of sampling steps for --measure_sampling.")
    p.add_argument("--output_dir", type=str, default="analysis/figures")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Deletion mode: {args.deletion_mode}")
    print(f"Batch size: {args.batch_size}, seq_len: {args.seq_len}")

    model_entries = []
    for entry in args.models:
        parts = entry.split(",", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed --models entry {entry!r}. Expected 'LABEL,PATH'.")
        model_entries.append({"label": parts[0].strip(), "path": parts[1].strip()})

    all_results = []
    all_sigma_data = {}

    for entry in model_entries:
        label = entry["label"]
        path = entry["path"]
        print(f"\nLoading {label} from {path} ...")

        model, is_mrsedd = load_model(path, device, args.deletion_mode)
        print(f"  Model type: {'MrSEDD' if is_mrsedd else 'SEDD baseline'}")

        stats = measure_per_step_runtime(
            model, is_mrsedd, device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )

        del_rate = stats["mean_deletion_rate"]
        eff_seq = args.seq_len * (1.0 - del_rate)

        print(f"  {label}: {stats['mean_ms_per_step']:.2f} ± {stats['std_ms_per_step']:.2f} ms/step"
              f"  |  deletion_rate: {del_rate:.1%}  |  eff_seq: {eff_seq:.0f}")

        all_results.append({
            "label": label,
            "mean_ms": stats["mean_ms_per_step"],
            "std_ms": stats["std_ms_per_step"],
            "deletion_rate": del_rate,
            "effective_seq_len": eff_seq,
        })

        if stats["deletion_rates_by_sigma"]:
            all_sigma_data[label] = stats["deletion_rates_by_sigma"]

        # Multi-step sampling measurement
        if args.measure_sampling:
            print(f"  Measuring full sampling ({args.num_steps} steps) ...")
            samp_stats = measure_sampling_runtime(
                model, is_mrsedd, device,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                n_runs=3,
            )
            print(f"    Total: {samp_stats['mean_ms_total']:.0f} ± {samp_stats['std_ms_total']:.0f} ms"
                  f"  |  {samp_stats['ms_per_step']:.2f} ms/step")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    baseline_ms = all_results[0]["mean_ms"]

    os.makedirs(args.output_dir, exist_ok=True)
    print_and_save_table(all_results, baseline_ms, args.output_dir)
    plot_runtime_comparison(all_results, baseline_ms, args.output_dir)
    plot_deletion_rate_vs_sigma(all_sigma_data, args.output_dir)

    # Print multi-step extrapolation
    if not args.measure_sampling:
        print(f"\n  Extrapolated sampling time ({args.num_steps} steps):")
        for r in all_results:
            total = r["mean_ms"] * args.num_steps
            speedup = (baseline_ms * args.num_steps) / total if total > 0 else 0
            print(f"    {r['label']:<22}: {total/1000:.2f}s  ({speedup:.2f}× vs baseline)")

    print("\nDone.")


if __name__ == "__main__":
    main()
