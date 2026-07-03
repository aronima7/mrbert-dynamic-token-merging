"""
compute_savings.py

Theoretical MACs saved by token deletion in MrSEDD as a function of deletion
ratio, accounting for the multi-step iterative sampling unique to diffusion.

Key differences from MrBERT compute_savings.py:
  - Diffusion sampling runs N denoising steps; savings multiply by N
  - Deletion rate varies with sigma (noise level) via linear_sigma schedule,
    so effective savings depend on the sigma distribution during sampling
  - DDiT architecture: hidden=768, n_heads=12, mlp_ratio=4, 12 blocks (small)
  - Gate at layer 3 by default (same as MrBERT convention)

Produces figures saved to analysis/figures/:
  1. macs_relative.pdf              — relative compute vs deletion ratio (gate layer 3)
  2. macs_by_gate_layer.pdf         — relative compute vs deletion ratio for layers 1,3,6,9
  3. macs_multistep.pdf             — total sampling FLOPs: baseline SEDD vs MrSEDD at various
                                      step counts (32, 64, 128, 256, 512)
  4. macs_sigma_schedule.pdf        — effective FLOPs savings under linear_sigma schedule
                                      vs constant deletion rate
  5. perplexity_vs_compute.pdf      — perplexity vs relative compute (requires --runs)
  6. gate_layer_ablation.pdf        — perplexity + runtime vs gate layer (requires --gate-layer-runs)

Usage:
    # Theoretical curves only (no run data needed)
    python analysis/compute_savings.py

    # With run data
    python analysis/compute_savings.py \
        --runs "SEDD,95.2,0.0" "MrSEDD-30%,96.1,0.30" "MrSEDD-50%,98.5,0.50"

    # Gate layer ablation
    python analysis/compute_savings.py \
        --gate-layer-runs "Layer 1,96.0,8.2,1" "Layer 3,96.5,6.8,3" "Layer 6,97.2,5.5,6"
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# MACs formulae (DDiT architecture)
# ---------------------------------------------------------------------------

def ddit_block_macs(seq_len: int, hidden_dim: int = 768, mlp_ratio: int = 4, n_heads: int = 12) -> float:
    """
    Approximate MACs for one DDiTBlock at a given sequence length.

    Attention:
      - Q, K, V projections (combined): 3 * seq * d²
      - O projection:                    seq * d²
      - QK^T + softmax·V:               2 * seq² * d
    FFN (mlp_ratio=4):
      - Up + down projections:           2 * seq * d * (mlp_ratio * d)
    AdaLN modulation:
      - Linear(cond_dim, 6*d):           ~6 * d² (per step, amortized over seq)
    """
    d = hidden_dim
    L = seq_len

    attn_proj = 4 * L * d * d
    attn_score = 2 * L * L * d
    ffn = 2 * L * d * (mlp_ratio * d)
    adaln = 6 * d * d

    return attn_proj + attn_score + ffn + adaln


def sedd_forward_macs(
    seq_len: int,
    deletion_rate: float = 0.0,
    gate_layer: int = 3,
    n_blocks: int = 12,
    hidden_dim: int = 768,
    mlp_ratio: int = 4,
    n_heads: int = 12,
) -> float:
    """
    Total MACs for one forward pass of MrSEDD (or baseline SEDD if deletion_rate=0).

    Blocks 0..gate_layer run at full seq_len.
    Blocks (gate_layer+1)..n_blocks-1 run at seq_len * (1 - deletion_rate).
    Gate overhead: ~2 * seq * d² (LayerNorm + Linear + sigma projection).
    """
    L = seq_len
    L_prime = int(L * (1.0 - deletion_rate))
    n_before = gate_layer + 1
    n_after = n_blocks - n_before

    before = ddit_block_macs(L, hidden_dim, mlp_ratio, n_heads) * n_before
    after = ddit_block_macs(L_prime, hidden_dim, mlp_ratio, n_heads) * n_after

    gate_overhead = 2 * L * hidden_dim * hidden_dim if deletion_rate > 0 else 0

    return before + after + gate_overhead


def sedd_baseline_macs(
    seq_len: int,
    n_blocks: int = 12,
    hidden_dim: int = 768,
    mlp_ratio: int = 4,
    n_heads: int = 12,
) -> float:
    """Total MACs for one forward pass of baseline SEDD (no deletion)."""
    return ddit_block_macs(seq_len, hidden_dim, mlp_ratio, n_heads) * n_blocks


def sampling_total_macs(
    seq_len: int,
    deletion_rate: float,
    num_steps: int,
    gate_layer: int = 3,
    n_blocks: int = 12,
    hidden_dim: int = 768,
) -> float:
    """Total MACs for full sampling (num_steps denoising steps)."""
    per_step = sedd_forward_macs(seq_len, deletion_rate, gate_layer, n_blocks, hidden_dim)
    return per_step * num_steps


def effective_deletion_rate_linear_sigma(
    r_min: float, r_max: float, num_steps: int, sigma_max: float = 20.0
) -> float:
    """
    Average deletion rate across all sampling steps under linear_sigma schedule.

    During sampling, sigma decreases from sigma_max to ~0 over num_steps.
    r(sigma) = r_min + (r_max - r_min) * (sigma / sigma_max)
    Average over uniform sigma grid: (r_min + r_max) / 2.
    """
    return (r_min + r_max) / 2.0


def macs_with_sigma_schedule(
    seq_len: int,
    r_min: float,
    r_max: float,
    num_steps: int,
    sigma_max: float = 20.0,
    gate_layer: int = 3,
    n_blocks: int = 12,
    hidden_dim: int = 768,
) -> float:
    """
    Total MACs for sampling with sigma-dependent deletion rate.

    At each step t, sigma = sigma_max * (1 - t/num_steps) approximately.
    r(sigma) = r_min + (r_max - r_min) * (sigma / sigma_max).
    We sum per-step MACs with the actual deletion rate at that step.
    """
    total = 0.0
    for t in range(num_steps):
        frac = 1.0 - t / num_steps
        sigma = sigma_max * frac
        rate = r_min + (r_max - r_min) * (sigma / sigma_max)
        total += sedd_forward_macs(seq_len, rate, gate_layer, n_blocks, hidden_dim)
    return total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_macs_vs_deletion_ratio(seq_len: int, gate_layer: int, output_dir: str):
    """Figure 1: relative compute vs deletion ratio for one gate layer (single step)."""
    deletion_ratios = np.linspace(0, 0.8, 81)
    baseline = sedd_baseline_macs(seq_len)
    relative = [sedd_forward_macs(seq_len, dr, gate_layer) / baseline for dr in deletion_ratios]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(deletion_ratios, relative, color="steelblue", linewidth=2)
    ax.axvline(x=0.30, color="gray", linestyle="--", linewidth=1, label="30% deletion")
    ax.axvline(x=0.50, color="gray", linestyle=":", linewidth=1, label="50% deletion")

    for dr in (0.30, 0.50):
        rel = sedd_forward_macs(seq_len, dr, gate_layer) / baseline
        ax.annotate(f"{rel:.3f}×", xy=(dr, rel), xytext=(dr + 0.02, rel + 0.02), fontsize=8)

    ax.set_xlabel("Deletion ratio (δ)")
    ax.set_ylabel("Compute vs baseline SEDD")
    ax.set_title(f"MrSEDD per-step compute savings\n(gate at layer {gate_layer}, seq_len={seq_len})")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0.2, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "macs_relative.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_macs_by_gate_layer(seq_len: int, output_dir: str):
    """Figure 2: relative compute vs deletion ratio for multiple gate layers."""
    deletion_ratios = np.linspace(0, 0.8, 81)
    baseline = sedd_baseline_macs(seq_len)
    gate_layers = [1, 3, 6, 9]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for gl, color in zip(gate_layers, colors):
        relative = [sedd_forward_macs(seq_len, dr, gl) / baseline for dr in deletion_ratios]
        ax.plot(deletion_ratios, relative, color=color, linewidth=2, label=f"Gate after layer {gl}")

    ax.set_xlabel("Deletion ratio (δ)")
    ax.set_ylabel("Compute vs baseline SEDD")
    ax.set_title(f"MrSEDD compute savings by gate layer\n(seq_len={seq_len})")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0.2, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "macs_by_gate_layer.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_macs_multistep(seq_len: int, gate_layer: int, output_dir: str):
    """Figure 3: total sampling FLOPs — baseline vs MrSEDD at various step counts."""
    step_counts = [32, 64, 128, 256, 512]
    deletion_rates = [0.0, 0.20, 0.30, 0.50]
    colors = ["#333333", "#4daf4a", "#377eb8", "#e41a1c"]
    labels = ["Baseline (0%)", "MrSEDD (20%)", "MrSEDD (30%)", "MrSEDD (50%)"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for dr, color, label in zip(deletion_rates, colors, labels):
        total_flops = [sampling_total_macs(seq_len, dr, ns, gate_layer) / 1e12 for ns in step_counts]
        ax.plot(step_counts, total_flops, color=color, linewidth=2, marker="o", markersize=5, label=label)

    ax.set_xlabel("Number of sampling steps")
    ax.set_ylabel("Total TFLOPs")
    ax.set_title(f"Total sampling compute: SEDD vs MrSEDD\n(seq_len={seq_len}, gate layer {gate_layer})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    path = os.path.join(output_dir, "macs_multistep.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_macs_sigma_schedule(seq_len: int, gate_layer: int, output_dir: str):
    """Figure 4: effective savings under linear_sigma schedule vs constant rate."""
    num_steps = 128
    sigma_max = 20.0

    constant_rates = np.linspace(0.05, 0.6, 12)
    baseline_total = sampling_total_macs(seq_len, 0.0, num_steps, gate_layer)

    # Constant schedule savings
    constant_savings = []
    for dr in constant_rates:
        total = sampling_total_macs(seq_len, dr, num_steps, gate_layer)
        constant_savings.append(1.0 - total / baseline_total)

    # Linear sigma schedule: vary r_max (r_min fixed at 0.05)
    r_min = 0.05
    r_maxes = np.linspace(0.1, 0.7, 12)
    schedule_savings = []
    schedule_avg_rates = []
    for r_max in r_maxes:
        total = macs_with_sigma_schedule(seq_len, r_min, r_max, num_steps, sigma_max, gate_layer)
        schedule_savings.append(1.0 - total / baseline_total)
        schedule_avg_rates.append(effective_deletion_rate_linear_sigma(r_min, r_max, num_steps, sigma_max))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(constant_rates, constant_savings, color="#377eb8", linewidth=2,
            marker="o", markersize=4, label="Constant rate")
    ax.plot(schedule_avg_rates, schedule_savings, color="#e41a1c", linewidth=2,
            marker="s", markersize=4, label="Linear σ schedule\n(r_min=0.05, vary r_max)")

    ax.set_xlabel("Average deletion rate")
    ax.set_ylabel("FLOPs savings (fraction)")
    ax.set_title(f"Sampling FLOPs savings: constant vs σ-schedule\n({num_steps} steps, seq_len={seq_len})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.6)

    path = os.path.join(output_dir, "macs_sigma_schedule.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_perplexity_vs_compute(runs: list, seq_len: int, gate_layer: int, output_dir: str):
    """
    Figure 5: perplexity vs relative compute for each model run.

    `runs` is a list of (label, perplexity, deletion_rate) tuples.
    """
    baseline = sedd_baseline_macs(seq_len)

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#377eb8", "#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]

    for i, (label, ppl, deletion_rate) in enumerate(runs):
        rel_compute = sedd_forward_macs(seq_len, deletion_rate, gate_layer) / baseline
        color = colors[i % len(colors)]
        ax.scatter(rel_compute, ppl, color=color, s=80, zorder=3)
        ax.annotate(label, xy=(rel_compute, ppl),
                    xytext=(rel_compute + 0.01, ppl + 0.5), fontsize=9, color=color)

    ax.set_xlabel("Relative compute vs baseline SEDD")
    ax.set_ylabel("Generative perplexity")
    ax.set_title("Quality vs compute tradeoff (MrSEDD)")
    ax.set_xlim(0.3, 1.1)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    path = os.path.join(output_dir, "perplexity_vs_compute.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_gate_layer_ablation(gate_layer_runs: list, output_dir: str):
    """
    Figure 6: perplexity + runtime vs gate layer — dual y-axis.

    `gate_layer_runs` is a list of (label, perplexity, runtime_ms, gate_layer) tuples.
    """
    gate_layer_runs_sorted = sorted(gate_layer_runs, key=lambda x: x[3])
    layers = [gl for _, _, _, gl in gate_layer_runs_sorted]
    ppls = [p for _, p, _, _ in gate_layer_runs_sorted]
    runtimes = [rt for _, _, rt, _ in gate_layer_runs_sorted]

    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()

    color_ppl = "#e41a1c"
    color_runtime = "#377eb8"

    ax1.plot(layers, ppls, color=color_ppl, linewidth=2, marker="o", markersize=6, label="Perplexity")
    ax2.plot(layers, runtimes, color=color_runtime, linewidth=2, marker="s", markersize=6,
             linestyle="--", label="Runtime (ms/step)")

    ax1.set_xlabel("Delete gate layer (0-indexed)")
    ax1.set_ylabel("Generative perplexity", color=color_ppl)
    ax2.set_ylabel("Inference runtime (ms/step)", color=color_runtime)
    ax1.tick_params(axis="y", labelcolor=color_ppl)
    ax2.tick_params(axis="y", labelcolor=color_runtime)
    ax1.set_xticks(layers)
    ax1.set_title("Perplexity and runtime vs delete gate layer\n(MrSEDD, hard deletion at inference)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "gate_layer_ablation.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_savings_table(seq_len: int, gate_layer: int, num_steps: int):
    """Print compute savings at common deletion ratios."""
    baseline_step = sedd_baseline_macs(seq_len)
    baseline_total = baseline_step * num_steps

    print(f"\nMrSEDD compute savings (seq_len={seq_len}, gate layer {gate_layer}, {num_steps} sampling steps)")
    print(f"{'Deletion':>10}  {'Per-step':>12}  {f'Total ({num_steps} steps)':>18}  {'Savings':>10}  {'Speedup':>8}")
    print("-" * 68)

    for dr in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        per_step = sedd_forward_macs(seq_len, dr, gate_layer)
        total = per_step * num_steps
        savings = 1.0 - total / baseline_total
        speedup = baseline_total / total if total > 0 else float("inf")
        print(f"{dr:>9.0%}  {per_step/1e9:>10.2f} G  {total/1e12:>14.3f} T  {savings:>9.1%}  {speedup:>7.2f}×")

    # Also show sigma schedule
    print(f"\n  With linear_sigma (r_min=0.05, r_max=0.50, avg≈27.5%):")
    total_sched = macs_with_sigma_schedule(seq_len, 0.05, 0.50, num_steps, 20.0, gate_layer)
    savings_sched = 1.0 - total_sched / baseline_total
    speedup_sched = baseline_total / total_sched
    print(f"    Total: {total_sched/1e12:.3f} T  |  Savings: {savings_sched:.1%}  |  Speedup: {speedup_sched:.2f}×")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot MrSEDD theoretical compute savings.")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--gate_layer", type=int, default=3)
    p.add_argument("--n_blocks", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_steps", type=int, default=128,
                   help="Number of sampling steps for multi-step analysis.")
    p.add_argument("--output_dir", type=str, default="analysis/figures")
    p.add_argument("--runs", nargs="+", metavar="LABEL,PPL,DEL_RATE", default=None,
                   help="Runs for perplexity-vs-compute plot. Each: label,perplexity,deletion_rate")
    p.add_argument("--gate-layer-runs", nargs="+", dest="gate_layer_runs",
                   metavar="LABEL,PPL,RUNTIME_MS,GATE_LAYER", default=None,
                   help="Runs for gate layer ablation. Each: label,ppl,runtime_ms,gate_layer")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print_savings_table(args.seq_len, args.gate_layer, args.num_steps)
    plot_macs_vs_deletion_ratio(args.seq_len, args.gate_layer, args.output_dir)
    plot_macs_by_gate_layer(args.seq_len, args.output_dir)
    plot_macs_multistep(args.seq_len, args.gate_layer, args.output_dir)
    plot_macs_sigma_schedule(args.seq_len, args.gate_layer, args.output_dir)

    if args.runs:
        runs = []
        for entry in args.runs:
            parts = entry.split(",")
            if len(parts) != 3:
                print(f"Warning: skipping malformed --runs entry {entry!r}")
                continue
            runs.append((parts[0], float(parts[1]), float(parts[2])))
        if runs:
            plot_perplexity_vs_compute(runs, args.seq_len, args.gate_layer, args.output_dir)
    else:
        print("\nSkipping perplexity_vs_compute — no --runs provided.")
        print("  Example: --runs 'SEDD,95.2,0.0' 'MrSEDD-30%,96.1,0.30'")

    if args.gate_layer_runs:
        gl_runs = []
        for entry in args.gate_layer_runs:
            parts = entry.split(",")
            if len(parts) != 4:
                print(f"Warning: skipping malformed --gate-layer-runs entry {entry!r}")
                continue
            gl_runs.append((parts[0], float(parts[1]), float(parts[2]), int(parts[3])))
        if gl_runs:
            plot_gate_layer_ablation(gl_runs, args.output_dir)
    else:
        print("\nSkipping gate_layer_ablation — no --gate-layer-runs provided.")

    print("\nDone.")


if __name__ == "__main__":
    main()
