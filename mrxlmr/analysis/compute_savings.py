"""
compute_savings.py

Plots theoretical MACs (multiply-accumulate operations) saved by token deletion
in MrXLMR as a function of deletion ratio, mirroring mrt5/analysis/compute_savings.ipynb
adapted for the XLM-RoBERTa encoder-only architecture.

Key differences from MrT5:
  - Encoder-only (no decoder, no cross-attention)
  - XLM-R-base dimensions: d_model=768, d_ff=3072, 12 encoder layers
  - Gate at layer 3 by default (0-indexed: deletion happens after layer 3)
  - Architecture identical to BERT-base; MACs formulas are the same

Produces figures saved to analysis/figures/:
  1. macs_relative.pdf              — relative compute vs deletion ratio for gate at layer 3
  2. macs_by_gate_layer.pdf         — relative compute vs deletion ratio for gate at layers 1, 3, 6, 9
  3. accuracy_vs_compute.pdf        — accuracy vs relative compute for each run (requires --runs)
  4. accuracy_vs_seq_reduction.pdf  — accuracy vs sequence length reduction % (requires --runs)
                                      mirrors MrT5 paper Figure 2
  5. gate_layer_ablation.pdf        — accuracy + runtime vs gate layer (requires --gate-layer-runs)
                                      mirrors MrT5 paper Figure 4

Usage:
    # Theoretical curves only (no run data needed)
    python analysis/compute_savings.py

    # Include accuracy plots once you have eval results
    python analysis/compute_savings.py \\
        --runs "XLM-R,0.9055,0.0" "MrXLMR-30%,0.8141,0.32" "MrXLMR-50%,0.78,0.50" "MrXLMR-70%,0.72,0.70"

    # Gate layer ablation chart (accuracy + runtime vs layer)
    python analysis/compute_savings.py \\
        --gate-layer-runs \\
            "Layer 1,0.88,32.1,1" \\
            "Layer 3,0.87,28.5,3" \\
            "Layer 6,0.85,24.1,6" \\
            "Layer 9,0.82,20.8,9"

    # Adjust seq_len to your actual mean post-deletion sequence length
    python analysis/compute_savings.py --seq_len 27 \\
        --runs "XLM-R,0.9055,0.0" "MrXLMR-30%,0.8141,0.32"
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# MACs formulae
# ---------------------------------------------------------------------------

def transformer_layer_macs(d_model: int, d_ff: int, seq_len: int) -> int:
    """
    Approximate MACs for one transformer encoder layer at a given sequence length.

    Attention (assuming d_head * n_heads = d_model):
      - Q, K, V, O projections: 4 * seq * d_model^2
      - QK^T and weighted-sum:   2 * seq^2 * d_model

    FFN (two linear projections):
      - 2 * seq * d_model * d_ff
    """
    att_macs = (
        seq_len * d_model * d_model * 4 +   # Q, K, V, O projections
        seq_len * d_model * seq_len * 2      # QK^T and (QK^T)V
    )
    ffn_macs = seq_len * d_ff * d_model * 2
    return att_macs + ffn_macs


def xlmr_macs(
    seq_len: int,
    keep_ratio: float,
    gate_layer: int = 3,
    n_layers: int = 12,
    d_model: int = 768,
    d_ff: int = 3072,
) -> int:
    """
    Total MACs for MrXLMR with a delete gate after layer `gate_layer`.

    Layers 0..gate_layer run at full seq_len.
    Layers (gate_layer+1)..n_layers-1 run at seq_len * keep_ratio.
    """
    reduced_seq = int(seq_len * keep_ratio)
    n_before = gate_layer + 1
    n_after  = n_layers - n_before

    before = transformer_layer_macs(d_model, d_ff, seq_len)     * n_before
    after  = transformer_layer_macs(d_model, d_ff, reduced_seq) * n_after
    return before + after


def xlmr_baseline_macs(
    seq_len: int,
    n_layers: int = 12,
    d_model: int = 768,
    d_ff: int = 3072,
) -> int:
    """Total MACs for standard XLM-R (no deletion)."""
    return transformer_layer_macs(d_model, d_ff, seq_len) * n_layers


# ---------------------------------------------------------------------------
# Exact savings formula (paper Appendix C, attention-only version)
# ---------------------------------------------------------------------------

def attention_compute_saved_pct(
    keep_ratio: float,
    gate_layer: int = 3,
    n_layers: int = 12,
) -> float:
    """
    Fraction of total attention MACs saved by deleting (1 - keep_ratio) tokens.
    Layers 0..gate_layer use full sequence; remaining layers use keep_ratio * seq.

    Returns value in [0, 1].
    """
    n_before = gate_layer + 1
    n_after  = n_layers - n_before
    baseline = n_layers
    with_deletion = n_before + n_after * (keep_ratio ** 2)
    saved = (baseline - with_deletion) / baseline
    return max(0.0, min(1.0, saved))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_macs_vs_deletion_ratio(seq_len: int, gate_layer: int, output_dir: str):
    """Figure 1: relative compute vs deletion ratio for one gate layer."""
    deletion_ratios = [i / 100 for i in range(0, 101)]
    keep_ratios     = [1.0 - d for d in deletion_ratios]

    baseline = xlmr_baseline_macs(seq_len)
    relative = [xlmr_macs(seq_len, kr, gate_layer=gate_layer) / baseline
                for kr in keep_ratios]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(deletion_ratios, relative, color="steelblue", linewidth=2)
    ax.axvline(x=0.30, color="gray", linestyle="--", linewidth=1, label="30% deletion")
    ax.axvline(x=0.50, color="gray", linestyle=":",  linewidth=1, label="50% deletion")

    for dr in (0.30, 0.50):
        kr  = 1.0 - dr
        rel = xlmr_macs(seq_len, kr, gate_layer=gate_layer) / baseline
        ax.annotate(
            f"{rel:.2f}×",
            xy=(dr, rel),
            xytext=(dr + 0.03, rel + 0.03),
            fontsize=8,
        )

    ax.set_xlabel("Deletion ratio (δ)")
    ax.set_ylabel("Compute vs baseline XLM-R")
    ax.set_title(f"MrXLMR compute savings\n(gate at layer {gate_layer}, seq_len={seq_len})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "macs_relative.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_macs_by_gate_layer(seq_len: int, output_dir: str):
    """Figure 2: relative compute vs deletion ratio for multiple gate layers."""
    deletion_ratios = [i / 100 for i in range(0, 101)]
    keep_ratios     = [1.0 - d for d in deletion_ratios]
    baseline        = xlmr_baseline_macs(seq_len)
    gate_layers     = [1, 3, 6, 9]
    colors          = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    fig, ax = plt.subplots(figsize=(5, 3))
    for gl, color in zip(gate_layers, colors):
        relative = [xlmr_macs(seq_len, kr, gate_layer=gl) / baseline
                    for kr in keep_ratios]
        ax.plot(deletion_ratios, relative, color=color, linewidth=2,
                label=f"Gate after layer {gl}")

    ax.set_xlabel("Deletion ratio (δ)")
    ax.set_ylabel("Compute vs baseline XLM-R")
    ax.set_title(f"MrXLMR compute savings by gate layer\n(seq_len={seq_len})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "macs_by_gate_layer.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_accuracy_vs_compute(runs: list, seq_len: int, gate_layer: int, output_dir: str):
    """
    Figure 3: accuracy vs relative compute for each model run.

    `runs` is a list of (label, accuracy, deletion_rate) tuples, e.g.:
        [("XLM-R", 0.9055, 0.0), ("MrXLMR-30%", 0.8141, 0.32), ("MrXLMR-50%", 0.78, 0.50)]

    The background curve shows the theoretical compute envelope (gate at `gate_layer`).
    Each run is plotted as a labelled point using its measured deletion_rate to determine
    its x-position (relative compute).
    """
    deletion_ratios = [i / 100 for i in range(0, 101)]
    keep_ratios     = [1.0 - d for d in deletion_ratios]
    baseline        = xlmr_baseline_macs(seq_len)
    bg_compute      = [xlmr_macs(seq_len, kr, gate_layer=gate_layer) / baseline
                       for kr in keep_ratios]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(bg_compute, [None] * len(bg_compute),
            color="lightgray", linewidth=1.5, linestyle="--", zorder=1)

    colors = ["#377eb8", "#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]

    for i, (label, accuracy, deletion_rate) in enumerate(runs):
        keep_ratio  = 1.0 - deletion_rate
        rel_compute = xlmr_macs(seq_len, keep_ratio, gate_layer=gate_layer) / baseline
        color       = colors[i % len(colors)]

        ax.scatter(rel_compute, accuracy * 100, color=color, s=80, zorder=3)
        ax.annotate(
            label,
            xy=(rel_compute, accuracy * 100),
            xytext=(rel_compute + 0.01, accuracy * 100 + 0.3),
            fontsize=9,
            color=color,
        )

    ax.set_xlabel("Relative compute vs XLM-R baseline")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs compute tradeoff\n(MrXLMR on SNLI)")
    ax.set_xlim(0, 1.1)
    if runs:
        accs = [r[1] * 100 for r in runs]
        ax.set_ylim(min(accs) - 3, max(accs) + 3)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "accuracy_vs_compute.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_accuracy_vs_seq_length_reduction(runs: list, output_dir: str):
    """
    Figure 4: accuracy vs sequence length reduction %.
    Mirrors MrT5 paper Figure 2 (BPB vs sequence length reduction).

    `runs` is a list of (label, accuracy, deletion_rate) tuples.
    X-axis: sequence length reduction % = deletion_rate * 100
    Y-axis: accuracy %

    The XLM-R baseline (deletion_rate == 0) is shown as a horizontal dashed line.
    """
    baseline_runs = [(l, a, d) for l, a, d in runs if d == 0.0]
    mrxlmr_runs   = [(l, a, d) for l, a, d in runs if d > 0.0]

    fig, ax = plt.subplots(figsize=(5, 4))

    mrxlmr_runs_sorted = sorted(mrxlmr_runs, key=lambda x: x[2])
    if mrxlmr_runs_sorted:
        xs = [d * 100 for _, _, d in mrxlmr_runs_sorted]
        ys = [a * 100 for _, a, _ in mrxlmr_runs_sorted]
        ax.plot(xs, ys, color="#e41a1c", linewidth=1.5, zorder=2)
        for label, acc, dr in mrxlmr_runs_sorted:
            ax.scatter(dr * 100, acc * 100, color="#e41a1c", s=70, zorder=3)
            ax.annotate(
                label,
                xy=(dr * 100, acc * 100),
                xytext=(dr * 100 + 1.0, acc * 100 + 0.3),
                fontsize=8,
                color="#e41a1c",
            )

    for label, acc, _ in baseline_runs:
        ax.axhline(y=acc * 100, color="#377eb8", linestyle="--", linewidth=1.5,
                   label=f"{label} (no deletion, {acc * 100:.1f}%)", zorder=1)

    ax.set_xlabel("Sequence length reduction (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs sequence length reduction\n(MrXLMR on SNLI)")
    ax.set_xlim(-2, max(d * 100 for _, _, d in runs) + 10 if runs else 80)
    if runs:
        accs = [a * 100 for _, a, _ in runs]
        ax.set_ylim(min(accs) - 3, max(accs) + 3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "accuracy_vs_seq_reduction.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_gate_layer_ablation(gate_layer_runs: list, output_dir: str):
    """
    Figure 5: accuracy + runtime vs gate layer — dual y-axis.
    Mirrors MrT5 paper Figure 4 (BPB + runtime vs gate layer).

    `gate_layer_runs` is a list of (label, accuracy, runtime_ms, gate_layer) tuples, e.g.:
        [("Layer 1", 0.88, 32.1, 1), ("Layer 3", 0.87, 28.5, 3), ...]

    Left y-axis:  accuracy (%)
    Right y-axis: inference runtime (ms/sample)
    X-axis:       gate layer index
    """
    gate_layer_runs_sorted = sorted(gate_layer_runs, key=lambda x: x[3])
    layers   = [gl for _, _, _, gl in gate_layer_runs_sorted]
    accs     = [a * 100 for _, a, _, _ in gate_layer_runs_sorted]
    runtimes = [rt for _, _, rt, _ in gate_layer_runs_sorted]

    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()

    color_acc     = "#e41a1c"
    color_runtime = "#377eb8"

    ax1.plot(layers, accs, color=color_acc, linewidth=2, marker="o", markersize=6,
             label="Accuracy", zorder=3)
    ax2.plot(layers, runtimes, color=color_runtime, linewidth=2, marker="s", markersize=6,
             linestyle="--", label="Runtime (ms/sample)", zorder=3)

    for gl, acc in zip(layers, accs):
        ax1.annotate(f"{acc:.1f}%", xy=(gl, acc),
                     xytext=(gl + 0.1, acc + 0.2), fontsize=8, color=color_acc)

    for gl, rt in zip(layers, runtimes):
        ax2.annotate(f"{rt:.1f}ms", xy=(gl, rt),
                     xytext=(gl + 0.1, rt - 0.8), fontsize=8, color=color_runtime)

    ax1.set_xlabel("Delete gate layer (0-indexed)")
    ax1.set_ylabel("Accuracy (%)", color=color_acc)
    ax2.set_ylabel("Inference runtime (ms/sample)", color=color_runtime)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_runtime)
    ax1.set_xticks(layers)
    ax1.set_title("Accuracy and runtime vs delete gate layer\n(MrXLMR, 30% deletion, SNLI)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "gate_layer_ablation.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_savings_table(seq_len: int = 27):
    """Print a quick table of compute savings at common deletion ratios."""
    baseline = xlmr_baseline_macs(seq_len)
    print(f"\nMACs savings for XLM-R-base (seq_len={seq_len}, gate at layer 3)")
    print(f"{'Deletion':>10}  {'Keep ratio':>12}  {'Relative MACs':>14}  {'Savings':>10}")
    print("-" * 52)
    for dr in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        kr  = 1.0 - dr
        rel = xlmr_macs(seq_len, kr, gate_layer=3) / baseline
        pct_saved = attention_compute_saved_pct(kr, gate_layer=3)
        print(f"{dr:>9.0%}  {kr:>12.2f}  {rel:>13.4f}×  {pct_saved:>9.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot MrXLMR theoretical compute savings.")
    p.add_argument("--seq_len", type=int, default=27,
                   help="Sequence length for MACs calculation. "
                        "Default 27 ≈ mean SNLI sequence length after 30%% deletion.")
    p.add_argument("--gate_layer", type=int, default=3,
                   help="Default gate layer for Figures 1 and 3.")
    p.add_argument("--output_dir", type=str, default="analysis/figures")
    p.add_argument(
        "--runs",
        nargs="+",
        metavar="LABEL,ACC,DEL_RATE",
        default=None,
        help=(
            "Runs to plot in accuracy-vs-compute and accuracy-vs-seq-reduction charts. "
            "Each entry is a comma-separated triple: label,accuracy,deletion_rate. "
            "Example: --runs 'XLM-R,0.9055,0.0' 'MrXLMR-30%%,0.8141,0.32' 'MrXLMR-50%%,0.78,0.50'"
        ),
    )
    p.add_argument(
        "--gate-layer-runs",
        nargs="+",
        metavar="LABEL,ACC,RUNTIME_MS,GATE_LAYER",
        default=None,
        dest="gate_layer_runs",
        help=(
            "Runs for the gate layer ablation chart (accuracy + runtime vs layer). "
            "Each entry is: label,accuracy,runtime_ms,gate_layer. "
            "Example: --gate-layer-runs 'Layer 1,0.88,32.1,1' 'Layer 3,0.87,28.5,3' 'Layer 6,0.85,24.1,6'"
        ),
    )
    args = p.parse_args()

    print_savings_table(seq_len=args.seq_len)
    plot_macs_vs_deletion_ratio(seq_len=args.seq_len, gate_layer=args.gate_layer,
                                output_dir=args.output_dir)
    plot_macs_by_gate_layer(seq_len=args.seq_len, output_dir=args.output_dir)

    if args.runs:
        runs = []
        for entry in args.runs:
            parts = entry.split(",")
            if len(parts) != 3:
                print(f"Warning: skipping malformed --runs entry {entry!r} (expected label,acc,del_rate)")
                continue
            label, acc, del_rate = parts[0], float(parts[1]), float(parts[2])
            runs.append((label, acc, del_rate))
        if runs:
            plot_accuracy_vs_compute(runs, seq_len=args.seq_len,
                                     gate_layer=args.gate_layer,
                                     output_dir=args.output_dir)
            plot_accuracy_vs_seq_length_reduction(runs, output_dir=args.output_dir)
    else:
        print("\nSkipping accuracy plots — no --runs provided.")
        print("Once you have eval results, re-run with:")
        print("  --runs 'XLM-R,0.9055,0.0' 'MrXLMR-30%,<acc>,0.32' 'MrXLMR-50%,<acc>,0.50'")

    if args.gate_layer_runs:
        gl_runs = []
        for entry in args.gate_layer_runs:
            parts = entry.split(",")
            if len(parts) != 4:
                print(f"Warning: skipping malformed --gate-layer-runs entry {entry!r} "
                      f"(expected label,acc,runtime_ms,gate_layer)")
                continue
            label, acc, runtime_ms, gate_layer = parts[0], float(parts[1]), float(parts[2]), int(parts[3])
            gl_runs.append((label, acc, runtime_ms, gate_layer))
        if gl_runs:
            plot_gate_layer_ablation(gl_runs, output_dir=args.output_dir)
    else:
        print("\nSkipping gate_layer_ablation.pdf — no --gate-layer-runs provided.")
        print("Once you have layer ablation results, re-run with:")
        print("  --gate-layer-runs 'Layer 1,<acc>,<ms>,1' 'Layer 3,<acc>,<ms>,3' ...")

    print("\nDone.")


if __name__ == "__main__":
    main()