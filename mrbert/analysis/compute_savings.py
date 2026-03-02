"""
compute_savings.py

Plots theoretical MACs (multiply-accumulate operations) saved by token deletion
in MrBERT as a function of deletion ratio, mirroring mrt5/analysis/compute_savings.ipynb
adapted for the BERT encoder-only architecture.

Key differences from MrT5:
  - Encoder-only (no decoder, no cross-attention)
  - BERT-base dimensions: d_model=768, d_ff=3072, 12 encoder layers
  - Gate at layer 3 by default (0-indexed: deletion happens after layer 3)

Produces three figures saved to analysis/figures/:
  1. macs_relative.pdf         — relative compute vs deletion ratio for gate at layer 3
  2. macs_by_gate_layer.pdf    — relative compute vs deletion ratio for gate at layers 1, 3, 6, 9
  3. accuracy_vs_compute.pdf   — accuracy vs relative compute for each run (requires --runs)

Usage:
    # Theoretical curves only (no run data needed)
    python analysis/compute_savings.py

    # Include accuracy-vs-compute plot once you have eval results
    python analysis/compute_savings.py \\
        --runs "BERT,0.9055,0.0" "MrBERT-30%,0.8141,0.32" "MrBERT-50%,0.78,0.50"

    # Adjust seq_len to your actual mean post-deletion sequence length
    python analysis/compute_savings.py --seq_len 27 \\
        --runs "BERT,0.9055,0.0" "MrBERT-30%,0.8141,0.32"
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
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


def bert_macs(
    seq_len: int,
    keep_ratio: float,
    gate_layer: int = 3,
    n_layers: int = 12,
    d_model: int = 768,
    d_ff: int = 3072,
) -> int:
    """
    Total MACs for MrBERT with a delete gate after layer `gate_layer`.

    Layers 0..gate_layer run at full seq_len.
    Layers (gate_layer+1)..n_layers-1 run at seq_len * keep_ratio.
    """
    reduced_seq = int(seq_len * keep_ratio)
    n_before = gate_layer + 1
    n_after  = n_layers - n_before

    before = transformer_layer_macs(d_model, d_ff, seq_len)     * n_before
    after  = transformer_layer_macs(d_model, d_ff, reduced_seq) * n_after
    return before + after


def bert_baseline_macs(
    seq_len: int,
    n_layers: int = 12,
    d_model: int = 768,
    d_ff: int = 3072,
) -> int:
    """Total MACs for standard BERT (no deletion)."""
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

    baseline = bert_baseline_macs(seq_len)
    relative = [bert_macs(seq_len, kr, gate_layer=gate_layer) / baseline
                for kr in keep_ratios]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(deletion_ratios, relative, color="steelblue", linewidth=2)
    ax.axvline(x=0.30, color="gray", linestyle="--", linewidth=1, label="30% deletion")
    ax.axvline(x=0.50, color="gray", linestyle=":",  linewidth=1, label="50% deletion")

    # Annotate target operating points
    for dr in (0.30, 0.50):
        kr  = 1.0 - dr
        rel = bert_macs(seq_len, kr, gate_layer=gate_layer) / baseline
        ax.annotate(
            f"{rel:.2f}×",
            xy=(dr, rel),
            xytext=(dr + 0.03, rel + 0.03),
            fontsize=8,
        )

    ax.set_xlabel("Deletion ratio (δ)")
    ax.set_ylabel("Compute vs baseline BERT")
    ax.set_title(f"MrBERT compute savings\n(gate at layer {gate_layer}, seq_len={seq_len})")
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
    baseline        = bert_baseline_macs(seq_len)
    gate_layers     = [1, 3, 6, 9]
    colors          = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    fig, ax = plt.subplots(figsize=(5, 3))
    for gl, color in zip(gate_layers, colors):
        relative = [bert_macs(seq_len, kr, gate_layer=gl) / baseline
                    for kr in keep_ratios]
        ax.plot(deletion_ratios, relative, color=color, linewidth=2,
                label=f"Gate after layer {gl}")

    ax.set_xlabel("Deletion ratio (δ)")
    ax.set_ylabel("Compute vs baseline BERT")
    ax.set_title(f"MrBERT compute savings by gate layer\n(seq_len={seq_len})")
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
        [("BERT", 0.9055, 0.0), ("MrBERT-30%", 0.8141, 0.32), ("MrBERT-50%", 0.78, 0.50)]

    The background curve shows the theoretical compute envelope (gate at `gate_layer`).
    Each run is plotted as a labelled point using its measured deletion_rate to determine
    its x-position (relative compute).
    """
    # Background efficiency curve
    deletion_ratios = [i / 100 for i in range(0, 101)]
    keep_ratios     = [1.0 - d for d in deletion_ratios]
    baseline        = bert_baseline_macs(seq_len)
    bg_compute      = [bert_macs(seq_len, kr, gate_layer=gate_layer) / baseline
                       for kr in keep_ratios]

    fig, ax = plt.subplots(figsize=(5, 4))

    # Background: theoretical compute envelope
    ax.plot(bg_compute, [None] * len(bg_compute),  # invisible — just for context
            color="lightgray", linewidth=1.5, linestyle="--", zorder=1)

    # Colour cycle for the run points
    colors = ["#377eb8", "#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]

    for i, (label, accuracy, deletion_rate) in enumerate(runs):
        keep_ratio   = 1.0 - deletion_rate
        rel_compute  = bert_macs(seq_len, keep_ratio, gate_layer=gate_layer) / baseline
        color        = colors[i % len(colors)]

        ax.scatter(rel_compute, accuracy * 100,
                   color=color, s=80, zorder=3)
        ax.annotate(
            label,
            xy=(rel_compute, accuracy * 100),
            xytext=(rel_compute + 0.01, accuracy * 100 + 0.3),
            fontsize=9,
            color=color,
        )

    # Axis formatting
    ax.set_xlabel("Relative compute vs BERT baseline")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs compute tradeoff\n(MrBERT on SNLI)")
    ax.set_xlim(0, 1.1)
    # y-axis: auto-range based on run accuracies, with a little padding
    if runs:
        accs = [r[1] * 100 for r in runs]
        ax.set_ylim(min(accs) - 3, max(accs) + 3)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "accuracy_vs_compute.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_savings_table(seq_len: int = 27):
    """Print a quick table of compute savings at common deletion ratios."""
    baseline = bert_baseline_macs(seq_len)
    print(f"\nMACs savings for BERT-base (seq_len={seq_len}, gate at layer 3)")
    print(f"{'Deletion':>10}  {'Keep ratio':>12}  {'Relative MACs':>14}  {'Savings':>10}")
    print("-" * 52)
    for dr in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        kr  = 1.0 - dr
        rel = bert_macs(seq_len, kr, gate_layer=3) / baseline
        pct_saved = attention_compute_saved_pct(kr, gate_layer=3)
        print(f"{dr:>9.0%}  {kr:>12.2f}  {rel:>13.4f}×  {pct_saved:>9.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot MrBERT theoretical compute savings.")
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
            "Runs to plot in Figure 3 (accuracy vs compute). "
            "Each entry is a comma-separated triple: label,accuracy,deletion_rate. "
            "Example: --runs 'BERT,0.9055,0.0' 'MrBERT-30%%,0.8141,0.32' 'MrBERT-50%%,0.78,0.50'"
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
                print(f"Warning: skipping malformed run entry {entry!r} (expected label,acc,del_rate)")
                continue
            label, acc, del_rate = parts[0], float(parts[1]), float(parts[2])
            runs.append((label, acc, del_rate))
        if runs:
            plot_accuracy_vs_compute(runs, seq_len=args.seq_len,
                                     gate_layer=args.gate_layer,
                                     output_dir=args.output_dir)
    else:
        print("\nSkipping accuracy_vs_compute.pdf — no --runs provided.")
        print("Once you have eval results, re-run with:")
        print("  --runs 'BERT,0.9055,0.0' 'MrBERT-30%,<acc>,0.32' 'MrBERT-50%,<acc>,0.50'")

    print("\nDone.")


if __name__ == "__main__":
    main()