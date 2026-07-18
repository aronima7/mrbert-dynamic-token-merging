"""
generate_figure4_gate_layer_runtime.py

Regenerates Figure 4 (runtime vs gate layer placement) with the correct
BERT baseline (1.440ms from the deletion percentage benchmarking run).

Data source:
  - mrbert/analysis/figures/snli_runtime_table_deletion_gate_layer.csv
  - mrbert/analysis/figures/snli_runtime_table_deletion_percentage.csv (BERT baseline)

Usage (from repo root):
  python mrbert/analysis/generate_figure4_gate_layer_runtime.py

Output:
  - mrbert/analysis/figures/snli_runtime_vs_deletion_gate_layer.pdf
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GATE_LAYER_CSV = REPO_ROOT / "mrbert" / "analysis" / "figures" / "snli_runtime_table_deletion_gate_layer.csv"
DELETION_PCT_CSV = REPO_ROOT / "mrbert" / "analysis" / "figures" / "snli_runtime_table_deletion_percentage.csv"
OUTPUT_DIR = REPO_ROOT / "mrbert" / "analysis" / "figures"

# Also copy to the COLM submission figures directory
COLM_FIGURES = REPO_ROOT / "COLM-Paper-dynamic-token-merging" / "COLM-submission" / "figures"


def load_gate_layer_data():
    """Load gate layer runtime data."""
    rows = []
    with open(GATE_LAYER_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "label": row["label"],
                "ms": float(row["ms_per_sample"]),
                "std": float(row["std_ms"]),
            })
    return rows


def load_bert_baseline():
    """Load BERT baseline runtime from the deletion percentage CSV."""
    with open(DELETION_PCT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "BERT":
                return float(row["ms_per_sample"])
    raise ValueError("BERT baseline not found in deletion percentage CSV")


def plot_figure(gate_data, bert_baseline_ms, output_path):
    """Generate the corrected bar chart."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    labels = [r["label"] for r in gate_data]
    means = [r["ms"] for r in gate_data]
    stds = [r["std"] for r in gate_data]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=stds, color="#e41a1c", capsize=5, width=0.6, zorder=3)

    # BERT baseline dashed line (correct value: 1.440ms)
    ax.axhline(y=bert_baseline_ms, color="#377eb8", linestyle="--", linewidth=1.5,
               label=f"BERT baseline ({bert_baseline_ms:.3f} ms/sample)", zorder=2)

    # Annotate speedup on each bar
    for i, (bar, ms) in enumerate(zip(bars, means)):
        speedup = bert_baseline_ms / ms
        pct_decrease = (1.0 - ms / bert_baseline_ms) * 100.0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + stds[i] + 0.02,
                f"{speedup:.2f}×\n(−{pct_decrease:.0f}%)",
                ha="center", va="bottom", fontsize=9, color="#e41a1c")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Inference time (ms / sample)")
    ax.set_title("Inference runtime vs. gate layer placement\n(MrBERT-30%, SNLI, A100, hard deletion)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(bert_baseline_ms, max(means)) * 1.2)

    os.makedirs(output_path.parent, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def main():
    gate_data = load_gate_layer_data()
    bert_baseline_ms = load_bert_baseline()

    print(f"BERT baseline: {bert_baseline_ms:.3f} ms/sample")
    print(f"Gate layer data:")
    for row in gate_data:
        speedup = bert_baseline_ms / row["ms"]
        print(f"  {row['label']}: {row['ms']:.3f} ± {row['std']:.3f} ms  "
              f"({speedup:.2f}× speedup vs BERT)")

    # Generate figure
    output_path = OUTPUT_DIR / "snli_runtime_vs_deletion_gate_layer.pdf"
    plot_figure(gate_data, bert_baseline_ms, output_path)

    # Also copy to COLM submission figures
    if COLM_FIGURES.exists():
        colm_path = COLM_FIGURES / "snli_runtime_vs_deletion_gate_layer.pdf"
        plot_figure(gate_data, bert_baseline_ms, colm_path)


if __name__ == "__main__":
    main()
