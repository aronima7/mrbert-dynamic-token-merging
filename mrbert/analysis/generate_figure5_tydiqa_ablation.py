"""
generate_figure5_tydiqa_ablation.py

Regenerates Figure 5 (TyDi QA pre-deletion blending ablation bar chart)
with correct values from W&B.

Data source:
  - mrbert/analysis/wandb_plots/all_runs_summary.csv

Usage (from repo root):
  python mrbert/analysis/generate_figure5_tydiqa_ablation.py

Output:
  - mrbert/analysis/figures/tydiqa_blending_ablation.png
  - COLM-Paper-dynamic-token-merging/COLM-submission/figures/tydiqa_blending_ablation.png
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WANDB_CSV = REPO_ROOT / "mrbert" / "analysis" / "wandb_plots" / "all_runs_summary.csv"
OUTPUT_DIR = REPO_ROOT / "mrbert" / "analysis" / "figures"
COLM_FIGURES = REPO_ROOT / "COLM-Paper-dynamic-token-merging" / "COLM-submission" / "figures"

RUNS = [
    {"run_name": "bert-tydiqa-baseline", "label": "BERT baseline"},
    {"run_name": "mrbert-tydiqa-30pct", "label": "MrBERT 30%\n(no blend, L3)"},
    {"run_name": "mrbert-tydiqa-30pct-predel", "label": "MrBERT 30%\n(blend, L3)"},
    {"run_name": "mrbert-tydiqa-30pct-layer9-predel", "label": "MrBERT 30%\n(blend, L9)"},
]


def load_wandb_data():
    """Load TyDi QA runs from W&B summary CSV."""
    data = {}
    with open(WANDB_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("project") != "mrbert-tydiqa":
                continue
            name = row.get("run_name", "")
            span_em = row.get("eval/span_em")
            start_acc = row.get("eval/start_acc")
            end_acc = row.get("eval/end_acc")
            data[name] = {
                "span_em": float(span_em) if span_em else None,
                "start_acc": float(start_acc) if start_acc else None,
                "end_acc": float(end_acc) if end_acc else None,
            }
    return data


def plot_figure(wandb_data, output_path):
    """Generate the ablation bar chart."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    labels = []
    end_accs = []
    start_accs = []
    span_ems = []

    for run in RUNS:
        d = wandb_data.get(run["run_name"])
        if d is None:
            print(f"WARNING: {run['run_name']} not found")
            continue
        labels.append(run["label"])
        end_accs.append(d["end_acc"])
        start_accs.append(d["start_acc"])
        span_ems.append(d["span_em"])

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_end = ax.bar(x - width, end_accs, width, label="End Acc", color="#64b5f6", zorder=3)
    bars_start = ax.bar(x, start_accs, width, label="Start Acc", color="#81c784", zorder=3)
    bars_span = ax.bar(x + width, span_ems, width, label="Span EM", color="#ffb74d", zorder=3)

    # Annotate span EM values
    for bar, val in zip(bars_span, span_ems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, color="#e65100")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("TyDi QA: Pre-Deletion Blending Ablation")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, 0.75)

    os.makedirs(output_path.parent, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main():
    wandb_data = load_wandb_data()

    print("TyDi QA ablation data from W&B:")
    for run in RUNS:
        d = wandb_data.get(run["run_name"])
        if d:
            print(f"  {run['run_name']:45s} span_em={d['span_em']:.4f}  "
                  f"start={d['start_acc']:.4f}  end={d['end_acc']:.4f}")

    # Generate figure
    output_path = OUTPUT_DIR / "tydiqa_blending_ablation.png"
    plot_figure(wandb_data, output_path)

    # Also copy to COLM submission figures
    if COLM_FIGURES.exists():
        colm_path = COLM_FIGURES / "tydiqa_blending_ablation.png"
        plot_figure(wandb_data, colm_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
