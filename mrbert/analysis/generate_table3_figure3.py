"""
generate_table3_figure3.py

Generates Table 3 (SNLI test results) and Figure 3 (snli_efficiency_frontier.png)
for the COLM 2026 paper from W&B summary data and runtime CSVs.

Data sources:
  - mrbert/analysis/wandb_plots/all_runs_summary.csv (accuracy, del rate, avg seq len)
  - mrbert/analysis/figures/snli_runtime_table_deletion_percentage.csv (ms/sample by del %)
  - mrbert/analysis/figures/snli_runtime_table_deletion_gate_layer.csv (ms/sample by gate layer)

Usage (from repo root):
  python mrbert/analysis/generate_table3_figure3.py

Output:
  - mrbert/analysis/figures/snli_efficiency_frontier.png  (Figure 3)
  - mrbert/analysis/figures/table3_snli_results.csv       (Table 3 data)
  - Prints LaTeX table to stdout
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
RUNTIME_DEL_CSV = REPO_ROOT / "mrbert" / "analysis" / "figures" / "snli_runtime_table_deletion_percentage.csv"
RUNTIME_LAYER_CSV = REPO_ROOT / "mrbert" / "analysis" / "figures" / "snli_runtime_table_deletion_gate_layer.csv"
OUTPUT_DIR = REPO_ROOT / "mrbert" / "analysis" / "figures"

# Run name -> display label mapping
SNLI_RUNS = {
    "bert-snli-baseline":    "BERT baseline",
    "mrbert-snli-0pct":      "MrBERT 0% (sanity)",
    "mrbert-snli-30pct":     "MrBERT 30%",
    "mrbert-snli-30pct-hd":  "MrBERT 30% (hard-train)",
    "mrbert-snli-50pct":     "MrBERT 50%",
    "mrbert-snli-70pct":     "MrBERT 70%",
    "mrbert-snli-random30":  "Random gate 30%",
    "mrbert-snli-nopi":      "No-PI 30%",
}

LAYER_RUNS = {
    "mrbert-snli-layer1": "Layer 1",
    "mrbert-snli-30pct":  "Layer 3 (default)",
    "mrbert-snli-layer6": "Layer 6",
    "mrbert-snli-layer9": "Layer 9",
}

# Runtime label -> W&B run name mapping
RUNTIME_LABEL_MAP = {
    "BERT": "bert-snli-baseline",
    "MrBERT-0%": "mrbert-snli-0pct",
    "MrBERT-30%": "mrbert-snli-30pct",
    "MrBERT-30%-HD": "mrbert-snli-30pct-hd",
    "MrBERT-50%": "mrbert-snli-50pct",
    "MrBERT-70%": "mrbert-snli-70pct",
    "Random-30%": "mrbert-snli-random30",
}

LAYER_LABEL_MAP = {
    "Layer-1": "mrbert-snli-layer1",
    "Layer-3": "mrbert-snli-30pct",
    "Layer-6": "mrbert-snli-layer6",
    "Layer-9": "mrbert-snli-layer9",
}


def load_wandb_data():
    """Load accuracy, deletion rate, and avg seq len from W&B summary CSV."""
    data = {}
    with open(WANDB_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("run_name", "")
            if not name:
                continue
            project = row.get("project", "")
            if "snli" not in project and "mrbert" not in project:
                continue
            accuracy = row.get("train/accuracy") or row.get("test/accuracy")
            del_rate = row.get("train/percent_non_pad_deleted_tokens")
            seq_len = row.get("train/new_seq_len")
            if accuracy:
                data[name] = {
                    "accuracy": float(accuracy),
                    "del_rate": float(del_rate) if del_rate else 0.0,
                    "avg_seq_len": float(seq_len) if seq_len else None,
                }
    return data


def load_runtime_csv(path):
    """Load runtime CSV into a dict keyed by label."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["label"]] = {
                "ms_per_sample": float(row["ms_per_sample"]),
                "std_ms": float(row["std_ms"]),
            }
    return data


def build_table(wandb_data, runtime_del, runtime_layer):
    """Combine W&B and runtime data into table rows."""
    baseline_acc = wandb_data["bert-snli-baseline"]["accuracy"]
    baseline_seq_len = 27.5  # average non-padding tokens in SNLI (max_seq_length=128)

    rows = []

    # Main ablation rows
    for run_name, label in SNLI_RUNS.items():
        wd = wandb_data.get(run_name, {})
        acc = wd.get("accuracy")
        del_rate = wd.get("del_rate", 0.0)
        seq_len = wd.get("avg_seq_len")
        if seq_len is None:
            seq_len = baseline_seq_len

        # Find runtime
        rt_label = next((k for k, v in RUNTIME_LABEL_MAP.items() if v == run_name), None)
        ms = runtime_del.get(rt_label, {}).get("ms_per_sample") if rt_label else None

        delta = (acc - baseline_acc) * 100 if acc else None  # in pp

        rows.append({
            "label": label,
            "accuracy": acc,
            "delta_pp": delta,
            "del_rate": del_rate,
            "avg_seq_len": seq_len,
            "ms_per_sample": ms,
            "section": "main",
        })

    # Gate layer ablation rows
    for run_name, label in LAYER_RUNS.items():
        wd = wandb_data.get(run_name, {})
        acc = wd.get("accuracy")
        del_rate = wd.get("del_rate", 0.0)
        seq_len = wd.get("avg_seq_len")

        rt_label = next((k for k, v in LAYER_LABEL_MAP.items() if v == run_name), None)
        ms = runtime_layer.get(rt_label, {}).get("ms_per_sample") if rt_label else None

        delta = (acc - baseline_acc) * 100 if acc else None

        rows.append({
            "label": label,
            "accuracy": acc,
            "delta_pp": delta,
            "del_rate": del_rate,
            "avg_seq_len": seq_len,
            "ms_per_sample": ms,
            "section": "layer",
        })

    return rows


def print_latex_table(rows):
    """Print Table 3 as LaTeX."""
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{@{}lccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Accuracy} & \textbf{$\Delta$ (pp)} & "
          r"\textbf{Del Rate} & \textbf{Avg Seq Len} & \textbf{ms/sample} \\")
    print(r"\midrule")

    for row in rows:
        if row["section"] == "layer" and row["label"] == "Layer 1":
            print(r"\midrule")
            print(r"\multicolumn{6}{@{}l}{\textit{Gate layer ablation (all at 30\% target)}} \\")

        acc_str = f"{row['accuracy']*100:.2f}\\%" if row["accuracy"] else "---"

        if row["delta_pp"] is None or row["label"] == "BERT baseline":
            delta_str = "---"
        elif row["delta_pp"] >= 0:
            delta_str = f"$+${row['delta_pp']:.2f}"
        else:
            delta_str = f"$-${abs(row['delta_pp']):.2f}"

        del_str = f"{row['del_rate']:.1f}\\%" if row["del_rate"] > 0 else "0\\%"
        seq_str = f"{row['avg_seq_len']:.1f}" if row["avg_seq_len"] else "---"

        if row["ms_per_sample"]:
            ms_str = f"{row['ms_per_sample']:.3f}"
        else:
            ms_str = "---"

        # Add markers
        if row["label"] == "Random gate 30%":
            seq_str += "*"
        if row["label"] == "MrBERT 0% (sanity)":
            ms_str += "$^\\dagger$"

        print(f"{row['label']:25s} & {acc_str} & {delta_str} & {del_str} & {seq_str} & {ms_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{SNLI test results. All inputs are padded to 128 tokens. "
          r"$\Delta$ is relative to the BERT baseline. Runtime measured on NVIDIA A100 "
          r"with hard deletion. Layer~3 provides the best accuracy--efficiency balance. "
          r"The No-PI run collapses to 89\% deletion (target: 30\%). "
          r"*Random gate does not always delete padding, so its Avg Seq Len includes "
          r"surviving pad tokens (unlike learned-gate rows, which always hard-delete padding). "
          r"$^\dagger$MrBERT-0\% is faster than BERT despite 0\% non-padding deletion "
          r"because hard deletion still removes all padding tokens, so post-gate layers "
          r"process only $\sim$27 tokens instead of 128.}")
    print(r"\label{tab:snli-results}")
    print(r"\end{table}")


def save_table_csv(rows, path):
    """Save table data as CSV for reproducibility."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "accuracy", "delta_pp", "del_rate_pct",
                         "avg_seq_len", "ms_per_sample", "section"])
        for row in rows:
            writer.writerow([
                row["label"],
                f"{row['accuracy']:.4f}" if row["accuracy"] else "",
                f"{row['delta_pp']:.2f}" if row["delta_pp"] is not None else "",
                f"{row['del_rate']:.2f}",
                f"{row['avg_seq_len']:.1f}" if row["avg_seq_len"] else "",
                f"{row['ms_per_sample']:.3f}" if row["ms_per_sample"] else "",
                row["section"],
            ])
    print(f"\nSaved table CSV → {path}")


def plot_efficiency_frontier(rows, output_path):
    """Generate the two-panel efficiency frontier scatter plot (Figure 3)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    # Filter to main ablation rows (not layer ablation) that have accuracy
    main_rows = [r for r in rows if r["section"] == "main" and r["accuracy"]]

    colors = {
        "BERT baseline":           "#1f77b4",
        "MrBERT 0% (sanity)":     "#2ca02c",
        "MrBERT 30%":             "#4daf4a",
        "MrBERT 30% (hard-train)":"#8cce8c",
        "MrBERT 50%":             "#ff7f0e",
        "MrBERT 70%":             "#d62728",
        "Random gate 30%":        "#9467bd",
        "No-PI 30%":              "#8c564b",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: Accuracy vs Deletion Rate
    for row in main_rows:
        color = colors.get(row["label"], "#333333")
        ax1.scatter(row["del_rate"], row["accuracy"] * 100,
                    color=color, s=200, zorder=3, edgecolors="white", linewidth=0.5)
        ax1.annotate(row["label"].replace("MrBERT ", "MrBERT\n") if len(row["label"]) > 15 else row["label"],
                     xy=(row["del_rate"], row["accuracy"] * 100),
                     xytext=(row["del_rate"] + 1.5, row["accuracy"] * 100 + 0.1),
                     fontsize=9, color=color)

    ax1.set_xlabel("Deletion Rate (%)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Accuracy vs. Deletion Rate (SNLI)")
    ax1.set_xlim(-5, 100)
    ax1.grid(True, alpha=0.3)

    # Right panel: Accuracy vs Inference Time
    main_with_runtime = [r for r in main_rows if r["ms_per_sample"]]
    for row in main_with_runtime:
        color = colors.get(row["label"], "#333333")
        ax2.scatter(row["ms_per_sample"], row["accuracy"] * 100,
                    color=color, s=200, zorder=3, edgecolors="white", linewidth=0.5)
        ax2.annotate(row["label"].replace("MrBERT ", "MrBERT\n") if len(row["label"]) > 15 else row["label"],
                     xy=(row["ms_per_sample"], row["accuracy"] * 100),
                     xytext=(row["ms_per_sample"] + 0.02, row["accuracy"] * 100 + 0.1),
                     fontsize=9, color=color)

    ax2.set_xlabel("Inference Time (ms/sample, A100)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Accuracy vs. Inference Speed (SNLI)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_path.parent, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {output_path}")


def main():
    wandb_data = load_wandb_data()
    runtime_del = load_runtime_csv(RUNTIME_DEL_CSV)
    runtime_layer = load_runtime_csv(RUNTIME_LAYER_CSV)

    rows = build_table(wandb_data, runtime_del, runtime_layer)

    print("=" * 70)
    print("TABLE 3: SNLI Test Results (LaTeX)")
    print("=" * 70)
    print()
    print_latex_table(rows)
    print()

    # Save CSV
    csv_path = OUTPUT_DIR / "table3_snli_results.csv"
    save_table_csv(rows, csv_path)

    # Generate Figure 3
    fig_path = OUTPUT_DIR / "snli_efficiency_frontier.png"
    plot_efficiency_frontier(rows, fig_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
