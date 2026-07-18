"""
generate_table4_classification.py

Generates Table 4 (Classification results on SST-2, MRPC, and IMDB)
for the COLM 2026 paper from the W&B summary CSV.

Data source:
  - mrbert/analysis/wandb_plots/all_runs_summary.csv

Usage (from repo root):
  python mrbert/analysis/generate_table4_classification.py

Output:
  - mrbert/analysis/figures/table4_classification_results.csv
  - Prints LaTeX table to stdout
"""

import csv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WANDB_CSV = REPO_ROOT / "mrbert" / "analysis" / "wandb_plots" / "all_runs_summary.csv"
OUTPUT_DIR = REPO_ROOT / "mrbert" / "analysis" / "figures"

# Dataset configs: (baseline_run, mrbert_run, max_seq_length)
DATASETS = {
    "SST-2": {
        "project": "mrbert-sst2",
        "baseline_run": "bert-sst2-baseline",
        "mrbert_run": "mrbert-sst2-30pct",
        "max_seq_length": 128,
    },
    "MRPC": {
        "project": "mrbert-mrpc",
        "baseline_run": "bert-mrpc-baseline",
        "mrbert_run": "mrbert-mrpc-30pct",
        "max_seq_length": 128,
    },
    "IMDB": {
        "project": "mrbert-imdb",
        "baseline_run": "bert-imdb-baseline",
        "mrbert_run": "mrbert-imdb-30pct",
        "max_seq_length": 512,
    },
}


def load_wandb_data():
    """Load relevant runs from W&B summary CSV."""
    data = {}
    with open(WANDB_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("run_name", "")
            project = row.get("project", "")
            if not name or not project:
                continue

            # Use test/accuracy if available, otherwise train/accuracy
            accuracy = row.get("test/accuracy") or row.get("train/accuracy")
            del_rate = row.get("train/percent_non_pad_deleted_tokens")
            seq_len = row.get("train/new_seq_len")

            if accuracy:
                key = (project, name)
                data[key] = {
                    "accuracy": float(accuracy),
                    "del_rate": float(del_rate) if del_rate else 0.0,
                    "avg_seq_len": float(seq_len) if seq_len else None,
                }
    return data


def build_table(wandb_data):
    """Combine W&B data into table rows."""
    rows = []

    for dataset_name, config in DATASETS.items():
        project = config["project"]
        max_seq = config["max_seq_length"]

        baseline_key = (project, config["baseline_run"])
        mrbert_key = (project, config["mrbert_run"])

        baseline = wandb_data.get(baseline_key, {})
        mrbert = wandb_data.get(mrbert_key, {})

        baseline_acc = baseline.get("accuracy")
        mrbert_acc = mrbert.get("accuracy")
        del_rate = mrbert.get("del_rate", 0.0)
        avg_seq_len = mrbert.get("avg_seq_len")

        delta = (mrbert_acc - baseline_acc) * 100 if (baseline_acc and mrbert_acc) else None
        seq_red = (1.0 - avg_seq_len / max_seq) * 100 if avg_seq_len else None

        rows.append({
            "dataset": dataset_name,
            "baseline_acc": baseline_acc,
            "mrbert_acc": mrbert_acc,
            "delta_pp": delta,
            "del_rate": del_rate,
            "avg_seq_len": avg_seq_len,
            "seq_red": seq_red,
            "max_seq_length": max_seq,
        })

    return rows


def print_latex_table(rows):
    """Print Table 4 as LaTeX."""
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{@{}lcccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Baseline} & \textbf{MrBERT-30\%} & "
          r"\textbf{$\Delta$ (pp)} & \textbf{Del Rate} & "
          r"\textbf{Avg Seq Len} & \textbf{Seq Red.} \\")
    print(r"\midrule")

    for row in rows:
        base_str = f"{row['baseline_acc']*100:.2f}\\%" if row["baseline_acc"] else "---"
        mrb_str = f"{row['mrbert_acc']*100:.2f}\\%" if row["mrbert_acc"] else "---"

        if row["delta_pp"] is not None:
            if row["delta_pp"] >= 0:
                delta_str = f"$+${row['delta_pp']:.2f}"
            else:
                delta_str = f"$-${abs(row['delta_pp']):.2f}"
        else:
            delta_str = "---"

        del_str = f"{row['del_rate']:.1f}\\%"
        seq_str = f"{row['avg_seq_len']:.2f}" if row["avg_seq_len"] else "---"
        red_str = f"{row['seq_red']:.1f}\\%" if row["seq_red"] else "---"

        # Add * for SST-2 and MRPC (padding-dominated Seq Red)
        if row["dataset"] in ("SST-2", "MRPC"):
            red_str += "*"

        print(f"{row['dataset']:5s} & {base_str} & {mrb_str} & {delta_str} & "
              f"{del_str} & {seq_str} & {red_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Classification results on SST-2, MRPC, and IMDB. "
          r"$\Delta$ is relative to the BERT baseline; "
          r"Del Rate is the fraction of non-padding tokens deleted; "
          r"Seq Red.\ is the reduction in average sequence length relative to the "
          r"padded maximum ($1 - \text{Avg Seq Len}/\text{max\_seq\_length}$). "
          r"SST-2 and IMDB are robust to deletion ($<$1pp drop); "
          r"MRPC collapses, likely due to small dataset size and paraphrase-sensitivity. "
          r"*Seq Red.\ includes padding removal (free at inference for all MrBERT variants); "
          r"for SST-2 and MRPC, padding dominates the reduction --- "
          r"Del Rate better reflects the gate's learned contribution.}")
    print(r"\label{tab:classification-results}")
    print(r"\end{table}")


def save_table_csv(rows, path):
    """Save table data as CSV for reproducibility."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "baseline_acc", "mrbert_acc", "delta_pp",
                         "del_rate_pct", "avg_seq_len", "seq_red_pct", "max_seq_length"])
        for row in rows:
            writer.writerow([
                row["dataset"],
                f"{row['baseline_acc']:.4f}" if row["baseline_acc"] else "",
                f"{row['mrbert_acc']:.4f}" if row["mrbert_acc"] else "",
                f"{row['delta_pp']:.2f}" if row["delta_pp"] is not None else "",
                f"{row['del_rate']:.2f}",
                f"{row['avg_seq_len']:.2f}" if row["avg_seq_len"] else "",
                f"{row['seq_red']:.1f}" if row["seq_red"] else "",
                row["max_seq_length"],
            ])
    print(f"\nSaved table CSV → {path}")


def print_summary(rows):
    """Print a human-readable summary for verification."""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    for row in rows:
        print(f"\n{row['dataset']} (max_seq_length={row['max_seq_length']}):")
        print(f"  Baseline accuracy:  {row['baseline_acc']*100:.2f}%")
        print(f"  MrBERT-30% accuracy: {row['mrbert_acc']*100:.2f}%")
        print(f"  Delta:              {row['delta_pp']:.2f}pp")
        print(f"  Deletion rate:      {row['del_rate']:.2f}%")
        print(f"  Avg seq len:        {row['avg_seq_len']:.2f}")
        print(f"  Seq reduction:      {row['seq_red']:.1f}% "
              f"(= 1 - {row['avg_seq_len']:.2f}/{row['max_seq_length']})")
        # Estimate non-pad input length
        if row["del_rate"] > 0:
            est_non_pad = row["avg_seq_len"] / (1 - row["del_rate"] / 100)
            pad_removal_only = (1 - est_non_pad / row["max_seq_length"]) * 100
            print(f"  Est. non-pad input: ~{est_non_pad:.1f} tokens")
            print(f"  Padding removal alone: ~{pad_removal_only:.1f}%")
            print(f"  Gate contribution:  ~{row['seq_red'] - pad_removal_only:.1f}pp")


def main():
    wandb_data = load_wandb_data()
    rows = build_table(wandb_data)

    print("=" * 70)
    print("TABLE 4: Classification Results (LaTeX)")
    print("=" * 70)
    print()
    print_latex_table(rows)

    # Save CSV
    csv_path = OUTPUT_DIR / "table4_classification_results.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_table_csv(rows, csv_path)

    # Print verification summary
    print_summary(rows)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
