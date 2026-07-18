"""
generate_table5_mrxlmr.py

Generates Table 5 (MrXLMR results on SNLI) for the COLM 2026 paper
from the W&B summary CSV.

Data source:
  - mrbert/analysis/wandb_plots/all_runs_summary.csv

Usage (from repo root):
  python mrbert/analysis/generate_table5_mrxlmr.py

Output:
  - mrbert/analysis/figures/table5_mrxlmr_results.csv
  - Prints LaTeX table to stdout
"""

import csv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WANDB_CSV = REPO_ROOT / "mrbert" / "analysis" / "wandb_plots" / "all_runs_summary.csv"
OUTPUT_DIR = REPO_ROOT / "mrbert" / "analysis" / "figures"

RUNS = [
    {
        "run_name": "xlmr-snli-baseline",
        "label": "XLM-R baseline",
        "is_baseline": True,
    },
    {
        "run_name": "mrxlmr-snli-0pct",
        "label": "MrXLMR 0\\% (sanity)",
        "is_baseline": False,
    },
    {
        "run_name": "mrxlmr-snli-30pct",
        "label": "MrXLMR 30\\%",
        "is_baseline": False,
    },
    {
        "run_name": "mrxlmr-snli-30pct-layer8-freeze-embeddings",
        "label": "MrXLMR 30\\% (L8, frozen embeddings)",
        "is_baseline": False,
    },
]


def load_wandb_data():
    """Load MrXLMR-SNLI runs from W&B summary CSV."""
    data = {}
    with open(WANDB_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("project") != "mrxlmr-snli":
                continue
            name = row.get("run_name", "")
            accuracy = row.get("test/accuracy")
            del_rate = row.get("train/percent_non_pad_deleted_tokens")
            seq_len = row.get("train/new_seq_len")

            data[name] = {
                "accuracy": float(accuracy) if accuracy else None,
                "del_rate": float(del_rate) if del_rate else 0.0,
                "avg_seq_len": float(seq_len) if seq_len else None,
            }
    return data


def build_table(wandb_data):
    """Build table rows from W&B data."""
    rows = []
    baseline_acc = None

    for run_cfg in RUNS:
        run_name = run_cfg["run_name"]
        run_data = wandb_data.get(run_name)

        if run_data is None:
            print(f"WARNING: run '{run_name}' not found in W&B data")
            continue

        if run_cfg["is_baseline"]:
            baseline_acc = run_data["accuracy"]

        rows.append({
            "label": run_cfg["label"],
            "accuracy": run_data["accuracy"],
            "del_rate": run_data["del_rate"],
            "avg_seq_len": run_data["avg_seq_len"],
            "is_baseline": run_cfg["is_baseline"],
        })

    # Compute deltas
    for row in rows:
        if row["is_baseline"] or baseline_acc is None:
            row["delta_pp"] = None
        else:
            row["delta_pp"] = (row["accuracy"] - baseline_acc) * 100

    return rows


def print_latex_table(rows):
    """Print Table 5 as LaTeX."""
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Accuracy} & \textbf{$\Delta$ (pp)} & "
          r"\textbf{Del Rate} & \textbf{Avg Seq Len} \\")
    print(r"\midrule")

    for row in rows:
        acc_str = f"{row['accuracy']*100:.2f}\\%" if row["accuracy"] else "---"

        if row["delta_pp"] is None:
            delta_str = "---"
        elif row["delta_pp"] >= 0:
            delta_str = f"$+${row['delta_pp']:.2f}"
        else:
            delta_str = f"$-${abs(row['delta_pp']):.2f}"

        del_str = f"{row['del_rate']:.1f}\\%" if not row["is_baseline"] else "0\\%"
        seq_str = f"{row['avg_seq_len']:.1f}" if row["avg_seq_len"] else "---"

        # Baseline has no gate so avg_seq_len comes from input stats
        if row["is_baseline"] and row["avg_seq_len"] is None:
            seq_str = "15.6"

        print(f"{row['label']:45s} & {acc_str} & {delta_str:10s} & "
              f"{del_str:6s} & {seq_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{MrXLMR results on SNLI. MrXLMR-0\% and MrXLMR-30\% both "
          r"degrade ($-$7.46pp and $-$7.58pp), but the L8 frozen-embeddings variant "
          r"recovers to baseline accuracy (89.86\%) while achieving 50.3\% deletion, "
          r"suggesting freezing embeddings resolves XLM-R's training instability.}")
    print(r"\label{tab:mrxlmr-results}")
    print(r"\end{table}")


def save_table_csv(rows, path):
    """Save table data as CSV for reproducibility."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "accuracy", "delta_pp", "del_rate_pct", "avg_seq_len"])
        for row in rows:
            writer.writerow([
                row["label"].replace("\\%", "%"),
                f"{row['accuracy']:.4f}" if row["accuracy"] else "",
                f"{row['delta_pp']:.2f}" if row["delta_pp"] is not None else "",
                f"{row['del_rate']:.2f}",
                f"{row['avg_seq_len']:.2f}" if row["avg_seq_len"] else "",
            ])
    print(f"\nSaved table CSV -> {path}")


def print_verification(rows):
    """Print verification summary."""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    baseline = next((r for r in rows if r["is_baseline"]), None)
    if baseline:
        print(f"\nXLM-R baseline: {baseline['accuracy']*100:.2f}%")

    for row in rows:
        if row["is_baseline"]:
            continue
        print(f"\n{row['label'].replace(chr(92)+'%', '%')}:")
        print(f"  Accuracy:    {row['accuracy']*100:.2f}%")
        print(f"  Delta:       {row['delta_pp']:+.2f}pp")
        print(f"  Del rate:    {row['del_rate']:.2f}%")
        print(f"  Avg seq len: {row['avg_seq_len']:.2f}" if row["avg_seq_len"] else "  Avg seq len: N/A")

    # Key finding
    l8_frozen = next((r for r in rows if "L8" in r["label"]), None)
    if l8_frozen and baseline:
        print(f"\n{'=' * 70}")
        print("KEY FINDING:")
        print(f"  L8 frozen-embeddings: {l8_frozen['accuracy']*100:.2f}% "
              f"(delta={l8_frozen['delta_pp']:+.2f}pp) at {l8_frozen['del_rate']:.1f}% deletion")
        print(f"  -> Matches baseline ({baseline['accuracy']*100:.2f}%) while deleting "
              f"half the tokens")


def main():
    wandb_data = load_wandb_data()

    print(f"Found {len(wandb_data)} runs in project mrxlmr-snli")
    print(f"Available runs: {sorted(wandb_data.keys())}")
    print()

    rows = build_table(wandb_data)

    print("=" * 70)
    print("TABLE 5: MrXLMR Results on SNLI (LaTeX)")
    print("=" * 70)
    print()
    print_latex_table(rows)

    # Save CSV
    csv_path = OUTPUT_DIR / "table5_mrxlmr_results.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_table_csv(rows, csv_path)

    # Print verification
    print_verification(rows)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
