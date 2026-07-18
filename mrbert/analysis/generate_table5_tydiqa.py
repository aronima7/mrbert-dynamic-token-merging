"""
generate_table5_tydiqa.py

Generates Table 5 (TyDi QA English results) for the COLM 2026 paper
from the W&B summary CSV.

Data source:
  - mrbert/analysis/wandb_plots/all_runs_summary.csv

Usage (from repo root):
  python mrbert/analysis/generate_table5_tydiqa.py

Output:
  - mrbert/analysis/figures/table5_tydiqa_results.csv
  - Prints LaTeX table to stdout
"""

import csv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WANDB_CSV = REPO_ROOT / "mrbert" / "analysis" / "wandb_plots" / "all_runs_summary.csv"
OUTPUT_DIR = REPO_ROOT / "mrbert" / "analysis" / "figures"

# Run mapping: (run_name, label, layer, blend, notes)
RUNS = [
    {
        "run_name": "bert-tydiqa-baseline",
        "label": "BERT baseline",
        "layer": "---",
        "blend": "---",
        "is_baseline": True,
    },
    {
        "run_name": "mrbert-tydiqa-30pct",
        "label": "MrBERT 30\\% (no blend)",
        "layer": "3",
        "blend": "No",
        "is_baseline": False,
    },
    {
        "run_name": "mrbert-tydiqa-30pct-predel",
        "label": "MrBERT 30\\% (blend, L3)",
        "layer": "3",
        "blend": "Yes",
        "is_baseline": False,
    },
    {
        "run_name": "mrbert-tydiqa-30pct-layer9-predel",
        "label": "MrBERT 30\\% (blend, L9)",
        "layer": "9",
        "blend": "Yes",
        "is_baseline": False,
    },
    {
        "run_name": "mrbert-tydiqa-0pct-0_1wt-predel",
        "label": "MrBERT 0\\% (no deletion, blend)",
        "layer": "3",
        "blend": "Yes",
        "is_baseline": False,
        "is_ablation": True,
    },
]


def load_wandb_data():
    """Load TyDi QA runs from W&B summary CSV."""
    data = {}
    with open(WANDB_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            project = row.get("project", "")
            name = row.get("run_name", "")
            if project != "mrbert-tydiqa":
                continue

            span_em = row.get("eval/span_em")
            start_acc = row.get("eval/start_acc")
            end_acc = row.get("eval/end_acc")
            del_rate = row.get("train/percent_non_pad_deleted_tokens")

            # Also grab test metrics for comparison
            test_span_em = row.get("test/span_em")
            test_start_acc = row.get("test/start_acc")
            test_end_acc = row.get("test/end_acc")

            data[name] = {
                "eval_span_em": float(span_em) if span_em else None,
                "eval_start_acc": float(start_acc) if start_acc else None,
                "eval_end_acc": float(end_acc) if end_acc else None,
                "del_rate": float(del_rate) if del_rate else 0.0,
                "test_span_em": float(test_span_em) if test_span_em else None,
                "test_start_acc": float(test_start_acc) if test_start_acc else None,
                "test_end_acc": float(test_end_acc) if test_end_acc else None,
            }
    return data


def build_table(wandb_data):
    """Build table rows from W&B data."""
    rows = []
    for run_cfg in RUNS:
        run_name = run_cfg["run_name"]
        run_data = wandb_data.get(run_name)

        if run_data is None:
            print(f"WARNING: run '{run_name}' not found in W&B data")
            continue

        rows.append({
            "label": run_cfg["label"],
            "layer": run_cfg["layer"],
            "blend": run_cfg["blend"],
            "span_em": run_data["eval_span_em"],
            "start_acc": run_data["eval_start_acc"],
            "end_acc": run_data["eval_end_acc"],
            "del_rate": run_data["del_rate"],
            "is_baseline": run_cfg.get("is_baseline", False),
            "is_ablation": run_cfg.get("is_ablation", False),
            # Keep test metrics for verification
            "test_span_em": run_data["test_span_em"],
            "test_start_acc": run_data["test_start_acc"],
            "test_end_acc": run_data["test_end_acc"],
        })
    return rows


def fmt_metric(val):
    """Format a metric to 2 decimal places."""
    if val is None:
        return "---"
    return f"{val:.2f}"


def fmt_del_rate(val, is_baseline=False):
    """Format deletion rate."""
    if is_baseline:
        return "0\\%"
    if val == 0.0:
        return "$\\sim$0\\%"
    return f"{val:.1f}\\%"


def print_latex_table(rows):
    """Print Table 5 as LaTeX."""
    print(r"\begin{table}[t]")
    print(r"\centering\small")
    print(r"\begin{tabular}{@{}lcccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Span EM} & \textbf{Start} & "
          r"\textbf{End} & \textbf{Layer} & \textbf{Blend} & \textbf{Del.} \\")
    print(r"\midrule")

    for i, row in enumerate(rows):
        # Add ablation separator
        if row["is_ablation"]:
            print(r"\midrule")
            print(r"\multicolumn{7}{@{}l}{\textit{Ablation: 0\% deletion target}} \\")

        # Bold the best non-baseline span_em and start_acc
        non_baseline = [r for r in rows if not r["is_baseline"]]
        best_span = max((r["span_em"] for r in non_baseline if r["span_em"]), default=0)
        best_start = max((r["start_acc"] for r in non_baseline if r["start_acc"]), default=0)

        span_str = fmt_metric(row["span_em"])
        start_str = fmt_metric(row["start_acc"])
        end_str = fmt_metric(row["end_acc"])

        if not row["is_baseline"] and row["span_em"] and row["span_em"] >= best_span:
            span_str = r"\textbf{" + span_str + "}"
        if not row["is_baseline"] and row["start_acc"] and row["start_acc"] >= best_start:
            start_str = r"\textbf{" + start_str + "}"

        del_str = fmt_del_rate(row["del_rate"], row["is_baseline"])

        print(f"{row['label']:40s} & {span_str} & {start_str} & {end_str} "
              f"& {row['layer']:3s} & {row['blend']:3s} & {del_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{TyDi~QA English results. Pre-deletion blending is essential; "
          r"gate at layer~9 nearly matches BERT baseline while deleting "
          r"$\sim$25\% of non-padding tokens.}")
    print(r"\label{tab:tydiqa-results}")
    print(r"\end{table}")


def save_table_csv(rows, path):
    """Save table data as CSV for reproducibility."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "span_em", "start_acc", "end_acc",
            "layer", "blend", "del_rate_pct",
            "test_span_em", "test_start_acc", "test_end_acc",
        ])
        for row in rows:
            writer.writerow([
                row["label"].replace("\\%", "%"),
                f"{row['span_em']:.4f}" if row["span_em"] else "",
                f"{row['start_acc']:.4f}" if row["start_acc"] else "",
                f"{row['end_acc']:.4f}" if row["end_acc"] else "",
                row["layer"],
                row["blend"],
                f"{row['del_rate']:.2f}",
                f"{row['test_span_em']:.4f}" if row["test_span_em"] else "",
                f"{row['test_start_acc']:.4f}" if row["test_start_acc"] else "",
                f"{row['test_end_acc']:.4f}" if row["test_end_acc"] else "",
            ])
    print(f"\nSaved table CSV -> {path}")


def print_verification(rows):
    """Print verification summary."""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    baseline = next((r for r in rows if r["is_baseline"]), None)
    if not baseline:
        print("ERROR: no baseline found")
        return

    print(f"\nBERT baseline: span_em={baseline['span_em']:.4f}, "
          f"start={baseline['start_acc']:.4f}, end={baseline['end_acc']:.4f}")

    for row in rows:
        if row["is_baseline"]:
            continue
        print(f"\n{row['label'].replace(chr(92)+'%', '%')} (Layer {row['layer']}, "
              f"Blend={row['blend']}):")
        print(f"  Eval:  span_em={row['span_em']:.4f}, "
              f"start={row['start_acc']:.4f}, end={row['end_acc']:.4f}")
        if row["test_span_em"]:
            print(f"  Test:  span_em={row['test_span_em']:.4f}, "
                  f"start={row['test_start_acc']:.4f}, end={row['test_end_acc']:.4f}")
        print(f"  Del rate: {row['del_rate']:.2f}%")

        # Relative to baseline
        if row["span_em"] and baseline["span_em"]:
            rel_drop = (baseline["span_em"] - row["span_em"]) / baseline["span_em"] * 100
            abs_drop = baseline["span_em"] - row["span_em"]
            print(f"  vs baseline: {abs_drop:+.4f} ({rel_drop:.1f}% relative drop)")

    # Summary of improvement from blending
    no_blend = next((r for r in rows if "no blend" in r["label"]), None)
    best = max((r for r in rows if not r["is_baseline"] and not r.get("is_ablation")),
               key=lambda r: r["span_em"] or 0, default=None)

    if no_blend and best:
        improvement = best["span_em"] / no_blend["span_em"]
        print(f"\n{'=' * 70}")
        print(f"KEY NARRATIVE:")
        print(f"  No blend -> Best (L{best['layer']}+blend): "
              f"{no_blend['span_em']:.2f} -> {best['span_em']:.2f} "
              f"({improvement:.1f}x improvement)")
        print(f"  Best vs BERT baseline: {best['span_em']:.2f} vs "
              f"{baseline['span_em']:.2f} (gap = {baseline['span_em'] - best['span_em']:.2f})")


def main():
    wandb_data = load_wandb_data()

    print(f"Found {len(wandb_data)} runs in project mrbert-tydiqa")
    print(f"Available runs: {sorted(wandb_data.keys())}")
    print()

    rows = build_table(wandb_data)

    print("=" * 70)
    print("TABLE 5: TyDi QA Results (LaTeX)")
    print("=" * 70)
    print()
    print_latex_table(rows)

    # Save CSV
    csv_path = OUTPUT_DIR / "table5_tydiqa_results.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_table_csv(rows, csv_path)

    # Print verification
    print_verification(rows)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
