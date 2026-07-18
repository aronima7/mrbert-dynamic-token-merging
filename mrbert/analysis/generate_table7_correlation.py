"""
generate_table7_correlation.py

Generates Table 7 (per-example deletion rate vs loss correlation) for the
COLM 2026 paper.

Table 7 reports Pearson and Spearman correlations between per-example deletion
rate and cross-entropy loss across multiple models, tasks, and deletion targets.

Data source:
  - Each row is computed by running inference on a model checkpoint and
    computing per-example (deletion_rate, loss) pairs, then correlating them.
  - Primary script: analyze_deletion_correlation.py (root level)
  - Variant: deletion_correlation_analysis.py (SNLI-specific, uses test set)

Checkpoints used:
  - BERT, delta=0.3, MRPC:   mrbert/test_mrpc_mrbert/final/
  - BERT, delta=0.5, MRPC:   (Modal checkpoint, not local)
  - BERT, delta=0.3, SST-2:  mrbert/test_sst2_mrbert/final/
  - BERT, delta=0.5, SST-2:  (Modal checkpoint, not local)
  - BERT, delta=0.3, SNLI:   mrbert/local_checkpoints/mrbert-snli-30pct/final/
  - BERT, No-PI, SNLI:       (Modal checkpoint, not local)
  - XLM-R, delta=0.3, SST-2: (Modal checkpoint, not local)
  - XLM-R, delta=0.3, MRPC:  (Modal checkpoint, not local)
  - XLM-R, delta=0.5, XNLI:  (Modal checkpoint, not local)
  - XLM-R, delta=0.5, SNLI:  mrxlmr/test_snli_mrxlmr/final/

Note: The SNLI row uses the TEST set (2,000 examples) via
deletion_correlation_analysis.py. The main-text analysis (r=0.074) uses the
VALIDATION set (1,000 examples) via analyze_deletion_correlation.py.

Usage (from repo root):
  python mrbert/analysis/generate_table7_correlation.py
  python mrbert/analysis/generate_table7_correlation.py --recompute  # requires GPU + checkpoints

Output:
  - mrbert/analysis/figures/table7_correlation_results.csv
  - Prints LaTeX table to stdout
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "mrbert" / "analysis" / "figures"

# Verified values from analysis runs.
# Each row: (model, delta, dataset, pearson, spearman, interpretation, notes)
VERIFIED_ROWS = [
    {
        "model": "BERT",
        "delta": "0.3",
        "dataset": "MRPC",
        "pearson": +0.064,
        "spearman": +0.090,
        "interpretation": r"Higher del $\to$ higher loss",
        "checkpoint": "mrbert/test_mrpc_mrbert/final/",
        "split": "validation",
        "n_examples": 408,
    },
    {
        "model": "BERT",
        "delta": "0.5",
        "dataset": "MRPC",
        "pearson": +0.068,
        "spearman": +0.057,
        "interpretation": r"Higher del $\to$ higher loss",
        "checkpoint": "(Modal: mrbert-mrpc-50pct)",
        "split": "validation",
        "n_examples": 408,
    },
    {
        "model": "BERT",
        "delta": "0.3",
        "dataset": "SST-2",
        "pearson": -0.022,
        "spearman": -0.010,
        "interpretation": "Near zero",
        "checkpoint": "mrbert/test_sst2_mrbert/final/",
        "split": "validation",
        "n_examples": 872,
    },
    {
        "model": "BERT",
        "delta": "0.5",
        "dataset": "SST-2",
        "pearson": +0.035,
        "spearman": +0.015,
        "interpretation": "Weak positive",
        "checkpoint": "(Modal: mrbert-sst2-50pct)",
        "split": "validation",
        "n_examples": 872,
    },
    {
        "model": "BERT",
        "delta": "0.3",
        "dataset": "SNLI",
        "pearson": -0.048,
        "spearman": +0.060,
        "interpretation": "Mixed",
        "checkpoint": "mrbert/local_checkpoints/mrbert-snli-30pct/final/",
        "split": "test",
        "n_examples": 2000,
        "note": "Test set (2k examples); validation set (1k) yields r=+0.074, rho=+0.053",
    },
    {
        "model": "BERT",
        "delta": "No-PI",
        "dataset": "SNLI",
        "pearson": -0.019,
        "spearman": -0.016,
        "interpretation": "Near zero",
        "checkpoint": "(Modal: mrbert-snli-nopi)",
        "split": "test",
        "n_examples": 2000,
    },
    {
        "model": "XLM-R",
        "delta": "0.3",
        "dataset": "SST-2",
        "pearson": +0.195,
        "spearman": +0.211,
        "interpretation": r"Higher del $\to$ higher loss",
        "checkpoint": "(Modal: mrxlmr-sst2-30pct)",
        "split": "validation",
        "n_examples": 872,
    },
    {
        "model": "XLM-R",
        "delta": "0.3",
        "dataset": "MRPC",
        "pearson": -0.130,
        "spearman": -0.208,
        "interpretation": r"Del $\to$ lower loss",
        "checkpoint": "(Modal: mrxlmr-mrpc-30pct)",
        "split": "validation",
        "n_examples": 408,
    },
    {
        "model": "XLM-R",
        "delta": "0.5",
        "dataset": "XNLI",
        "pearson": -0.046,
        "spearman": -0.084,
        "interpretation": "Slight negative",
        "checkpoint": "(Modal: mrxlmr-xnli-50pct)",
        "split": "test",
        "n_examples": 2000,
    },
    {
        "model": "XLM-R",
        "delta": "0.5",
        "dataset": "SNLI",
        "pearson": +0.028,
        "spearman": +0.012,
        "interpretation": "Near zero",
        "checkpoint": "mrxlmr/test_snli_mrxlmr/final/",
        "split": "test",
        "n_examples": 2000,
    },
]


def recompute_row(row):
    """
    Recompute a single row by running inference on the checkpoint.
    Requires GPU and the checkpoint to exist locally.
    """
    from scipy import stats
    import torch

    checkpoint_path = REPO_ROOT / row["checkpoint"]
    if not checkpoint_path.exists():
        print(f"  SKIP (checkpoint not found): {row['checkpoint']}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: Running on CPU, this will be slow")

    model_type = row["model"]
    dataset_name = row["dataset"]

    if model_type == "BERT":
        sys.path.insert(0, str(REPO_ROOT / "mrbert" / "models"))
        from modeling_mrbert import MrBertForSequenceClassification
        from configuration_mrbert import MrBertConfig
        from transformers import BertTokenizerFast

        config = MrBertConfig.from_pretrained(str(checkpoint_path))
        model = MrBertForSequenceClassification.from_pretrained(
            str(checkpoint_path), config=config
        )
        tokenizer = BertTokenizerFast.from_pretrained(str(checkpoint_path))
    elif model_type == "XLM-R":
        sys.path.insert(0, str(REPO_ROOT / "mrxlmr" / "models"))
        from modeling_mrxlmr import MrXLMRForSequenceClassification
        from configuration_mrxlmr import MrXLMRConfig
        from transformers import AutoTokenizer

        config = MrXLMRConfig.from_pretrained(str(checkpoint_path))
        model = MrXLMRForSequenceClassification.from_pretrained(
            str(checkpoint_path), config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    else:
        print(f"  SKIP (unknown model type): {model_type}")
        return None

    model.to(device)
    model.eval()

    # Load dataset
    from datasets import load_dataset

    if dataset_name == "SNLI":
        ds = load_dataset("snli", split=row["split"])
        ds = ds.filter(lambda x: x["label"] != -1)
        text_cols = ("premise", "hypothesis")
    elif dataset_name == "SST-2":
        ds = load_dataset("glue", "sst2", split=row["split"])
        text_cols = ("sentence",)
    elif dataset_name == "MRPC":
        ds = load_dataset("glue", "mrpc", split=row["split"])
        text_cols = ("sentence1", "sentence2")
    elif dataset_name == "XNLI":
        ds = load_dataset("xnli", "en", split=row["split"])
        text_cols = ("premise", "hypothesis")
    else:
        print(f"  SKIP (unknown dataset): {dataset_name}")
        return None

    n = min(row["n_examples"], len(ds))
    ds = ds.select(range(n))

    # Compute per-example deletion rate and loss
    from torch.nn import CrossEntropyLoss
    ce = CrossEntropyLoss(reduction="none")

    deletion_rates = []
    losses = []

    with torch.no_grad():
        for i in range(0, n, 32):
            batch_examples = ds[i:i+32]
            if len(text_cols) == 1:
                enc = tokenizer(
                    batch_examples[text_cols[0]],
                    max_length=128, truncation=True, padding="max_length",
                    return_tensors="pt",
                )
            else:
                enc = tokenizer(
                    batch_examples[text_cols[0]], batch_examples[text_cols[1]],
                    max_length=128, truncation=True, padding="max_length",
                    return_tensors="pt",
                )

            labels = torch.tensor(batch_examples["label"])
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            outputs = model(**enc, labels=labels)
            batch_loss = ce(outputs.logits, labels).cpu().numpy()
            losses.extend(batch_loss.tolist())

            # Get deletion rate
            gate_mask = getattr(outputs, "delete_gate_mask", None)
            if gate_mask is None:
                gate_mask = getattr(outputs, "gate_values", None)

            if gate_mask is not None:
                if gate_mask.dim() == 3:
                    gate_mask = gate_mask.squeeze(-1)
                threshold = config.sigmoid_mask_scale / 2.0
                attn_mask = enc["attention_mask"].bool()
                for j in range(gate_mask.size(0)):
                    non_pad = attn_mask[j]
                    n_non_pad = non_pad.sum().item()
                    if n_non_pad > 0:
                        deleted = (gate_mask[j] < threshold) & non_pad
                        dr = deleted.sum().item() / n_non_pad
                    else:
                        dr = 0.0
                    deletion_rates.append(dr)
            else:
                deletion_rates.extend([0.0] * len(batch_loss))

            if (i // 32) % 10 == 0:
                print(f"    {i+min(32, n-i)}/{n}...")

    deletion_rates = np.array(deletion_rates)
    losses = np.array(losses)

    if deletion_rates.std() < 1e-6:
        print(f"  WARNING: constant deletion rate ({deletion_rates.mean()*100:.1f}%)")
        return {"pearson": 0.0, "spearman": 0.0}

    pr, _ = stats.pearsonr(deletion_rates, losses)
    sr, _ = stats.spearmanr(deletion_rates, losses)

    print(f"  Pearson={pr:+.3f}, Spearman={sr:+.3f} (n={len(deletion_rates)})")
    return {"pearson": float(pr), "spearman": float(sr)}


def print_latex_table(rows):
    """Print Table 7 as LaTeX."""
    print(r"\begin{table}[t]")
    print(r"\centering\small")
    print(r"\begin{tabular}{@{}llcc l@{}}")
    print(r"\toprule")
    print(r"\textbf{Run} & \textbf{Dataset} & \textbf{Pearson} & "
          r"\textbf{Spearman} & \textbf{Interpretation} \\")
    print(r"\midrule")

    for row in rows:
        if row["delta"] == "No-PI":
            run_str = f"{row['model']}, No-PI"
        else:
            run_str = f"{row['model']}, $\\delta$={row['delta']}"

        p_sign = "$+$" if row["pearson"] >= 0 else "$-$"
        s_sign = "$+$" if row["spearman"] >= 0 else "$-$"
        p_str = f"{p_sign}{abs(row['pearson']):.3f}"
        s_str = f"{s_sign}{abs(row['spearman']):.3f}"

        interp = row["interpretation"]
        # Add footnote marker for SNLI test-set row
        if row.get("note") and "Test set" in row["note"]:
            interp += r"$^\ddagger$"

        print(f"{run_str:25s} & {row['dataset']:6s} & {p_str:10s} & "
              f"{s_str:10s} & {interp} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Pearson and Spearman correlations between per-example "
          r"deletion rate and cross-entropy loss. Most correlations are near "
          r"zero, confirming the gate deletes wisely. Positive correlations on "
          r"MRPC and XLM-R SST-2 suggest over-deletion harms hard examples in "
          r"sensitive regimes. $^\ddagger$Computed on the SNLI test set (2,000 "
          r"examples); the validation-set analysis (1,000 examples) yields "
          r"$r{=}+0.074$, $\rho{=}+0.053$ --- both are weak, confirming the "
          r"same conclusion.}")
    print(r"\label{tab:loss-deletion-corr}")
    print(r"\end{table}")


def save_csv(rows, path):
    """Save table data as CSV."""
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "delta", "dataset", "pearson", "spearman",
            "interpretation", "split", "n_examples", "checkpoint",
        ])
        for row in rows:
            writer.writerow([
                row["model"],
                row["delta"],
                row["dataset"],
                f"{row['pearson']:+.3f}",
                f"{row['spearman']:+.3f}",
                row["interpretation"].replace(r"$\to$", "->"),
                row["split"],
                row["n_examples"],
                row["checkpoint"],
            ])
    print(f"\nSaved CSV -> {path}")


def print_verification(rows):
    """Print verification summary."""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<12} {'Delta':<7} {'Dataset':<7} {'Pearson':>8} "
          f"{'Spearman':>9} {'Split':<6} {'N':>5}")
    print("-" * 70)
    for row in rows:
        print(f"{row['model']:<12} {row['delta']:<7} {row['dataset']:<7} "
              f"{row['pearson']:>+8.3f} {row['spearman']:>+9.3f} "
              f"{row['split']:<6} {row['n_examples']:>5}")

    # Key findings
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS:")
    bert_sst2 = [r for r in rows if r["model"] == "BERT" and r["dataset"] == "SST-2"]
    if bert_sst2:
        max_r = max(abs(r["pearson"]) for r in bert_sst2)
        print(f"  SST-2 (BERT): max |r| = {max_r:.3f} — near zero, task is redundant")

    xlmr_sst2 = [r for r in rows if r["model"] == "XLM-R" and r["dataset"] == "SST-2"]
    if xlmr_sst2:
        print(f"  SST-2 (XLM-R): r = {xlmr_sst2[0]['pearson']:+.3f} — "
              f"notable positive, over-deletion on hard examples")

    mrpc = [r for r in rows if r["dataset"] == "MRPC" and r["model"] == "BERT"]
    if mrpc:
        print(f"  MRPC (BERT): r in [{mrpc[0]['pearson']:+.3f}, {mrpc[-1]['pearson']:+.3f}] — "
              f"weak positive, paraphrase-sensitive")

    # Note about SNLI discrepancy
    snli_row = next((r for r in rows if r["model"] == "BERT"
                     and r["delta"] == "0.3" and r["dataset"] == "SNLI"), None)
    if snli_row and snli_row.get("note"):
        print(f"\n  NOTE: {snli_row['note']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Table 7 (deletion-loss correlation)")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute correlations from checkpoints (requires GPU)")
    args = parser.parse_args()

    rows = VERIFIED_ROWS

    if args.recompute:
        print("Recomputing correlations from available checkpoints...\n")
        for row in rows:
            ckpt = row["checkpoint"]
            if ckpt.startswith("("):
                print(f"  SKIP (remote only): {row['model']} {row['delta']} {row['dataset']}")
                continue
            ckpt_path = REPO_ROOT / ckpt
            if not ckpt_path.exists():
                print(f"  SKIP (not found): {ckpt}")
                continue
            print(f"  Computing: {row['model']} {row['delta']} {row['dataset']}...")
            result = recompute_row(row)
            if result:
                print(f"    Old: Pearson={row['pearson']:+.3f}, Spearman={row['spearman']:+.3f}")
                print(f"    New: Pearson={result['pearson']:+.3f}, Spearman={result['spearman']:+.3f}")
                diff_p = abs(result["pearson"] - row["pearson"])
                diff_s = abs(result["spearman"] - row["spearman"])
                if diff_p > 0.01 or diff_s > 0.01:
                    print(f"    ⚠ DISCREPANCY: Pearson diff={diff_p:.3f}, Spearman diff={diff_s:.3f}")
        print()

    print("=" * 70)
    print("TABLE 7: Per-Example Deletion Rate vs Loss Correlation (LaTeX)")
    print("=" * 70)
    print()
    print_latex_table(rows)

    # Save CSV
    csv_path = OUTPUT_DIR / "table7_correlation_results.csv"
    save_csv(rows, csv_path)

    # Print verification
    print_verification(rows)

    # Reproduction commands
    print(f"\n{'=' * 70}")
    print("REPRODUCTION COMMANDS (requires GPU + checkpoints):")
    print("=" * 70)
    print()
    print("# Each row can be reproduced with analyze_deletion_correlation.py:")
    for row in rows:
        if row["checkpoint"].startswith("("):
            continue
        ds_arg = ""
        if row["dataset"] == "SST-2":
            ds_arg = "--dataset_name glue --dataset_config sst2"
        elif row["dataset"] == "MRPC":
            ds_arg = "--dataset_name glue --dataset_config mrpc"
        elif row["dataset"] == "SNLI":
            ds_arg = "--dataset_name snli"
        elif row["dataset"] == "XNLI":
            ds_arg = "--dataset_name xnli --dataset_config en"
        print(f"# {row['model']} delta={row['delta']} {row['dataset']}:")
        print(f"python analyze_deletion_correlation.py \\")
        print(f"    --model_path {row['checkpoint']} \\")
        print(f"    --task sequence_classification {ds_arg} \\")
        print(f"    --split {row['split']} "
              f"--max_samples {row['n_examples']} \\")
        print(f"    --output_dir ./deletion_analysis_{row['model'].lower()}_{row['dataset'].lower()}")
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
