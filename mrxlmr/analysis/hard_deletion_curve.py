"""
hard_deletion_curve.py

Post-hoc evaluation of soft vs hard deletion accuracy across training checkpoints.
Loads each saved checkpoint, runs SNLI test-set eval under both modes, and plots
the two accuracy curves vs training step.

Adapted from mrbert/analysis/hard_deletion_curve.py for XLM-RoBERTa:
  - XLMRobertaTokenizerFast instead of BertTokenizer
  - MrXLMRForSequenceClassification instead of MrBertForSequenceClassification
  - XLM-R pad_token_id = 1 (not 0)

Usage (from mrxlmr/ directory):
  python analysis/hard_deletion_curve.py \\
      --checkpoint_dir ./local_checkpoints/mrxlmr-snli-30pct \\
      --local_snli_dir ./snli_datasets \\
      --output_dir ./analysis/figures

  # Specify run name for output filenames
  python analysis/hard_deletion_curve.py \\
      --checkpoint_dir ./local_checkpoints/mrxlmr-snli-30pct \\
      --local_snli_dir ./snli_datasets \\
      --run_name mrxlmr-snli-30pct \\
      --output_dir ./analysis/figures

Output:
  <output_dir>/<run_name>_hard_deletion_curve.csv
  <output_dir>/<run_name>_hard_deletion_curve.pdf
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import XLMRobertaTokenizerFast

from configuration_mrxlmr import MrXLMRConfig
from modeling_mrxlmr import MrXLMRForSequenceClassification


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """
    Return (step, path) pairs for all checkpoint-* subdirs and the final/ dir,
    sorted by step. The 'final' checkpoint is assigned the largest step + 1
    so it appears last.
    """
    entries = []
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                entries.append((step, path))
            except (IndexError, ValueError):
                pass

    entries.sort(key=lambda x: x[0])

    final_path = os.path.join(checkpoint_dir, "final")
    if os.path.isdir(final_path):
        last_step = entries[-1][0] if entries else 0
        entries.append((last_step + 1, final_path))

    if not entries:
        raise RuntimeError(
            f"No checkpoint-* directories or final/ found in {checkpoint_dir}"
        )

    return entries


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_snli_test(local_snli_dir: str, tokenizer, max_seq_length: int, n_samples: int):
    """
    Load SNLI test examples as a TensorDataset of (input_ids, attention_mask, labels).
    Reads pre-tokenized NDJSON produced by preprocess_snli.py.

    XLM-R pad_token_id = 1 (not 0 as in BERT).
    """
    test_path = os.path.join(local_snli_dir, "snli-test.json")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"SNLI test file not found at {test_path}. "
            "Run data/preprocess_snli.py first."
        )

    def pad(seq, length, pad_val=1):
        seq = seq[:length]
        return seq + [pad_val] * (length - len(seq))

    input_ids_list, mask_list, label_list = [], [], []

    with open(test_path) as f:
        for line in f:
            if len(input_ids_list) >= n_samples:
                break
            ex = json.loads(line)
            ids  = ex["input_ids"]
            mask = ex["attention_mask"]
            lbl  = ex["labels"]
            if ids and isinstance(ids[0], list):
                seq_ids, seq_mask = ids[0], mask[0]
            else:
                seq_ids, seq_mask = ids, mask
            input_ids_list.append(pad(seq_ids, max_seq_length))
            mask_list.append(pad(seq_mask, max_seq_length, 0))
            label_list.append(lbl)

    input_ids      = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(mask_list,       dtype=torch.long)
    labels         = torch.tensor(label_list,      dtype=torch.long)

    print(f"  Loaded {len(input_ids)} SNLI test examples (max_seq_length={max_seq_length})")
    return TensorDataset(input_ids, attention_mask, labels)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, dataset: TensorDataset, batch_size: int, device: str,
                       hard_delete: bool) -> float:
    """
    Run inference over dataset and return accuracy.
    hard_delete=True physically removes tokens; hard_delete=False uses soft deletion.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    is_mrxlmr = getattr(model.config, "model_type", "") == "mrxlmr"

    correct = 0
    total   = 0

    model.eval()
    with torch.no_grad():
        for ids, mask, lbls in loader:
            ids, mask, lbls = ids.to(device), mask.to(device), lbls.to(device)
            if is_mrxlmr:
                outputs = model(input_ids=ids, attention_mask=mask, hard_delete=hard_delete)
            else:
                outputs = model(input_ids=ids, attention_mask=mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Plotting and CSV
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("step,step_label,soft_accuracy,hard_accuracy\n")
        for r in results:
            f.write(f"{r['step']},{r['step_label']},{r['soft_acc']:.6f},{r['hard_acc']:.6f}\n")
    print(f"Saved → {output_path}")


def plot_curve(results: list[dict], output_path: str, run_name: str):
    """
    Two-line plot: soft deletion accuracy and hard deletion accuracy vs training step.
    """
    step_labels = [r["step_label"] for r in results]
    soft_accs   = [r["soft_acc"]   for r in results]
    hard_accs   = [r["hard_acc"]   for r in results]
    x = list(range(len(step_labels)))

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 0.6 + 2), 4))

    ax.plot(x, soft_accs, marker="o", color="#377eb8", linewidth=1.8,
            markersize=4, label="Soft deletion (training mode)")
    ax.plot(x, hard_accs, marker="s", color="#e41a1c", linewidth=1.8,
            markersize=4, label="Hard deletion (inference mode)", linestyle="--")

    ax.fill_between(x, soft_accs, hard_accs, alpha=0.08, color="#e41a1c",
                    label="Gap (soft − hard)")

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Soft vs hard deletion accuracy over training — {run_name}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    last = results[-1]
    gap = abs(last["soft_acc"] - last["hard_acc"])
    ax.annotate(
        f"final gap: {gap*100:.2f}pp",
        xy=(x[-1], (last["soft_acc"] + last["hard_acc"]) / 2),
        xytext=(x[-1] - max(1, len(x) // 6), (last["soft_acc"] + last["hard_acc"]) / 2),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8,
        color="black",
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot soft vs hard deletion accuracy across training checkpoints (MrXLMR)."
    )
    p.add_argument(
        "--checkpoint_dir", required=True,
        help="Directory containing checkpoint-* subdirs and/or final/. "
             "E.g. ./local_checkpoints/mrxlmr-snli-30pct",
    )
    p.add_argument(
        "--local_snli_dir", required=True,
        help="Path to pre-tokenized SNLI NDJSON files (from data/preprocess_snli.py).",
    )
    p.add_argument(
        "--run_name", default="",
        help="Name prefix for output files (default: basename of checkpoint_dir).",
    )
    p.add_argument(
        "--output_dir", default="analysis/figures",
        help="Where to save the CSV and PDF (default: analysis/figures).",
    )
    p.add_argument(
        "--max_seq_length", type=int, default=128,
        help="Sequence length used during training (default: 128).",
    )
    p.add_argument(
        "--n_samples", type=int, default=9824,
        help="Number of SNLI test examples to evaluate (default: full test set ~9824).",
    )
    p.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for inference (default: 64).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or os.path.basename(args.checkpoint_dir.rstrip("/"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Run: {run_name}")

    # Discover checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir)
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  step={step}  {path}")

    # Load tokenizer from the first available checkpoint
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(checkpoints[0][1])

    # Load SNLI test set once
    print("\nLoading SNLI test set...")
    dataset = load_snli_test(
        local_snli_dir=args.local_snli_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        n_samples=args.n_samples,
    )

    # Evaluate each checkpoint
    results = []
    n = len(checkpoints)
    for i, (step, ckpt_path) in enumerate(checkpoints):
        step_label = "final" if i == n - 1 and os.path.basename(ckpt_path) == "final" else str(step)
        print(f"\n[{i+1}/{n}] {step_label}  ({ckpt_path})")

        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "mrxlmr":
            config = MrXLMRConfig.from_pretrained(ckpt_path)
            model = MrXLMRForSequenceClassification.from_pretrained(ckpt_path, config=config)
        else:
            from transformers import XLMRobertaForSequenceClassification
            model = XLMRobertaForSequenceClassification.from_pretrained(ckpt_path)
        model.to(device)

        soft_acc = evaluate_accuracy(model, dataset, args.batch_size, device, hard_delete=False)
        hard_acc = evaluate_accuracy(model, dataset, args.batch_size, device, hard_delete=True)

        print(f"  soft={soft_acc:.4f}  hard={hard_acc:.4f}  gap={abs(soft_acc-hard_acc)*100:.2f}pp")
        results.append({
            "step":       step,
            "step_label": step_label,
            "soft_acc":   soft_acc,
            "hard_acc":   hard_acc,
        })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print()
    csv_path = os.path.join(args.output_dir, f"{run_name}_hard_deletion_curve.csv")
    pdf_path = os.path.join(args.output_dir, f"{run_name}_hard_deletion_curve.pdf")
    save_csv(results, csv_path)
    plot_curve(results, pdf_path, run_name)
    print("\nDone.")


if __name__ == "__main__":
    main()