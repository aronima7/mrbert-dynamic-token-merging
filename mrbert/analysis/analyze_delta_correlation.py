#!/usr/bin/env python3
"""
Analyze per-example correlation between deletion rate and DELTA loss (MrBERT - BERT).

Inspired by MrT5 (Kallini et al., 2024) Section 7 "Per-sample Sequence Length Reduction":
they correlate per-sample sequence length reduction with the percent increase in BPB
relative to baseline ByT5, finding r=0.103 for MrT5 (learned) vs r=0.295 for random.

This script computes the analogous analysis for MrBERT:
  - delta[i] = loss_MrBERT[i] - loss_BERT[i]  (per-example degradation vs baseline)
  - Correlate deletion_rate[i] with delta[i]
  - Compare learned gate (MrBERT) vs random gate to show strategic deletion

This is stronger than correlating deletion_rate vs absolute loss because it isolates
the *degradation caused by deletion* from the *inherent difficulty of the sample*.

Usage:
    # Basic: MrBERT vs BERT on SNLI
    python analyze_delta_correlation.py \
        --baseline_path ./path/to/bert-snli-baseline/final \
        --mrbert_path ./path/to/mrbert-snli-30pct/final \
        --dataset_name snli \
        --output_dir ./delta_analysis_snli

    # Full comparison including random baseline
    python analyze_delta_correlation.py \
        --baseline_path ./path/to/bert-snli-baseline/final \
        --mrbert_path ./path/to/mrbert-snli-30pct/final \
        --random_path ./path/to/mrbert-snli-random30/final \
        --dataset_name snli \
        --output_dir ./delta_analysis_snli

    # On SNLI with local checkpoints
    python analyze_delta_correlation.py \
        --baseline_path bert-base-uncased \
        --baseline_is_pretrained \
        --mrbert_path ./mrbert/local_checkpoints/mrbert-snli-30pct/final \
        --dataset_name snli \
        --split validation \
        --max_samples 1000 \
        --output_dir ./delta_analysis_snli
"""

import sys
import os
import argparse
import json
from typing import Optional, List, Dict

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DefaultDataCollator,
)
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mrbert", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mrbert", "training"))

from configuration_mrbert import MrBertConfig
from modeling_mrbert import MrBertForSequenceClassification


def load_baseline_model(path: str, is_pretrained: bool, num_labels: int, device: str):
    """Load baseline BERT model (no delete gate)."""
    print(f"Loading baseline BERT from: {path}")
    if is_pretrained:
        model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(path)
    return model, tokenizer


def load_mrbert_model(path: str, device: str):
    """Load MrBERT model (with delete gate)."""
    print(f"Loading MrBERT from: {path}")
    model = MrBertForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(path)
    config = MrBertConfig.from_pretrained(path)
    return model, tokenizer, config


def prepare_dataset(dataset_name: str, dataset_config: Optional[str],
                    split: str, tokenizer, max_seq_length: int,
                    max_samples: Optional[int]):
    """Load and tokenize dataset."""
    print(f"Loading dataset: {dataset_name}/{dataset_config or ''} split={split}")

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    if split not in dataset:
        print(f"Split '{split}' not found, trying 'test'...")
        split = "test"

    eval_dataset = dataset[split]

    if dataset_name in ("snli", "multi_nli"):
        text_columns = ["premise", "hypothesis"]
    elif dataset_name == "xnli":
        text_columns = ["premise", "hypothesis"]
    elif dataset_name == "glue":
        if dataset_config in ["sst2", "cola"]:
            text_columns = ["sentence"]
        elif dataset_config in ["mrpc", "stsb", "rte", "wnli"]:
            text_columns = ["sentence1", "sentence2"]
        elif dataset_config == "mnli":
            text_columns = ["premise", "hypothesis"]
        elif dataset_config == "qnli":
            text_columns = ["question", "sentence"]
        elif dataset_config == "qqp":
            text_columns = ["question1", "question2"]
        else:
            text_columns = ["sentence"]
    else:
        cols = eval_dataset.column_names
        if "premise" in cols and "hypothesis" in cols:
            text_columns = ["premise", "hypothesis"]
        elif "sentence" in cols:
            text_columns = ["sentence"]
        else:
            text_columns = [cols[0]]

    def tokenize_fn(examples):
        if len(text_columns) == 1:
            return tokenizer(examples[text_columns[0]], truncation=True,
                           max_length=max_seq_length, padding="max_length")
        return tokenizer(examples[text_columns[0]], examples[text_columns[1]],
                       truncation=True, max_length=max_seq_length, padding="max_length")

    tokenized = eval_dataset.map(tokenize_fn, batched=True)
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")

    if "labels" in tokenized.column_names:
        tokenized = tokenized.filter(lambda x: x["labels"] >= 0)

    if max_samples:
        tokenized = tokenized.select(range(min(max_samples, len(tokenized))))
        print(f"  Using {len(tokenized)} samples")

    return tokenized


def compute_per_example_loss(model, dataloader, device: str, is_mrbert: bool = False,
                             config=None, pad_token_id: int = 0):
    """
    Compute per-example loss (and deletion rate if MrBERT).

    Returns:
        losses: np.ndarray of shape (N,)
        deletion_rates: np.ndarray of shape (N,) — all zeros if not MrBERT
    """
    all_losses = []
    all_deletion_rates = []

    model_input_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'labels'}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}...")

            batch = {k: v.to(device) for k, v in batch.items() if k in model_input_keys}
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)

            outputs = model(**batch)

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_example_loss = loss_fct(outputs.logits, labels)
            all_losses.extend(per_example_loss.cpu().numpy().tolist())

            if is_mrbert and config is not None:
                delete_gate_mask = getattr(outputs, "delete_gate_mask", None)
                if delete_gate_mask is not None:
                    if delete_gate_mask.dim() == 3:
                        delete_gate_mask = delete_gate_mask.squeeze(-1)

                    deletion_threshold = config.sigmoid_mask_scale / 2.0

                    for i in range(batch_size):
                        non_pad_mask = input_ids[i] != pad_token_id
                        num_non_pad = non_pad_mask.sum().item()
                        if num_non_pad > 0:
                            deleted = (delete_gate_mask[i] < deletion_threshold) & non_pad_mask
                            dr = deleted.sum().item() / num_non_pad
                        else:
                            dr = 0.0
                        all_deletion_rates.append(dr)
                else:
                    all_deletion_rates.extend([0.0] * batch_size)
            else:
                all_deletion_rates.extend([0.0] * batch_size)

    return np.array(all_losses), np.array(all_deletion_rates)


def compute_delta_correlation(deletion_rates, deltas, label: str):
    """Compute and print correlation between deletion rate and loss delta."""
    valid = np.isfinite(deletion_rates) & np.isfinite(deltas)
    dr = deletion_rates[valid]
    d = deltas[valid]

    if len(dr) < 3 or dr.std() < 1e-6:
        print(f"  [{label}] Cannot compute correlation (constant deletion rate)")
        return {"pearson": 0.0, "spearman": 0.0, "n": len(dr)}

    pearson_r, pearson_p = stats.pearsonr(dr, d)
    spearman_r, spearman_p = stats.spearmanr(dr, d)

    print(f"  [{label}]")
    print(f"    Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.4e})")
    print(f"    Spearman ρ = {spearman_r:+.4f}  (p = {spearman_p:.4e})")
    print(f"    N = {len(dr)}")

    return {
        "pearson": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n": int(len(dr)),
    }


def create_plots(output_dir: str, mrbert_dr, mrbert_delta,
                 random_dr=None, random_delta=None):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    has_random = random_dr is not None and random_delta is not None

    # --- Plot 1: Scatter comparison ---
    fig, axes = plt.subplots(1, 2 if has_random else 1,
                             figsize=(14 if has_random else 8, 6))
    if not has_random:
        axes = [axes]

    # MrBERT panel
    ax = axes[0]
    ax.scatter(mrbert_dr * 100, mrbert_delta, alpha=0.3, s=10, color='tab:blue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if mrbert_dr.std() > 1e-6:
        z = np.polyfit(mrbert_dr, mrbert_delta, 1)
        p = np.poly1d(z)
        x_line = np.linspace(mrbert_dr.min(), mrbert_dr.max(), 100)
        r = stats.pearsonr(mrbert_dr, mrbert_delta)[0]
        ax.plot(x_line * 100, p(x_line), "r-", linewidth=2,
                label=f'r = {r:+.3f}')
        ax.legend(fontsize=12)
    ax.set_xlabel('Deletion Rate (%)', fontsize=12)
    ax.set_ylabel('Δ Loss (MrBERT − BERT)', fontsize=12)
    ax.set_title('MrBERT (Learned Gate)', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Random panel
    if has_random:
        ax = axes[1]
        ax.scatter(random_dr * 100, random_delta, alpha=0.3, s=10, color='tab:orange')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if random_dr.std() > 1e-6:
            z = np.polyfit(random_dr, random_delta, 1)
            p = np.poly1d(z)
            x_line = np.linspace(random_dr.min(), random_dr.max(), 100)
            r = stats.pearsonr(random_dr, random_delta)[0]
            ax.plot(x_line * 100, p(x_line), "r-", linewidth=2,
                    label=f'r = {r:+.3f}')
            ax.legend(fontsize=12)
        ax.set_xlabel('Deletion Rate (%)', fontsize=12)
        ax.set_ylabel('Δ Loss (Random − BERT)', fontsize=12)
        ax.set_title('Random Deletion Baseline', fontsize=13)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "delta_correlation_scatter.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 2: Combined overlay (if both exist) ---
    if has_random:
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.scatter(mrbert_dr * 100, mrbert_delta, alpha=0.25, s=10,
                   color='tab:blue', label='MrBERT (learned)')
        ax.scatter(random_dr * 100, random_delta, alpha=0.25, s=10,
                   color='tab:orange', label='Random deletion')

        # Regression lines
        if mrbert_dr.std() > 1e-6:
            z = np.polyfit(mrbert_dr, mrbert_delta, 1)
            x_line = np.linspace(0, max(mrbert_dr.max(), random_dr.max()), 100)
            r_learned = stats.pearsonr(mrbert_dr, mrbert_delta)[0]
            ax.plot(x_line * 100, np.poly1d(z)(x_line), color='tab:blue',
                    linewidth=2.5, linestyle='-',
                    label=f'Learned fit (r={r_learned:+.3f})')

        if random_dr.std() > 1e-6:
            z = np.polyfit(random_dr, random_delta, 1)
            r_random = stats.pearsonr(random_dr, random_delta)[0]
            ax.plot(x_line * 100, np.poly1d(z)(x_line), color='tab:orange',
                    linewidth=2.5, linestyle='--',
                    label=f'Random fit (r={r_random:+.3f})')

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Deletion Rate (%)', fontsize=13)
        ax.set_ylabel('Δ Loss (Model − Baseline BERT)', fontsize=13)
        ax.set_title('Per-Example Degradation vs Deletion Rate:\nLearned Gate vs Random', fontsize=14)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "delta_correlation_overlay.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # --- Plot 3: Binned delta by deletion rate ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_labels = [f"{lo*100:.0f}-{hi*100:.0f}%" for lo, hi in bins]

    mrbert_means = []
    random_means = []
    mrbert_stds = []
    random_stds = []

    for lo, hi in bins:
        mask = (mrbert_dr >= lo) & (mrbert_dr < hi)
        if mask.sum() > 0:
            mrbert_means.append(mrbert_delta[mask].mean())
            mrbert_stds.append(mrbert_delta[mask].std() / np.sqrt(mask.sum()))
        else:
            mrbert_means.append(0)
            mrbert_stds.append(0)

        if has_random:
            mask_r = (random_dr >= lo) & (random_dr < hi)
            if mask_r.sum() > 0:
                random_means.append(random_delta[mask_r].mean())
                random_stds.append(random_delta[mask_r].std() / np.sqrt(mask_r.sum()))
            else:
                random_means.append(0)
                random_stds.append(0)

    x = np.arange(len(bins))
    width = 0.35 if has_random else 0.6

    ax.bar(x - (width/2 if has_random else 0), mrbert_means,
           width, yerr=mrbert_stds, capsize=4,
           label='MrBERT (learned)', color='tab:blue', alpha=0.7)

    if has_random:
        ax.bar(x + width/2, random_means, width, yerr=random_stds,
               capsize=4, label='Random', color='tab:orange', alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Deletion Rate Bin', fontsize=12)
    ax.set_ylabel('Mean Δ Loss (± SE)', fontsize=12)
    ax.set_title('Mean Degradation by Deletion Rate Bin', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, "delta_binned_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-example delta-loss correlation (MrT5-style analysis)")
    parser.add_argument("--baseline_path", type=str, required=True,
                       help="Path to baseline BERT checkpoint (no delete gate)")
    parser.add_argument("--baseline_is_pretrained", action="store_true",
                       help="If set, load baseline from HuggingFace pretrained "
                            "(requires --num_labels)")
    parser.add_argument("--mrbert_path", type=str, required=True,
                       help="Path to MrBERT checkpoint (learned gate)")
    parser.add_argument("--random_path", type=str, default=None,
                       help="Path to random-deletion MrBERT checkpoint (optional)")
    parser.add_argument("--dataset_name", type=str, default="snli",
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Dataset config/subset (e.g., 'sst2' for glue)")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split to evaluate on")
    parser.add_argument("--max_seq_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Number of examples to analyze")
    parser.add_argument("--num_labels", type=int, default=3,
                       help="Number of labels (needed if --baseline_is_pretrained)")
    parser.add_argument("--output_dir", type=str, default="./delta_analysis",
                       help="Output directory")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # --- Load models ---
    print("=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    mrbert_model, mrbert_tokenizer, mrbert_config = load_mrbert_model(
        args.mrbert_path, args.device)
    num_labels = mrbert_model.config.num_labels

    baseline_model, baseline_tokenizer = load_baseline_model(
        args.baseline_path, args.baseline_is_pretrained, num_labels, args.device)

    random_model, random_config = None, None
    if args.random_path:
        random_model, _, random_config = load_mrbert_model(
            args.random_path, args.device)

    # --- Prepare dataset (tokenize once, use for all models) ---
    print("\n" + "=" * 70)
    print("PREPARING DATASET")
    print("=" * 70)

    eval_dataset = prepare_dataset(
        args.dataset_name, args.dataset_config, args.split,
        mrbert_tokenizer, args.max_seq_length, args.max_samples)

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DefaultDataCollator(),
    )

    pad_token_id = mrbert_tokenizer.pad_token_id

    # --- Run inference on all models ---
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)

    print("\n--- Baseline BERT ---")
    baseline_losses, _ = compute_per_example_loss(
        baseline_model, dataloader, args.device,
        is_mrbert=False, pad_token_id=pad_token_id)

    print("\n--- MrBERT (learned gate) ---")
    mrbert_losses, mrbert_dr = compute_per_example_loss(
        mrbert_model, dataloader, args.device,
        is_mrbert=True, config=mrbert_config, pad_token_id=pad_token_id)

    random_losses, random_dr = None, None
    if random_model is not None:
        print("\n--- MrBERT (random gate) ---")
        random_losses, random_dr = compute_per_example_loss(
            random_model, dataloader, args.device,
            is_mrbert=True, config=random_config, pad_token_id=pad_token_id)

    # --- Compute deltas ---
    mrbert_delta = mrbert_losses - baseline_losses
    random_delta = random_losses - baseline_losses if random_losses is not None else None

    # --- Correlation analysis ---
    print("\n" + "=" * 70)
    print("CORRELATION: deletion_rate vs Δ loss (model − baseline)")
    print("=" * 70)
    print()

    mrbert_corr = compute_delta_correlation(mrbert_dr, mrbert_delta, "MrBERT (learned)")
    print()

    random_corr = None
    if random_dr is not None and random_delta is not None:
        random_corr = compute_delta_correlation(random_dr, random_delta, "Random baseline")
        print()

    # --- Summary comparison ---
    print("=" * 70)
    print("SUMMARY (cf. MrT5 paper: learned r=0.103, random r=0.295)")
    print("=" * 70)
    print(f"\n  MrBERT (learned gate):   r = {mrbert_corr['pearson']:+.3f}")
    if random_corr:
        print(f"  MrBERT (random gate):    r = {random_corr['pearson']:+.3f}")
        print(f"\n  Ratio (random/learned):  {abs(random_corr['pearson']) / max(abs(mrbert_corr['pearson']), 1e-6):.1f}×")
    print(f"\n  Interpretation:")
    if abs(mrbert_corr['pearson']) < 0.15:
        print(f"    ✓ Learned gate: weak correlation — deletion does NOT proportionally")
        print(f"      degrade examples. The gate deletes strategically.")
    else:
        print(f"    ⚠ Learned gate: moderate correlation — some degradation scales")
        print(f"      with deletion rate.")
    if random_corr and abs(random_corr['pearson']) > abs(mrbert_corr['pearson']):
        print(f"    ✓ Random baseline shows HIGHER correlation ({random_corr['pearson']:+.3f}),")
        print(f"      confirming learned gate is strategic, not just lucky.")

    # --- Statistics ---
    print(f"\n  Deletion rate stats (learned):")
    print(f"    Mean: {mrbert_dr.mean()*100:.1f}%  Std: {mrbert_dr.std()*100:.1f}%  "
          f"Range: [{mrbert_dr.min()*100:.1f}%, {mrbert_dr.max()*100:.1f}%]")
    print(f"  Delta loss stats (learned):")
    print(f"    Mean: {mrbert_delta.mean():.4f}  Std: {mrbert_delta.std():.4f}  "
          f"Median: {np.median(mrbert_delta):.4f}")
    pct_worse = (mrbert_delta > 0).mean() * 100
    print(f"    % examples where MrBERT is worse than BERT: {pct_worse:.1f}%")

    if random_dr is not None:
        print(f"\n  Deletion rate stats (random):")
        print(f"    Mean: {random_dr.mean()*100:.1f}%  Std: {random_dr.std()*100:.1f}%  "
              f"Range: [{random_dr.min()*100:.1f}%, {random_dr.max()*100:.1f}%]")
        print(f"  Delta loss stats (random):")
        print(f"    Mean: {random_delta.mean():.4f}  Std: {random_delta.std():.4f}  "
              f"Median: {np.median(random_delta):.4f}")
        pct_worse_r = (random_delta > 0).mean() * 100
        print(f"    % examples where Random is worse than BERT: {pct_worse_r:.1f}%")

    # --- Plots ---
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print("=" * 70)

    create_plots(args.output_dir, mrbert_dr, mrbert_delta, random_dr, random_delta)

    # --- Save results ---
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "config": {
            "baseline_path": args.baseline_path,
            "mrbert_path": args.mrbert_path,
            "random_path": args.random_path,
            "dataset": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "max_samples": args.max_samples,
            "n_examples": int(len(mrbert_dr)),
        },
        "mrbert_learned": {
            "correlation": mrbert_corr,
            "deletion_rate_mean": float(mrbert_dr.mean()),
            "deletion_rate_std": float(mrbert_dr.std()),
            "delta_loss_mean": float(mrbert_delta.mean()),
            "delta_loss_std": float(mrbert_delta.std()),
            "pct_worse_than_baseline": float((mrbert_delta > 0).mean()),
        },
    }
    if random_corr:
        results["random_baseline"] = {
            "correlation": random_corr,
            "deletion_rate_mean": float(random_dr.mean()),
            "deletion_rate_std": float(random_dr.std()),
            "delta_loss_mean": float(random_delta.mean()),
            "delta_loss_std": float(random_delta.std()),
            "pct_worse_than_baseline": float((random_delta > 0).mean()),
        }
    results["comparison_to_mrt5"] = {
        "mrt5_learned_r": 0.103,
        "mrt5_random_r": 0.295,
        "mrbert_learned_r": mrbert_corr["pearson"],
        "mrbert_random_r": random_corr["pearson"] if random_corr else None,
    }

    # Save per-example data for further analysis
    np.savez(
        os.path.join(args.output_dir, "per_example_data.npz"),
        mrbert_deletion_rates=mrbert_dr,
        mrbert_delta_losses=mrbert_delta,
        baseline_losses=baseline_losses,
        mrbert_losses=mrbert_losses,
        random_deletion_rates=random_dr if random_dr is not None else np.array([]),
        random_delta_losses=random_delta if random_delta is not None else np.array([]),
    )

    json_path = os.path.join(args.output_dir, "delta_analysis_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results: {json_path}")
    print(f"  Saved per-example data: {os.path.join(args.output_dir, 'per_example_data.npz')}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
