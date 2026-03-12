#!/usr/bin/env python3
"""
Analyze per-example correlation between deletion rate and loss.

This script evaluates whether MrBERT makes "smart" decisions about token deletion:
if examples with higher deletion rates don't show proportionally higher losses,
that suggests the model is successfully identifying and removing less important tokens.

Usage:
    python analyze_deletion_correlation.py \
        --model_path ./mrbert_sst2/final \
        --task sequence_classification \
        --dataset_name glue \
        --dataset_config sst2 \
        --output_dir ./deletion_analysis

    python analyze_deletion_correlation.py \
        --model_path ./mrbert_wikitext/final \
        --task mlm \
        --dataset_name wikitext \
        --dataset_config wikitext-2-raw-v1

    python analyze_deletion_correlation.py \
        --model_path ./mrbert_squad/final \
        --task question_answering \
        --dataset_name squad
"""

import sys
import os
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from transformers import BertTokenizer, DefaultDataCollator
from datasets import load_dataset

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mrbert", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mrbert", "training"))

from configuration_mrbert import MrBertConfig
from modeling_mrbert import (
    MrBertForMaskedLM,
    MrBertForSequenceClassification,
    MrBertForTokenClassification,
    MrBertForQuestionAnswering,
)


@dataclass
class AnalysisArgs:
    """Arguments for deletion rate vs loss analysis."""
    model_path: str
    task: str = "sequence_classification"  # mlm | sequence_classification | token_classification | question_answering
    dataset_name: str = "glue"
    dataset_config: Optional[str] = "sst2"
    max_seq_length: int = 512
    batch_size: int = 8
    max_samples: Optional[int] = None
    output_dir: str = "./deletion_analysis"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    split: str = "validation"  # which dataset split to analyze


def load_model(args: AnalysisArgs):
    """Load trained MrBERT model from checkpoint."""
    print(f"Loading model from: {args.model_path}")

    if args.task == "mlm":
        model = MrBertForMaskedLM.from_pretrained(args.model_path)
    elif args.task == "sequence_classification":
        model = MrBertForSequenceClassification.from_pretrained(args.model_path)
    elif args.task == "token_classification":
        model = MrBertForTokenClassification.from_pretrained(args.model_path)
    elif args.task == "question_answering":
        model = MrBertForQuestionAnswering.from_pretrained(args.model_path)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    model.to(args.device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    return model, tokenizer


def prepare_dataset(args: AnalysisArgs, tokenizer):
    """Load and tokenize dataset."""
    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config}")

    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)

    # Get the appropriate split
    if args.split not in dataset:
        print(f"Split '{args.split}' not found, trying 'test'...")
        args.split = "test"

    eval_dataset = dataset[args.split]

    # Tokenize based on task
    if args.task == "mlm":
        text_column = "text" if "text" in eval_dataset.column_names else eval_dataset.column_names[0]

        def tokenize_fn(examples):
            texts = [t for t in examples[text_column] if t and len(t.strip()) > 0]
            if not texts:
                return {"input_ids": [], "attention_mask": []}
            return tokenizer(texts, truncation=True, max_length=args.max_seq_length,
                           padding="max_length", return_special_tokens_mask=True)

        tokenized = eval_dataset.map(tokenize_fn, batched=True,
                                     remove_columns=eval_dataset.column_names)
        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

    elif args.task == "sequence_classification":
        # Handle SNLI, MNLI, XNLI
        if args.dataset_name in ("snli", "multi_nli", "xnli"):
            text_columns = ["premise", "hypothesis"]
        # Handle GLUE tasks
        elif args.dataset_name == "glue":
            if args.dataset_config in ["sst2", "cola"]:
                text_columns = ["sentence"]
            elif args.dataset_config in ["mrpc", "stsb", "rte", "wnli"]:
                text_columns = ["sentence1", "sentence2"]
            elif args.dataset_config == "mnli":
                text_columns = ["premise", "hypothesis"]
            elif args.dataset_config == "qnli":
                text_columns = ["question", "sentence"]
            elif args.dataset_config == "qqp":
                text_columns = ["question1", "question2"]
            else:
                text_columns = ["sentence"]
        else:
            cols = eval_dataset.column_names
            text_columns = ["text"] if "text" in cols else (["sentence"] if "sentence" in cols else [cols[0]])

        def tokenize_fn(examples):
            if len(text_columns) == 1:
                return tokenizer(examples[text_columns[0]], truncation=True,
                               max_length=args.max_seq_length, padding="max_length")
            return tokenizer(examples[text_columns[0]], examples[text_columns[1]],
                           truncation=True, max_length=args.max_seq_length, padding="max_length")

        tokenized = eval_dataset.map(tokenize_fn, batched=True)
        if "label" in tokenized.column_names:
            tokenized = tokenized.rename_column("label", "labels")

        # Filter out examples with invalid labels (e.g., -1 in SNLI)
        if "labels" in tokenized.column_names:
            tokenized = tokenized.filter(lambda x: x["labels"] >= 0)

    elif args.task == "question_answering":
        def tokenize_fn(examples):
            return tokenizer(
                examples["question"], examples["context"],
                truncation="only_second",
                max_length=args.max_seq_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

        tokenized = eval_dataset.map(tokenize_fn, batched=True,
                                     remove_columns=eval_dataset.column_names)

    else:
        raise ValueError(f"Task {args.task} not yet supported in this analysis script")

    if args.max_samples:
        tokenized = tokenized.select(range(min(args.max_samples, len(tokenized))))
        print(f"Limited to {len(tokenized)} samples")

    return tokenized


def compute_per_example_metrics(model, dataloader, config, pad_token_id: int, device: str):
    """
    Compute loss and deletion rate for each example.

    Returns:
        List[Tuple[float, float]]: List of (deletion_rate, loss) tuples for each example
    """
    results = []

    print("Computing per-example metrics...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx * dataloader.batch_size} examples...")

            # Move batch to device and filter to only model inputs
            # Remove any extra keys that the model doesn't expect (like 'idx')
            model_inputs = ['input_ids', 'attention_mask', 'token_type_ids', 'labels',
                           'start_positions', 'end_positions']
            batch = {k: v.to(device) for k, v in batch.items() if k in model_inputs}

            # Save labels before forward pass (some models pop it)
            labels_backup = batch.get("labels", None)
            start_pos_backup = batch.get("start_positions", None)
            end_pos_backup = batch.get("end_positions", None)

            # Forward pass
            outputs = model(**batch)

            # Get per-example losses
            # Most models return mean loss, so we need to recompute per-example
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)

            # Compute per-example loss
            if hasattr(outputs, "logits") and labels_backup is not None:
                logits = outputs.logits
                labels = labels_backup

                # For classification tasks
                if len(logits.shape) == 2:
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    per_example_loss = loss_fct(logits, labels)
                # For MLM or token classification
                elif len(logits.shape) == 3:
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                    per_example_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    # Average over non-ignored positions
                    per_example_loss = per_example_loss.view(batch_size, -1)
                    valid_mask = (labels != -100).float()
                    per_example_loss = (per_example_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
                else:
                    # Fallback: use the mean loss for all examples (not ideal but works)
                    per_example_loss = torch.full((batch_size,), outputs.loss.item())

            elif hasattr(outputs, "start_logits") and hasattr(outputs, "end_logits"):
                # Question answering task
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                start_positions = start_pos_backup
                end_positions = end_pos_backup

                if start_positions is not None and end_positions is not None:
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    per_example_loss = (start_loss + end_loss) / 2
                else:
                    per_example_loss = torch.full((batch_size,), outputs.loss.item())
            else:
                # Fallback
                per_example_loss = torch.full((batch_size,), outputs.loss.item())

            # Get deletion rates
            delete_gate_mask = getattr(outputs, "delete_gate_mask", None)

            if delete_gate_mask is not None:
                # delete_gate_mask shape: (batch, seq_len, 1) or (batch, seq_len)
                if delete_gate_mask.dim() == 3:
                    delete_gate_mask = delete_gate_mask.squeeze(-1)

                # Get deletion threshold from config, or default to sigmoid_mask_scale / 2.0
                if hasattr(config, 'deletion_threshold') and config.deletion_threshold is not None:
                    deletion_threshold = config.deletion_threshold
                else:
                    deletion_threshold = config.sigmoid_mask_scale / 2.0

                # For each example, compute deletion rate over non-pad tokens
                for i in range(batch_size):
                    non_pad_mask = input_ids[i] != pad_token_id
                    num_non_pad = non_pad_mask.sum().item()

                    if num_non_pad > 0:
                        # Count tokens below deletion threshold
                        deleted = (delete_gate_mask[i] < deletion_threshold) & non_pad_mask
                        num_deleted = deleted.sum().item()
                        deletion_rate = num_deleted / num_non_pad
                    else:
                        deletion_rate = 0.0

                    loss = per_example_loss[i].item()
                    results.append((deletion_rate, loss))
            else:
                # No deletion (baseline BERT or gate disabled)
                for i in range(batch_size):
                    results.append((0.0, per_example_loss[i].item()))

    print(f"Computed metrics for {len(results)} examples")
    return results


def analyze_correlation(results: List[Tuple[float, float]], output_dir: str):
    """
    Analyze correlation between deletion rate and loss.

    Creates visualizations and summary statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    deletion_rates = np.array([r[0] for r in results])
    losses = np.array([r[1] for r in results])

    # Remove any NaN or inf values
    valid_mask = np.isfinite(deletion_rates) & np.isfinite(losses)
    deletion_rates = deletion_rates[valid_mask]
    losses = losses[valid_mask]

    print(f"\n{'='*70}")
    print("DELETION RATE vs LOSS ANALYSIS")
    print(f"{'='*70}")

    # Summary statistics
    print(f"\nDeletion Rate Statistics:")
    print(f"  Mean:   {deletion_rates.mean()*100:.2f}%")
    print(f"  Median: {np.median(deletion_rates)*100:.2f}%")
    print(f"  Std:    {deletion_rates.std()*100:.2f}%")
    print(f"  Min:    {deletion_rates.min()*100:.2f}%")
    print(f"  Max:    {deletion_rates.max()*100:.2f}%")

    print(f"\nLoss Statistics:")
    print(f"  Mean:   {losses.mean():.4f}")
    print(f"  Median: {np.median(losses):.4f}")
    print(f"  Std:    {losses.std():.4f}")
    print(f"  Min:    {losses.min():.4f}")
    print(f"  Max:    {losses.max():.4f}")

    # Correlation analysis
    if len(deletion_rates) > 2 and deletion_rates.std() > 0:
        pearson_r, pearson_p = stats.pearsonr(deletion_rates, losses)
        spearman_r, spearman_p = stats.spearmanr(deletion_rates, losses)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_r, spearman_p = 0.0, 1.0

    print(f"\nCorrelation Analysis:")

    if deletion_rates.std() < 1e-6:
        print(f"  ⚠ WARNING: Deletion rate is constant ({deletion_rates.mean()*100:.2f}%) across all examples!")
        print(f"  Cannot compute meaningful correlation.")
        print(f"\n  Possible reasons:")
        print(f"    - Model was trained with bypass_gate=True (deletion disabled)")
        print(f"    - Delete gate didn't learn (stuck at initialization)")
        print(f"    - Incorrect deletion threshold setting")
    else:
        print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.4e})")
        print(f"  Spearman r = {spearman_r:.4f}  (p = {spearman_p:.4e})")

        if abs(pearson_r) < 0.1:
            interpretation = "very weak"
        elif abs(pearson_r) < 0.3:
            interpretation = "weak"
        elif abs(pearson_r) < 0.5:
            interpretation = "moderate"
        elif abs(pearson_r) < 0.7:
            interpretation = "strong"
        else:
            interpretation = "very strong"

        direction = "positive" if pearson_r > 0 else "negative"

        print(f"\n  → {interpretation.upper()} {direction} correlation")

        if abs(pearson_r) < 0.3:
            print(f"\n  ✓ This suggests the model is making SMART deletion decisions!")
            print(f"    Examples with higher deletion rates don't have substantially higher losses.")
        else:
            print(f"\n  ⚠ This suggests deletion may be impacting performance.")
            print(f"    Examples with higher deletion rates tend to have {'higher' if pearson_r > 0 else 'lower'} losses.")

    # Binned analysis
    print(f"\nBinned Analysis:")
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    bin_stats = []

    for low, high in bins:
        mask = (deletion_rates >= low) & (deletion_rates < high)
        if mask.sum() > 0:
            bin_losses = losses[mask]
            count = mask.sum()
            mean_loss = bin_losses.mean()
            std_loss = bin_losses.std()
            print(f"  Deletion {low*100:.0f}-{high*100:.0f}%: n={count:4d}  mean_loss={mean_loss:.4f}  std={std_loss:.4f}")
            bin_stats.append({
                'range': f"{low*100:.0f}-{high*100:.0f}%",
                'count': int(count),
                'mean_loss': float(mean_loss),
                'std_loss': float(std_loss)
            })

    # Visualizations
    print(f"\nCreating visualizations...")

    # 1. Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(deletion_rates * 100, losses, alpha=0.3, s=10)
    plt.xlabel('Deletion Rate (%)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Per-Example Deletion Rate vs Loss', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add regression line (only if there's variance in deletion rates)
    if len(deletion_rates) > 2 and deletion_rates.std() > 1e-6:
        try:
            z = np.polyfit(deletion_rates, losses, 1)
            p = np.poly1d(z)
            x_line = np.linspace(deletion_rates.min(), deletion_rates.max(), 100)
            plt.plot(x_line * 100, p(x_line), "r--", alpha=0.8, linewidth=2,
                    label=f'Linear fit (r={pearson_r:.3f})')
            plt.legend()
        except np.linalg.LinAlgError:
            # Skip regression line if fitting fails
            pass

    scatter_path = os.path.join(output_dir, "deletion_vs_loss_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {scatter_path}")

    # 2. Binned bar plot
    if bin_stats:
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(bin_stats))
        means = [b['mean_loss'] for b in bin_stats]
        stds = [b['std_loss'] for b in bin_stats]
        labels = [b['range'] for b in bin_stats]

        plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel('Deletion Rate Range', fontsize=12)
        plt.ylabel('Mean Loss', fontsize=12)
        plt.title('Mean Loss by Deletion Rate Bin', fontsize=14)
        plt.xticks(x_pos, labels)
        plt.grid(True, alpha=0.3, axis='y')

        bar_path = os.path.join(output_dir, "deletion_bins_mean_loss.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {bar_path}")

    # 3. Histogram of deletion rates
    plt.figure(figsize=(10, 6))
    plt.hist(deletion_rates * 100, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Deletion Rate (%)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Deletion Rates', fontsize=14)
    plt.axvline(deletion_rates.mean() * 100, color='r', linestyle='--',
               linewidth=2, label=f'Mean = {deletion_rates.mean()*100:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    hist_path = os.path.join(output_dir, "deletion_rate_histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {hist_path}")

    # 4. Joint distribution (2D histogram)
    plt.figure(figsize=(10, 8))
    plt.hist2d(deletion_rates * 100, losses, bins=50, cmap='Blues')
    plt.colorbar(label='Count')
    plt.xlabel('Deletion Rate (%)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Joint Distribution: Deletion Rate vs Loss', fontsize=14)

    joint_path = os.path.join(output_dir, "deletion_vs_loss_2dhist.png")
    plt.savefig(joint_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {joint_path}")

    # Save results to JSON
    results_dict = {
        'num_examples': len(deletion_rates),
        'deletion_rate_stats': {
            'mean': float(deletion_rates.mean()),
            'median': float(np.median(deletion_rates)),
            'std': float(deletion_rates.std()),
            'min': float(deletion_rates.min()),
            'max': float(deletion_rates.max()),
        },
        'loss_stats': {
            'mean': float(losses.mean()),
            'median': float(np.median(losses)),
            'std': float(losses.std()),
            'min': float(losses.min()),
            'max': float(losses.max()),
        },
        'correlation': {
            'pearson_r': float(pearson_r) if len(deletion_rates) > 2 else None,
            'pearson_p': float(pearson_p) if len(deletion_rates) > 2 else None,
            'spearman_r': float(spearman_r) if len(deletion_rates) > 2 else None,
            'spearman_p': float(spearman_p) if len(deletion_rates) > 2 else None,
        },
        'binned_analysis': bin_stats,
    }

    json_path = os.path.join(output_dir, "analysis_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze deletion rate vs loss correlation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained MrBERT model checkpoint")
    parser.add_argument("--task", type=str, default="sequence_classification",
                       choices=["mlm", "sequence_classification", "token_classification", "question_answering"],
                       help="Task type")
    parser.add_argument("--dataset_name", type=str, default="glue",
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="sst2",
                       help="Dataset config/subset")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to analyze")
    parser.add_argument("--output_dir", type=str, default="./deletion_analysis",
                       help="Output directory for results")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split to analyze")

    args_dict = vars(parser.parse_args())
    args = AnalysisArgs(**args_dict)

    # Load model and tokenizer
    model, tokenizer = load_model(args)

    # Prepare dataset
    eval_dataset = prepare_dataset(args, tokenizer)

    # Create dataloader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DefaultDataCollator(),
    )

    # Compute per-example metrics
    results = compute_per_example_metrics(
        model, dataloader, model.config,
        pad_token_id=tokenizer.pad_token_id,
        device=args.device
    )

    # Analyze and visualize
    analyze_correlation(results, args.output_dir)


if __name__ == "__main__":
    main()
