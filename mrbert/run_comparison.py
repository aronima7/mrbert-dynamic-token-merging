#!/usr/bin/env python3
"""
Comparison script for BERT vs MrBERT.

This script runs both baseline BERT and MrBERT on the same task and
compares their performance.

Usage:
    # Quick comparison (no training, just evaluate)
    python run_comparison.py --task mlm

    # Full comparison with training
    python run_comparison.py --task mlm --train --num_epochs 10

    # Classification comparison
    python run_comparison.py --task sequence_classification --dataset_name glue --dataset_config sst2 --train
"""

import argparse
import math
import os
import json
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    DefaultDataCollator,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

from mrbert.models.configuration_mrbert import MrBertConfig
from mrbert.models.modeling_mrbert import MrBertForMaskedLM, MrBertForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Compare BERT vs MrBERT")
    
    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="mlm",
        choices=["mlm", "sequence_classification"],
        help="Task to evaluate on",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model",
    )
    parser.add_argument(
        "--delete_gate_layer",
        type=int,
        default=2,
        help="Which encoder layer to place the delete gate",
    )
    parser.add_argument(
        "--deletion_type",
        type=str,
        default="scaled_sigmoid",
        help="Type of delete gate",
    )
    parser.add_argument(
        "--sigmoid_mask_scale",
        type=float,
        default=-10.0,
        help="Scale for sigmoid mask",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name from HuggingFace datasets",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of masking tokens for MLM",
    )
    
    # Training arguments
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune the models before evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of steps (-1 for full training)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    
    # MrBERT specific
    parser.add_argument(
        "--target_deletion_rate",
        type=float,
        default=0.3,
        help="Target deletion rate for MrBERT",
    )
    parser.add_argument(
        "--deletion_loss_weight",
        type=float,
        default=0.01,  # Small value to avoid aggressive deletion
        help="Weight for deletion loss (keep small to avoid 100% deletion)",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=500,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=5000,
        help="Number of samples for training (for quick experiments)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use",
    )
    
    return parser.parse_args()


def load_mlm_dataset(args, tokenizer):
    """Load and preprocess MLM dataset."""
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )
    
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    return tokenized_dataset


def load_classification_dataset(args, tokenizer):
    """Load and preprocess classification dataset."""
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    # Get text column
    if args.dataset_name == "glue":
        if args.dataset_config in ["sst2", "cola"]:
            text_col = "sentence"
        else:
            text_col = "sentence"
    else:
        text_col = "text"
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset


def evaluate_mlm(model, eval_dataloader, device, model_name="Model"):
    """Evaluate MLM and return metrics."""
    model.eval()
    total_loss = 0
    total_samples = 0
    deletion_rates = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating {model_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["input_ids"].shape[0]
            total_samples += batch["input_ids"].shape[0]
            
            # Get deletion rate for MrBERT
            if hasattr(outputs, "delete_gate_mask") and outputs.delete_gate_mask is not None:
                mask = outputs.delete_gate_mask.squeeze(-1)
                # Tokens with mask < -5 are "deleted"
                deletion_rate = (mask < -5.0).float().mean().item()
                deletion_rates.append(deletion_rate)
    
    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    
    metrics = {"loss": avg_loss, "perplexity": perplexity}
    if deletion_rates:
        metrics["deletion_rate"] = sum(deletion_rates) / len(deletion_rates)
    
    return metrics


def evaluate_classification(model, eval_dataloader, device, model_name="Model"):
    """Evaluate classification accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    deletion_rates = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating {model_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            predictions = outputs.logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.shape[0]
            total_loss += outputs.loss.item() * labels.shape[0]
            
            # Get deletion rate for MrBERT
            if hasattr(outputs, "delete_gate_mask") and outputs.delete_gate_mask is not None:
                mask = outputs.delete_gate_mask.squeeze(-1)
                deletion_rate = (mask < -5.0).float().mean().item()
                deletion_rates.append(deletion_rate)
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    
    metrics = {"loss": avg_loss, "accuracy": accuracy}
    if deletion_rates:
        metrics["deletion_rate"] = sum(deletion_rates) / len(deletion_rates)
    
    return metrics


def train_model(model, train_dataloader, args, is_mrbert=False):
    """Train a model."""
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    num_training_steps = len(train_dataloader) * args.num_epochs
    if args.max_steps > 0:
        num_training_steps = min(num_training_steps, args.max_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_steps = 0
        
        model_name = "MrBERT" if is_mrbert else "BERT"
        progress_bar = tqdm(train_dataloader, desc=f"Training {model_name} Epoch {epoch + 1}")
        
        for batch in progress_bar:
            batch = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Add deletion loss for MrBERT
            if is_mrbert and hasattr(outputs, "delete_gate_logits") and outputs.delete_gate_logits is not None:
                deletion_probs = torch.sigmoid(outputs.delete_gate_logits).mean()
                deletion_loss = (deletion_probs - args.target_deletion_rate) ** 2
                loss = loss + args.deletion_loss_weight * deletion_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    
    return model


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("   BERT vs MrBERT COMPARISON")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset_name}/{args.dataset_config}")
    print(f"Device: {args.device}")
    print(f"Training: {args.train}")
    if args.train:
        print(f"  - Epochs: {args.num_epochs}")
        print(f"  - Learning rate: {args.learning_rate}")
        print(f"  - Target deletion rate: {args.target_deletion_rate}")
    print("=" * 70 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Load dataset based on task
    if args.task == "mlm":
        dataset = load_mlm_dataset(args, tokenizer)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=args.mlm_probability,
        )
        evaluate_fn = evaluate_mlm
    elif args.task == "sequence_classification":
        dataset = load_classification_dataset(args, tokenizer)
        data_collator = DefaultDataCollator()
        evaluate_fn = evaluate_classification
    
    # Limit dataset size for quick experiments
    train_dataset = dataset["train"]
    if args.num_train_samples > 0 and len(train_dataset) > args.num_train_samples:
        train_dataset = train_dataset.select(range(args.num_train_samples))
    
    if "validation" in dataset:
        eval_dataset = dataset["validation"]
    elif "test" in dataset:
        eval_dataset = dataset["test"]
    else:
        eval_dataset = dataset["train"]
    
    if args.num_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.num_eval_samples)))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    
    results = {}
    
    # =========================================================================
    # STEP 1: Baseline BERT
    # =========================================================================
    print("\n" + "=" * 70)
    print("   STEP 1: BASELINE BERT")
    print("=" * 70)
    
    if args.task == "mlm":
        bert_model = BertForMaskedLM.from_pretrained(args.model_name)
    elif args.task == "sequence_classification":
        temp_dataset = load_dataset(args.dataset_name, args.dataset_config)
        num_labels = temp_dataset["train"].features["label"].num_classes
        bert_model = BertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels
        )
    
    bert_model = bert_model.to(args.device)
    print(f"BERT parameters: {sum(p.numel() for p in bert_model.parameters()):,}")
    
    if args.train:
        print("\nTraining baseline BERT...")
        bert_model = train_model(bert_model, train_dataloader, args, is_mrbert=False)
    
    print("\nEvaluating baseline BERT...")
    bert_metrics = evaluate_fn(bert_model, eval_dataloader, args.device, "BERT")
    results["bert"] = bert_metrics
    
    print("\nBERT Results:")
    for key, value in bert_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Free memory
    del bert_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =========================================================================
    # STEP 2: MrBERT
    # =========================================================================
    print("\n" + "=" * 70)
    print("   STEP 2: MrBERT (with Delete Gate)")
    print("=" * 70)
    
    # Create MrBERT config
    mrbert_config = MrBertConfig.from_pretrained(
        args.model_name,
        delete_gate_layer=args.delete_gate_layer,
        deletion_type=args.deletion_type,
        sigmoid_mask_scale=args.sigmoid_mask_scale,
    )
    
    if args.task == "mlm":
        mrbert_model = MrBertForMaskedLM.from_pretrained(args.model_name, config=mrbert_config)
    elif args.task == "sequence_classification":
        mrbert_config.num_labels = num_labels
        mrbert_model = MrBertForSequenceClassification.from_pretrained(
            args.model_name, config=mrbert_config
        )
    
    mrbert_model = mrbert_model.to(args.device)
    print(f"MrBERT parameters: {sum(p.numel() for p in mrbert_model.parameters()):,}")
    print(f"Delete gate layer: {args.delete_gate_layer}")
    print(f"Deletion type: {args.deletion_type}")
    
    if args.train:
        print("\nTraining MrBERT...")
        mrbert_model = train_model(mrbert_model, train_dataloader, args, is_mrbert=True)
    
    print("\nEvaluating MrBERT...")
    mrbert_metrics = evaluate_fn(mrbert_model, eval_dataloader, args.device, "MrBERT")
    results["mrbert"] = mrbert_metrics
    
    print("\nMrBERT Results:")
    for key, value in mrbert_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("   COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'BERT':<15} {'MrBERT':<15} {'Difference':<15}")
    print("-" * 65)
    
    for key in bert_metrics.keys():
        bert_val = bert_metrics[key]
        mrbert_val = mrbert_metrics.get(key, 0)
        diff = mrbert_val - bert_val
        diff_pct = (diff / bert_val * 100) if bert_val != 0 else 0
        print(f"{key:<20} {bert_val:<15.4f} {mrbert_val:<15.4f} {diff:+.4f} ({diff_pct:+.1f}%)")
    
    # Extra MrBERT metrics
    if "deletion_rate" in mrbert_metrics:
        print(f"\nMrBERT Deletion Rate: {mrbert_metrics['deletion_rate']:.1%}")
    
    # Save results
    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "task": args.task,
        "dataset": f"{args.dataset_name}/{args.dataset_config}",
        "trained": args.train,
        "config": {
            "num_epochs": args.num_epochs if args.train else 0,
            "learning_rate": args.learning_rate,
            "delete_gate_layer": args.delete_gate_layer,
            "deletion_type": args.deletion_type,
            "target_deletion_rate": args.target_deletion_rate,
        },
        "results": results,
    }
    
    results_file = os.path.join(args.output_dir, f"comparison_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
