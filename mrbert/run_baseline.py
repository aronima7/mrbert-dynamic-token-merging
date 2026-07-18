#!/usr/bin/env python3
"""
Baseline BERT evaluation script.

This script evaluates standard BERT (without delete gates) to establish
a baseline for comparison with MrBERT.

Usage:
    # Evaluate baseline BERT on MLM
    python run_baseline.py --task mlm

    # Evaluate on classification (SST-2)
    python run_baseline.py --task sequence_classification --dataset_name glue --dataset_config sst2

    # Fine-tune baseline BERT first, then evaluate
    python run_baseline.py --task mlm --train --num_epochs 3
"""

import argparse
import math
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForQuestionAnswering,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    DefaultDataCollator,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline BERT")
    
    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="mlm",
        choices=["mlm", "sequence_classification", "token_classification", "question_answering"],
        help="Task to evaluate on",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model",
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
        help="Dataset configuration (optional)",
    )
    parser.add_argument(
        "--local_mc4_dir",
        type=str,
        default=None,
        help="Path to local mC4 dataset directory (for local_mc4 dataset)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to use (-1 for all)",
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
    
    # Training arguments (if --train is specified)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune the model before evaluation",
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
        default=500,
        help="Number of warmup steps",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=1000,
        help="Number of samples for evaluation",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_results",
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
    
    # Handle local mC4 dataset
    if args.dataset_name == "local_mc4":
        from mrt5.mc4_dataset import load_mc4_dataset
        
        print(f"Loading LOCAL mC4 dataset from: {args.local_mc4_dir}")
        
        max_samples = args.max_samples if args.max_samples > 0 else None
        train_dataset = load_mc4_dataset(
            split="train",
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            data_dir=args.local_mc4_dir,
            max_samples=max_samples,
            streaming=False
        )
        val_dataset = load_mc4_dataset(
            split="validation",
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            data_dir=args.local_mc4_dir,
            max_samples=args.num_eval_samples if args.num_eval_samples > 0 else max_samples,
            streaming=False
        )
        
        return {"train": train_dataset, "validation": val_dataset}
    
    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config}")
    
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
    
    # Filter empty examples
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Apply max_samples limit if specified
    if args.max_samples > 0:
        if "train" in tokenized_dataset:
            tokenized_dataset["train"] = tokenized_dataset["train"].select(range(min(args.max_samples, len(tokenized_dataset["train"]))))
        if "validation" in tokenized_dataset:
            tokenized_dataset["validation"] = tokenized_dataset["validation"].select(range(min(args.max_samples, len(tokenized_dataset["validation"]))))
    
    return tokenized_dataset


def load_classification_dataset(args, tokenizer):
    """Load and preprocess classification dataset."""
    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config}")
    
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    # Get text columns based on dataset
    if args.dataset_name == "glue":
        if args.dataset_config in ["sst2", "cola"]:
            text_col = "sentence"
        elif args.dataset_config in ["mrpc", "stsb", "rte", "wnli"]:
            text_col = ("sentence1", "sentence2")
        elif args.dataset_config in ["qnli"]:
            text_col = ("question", "sentence")
        elif args.dataset_config in ["qqp"]:
            text_col = ("question1", "question2")
        elif args.dataset_config in ["mnli"]:
            text_col = ("premise", "hypothesis")
        else:
            text_col = "sentence"
    else:
        text_col = "text"
    
    def tokenize_function(examples):
        if isinstance(text_col, tuple):
            return tokenizer(
                examples[text_col[0]],
                examples[text_col[1]],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )
        else:
            return tokenizer(
                examples[text_col],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
    )
    
    # Rename 'label' to 'labels' for HuggingFace compatibility
    if "label" in tokenized_dataset["train"].column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    # Set format to PyTorch tensors
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_dataset


def evaluate_mlm(model, eval_dataloader, device):
    """Evaluate MLM perplexity."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating MLM"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["input_ids"].shape[0]
            total_samples += batch["input_ids"].shape[0]
    
    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    
    return {"loss": avg_loss, "perplexity": perplexity}


def evaluate_classification(model, eval_dataloader, device):
    """Evaluate classification accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating Classification"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            predictions = outputs.logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.shape[0]
            total_loss += outputs.loss.item() * labels.shape[0]
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    
    return {"loss": avg_loss, "accuracy": accuracy}


def train_model(model, train_dataloader, eval_dataloader, args, evaluate_fn):
    """Train the model."""
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
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")
        
        epoch_loss = 0
        epoch_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        for batch in progress_bar:
            batch = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**batch)
            loss = outputs.loss
            
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
        
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Average training loss: {avg_epoch_loss:.4f}")
        
        # Evaluate at end of epoch
        metrics = evaluate_fn(model, eval_dataloader, args.device)
        print(f"Evaluation metrics: {metrics}")
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    
    return model


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BASELINE BERT EVALUATION")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}/{args.dataset_config}")
    print(f"Device: {args.device}")
    print(f"Training: {args.train}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Load model and dataset based on task
    if args.task == "mlm":
        model = BertForMaskedLM.from_pretrained(args.model_name)
        dataset = load_mlm_dataset(args, tokenizer)
        
        # For local_mc4, the dataset already has MLM labels, use DefaultDataCollator
        # For HuggingFace datasets, use DataCollatorForLanguageModeling to create masks
        if args.dataset_name == "local_mc4":
            data_collator = DefaultDataCollator()
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=args.mlm_probability,
            )
        
        evaluate_fn = evaluate_mlm
        
    elif args.task == "sequence_classification":
        # Get number of labels
        if args.dataset_config:
            temp_dataset = load_dataset(args.dataset_name, args.dataset_config)
        else:
            temp_dataset = load_dataset(args.dataset_name)
        
        # Determine number of labels
        if "label" in temp_dataset["train"].features:
            num_labels = temp_dataset["train"].features["label"].num_classes
        else:
            num_labels = 2
        
        model = BertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels
        )
        dataset = load_classification_dataset(args, tokenizer)
        data_collator = DefaultDataCollator()
        evaluate_fn = evaluate_classification
        
    else:
        raise NotImplementedError(f"Task {args.task} not yet implemented for baseline")
    
    model = model.to(args.device)
    
    # Check if this is a local_mc4 dataset (PyTorch Dataset) or HuggingFace DatasetDict
    is_pytorch_dataset = args.dataset_name == "local_mc4"
    
    # Create dataloaders
    if "train" in dataset:
        train_dataset = dataset["train"]
        # Only use .select() for HuggingFace datasets
        if not is_pytorch_dataset and args.num_eval_samples > 0 and len(train_dataset) > args.num_eval_samples:
            train_dataset = train_dataset.select(range(min(len(train_dataset), 10000)))
    
    if "validation" in dataset:
        eval_dataset = dataset["validation"]
    elif "test" in dataset:
        eval_dataset = dataset["test"]
    else:
        eval_dataset = dataset["train"]
    
    # Only use .select() for HuggingFace datasets
    if not is_pytorch_dataset and args.num_eval_samples > 0:
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
    
    # Train if requested
    if args.train:
        print("\n" + "=" * 60)
        print("TRAINING BASELINE BERT")
        print("=" * 60)
        model = train_model(model, train_dataloader, eval_dataloader, args, evaluate_fn)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    metrics = evaluate_fn(model, eval_dataloader, args.device)
    
    print("\nBaseline BERT Results:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    results = {
        "model": "bert-base-uncased",
        "task": args.task,
        "dataset": f"{args.dataset_name}/{args.dataset_config}",
        "trained": args.train,
        "metrics": metrics,
    }
    
    results_file = os.path.join(args.output_dir, f"baseline_{args.task}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return metrics


if __name__ == "__main__":
    main()
