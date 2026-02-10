#!/usr/bin/env python3
"""
Evaluation script for MrBERT model.

This script evaluates MrBERT on:
1. MLM Perplexity - How well does the model predict masked tokens?
2. Deletion Rate - What fraction of tokens are being deleted?
3. Token Analysis - Which types of tokens are being deleted?
4. Comparison with BERT - How does MrBERT compare to baseline BERT?

Usage:
    python eval_mrbert.py --model_path ./mrbert_checkpoints/final

For evaluation on a fresh model (no training):
    python eval_mrbert.py --from_pretrained bert-base-uncased
"""

import argparse
import math
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from tqdm import tqdm

from configuration_mrbert import MrBertConfig
from modeling_mrbert import MrBertForMaskedLM, MrBertModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MrBERT model")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained MrBERT checkpoint",
    )
    parser.add_argument(
        "--from_pretrained",
        type=str,
        default="bert-base-uncased",
        help="Create MrBERT from pretrained BERT (for testing without training)",
    )
    parser.add_argument(
        "--delete_gate_layer",
        type=int,
        default=3,  # MrT5 default
        help="Delete gate layer (only used with --from_pretrained)",
    )
    parser.add_argument(
        "--deletion_type",
        type=str,
        default="scaled_sigmoid",
        help="Deletion type (only used with --from_pretrained)",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name (use 'local_mc4' for preprocessed mC4)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset config",
    )
    parser.add_argument(
        "--local_mc4_dir",
        type=str,
        default="mrt5/lm_datasets",
        help="Directory for local mC4 data (used when dataset_name='local_mc4')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to evaluate (-1 for all)",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="MLM masking probability",
    )
    parser.add_argument(
        "--hard_delete",
        action="store_true",
        help="Use hard deletion during evaluation",
    )
    parser.add_argument(
        "--deletion_threshold",
        type=float,
        default=-5.0,
        help="Threshold for hard deletion",
    )
    parser.add_argument(
        "--compare_bert",
        action="store_true",
        help="Compare with baseline BERT",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="Number of example deletions to show",
    )
    
    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def load_model(args, tokenizer):
    """Load MrBERT model from checkpoint or create from pretrained."""
    if args.model_path:
        print(f"Loading MrBERT from {args.model_path}")
        model = MrBertForMaskedLM.from_pretrained(args.model_path)
    else:
        print(f"Creating MrBERT from {args.from_pretrained}")
        config = MrBertConfig.from_pretrained(
            args.from_pretrained,
            deletion_type=args.deletion_type,
            delete_gate_layer=args.delete_gate_layer,
            sigmoid_mask_scale=-30.0,  # MrT5 default
            deletion_threshold=-15.0,  # MrT5 default
        )
        model = MrBertForMaskedLM(config)
    
    return model


def prepare_dataset(args, tokenizer):
    """Load and prepare evaluation dataset."""
    
    # Handle local mC4 dataset
    if args.dataset_name == "local_mc4":
        print(f"Loading LOCAL mC4 dataset from: {args.local_mc4_dir} ({args.split})")
        from mc4_dataset import load_mc4_dataset
        
        dataset = load_mc4_dataset(
            split=args.split,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            max_samples=args.max_samples if args.max_samples > 0 else None,
            streaming=False,
            data_dir=args.local_mc4_dir,
        )
        return dataset
    
    # Standard HuggingFace dataset
    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config} ({args.split})")
    
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    
    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
    
    def tokenize_function(examples):
        texts = [t for t in examples[text_column] if t and len(t.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    if args.max_samples > 0:
        tokenized_dataset = tokenized_dataset.select(range(min(args.max_samples, len(tokenized_dataset))))
    
    return tokenized_dataset


def compute_perplexity(model, dataloader, device, hard_delete=False, deletion_threshold=None):
    """Compute MLM perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Count masked tokens
            masked_tokens = (batch["labels"] != -100).sum().item()
            if masked_tokens == 0:
                continue
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                hard_delete=hard_delete,
                deletion_threshold=deletion_threshold,
            )
            
            total_loss += outputs.loss.item() * masked_tokens
            total_tokens += masked_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return perplexity, avg_loss


def analyze_deletions(model, dataloader, tokenizer, device, num_examples=5):
    """Analyze which tokens are being deleted."""
    model.eval()
    
    deletion_stats = {
        "total_tokens": 0,
        "deleted_tokens": 0,
        "token_type_counts": defaultdict(lambda: {"total": 0, "deleted": 0}),
    }
    
    examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing deletions")):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get model outputs with delete gate info
            outputs = model.bert(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            if outputs.delete_gate_mask is None:
                continue
            
            # Process each sequence in batch
            for seq_idx in range(batch["input_ids"].size(0)):
                input_ids = batch["input_ids"][seq_idx]
                attention_mask = batch["attention_mask"][seq_idx]
                gate_values = outputs.delete_gate_mask[seq_idx].squeeze(-1)
                
                # Get tokens
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                
                # Analyze each token
                for pos, (token, gate_val, mask) in enumerate(zip(tokens, gate_values, attention_mask)):
                    if mask == 0:  # Skip padding
                        continue
                    
                    deletion_stats["total_tokens"] += 1
                    
                    # Determine token type
                    if token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]:
                        token_type = "special"
                    elif token.startswith("##"):
                        token_type = "subword"
                    elif token.isalpha():
                        token_type = "word"
                    elif token.isdigit():
                        token_type = "number"
                    elif token in ".,;:!?'\"()-":
                        token_type = "punctuation"
                    else:
                        token_type = "other"
                    
                    deletion_stats["token_type_counts"][token_type]["total"] += 1
                    
                    # Check if deleted (gate value < -5 is strong deletion signal)
                    is_deleted = gate_val.item() < -5.0
                    if is_deleted:
                        deletion_stats["deleted_tokens"] += 1
                        deletion_stats["token_type_counts"][token_type]["deleted"] += 1
                
                # Save example
                if len(examples) < num_examples:
                    example = {
                        "tokens": tokens,
                        "gate_values": gate_values.cpu().numpy(),
                        "attention_mask": attention_mask.cpu().numpy(),
                    }
                    examples.append(example)
    
    return deletion_stats, examples


def print_deletion_examples(examples, tokenizer):
    """Print examples showing which tokens are deleted."""
    print("\n" + "=" * 60)
    print("DELETION EXAMPLES")
    print("=" * 60)
    
    for i, example in enumerate(examples):
        print(f"\nExample {i + 1}:")
        print("-" * 40)
        
        # Color-code tokens: green = keep, red = delete
        kept_tokens = []
        deleted_tokens = []
        
        for token, gate_val, mask in zip(
            example["tokens"], 
            example["gate_values"], 
            example["attention_mask"]
        ):
            if mask == 0 or token in ["[PAD]"]:
                continue
            
            if gate_val < -5.0:
                deleted_tokens.append(token)
            else:
                kept_tokens.append(token)
        
        # Print original
        all_tokens = [t for t, m in zip(example["tokens"], example["attention_mask"]) 
                      if m == 1 and t != "[PAD]"]
        print(f"Original ({len(all_tokens)} tokens):")
        print("  " + " ".join(all_tokens[:50]) + ("..." if len(all_tokens) > 50 else ""))
        
        # Print with deletion info
        print(f"\nKept ({len(kept_tokens)} tokens):")
        print("  " + " ".join(kept_tokens[:50]) + ("..." if len(kept_tokens) > 50 else ""))
        
        print(f"\nDeleted ({len(deleted_tokens)} tokens):")
        print("  " + " ".join(deleted_tokens[:30]) + ("..." if len(deleted_tokens) > 30 else ""))
        
        deletion_rate = len(deleted_tokens) / len(all_tokens) * 100 if all_tokens else 0
        print(f"\nDeletion rate: {deletion_rate:.1f}%")


def print_deletion_stats(stats):
    """Print deletion statistics."""
    print("\n" + "=" * 60)
    print("DELETION STATISTICS")
    print("=" * 60)
    
    total = stats["total_tokens"]
    deleted = stats["deleted_tokens"]
    rate = deleted / total * 100 if total > 0 else 0
    
    print(f"\nOverall:")
    print(f"  Total tokens: {total:,}")
    print(f"  Deleted tokens: {deleted:,}")
    print(f"  Deletion rate: {rate:.2f}%")
    
    print(f"\nBy token type:")
    print(f"  {'Type':<15} {'Total':>10} {'Deleted':>10} {'Rate':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    
    for token_type, counts in sorted(stats["token_type_counts"].items()):
        type_total = counts["total"]
        type_deleted = counts["deleted"]
        type_rate = type_deleted / type_total * 100 if type_total > 0 else 0
        print(f"  {token_type:<15} {type_total:>10,} {type_deleted:>10,} {type_rate:>9.1f}%")


def compare_with_bert(args, tokenizer, dataloader, device):
    """Compare MrBERT with baseline BERT."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE BERT")
    print("=" * 60)
    
    # Load baseline BERT
    print("\nLoading baseline BERT...")
    bert_model = BertForMaskedLM.from_pretrained(args.from_pretrained)
    bert_model.to(device)
    bert_model.eval()
    
    # Load MrBERT
    print("Loading MrBERT...")
    mrbert_model = load_model(args, tokenizer)
    mrbert_model.to(device)
    mrbert_model.eval()
    
    # Compute perplexities
    print("\nEvaluating BERT...")
    
    # For BERT, we need to handle the DataCollator output format
    bert_total_loss = 0
    bert_total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="BERT evaluation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            masked_tokens = (batch["labels"] != -100).sum().item()
            if masked_tokens == 0:
                continue
            
            outputs = bert_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            bert_total_loss += outputs.loss.item() * masked_tokens
            bert_total_tokens += masked_tokens
    
    bert_ppl = math.exp(bert_total_loss / bert_total_tokens) if bert_total_tokens > 0 else float('inf')
    
    print("Evaluating MrBERT (soft deletion)...")
    mrbert_ppl_soft, _ = compute_perplexity(mrbert_model, dataloader, device, hard_delete=False)
    
    if args.hard_delete:
        print("Evaluating MrBERT (hard deletion)...")
        mrbert_ppl_hard, _ = compute_perplexity(
            mrbert_model, dataloader, device, 
            hard_delete=True, 
            deletion_threshold=args.deletion_threshold
        )
    
    # Print comparison
    print("\n" + "-" * 40)
    print(f"{'Model':<25} {'Perplexity':>15}")
    print("-" * 40)
    print(f"{'BERT (baseline)':<25} {bert_ppl:>15.2f}")
    print(f"{'MrBERT (soft delete)':<25} {mrbert_ppl_soft:>15.2f}")
    if args.hard_delete:
        print(f"{'MrBERT (hard delete)':<25} {mrbert_ppl_hard:>15.2f}")
    print("-" * 40)
    
    return {
        "bert_ppl": bert_ppl,
        "mrbert_soft_ppl": mrbert_ppl_soft,
        "mrbert_hard_ppl": mrbert_ppl_hard if args.hard_delete else None,
    }


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("MrBERT Evaluation")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset_name}/{args.dataset_config}")
    print(f"Split: {args.split}")
    print()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        args.model_path if args.model_path else args.from_pretrained
    )
    
    # Load model
    model = load_model(args, tokenizer)
    model.to(args.device)
    model.eval()
    
    # Prepare dataset
    dataset = prepare_dataset(args, tokenizer)
    
    # For local_mc4, the dataset already has MLM labels, use DefaultDataCollator
    # For HuggingFace datasets, use DataCollatorForLanguageModeling to create masks
    if args.dataset_name == "local_mc4":
        from transformers import DefaultDataCollator
        data_collator = DefaultDataCollator()
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=args.mlm_probability,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    
    # Evaluation 1: Perplexity
    print("\n" + "=" * 60)
    print("MLM PERPLEXITY")
    print("=" * 60)
    
    print("\nSoft deletion:")
    ppl_soft, loss_soft = compute_perplexity(model, dataloader, args.device, hard_delete=False)
    print(f"  Perplexity: {ppl_soft:.2f}")
    print(f"  Loss: {loss_soft:.4f}")
    
    if args.hard_delete:
        print("\nHard deletion:")
        ppl_hard, loss_hard = compute_perplexity(
            model, dataloader, args.device, 
            hard_delete=True, 
            deletion_threshold=args.deletion_threshold
        )
        print(f"  Perplexity: {ppl_hard:.2f}")
        print(f"  Loss: {loss_hard:.4f}")
    
    # Evaluation 2: Deletion Analysis
    deletion_stats, examples = analyze_deletions(
        model, dataloader, tokenizer, args.device, 
        num_examples=args.show_examples
    )
    
    print_deletion_stats(deletion_stats)
    
    if args.show_examples > 0:
        print_deletion_examples(examples, tokenizer)
    
    # Evaluation 3: Comparison with BERT
    if args.compare_bert:
        compare_with_bert(args, tokenizer, dataloader, args.device)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
