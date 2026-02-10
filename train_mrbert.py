#!/usr/bin/env python3
"""
Fine-tuning script for MrBERT model.

This script supports fine-tuning MrBERT for multiple tasks:
- Masked Language Modeling (MLM)
- Sequence Classification (e.g., sentiment analysis, GLUE tasks)
- Token Classification (e.g., NER)
- Question Answering (e.g., SQuAD)

Usage:
    # MLM (default)
    python train_mrbert.py --task mlm --dataset_name wikitext --dataset_config wikitext-2-raw-v1

    # Sequence Classification (e.g., SST-2)
    python train_mrbert.py --task sequence_classification --dataset_name glue --dataset_config sst2

    # Token Classification (e.g., CoNLL-2003 NER)
    python train_mrbert.py --task token_classification --dataset_name conll2003

    # Question Answering (e.g., SQuAD)
    python train_mrbert.py --task question_answering --dataset_name squad

For a quick test run:
    python train_mrbert.py --max_steps 100 --logging_steps 10
"""

import argparse
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    DefaultDataCollator,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

from configuration_mrbert import MrBertConfig
from modeling_mrbert import (
    MrBertForMaskedLM,
    MrBertForSequenceClassification,
    MrBertForTokenClassification,
    MrBertForQuestionAnswering,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MrBERT for various tasks")
    
    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="mlm",
        choices=["mlm", "sequence_classification", "token_classification", "question_answering"],
        help="Task to fine-tune on",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model to start from",
    )
    parser.add_argument(
        "--delete_gate_layer",
        type=int,
        default=2,
        help="Which encoder layer to place the delete gate (0-indexed)",
    )
    parser.add_argument(
        "--deletion_type",
        type=str,
        default="scaled_sigmoid",
        choices=["scaled_sigmoid", "log_sigmoid", "random", "fixed"],
        help="Type of delete gate",
    )
    parser.add_argument(
        "--sigmoid_mask_scale",
        type=float,
        default=-10.0,
        help="Scale for sigmoid mask (more negative = stronger deletion signal)",
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
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--delete_gate_lr",
        type=float,
        default=1e-4,
        help="Learning rate for delete gate (can be higher than main model)",
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
        help="Maximum number of training steps (-1 for full training)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    
    # Delete gate loss arguments
    parser.add_argument(
        "--deletion_loss_weight",
        type=float,
        default=0.01,  # Reduced from 0.1 - too high causes 100% deletion
        help="Weight for deletion regularization loss (MrT5 paper uses small values)",
    )
    parser.add_argument(
        "--target_deletion_rate",
        type=float,
        default=0.3,
        help="Target fraction of tokens to delete (for regularization)",
    )
    
    # Other arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mrbert_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def compute_deletion_loss(delete_gate_output, target_deletion_rate, sigmoid_mask_scale):
    """
    Compute auxiliary loss to encourage the model to delete tokens.
    """
    if delete_gate_output is None:
        return torch.tensor(0.0)
    
    # Convert gate values to deletion probabilities
    deletion_probs = -delete_gate_output.squeeze(-1) / sigmoid_mask_scale
    deletion_probs = deletion_probs.clamp(0, 1)
    
    # Compute actual deletion rate
    actual_deletion_rate = deletion_probs.mean()
    
    # L2 loss to push toward target rate
    deletion_loss = (actual_deletion_rate - target_deletion_rate) ** 2
    
    return deletion_loss


# =============================================================================
# Dataset Preparation Functions
# =============================================================================

def prepare_mlm_dataset(args, tokenizer):
    """Prepare dataset for Masked Language Modeling."""
    print(f"Loading MLM dataset: {args.dataset_name}/{args.dataset_config}")
    
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    text_column = "text" if "text" in dataset["train"].column_names else dataset["train"].column_names[0]
    
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
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    
    return tokenized_dataset, data_collator


def prepare_sequence_classification_dataset(args, tokenizer):
    """Prepare dataset for Sequence Classification (e.g., GLUE tasks)."""
    print(f"Loading classification dataset: {args.dataset_name}/{args.dataset_config}")
    
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    # Determine text columns based on dataset
    if args.dataset_name == "glue":
        if args.dataset_config in ["sst2", "cola"]:
            text_columns = ["sentence"]
        elif args.dataset_config in ["mrpc", "stsb", "rte", "wnli"]:
            text_columns = ["sentence1", "sentence2"]
        elif args.dataset_config in ["mnli", "qnli"]:
            text_columns = ["premise", "hypothesis"] if args.dataset_config == "mnli" else ["question", "sentence"]
        elif args.dataset_config == "qqp":
            text_columns = ["question1", "question2"]
        else:
            text_columns = ["sentence"]
    else:
        # Default: look for common column names
        cols = dataset["train"].column_names
        if "text" in cols:
            text_columns = ["text"]
        elif "sentence" in cols:
            text_columns = ["sentence"]
        else:
            text_columns = [cols[0]]
    
    label_column = "label"
    
    def tokenize_function(examples):
        if len(text_columns) == 1:
            return tokenizer(
                examples[text_columns[0]],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )
        else:
            return tokenizer(
                examples[text_columns[0]],
                examples[text_columns[1]],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
    )
    
    # Rename label column if needed
    if label_column in tokenized_dataset["train"].column_names:
        tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
    
    # Get number of labels
    num_labels = len(set(dataset["train"][label_column]))
    
    data_collator = DefaultDataCollator()
    
    return tokenized_dataset, data_collator, num_labels


def prepare_token_classification_dataset(args, tokenizer):
    """Prepare dataset for Token Classification (e.g., NER)."""
    print(f"Loading token classification dataset: {args.dataset_name}")
    
    dataset = load_dataset(args.dataset_name)
    
    # Get label list
    if "ner_tags" in dataset["train"].features:
        label_list = dataset["train"].features["ner_tags"].feature.names
    elif "pos_tags" in dataset["train"].features:
        label_list = dataset["train"].features["pos_tags"].feature.names
    else:
        # Default for CoNLL-2003
        label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    
    num_labels = len(label_list)
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            is_split_into_words=True,
        )
        
        labels = []
        label_key = "ner_tags" if "ner_tags" in examples else "pos_tags"
        
        for i, label in enumerate(examples[label_key]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx] if word_idx < len(label) else -100)
                else:
                    label_ids.append(-100)  # Subword tokens get -100
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing",
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    return tokenized_dataset, data_collator, num_labels


def prepare_question_answering_dataset(args, tokenizer):
    """Prepare dataset for Question Answering (e.g., SQuAD)."""
    print(f"Loading QA dataset: {args.dataset_name}")
    
    dataset = load_dataset(args.dataset_name)
    
    def prepare_train_features(examples):
        # Tokenize questions and contexts
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=args.max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        
        return tokenized_examples
    
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    
    data_collator = DefaultDataCollator()
    
    return tokenized_dataset, data_collator


# =============================================================================
# Model Creation
# =============================================================================

def create_model(args, tokenizer, num_labels=None):
    """Create the appropriate MrBERT model for the task."""
    
    config = MrBertConfig.from_pretrained(
        args.model_name,
        deletion_type=args.deletion_type,
        delete_gate_layer=args.delete_gate_layer,
        sigmoid_mask_scale=args.sigmoid_mask_scale,
        use_gumbel_noise=True,
    )
    
    if args.task == "mlm":
        model = MrBertForMaskedLM(config)
    elif args.task == "sequence_classification":
        config.num_labels = num_labels
        model = MrBertForSequenceClassification(config)
    elif args.task == "token_classification":
        config.num_labels = num_labels
        model = MrBertForTokenClassification(config)
    elif args.task == "question_answering":
        config.num_labels = 2  # start and end
        model = MrBertForQuestionAnswering(config)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    return model


# =============================================================================
# Training Loop
# =============================================================================

def train(args, model, train_dataloader, eval_dataloader, tokenizer):
    """Training loop."""
    
    # Separate parameters for different learning rates
    delete_gate_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if "delete_gate" in name:
            delete_gate_params.append(param)
        else:
            other_params.append(param)
    
    print(f"Delete gate parameters: {sum(p.numel() for p in delete_gate_params):,}")
    print(f"Other parameters: {sum(p.numel() for p in other_params):,}")
    
    # Create optimizer
    optimizer = AdamW([
        {"params": other_params, "lr": args.learning_rate},
        {"params": delete_gate_params, "lr": args.delete_gate_lr},
    ])
    
    # Calculate total steps
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = len(train_dataloader) * args.num_epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\nTotal training steps: {total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print()
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    total_task_loss = 0
    total_deletion_loss = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Task loss
            task_loss = outputs.loss
            
            # Deletion regularization loss
            delete_gate_output = getattr(outputs, 'delete_gate_output', None)
            deletion_loss = compute_deletion_loss(
                delete_gate_output,
                args.target_deletion_rate,
                args.sigmoid_mask_scale,
            )
            
            # Combined loss
            loss = task_loss + args.deletion_loss_weight * deletion_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update tracking
            global_step += 1
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_deletion_loss += deletion_loss.item() if isinstance(deletion_loss, torch.Tensor) else deletion_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "task": f"{task_loss.item():.4f}",
            })
            
            # Logging
            if global_step % args.logging_steps == 0:
                avg_loss = total_loss / args.logging_steps
                avg_task_loss = total_task_loss / args.logging_steps
                avg_deletion_loss = total_deletion_loss / args.logging_steps
                
                # Calculate actual deletion rate
                if delete_gate_output is not None:
                    deletion_probs = -delete_gate_output.squeeze(-1) / args.sigmoid_mask_scale
                    deletion_probs = deletion_probs.clamp(0, 1)
                    actual_del_rate = deletion_probs.mean().item()
                else:
                    actual_del_rate = 0.0
                
                print(f"\nStep {global_step}:")
                print(f"  Loss: {avg_loss:.4f} (Task: {avg_task_loss:.4f}, Del: {avg_deletion_loss:.4f})")
                print(f"  Deletion rate: {actual_del_rate:.2%} (target: {args.target_deletion_rate:.2%})")
                print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
                
                total_loss = 0
                total_task_loss = 0
                total_deletion_loss = 0
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_path = f"{args.output_dir}/checkpoint-{global_step}"
                print(f"\nSaving checkpoint to {save_path}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            
            # Check max steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    
    # Save final model
    final_path = f"{args.output_dir}/final"
    print(f"\nSaving final model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    return model


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 60)
    print(f"MrBERT Fine-tuning - Task: {args.task.upper()}")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Delete gate layer: {args.delete_gate_layer}")
    print(f"Deletion type: {args.deletion_type}")
    print(f"Target deletion rate: {args.target_deletion_rate}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Prepare dataset based on task
    num_labels = None
    
    if args.task == "mlm":
        tokenized_dataset, data_collator = prepare_mlm_dataset(args, tokenizer)
    elif args.task == "sequence_classification":
        tokenized_dataset, data_collator, num_labels = prepare_sequence_classification_dataset(args, tokenizer)
    elif args.task == "token_classification":
        tokenized_dataset, data_collator, num_labels = prepare_token_classification_dataset(args, tokenizer)
    elif args.task == "question_answering":
        tokenized_dataset, data_collator = prepare_question_answering_dataset(args, tokenizer)
    
    # Create model
    print("Creating MrBERT model...")
    model = create_model(args, tokenizer, num_labels)
    model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    eval_dataloader = None
    if "validation" in tokenized_dataset:
        eval_dataloader = DataLoader(
            tokenized_dataset["validation"],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
    
    # Train
    model = train(args, model, train_dataloader, eval_dataloader, tokenizer)
    
    print("\nTraining complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
