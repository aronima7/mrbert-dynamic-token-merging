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
        default=3,  # MrT5 default
        help="Which encoder layer to place the delete gate (0-indexed). MrT5 uses layer 3.",
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
        default=-30.0,  # MrT5 default
        help="Scale for sigmoid mask (more negative = stronger deletion signal)",
    )
    parser.add_argument(
        "--deletion_threshold",
        type=float,
        default=-15.0,  # MrT5 default (sigmoid_mask_scale / 2)
        help="Threshold for counting a token as deleted (gate < threshold)",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name from HuggingFace datasets (or 'local_mc4' for local mC4)",
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
        default="mrt5/lm_datasets",
        help="Directory containing local mC4 preprocessed files (used when dataset_name='local_mc4')",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of masking tokens for MLM",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming for large datasets (memory efficient)",
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
    
    # Delete gate loss arguments (PI-controller from MrT5 paper)
    parser.add_argument(
        "--deletion_loss_weight",
        type=float,
        default=0.0,  # Starting α_0 (PI-controller will adjust this)
        help="Initial weight for deletion regularization loss (α_0 in MrT5 paper)",
    )
    parser.add_argument(
        "--target_deletion_rate",
        type=float,
        default=0.4,  # δ in MrT5 paper
        help="Target fraction of tokens to delete (δ in MrT5 paper)",
    )
    parser.add_argument(
        "--controller_p",
        type=float,
        default=0.5,  # k_p in MrT5 paper
        help="Proportional gain for PI-controller (k_p in MrT5 paper)",
    )
    parser.add_argument(
        "--controller_i",
        type=float,
        default=5e-5,  # k_i in MrT5 paper
        help="Integral gain for PI-controller (k_i in MrT5 paper)",
    )
    parser.add_argument(
        "--regularizer_delay",
        type=int,
        default=0,
        help="Number of steps before applying delete gate regularizer",
    )
    parser.add_argument(
        "--use_pi_controller",
        action="store_true",
        default=True,
        help="Use PI-controller for deletion rate (recommended by MrT5 paper)",
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


class PIController:
    """
    PI-controller for dynamically adjusting deletion loss coefficient.
    
    Adapted from MrT5 trainer.py (lines 262-266):
    Uses exponential moving average for proportional term.
    
    α_t = p_acc + i_acc
    where:
      p_acc = 0.9 * p_acc + 0.1 * k_p * error
      i_acc = i_acc + k_i * error
      error = target_rate - actual_rate
    """
    
    def __init__(self, target_rate: float, kp: float = 0.5, ki: float = 1e-5, alpha_0: float = 0.0):
        self.target_rate = target_rate
        self.kp = kp
        self.ki = ki
        self.p_acc = 0.0  # Proportional accumulator (with EMA)
        self.i_acc = 0.0  # Integral accumulator
    
    def update(self, actual_rate: float) -> float:
        """
        Update the controller and return the new deletion loss coefficient (α).
        
        Args:
            actual_rate: Current deletion rate (0 to 1, as percentage/100)
            
        Returns:
            α: Updated deletion loss coefficient
        """
        # Error: how much more we want to delete
        error = self.target_rate - actual_rate
        
        # Update accumulators (from MrT5 trainer.py)
        self.p_acc = 0.9 * self.p_acc + 0.1 * self.kp * error
        self.i_acc = self.i_acc + self.ki * error
        
        # PI control law
        alpha = max(0.0, self.p_acc + self.i_acc)
        
        return alpha


def compute_deletion_loss(delete_gate_output, input_ids, deletion_threshold, sigmoid_mask_scale, pad_token_id=0):
    """
    Compute the deletion regularization loss.
    
    Adapted from MrT5 trainer.py __compute_loss method.
    
    Returns:
        deletion_loss: The gate mean loss (to encourage deletion)
        percent_deleted: Percentage of non-pad tokens deleted (0-100)
    """
    if delete_gate_output is None:
        return torch.tensor(0.0), 0.0
    
    delete_gate_output = delete_gate_output.squeeze(-1)
    
    # Create mask to exclude PAD tokens (from MrT5 trainer.py line 286)
    non_pad_mask = input_ids != pad_token_id
    
    # Compute delete gate loss: mean of gate values for non-pad tokens
    # This encourages deletion (more negative = more deletion)
    # From MrT5 trainer.py line 290: delete_gate_loss = delete_gate_output[non_pad_mask].mean()
    if non_pad_mask.any():
        deletion_loss = delete_gate_output[non_pad_mask].mean()
    else:
        deletion_loss = delete_gate_output.mean()
    
    # Count deleted tokens (where gate < threshold)
    # From MrT5 trainer.py lines 323-328
    num_non_pad_tokens = non_pad_mask.sum()
    num_deleted = ((delete_gate_output < deletion_threshold) & non_pad_mask).sum()
    
    if num_non_pad_tokens > 0:
        percent_deleted = (num_deleted / num_non_pad_tokens * 100).item()
    else:
        percent_deleted = 0.0
    
    return deletion_loss, percent_deleted


# =============================================================================
# Dataset Preparation Functions
# =============================================================================

def prepare_mlm_dataset(args, tokenizer):
    """Prepare dataset for Masked Language Modeling."""
    
    # Check if using local mC4 dataset
    if args.dataset_name == "local_mc4":
        print(f"Loading LOCAL mC4 dataset from: {args.local_mc4_dir}")
        from mc4_dataset import load_mc4_dataset
        
        train_dataset = load_mc4_dataset(
            split="train",
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            mlm_probability=args.mlm_probability,
            streaming=args.streaming,
            data_dir=args.local_mc4_dir,
        )
        
        eval_dataset = load_mc4_dataset(
            split="validation",
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            mlm_probability=args.mlm_probability,
            streaming=False,  # Don't stream validation
            max_samples=1000,  # Limit validation size
            data_dir=args.local_mc4_dir,
        )
        
        # Return dict-like structure for compatibility
        class DatasetDict:
            def __init__(self, train, validation):
                self._train = train
                self._validation = validation
            def __getitem__(self, key):
                if key == "train":
                    return self._train
                elif key in ["validation", "test"]:
                    return self._validation
                raise KeyError(key)
        
        # No data collator needed - dataset already applies MLM
        return DatasetDict(train_dataset, eval_dataset), None
    
    # Standard HuggingFace dataset loading
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
    print(f"Target deletion rate: {args.target_deletion_rate:.1%}")
    if args.use_pi_controller:
        print(f"Using PI-controller: k_p={args.controller_p}, k_i={args.controller_i}")
    print()
    
    # Initialize PI-controller for deletion rate targeting
    pi_controller = PIController(
        target_rate=args.target_deletion_rate,
        kp=args.controller_p,
        ki=args.controller_i,
        alpha_0=args.deletion_loss_weight,
    )
    current_alpha = args.deletion_loss_weight  # Current deletion loss coefficient
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    total_task_loss = 0
    total_deletion_loss = 0
    total_deletion_rate = 0
    
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
            
            # Task loss (cross-entropy)
            task_loss = outputs.loss
            
            # Get input_ids for deletion loss calculation
            input_ids = batch.get('input_ids')
            
            # Deletion regularization loss with PI-controller
            delete_gate_output = getattr(outputs, 'delete_gate_output', None)
            deletion_loss, percent_deleted = compute_deletion_loss(
                delete_gate_output,
                input_ids,
                deletion_threshold=args.deletion_threshold,
                sigmoid_mask_scale=args.sigmoid_mask_scale,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # Convert percentage to rate (0-1) for PI-controller
            actual_del_rate = percent_deleted / 100.0
            
            # Update PI-controller to get current α (only after regularizer_delay)
            if args.use_pi_controller and global_step >= args.regularizer_delay:
                current_alpha = pi_controller.update(actual_del_rate)
            
            # Combined loss: task_loss + α * deletion_loss
            # Apply regularizer only after delay (from MrT5 trainer.py lines 330-337)
            if global_step >= args.regularizer_delay:
                loss = task_loss + current_alpha * deletion_loss
            else:
                loss = task_loss
            
            # Track deletion rate for logging
            total_deletion_rate += percent_deleted
            
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
                avg_del_pct = total_deletion_rate / args.logging_steps  # Percentage (0-100)
                
                print(f"\nStep {global_step}:")
                print(f"  Loss: {avg_loss:.4f} (Task: {avg_task_loss:.4f}, Del: {avg_deletion_loss:.4f})")
                print(f"  Deleted tokens: {avg_del_pct:.1f}% (target: {args.target_deletion_rate*100:.1f}%)")
                print(f"  α (deletion coeff): {current_alpha:.6f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
                
                total_loss = 0
                total_task_loss = 0
                total_deletion_loss = 0
                total_deletion_rate = 0
            
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
    # Handle both regular datasets and iterable datasets (for local_mc4)
    from torch.utils.data import IterableDataset
    
    train_dataset = tokenized_dataset["train"]
    is_iterable = isinstance(train_dataset, IterableDataset)
    
    # Custom collate function for when data_collator is None
    def default_collate(batch):
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        }
    
    collate_fn = data_collator if data_collator is not None else default_collate
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not is_iterable,  # Can't shuffle iterable datasets
        collate_fn=collate_fn,
    )
    
    eval_dataloader = None
    try:
        eval_dataset = tokenized_dataset["validation"]
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    except (KeyError, TypeError):
        pass  # No validation set
    
    # Train
    model = train(args, model, train_dataloader, eval_dataloader, tokenizer)
    
    print("\nTraining complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
