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

import sys
import os
# Allow running from any directory by adding the models/ directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForQuestionAnswering,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    DefaultDataCollator,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import wandb

from configuration_mrbert import MrBertConfig
from modeling_mrbert import (
    MrBertForMaskedLM,
    MrBertForSequenceClassification,
    MrBertForTokenClassification,
    MrBertForQuestionAnswering,
)
from pi_controller import PIController


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MrBERT for various tasks")
    
    # Task arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="MrBERT",
        choices=["MrBERT", "BERT"],
        help="Model architecture: MrBERT (with delete gate) or BERT (baseline, no gate)",
    )
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
        "--use_softmax1",
        action="store_true",
        default=True,
        help="Use softmax1 (n+1 denominator) for attention. Recommended by MrT5 paper.",
    )
    parser.add_argument(
        "--no_use_softmax1",
        action="store_false",
        dest="use_softmax1",
        help="Disable softmax1; use standard softmax.",
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
        "--local_snli_dir",
        type=str,
        default="snli_datasets",
        help="Directory containing local SNLI preprocessed NDJSON files (used when dataset_name='local_snli')",
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
        help="Use PI-controller for deletion rate (recommended by MrT5 paper).",
    )
    parser.add_argument(
        "--no_use_pi_controller",
        action="store_false",
        dest="use_pi_controller",
        help="Disable PI-controller; use fixed deletion_loss_weight throughout training.",
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
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mrbert",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name (default: auto-generated)",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="training-only",
        choices=["training-only", "eval-only", "training-and-eval"],
        help="Whether to run training, eval, or both (default: training-only)",
    )

    return parser.parse_args()


def compute_deletion_loss(delete_gate_output, input_ids, deletion_threshold, sigmoid_mask_scale, pad_token_id=0):
    """
    Compute the deletion regularization loss.
    
    Adapted from MrT5 trainer.py __compute_loss method.
    
    Returns:
        deletion_loss: The gate mean loss (to encourage deletion)
        percent_deleted: Percentage of non-pad tokens deleted (0-100)
    """
    if delete_gate_output is None:
        return torch.tensor(0.0), 0.0, 0.0
    
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
        percent_deleted_non_pad = (num_deleted / num_non_pad_tokens * 100).item()
    else:
        percent_deleted_non_pad = 0.0

    total_tokens = delete_gate_output.numel()
    num_deleted_all = (delete_gate_output < deletion_threshold).sum()
    percent_deleted_all = (num_deleted_all / total_tokens * 100).item() if total_tokens > 0 else 0.0

    return deletion_loss, percent_deleted_non_pad, percent_deleted_all


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

    if args.dataset_name == "local_snli":
        print(f"Loading LOCAL SNLI dataset from: {args.local_snli_dir}")
        from datasets import load_dataset as hf_load_dataset
        dataset = hf_load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_snli_dir}/snli-train.json",
                "validation": f"{args.local_snli_dir}/snli-validation.json",
                "test":       f"{args.local_snli_dir}/snli-test.json",
            },
        )
        # input_ids and attention_mask are already tokenized; unwrap the outer list
        # added by the preprocess script (shape was [1, seq_len] → [seq_len])
        dataset = dataset.map(
            lambda x: {
                "input_ids":      x["input_ids"][0],
                "attention_mask": x["attention_mask"][0],
                "labels":         x["labels"],
            },
            desc="Unwrapping precomputed features",
        )
        num_labels = 3  # entailment, neutral, contradiction
        data_collator = DefaultDataCollator()
        return dataset, data_collator, num_labels

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
    """Create the appropriate model for the task."""

    if args.model_type == "MrBERT":
        config = MrBertConfig.from_pretrained(
            args.model_name,
            deletion_type=args.deletion_type,
            delete_gate_layer=args.delete_gate_layer,
            sigmoid_mask_scale=args.sigmoid_mask_scale,
            use_gumbel_noise=True,
            use_softmax1=args.use_softmax1,
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
            model = MrBertForQuestionAnswering(config)
        else:
            raise ValueError(f"Unknown task: {args.task}")

    elif args.model_type == "BERT":
        config = BertConfig.from_pretrained(args.model_name)
        if args.task == "mlm":
            model = BertForMaskedLM.from_pretrained(args.model_name, config=config)
        elif args.task == "sequence_classification":
            config.num_labels = num_labels
            model = BertForSequenceClassification.from_pretrained(args.model_name, config=config)
        elif args.task == "token_classification":
            config.num_labels = num_labels
            model = BertForTokenClassification.from_pretrained(args.model_name, config=config)
        elif args.task == "question_answering":
            model = BertForQuestionAnswering.from_pretrained(args.model_name, config=config)
        else:
            raise ValueError(f"Unknown task: {args.task}")

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    return model


# =============================================================================
# Evaluation Loop
# =============================================================================

SNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def print_deletion_samples(args, model, eval_dataloader, tokenizer, n_samples=20):
    """Print n_samples examples showing which tokens were kept vs deleted."""
    model.eval()
    samples_printed = 0

    print("\n" + "=" * 70)
    print(f"DELETION SAMPLES (first {n_samples})")
    print("=" * 70)

    with torch.no_grad():
        for batch in eval_dataloader:
            if samples_printed >= n_samples:
                break

            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)

            input_ids = batch["input_ids"]
            labels = batch.get("labels")
            delete_gate_mask = getattr(outputs, "delete_gate_mask", None)
            logits = getattr(outputs, "logits", None)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                if samples_printed >= n_samples:
                    break

                ids = input_ids[i].tolist()
                tokens = tokenizer.convert_ids_to_tokens(ids)

                # Gate values for this example
                if delete_gate_mask is not None:
                    gate_vals = delete_gate_mask[i].squeeze(-1).tolist()
                    deleted = [gate_vals[j] < args.deletion_threshold for j in range(len(tokens))]
                else:
                    deleted = [False] * len(tokens)

                # Strip padding
                pad_id = tokenizer.pad_token_id
                non_pad = [(tok, d, ids[j]) for j, (tok, d) in enumerate(zip(tokens, deleted)) if ids[j] != pad_id]

                kept_tokens   = [tok for tok, d, _ in non_pad if not d]
                deleted_tokens = [tok for tok, d, _ in non_pad if d]

                # Prediction and ground truth
                pred_label = SNLI_LABELS.get(logits[i].argmax().item(), "?") if logits is not None else "?"
                true_label = SNLI_LABELS.get(labels[i].item(), "?") if labels is not None else "?"
                correct = "✓" if pred_label == true_label else "✗"

                # Full sequence (mark deleted tokens with [X])
                annotated = []
                for tok, d, tid in non_pad:
                    if tid in (tokenizer.cls_token_id, tokenizer.sep_token_id):
                        annotated.append(tok)
                    elif d:
                        annotated.append(f"[{tok}]")
                    else:
                        annotated.append(tok)

                print(f"\nSample {samples_printed + 1}:")
                print(f"  Full sequence (deleted tokens in [brackets]):")
                print(f"    {' '.join(annotated)}")
                print(f"  Kept    ({len(kept_tokens):2d} tokens): {' '.join(kept_tokens)}")
                print(f"  Deleted ({len(deleted_tokens):2d} tokens): {' '.join(deleted_tokens) if deleted_tokens else '(none)'}")
                print(f"  Label: {true_label} | Predicted: {pred_label} {correct}")

                samples_printed += 1

    print("\n" + "=" * 70)
    model.train()


def evaluate(args, model, eval_dataloader, tokenizer, step=None):
    """Run evaluation and return metrics. Optionally logs to W&B."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_deletion_rate = 0
    total_seq_len = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)

            total_loss += outputs.loss.item()

            if hasattr(outputs, 'logits') and outputs.logits is not None and 'labels' in batch:
                preds = outputs.logits.argmax(dim=-1)
                total_accuracy += (preds == batch['labels']).float().mean().item()

            input_ids = batch.get('input_ids')
            delete_gate_output = getattr(outputs, 'delete_gate_output', None)
            _, percent_deleted, _ = compute_deletion_loss(
                delete_gate_output,
                input_ids,
                deletion_threshold=args.deletion_threshold,
                sigmoid_mask_scale=args.sigmoid_mask_scale,
                pad_token_id=tokenizer.pad_token_id,
            )
            total_deletion_rate += percent_deleted

            delete_gate_mask = getattr(outputs, 'delete_gate_mask', None)
            if delete_gate_mask is not None:
                kept = (delete_gate_mask.squeeze(-1) > args.deletion_threshold).float()
                total_seq_len += kept.sum(dim=1).mean().item()
            else:
                total_seq_len += (input_ids != tokenizer.pad_token_id).float().sum(dim=1).mean().item()

            num_batches += 1

    model.train()

    metrics = {
        "eval/loss": round(total_loss / num_batches, 4),
        "eval/accuracy": round(total_accuracy / num_batches, 4),
        "eval/percent_deleted_tokens": round(total_deletion_rate / num_batches, 4),
        "eval/avg_seq_len": round(total_seq_len / num_batches, 2),
    }

    print(f"\nEval results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.model_type == "MrBERT":
        # Dropped token analysis and compute savings
        from diagnostics import (
            print_dropped_token_summary,
            theoretical_compute_saved_pct,
            aggregate_dropped_stats,
            analyze_dropped_tokens,
        )

        # Collect dropped token stats across first 5 batches for a representative sample
        model.eval()
        stats_list = []
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= 5:
                    break
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                gate_mask = getattr(outputs, "delete_gate_mask", None)
                if gate_mask is None:
                    break
                gate = gate_mask.squeeze(-1).cpu()
                input_ids = batch["input_ids"].cpu()
                keep_mask = gate > args.deletion_threshold
                for b in range(input_ids.size(0)):
                    stats_list.append(
                        analyze_dropped_tokens(tokenizer, input_ids, keep_mask, batch_index=b)
                    )
        model.train()

        if stats_list:
            agg = aggregate_dropped_stats(stats_list)
            print(f"\n  [Diagnostics] Dropped token analysis ({agg['n_samples']} examples):")
            print(f"    avg kept={agg['avg_kept']:.1f}  avg dropped={agg['avg_dropped']:.1f}")
            print(f"    dropped by type: {agg['dropped_by_type']}")
            print(f"    kept by type:    {agg['kept_by_type']}")

            # Compute savings using avg_seq_len from metrics
            avg_seq_before = agg["avg_kept"] + agg["avg_dropped"]
            avg_seq_after  = agg["avg_kept"]
            compute_saved  = theoretical_compute_saved_pct(
                seq_before=round(avg_seq_before),
                seq_after=round(avg_seq_after),
                num_layers=args.num_hidden_layers if hasattr(args, "num_hidden_layers") else 12,
                gate_layer_index=args.delete_gate_layer,
            )
            print(f"    theoretical compute saved (MACs, Appendix C): ~{compute_saved * 100:.1f}%")
            metrics["eval/compute_saved_pct"] = round(compute_saved * 100, 2)
            metrics["eval/dropped_words_pct"] = round(
                agg["dropped_by_type"]["word"] / max(agg["avg_dropped"], 1e-6), 4
            )

        # Print first-batch deletion sample
        print_deletion_samples(args, model, eval_dataloader, tokenizer)

    if not args.disable_wandb:
        wandb.log(metrics, step=step)

    return metrics


# =============================================================================
# Training Loop
# =============================================================================

def train(args, model, train_dataloader, eval_dataloader, tokenizer):
    """Training loop."""

    # Initialize wandb
    if not args.disable_wandb:
        run_name = args.wandb_run_name or f"{args.model_type}_{args.model_name.split('/')[-1]}_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_type": args.model_type,
                "model_name": args.model_name,
                "task": args.task,
                "delete_gate_layer": args.delete_gate_layer,
                "deletion_type": args.deletion_type,
                "sigmoid_mask_scale": args.sigmoid_mask_scale,
                "deletion_threshold": args.deletion_threshold,
                "target_deletion_rate": args.target_deletion_rate,
                "learning_rate": args.learning_rate,
                "delete_gate_lr": args.delete_gate_lr,
                "batch_size": args.batch_size,
                "max_steps": args.max_steps,
                "warmup_steps": args.warmup_steps,
                "controller_p": args.controller_p,
                "controller_i": args.controller_i,
                "deletion_loss_weight": args.deletion_loss_weight,
                "regularizer_delay": args.regularizer_delay,
                "seed": args.seed,
            },
        )
    
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
    print(f"Steps per epoch: {len(train_dataloader)}")
    print(f"Epochs: {args.num_epochs}")
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
    
    import time

    # Training loop
    model.train()
    train_start = time.time()
    global_step = 0
    total_loss = 0
    total_task_loss = 0
    total_deletion_loss = 0
    total_deletion_rate = 0
    total_deletion_rate_all = 0
    total_gate_avg = 0
    total_gate_std = 0
    total_gate_max = 0
    total_gate_min = 0
    total_accuracy = 0
    total_seq_len = 0
    
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
            deletion_loss, percent_deleted, percent_deleted_all = compute_deletion_loss(
                delete_gate_output,
                input_ids,
                deletion_threshold=args.deletion_threshold,
                sigmoid_mask_scale=args.sigmoid_mask_scale,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Convert percentage to rate (0-1) for PI-controller (uses non-pad rate)
            actual_del_rate = percent_deleted / 100.0

            # Accumulate per-step gate distribution stats
            if delete_gate_output is not None:
                gate_vals = delete_gate_output.squeeze(-1)
                total_gate_avg += gate_vals.mean().item()
                total_gate_std += gate_vals.std(dim=1).mean().item()
                total_gate_max += gate_vals.max(dim=1).values.mean().item()
                total_gate_min += gate_vals.min(dim=1).values.mean().item()

            # Accuracy: fraction of correct predictions in batch
            if hasattr(outputs, 'logits') and outputs.logits is not None and 'labels' in batch:
                preds = outputs.logits.argmax(dim=-1)
                total_accuracy += (preds == batch['labels']).float().mean().item()

            # Effective sequence length after soft deletion (tokens with gate > threshold)
            # For BERT baseline (no gate), use full input length minus padding
            delete_gate_mask = getattr(outputs, 'delete_gate_mask', None)
            if delete_gate_mask is not None:
                kept = (delete_gate_mask.squeeze(-1) > args.deletion_threshold).float()
                total_seq_len += kept.sum(dim=1).mean().item()
            else:
                # No gate: count non-pad tokens
                total_seq_len += (input_ids != tokenizer.pad_token_id).float().sum(dim=1).mean().item()
            
            # Update PI-controller to get current α (only after regularizer_delay, only for MrBERT)
            if args.model_type == "MrBERT" and args.use_pi_controller and global_step >= args.regularizer_delay:
                current_alpha = pi_controller.update(actual_del_rate)
            
            # Combined loss: task_loss + α * deletion_loss
            # Apply regularizer only after delay (from MrT5 trainer.py lines 330-337)
            if global_step >= args.regularizer_delay:
                loss = task_loss + current_alpha * deletion_loss
            else:
                loss = task_loss
            
            # Track deletion rate for logging
            total_deletion_rate += percent_deleted
            total_deletion_rate_all += percent_deleted_all
            
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
                "acc": f"{(outputs.logits.argmax(-1) == batch['labels']).float().mean().item():.3f}" if hasattr(outputs, 'logits') and 'labels' in batch else "n/a",
            })
            
            # Logging
            if global_step % args.logging_steps == 0:
                avg_loss = total_loss / args.logging_steps
                avg_task_loss = total_task_loss / args.logging_steps
                avg_deletion_loss = total_deletion_loss / args.logging_steps
                avg_del_pct_non_pad = total_deletion_rate / args.logging_steps
                avg_del_pct_all = total_deletion_rate_all / args.logging_steps
                avg_gate_avg = total_gate_avg / args.logging_steps
                avg_gate_std = total_gate_std / args.logging_steps
                avg_gate_max = total_gate_max / args.logging_steps
                avg_gate_min = total_gate_min / args.logging_steps
                avg_accuracy = total_accuracy / args.logging_steps
                avg_seq_len = total_seq_len / args.logging_steps

                elapsed = time.time() - train_start
                elapsed_str = f"{elapsed/3600:.1f}h" if elapsed >= 3600 else f"{elapsed/60:.1f}m"

                print(f"\nStep {global_step} [{elapsed_str}]:")
                print(f"  Loss: {avg_loss:.4f} (Task: {avg_task_loss:.4f}, Del: {avg_deletion_loss:.4f})")
                print(f"  Accuracy: {avg_accuracy:.4f}")
                print(f"  Seq length (effective): {avg_seq_len:.1f}")
                print(f"  Deleted tokens (non-pad): {avg_del_pct_non_pad:.1f}% (target: {args.target_deletion_rate*100:.1f}%)")
                print(f"  α (deletion coeff): {current_alpha:.6f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

                if not args.disable_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "loss": round(avg_loss, 4),
                        "cross_entropy_loss": round(avg_task_loss, 4),
                        "delete_gate_loss": round(avg_deletion_loss, 4),
                        "total_loss": round(avg_loss, 4),
                        "accuracy": round(avg_accuracy, 4),
                        "new_seq_len": round(avg_seq_len, 2),
                        "percent_deleted_tokens": round(avg_del_pct_all, 4),
                        "percent_non_pad_deleted_tokens": round(avg_del_pct_non_pad, 4),
                        "delete_gate_average": round(avg_gate_avg, 4),
                        "delete_gate_std": round(avg_gate_std, 4),
                        "delete_gate_max_value": round(avg_gate_max, 4),
                        "delete_gate_min_value": round(avg_gate_min, 4),
                        "delete_gate_loss_coeff": round(current_alpha, 6),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }, step=global_step)

                total_loss = 0
                total_task_loss = 0
                total_deletion_loss = 0
                total_deletion_rate = 0
                total_deletion_rate_all = 0
                total_gate_avg = 0
                total_gate_std = 0
                total_gate_max = 0
                total_gate_min = 0
                total_accuracy = 0
                total_seq_len = 0
            
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

    if not args.disable_wandb and args.mode == "training-only":
        wandb.finish()

    return model, global_step


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 60)
    print(f"Training - Model: {args.model_type}, Task: {args.task.upper()}")
    print("=" * 60)
    print(f"Device: {args.device}")
    if args.model_type == "MrBERT":
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

    from diagnostics import log_parameter_summary
    log_parameter_summary(model, label=args.model_type)
    
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
    
    # Train / eval based on --mode
    global_step = 0
    if args.mode in ("training-only", "training-and-eval"):
        model, global_step = train(args, model, train_dataloader, eval_dataloader, tokenizer)
        print("\nTraining complete!")
        print("=" * 60)

    if args.mode in ("eval-only", "training-and-eval"):
        if eval_dataloader is None:
            print("No validation set available — skipping eval.")
        else:
            if not args.disable_wandb and args.mode == "eval-only":
                run_name = args.wandb_run_name or f"{args.model_type}_{args.model_name.split('/')[-1]}_seed{args.seed}_eval"
                wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            evaluate(args, model, eval_dataloader, tokenizer, step=global_step)
            if not args.disable_wandb:
                wandb.finish()
        print("\nEval complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
