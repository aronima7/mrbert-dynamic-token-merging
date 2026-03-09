#!/usr/bin/env python3
"""
Fine-tuning script for MrXLMR using HuggingFace Trainer.

Supports: MLM, sequence classification, token classification, question answering.

Usage:
    # Sequence Classification (SNLI)
    python train_mrxlmr.py --task sequence_classification --dataset_name local_snli \
        --local_snli_dir /checkpoints/snli_datasets --output_dir ./mrxlmr_snli

    # Token Classification (CoNLL-2003 NER)
    python train_mrxlmr.py --task token_classification --dataset_name conll2003 \
        --output_dir ./mrxlmr_ner

    # Question Answering (SQuAD)
    python train_mrxlmr.py --task question_answering --dataset_name local_squad \
        --local_squad_dir /checkpoints/squad_datasets --output_dir ./mrxlmr_squad

    # MLM (wikitext)
    python train_mrxlmr.py --task mlm --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
        --output_dir ./mrxlmr_mlm

    # Quick smoke test (100 steps)
    python train_mrxlmr.py --max_steps 100 --logging_steps 10 --output_dir ./test_run

HuggingFace TrainingArguments flags apply directly, e.g.:
    --per_device_train_batch_size 16
    --num_train_epochs 3
    --eval_steps 500
    --save_steps 1000
    --learning_rate 2e-5
"""

import sys
import os
import random
import statistics

# Add the models/ and training/ directories to the path so local modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaForQuestionAnswering,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from datasets import load_dataset, DatasetDict

from configuration_mrxlmr import MrXLMRConfig
from modeling_mrxlmr import (
    MrXLMRForMaskedLM,
    MrXLMRForSequenceClassification,
    MrXLMRForTokenClassification,
    MrXLMRForQuestionAnswering,
)
from pi_controller import PIController


# =============================================================================
# Training Arguments
# =============================================================================

@dataclass
class MrXLMRTrainingArguments(TrainingArguments):
    """
    Extended TrainingArguments for MrXLMR training.

    Adds MrXLMR-specific parameters on top of the standard HuggingFace
    TrainingArguments (output_dir, learning_rate, num_train_epochs, etc.).
    """

    # ---- Model ----
    model_type: str = field(
        default="MrXLMR",
        metadata={"help": "Model type: MrXLMR (with delete gate) or XLMR (baseline)"},
    )
    model_name: str = field(
        default="xlm-roberta-base",
        metadata={"help": "Pretrained model name or path (default: xlm-roberta-base)"},
    )
    delete_gate_layer: int = field(
        default=3,
        metadata={"help": "Encoder layer index where the delete gate is applied (0-indexed)"},
    )
    deletion_type: str = field(
        default="scaled_sigmoid",
        metadata={"help": "Delete gate type: scaled_sigmoid | log_sigmoid | random | fixed"},
    )
    sigmoid_mask_scale: float = field(
        default=-30.0,
        metadata={"help": "Scale factor for the sigmoid delete gate (MrT5 default: -30.0)"},
    )
    deletion_threshold: float = field(
        default=-15.0,
        metadata={"help": "Gate value below which a token is counted as deleted (MrT5 default: -15.0)"},
    )
    bypass_gate: bool = field(
        default=False,
        metadata={"help": "Disable the delete gate entirely (useful for ablation)"},
    )
    use_gumbel_noise: bool = field(
        default=False,
        metadata={"help": "Add Gumbel noise to gate logits during training for exploration"},
    )
    use_softmax1: bool = field(
        default=True,
        metadata={"help": "Use softmax1 (n+1 denominator) for attention — recommended by MrT5"},
    )
    use_pre_deletion_blend: bool = field(
        default=True,
        metadata={"help": "Blend pre-deletion hidden states into deleted-token representations"},
    )

    # ---- Dataset ----
    task: str = field(
        default="sequence_classification",
        metadata={"help": "Task type: mlm | sequence_classification | token_classification | question_answering"},
    )
    dataset_name: str = field(
        default="local_snli",
        metadata={"help": "Dataset name (HuggingFace hub or local_snli/local_squad/local_sst2/local_mrpc/local_imdb/local_tydiqa)"},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config (e.g. 'sst2' for GLUE)"},
    )
    local_snli_dir: str = field(
        default="snli_datasets",
        metadata={"help": "Directory with pre-tokenized SNLI NDJSON files (snli-train.json etc.)"},
    )
    local_squad_dir: str = field(
        default="squad_datasets",
        metadata={"help": "Directory with pre-tokenized SQuAD NDJSON files"},
    )
    local_sst2_dir: str = field(
        default="sst2_datasets",
        metadata={"help": "Directory with pre-tokenized SST-2 NDJSON files"},
    )
    local_mrpc_dir: str = field(
        default="mrpc_datasets",
        metadata={"help": "Directory with pre-tokenized MRPC NDJSON files"},
    )
    local_imdb_dir: str = field(
        default="imdb_datasets",
        metadata={"help": "Directory with pre-tokenized IMDB NDJSON files"},
    )
    local_tydiqa_dir: str = field(
        default="tydiqa_datasets",
        metadata={"help": "Directory with pre-tokenized TyDi QA NDJSON files"},
    )
    local_xnli_dir: str = field(
        default="xnli_datasets",
        metadata={"help": "Directory with pre-tokenized XNLI NDJSON files (from data/preprocess_xnli.py)"},
    )
    local_mc4_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory with local mC4 dataset (optional, for MLM)"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate training set to this many examples (useful for quick local tests)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate eval/test sets to this many examples (useful for quick local tests)."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Maximum tokenized sequence length"},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Probability of masking each token for MLM"},
    )

    # ---- Deletion Loss / PI Controller ----
    deletion_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Initial weight α for the deletion regularisation loss"},
    )
    target_deletion_rate: float = field(
        default=0.4,
        metadata={"help": "Target fraction of non-pad tokens to delete (δ in the MrT5 paper)"},
    )
    controller_p: float = field(
        default=0.5,
        metadata={"help": "Proportional gain k_p for the PI controller"},
    )
    controller_i: float = field(
        default=5e-5,
        metadata={"help": "Integral gain k_i for the PI controller"},
    )
    regularizer_delay: int = field(
        default=0,
        metadata={"help": "Steps to train on task loss only before turning on the deletion regulariser"},
    )
    use_pi_controller: bool = field(
        default=True,
        metadata={"help": "Dynamically adjust deletion loss weight α with a PI controller"},
    )
    hard_delete_train_prob: float = field(
        default=0.0,
        metadata={"help": "Probability of using hard deletion (physical token removal) on each training step"},
    )

    # ---- W&B ----
    wandb_project: str = field(
        default="mrxlmr",
        metadata={"help": "W&B project name"},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "W&B run name (default: auto-generated)"},
    )
    disable_wandb: bool = field(
        default=False,
        metadata={"help": "Disable W&B logging entirely"},
    )

    # ---- Backward-compatible aliases ----
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Alias for --per_device_train/eval_batch_size"},
    )
    num_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "Alias for --num_train_epochs"},
    )
    mode: str = field(
        default="training-only",
        metadata={"help": "training-only | eval-only | training-and-eval"},
    )

    def __post_init__(self):
        if self.batch_size is not None:
            self.per_device_train_batch_size = self.batch_size
            self.per_device_eval_batch_size = self.batch_size
        if self.num_epochs is not None:
            self.num_train_epochs = float(self.num_epochs)

        if getattr(self, "eval_strategy", "no") == "no":
            self.eval_strategy = "steps"
            self.evaluation_strategy = "steps"

        if self.disable_wandb:
            self.report_to = ["none"]
        else:
            self.report_to = ["wandb"]
            os.environ["WANDB_PROJECT"] = self.wandb_project
            run_name = self.wandb_run_name or (
                f"{self.model_type}_{self.model_name.split('/')[-1]}_seed{self.seed}"
            )
            self.run_name = run_name

        super().__post_init__()


# =============================================================================
# Deletion Loss Utility
# =============================================================================

def compute_deletion_loss(delete_gate_output, input_ids, deletion_threshold, sigmoid_mask_scale, pad_token_id=1):
    """
    Compute the deletion regularisation loss and deletion statistics.

    XLM-R pad_token_id = 1 (default).

    Returns:
        deletion_loss: Scalar tensor (mean gate value over non-pad tokens).
        percent_deleted_non_pad: % of non-pad tokens with gate < threshold.
        percent_deleted_all: % of all tokens with gate < threshold.
    """
    if delete_gate_output is None:
        return torch.tensor(0.0), 0.0, 0.0

    delete_gate_output = delete_gate_output.squeeze(-1)
    non_pad_mask = input_ids != pad_token_id

    if non_pad_mask.any():
        deletion_loss = delete_gate_output[non_pad_mask].mean()
    else:
        deletion_loss = delete_gate_output.mean()

    num_non_pad_tokens = non_pad_mask.sum()
    num_deleted = ((delete_gate_output < deletion_threshold) & non_pad_mask).sum()
    percent_deleted_non_pad = (num_deleted / num_non_pad_tokens * 100).item() if num_non_pad_tokens > 0 else 0.0

    total_tokens = delete_gate_output.numel()
    num_deleted_all = (delete_gate_output < deletion_threshold).sum()
    percent_deleted_all = (num_deleted_all / total_tokens * 100).item() if total_tokens > 0 else 0.0

    return deletion_loss, percent_deleted_non_pad, percent_deleted_all


# =============================================================================
# Dataset Preparation
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
        return tokenizer(texts, truncation=True, max_length=args.max_seq_length,
                         padding="max_length", return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True,
        remove_columns=dataset["train"].column_names, desc="Tokenizing",
    )
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
    )
    return tokenized_dataset, data_collator


def prepare_sequence_classification_dataset(args, tokenizer):
    """
    Prepare dataset for Sequence Classification.

    Supports GLUE tasks, local SNLI/SST-2/MRPC/IMDB, and generic datasets.
    Returns (dataset_dict, data_collator, num_labels).
    """
    if args.dataset_name == "local_snli":
        print(f"Loading LOCAL SNLI dataset from: {args.local_snli_dir}")
        dataset = load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_snli_dir}/snli-train.json",
                "validation": f"{args.local_snli_dir}/snli-validation.json",
                "test":       f"{args.local_snli_dir}/snli-test.json",
            },
        )
        dataset = dataset.map(
            lambda x: {
                "input_ids":      x["input_ids"][0],
                "attention_mask": x["attention_mask"][0],
                "labels":         x["labels"],
            },
            desc="Unwrapping precomputed features",
        )
        return dataset, DefaultDataCollator(), 3

    if args.dataset_name == "local_sst2":
        print(f"Loading LOCAL SST-2 dataset from: {args.local_sst2_dir}")
        dataset = load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_sst2_dir}/sst2-train.json",
                "validation": f"{args.local_sst2_dir}/sst2-validation.json",
            },
        )
        return dataset, DefaultDataCollator(), 2

    if args.dataset_name == "local_mrpc":
        print(f"Loading LOCAL MRPC dataset from: {args.local_mrpc_dir}")
        dataset = load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_mrpc_dir}/mrpc-train.json",
                "validation": f"{args.local_mrpc_dir}/mrpc-validation.json",
                "test":       f"{args.local_mrpc_dir}/mrpc-test.json",
            },
        )
        return dataset, DefaultDataCollator(), 2

    if args.dataset_name == "local_imdb":
        print(f"Loading LOCAL IMDB dataset from: {args.local_imdb_dir}")
        dataset = load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_imdb_dir}/imdb-train.json",
                "validation": f"{args.local_imdb_dir}/imdb-validation.json",
                "test":       f"{args.local_imdb_dir}/imdb-test.json",
            },
        )
        return dataset, DefaultDataCollator(), 2

    if args.dataset_name == "local_xnli":
        print(f"Loading LOCAL XNLI dataset from: {args.local_xnli_dir}")
        dataset = load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_xnli_dir}/xnli-train.json",
                "validation": f"{args.local_xnli_dir}/xnli-validation.json",
                "test":       f"{args.local_xnli_dir}/xnli-test.json",
            },
        )
        dataset = dataset.map(
            lambda x: {
                "input_ids":      x["input_ids"][0],
                "attention_mask": x["attention_mask"][0],
                "labels":         x["labels"],
            },
            desc="Unwrapping precomputed features",
        )
        return dataset, DefaultDataCollator(), 3

    if args.dataset_name in ("imdb", "stanfordnlp/imdb"):
        print(f"Loading IMDB dataset (with 90/10 train/validation split)...")
        dataset = load_dataset(args.dataset_name)
        train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            "train":      train_val["train"],
            "validation": train_val["test"],
            "test":       dataset["test"],
        })

        def tokenize_imdb(examples):
            return tokenizer(examples["text"], truncation=True,
                             max_length=args.max_seq_length, padding="max_length")

        tokenized = dataset.map(tokenize_imdb, batched=True, desc="Tokenizing")
        tokenized = tokenized.rename_column("label", "labels")
        return tokenized, DefaultDataCollator(), 2

    print(f"Loading classification dataset: {args.dataset_name}/{args.dataset_config}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)

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
    elif args.dataset_name in ("stanfordnlp/snli", "snli"):
        text_columns = ["premise", "hypothesis"]
    else:
        cols = dataset["train"].column_names
        text_columns = ["text"] if "text" in cols else (["sentence"] if "sentence" in cols else [cols[0]])

    label_column = "label"

    def tokenize_function(examples):
        if len(text_columns) == 1:
            return tokenizer(examples[text_columns[0]], truncation=True,
                             max_length=args.max_seq_length, padding="max_length")
        return tokenizer(examples[text_columns[0]], examples[text_columns[1]],
                         truncation=True, max_length=args.max_seq_length, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, desc="Tokenizing")
    if label_column in tokenized_dataset["train"].column_names:
        tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")

    num_labels = len(set(dataset["train"][label_column]))
    return tokenized_dataset, DefaultDataCollator(), num_labels


def prepare_token_classification_dataset(args, tokenizer):
    """
    Prepare dataset for Token Classification (e.g. NER with CoNLL-2003).

    Aligns word-level labels to subword tokens: only the first subword of each
    word gets the real label; subsequent subwords get -100 (ignored by the loss).
    Returns (dataset_dict, data_collator, num_labels).
    """
    print(f"Loading token classification dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    if "ner_tags" in dataset["train"].features:
        label_list = dataset["train"].features["ner_tags"].feature.names
    elif "pos_tags" in dataset["train"].features:
        label_list = dataset["train"].features["pos_tags"].feature.names
    else:
        label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    num_labels = len(label_list)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, max_length=args.max_seq_length,
            padding="max_length", is_split_into_words=True,
        )
        labels = []
        label_key = "ner_tags" if "ner_tags" in examples else "pos_tags"
        for i, label in enumerate(examples[label_key]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids, previous_word_idx = [], None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx] if word_idx < len(label) else -100)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, desc="Tokenizing")
    return tokenized_dataset, DataCollatorForTokenClassification(tokenizer), num_labels


def prepare_question_answering_dataset(args, tokenizer):
    """
    Prepare dataset for extractive Question Answering (SQuAD / TyDi QA).

    Supports pre-tokenized local datasets and live loading from HuggingFace Hub.
    Returns (dataset_dict, data_collator).
    """
    if args.dataset_name == "local_squad":
        print(f"Loading LOCAL SQuAD dataset from: {args.local_squad_dir}")
        dataset = load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_squad_dir}/squad-train.json",
                "validation": f"{args.local_squad_dir}/squad-validation.json",
                "test":       f"{args.local_squad_dir}/squad-validation.json",
            },
        )
        return dataset, DefaultDataCollator()

    if args.dataset_name == "local_tydiqa":
        print(f"Loading LOCAL TyDi QA dataset from: {args.local_tydiqa_dir}")
        dataset = load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_tydiqa_dir}/tydiqa-train.json",
                "validation": f"{args.local_tydiqa_dir}/tydiqa-validation.json",
                "test":       f"{args.local_tydiqa_dir}/tydiqa-validation.json",
            },
        )
        return dataset, DefaultDataCollator()

    print(f"Loading QA dataset: {args.dataset_name}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)

    if args.dataset_name == "tydiqa":
        print("Filtering TyDi QA for English-only examples...")
        dataset = DatasetDict({
            split: dataset[split].filter(lambda x: x["id"].startswith("english-"))
            for split in dataset.keys()
        })

    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["question"], examples["context"],
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
        prepare_train_features, batched=True,
        remove_columns=dataset["train"].column_names, desc="Tokenizing",
    )
    return tokenized_dataset, DefaultDataCollator()


# =============================================================================
# Model Creation
# =============================================================================

def create_model(args, tokenizer, num_labels=None):
    """
    Instantiate the appropriate model for the given task and model_type.

    MrXLMR: loads pretrained XLM-R weights via from_pretrained, then randomly
            initialises only the delete gate (bias=10, weight_std=0.001).
    XLMR:   loads pretrained weights from HuggingFace Hub via from_pretrained.
    """
    if args.model_type == "MrXLMR":
        config = MrXLMRConfig.from_pretrained(
            args.model_name,
            deletion_type=args.deletion_type,
            delete_gate_layer=args.delete_gate_layer,
            sigmoid_mask_scale=args.sigmoid_mask_scale,
            use_gumbel_noise=args.use_gumbel_noise,
            use_softmax1=args.use_softmax1,
            bypass_gate=args.bypass_gate,
            use_pre_deletion_blend=args.use_pre_deletion_blend,
        )
        if args.task == "mlm":
            return MrXLMRForMaskedLM.from_pretrained(
                args.model_name, config=config, ignore_mismatched_sizes=True)
        elif args.task == "sequence_classification":
            config.num_labels = num_labels
            return MrXLMRForSequenceClassification.from_pretrained(
                args.model_name, config=config, ignore_mismatched_sizes=True)
        elif args.task == "token_classification":
            config.num_labels = num_labels
            return MrXLMRForTokenClassification.from_pretrained(
                args.model_name, config=config, ignore_mismatched_sizes=True)
        elif args.task == "question_answering":
            return MrXLMRForQuestionAnswering.from_pretrained(
                args.model_name, config=config, ignore_mismatched_sizes=True)

    elif args.model_type == "XLMR":
        config = XLMRobertaConfig.from_pretrained(args.model_name)
        if args.task == "mlm":
            return XLMRobertaForMaskedLM.from_pretrained(args.model_name, config=config)
        elif args.task == "sequence_classification":
            config.num_labels = num_labels
            return XLMRobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
        elif args.task == "token_classification":
            config.num_labels = num_labels
            return XLMRobertaForTokenClassification.from_pretrained(args.model_name, config=config)
        elif args.task == "question_answering":
            return XLMRobertaForQuestionAnswering.from_pretrained(args.model_name, config=config)

    raise ValueError(f"Unknown model_type={args.model_type} / task={args.task}")


# =============================================================================
# Deletion Sample Diagnostics
# =============================================================================

SNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def print_deletion_samples(args, model, dataloader, tokenizer, n_samples=20, split="TEST"):
    """Print the first n_samples examples showing which tokens the gate kept vs. deleted."""
    model.eval()
    samples_printed = 0

    print("\n" + "=" * 70)
    print(f"DELETION {split} SAMPLES (first {n_samples})")
    print("=" * 70)

    with torch.no_grad():
        for batch in dataloader:
            if samples_printed >= n_samples:
                break
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)

            input_ids = batch["input_ids"]
            labels = batch.get("labels")
            delete_gate_mask = getattr(outputs, "delete_gate_mask", None)
            logits = getattr(outputs, "logits", None)

            for i in range(input_ids.size(0)):
                if samples_printed >= n_samples:
                    break
                ids = input_ids[i].tolist()
                tokens = tokenizer.convert_ids_to_tokens(ids)

                if delete_gate_mask is not None:
                    gate_vals = delete_gate_mask[i].squeeze(-1).tolist()
                    deleted = [gate_vals[j] < args.deletion_threshold for j in range(len(tokens))]
                else:
                    deleted = [False] * len(tokens)

                pad_id = tokenizer.pad_token_id
                non_pad = [(tok, d, ids[j]) for j, (tok, d) in enumerate(zip(tokens, deleted)) if ids[j] != pad_id]
                kept_tokens    = [tok for tok, d, _ in non_pad if not d]
                deleted_tokens = [tok for tok, d, _ in non_pad if d]

                pred_label = SNLI_LABELS.get(logits[i].argmax().item(), "?") if logits is not None else "?"
                true_label = SNLI_LABELS.get(labels[i].item(), "?") if labels is not None else "?"
                correct = "✓" if pred_label == true_label else "✗"

                annotated = []
                for tok, d, tid in non_pad:
                    if tid in (tokenizer.cls_token_id, tokenizer.sep_token_id):
                        annotated.append(tok)
                    elif d:
                        annotated.append(f"[{tok}]")
                    else:
                        annotated.append(tok)

                print(f"\nSample {samples_printed + 1}:")
                print(f"  Full sequence (deleted in [brackets]):")
                print(f"    {' '.join(annotated)}")
                print(f"  Kept    ({len(kept_tokens):2d}): {' '.join(kept_tokens)}")
                print(f"  Deleted ({len(deleted_tokens):2d}): {' '.join(deleted_tokens) or '(none)'}")
                print(f"  Label: {true_label} | Predicted: {pred_label} {correct}")
                samples_printed += 1

    print("\n" + "=" * 70)
    model.train()


# =============================================================================
# SQuAD EM / F1 Evaluation
# =============================================================================

def compute_squad_em_f1(args, model, test_dataset, tokenizer):
    """Compute Exact Match (EM) and F1 for SQuAD / TyDi QA evaluation."""
    import re
    import string
    from collections import Counter

    def normalize_answer(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return " ".join(s.split())

    def f1_score(pred, gold):
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall    = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    if "answer_text" not in test_dataset.column_names:
        print("\n  Skipping EM/F1: 'answer_text' not in dataset.")
        return {}

    model.eval()
    em_scores, f1_scores = [], []
    batch_size = args.per_device_eval_batch_size

    print(f"\nComputing SQuAD EM/F1 on {len(test_dataset)} features...")

    with torch.no_grad():
        for start in range(0, len(test_dataset), batch_size):
            batch = test_dataset[start: start + batch_size]

            input_ids      = torch.tensor(batch["input_ids"]).to(args.device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(args.device)
            start_positions = batch["start_positions"]
            answer_texts    = batch["answer_text"]

            outputs      = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits   = outputs.end_logits

            for i in range(len(answer_texts)):
                if start_positions[i] == 0:
                    continue
                gold = answer_texts[i]
                if not gold:
                    continue

                start_pred = start_logits[i].argmax().item()
                end_scores = end_logits[i].clone()
                end_scores[:start_pred] = float("-inf")
                end_pred = end_scores.argmax().item()

                pred_ids  = input_ids[i][start_pred: end_pred + 1].tolist()
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

                em_scores.append(float(normalize_answer(pred_text) == normalize_answer(gold)))
                f1_scores.append(f1_score(pred_text, gold))

    model.train()

    if not em_scores:
        print("  No evaluable features found.")
        return {}

    avg_em = 100.0 * sum(em_scores) / len(em_scores)
    avg_f1 = 100.0 * sum(f1_scores) / len(f1_scores)

    print(f"\nQA Results ({len(em_scores)} features with answer in window):")
    print(f"  Exact Match (EM): {avg_em:.2f}%")
    print(f"  F1 Score:         {avg_f1:.2f}%")

    return {"test/squad_em": round(avg_em, 4), "test/squad_f1": round(avg_f1, 4)}


# =============================================================================
# compute_metrics helpers
# =============================================================================

def make_classification_compute_metrics():
    """Return a compute_metrics fn for sequence classification (accuracy)."""
    import numpy as np
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}
    return compute_metrics


def make_qa_compute_metrics():
    """Return a compute_metrics fn for question answering (start/end accuracy + span EM)."""
    import numpy as np
    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        if isinstance(predictions, tuple):
            start_logits, end_logits = predictions[0], predictions[1]
        else:
            return {}
        if isinstance(label_ids, tuple):
            start_labels, end_labels = label_ids[0], label_ids[1]
        else:
            return {}
        pred_starts = np.argmax(start_logits, axis=-1)
        pred_ends   = np.argmax(end_logits,   axis=-1)
        start_acc = float((pred_starts == start_labels).mean())
        end_acc   = float((pred_ends   == end_labels).mean())
        exact_match = float(((pred_starts == start_labels) & (pred_ends == end_labels)).mean())
        return {"start_acc": start_acc, "end_acc": end_acc, "span_em": exact_match}
    return compute_metrics


# =============================================================================
# Trainers
# =============================================================================

class MrXLMRTrainer(Trainer):
    """
    HuggingFace Trainer subclass for MrXLMR.

    Combines task loss with PI-controller-weighted deletion loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = self._init_metrics()
        self.current_alpha = self.args.deletion_loss_weight
        self._pi_controller = PIController(
            target_rate=self.args.target_deletion_rate,
            kp=self.args.controller_p,
            ki=self.args.controller_i,
            alpha_0=self.args.deletion_loss_weight,
        ) if self.args.use_pi_controller else None
        _proc = getattr(self, "processing_class", None) or self.tokenizer
        self._pad_token_id = _proc.pad_token_id  # 1 for XLM-R
        self._rng = random.Random()

    def train(self, *args, **kwargs):
        import time
        self._train_start_time = time.time()
        result = super().train(*args, **kwargs)
        if not self.args.disable_wandb:
            import wandb
            if wandb.run is not None:
                wandb.config.update({
                    "model_type":           self.args.model_type,
                    "model_name":           self.args.model_name,
                    "task":                 self.args.task,
                    "delete_gate_layer":    self.args.delete_gate_layer,
                    "deletion_type":        self.args.deletion_type,
                    "sigmoid_mask_scale":   self.args.sigmoid_mask_scale,
                    "deletion_threshold":   self.args.deletion_threshold,
                    "use_softmax1":         self.args.use_softmax1,
                    "target_deletion_rate": self.args.target_deletion_rate,
                    "deletion_loss_weight": self.args.deletion_loss_weight,
                    "use_pi_controller":       self.args.use_pi_controller,
                    "hard_delete_train_prob":  self.args.hard_delete_train_prob,
                    "controller_p":            self.args.controller_p,
                    "controller_i":            self.args.controller_i,
                    "regularizer_delay":       self.args.regularizer_delay,
                    "dataset_name":            self.args.dataset_name,
                    "dataset_config":          self.args.dataset_config,
                    "max_seq_length":          self.args.max_seq_length,
                }, allow_val_change=True)
        return result

    def _init_metrics(self):
        return {
            "loss":                          [],
            "cross_entropy_loss":            [],
            "delete_gate_loss":              [],
            "accuracy":                      [],
            "percent_deleted_tokens":        [],
            "percent_non_pad_deleted_tokens":[],
            "delete_gate_average":           [],
            "delete_gate_std":               [],
            "delete_gate_max_value":         [],
            "delete_gate_min_value":         [],
            "delete_gate_loss_coeff":        [],
            "new_seq_len":                   [],
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Forward pass + combined loss: task_loss + α * deletion_loss."""
        input_ids = inputs.get("input_ids")
        use_hard_delete = (
            model.training
            and self.args.hard_delete_train_prob > 0.0
            and self._rng.random() < self.args.hard_delete_train_prob
        )
        outputs = model(**inputs, hard_delete=use_hard_delete)
        task_loss = outputs.loss

        delete_gate_output = getattr(outputs, "delete_gate_output", None)
        deletion_loss, percent_deleted, percent_deleted_all = compute_deletion_loss(
            delete_gate_output,
            input_ids,
            deletion_threshold=self.args.deletion_threshold,
            sigmoid_mask_scale=self.args.sigmoid_mask_scale,
            pad_token_id=self._pad_token_id,
        )

        actual_del_rate = percent_deleted / 100.0

        if (model.training
                and self._pi_controller is not None
                and self.state.global_step >= self.args.regularizer_delay):
            self.current_alpha = self._pi_controller.update(actual_del_rate)

        if self.state.global_step >= self.args.regularizer_delay:
            loss = task_loss + self.current_alpha * deletion_loss
        else:
            loss = task_loss

        self.metrics["loss"].append(loss.detach().item())
        self.metrics["cross_entropy_loss"].append(task_loss.detach().item())
        self.metrics["percent_deleted_tokens"].append(percent_deleted_all)
        self.metrics["percent_non_pad_deleted_tokens"].append(percent_deleted)

        if delete_gate_output is not None:
            gate_vals = delete_gate_output.squeeze(-1)
            self.metrics["delete_gate_average"].append(gate_vals.mean().detach().item())
            if model.training:
                del_loss_val = deletion_loss.item() if isinstance(deletion_loss, torch.Tensor) else float(deletion_loss)
                self.metrics["delete_gate_loss"].append(del_loss_val)
                self.metrics["delete_gate_std"].append(gate_vals.std(dim=1).mean().detach().item())
                self.metrics["delete_gate_max_value"].append(gate_vals.max(dim=1).values.mean().detach().item())
                self.metrics["delete_gate_min_value"].append(gate_vals.min(dim=1).values.mean().detach().item())
                self.metrics["delete_gate_loss_coeff"].append(self.current_alpha)

        delete_gate_mask = getattr(outputs, "delete_gate_mask", None)
        if delete_gate_mask is not None:
            kept = (delete_gate_mask.squeeze(-1) > self.args.deletion_threshold).float()
            self.metrics["new_seq_len"].append(kept.sum(dim=1).mean().item())
        elif input_ids is not None:
            self.metrics["new_seq_len"].append(
                (input_ids != self._pad_token_id).float().sum(dim=1).mean().item()
            )

        if hasattr(outputs, "logits") and outputs.logits is not None and "labels" in inputs:
            preds = outputs.logits.argmax(dim=-1)
            acc = (preds == inputs["labels"]).float().mean().item()
            self.metrics["accuracy"].append(acc)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        import time
        aggregated = {
            k: round(statistics.fmean(v), 4)
            for k, v in self.metrics.items() if v
        }
        logs.update(aggregated)

        if "new_seq_len" in aggregated and self.args.max_seq_length > 0:
            logs["seq_len_reduction_pct"] = round(
                (1.0 - aggregated["new_seq_len"] / self.args.max_seq_length) * 100.0, 2
            )

        total_epochs = self.args.num_train_epochs
        current_epoch = self.state.epoch or 0.0
        logs["epoch_progress"] = f"Epoch {current_epoch:.2f}/{total_epochs}"
        elapsed = time.time() - getattr(self, "_train_start_time", time.time())
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        logs["elapsed_time"] = f"{h}h {m:02d}m {s:02d}s"

        self.metrics = self._init_metrics()
        self._last_logs = dict(logs)
        super().log(logs, *args, **kwargs)


class XLMRTrainer(Trainer):
    """
    HuggingFace Trainer for baseline XLM-R (no delete gate).
    Tracks cross-entropy loss and accuracy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = self._init_metrics()

    def _init_metrics(self):
        return {
            "cross_entropy_loss": [],
            "accuracy":           [],
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        self.metrics["cross_entropy_loss"].append(loss.detach().item())

        if hasattr(outputs, "logits") and outputs.logits is not None and "labels" in inputs:
            preds = outputs.logits.argmax(dim=-1)
            self.metrics["accuracy"].append(
                (preds == inputs["labels"]).float().mean().item()
            )

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        aggregated = {
            k: round(statistics.fmean(v), 4)
            for k, v in self.metrics.items() if v
        }
        logs.update(aggregated)
        self.metrics = self._init_metrics()
        self._last_logs = dict(logs)
        super().log(logs, *args, **kwargs)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = HfArgumentParser(MrXLMRTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print(f"Model: {args.model_type}  Task: {args.task.upper()}")
    print("=" * 60)
    print(f"Device: {args.device}")
    if args.model_type == "MrXLMR":
        print(f"Delete gate layer: {args.delete_gate_layer}")
        print(f"Deletion type:     {args.deletion_type}")
        print(f"Target del. rate:  {args.target_deletion_rate}")
    print()

    # Load XLM-R tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)

    # Prepare datasets
    num_labels = None
    if args.task == "mlm":
        tokenized_dataset, data_collator = prepare_mlm_dataset(args, tokenizer)
    elif args.task == "sequence_classification":
        tokenized_dataset, data_collator, num_labels = prepare_sequence_classification_dataset(args, tokenizer)
    elif args.task == "token_classification":
        tokenized_dataset, data_collator, num_labels = prepare_token_classification_dataset(args, tokenizer)
    elif args.task == "question_answering":
        tokenized_dataset, data_collator = prepare_question_answering_dataset(args, tokenizer)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print(f"Creating {args.model_type} model from pretrained '{args.model_name}'...")
    model = create_model(args, tokenizer, num_labels)
    model.to(args.device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    from diagnostics import log_parameter_summary
    log_parameter_summary(model, label=args.model_type)

    train_dataset = tokenized_dataset["train"]

    eval_dataset = None
    for split in ("validation", "test"):
        try:
            eval_dataset = tokenized_dataset[split]
            break
        except (KeyError, TypeError):
            pass

    test_dataset = None
    try:
        test_split = "test_matched" if (args.dataset_name == "glue" and args.dataset_config == "mnli") else "test"
        test_dataset = tokenized_dataset[test_split]
    except (KeyError, TypeError):
        pass

    if args.max_train_samples is not None and train_dataset is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
        print(f"Truncated train set to {len(train_dataset)} examples (--max_train_samples)")
    if args.max_eval_samples is not None:
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
        if test_dataset is not None:
            test_dataset = test_dataset.select(range(min(args.max_eval_samples, len(test_dataset))))

    TrainerClass = MrXLMRTrainer if args.model_type == "MrXLMR" else XLMRTrainer

    if args.task == "question_answering":
        _compute_metrics = make_qa_compute_metrics()
    else:
        _compute_metrics = make_classification_compute_metrics()

    import inspect
    _tokenizer_kwarg = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )

    trainer = TrainerClass(
        model=model,
        args=args,
        train_dataset=train_dataset if args.mode in ("training-only", "training-and-eval") else None,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
        data_collator=data_collator,
        **{_tokenizer_kwarg: tokenizer},
    )

    if args.mode in ("training-only", "training-and-eval"):
        print("\nStarting training...")
        print("-" * 60)
        trainer.train()
        print("\nTraining complete!")

        final_dir = os.path.join(args.output_dir, "final")
        print(f"\nSaving final model to {final_dir} ...")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Final model saved -> {final_dir}")
        print("=" * 60)

        if test_dataset is not None:
            print("\nRunning final evaluation on test set...")
            print("=" * 60)
            test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

            custom_keys = {
                "accuracy", "cross_entropy_loss", "delete_gate_loss",
                "percent_deleted_tokens", "percent_non_pad_deleted_tokens",
                "delete_gate_average", "new_seq_len", "seq_len_reduction_pct",
            }
            last_logs = getattr(trainer, "_last_logs", {})
            test_summary = {
                f"test/{k}": v for k, v in last_logs.items()
                if k in custom_keys
            }
            if test_summary and not args.disable_wandb:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary.update(test_summary)
                    print(f"Logged {len(test_summary)} test metrics to W&B summary.")

            print("\nTest metrics:")
            for k, v in test_metrics.items():
                print(f"  {k}: {v}")

        # QA: additionally compute EM/F1
        if args.task == "question_answering" and test_dataset is not None:
            qa_metrics = compute_squad_em_f1(args, model, test_dataset, tokenizer)
            if qa_metrics and not args.disable_wandb:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary.update(qa_metrics)

    elif args.mode == "eval-only":
        print("\nRunning evaluation...")
        metrics = trainer.evaluate(eval_dataset=eval_dataset or test_dataset, metric_key_prefix="eval")
        print("\nEval metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Print deletion samples for MrXLMR
    if args.model_type == "MrXLMR" and (eval_dataset or test_dataset) is not None:
        sample_dataset = eval_dataset or test_dataset
        sample_loader = DataLoader(
            sample_dataset,
            batch_size=8,
            collate_fn=data_collator or DefaultDataCollator(),
        )
        print_deletion_samples(args, model, sample_loader, tokenizer, n_samples=10)


if __name__ == "__main__":
    main()