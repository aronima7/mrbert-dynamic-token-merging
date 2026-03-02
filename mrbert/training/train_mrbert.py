#!/usr/bin/env python3
"""
Fine-tuning script for MrBERT using HuggingFace Trainer.

Supports: MLM, sequence classification, token classification, question answering.

Usage:
    # MLM
    python train_mrbert.py --task mlm --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
        --output_dir ./mrbert_checkpoints

    # Sequence Classification (SST-2)
    python train_mrbert.py --task sequence_classification --dataset_name glue --dataset_config sst2 \
        --output_dir ./mrbert_sst2

    # Token Classification (CoNLL-2003 NER)
    python train_mrbert.py --task token_classification --dataset_name conll2003 \
        --output_dir ./mrbert_ner

    # Quick smoke test
    python train_mrbert.py --max_steps 100 --logging_steps 10 --output_dir ./test_run

HuggingFace TrainingArguments flags apply directly, e.g.:
    --per_device_train_batch_size 16
    --num_train_epochs 3
    --eval_steps 500
    --save_steps 1000
    --learning_rate 5e-5
"""

import sys
import os
import statistics

# Add the models/ and training/ directories to the path so local modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
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
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset

from configuration_mrbert import MrBertConfig
from modeling_mrbert import (
    MrBertForMaskedLM,
    MrBertForSequenceClassification,
    MrBertForTokenClassification,
    MrBertForQuestionAnswering,
)
from pi_controller import PIController


# =============================================================================
# Training Arguments
# =============================================================================

@dataclass
class MrBertTrainingArguments(TrainingArguments):
    """
    HuggingFace TrainingArguments extended with MrBERT-specific fields.

    All standard TrainingArguments flags (--learning_rate, --num_train_epochs,
    --per_device_train_batch_size, --eval_steps, etc.) are available alongside
    the custom fields below.
    """

    # ---- Model ----
    model_type: str = field(
        default="MrBERT",
        metadata={"help": "MrBERT (with delete gate) or BERT (baseline, no gate)"},
    )
    model_name: str = field(
        default="bert-base-uncased",
        metadata={"help": "Pretrained BERT model to initialise from"},
    )
    delete_gate_layer: int = field(
        default=3,
        metadata={"help": "Encoder layer that emits the delete gate (0-indexed)"},
    )
    deletion_type: str = field(
        default="scaled_sigmoid",
        metadata={"help": "Gate type: scaled_sigmoid | log_sigmoid | random | fixed"},
    )
    use_softmax1: bool = field(
        default=True,
        metadata={"help": "Use softmax1 (n+1 denominator) for attention (recommended by MrT5 paper)"},
    )
    sigmoid_mask_scale: float = field(
        default=-30.0,
        metadata={"help": "Sigmoid mask scale applied to deleted tokens' attention scores (more negative = stronger deletion signal)"},
    )
    deletion_threshold: float = field(
        default=-15.0,
        metadata={"help": "Gate value below which a token is counted as deleted (= sigmoid_mask_scale / 2 by default)"},
    )

    # ---- Dataset ----
    task: str = field(
        default="mlm",
        metadata={"help": "mlm | sequence_classification | token_classification | question_answering"},
    )
    dataset_name: str = field(
        default="wikitext",
        metadata={"help": "HuggingFace dataset name (or 'local_mc4' / 'local_snli' for local datasets)"},
    )
    dataset_config: Optional[str] = field(
        default="wikitext-2-raw-v1",
        metadata={"help": "Dataset config / subset name (e.g. 'sst2' for GLUE)"},
    )
    local_mc4_dir: str = field(
        default="mrt5/lm_datasets",
        metadata={"help": "Directory with local mC4 pre-processed files (used when dataset_name='local_mc4')"},
    )
    local_snli_dir: str = field(
        default="snli_datasets",
        metadata={"help": "Directory with local SNLI NDJSON files (used when dataset_name='local_snli')"},
    )
    local_squad_dir: str = field(
        default="squad_datasets",
        metadata={"help": "Directory with local SQuAD NDJSON files (used when dataset_name='local_squad')"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum token sequence length; inputs are truncated/padded to this length"},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Fraction of tokens randomly masked for MLM pre-training"},
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Stream large datasets rather than loading them fully into memory"},
    )

    # ---- Deletion loss / PI controller ----
    # The deletion loss encourages the gate to delete a target fraction of tokens.
    # A PI controller dynamically adjusts the loss coefficient α so the observed
    # deletion rate tracks the target rate (Section 3.2 of the MrT5 paper).
    deletion_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Initial deletion loss coefficient α₀ (PI controller adjusts this during training)"},
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
        metadata={"help": "Number of steps to train on task loss only before switching on the deletion regulariser"},
    )
    use_pi_controller: bool = field(
        default=True,
        metadata={"help": "Dynamically adjust deletion loss weight α with a PI controller to hit target_deletion_rate"},
    )

    # ---- W&B ----
    wandb_project: str = field(
        default="mrbert",
        metadata={"help": "W&B project name"},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "W&B run name (default: auto-generated from model_type, model_name and seed)"},
    )
    disable_wandb: bool = field(
        default=False,
        metadata={"help": "Disable W&B logging entirely"},
    )

    # ---- Backward-compatible aliases ----
    # These map old CLI flag names to the standard TrainingArguments equivalents.
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
        # Resolve backward-compat aliases before parent validation
        if self.batch_size is not None:
            self.per_device_train_batch_size = self.batch_size
            self.per_device_eval_batch_size = self.batch_size
        if self.num_epochs is not None:
            self.num_train_epochs = float(self.num_epochs)

        # Default to step-based periodic evaluation so validation runs every eval_steps
        # (TrainingArguments defaults evaluation_strategy to "no")
        if getattr(self, "evaluation_strategy", "no") == "no":
            self.evaluation_strategy = "steps"

        # Configure W&B reporting
        if self.disable_wandb:
            self.report_to = ["none"]
        else:
            self.report_to = ["wandb"]
            os.environ["WANDB_PROJECT"] = self.wandb_project
            # Auto-generate a run name if not provided
            run_name = self.wandb_run_name or (
                f"{self.model_type}_{self.model_name.split('/')[-1]}_seed{self.seed}"
            )
            self.run_name = run_name

        super().__post_init__()


# =============================================================================
# Deletion Loss Utility
# =============================================================================

def compute_deletion_loss(delete_gate_output, input_ids, deletion_threshold, sigmoid_mask_scale, pad_token_id=0):
    """
    Compute the deletion regularisation loss and deletion statistics.

    The loss is the mean gate value over non-pad tokens. Minimising this
    encourages the gate to produce lower (more negative) values, which
    increases the deletion signal applied to attention scores.

    Args:
        delete_gate_output: Gate values from the model, shape (batch, seq_len, 1).
        input_ids: Token IDs used to identify pad positions.
        deletion_threshold: Gate value below which a token is considered deleted.
        sigmoid_mask_scale: Unused; kept for API compatibility.
        pad_token_id: Token ID for padding.

    Returns:
        deletion_loss: Scalar tensor (mean gate value over non-pad tokens).
        percent_deleted_non_pad: % of non-pad tokens with gate < threshold.
        percent_deleted_all: % of all tokens (including pad) with gate < threshold.
    """
    if delete_gate_output is None:
        return torch.tensor(0.0), 0.0, 0.0

    delete_gate_output = delete_gate_output.squeeze(-1)
    non_pad_mask = input_ids != pad_token_id

    # Gate mean over non-pad tokens (pad tokens are excluded to avoid biasing the loss)
    if non_pad_mask.any():
        deletion_loss = delete_gate_output[non_pad_mask].mean()
    else:
        deletion_loss = delete_gate_output.mean()

    # Count deleted non-pad tokens
    num_non_pad_tokens = non_pad_mask.sum()
    num_deleted = ((delete_gate_output < deletion_threshold) & non_pad_mask).sum()
    percent_deleted_non_pad = (num_deleted / num_non_pad_tokens * 100).item() if num_non_pad_tokens > 0 else 0.0

    # Count deleted tokens across the full sequence (including pad)
    total_tokens = delete_gate_output.numel()
    num_deleted_all = (delete_gate_output < deletion_threshold).sum()
    percent_deleted_all = (num_deleted_all / total_tokens * 100).item() if total_tokens > 0 else 0.0

    return deletion_loss, percent_deleted_non_pad, percent_deleted_all


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_mlm_dataset(args, tokenizer):
    """
    Prepare dataset for Masked Language Modeling.

    Supports HuggingFace datasets (e.g. wikitext) and a local mC4 dataset.
    Returns (dataset_dict, data_collator). The collator randomly masks tokens
    at mlm_probability during batching.
    """

    if args.dataset_name == "local_mc4":
        # Local mC4 dataset pre-processed into tokenized chunks
        print(f"Loading LOCAL mC4 dataset from: {args.local_mc4_dir}")
        from mc4_dataset import load_mc4_dataset

        train_dataset = load_mc4_dataset(
            split="train", tokenizer=tokenizer,
            max_length=args.max_seq_length, mlm_probability=args.mlm_probability,
            streaming=args.streaming, data_dir=args.local_mc4_dir,
        )
        eval_dataset = load_mc4_dataset(
            split="validation", tokenizer=tokenizer,
            max_length=args.max_seq_length, mlm_probability=args.mlm_probability,
            streaming=False, max_samples=1000, data_dir=args.local_mc4_dir,
        )

        # Wrap in a simple dict-like object so callers can use dataset["train"] etc.
        class DatasetDict:
            def __init__(self, train, validation):
                self._d = {"train": train, "validation": validation}
            def __getitem__(self, key):
                if key == "test":
                    return self._d["validation"]
                return self._d[key]

        # The mc4 dataset already applies masking, so no collator is needed
        return DatasetDict(train_dataset, eval_dataset), None

    print(f"Loading MLM dataset: {args.dataset_name}/{args.dataset_config}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)

    text_column = "text" if "text" in dataset["train"].column_names else dataset["train"].column_names[0]

    def tokenize_function(examples):
        # Filter out empty strings before tokenizing
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

    # DataCollatorForLanguageModeling applies random masking at batch time
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
    )
    return tokenized_dataset, data_collator


def prepare_sequence_classification_dataset(args, tokenizer):
    """
    Prepare dataset for Sequence Classification.

    Supports GLUE tasks (sst2, mnli, etc.), local SNLI, and generic datasets.
    Returns (dataset_dict, data_collator, num_labels).
    """

    if args.dataset_name == "local_snli":
        # Pre-tokenized SNLI stored as NDJSON; input_ids shape is [1, seq_len] so we unwrap
        print(f"Loading LOCAL SNLI dataset from: {args.local_snli_dir}")
        dataset = load_dataset(
            "json",
            data_files={
                "train":      f"{args.local_snli_dir}/snli-train.json",
                "validation": f"{args.local_snli_dir}/snli-validation.json",
                "test":       f"{args.local_snli_dir}/snli-test.json",
            },
        )
        # Unwrap the outer list dimension added by the preprocessing script
        dataset = dataset.map(
            lambda x: {
                "input_ids":      x["input_ids"][0],
                "attention_mask": x["attention_mask"][0],
                "labels":         x["labels"],
            },
            desc="Unwrapping precomputed features",
        )
        return dataset, DefaultDataCollator(), 3  # 3 labels: entailment, neutral, contradiction

    print(f"Loading classification dataset: {args.dataset_name}/{args.dataset_config}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)

    # Determine which column(s) contain the input text, based on dataset/config
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
        # Generic fallback: look for common column names
        cols = dataset["train"].column_names
        text_columns = ["text"] if "text" in cols else (["sentence"] if "sentence" in cols else [cols[0]])

    label_column = "label"

    def tokenize_function(examples):
        # Tokenize single-sentence or sentence-pair inputs
        if len(text_columns) == 1:
            return tokenizer(examples[text_columns[0]], truncation=True,
                             max_length=args.max_seq_length, padding="max_length")
        return tokenizer(examples[text_columns[0]], examples[text_columns[1]],
                         truncation=True, max_length=args.max_seq_length, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, desc="Tokenizing")
    # Rename "label" → "labels" to match what the model expects
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

    # Infer label list from dataset features
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
                    # Special tokens ([CLS], [SEP], [PAD]) → ignored
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word → assign the real label
                    label_ids.append(label[word_idx] if word_idx < len(label) else -100)
                else:
                    # Subsequent subwords of the same word → ignored
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, desc="Tokenizing")
    return tokenized_dataset, DataCollatorForTokenClassification(tokenizer), num_labels


def prepare_question_answering_dataset(args, tokenizer):
    """
    Prepare dataset for extractive Question Answering (e.g. SQuAD).

    Supports pre-tokenized local SQuAD (dataset_name='local_squad') and
    live loading from HuggingFace Hub (dataset_name='rajpurkar/squad' etc.).

    Long contexts are split into overlapping windows with stride=128.
    Start/end positions are mapped from character offsets to token indices.
    Returns (dataset_dict, data_collator).
    """

    if args.dataset_name == "local_squad":
        # Pre-tokenized SQuAD stored as NDJSON by preprocess_squad.py.
        # Validation split is reused as test (SQuAD has no public test set).
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

    print(f"Loading QA dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second",       # only truncate context, not question
            max_length=args.max_seq_length,
            stride=128,                     # overlap between windows for long contexts
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
                # No answer → point to [CLS]
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                # Find the token span that covers the character-level answer span
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    # Answer is outside this window → point to [CLS]
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

    MrBERT: loads pretrained BERT weights via from_pretrained, then randomly
            initialises only the delete gate (bias=10, weight_std=0.001).
            All other encoder weights are identical to the BERT baseline,
            making the comparison fair.
    BERT:   loads pretrained weights from HuggingFace Hub via from_pretrained.
    """
    if args.model_type == "MrBERT":
        # Build MrBertConfig from a pretrained BERT config, injecting gate params
        config = MrBertConfig.from_pretrained(
            args.model_name,
            deletion_type=args.deletion_type,
            delete_gate_layer=args.delete_gate_layer,
            sigmoid_mask_scale=args.sigmoid_mask_scale,
            use_gumbel_noise=True,       # adds exploration noise to the gate during training
            use_softmax1=args.use_softmax1,
        )
        # Load pretrained BERT weights; the delete gate is absent from the checkpoint
        # so it gets randomly initialised by _init_delete_gates() (bias=10, weight_std=0.001).
        # ignore_mismatched_sizes=True suppresses the warning about the gate being new.
        if args.task == "mlm":
            return MrBertForMaskedLM.from_pretrained(
                args.model_name, config=config, ignore_mismatched_sizes=True)
        elif args.task == "sequence_classification":
            config.num_labels = num_labels
            return MrBertForSequenceClassification.from_pretrained(
                args.model_name, config=config, ignore_mismatched_sizes=True)
        elif args.task == "token_classification":
            config.num_labels = num_labels
            return MrBertForTokenClassification.from_pretrained(
                args.model_name, config=config, ignore_mismatched_sizes=True)
        elif args.task == "question_answering":
            return MrBertForQuestionAnswering.from_pretrained(
                args.model_name, config=config, ignore_mismatched_sizes=True)

    elif args.model_type == "BERT":
        # Standard BERT loaded with pretrained weights
        config = BertConfig.from_pretrained(args.model_name)
        if args.task == "mlm":
            return BertForMaskedLM.from_pretrained(args.model_name, config=config)
        elif args.task == "sequence_classification":
            config.num_labels = num_labels
            return BertForSequenceClassification.from_pretrained(args.model_name, config=config)
        elif args.task == "token_classification":
            config.num_labels = num_labels
            return BertForTokenClassification.from_pretrained(args.model_name, config=config)
        elif args.task == "question_answering":
            return BertForQuestionAnswering.from_pretrained(args.model_name, config=config)

    raise ValueError(f"Unknown model_type={args.model_type} / task={args.task}")


# =============================================================================
# Deletion Sample Diagnostics
# =============================================================================

# Label map used for SNLI classification samples
SNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def print_deletion_samples(args, model, dataloader, tokenizer, n_samples=20, split="TEST"):
    """
    Print the first n_samples examples showing which tokens the gate kept vs. deleted.

    Deleted tokens are shown in [brackets]. For SNLI, the ground-truth and
    predicted labels are shown alongside a ✓/✗ correctness indicator.
    """
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
            # delete_gate_mask contains the raw gate values (pre-threshold)
            delete_gate_mask = getattr(outputs, "delete_gate_mask", None)
            logits = getattr(outputs, "logits", None)

            for i in range(input_ids.size(0)):
                if samples_printed >= n_samples:
                    break
                ids = input_ids[i].tolist()
                tokens = tokenizer.convert_ids_to_tokens(ids)

                # Determine which tokens are deleted based on gate threshold
                if delete_gate_mask is not None:
                    gate_vals = delete_gate_mask[i].squeeze(-1).tolist()
                    deleted = [gate_vals[j] < args.deletion_threshold for j in range(len(tokens))]
                else:
                    deleted = [False] * len(tokens)

                # Strip padding tokens from the display
                pad_id = tokenizer.pad_token_id
                non_pad = [(tok, d, ids[j]) for j, (tok, d) in enumerate(zip(tokens, deleted)) if ids[j] != pad_id]
                kept_tokens    = [tok for tok, d, _ in non_pad if not d]
                deleted_tokens = [tok for tok, d, _ in non_pad if d]

                # Prediction vs. ground truth (SNLI labels if available)
                pred_label = SNLI_LABELS.get(logits[i].argmax().item(), "?") if logits is not None else "?"
                true_label = SNLI_LABELS.get(labels[i].item(), "?") if labels is not None else "?"
                correct = "✓" if pred_label == true_label else "✗"

                # Build annotated sequence: deleted tokens wrapped in [brackets],
                # [CLS]/[SEP] shown as-is
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
    """
    Compute Exact Match (EM) and F1 for SQuAD QA evaluation.

    Iterates through test_dataset in batches, predicts answer spans from
    start_logits / end_logits, decodes them to strings, and compares against
    the stored answer_text using the standard SQuAD normalization rules.

    Only features where the answer falls within the context window
    (start_positions != 0) are scored — features where the answer was
    outside the window were assigned CLS (position 0) during preprocessing
    and cannot be evaluated fairly.

    Args:
        args:         MrBertTrainingArguments (uses device, per_device_eval_batch_size).
        model:        Trained QA model.
        test_dataset: HuggingFace Dataset with input_ids, attention_mask,
                      start_positions, end_positions, answer_text columns.
        tokenizer:    BertTokenizer for decoding predicted spans.

    Returns:
        dict with "test/squad_em" and "test/squad_f1" (both as percentages).
    """
    import re
    import string
    from collections import Counter

    def normalize_answer(s):
        """Lowercase, remove articles, punctuation and extra whitespace."""
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
        print("  Re-run preprocess_squad.py to regenerate the dataset with answer text.")
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
            start_logits = outputs.start_logits  # (batch, seq_len)
            end_logits   = outputs.end_logits    # (batch, seq_len)

            for i in range(len(answer_texts)):
                # Skip features where the answer was outside this context window
                if start_positions[i] == 0:
                    continue
                gold = answer_texts[i]
                if not gold:
                    continue

                # Predict start; then find best end position at or after start
                start_pred = start_logits[i].argmax().item()
                end_scores = end_logits[i].clone()
                end_scores[:start_pred] = float("-inf")
                end_pred = end_scores.argmax().item()

                # Decode predicted token span back to a string
                pred_ids  = input_ids[i][start_pred: end_pred + 1].tolist()
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

                em_scores.append(float(normalize_answer(pred_text) == normalize_answer(gold)))
                f1_scores.append(f1_score(pred_text, gold))

    model.train()

    if not em_scores:
        print("  No evaluable features found (all features had CLS start position).")
        return {}

    avg_em = 100.0 * sum(em_scores) / len(em_scores)
    avg_f1 = 100.0 * sum(f1_scores) / len(f1_scores)

    print(f"\nSQuAD Results ({len(em_scores)} features with answer in window):")
    print(f"  Exact Match (EM): {avg_em:.2f}%")
    print(f"  F1 Score:         {avg_f1:.2f}%")

    return {"test/squad_em": round(avg_em, 4), "test/squad_f1": round(avg_f1, 4)}


# =============================================================================
# Trainers
# =============================================================================

class MrBertTrainer(Trainer):
    """
    HuggingFace Trainer subclass for MrBERT.

    Key overrides:
      - train:        records training start time; patches W&B config after init.
      - compute_loss: combines task loss with a PI-controller-weighted deletion loss.
      - log:          averages per-step metrics and injects them into the log dict.

    Metrics are accumulated in self.metrics across steps and flushed every
    logging_steps via log(). The train/ prefix is used during training and
    eval/ during evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = self._init_metrics()
        # α starts at deletion_loss_weight and is updated by the PI controller
        self.current_alpha = self.args.deletion_loss_weight
        # Instantiate the PI controller if enabled
        self._pi_controller = PIController(
            target_rate=self.args.target_deletion_rate,
            kp=self.args.controller_p,
            ki=self.args.controller_i,
            alpha_0=self.args.deletion_loss_weight,
        ) if self.args.use_pi_controller else None
        # Cache pad token ID to avoid accessing self.tokenizer in the hot path,
        # which triggers a deprecation warning on every step in newer Transformers.
        _proc = getattr(self, "processing_class", None) or self.tokenizer
        self._pad_token_id = _proc.pad_token_id

    def train(self, *args, **kwargs):
        import time
        self._train_start_time = time.time()
        result = super().train(*args, **kwargs)
        # W&B is initialised by WandbCallback inside the first training step,
        # so we patch the run config here, after training, when wandb.run is live.
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
                    "use_pi_controller":    self.args.use_pi_controller,
                    "controller_p":         self.args.controller_p,
                    "controller_i":         self.args.controller_i,
                    "regularizer_delay":    self.args.regularizer_delay,
                    "dataset_name":         self.args.dataset_name,
                    "dataset_config":       self.args.dataset_config,
                    "max_seq_length":       self.args.max_seq_length,
                }, allow_val_change=True)
        return result

    # ------------------------------------------------------------------
    # Metrics bookkeeping
    # ------------------------------------------------------------------

    def _init_metrics(self):
        """Return a fresh dict of empty metric lists for one logging interval."""
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
            "delete_gate_loss_coeff":        [],  # current α from PI controller
            "new_seq_len":                   [],  # effective seq length after deletion
        }

    # ------------------------------------------------------------------
    # Loss: task loss + α * deletion loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Forward pass + combined loss computation.

        total_loss = cross_entropy_loss + α * deletion_loss

        α is updated each step by the PI controller (if enabled) to drive the
        observed deletion rate toward target_deletion_rate. Before regularizer_delay
        steps, only the task loss is used (deletion regulariser is off).
        """
        input_ids = inputs.get("input_ids")
        outputs = model(**inputs)
        task_loss = outputs.loss

        # Compute deletion loss and deletion rate statistics
        delete_gate_output = getattr(outputs, "delete_gate_output", None)
        deletion_loss, percent_deleted, percent_deleted_all = compute_deletion_loss(
            delete_gate_output,
            input_ids,
            deletion_threshold=self.args.deletion_threshold,
            sigmoid_mask_scale=self.args.sigmoid_mask_scale,
            pad_token_id=self._pad_token_id,
        )

        actual_del_rate = percent_deleted / 100.0

        # Update PI controller during training only, after the regulariser delay
        if (model.training
                and self._pi_controller is not None
                and self.state.global_step >= self.args.regularizer_delay):
            self.current_alpha = self._pi_controller.update(actual_del_rate)

        # Apply deletion regulariser only after the delay period
        if self.state.global_step >= self.args.regularizer_delay:
            loss = task_loss + self.current_alpha * deletion_loss
        else:
            loss = task_loss

        # Accumulate metrics (no prefix — W&B applies train/ or eval/ grouping automatically)
        self.metrics["loss"].append(loss.detach().item())
        self.metrics["cross_entropy_loss"].append(task_loss.detach().item())
        self.metrics["percent_deleted_tokens"].append(percent_deleted_all)
        self.metrics["percent_non_pad_deleted_tokens"].append(percent_deleted)

        if delete_gate_output is not None:
            gate_vals = delete_gate_output.squeeze(-1)
            self.metrics["delete_gate_average"].append(gate_vals.mean().detach().item())
            if model.training:
                # Extra gate distribution stats are only tracked during training
                del_loss_val = deletion_loss.item() if isinstance(deletion_loss, torch.Tensor) else float(deletion_loss)
                self.metrics["delete_gate_loss"].append(del_loss_val)
                self.metrics["delete_gate_std"].append(gate_vals.std(dim=1).mean().detach().item())
                self.metrics["delete_gate_max_value"].append(gate_vals.max(dim=1).values.mean().detach().item())
                self.metrics["delete_gate_min_value"].append(gate_vals.min(dim=1).values.mean().detach().item())
                self.metrics["delete_gate_loss_coeff"].append(self.current_alpha)

        # Effective sequence length: tokens with gate > threshold (or all non-pad for BERT)
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

    # ------------------------------------------------------------------
    # Logging: flush accumulated metrics then delegate to parent
    # ------------------------------------------------------------------

    def log(self, logs, *args, **kwargs):
        """
        Average all accumulated metrics and inject them into the log dict,
        then call the parent log() which handles W&B, console output, etc.
        Metrics are reset after each log call.
        """
        import time
        # Average each metric list; skip empty lists (metric wasn't hit this interval)
        aggregated = {
            k: round(statistics.fmean(v), 4)
            for k, v in self.metrics.items() if v
        }
        logs.update(aggregated)

        # Epoch progress (fractional) and wall-clock elapsed time
        total_epochs = self.args.num_train_epochs
        current_epoch = self.state.epoch or 0.0
        logs["epoch_progress"] = f"Epoch {current_epoch:.2f}/{total_epochs}"
        elapsed = time.time() - getattr(self, "_train_start_time", time.time())
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        logs["elapsed_time"] = f"{h}h {m:02d}m {s:02d}s"

        self.metrics = self._init_metrics()
        super().log(logs, *args, **kwargs)


class BertTrainer(Trainer):
    """
    HuggingFace Trainer for baseline BERT (no delete gate).

    Tracks cross-entropy loss and accuracy for train and eval splits.
    All other training behaviour (optimizer, scheduler, checkpointing, W&B)
    is handled by the parent Trainer class.
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
        super().log(logs, *args, **kwargs)


# =============================================================================
# Main
# =============================================================================

def main():
    # Parse all arguments from the command line via the MrBertTrainingArguments dataclass
    parser = HfArgumentParser(MrBertTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print(f"Model: {args.model_type}  Task: {args.task.upper()}")
    print("=" * 60)
    print(f"Device: {args.device}")
    if args.model_type == "MrBERT":
        print(f"Delete gate layer: {args.delete_gate_layer}")
        print(f"Deletion type:     {args.deletion_type}")
        print(f"Target del. rate:  {args.target_deletion_rate}")
    print()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Prepare tokenized datasets and data collator based on task
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

    # Build model (MrBERT loads pretrained BERT weights + randomly inits the delete gate)
    print(f"Creating {args.model_type} model from pretrained '{args.model_name}'...")
    model = create_model(args, tokenizer, num_labels)
    model.to(args.device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    from diagnostics import log_parameter_summary
    log_parameter_summary(model, label=args.model_type)

    # Extract dataset splits
    train_dataset = tokenized_dataset["train"]

    # Use "validation" split for periodic eval during training; fall back to "test"
    eval_dataset = None
    for split in ("validation", "test"):
        try:
            eval_dataset = tokenized_dataset[split]
            break
        except (KeyError, TypeError):
            pass

    # Test split for final evaluation after training
    test_dataset = None
    try:
        # GLUE MNLI uses "test_matched" instead of "test"
        test_split = "test_matched" if (args.dataset_name == "glue" and args.dataset_config == "mnli") else "test"
        test_dataset = tokenized_dataset[test_split]
    except (KeyError, TypeError):
        pass

    # Select trainer class based on model type
    TrainerClass = MrBertTrainer if args.model_type == "MrBERT" else BertTrainer

    # Use processing_class (new API) if available, else fall back to tokenizer (old API)
    import inspect
    _tokenizer_kwarg = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )

    trainer = TrainerClass(
        model=model,
        args=args,
        # Only pass train_dataset when actually training
        train_dataset=train_dataset if args.mode in ("training-only", "training-and-eval") else None,
        eval_dataset=eval_dataset,   # used for periodic validation during training
        data_collator=data_collator,
        **{_tokenizer_kwarg: tokenizer},
    )

    # --- Train ---
    if args.mode in ("training-only", "training-and-eval"):
        print("\nStarting training...")
        print("-" * 60)
        trainer.train()
        print("\nTraining complete!")
        print("=" * 60)

        # Final test-set evaluation after training completes
        if test_dataset is not None:
            print("\nRunning final evaluation on test set...")
            print("=" * 60)
            # metric_key_prefix="test" prefixes all logged metrics with "test/"
            test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            print("\nTest set metrics:")
            for k, v in sorted(test_metrics.items()):
                print(f"  {k}: {v}")
            print("\nTest set evaluation complete!")
            print("=" * 60)
            # For QA, compute standard SQuAD EM and F1 metrics
            if args.task == "question_answering":
                compute_squad_em_f1(args, model, test_dataset, tokenizer)
            if args.model_type == "MrBERT":
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.per_device_eval_batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                )
                print_deletion_samples(args, model, test_loader, tokenizer, split="TEST")
        else:
            print("No test set available — skipping final test evaluation.")

        # Print deletion samples on the validation set (MrBERT only)
        if args.model_type == "MrBERT" and eval_dataset is not None:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
            print_deletion_samples(args, model, eval_loader, tokenizer, split="VALIDATION")

    # --- Eval-only ---
    if args.mode == "eval-only":
        if eval_dataset is None:
            print("No validation set available — skipping eval.")
        else:
            print("\nRunning evaluation...")
            trainer.evaluate()
            print("\nEval complete!")
            print("=" * 60)


if __name__ == "__main__":
    main()