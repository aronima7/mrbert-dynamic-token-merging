# preprocess_snli.py
#
# Downloads the Stanford NLI dataset and saves tokenized train/validation/test
# splits to NDJSON files, one example per line, matching the format used by
# mrt5/data/preprocess_diagnostic_dataset.py.
#
# Usage:
#   python preprocess_snli.py                        # all splits
#   python preprocess_snli.py --split train
#   python preprocess_snli.py --output_dir /checkpoints/snli_datasets
#
# Output files:
#   <output_dir>/snli-train.json
#   <output_dir>/snli-validation.json
#   <output_dir>/snli-test.json
#
# Each line is a JSON object with keys:
#   input_ids       : list[list[int]]  (batch_size=1, seq_len)
#   attention_mask  : list[list[int]]
#   labels          : int  (0=entailment, 1=neutral, 2=contradiction)

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def preprocess_and_save(dataset_split, tokenizer, max_length, output_path):
    """Tokenize a dataset split and write examples to an NDJSON file."""

    # Filter out unlabeled examples (label == -1)
    dataset_split = dataset_split.filter(lambda x: x["label"] != -1)

    def save():
        for example in tqdm(dataset_split, desc=f"Writing {os.path.basename(output_path)}"):
            tokenized = tokenizer(
                example["premise"],
                example["hypothesis"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            yield {
                "input_ids": [tokenized["input_ids"]],
                "attention_mask": [tokenized["attention_mask"]],
                "labels": example["label"],
            }

    with open(output_path, "w") as f:
        for item in save():
            json.dump(item, f)
            f.write("\n")

    print(f"Saved {len(dataset_split)} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SNLI dataset for MrBERT.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="snli_datasets",
        help="Directory to write output NDJSON files.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum tokenized sequence length (premise + hypothesis).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="BERT tokenizer to use for tokenization.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test"],
        help="Single split to preprocess. If omitted, all three splits are processed.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap each split at this many examples (useful for local sanity tests).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SNLI dataset...")
    ds = load_dataset("stanfordnlp/snli")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    splits = [args.split] if args.split is not None else ["train", "validation", "test"]

    for split in splits:
        split_data = ds[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"snli-{split}.json")
        print(f"\nPreprocessing {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, output_path)

    print("\nDone.")