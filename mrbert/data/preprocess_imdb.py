# preprocess_imdb.py
#
# Downloads IMDB and saves tokenized splits to NDJSON files for use with MrBERT
# and BERT sequence classification training.
#
# Since IMDB only has train/test splits (no validation), the training set is
# split 90/10 to create a dedicated validation set.
#
# Usage:
#   python preprocess_imdb.py                           # all splits
#   python preprocess_imdb.py --output_dir /path/to/imdb_datasets
#   python preprocess_imdb.py --max_samples 1000        # local sanity test
#   python preprocess_imdb.py --validation_split 0.1    # fraction for validation
#
# Output files:
#   <output_dir>/imdb-train.json       (90% of original train, ~22,500 examples)
#   <output_dir>/imdb-validation.json  (10% of original train, ~2,500 examples)
#   <output_dir>/imdb-test.json        (original test set, 25,000 examples)
#
# Each line is a JSON object with keys:
#   input_ids       : list[int]
#   attention_mask  : list[int]
#   labels          : int (0=negative, 1=positive)

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer


def preprocess_and_save(dataset_split, tokenizer, max_length, output_path):
    """Tokenize a dataset split and write examples to an NDJSON file."""
    total = 0
    with open(output_path, "w") as f:
        for example in tqdm(dataset_split, desc=f"Writing {os.path.basename(output_path)}"):
            tokenized = tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            json.dump({
                "input_ids":      tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels":         example["label"],
            }, f)
            f.write("\n")
            total += 1
    print(f"Saved {total} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IMDB for MrBERT.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="imdb_datasets",
        help="Directory to write output NDJSON files.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length. IMDB reviews can be long; 512 is the BERT max.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="BERT tokenizer to use for tokenization.",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of training data to hold out as validation (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split reproducibility.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap each original split at this many examples (useful for local sanity tests).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading IMDB dataset...")
    ds = load_dataset("stanfordnlp/imdb")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # --- Train / validation split ---
    train_data = ds["train"]
    if args.max_samples is not None:
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))

    split_result = train_data.train_test_split(
        test_size=args.validation_split, seed=args.seed
    )
    train_split = split_result["train"]
    val_split   = split_result["test"]

    print(f"\nTrain split:      {len(train_split)} examples")
    print(f"Validation split: {len(val_split)} examples")

    output_path = os.path.join(args.output_dir, "imdb-train.json")
    print(f"\nPreprocessing train split...")
    preprocess_and_save(train_split, tokenizer, args.max_length, output_path)

    output_path = os.path.join(args.output_dir, "imdb-validation.json")
    print(f"\nPreprocessing validation split...")
    preprocess_and_save(val_split, tokenizer, args.max_length, output_path)

    # --- Test split ---
    test_data = ds["test"]
    if args.max_samples is not None:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))

    output_path = os.path.join(args.output_dir, "imdb-test.json")
    print(f"\nPreprocessing test split ({len(test_data)} examples)...")
    preprocess_and_save(test_data, tokenizer, args.max_length, output_path)

    print("\nDone.")