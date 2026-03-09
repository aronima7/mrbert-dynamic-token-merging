# preprocess_imdb.py
#
# Downloads the IMDB sentiment dataset and saves tokenized splits to NDJSON.
# Uses XLM-RoBERTa tokenizer.
# Since IMDB has no validation split, we carve 10% from train (seed=42).
#
# Usage:
#   python preprocess_imdb.py
#   python preprocess_imdb.py --output_dir /checkpoints/imdb_datasets
#
# Output files:
#   <output_dir>/imdb-train.json
#   <output_dir>/imdb-validation.json
#   <output_dir>/imdb-test.json
#
# Each line is a JSON object with keys:
#   input_ids      : list[int]
#   attention_mask : list[int]
#   labels         : int  (0=negative, 1=positive)

import argparse
import json
import os

from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast


def preprocess_and_save(dataset_split, tokenizer, max_length, output_path):
    """Tokenize IMDB reviews and write to an NDJSON file."""
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

    print(f"Saved {len(dataset_split)} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IMDB for MrXLMR.")

    parser.add_argument("--output_dir", type=str, default="imdb_datasets")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max sequence length (IMDB reviews can be long; 256 is a good default).")
    parser.add_argument(
        "--model_name", type=str, default="xlm-roberta-base",
        help="XLM-R tokenizer to use.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading IMDB dataset...")
    ds = load_dataset("stanfordnlp/imdb")

    # IMDB has no validation split — carve 10% from train
    train_val = ds["train"].train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({
        "train":      train_val["train"],
        "validation": train_val["test"],
        "test":       ds["test"],
    })
    print(f"  train: {len(ds['train'])}  validation: {len(ds['validation'])}  test: {len(ds['test'])}")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)

    for split in ["train", "validation", "test"]:
        split_data = ds[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"imdb-{split}.json")
        print(f"\nPreprocessing {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, output_path)

    print("\nDone.")