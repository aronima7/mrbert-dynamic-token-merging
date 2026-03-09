# preprocess_sst2.py
#
# Downloads SST-2 (GLUE) and saves tokenized train/validation splits to NDJSON.
# Uses XLM-RoBERTa tokenizer.
#
# Usage:
#   python preprocess_sst2.py
#   python preprocess_sst2.py --output_dir /checkpoints/sst2_datasets
#
# Output files:
#   <output_dir>/sst2-train.json
#   <output_dir>/sst2-validation.json
#
# Each line is a JSON object with keys:
#   input_ids      : list[int]
#   attention_mask : list[int]
#   labels         : int  (0=negative, 1=positive)

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast


def preprocess_and_save(dataset_split, tokenizer, max_length, output_path):
    """Tokenize SST-2 and write to an NDJSON file."""
    with open(output_path, "w") as f:
        for example in tqdm(dataset_split, desc=f"Writing {os.path.basename(output_path)}"):
            tokenized = tokenizer(
                example["sentence"],
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
    parser = argparse.ArgumentParser(description="Preprocess SST-2 for MrXLMR.")

    parser.add_argument("--output_dir", type=str, default="sst2_datasets")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--model_name", type=str, default="xlm-roberta-base",
        help="XLM-R tokenizer to use.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SST-2 (GLUE) dataset...")
    ds = load_dataset("glue", "sst2")
    # SST-2 test labels are -1 (unlabeled); skip the test split
    splits = ["train", "validation"]

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)

    for split in splits:
        split_data = ds[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"sst2-{split}.json")
        print(f"\nPreprocessing {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, output_path)

    print("\nDone.")