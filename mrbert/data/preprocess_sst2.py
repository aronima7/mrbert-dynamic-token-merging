# preprocess_sst2.py
#
# Downloads GLUE SST-2 and saves tokenized train/validation splits to NDJSON
# files for use with MrBERT and BERT sequence classification training.
#
# Note: The GLUE SST-2 test split has no public labels (-1), so only
# train and validation splits are preprocessed.
#
# Usage:
#   python preprocess_sst2.py                           # all splits
#   python preprocess_sst2.py --split train
#   python preprocess_sst2.py --output_dir /path/to/sst2_datasets
#   python preprocess_sst2.py --max_samples 1000        # local sanity test
#
# Output files:
#   <output_dir>/sst2-train.json
#   <output_dir>/sst2-validation.json
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
            total += 1
    print(f"Saved {total} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GLUE SST-2 for MrBERT.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sst2_datasets",
        help="Directory to write output NDJSON files.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum tokenized sequence length.",
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
        choices=["train", "validation"],
        help="Single split to preprocess. If omitted, both splits are processed.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap each split at this many examples (useful for local sanity tests).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GLUE SST-2 dataset...")
    ds = load_dataset("glue", "sst2")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # SST-2 test labels are -1 (not publicly released); only process train/validation
    splits = [args.split] if args.split is not None else ["train", "validation"]

    for split in splits:
        split_data = ds[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"sst2-{split}.json")
        print(f"\nPreprocessing {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, output_path)

    print("\nDone.")