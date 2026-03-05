# preprocess_mrpc.py
#
# Downloads GLUE MRPC and saves tokenized train/validation/test splits to NDJSON
# files for use with MrBERT and BERT sequence classification training.
#
# Usage:
#   python preprocess_mrpc.py                           # all splits
#   python preprocess_mrpc.py --split train
#   python preprocess_mrpc.py --output_dir /path/to/mrpc_datasets
#   python preprocess_mrpc.py --max_samples 1000        # local sanity test
#
# Output files:
#   <output_dir>/mrpc-train.json
#   <output_dir>/mrpc-validation.json
#   <output_dir>/mrpc-test.json
#
# Each line is a JSON object with keys:
#   input_ids       : list[int]
#   attention_mask  : list[int]
#   labels          : int (0=not paraphrase, 1=paraphrase)

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer


def preprocess_and_save(dataset_split, tokenizer, max_length, output_path):
    """Tokenize a sentence-pair dataset split and write examples to an NDJSON file."""
    total = 0
    with open(output_path, "w") as f:
        for example in tqdm(dataset_split, desc=f"Writing {os.path.basename(output_path)}"):
            tokenized = tokenizer(
                example["sentence1"],
                example["sentence2"],
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
    parser = argparse.ArgumentParser(description="Preprocess GLUE MRPC for MrBERT.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="mrpc_datasets",
        help="Directory to write output NDJSON files.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum tokenized sequence length (sentence1 + sentence2).",
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

    print("Loading GLUE MRPC dataset...")
    ds = load_dataset("glue", "mrpc")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    splits = [args.split] if args.split is not None else ["train", "validation", "test"]

    for split in splits:
        split_data = ds[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"mrpc-{split}.json")
        print(f"\nPreprocessing {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, output_path)

    print("\nDone.")