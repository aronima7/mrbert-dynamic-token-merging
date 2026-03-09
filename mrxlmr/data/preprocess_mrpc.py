# preprocess_mrpc.py
#
# Downloads MRPC (GLUE) and saves tokenized train/validation/test splits to NDJSON.
# Uses XLM-RoBERTa tokenizer.
#
# Usage:
#   python preprocess_mrpc.py
#   python preprocess_mrpc.py --output_dir /checkpoints/mrpc_datasets
#
# Output files:
#   <output_dir>/mrpc-train.json
#   <output_dir>/mrpc-validation.json
#   <output_dir>/mrpc-test.json
#
# Each line is a JSON object with keys:
#   input_ids      : list[int]
#   attention_mask : list[int]
#   labels         : int  (0=not paraphrase, 1=paraphrase)

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast


def preprocess_and_save(dataset_split, tokenizer, max_length, output_path):
    """Tokenize MRPC sentence pairs and write to an NDJSON file."""
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

    print(f"Saved {len(dataset_split)} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MRPC for MrXLMR.")

    parser.add_argument("--output_dir", type=str, default="mrpc_datasets")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--model_name", type=str, default="xlm-roberta-base",
        help="XLM-R tokenizer to use.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading MRPC (GLUE) dataset...")
    ds = load_dataset("glue", "mrpc")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)

    for split in ["train", "validation", "test"]:
        split_data = ds[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"mrpc-{split}.json")
        print(f"\nPreprocessing {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, output_path)

    print("\nDone.")