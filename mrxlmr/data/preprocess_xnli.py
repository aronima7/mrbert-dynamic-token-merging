# preprocess_xnli.py
#
# Downloads the XNLI dataset and saves tokenized train/validation/test splits
# to NDJSON files for use with MrXLMR.
# Uses XLM-RoBERTa tokenizer (SentencePiece, 250K vocab).
#
# XNLI structure: premise + hypothesis → 3-class NLI (same as SNLI)
# Training: English by default (XNLI English ≈ MultiNLI translated)
# Test:     Any of the 15 XNLI languages, enabling zero-shot cross-lingual eval
#
# Usage:
#   python preprocess_xnli.py                                    # English only
#   python preprocess_xnli.py --test_languages zh,de,sw,fr       # + cross-lingual test files
#   python preprocess_xnli.py --output_dir /checkpoints/xnli_datasets
#
# Output files:
#   <output_dir>/xnli-train.json
#   <output_dir>/xnli-validation.json
#   <output_dir>/xnli-test.json          (English)
#   <output_dir>/xnli-test-{lang}.json   (one per --test_languages entry)
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
from transformers import XLMRobertaTokenizerFast

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}

SUPPORTED_LANGUAGES = [
    "ar", "bg", "de", "el", "en", "es", "fr",
    "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]


def preprocess_and_save(dataset_split, tokenizer, max_length, output_path):
    """Tokenize a dataset split and write examples to an NDJSON file."""
    dataset_split = dataset_split.filter(lambda x: x["label"] != -1)

    with open(output_path, "w") as f:
        for example in tqdm(dataset_split, desc=f"Writing {os.path.basename(output_path)}"):
            tokenized = tokenizer(
                example["premise"],
                example["hypothesis"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            item = {
                "input_ids": [tokenized["input_ids"]],
                "attention_mask": [tokenized["attention_mask"]],
                "labels": example["label"],
            }
            json.dump(item, f)
            f.write("\n")

    print(f"Saved {len(dataset_split)} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess XNLI dataset for MrXLMR.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="xnli_datasets",
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
        default="xlm-roberta-base",
        help="XLM-R tokenizer to use for tokenization.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap each split at this many examples (useful for local sanity tests).",
    )
    parser.add_argument(
        "--test_languages",
        type=str,
        default=None,
        help=(
            "Comma-separated list of XNLI language codes to produce additional test files, "
            "e.g. 'zh,de,sw,fr'. Available: " + ", ".join(SUPPORTED_LANGUAGES)
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)

    # English: train + validation + test
    print("Loading XNLI English splits...")
    ds_en = load_dataset("xnli", "en")

    for split in ["train", "validation", "test"]:
        split_data = ds_en[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"xnli-{split}.json")
        print(f"\nPreprocessing English {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, output_path)

    # Optional: additional language test files for zero-shot cross-lingual eval
    if args.test_languages:
        langs = [l.strip() for l in args.test_languages.split(",") if l.strip()]
        for lang in langs:
            if lang not in SUPPORTED_LANGUAGES:
                print(f"Warning: unsupported language '{lang}', skipping.")
                continue
            if lang == "en":
                continue  # already written above
            print(f"\nLoading XNLI test split for language: {lang}")
            ds_lang = load_dataset("xnli", lang)
            test_data = ds_lang["test"]
            if args.max_samples is not None:
                test_data = test_data.select(range(min(args.max_samples, len(test_data))))
            output_path = os.path.join(args.output_dir, f"xnli-test-{lang}.json")
            print(f"Preprocessing {lang} test split ({len(test_data)} examples)...")
            preprocess_and_save(test_data, tokenizer, args.max_length, output_path)

    print("\nDone.")