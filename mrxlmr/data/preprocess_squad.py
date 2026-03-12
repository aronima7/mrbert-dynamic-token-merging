# preprocess_squad.py
#
# Downloads SQuAD and saves tokenized train/validation splits to NDJSON files.
# Uses XLM-RoBERTa tokenizer. Long contexts are split into overlapping windows.
#
# Usage:
#   python preprocess_squad.py
#   python preprocess_squad.py --output_dir /checkpoints/squad_datasets
#   python preprocess_squad.py --max_samples 1000
#
# Output files:
#   <output_dir>/squad-train.json
#   <output_dir>/squad-validation.json
#
# Each line is a JSON object with keys:
#   input_ids        : list[int]
#   attention_mask   : list[int]
#   start_positions  : int
#   end_positions    : int
#   answer_text      : str

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast


def convert_examples_to_features(examples, tokenizer, max_length, stride):
    """Tokenize a batch of SQuAD examples into model-ready features."""
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    features = []
    for feature_idx, offsets in enumerate(offset_mapping):
        input_ids      = tokenized["input_ids"][feature_idx]
        attention_mask = tokenized["attention_mask"][feature_idx]
        cls_index      = input_ids.index(tokenizer.cls_token_id)
        sequence_ids   = tokenized.sequence_ids(feature_idx)

        sample_idx = sample_mapping[feature_idx]
        answers    = examples["answers"][sample_idx]

        if len(answers["answer_start"]) == 0:
            start_position = cls_index
            end_position   = cls_index
            answer_text    = ""
        else:
            start_char  = answers["answer_start"][0]
            end_char    = start_char + len(answers["text"][0])
            answer_text = answers["text"][0]

            token_start = 0
            while token_start < len(sequence_ids) and sequence_ids[token_start] != 1:
                token_start += 1
            token_end = len(input_ids) - 1
            while token_end >= 0 and sequence_ids[token_end] != 1:
                token_end -= 1

            if offsets[token_start][0] > start_char or offsets[token_end][1] < end_char:
                start_position = cls_index
                end_position   = cls_index
            else:
                pos = token_start
                while pos <= token_end and offsets[pos][0] <= start_char:
                    pos += 1
                start_position = pos - 1

                pos = token_end
                while pos >= token_start and offsets[pos][1] >= end_char:
                    pos -= 1
                end_position = pos + 1

        features.append({
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "start_positions": start_position,
            "end_positions":   end_position,
            "answer_text":     answer_text,
        })

    return features


def preprocess_and_save(dataset_split, tokenizer, max_length, stride, output_path):
    """Convert a SQuAD split into features and write to an NDJSON file."""
    batch_size     = 500
    total_features = 0

    with open(output_path, "w") as f:
        for start in tqdm(
            range(0, len(dataset_split), batch_size),
            desc=f"Writing {os.path.basename(output_path)}",
        ):
            batch    = dataset_split[start: start + batch_size]
            features = convert_examples_to_features(batch, tokenizer, max_length, stride)
            for feat in features:
                json.dump(feat, f)
                f.write("\n")
                total_features += 1

    print(f"Saved {total_features} features from {len(dataset_split)} examples to {output_path}")
    print(f"  (expansion ratio: {total_features / len(dataset_split):.2f}x due to sliding window)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SQuAD dataset for MrXLMR.")

    parser.add_argument("--output_dir", type=str, default="squad_datasets")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--model_name", type=str, default="xlm-roberta-base",
        help="XLM-R tokenizer to use.",
    )
    parser.add_argument(
        "--split", type=str, default=None, choices=["train", "validation"],
        help="Single split to preprocess. If omitted, both splits are processed.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SQuAD dataset...")
    ds = load_dataset("rajpurkar/squad")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)

    splits = [args.split] if args.split is not None else ["train", "validation"]

    for split in splits:
        split_data = ds[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"squad-{split}.json")
        print(f"\nPreprocessing {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, args.stride, output_path)

    print("\nDone.")