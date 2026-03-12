# preprocess_tydiqa.py
#
# Downloads TyDi QA (secondary_task / GoldP) and saves tokenized English-only
# splits to NDJSON files for use with MrBERT and BERT QA training.
#
# The secondary_task (GoldP) config provides a single gold passage per question,
# making it directly comparable to SQuAD.  By default only English examples are
# kept because bert-base-uncased is primarily an English model.
#
# Usage:
#   python preprocess_tydiqa.py                           # all splits, English only
#   python preprocess_tydiqa.py --split train
#   python preprocess_tydiqa.py --output_dir /path/to/tydiqa_datasets
#   python preprocess_tydiqa.py --max_samples 1000        # local sanity test
#   python preprocess_tydiqa.py --all_languages           # include all 11 languages
#
# Output files:
#   <output_dir>/tydiqa-train.json
#   <output_dir>/tydiqa-validation.json
#
# Each line is a JSON object with keys:
#   input_ids        : list[int]
#   attention_mask   : list[int]
#   start_positions  : int         (token index of answer start; 0=[CLS] for out-of-window)
#   end_positions    : int         (token index of answer end)
#   answer_text      : str         (original answer string; used for EM/F1 evaluation)
#
# Long contexts are split into overlapping windows (stride=128). Each window
# becomes its own example. If the answer falls outside a given window,
# start/end are set to 0 ([CLS]).

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast


def convert_examples_to_features(examples, tokenizer, max_length, stride):
    """
    Tokenize a batch of TyDi QA examples into model-ready features.

    Format is identical to SQuAD: question + context with sliding window
    for long passages.  Returns a flat list of feature dicts.
    """
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

            # Locate context token boundaries in this window
            token_start = 0
            while token_start < len(sequence_ids) and sequence_ids[token_start] != 1:
                token_start += 1
            token_end = len(input_ids) - 1
            while token_end >= 0 and sequence_ids[token_end] != 1:
                token_end -= 1

            if offsets[token_start][0] > start_char or offsets[token_end][1] < end_char:
                # Answer is outside this window → point to [CLS]
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
    """Convert a TyDi QA split into features and write them to an NDJSON file."""
    batch_size    = 500
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
    parser = argparse.ArgumentParser(
        description="Preprocess TyDi QA (secondary_task / GoldP) for MrBERT."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="tydiqa_datasets",
        help="Directory to write output NDJSON files.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Maximum tokenized sequence length (question + context). 384 is standard for QA.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Overlap in tokens between sliding windows for long contexts.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="BERT tokenizer to use.",
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
    parser.add_argument(
        "--all_languages",
        action="store_true",
        help="Include all 11 TyDi QA languages (default: English only).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading TyDi QA (secondary_task / GoldP) dataset...")
    ds = load_dataset("tydiqa", "secondary_task")

    if not args.all_languages:
        print("Filtering for English-only examples (pass --all_languages to include all)...")
        ds = ds.filter(lambda x: x["id"].startswith("english-"))
        for split_name in ds.keys():
            print(f"  {split_name}: {len(ds[split_name])} English examples")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    splits = [args.split] if args.split is not None else ["train", "validation"]

    for split in splits:
        split_data = ds[split]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        output_path = os.path.join(args.output_dir, f"tydiqa-{split}.json")
        print(f"\nPreprocessing {split} split ({len(split_data)} examples)...")
        preprocess_and_save(split_data, tokenizer, args.max_length, args.stride, output_path)

    print("\nDone.")