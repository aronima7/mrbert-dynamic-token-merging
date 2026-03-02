"""
get_deletion_patterns.py

Run a trained MrBERT model over the SNLI test set (or a sample of it) and save
per-token gate decisions to a JSON file for downstream analysis.

Equivalent to mrt5/analysis/get_deletion_patterns.py, adapted for:
  - BertTokenizer word-piece tokens instead of byte-level ByT5 tokens
  - SNLI sequence-classification task instead of span corruption
  - MrBertForSequenceClassification instead of MrT5

Output JSON (one object per line):
    {
        "premise":          "A man is playing guitar.",
        "hypothesis":       "A person is making music.",
        "label":            0,
        "label_str":        "entailment",
        "prediction":       0,
        "prediction_str":   "entailment",
        "correct":          true,
        "decoded_tokens":   ["[CLS]", "a", "man", "is", ...],
        "gate_values":      [-0.1, -27.3, -1.2, ...],
        "deletion_mask":    [false, true, false, ...]   <- true = deleted
    }

Usage:
    # From mrbert/ directory
    python analysis/get_deletion_patterns.py \\
        --model_path ./mrbert_checkpoints/final \\
        --local_snli_dir ./snli_datasets \\
        --output_dir ./analysis/deletion_patterns \\
        --sample_size 1000

    # Larger sample, GPU
    python analysis/get_deletion_patterns.py \\
        --model_path ./mrbert_checkpoints/final \\
        --local_snli_dir ./snli_datasets \\
        --sample_size -1 \\
        --batch_size 64
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import argparse
import json
import torch
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm

from configuration_mrbert import MrBertConfig
from modeling_mrbert import MrBertForSequenceClassification


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def parse_args():
    p = argparse.ArgumentParser(description="Save MrBERT per-token gate decisions to JSON.")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to trained MrBERT checkpoint directory.")
    p.add_argument("--local_snli_dir", type=str, default=None,
                   help="Path to pre-tokenized SNLI JSON files (snli-test.json etc.). "
                        "If not set, raw SNLI is loaded from HuggingFace.")
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--sample_size", type=int, default=1000,
                   help="Number of examples to process. -1 for the full split.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--deletion_threshold", type=float, default=-15.0,
                   help="Gate value below which a token is counted as deleted.")
    p.add_argument("--output_dir", type=str, default="analysis/deletion_patterns")
    p.add_argument("--output_file", type=str, default=None,
                   help="Override output filename. Defaults to <model_basename>_<split>.json.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_snli(args, tokenizer):
    """
    Load and tokenize the SNLI split.
    Always loads raw text from HuggingFace so we can store premise/hypothesis strings.
    The local_snli_dir pre-tokenized files do not store the original text, so we
    re-tokenize here.
    """
    print(f"Loading SNLI {args.split} from HuggingFace...")
    dataset = load_dataset("snli", split=args.split)
    dataset = dataset.filter(lambda x: x["label"] != -1)
    if args.sample_size > 0:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))

    print(f"  {len(dataset)} examples loaded.")
    return dataset


def run_model(model, tokenizer, dataset, args, device):
    """
    Run MrBERT over dataset in batches. Returns list of result dicts.
    """
    model.eval()
    results = []

    # Build batches manually to keep premise/hypothesis text
    batch_size = args.batch_size
    examples = list(dataset)

    for start in tqdm(range(0, len(examples), batch_size), desc="Inferring"):
        batch_examples = examples[start : start + batch_size]

        premises    = [e["premise"]    for e in batch_examples]
        hypotheses  = [e["hypothesis"] for e in batch_examples]
        labels      = [e["label"]      for e in batch_examples]

        enc = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Predictions
        predictions = outputs.logits.argmax(dim=-1).cpu().tolist()

        # Gate values: (batch, seq, 1) → (batch, seq)
        gate = None
        if outputs.delete_gate_mask is not None:
            gate = outputs.delete_gate_mask.squeeze(-1).cpu()

        for i, example in enumerate(batch_examples):
            ids_i  = input_ids[i].cpu().tolist()
            mask_i = attention_mask[i].cpu().tolist()

            decoded = [tokenizer.decode([t]) for t in ids_i]

            if gate is not None:
                gate_i   = gate[i].tolist()
                del_mask = [g < args.deletion_threshold for g in gate_i]
            else:
                gate_i   = [0.0] * len(ids_i)
                del_mask = [False] * len(ids_i)

            results.append({
                "premise":          example["premise"],
                "hypothesis":       example["hypothesis"],
                "label":            example["label"],
                "label_str":        LABEL_NAMES.get(example["label"], "unknown"),
                "prediction":       predictions[i],
                "prediction_str":   LABEL_NAMES.get(predictions[i], "unknown"),
                "correct":          predictions[i] == example["label"],
                "decoded_tokens":   decoded,
                "attention_mask":   mask_i,
                "gate_values":      gate_i,
                "deletion_mask":    del_mask,
            })

    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading MrBERT from {args.model_path} ...")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = MrBertForSequenceClassification.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    print(f"  num_labels={model.config.num_labels}  "
          f"delete_gate_layer={model.config.delete_gate_layer}  "
          f"deletion_threshold={args.deletion_threshold}")

    # Load data
    dataset = load_snli(args, tokenizer)

    # Run
    results = run_model(model, tokenizer, dataset, args, device)

    # Accuracy summary
    correct = sum(r["correct"] for r in results)
    deleted_frac = [
        sum(d and m for d, m in zip(r["deletion_mask"], r["attention_mask"]))
        / max(sum(r["attention_mask"]), 1)
        for r in results
    ]
    print(f"\nSummary over {len(results)} examples:")
    print(f"  Accuracy:              {correct / len(results):.4f}")
    print(f"  Mean deletion rate:    {sum(deleted_frac) / len(deleted_frac):.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(args.model_path))
    out_file = args.output_file or f"{model_name}_{args.split}.json"
    out_path = os.path.join(args.output_dir, out_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} examples → {out_path}")


if __name__ == "__main__":
    main()