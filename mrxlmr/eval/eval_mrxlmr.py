#!/usr/bin/env python3
"""
Evaluation script for trained MrXLMR models.

Usage:
    # Evaluate on SNLI (sequence classification)
    python eval_mrxlmr.py --model_path ./mrxlmr_checkpoints/final \
        --dataset_name local_snli --local_snli_dir ./snli_datasets

    # Evaluate with a fresh XLM-R model (no training)
    python eval_mrxlmr.py --from_pretrained xlm-roberta-base

    # Evaluate with hard deletion
    python eval_mrxlmr.py --model_path ./mrxlmr_checkpoints/final --hard_delete

    # Compare MrXLMR vs baseline XLM-R
    python eval_mrxlmr.py --model_path ./mrxlmr_checkpoints/final --compare_xlmr

    # MLM perplexity evaluation
    python eval_mrxlmr.py --model_path ./mrxlmr_checkpoints/final \
        --dataset_name wikitext --dataset_config wikitext-2-raw-v1 --task mlm
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import torch
from torch.utils.data import DataLoader
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    XLMRobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    DefaultDataCollator,
)
from datasets import load_dataset

from configuration_mrxlmr import MrXLMRConfig
from modeling_mrxlmr import (
    MrXLMRForMaskedLM,
    MrXLMRForSequenceClassification,
    MrXLMRForTokenClassification,
    MrXLMRForQuestionAnswering,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MrXLMR models.")

    # Model arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved MrXLMR checkpoint directory.")
    parser.add_argument("--from_pretrained", type=str, default=None,
                        help="Load from a pretrained XLM-R model name (creates fresh MrXLMR with random gate).")
    parser.add_argument("--delete_gate_layer", type=int, default=3)
    parser.add_argument("--deletion_type", type=str, default="scaled_sigmoid")
    parser.add_argument("--deletion_threshold", type=float, default=-15.0)
    parser.add_argument("--sigmoid_mask_scale", type=float, default=-30.0)

    # Task
    parser.add_argument("--task", type=str, default="sequence_classification",
                        choices=["mlm", "sequence_classification", "token_classification", "question_answering"])
    parser.add_argument("--num_labels", type=int, default=3,
                        help="Number of labels for classification tasks.")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="local_snli")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--local_snli_dir", type=str, default="snli_datasets")
    parser.add_argument("--local_squad_dir", type=str, default="squad_datasets")
    parser.add_argument("--local_sst2_dir", type=str, default="sst2_datasets")
    parser.add_argument("--local_mrpc_dir", type=str, default="mrpc_datasets")
    parser.add_argument("--local_imdb_dir", type=str, default="imdb_datasets")
    parser.add_argument("--local_tydiqa_dir", type=str, default="tydiqa_datasets")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hard_delete", action="store_true",
                        help="Evaluate with hard deletion (physically remove tokens).")
    parser.add_argument("--compare_xlmr", action="store_true",
                        help="Compare MrXLMR against baseline XLM-R.")
    parser.add_argument("--show_examples", action="store_true",
                        help="Print deletion examples.")

    return parser.parse_args()


def load_model(args):
    """Load MrXLMR model from checkpoint or fresh pretrained."""
    if args.model_path is not None:
        print(f"Loading MrXLMR from checkpoint: {args.model_path}")
        config = MrXLMRConfig.from_pretrained(args.model_path)
        if args.task == "mlm":
            model = MrXLMRForMaskedLM.from_pretrained(args.model_path, config=config)
        elif args.task == "sequence_classification":
            model = MrXLMRForSequenceClassification.from_pretrained(args.model_path, config=config)
        elif args.task == "token_classification":
            model = MrXLMRForTokenClassification.from_pretrained(args.model_path, config=config)
        elif args.task == "question_answering":
            model = MrXLMRForQuestionAnswering.from_pretrained(args.model_path, config=config)
    else:
        base_model = args.from_pretrained or "xlm-roberta-base"
        print(f"Creating fresh MrXLMR from pretrained '{base_model}'...")
        config = MrXLMRConfig.from_pretrained(
            base_model,
            delete_gate_layer=args.delete_gate_layer,
            deletion_type=args.deletion_type,
            deletion_threshold=args.deletion_threshold,
            sigmoid_mask_scale=args.sigmoid_mask_scale,
        )
        if args.task == "mlm":
            model = MrXLMRForMaskedLM.from_pretrained(base_model, config=config, ignore_mismatched_sizes=True)
        elif args.task == "sequence_classification":
            config.num_labels = args.num_labels
            model = MrXLMRForSequenceClassification.from_pretrained(base_model, config=config, ignore_mismatched_sizes=True)
        elif args.task == "token_classification":
            config.num_labels = args.num_labels
            model = MrXLMRForTokenClassification.from_pretrained(base_model, config=config, ignore_mismatched_sizes=True)
        elif args.task == "question_answering":
            model = MrXLMRForQuestionAnswering.from_pretrained(base_model, config=config, ignore_mismatched_sizes=True)
    return model


def load_dataset_split(args, tokenizer):
    """Load and tokenize the evaluation dataset split."""
    if args.dataset_name == "local_snli":
        path = f"{args.local_snli_dir}/snli-{args.split}.json"
        dataset = load_dataset("json", data_files={args.split: path})[args.split]
        dataset = dataset.map(
            lambda x: {"input_ids": x["input_ids"][0], "attention_mask": x["attention_mask"][0], "labels": x["labels"]},
        )
    elif args.dataset_name == "local_sst2":
        path = f"{args.local_sst2_dir}/sst2-{args.split}.json"
        dataset = load_dataset("json", data_files={args.split: path})[args.split]
    elif args.dataset_name == "local_mrpc":
        path = f"{args.local_mrpc_dir}/mrpc-{args.split}.json"
        dataset = load_dataset("json", data_files={args.split: path})[args.split]
    elif args.dataset_name == "local_imdb":
        path = f"{args.local_imdb_dir}/imdb-{args.split}.json"
        dataset = load_dataset("json", data_files={args.split: path})[args.split]
    elif args.dataset_name == "local_squad":
        path = f"{args.local_squad_dir}/squad-{'validation' if args.split == 'test' else args.split}.json"
        dataset = load_dataset("json", data_files={args.split: path})[args.split]
    elif args.dataset_name == "local_tydiqa":
        path = f"{args.local_tydiqa_dir}/tydiqa-{'validation' if args.split == 'test' else args.split}.json"
        dataset = load_dataset("json", data_files={args.split: path})[args.split]
    else:
        if args.dataset_config:
            raw = load_dataset(args.dataset_name, args.dataset_config)
        else:
            raw = load_dataset(args.dataset_name)
        split = args.split if args.split in raw else "validation"
        text_col = "text" if "text" in raw[split].column_names else raw[split].column_names[0]
        dataset = raw[split].map(
            lambda x: tokenizer(x[text_col], truncation=True, max_length=args.max_seq_length, padding="max_length"),
            batched=True,
        )

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    dataset = dataset.with_format("torch")
    return dataset


def evaluate_classification(model, dataloader, device):
    """Evaluate accuracy for sequence classification."""
    model.eval()
    correct = 0
    total = 0
    deletion_rates = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**{k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids", "labels")})

            logits = outputs.logits
            labels = batch["labels"]
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Track deletion rate
            delete_gate_output = getattr(outputs, "delete_gate_output", None)
            if delete_gate_output is not None:
                input_ids = batch["input_ids"]
                non_pad = (input_ids != 1)  # XLM-R pad_token_id = 1
                gate_vals = delete_gate_output.squeeze(-1)
                threshold = model.config.deletion_threshold
                deleted = ((gate_vals < threshold) & non_pad).float().sum()
                total_non_pad = non_pad.float().sum()
                if total_non_pad > 0:
                    deletion_rates.append((deleted / total_non_pad).item())

    accuracy = correct / total if total > 0 else 0.0
    avg_deletion_rate = sum(deletion_rates) / len(deletion_rates) if deletion_rates else 0.0
    return accuracy, avg_deletion_rate


def evaluate_mlm(model, dataloader, device):
    """Evaluate MLM perplexity."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
                total_batches += 1

    if total_batches == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity, avg_loss


def analyze_deletions(model, dataloader, tokenizer, device, n_samples=5):
    """Analyze and print deletion patterns."""
    model.eval()
    samples_shown = 0
    total_deletion_rates = []

    print("\n" + "=" * 70)
    print("DELETION ANALYSIS")
    print("=" * 70)

    with torch.no_grad():
        for batch in dataloader:
            if samples_shown >= n_samples:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            input_ids_batch = batch["input_ids"]
            outputs = model(**{k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")})

            delete_gate_mask = getattr(outputs, "delete_gate_mask", None)
            if delete_gate_mask is None:
                print("No delete gate output found.")
                break

            threshold = model.config.deletion_threshold

            for i in range(min(input_ids_batch.size(0), n_samples - samples_shown)):
                ids = input_ids_batch[i].tolist()
                tokens = tokenizer.convert_ids_to_tokens(ids)
                gate_vals = delete_gate_mask[i].squeeze(-1).tolist()
                deleted = [g < threshold for g in gate_vals]

                pad_id = tokenizer.pad_token_id
                non_pad = [(tok, d, tid) for tid, (tok, d) in enumerate(zip(tokens, deleted)) if ids[tid] != pad_id]
                kept = [tok for tok, d, _ in non_pad if not d]
                del_toks = [tok for tok, d, _ in non_pad if d]

                del_rate = len(del_toks) / len(non_pad) * 100 if non_pad else 0.0
                total_deletion_rates.append(del_rate / 100.0)

                print(f"\nSample {samples_shown + 1}:")
                annotated = []
                for tok, d, idx in non_pad:
                    if ids[idx] in (tokenizer.cls_token_id, tokenizer.sep_token_id):
                        annotated.append(tok)
                    elif d:
                        annotated.append(f"[{tok}]")
                    else:
                        annotated.append(tok)
                print(f"  Sequence: {' '.join(annotated)}")
                print(f"  Kept ({len(kept)}): {' '.join(kept[:30])}")
                print(f"  Deleted ({len(del_toks)}, {del_rate:.1f}%): {' '.join(del_toks[:20]) or '(none)'}")
                samples_shown += 1

    avg_del = sum(total_deletion_rates) / len(total_deletion_rates) * 100 if total_deletion_rates else 0.0
    print(f"\nAverage deletion rate (non-pad): {avg_del:.1f}%")
    print("=" * 70)
    return avg_del


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        args.model_path or args.from_pretrained or "xlm-roberta-base"
    )

    model = load_model(args)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    if args.task == "mlm":
        dataset = load_dataset_split(args, tokenizer)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

        print(f"\nEvaluating MLM perplexity on {len(dataset)} examples...")
        ppl, loss = evaluate_mlm(model, dataloader, device)
        print(f"  MLM Perplexity: {ppl:.2f}")
        print(f"  MLM Loss:       {loss:.4f}")

    elif args.task in ("sequence_classification",):
        dataset = load_dataset_split(args, tokenizer)
        keep_cols = [c for c in ["input_ids", "attention_mask", "token_type_ids", "labels"] if c in dataset.column_names]
        dataset.set_format(type="torch", columns=keep_cols)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=DefaultDataCollator())

        print(f"\nEvaluating on {len(dataset)} examples...")
        if args.hard_delete:
            print("Using HARD deletion (physical token removal)")
            # Monkey-patch forward to use hard_delete
            _orig_fwd = model.forward
            def _hard_fwd(**kwargs):
                kwargs["hard_delete"] = True
                return _orig_fwd(**kwargs)
            model.forward = _hard_fwd

        accuracy, avg_del_rate = evaluate_classification(model, dataloader, device)
        print(f"\nResults on {args.dataset_name} [{args.split}]:")
        print(f"  Accuracy:       {accuracy * 100:.2f}%")
        print(f"  Deletion rate:  {avg_del_rate * 100:.1f}%")

        if args.show_examples:
            analyze_deletions(model, dataloader, tokenizer, device)

        if args.compare_xlmr:
            print("\n--- Baseline XLM-R ---")
            xlmr_model = XLMRobertaForSequenceClassification.from_pretrained(
                "xlm-roberta-base", num_labels=model.config.num_labels
            ).to(device)
            xlmr_acc, _ = evaluate_classification(xlmr_model, dataloader, device)
            print(f"  Accuracy (XLM-R baseline): {xlmr_acc * 100:.2f}%")
            print(f"  Accuracy (MrXLMR):         {accuracy * 100:.2f}%")
            print(f"  Diff:                      {(accuracy - xlmr_acc) * 100:+.2f}%")


if __name__ == "__main__":
    main()