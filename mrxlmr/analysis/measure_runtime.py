"""
measure_runtime.py

Measure per-sample inference runtime (ms/sample) for XLM-R and MrXLMR checkpoints.
Equivalent to MrT5 paper Table 3, adapted for XLM-RoBERTa.

For each model provided, runs N warmup passes then N timed forward passes on a
fixed batch drawn from the SNLI test set and reports:
  - mean runtime (ms/sample)
  - runtime decrease vs XLM-R baseline (%)
  - actual sequence length reduction (%) from the delete gate
  - theoretical compute savings (MACs, relative to XLM-R baseline)

Usage (from mrxlmr/ directory):

  # Measure two checkpoints against each other
  python analysis/measure_runtime.py \\
      --models "XLM-R,./xlmr_snli/final" "MrXLMR-30%,./mrxlmr_snli_del30/final" \\
      --local_snli_dir ./snli_datasets

  # Quick CPU sanity test (small n)
  python analysis/measure_runtime.py \\
      --models "XLM-R,./xlmr_snli/final" "MrXLMR-30%,./mrxlmr_snli_del30/final" \\
      --local_snli_dir ./snli_datasets \\
      --n_warmup 2 --n_timed 10 --batch_size 8

  # Soft deletion only (no actual speedup — for comparison)
  python analysis/measure_runtime.py \\
      --models "XLM-R,./xlmr_snli/final" "MrXLMR-30%,./mrxlmr_snli_del30/final" \\
      --local_snli_dir ./snli_datasets --no_hard_delete

  # After downloading Modal checkpoints:
  python analysis/measure_runtime.py \\
      --models \\
          "XLM-R,./local_mrxlmr_checkpoints/xlmr-snli/final" \\
          "MrXLMR-30%,./local_mrxlmr_checkpoints/mrxlmr-snli-del30/final" \\
          "MrXLMR-50%,./local_mrxlmr_checkpoints/mrxlmr-snli-del50/final" \\
          "MrXLMR-70%,./local_mrxlmr_checkpoints/mrxlmr-snli-del70/final" \\
      --local_snli_dir ./snli_datasets \\
      --output_dir analysis/figures

Output:
  Prints a runtime table to stdout (mirrors MrT5 Table 3).
  Saves analysis/figures/runtime_table.csv and runtime_vs_deletion.pdf.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification

from configuration_mrxlmr import MrXLMRConfig
from modeling_mrxlmr import MrXLMRForSequenceClassification


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str):
    """
    Load an XLM-R or MrXLMR checkpoint for sequence classification.
    Detects model type from config.json.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found at {model_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("model_type") == "mrxlmr":
        config = MrXLMRConfig.from_pretrained(model_path)
        model = MrXLMRForSequenceClassification.from_pretrained(model_path, config=config)
    else:
        model = XLMRobertaForSequenceClassification.from_pretrained(model_path)

    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_snli_batch(local_snli_dir: str, tokenizer, max_seq_length: int, n_samples: int):
    """
    Load n_samples from the SNLI test split as a fixed TensorDataset.
    Reads pre-tokenized NDJSON produced by preprocess_snli.py.
    Falls back to HuggingFace download if local files are not present.

    XLM-R pad_token_id = 1; padding is done with 1 (not 0 as in BERT).
    """
    local_test = os.path.join(local_snli_dir, "snli-test.json") if local_snli_dir else None

    if local_test and os.path.exists(local_test):
        print(f"Loading SNLI test from {local_test} ...")
        input_ids_list, mask_list = [], []
        with open(local_test) as f:
            for line in f:
                if len(input_ids_list) >= n_samples:
                    break
                ex = json.loads(line)
                ids  = ex["input_ids"]
                mask = ex["attention_mask"]
                if ids and isinstance(ids[0], list):
                    for seq_ids, seq_mask in zip(ids, mask):
                        if len(input_ids_list) >= n_samples:
                            break
                        input_ids_list.append(seq_ids[:max_seq_length])
                        mask_list.append(seq_mask[:max_seq_length])
                else:
                    input_ids_list.append(ids[:max_seq_length])
                    mask_list.append(mask[:max_seq_length])
    else:
        print("Loading SNLI test from HuggingFace (this may take a moment)...")
        from datasets import load_dataset
        ds = load_dataset("snli", split="test")
        ds = ds.filter(lambda x: x["label"] != -1)
        ds = ds.select(range(min(n_samples, len(ds))))
        enc = tokenizer(
            ds["premise"], ds["hypothesis"],
            truncation=True, max_length=max_seq_length,
            padding="max_length",
        )
        input_ids_list = enc["input_ids"]
        mask_list      = enc["attention_mask"]

    # Pad/truncate all sequences to max_seq_length
    # XLM-R pad_token_id = 1 (not 0 as in BERT)
    def pad(seq, length, pad_val=1):
        seq = seq[:length]
        return seq + [pad_val] * (length - len(seq))

    input_ids      = torch.tensor([pad(x, max_seq_length)    for x in input_ids_list], dtype=torch.long)
    attention_mask = torch.tensor([pad(x, max_seq_length, 0) for x in mask_list],      dtype=torch.long)

    print(f"  Loaded {len(input_ids)} examples (max_seq_length={max_seq_length})")
    return TensorDataset(input_ids, attention_mask)


# ---------------------------------------------------------------------------
# Runtime measurement
# ---------------------------------------------------------------------------

def measure_runtime(
    model,
    dataset: TensorDataset,
    batch_size: int,
    n_warmup: int,
    n_timed: int,
    device: str,
    hard_delete: bool = True,
):
    """
    Run n_warmup + n_timed batches. Return timing stats and gate statistics.

    Returns dict with:
      - mean_ms_per_sample: float
      - std_ms_per_sample:  float
      - mean_seq_len_after: float   (avg kept tokens from delete gate; None for XLM-R baseline)
      - original_seq_len:   int

    hard_delete=True (default): MrXLMR physically removes tokens, giving real speedup.
    hard_delete=False: soft deletion only (attention masking); near-zero runtime difference.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loader_iter = iter(loader)

    original_seq_len = dataset[0][0].shape[0]  # max_seq_length

    is_mrxlmr = getattr(model.config, "model_type", "") == "mrxlmr"

    # Warmup passes (not timed)
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                ids, mask = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                ids, mask = next(loader_iter)
            ids, mask = ids.to(device), mask.to(device)
            if is_mrxlmr:
                _ = model(input_ids=ids, attention_mask=mask, hard_delete=hard_delete)
            else:
                _ = model(input_ids=ids, attention_mask=mask)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed passes
    batch_times = []
    all_new_seq_lens = []

    with torch.no_grad():
        for _ in range(n_timed):
            try:
                ids, mask = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                ids, mask = next(loader_iter)
            ids, mask = ids.to(device), mask.to(device)
            n_in_batch = ids.shape[0]

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if is_mrxlmr:
                outputs = model(input_ids=ids, attention_mask=mask, hard_delete=hard_delete)
            else:
                outputs = model(input_ids=ids, attention_mask=mask)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000.0
            batch_times.append(elapsed_ms / n_in_batch)  # ms per sample

            # Collect kept token count from delete_gate_mask.
            # Gate values are in [sigmoid_mask_scale, 0] (default [-30, 0]).
            # Tokens with gate_value > deletion_threshold (default -15) are kept.
            if is_mrxlmr and hasattr(outputs, "delete_gate_mask") and outputs.delete_gate_mask is not None:
                threshold = getattr(model.config, "deletion_threshold", -15.0)
                kept = (outputs.delete_gate_mask.squeeze(-1) > threshold).float().sum(dim=1).mean().item()
                all_new_seq_lens.append(kept)

    mean_ms = sum(batch_times) / len(batch_times)
    std_ms  = (sum((x - mean_ms) ** 2 for x in batch_times) / len(batch_times)) ** 0.5

    mean_new_seq = (sum(all_new_seq_lens) / len(all_new_seq_lens)) if all_new_seq_lens else None

    return {
        "mean_ms_per_sample": mean_ms,
        "std_ms_per_sample":  std_ms,
        "mean_seq_len_after": mean_new_seq,
        "original_seq_len":   original_seq_len,
    }


# ---------------------------------------------------------------------------
# Table printing and CSV output
# ---------------------------------------------------------------------------

def print_and_save_table(results: list, baseline_ms: float, max_seq_len: int,
                         output_dir: str, filename: str = "runtime_table.csv"):
    """
    Print a runtime table (mirrors MrT5 paper Table 3) and save as CSV.
    """
    print()
    print("=" * 85)
    print(f"  {'Model':<22}  {'ms/sample':>10}  {'±':>6}  {'vs XLM-R':>10}  {'Seq Δ':>10}  {'Seq Δ %':>10}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

    rows = []
    for r in results:
        ms        = r["mean_ms"]
        std       = r["std_ms"]
        pct_dec   = (1.0 - ms / baseline_ms) * 100.0 if baseline_ms > 0 else 0.0
        new_seq   = r["mean_new_seq_len"]
        seq_delta = (new_seq - max_seq_len) if new_seq is not None else None
        seq_pct   = ((1.0 - new_seq / max_seq_len) * 100.0) if new_seq is not None else None

        seq_delta_str = f"{seq_delta:+.1f}" if seq_delta is not None else "  n/a"
        seq_pct_str   = f"{seq_pct:.1f}%"   if seq_pct   is not None else "  n/a"
        pct_dec_str   = f"{pct_dec:+.1f}%" if r["label"] != results[0]["label"] else "baseline"

        print(f"  {r['label']:<22}  {ms:>10.2f}  {std:>6.2f}  {pct_dec_str:>10}  {seq_delta_str:>10}  {seq_pct_str:>10}")
        rows.append({
            "label":                    r["label"],
            "ms_per_sample":            f"{ms:.3f}",
            "std_ms":                   f"{std:.3f}",
            "pct_decrease_vs_xlmr":     f"{pct_dec:.2f}",
            "mean_seq_len_after":       f"{new_seq:.1f}" if new_seq is not None else "",
            "seq_len_reduction_pct":    f"{seq_pct:.2f}" if seq_pct is not None else "",
        })

    print("=" * 85)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, "w") as f:
        headers = ["label", "ms_per_sample", "std_ms", "pct_decrease_vs_xlmr",
                   "mean_seq_len_after", "seq_len_reduction_pct"]
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(row[h] for h in headers) + "\n")
    print(f"\nSaved → {csv_path}")


# ---------------------------------------------------------------------------
# Runtime vs deletion rate plot
# ---------------------------------------------------------------------------

def plot_runtime_vs_deletion(results: list, baseline_ms: float, output_dir: str,
                              filename: str = "runtime_vs_deletion.pdf"):
    """
    Bar chart: runtime (ms/sample) per model, with XLM-R baseline as horizontal dashed line.
    """
    labels = [r["label"] for r in results]
    means  = [r["mean_ms"] for r in results]
    stds   = [r["std_ms"]  for r in results]
    colors = ["#377eb8" if r["mean_new_seq_len"] is None else "#e41a1c" for r in results]

    fig, ax = plt.subplots(figsize=(max(5, len(results) * 1.4), 4))
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, width=0.6, zorder=3)

    ax.axhline(y=baseline_ms, color="#377eb8", linestyle="--", linewidth=1.2,
               label=f"XLM-R baseline ({baseline_ms:.1f} ms/sample)", zorder=2)

    for i, (bar, ms) in enumerate(zip(bars, means)):
        if ms < baseline_ms and baseline_ms > 0:
            pct = (1.0 - ms / baseline_ms) * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stds[i] + 0.2,
                    f"−{pct:.1f}%", ha="center", va="bottom", fontsize=8, color="#e41a1c")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Inference time (ms / sample)")
    ax.set_title("Inference runtime: XLM-R vs MrXLMR at different deletion rates")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(means) * 1.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Measure MrXLMR vs XLM-R inference runtime (ms/sample)."
    )
    p.add_argument(
        "--models",
        nargs="+",
        metavar="LABEL,PATH",
        required=True,
        help=(
            "Models to benchmark. Each entry is 'label,/path/to/checkpoint'. "
            "The FIRST entry is treated as the baseline (XLM-R). "
            "Example: --models 'XLM-R,./xlmr_snli/final' 'MrXLMR-30%%,./mrxlmr_del30/final'"
        ),
    )
    p.add_argument(
        "--local_snli_dir",
        type=str,
        default=None,
        help="Path to pre-tokenized SNLI NDJSON files. Falls back to HuggingFace download if not set.",
    )
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Sequence length used during training (default: 128 for SNLI).",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=512,
        help="Number of test examples to load (default: 512).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    p.add_argument(
        "--n_warmup",
        type=int,
        default=5,
        help="Warmup batches before timing (default: 5).",
    )
    p.add_argument(
        "--n_timed",
        type=int,
        default=50,
        help="Timed batches to average over (default: 50).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="analysis/figures",
        help="Where to save runtime_table.csv and runtime_vs_deletion.pdf.",
    )
    p.add_argument(
        "--hard_delete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use hard deletion (physically remove tokens) for MrXLMR inference. "
            "Default True — gives real runtime speedup. "
            "Use --no_hard_delete to benchmark soft deletion (attention masking only)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Hard delete at inference: {args.hard_delete}")

    # Parse model entries: "label,path"
    model_entries = []
    for entry in args.models:
        parts = entry.split(",", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed --models entry {entry!r}. Expected 'LABEL,PATH'.")
        model_entries.append({"label": parts[0].strip(), "path": parts[1].strip()})

    # Load tokenizer from the first (baseline) checkpoint
    print(f"Loading tokenizer from {model_entries[0]['path']} ...")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_entries[0]["path"])

    # Load fixed dataset batch once (same inputs for all models — fair comparison)
    dataset = load_snli_batch(
        local_snli_dir=args.local_snli_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        n_samples=args.n_samples,
    )

    # Benchmark each model
    all_results = []
    for entry in model_entries:
        label = entry["label"]
        path  = entry["path"]
        print(f"\nLoading {label} from {path} ...")
        model = load_model(path, device)

        print(f"  Warmup ({args.n_warmup} batches) ...")
        stats = measure_runtime(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
            device=device,
            hard_delete=args.hard_delete,
        )
        print(f"  {label}: {stats['mean_ms_per_sample']:.2f} ± {stats['std_ms_per_sample']:.2f} ms/sample", end="")
        if stats["mean_seq_len_after"] is not None:
            red_pct = (1.0 - stats["mean_seq_len_after"] / stats["original_seq_len"]) * 100.0
            print(f"  |  seq len reduction: {red_pct:.1f}%", end="")
        print()

        all_results.append({
            "label":            label,
            "mean_ms":          stats["mean_ms_per_sample"],
            "std_ms":           stats["std_ms_per_sample"],
            "mean_new_seq_len": stats["mean_seq_len_after"],
        })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    baseline_ms = all_results[0]["mean_ms"]

    print_and_save_table(all_results, baseline_ms, args.max_seq_length, args.output_dir)
    plot_runtime_vs_deletion(all_results, baseline_ms, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()