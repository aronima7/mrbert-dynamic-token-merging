"""
Diagnostics and logging for MrBERT: parameters, tensor shapes, dropped-token analysis,
and theoretical compute savings.

Adapted from Alina's mrbert/diagnostics.py.

Usage:
    # Parameter summary at model creation
    from mrbert.models.diagnostics import log_parameter_summary
    log_parameter_summary(model)

    # After eval — dropped token analysis
    from mrbert.models.diagnostics import print_dropped_token_summary
    print_dropped_token_summary(tokenizer, input_ids, gate, threshold)

    # Standalone interpretability demo
    python mrbert/models/diagnostics.py --model_path ./mrbert_checkpoints/final
"""
from __future__ import annotations

import re
from typing import Any

import torch
from torch.nn import Module


# Special token IDs in BERT vocab (bert-base-uncased)
CLS_ID = 101
SEP_ID = 102
PAD_ID = 0


# =============================================================================
# Parameter counting
# =============================================================================

def count_parameters(module: Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def log_parameter_summary(model: Module, label: str = "MrBERT") -> dict[str, int]:
    """
    Log total, trainable, and DeleteGate-only parameter counts.
    Returns dict with keys: total, trainable, gate_params.
    """
    total, trainable = count_parameters(model)
    gate_params = sum(
        p.numel() for name, p in model.named_parameters() if "delete_gate" in name
    )
    out = {"total": total, "trainable": trainable, "gate_params": gate_params}
    print(f"[{label}] Parameters: total={total:,}  trainable={trainable:,}  DeleteGate={gate_params:,}")
    return out


# =============================================================================
# Compute savings estimation (paper Appendix C)
# =============================================================================

def theoretical_compute_saved_pct(
    seq_before: int,
    seq_after: int,
    num_layers: int = 12,
    gate_layer_index: int = 3,
) -> float:
    """
    Approximate MACs saved by token deletion (paper Appendix C).

    Self-attention cost is O(seq^2) per layer. Layers 0..gate_layer_index use
    seq_before; layers (gate_layer_index+1)..(num_layers-1) use seq_after.
    Returns fraction in [0, 1] (e.g. 0.425 for 42.5% savings).
    """
    if seq_before <= 0:
        return 0.0
    layers_before = gate_layer_index + 1
    layers_after = num_layers - layers_before
    total_before = num_layers * (seq_before ** 2)
    total_after = layers_before * (seq_before ** 2) + layers_after * (seq_after ** 2)
    if total_before <= 0:
        return 0.0
    saved = (total_before - total_after) / total_before
    return max(0.0, min(1.0, saved))


def log_shape_after_gate(
    batch_size: int,
    seq_before: int,
    seq_after: int,
    hidden_size: int,
    gate_layer_index: int,
    num_layers: int = 12,
) -> None:
    """Log tensor shape change at deletion and theoretical compute saved (paper Appendix C)."""
    print(
        f"  [Gate] After layer {gate_layer_index}: "
        f"[{batch_size}, {seq_before}, {hidden_size}] -> [{batch_size}, {seq_after}, {hidden_size}] "
        f"(kept {seq_after}/{seq_before} tokens)"
    )
    pct = theoretical_compute_saved_pct(seq_before, seq_after, num_layers, gate_layer_index)
    print(f"  [Gate] Theoretical compute saved (MACs, Appendix C): ~{pct * 100:.1f}%")


# =============================================================================
# Dropped token analysis
# =============================================================================

def _token_type(token_str: str, token_id: int) -> str:
    """Classify token as: special, subword, punctuation, or word."""
    if token_id in (CLS_ID, SEP_ID, PAD_ID):
        return "special"
    if token_str.startswith("##"):
        return "subword"
    if re.match(r"^[\W_]+$", token_str) or token_str in ("'", '"', ".", ",", "!", "?", "-", ";"):
        return "punctuation"
    return "word"


def analyze_dropped_tokens(
    tokenizer: Any,
    input_ids: torch.Tensor,
    keep_mask: torch.Tensor,
    batch_index: int = 0,
) -> dict[str, Any]:
    """
    For one batch item, analyze which tokens were kept vs dropped and their types.

    Args:
        tokenizer: HuggingFace tokenizer.
        input_ids: (batch, seq_len) token IDs.
        keep_mask: (batch, seq_len) bool tensor — True = kept.
        batch_index: Which example in the batch to analyze.

    Returns dict with: dropped_tokens, kept_tokens, dropped_by_type, kept_by_type,
    n_dropped, n_kept.
    """
    ids = input_ids[batch_index].tolist()
    mask = keep_mask[batch_index].tolist()
    dropped_tokens, kept_tokens = [], []
    dropped_by_type = {"special": 0, "subword": 0, "punctuation": 0, "word": 0}
    kept_by_type    = {"special": 0, "subword": 0, "punctuation": 0, "word": 0}

    for tid, kept in zip(ids, mask):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            s = f"[id={tid}]"
        ttype = _token_type(s, tid)
        if kept:
            kept_tokens.append(s)
            kept_by_type[ttype] += 1
        else:
            dropped_tokens.append(s)
            dropped_by_type[ttype] += 1

    return {
        "dropped_tokens": dropped_tokens,
        "kept_tokens": kept_tokens,
        "dropped_by_type": dropped_by_type,
        "kept_by_type": kept_by_type,
        "n_dropped": len(dropped_tokens),
        "n_kept": len(kept_tokens),
    }


def print_dropped_token_summary(
    tokenizer: Any,
    input_ids: torch.Tensor,
    gate: torch.Tensor,
    threshold: float,
    batch_index: int = 0,
    max_show: int = 25,
) -> dict[str, Any]:
    """
    Derive keep_mask from gate values and threshold, then print a dropped-token breakdown.

    Args:
        tokenizer: HuggingFace tokenizer.
        input_ids: (batch, seq_len).
        gate: (batch, seq_len) gate values — MrBERT gate_mask squeezed to 2D.
        threshold: gate value below which a token is counted as deleted (e.g. -15.0).
        batch_index: Which example to analyze.
        max_show: Max dropped tokens to display.
    """
    keep_mask = (gate > threshold)
    stats = analyze_dropped_tokens(tokenizer, input_ids, keep_mask, batch_index)
    n_d, n_k = stats["n_dropped"], stats["n_kept"]
    total = n_d + n_k
    print(f"  [Dropped tokens] batch_idx={batch_index}  kept={n_k}  dropped={n_d}  (del={n_d / total * 100:.1f}%)")
    print(f"    by type  dropped: {stats['dropped_by_type']}")
    print(f"             kept:    {stats['kept_by_type']}")
    if stats["dropped_tokens"] and max_show > 0:
        show = [f"[{t}]" for t in stats["dropped_tokens"][:max_show]]
        print(f"    dropped tokens: {' '.join(show)}")
    return stats


def aggregate_dropped_stats(stats_list: list[dict]) -> dict[str, float]:
    """Aggregate multiple analyze_dropped_tokens outputs into mean counts by type."""
    n = len(stats_list)
    if n == 0:
        return {}
    out: dict[str, Any] = {"n_samples": n}
    for key in ("dropped_by_type", "kept_by_type"):
        agg: dict[str, float] = {"special": 0.0, "subword": 0.0, "punctuation": 0.0, "word": 0.0}
        for s in stats_list:
            for k, v in s.get(key, {}).items():
                agg[k] = agg.get(k, 0.0) + v
        out[key] = {k: round(v / n, 2) for k, v in agg.items()}
    out["avg_dropped"] = round(sum(s["n_dropped"] for s in stats_list) / n, 2)
    out["avg_kept"]    = round(sum(s["n_kept"]    for s in stats_list) / n, 2)
    return out


# =============================================================================
# Gate interpretability demo
# Run: python mrbert/models/diagnostics.py [--model_path PATH] [--sentences ...]
# =============================================================================

def run_gate_interpretability_demo(
    sentences: list[str],
    model_path: str = "bert-base-uncased",
    max_length: int = 64,
    deletion_threshold: float = -15.0,
    stats_only: bool = False,
    num_labels: int = 3,
    delete_gate_layer: int = 3,
) -> None:
    """
    Load a MrBERT model and print per-token keep/delete decisions for each sentence.

    Args:
        sentences: List of strings to run through the gate.
        model_path: Path to a saved MrBERT checkpoint, or 'bert-base-uncased' for a
                    freshly initialized model (gate weights will be random).
        max_length: Tokenization max length.
        deletion_threshold: Gate value below which a token is deleted (default: -15.0).
        stats_only: If True, print deletion rate by token type instead of per-token output.
        num_labels: Number of classification labels (3 for SNLI).
        delete_gate_layer: Which encoder layer has the delete gate.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from transformers import BertTokenizer
    from mrbert.models.configuration_mrbert import MrBertConfig
    from mrbert.models.modeling_mrbert import MrBertForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    config = MrBertConfig(
        num_labels=num_labels,
        delete_gate_layer=delete_gate_layer,
        deletion_threshold=deletion_threshold,
    )
    if model_path == "bert-base-uncased":
        print("[diagnostics] Loading fresh MrBERT (random gate weights, pretrained BERT backbone).")
        model = MrBertForSequenceClassification.from_pretrained("bert-base-uncased", config=config, ignore_mismatched_sizes=True)
    else:
        print(f"[diagnostics] Loading MrBERT from {model_path}")
        model = MrBertForSequenceClassification.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)

    model = model.to(device).eval()

    enc = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    gate_mask = getattr(outputs, "delete_gate_mask", None)
    if gate_mask is None:
        print("[diagnostics] No gate output found. Is this a MrBERT model?")
        return

    # gate_mask shape: (batch, seq_len, 1) → squeeze to (batch, seq_len)
    gate = gate_mask.squeeze(-1).cpu()
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    keep_mask = gate > deletion_threshold

    if stats_only:
        by_type: dict[str, list[float]] = {"word": [], "subword": [], "punctuation": [], "special": []}
        for b in range(gate.shape[0]):
            stats = analyze_dropped_tokens(tokenizer, input_ids, keep_mask, batch_index=b)
            for ttype in by_type:
                dropped = stats["dropped_by_type"][ttype]
                total   = dropped + stats["kept_by_type"][ttype]
                if total > 0:
                    by_type[ttype].append(dropped / total)
        print("Deletion rate by token type:")
        for ttype, rates in by_type.items():
            if rates:
                print(f"  {ttype:12s}: {sum(rates) / len(rates):.2%}")
        return

    print(f"Per-token gate score (deleted if G < {deletion_threshold}):")
    for b in range(gate.shape[0]):
        ids_b  = input_ids[b].tolist()
        mask_b = attention_mask[b].tolist()
        gate_b = gate[b].tolist()
        print(f"\nSentence {b + 1}: {sentences[b]!r}")
        for i in range(len(ids_b)):
            if mask_b[i] == 0:
                break
            s    = tokenizer.decode([ids_b[i]])
            g    = gate_b[i]
            kept = "KEEP" if g >= deletion_threshold else "DEL "
            print(f"  {s!r:20s}  G={g:6.1f}  [{kept}]")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MrBERT gate interpretability: which tokens are kept vs deleted")
    p.add_argument("--model_path", type=str, default="bert-base-uncased",
                   help="Path to MrBERT checkpoint, or 'bert-base-uncased' for fresh model")
    p.add_argument("--sentences", nargs="+",
                   default=["A man is playing guitar on a stage.", "The animal rested quietly."])
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--deletion_threshold", type=float, default=-15.0)
    p.add_argument("--delete_gate_layer", type=int, default=3)
    p.add_argument("--num_labels", type=int, default=3)
    p.add_argument("--stats", action="store_true", help="Print deletion rate by token type only")
    a = p.parse_args()
    run_gate_interpretability_demo(
        sentences=a.sentences,
        model_path=a.model_path,
        max_length=a.max_length,
        deletion_threshold=a.deletion_threshold,
        stats_only=a.stats,
        num_labels=a.num_labels,
        delete_gate_layer=a.delete_gate_layer,
    )