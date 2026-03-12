"""
Diagnostics and logging for MrXLMR: parameters, tensor shapes, dropped-token analysis.

XLM-R special token IDs: <s>=0 (CLS), <pad>=1, </s>=2 (SEP), <mask>=250001

Usage:
    from diagnostics import log_parameter_summary
    log_parameter_summary(model, label="MrXLMR")
"""
from __future__ import annotations

import re
from typing import Any

import torch
from torch.nn import Module


# XLM-R special token IDs
CLS_ID = 0       # <s>
PAD_ID = 1       # <pad>
SEP_ID = 2       # </s>
MASK_ID = 250001  # <mask>


# =============================================================================
# Parameter counting
# =============================================================================

def count_parameters(module: Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def log_parameter_summary(model: Module, label: str = "MrXLMR") -> dict[str, int]:
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
# Compute savings estimation
# =============================================================================

def theoretical_compute_saved_pct(
    seq_before: int,
    seq_after: int,
    num_layers: int = 12,
    gate_layer_index: int = 3,
) -> float:
    """
    Approximate MACs saved by token deletion.
    Self-attention cost is O(seq^2) per layer.
    Returns fraction in [0, 1].
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


# =============================================================================
# Dropped token analysis
# =============================================================================

def _token_type(token_str: str, token_id: int) -> str:
    """Classify token as: special, subword, punctuation, or word."""
    if token_id in (CLS_ID, SEP_ID, PAD_ID, MASK_ID):
        return "special"
    # XLM-R uses ▁ prefix for word-initial tokens (SentencePiece)
    if token_str.startswith("▁"):
        return "word"
    if re.match(r"^[\W_]+$", token_str) or token_str in (".", ",", "!", "?", "-", ";", ":", "'", '"'):
        return "punctuation"
    # Tokens without ▁ prefix are continuation/subword pieces
    return "subword"


def analyze_dropped_tokens(
    tokenizer: Any,
    input_ids: torch.Tensor,
    keep_mask: torch.Tensor,
    batch_index: int = 0,
) -> dict[str, Any]:
    """
    For one batch item, analyze which tokens were kept vs dropped and their types.

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
    """Print a dropped-token breakdown for one batch item."""
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