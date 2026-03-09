"""
deletion_pattern_analysis.py

Visualise and analyse per-token deletion patterns produced by get_deletion_patterns.py.
Adapted from mrbert/analysis/deletion_pattern_analysis.py for XLM-RoBERTa / SentencePiece.

Key XLM-R differences:
  - Tokens are SentencePiece pieces; word-initial tokens start with "▁" (not "##" for subwords)
  - Special tokens: "<s>" (CLS), "</s>" (SEP), "<pad>", "<mask>", "<unk>"
  - SNLI pair format: <s> premise </s> </s> hypothesis </s>
    (two consecutive </s> tokens separate premise from hypothesis)

Analyses performed:
  1. Colored terminal output — KEEP tokens in green, DELETED tokens in red
  2. Token-type deletion rates — word / subword / punctuation / special
  3. Top-N most and least deleted individual tokens
  4. Bar chart of deletion rates by token type  (saved to figures/)
  5. SNLI-specific: deletion rate in the premise segment vs hypothesis segment

Usage:
    # From mrxlmr/ directory
    python analysis/deletion_pattern_analysis.py \\
        --input_file analysis/deletion_patterns/final_test.json

    python analysis/deletion_pattern_analysis.py \\
        --input_file analysis/deletion_patterns/final_test.json \\
        --n_print 30 \\
        --output_dir analysis/figures
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import argparse
import json
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# XLM-R special token strings (SentencePiece)
CLS_TOK  = "<s>"
SEP_TOK  = "</s>"
PAD_TOK  = "<pad>"
MASK_TOK = "<mask>"
UNK_TOK  = "<unk>"

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}

# ANSI colours
RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_type(token_str: str) -> str:
    """
    Classify a decoded XLM-R SentencePiece token string.

    XLM-R SentencePiece convention:
      - Word-initial tokens start with "▁" (e.g. "▁the", "▁hello")
      - Continuation tokens have no prefix (e.g. "ing", "tion")
      - Special tokens: <s>, </s>, <pad>, <mask>, <unk>
    """
    s = token_str.strip()
    if s in (CLS_TOK, SEP_TOK, PAD_TOK, MASK_TOK, UNK_TOK):
        return "special"
    # Strip leading ▁ to get the surface form
    surface = s.lstrip("▁")
    # Continuation piece (no ▁ prefix and not a special token)
    if not s.startswith("▁") and s not in (CLS_TOK, SEP_TOK, PAD_TOK, MASK_TOK, UNK_TOK):
        return "subword"
    # Punctuation: surface is entirely non-word characters
    if re.match(r"^[\W_]+$", surface) or surface in ("'", '"', ".", ",", "!", "?", "-", ";", ":"):
        return "punctuation"
    return "word"


def _non_pad_tokens(example):
    """Iterate (token_str, gate_val, is_deleted) for non-pad positions."""
    for tok, gate, deleted, mask in zip(
        example["decoded_tokens"],
        example["gate_values"],
        example["deletion_mask"],
        example["attention_mask"],
    ):
        if mask == 1:
            yield tok, gate, deleted


def _find_sep_positions(example):
    """
    Return indices (in the non-padded token list) of all </s> tokens.

    XLM-R SNLI format: <s> premise </s> </s> hypothesis </s>
    So there will be three </s> tokens (indices 1, 2, 3 from the right counting non-pad).
    We return all of them; the caller picks the first for premise/hypothesis split.
    """
    seps = []
    for i, (tok, mask) in enumerate(zip(example["decoded_tokens"], example["attention_mask"])):
        if mask == 0:
            break
        if tok.strip() == SEP_TOK:
            seps.append(i)
    return seps


# ---------------------------------------------------------------------------
# 1. Colored terminal output
# ---------------------------------------------------------------------------

def print_colored_examples(examples, n: int = 20, deletion_threshold: float = -15.0):
    """Print n examples with deleted tokens in red, kept tokens in green."""
    print("\n" + "=" * 70)
    print("DELETION PATTERNS  (RED = deleted, GREEN = kept)")
    print("=" * 70)

    for i, ex in enumerate(examples[:n]):
        label_str = LABEL_NAMES.get(ex["label"], "?")
        pred_str  = LABEL_NAMES.get(ex["prediction"], "?")
        correct   = "✓" if ex["correct"] else "✗"

        deleted_count = sum(
            1 for d, m in zip(ex["deletion_mask"], ex["attention_mask"])
            if d and m == 1
        )
        total_count = sum(ex["attention_mask"])
        del_rate = deleted_count / total_count * 100 if total_count > 0 else 0.0

        print(f"\n[{i+1}] label={label_str}  pred={pred_str} {correct}  "
              f"del={del_rate:.1f}% ({deleted_count}/{total_count})")

        parts = []
        for tok, gate, deleted in _non_pad_tokens(ex):
            color = RED if deleted else GREEN
            parts.append(f"{color}{tok}{RESET}")
        print("  " + " ".join(parts))


# ---------------------------------------------------------------------------
# 2. Deletion rates by token type
# ---------------------------------------------------------------------------

def compute_token_type_rates(examples):
    """
    Returns dict: token_type -> {"total": int, "deleted": int, "rate": float}
    """
    counts = defaultdict(lambda: {"total": 0, "deleted": 0})
    for ex in examples:
        for tok, gate, deleted in _non_pad_tokens(ex):
            ttype = _token_type(tok)
            counts[ttype]["total"] += 1
            if deleted:
                counts[ttype]["deleted"] += 1

    result = {}
    for ttype, c in counts.items():
        result[ttype] = {
            **c,
            "rate": c["deleted"] / c["total"] if c["total"] > 0 else 0.0,
        }
    return result


def print_token_type_table(rates: dict):
    print("\n" + "=" * 55)
    print("DELETION RATE BY TOKEN TYPE")
    print("=" * 55)
    print(f"  {'Type':<15}  {'Total':>8}  {'Deleted':>8}  {'Rate':>8}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    for ttype in ("word", "subword", "punctuation", "special"):
        if ttype not in rates:
            continue
        c = rates[ttype]
        print(f"  {ttype:<15}  {c['total']:>8,}  {c['deleted']:>8,}  {c['rate']:>7.1%}")


# ---------------------------------------------------------------------------
# 3. Top-N most / least deleted tokens
# ---------------------------------------------------------------------------

def compute_per_token_rates(examples, min_count: int = 10):
    """
    Returns (deleted_counter, kept_counter) over individual token strings,
    filtered to tokens appearing at least min_count times.
    """
    deleted_counter = Counter()
    kept_counter    = Counter()
    for ex in examples:
        for tok, gate, deleted in _non_pad_tokens(ex):
            tok = tok.strip()
            if deleted:
                deleted_counter[tok] += 1
            else:
                kept_counter[tok] += 1

    all_tokens = set(deleted_counter) | set(kept_counter)
    filtered_del  = Counter()
    filtered_kept = Counter()
    for tok in all_tokens:
        if deleted_counter[tok] + kept_counter[tok] >= min_count:
            filtered_del[tok]  = deleted_counter[tok]
            filtered_kept[tok] = kept_counter[tok]

    return filtered_del, filtered_kept


def print_top_tokens(deleted_counter, kept_counter, n: int = 15):
    all_tokens = set(deleted_counter) | set(kept_counter)
    rates = {}
    for tok in all_tokens:
        total = deleted_counter[tok] + kept_counter[tok]
        rates[tok] = deleted_counter[tok] / total

    sorted_by_rate = sorted(rates.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 55)
    print(f"TOP {n} MOST DELETED TOKENS  (among tokens with ≥10 occurrences)")
    print("=" * 55)
    print(f"  {'Token':<20}  {'Del':>6}  {'Total':>6}  {'Rate':>7}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*7}")
    for tok, rate in sorted_by_rate[:n]:
        total = deleted_counter[tok] + kept_counter[tok]
        print(f"  {tok!r:<20}  {deleted_counter[tok]:>6}  {total:>6}  {rate:>6.1%}")

    print(f"\nTOP {n} LEAST DELETED TOKENS")
    print(f"  {'Token':<20}  {'Del':>6}  {'Total':>6}  {'Rate':>7}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*7}")
    for tok, rate in sorted_by_rate[-n:][::-1]:
        total = deleted_counter[tok] + kept_counter[tok]
        print(f"  {tok!r:<20}  {deleted_counter[tok]:>6}  {total:>6}  {rate:>6.1%}")


# ---------------------------------------------------------------------------
# 4. Bar chart — deletion rate by token type
# ---------------------------------------------------------------------------

def plot_deletion_by_type(rates: dict, output_dir: str, filename: str = "deletion_by_type.pdf"):
    types    = [t for t in ("word", "subword", "punctuation", "special") if t in rates]
    del_pct  = [rates[t]["rate"] * 100 for t in types]
    kept_pct = [100 - d for d in del_pct]

    x = range(len(types))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x, kept_pct, color="#4daf4a", label="Kept")
    ax.bar(x, del_pct,  bottom=kept_pct, color="#e41a1c", label="Deleted")

    for i, d in enumerate(del_pct):
        ax.text(i, 103, f"{d:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(types)
    ax.set_ylabel("Percentage of tokens")
    ax.set_title("Deletion rate by token type (MrXLMR)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {path}")


# ---------------------------------------------------------------------------
# 5. SNLI-specific: premise vs hypothesis deletion
# ---------------------------------------------------------------------------

def compute_premise_hypothesis_rates(examples):
    """
    For each example, split tokens into premise segment and hypothesis segment.

    XLM-R SNLI format: <s> premise </s> </s> hypothesis </s>
    The first </s> ends the premise; tokens between the two </s> separators
    and the final </s> form the hypothesis.

    We use the first two </s> occurrences to find the boundary:
      positions 0         : <s>   (skip)
      positions 1..sep0-1 : premise
      position  sep0      : </s>  (skip)
      position  sep1      : </s>  (skip — second separator before hypothesis)
      positions sep1+1..sep2-1: hypothesis
      position  sep2      : </s>  (skip)
    """
    prem_del = prem_total = hyp_del = hyp_total = 0

    for ex in examples:
        seps = _find_sep_positions(ex)
        if len(seps) < 3:
            continue  # need at least 3 </s> tokens for the XLM-R pair format

        sep0 = seps[0]   # end of premise
        sep1 = seps[1]   # second separator (before hypothesis)
        sep2 = seps[2]   # end of hypothesis

        for i, (tok, gate, deleted) in enumerate(_non_pad_tokens(ex)):
            if i == 0:          # <s>
                continue
            if 1 <= i < sep0:   # premise tokens
                prem_total += 1
                if deleted:
                    prem_del += 1
            elif sep1 < i < sep2:  # hypothesis tokens
                hyp_total += 1
                if deleted:
                    hyp_del += 1
            # sep tokens and anything beyond sep2 are skipped

    prem_rate = prem_del / prem_total if prem_total > 0 else 0.0
    hyp_rate  = hyp_del  / hyp_total  if hyp_total  > 0 else 0.0
    return prem_rate, hyp_rate, prem_total, hyp_total


def plot_premise_vs_hypothesis(prem_rate, hyp_rate, output_dir, filename="premise_vs_hypothesis.pdf"):
    fig, ax = plt.subplots(figsize=(4, 3))
    segments = ["Premise", "Hypothesis"]
    rates    = [prem_rate * 100, hyp_rate * 100]
    colors   = ["#377eb8", "#e41a1c"]

    bars = ax.bar(segments, rates, color=colors, width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Deletion rate (%)")
    ax.set_title("Deletion rate: premise vs hypothesis\n(SNLI, MrXLMR)")
    ax.set_ylim(0, max(rates) * 1.3 + 5)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def print_premise_hypothesis_table(prem_rate, hyp_rate, prem_total, hyp_total):
    print("\n" + "=" * 55)
    print("PREMISE vs HYPOTHESIS DELETION RATE")
    print("=" * 55)
    print(f"  {'Segment':<15}  {'Tokens':>8}  {'Del rate':>9}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*9}")
    print(f"  {'Premise':<15}  {prem_total:>8,}  {prem_rate:>8.1%}")
    print(f"  {'Hypothesis':<15}  {hyp_total:>8,}  {hyp_rate:>8.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse MrXLMR deletion patterns on SNLI.")
    p.add_argument("--input_file", type=str, required=True,
                   help="JSON file produced by get_deletion_patterns.py")
    p.add_argument("--n_print", type=int, default=20,
                   help="Number of colored examples to print to terminal.")
    p.add_argument("--top_n_tokens", type=int, default=15,
                   help="Number of top/bottom tokens to show.")
    p.add_argument("--min_token_count", type=int, default=10,
                   help="Minimum occurrences for per-token rate to be included.")
    p.add_argument("--output_dir", type=str, default="analysis/figures")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input_file} ...")
    with open(args.input_file) as f:
        examples = json.load(f)
    print(f"  {len(examples)} examples loaded.")

    # Overall stats
    n_correct = sum(e["correct"] for e in examples)
    all_deleted = [
        sum(d and m for d, m in zip(e["deletion_mask"], e["attention_mask"]))
        / max(sum(e["attention_mask"]), 1)
        for e in examples
    ]
    print(f"\nOverall accuracy:        {n_correct / len(examples):.4f}")
    print(f"Mean deletion rate:      {sum(all_deleted) / len(all_deleted):.4f}")

    # 1. Colored examples
    print_colored_examples(examples, n=args.n_print)

    # 2. Token-type rates
    rates = compute_token_type_rates(examples)
    print_token_type_table(rates)

    # 3. Per-token top/bottom
    deleted_c, kept_c = compute_per_token_rates(examples, min_count=args.min_token_count)
    print_top_tokens(deleted_c, kept_c, n=args.top_n_tokens)

    # 4. Bar chart
    base = os.path.splitext(os.path.basename(args.input_file))[0]
    plot_deletion_by_type(rates, args.output_dir, filename=f"{base}_by_type.pdf")

    # 5. SNLI premise vs hypothesis
    prem_rate, hyp_rate, prem_total, hyp_total = compute_premise_hypothesis_rates(examples)
    print_premise_hypothesis_table(prem_rate, hyp_rate, prem_total, hyp_total)
    plot_premise_vs_hypothesis(prem_rate, hyp_rate, args.output_dir,
                               filename=f"{base}_premise_vs_hyp.pdf")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()