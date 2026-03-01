# MrBERT Training Run Plan

A structured plan for training runs with different parameter variations, including the intuition behind each run and the expected metrics outcomes. Runs are ordered from foundational baselines to ablation studies.

For metric interpretation and success criteria, see `README_training_eval_run_analysis.md`.

---

## Overview

| Run | Model | Purpose | Key variation |
|---|---|---|---|
| [Run 1](#run-1-bert-baseline) | BERT | Baseline reference | No delete gate |
| [Run 2](#run-2-mrbert-standard) | MrBERT | Primary model | Default settings, 30% deletion |
| [Run 3](#run-3-mrbert-higher-deletion-rate) | MrBERT | Compression vs accuracy tradeoff | 50% deletion target |
| [Run 4](#run-4-mrbert-lower-deletion-rate) | MrBERT | Conservative compression | 20% deletion target |
| [Run 5](#run-5-mrbert-no-pi-controller) | MrBERT | PI controller ablation | Fixed α, no dynamic adjustment |
| [Run 6](#run-6-mrbert-earlier-gate-layer) | MrBERT | Gate placement ablation | Gate at layer 1 instead of 3 |
| [Run 7](#run-7-mrbert-later-gate-layer) | MrBERT | Gate placement ablation | Gate at layer 6 instead of 3 |
| [Run 8](#run-8-mrbert-random-gate) | MrBERT | Learned vs random deletion | Random gate, 30% deletion |

---

## Run 1: BERT Baseline

**Command:**
```bash
modal run --detach train_modal.py \
  --model-type BERT \
  --max-steps -1 \
  --num-epochs 3 \
  --mode training-and-eval
```

**Intuition:**
This is the reference point for all other runs. Standard BERT fine-tuned on SNLI with no deletion. Every subsequent MrBERT run will be measured against this. Without this baseline, there is no way to know whether MrBERT's accuracy drop (if any) is an acceptable tradeoff for the efficiency gain.

**Expected metrics:**
| Metric | Expected value |
|---|---|
| `eval/accuracy` | ~90–91% (standard BERT on SNLI) |
| `cross_entropy_loss` | Smooth decay to ~0.25–0.30 by epoch 3 |
| `accuracy` (train) | ~91–92% |
| `percent_non_pad_deleted_tokens` | N/A — no gate |
| `new_seq_len` | N/A — no gate |

**What to look for:**
- This establishes the accuracy ceiling. Any MrBERT run that matches within ~1–2% is a successful compression.
- If BERT baseline underperforms (< 88%), it likely indicates a data or training setup issue that would affect all subsequent runs.

---

## Run 2: MrBERT Standard

**Command:**
```bash
modal run --detach train_modal.py \
  --model-type MrBERT \
  --max-steps -1 \
  --num-epochs 3 \
  --target-deletion-rate 0.3 \
  --mode training-and-eval
```

**Intuition:**
This is the primary MrBERT run. The 30% deletion target is chosen as a moderate tradeoff: enough compression to reduce compute meaningfully (~49% reduction in attention compute for layers 4–11, since attention scales as O(n²)), while being conservative enough that the model retains most of the informative tokens. The PI controller is enabled to dynamically stabilize the deletion rate at exactly 30%.

**Expected metrics:**
| Metric | Expected value |
|---|---|
| `eval/accuracy` | 88–90% (within ~1–2% of BERT baseline) |
| `percent_non_pad_deleted_tokens` | Converges to 27–33% by epoch 2 |
| `delete_gate_std` | > 1.0 — gate is discriminating between tokens |
| `new_seq_len` | ~19–22 tokens (70% of avg non-pad length) |
| `delete_gate_loss_coeff` | Stabilizes after initial oscillation |
| `cross_entropy_loss` | Slightly higher than baseline but similar curve shape |

**What to look for:**
- Compare `eval/accuracy` directly against Run 1. The gap is the cost of 30% compression.
- `delete_gate_std` rising over training indicates the gate is learning which tokens to delete, not just uniformly suppressing everything.
- If `percent_non_pad_deleted_tokens` never reaches 30%, the PI controller gains are too conservative — see Run 5 for comparison.

---

## Run 3: MrBERT Higher Deletion Rate

**Command:**
```bash
modal run --detach train_modal.py \
  --model-type MrBERT \
  --max-steps -1 \
  --num-epochs 3 \
  --target-deletion-rate 0.5 \
  --mode training-and-eval
```

**Intuition:**
Deleting 50% of tokens is a much more aggressive compression. At this rate, the model must learn to identify the most informative ~half of each sequence and discard the rest. This run tests the upper bound of MrBERT's compression capability. It is expected to hurt accuracy more than Run 2, but if the gate is truly learning meaningful deletions (preserving verbs, key nouns, negations), the accuracy drop may be smaller than expected. In NLI, many tokens are redundant — determiners, prepositions, repeated words — so there is reason to believe 50% deletion is viable.

**Expected metrics:**
| Metric | Expected value |
|---|---|
| `eval/accuracy` | 85–88% (larger gap vs baseline than Run 2) |
| `percent_non_pad_deleted_tokens` | Converges to ~47–53% |
| `new_seq_len` | ~12–15 tokens (50% of avg non-pad length) |
| `delete_gate_std` | Should be higher than Run 2 — gate is forced to make harder decisions |
| `cross_entropy_loss` | Higher than Run 2 throughout training |

**What to look for:**
- Compare `eval/accuracy` gap vs Run 2. If accuracy drops more than 3–4% compared to the baseline, 50% deletion is too aggressive for this architecture.
- Compare `delete_gate_std` to Run 2. Higher std here means the gate is making sharper keep/delete decisions, which is a good sign.
- Watch for gate collapse early in training — with a 50% target the PI controller will apply stronger pressure, which can destabilize the gate in the first epoch.

---

## Run 4: MrBERT Lower Deletion Rate

**Command:**
```bash
modal run --detach train_modal.py \
  --model-type MrBERT \
  --max-steps -1 \
  --num-epochs 3 \
  --target-deletion-rate 0.2 \
  --mode training-and-eval
```

**Intuition:**
Deleting only 20% of tokens is a conservative setting. The expectation is that accuracy should be very close to the BERT baseline (the model retains 80% of tokens, so very little information is lost), while still providing a modest compute reduction (~36% reduction in attention compute for layers 4–11). This run establishes the lower end of the accuracy-compression tradeoff curve. It is also a sanity check: if 20% deletion significantly hurts accuracy, something is fundamentally wrong with how the gate interacts with the task head.

**Expected metrics:**
| Metric | Expected value |
|---|---|
| `eval/accuracy` | 89–91% (very close to BERT baseline) |
| `percent_non_pad_deleted_tokens` | Converges to ~17–23% |
| `new_seq_len` | ~22–27 tokens (80% of avg non-pad length) |
| `delete_gate_std` | Lower than Run 2 — less pressure to discriminate |
| `cross_entropy_loss` | Very close to baseline Run 1 |

**What to look for:**
- If accuracy matches the BERT baseline within 0.5%, the gate is truly selecting only redundant tokens to delete — a very good result.
- Together with Runs 2 and 3, this run completes the accuracy-vs-compression tradeoff curve: 20% → 30% → 50% deletion vs. accuracy loss.

---

## Run 5: MrBERT No PI Controller

**Command:**
```bash
modal run --detach train_modal.py \
  --model-type MrBERT \
  --max-steps -1 \
  --num-epochs 3 \
  --mode training-and-eval \
  --no-use-pi-controller
```

**Intuition:**
Without the PI controller, `α` (the deletion loss weight) stays fixed at `--deletion_loss_weight` (default `0.01`) for the entire run. The model has no feedback mechanism to correct the deletion rate toward 30% — it will find whatever equilibrium the fixed `α` produces. This run directly tests the value of the PI controller. Two outcomes are possible:

1. The deletion rate happens to land near 30% anyway → the PI controller adds little value for this task
2. The deletion rate drifts far from 30% (too high or too low) → the PI controller is essential for targeted compression

The intuition from the MrT5 paper is that a fixed α is fragile: even small changes in the learning dynamics can push the gate to collapse or become inactive. The PI controller provides robustness.

**Expected metrics:**
| Metric | Expected value |
|---|---|
| `eval/accuracy` | Similar to Run 2, or worse if gate drifts |
| `percent_non_pad_deleted_tokens` | Unpredictable — likely lower than 30% (weak fixed α) or oscillating without correction |
| `delete_gate_loss_coeff` | Flat line at 0.01 throughout training (no adjustment) |
| `cross_entropy_loss` | May be slightly lower if less deletion pressure frees the model |

**What to look for:**
- If `percent_non_pad_deleted_tokens` drifts far from 30% (e.g. stabilizes at 10% or 60%), this confirms the PI controller is necessary to hit a specific target.
- If `eval/accuracy` is worse than Run 2 with a similar deletion rate, it suggests the PI controller's smoother α trajectory leads to better optimization.
- Compare `delete_gate_std` — a well-calibrated gate should show similar std whether or not the PI controller is used, as long as the deletion rate is similar.

---

## Run 6: MrBERT Earlier Gate Layer

**Command:**
```bash
modal run --detach train_modal.py \
  --model-type MrBERT \
  --max-steps -1 \
  --num-epochs 3 \
  --target-deletion-rate 0.3 \
  --mode training-and-eval
```

> Then pass `--delete_gate_layer 1` via extra_args, or temporarily change `delete_gate_layer` default in `configuration_mrbert.py` to `1`.

**Intuition:**
In Run 2, the gate sits at layer 3. After only 3 transformer layers, the model has built some contextual representation but not deep understanding of the sentence. Moving the gate to layer 1 means deletion decisions are made almost immediately — the model sees each token's representation after only 1 layer of context. This is a more challenging setting because the gate has less information to work with when deciding what to delete.

The hypothesis: earlier deletion means more layers (2–11) benefit from the compressed sequence, improving efficiency — but accuracy may suffer because the gate is making poorer decisions with limited context.

**Expected metrics:**
| Metric | Expected value |
|---|---|
| `eval/accuracy` | Lower than Run 2 — gate makes worse decisions with less context |
| `percent_non_pad_deleted_tokens` | Should still converge to ~30% (same PI controller) |
| `delete_gate_std` | May be lower — less contextual representation → harder to discriminate |
| `new_seq_len` | Same target as Run 2, but potentially noisier |
| `cross_entropy_loss` | Likely higher than Run 2 |

**What to look for:**
- How much does moving the gate earlier hurt accuracy? A small drop suggests the gate is learning robust token-level importance signals even from shallow representations.
- Compare `delete_gate_std` to Run 2. Lower std here would support the hypothesis that deeper representations enable better deletion decisions.

---

## Run 7: MrBERT Later Gate Layer

> Change `delete_gate_layer` default in `configuration_mrbert.py` to `6`, or pass it via extra_args.

**Intuition:**
Moving the gate to layer 6 (the halfway point of the 12-layer encoder) means the model has built rich contextual representations before making deletion decisions. The gate should make better-informed keep/delete choices. However, since only layers 7–11 (5 layers) benefit from the compressed sequence instead of layers 4–11 (8 layers), the efficiency gain is smaller.

This run tests whether gate quality (better decisions from richer representations) outweighs gate placement (earlier = more efficient).

**Expected metrics:**
| Metric | Expected value |
|---|---|
| `eval/accuracy` | Higher than Run 2 and Run 6 — better gate decisions |
| `percent_non_pad_deleted_tokens` | Converges to ~30% |
| `delete_gate_std` | Higher than Run 2 — richer representations → sharper decisions |
| `new_seq_len` | Same target as Run 2 |
| `cross_entropy_loss` | Likely lower than Run 2 (closer to BERT baseline) |

**What to look for:**
- Together with Runs 2 and 6, this completes the gate layer sweep: layer 1 → layer 3 → layer 6. The accuracy trend across these three runs tells you where the best accuracy-efficiency tradeoff point is.
- If accuracy at layer 6 is not meaningfully better than layer 3, layer 3 is the better choice because it provides more efficiency gain for the same accuracy.

---

## Run 8: MrBERT Random Gate

> Change `deletion_type` to `"random"` in `configuration_mrbert.py`, or pass via extra_args.

**Intuition:**
The random gate deletes 30% of non-special tokens at random, with no learning. This is the most important ablation because it directly answers: **does the model actually learn which tokens to delete, or is any 30% deletion equally good?**

If MrBERT (Run 2) significantly outperforms the random gate, it proves the learned gate is adding value — it is identifying and removing the least informative tokens. If performance is similar, the result is that token selection does not matter and any compression at the target rate performs the same. Either outcome is a meaningful finding.

**Expected metrics:**
| Metric | Expected value |
|---|---|
| `eval/accuracy` | Lower than Run 2 — random deletion randomly removes informative tokens |
| `percent_non_pad_deleted_tokens` | ~30% (random gate is designed to hit the target rate) |
| `delete_gate_std` | Near 0 — no learned discrimination, purely random |
| `delete_gate_average` | Constant — not learned |
| `cross_entropy_loss` | Higher than Run 2 — task signal degraded by random deletion |

**What to look for:**
- The accuracy gap between Run 2 (learned gate) and Run 8 (random gate) is the key result. A gap of 2–3%+ means the gate is genuinely learning meaningful deletions.
- If the gap is small (< 1%), it suggests the NLI task is robust to the specific tokens deleted — the model compensates via attention, and random deletion at 30% is essentially as good as learned deletion. This would be a surprising but publishable result.
- `delete_gate_std` near 0 in Run 8 vs. > 1.0 in Run 2 confirms the learned gate is doing qualitatively different work.

---

## Summary: What Each Run Tells You

| Run | Core question answered |
|---|---|
| Run 1 (BERT baseline) | What is the accuracy ceiling without any compression? |
| Run 2 (MrBERT 30%) | Can MrBERT match BERT accuracy with 30% deletion? |
| Run 3 (MrBERT 50%) | How much accuracy do we sacrifice for aggressive compression? |
| Run 4 (MrBERT 20%) | Is conservative compression essentially free in terms of accuracy? |
| Run 5 (No PI controller) | Is dynamic α adjustment necessary for stable, targeted deletion? |
| Run 6 (Gate at layer 1) | Does early deletion (more efficiency) hurt accuracy significantly? |
| Run 7 (Gate at layer 6) | Does late deletion (better decisions) recover accuracy? |
| Run 8 (Random gate) | Is the learned gate actually learning something meaningful? |

## Recommended run order

1. **Run 1** and **Run 2** in parallel — establish baseline and primary model simultaneously
2. **Run 8** — most important ablation, run early to validate the core premise
3. **Runs 3 and 4** in parallel — complete the compression tradeoff curve
4. **Run 5** — PI controller ablation
5. **Runs 6 and 7** in parallel — gate layer sweep