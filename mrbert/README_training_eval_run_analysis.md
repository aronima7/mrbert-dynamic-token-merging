# MrBERT Training Run Analysis Guide

A step-by-step guide for launching training, interpreting W&B metrics, diagnosing problems, adjusting parameters, and deciding when a run is complete.

---

## Table of Contents

1. [Concepts: Steps vs Epochs](#1-concepts-steps-vs-epochs)
2. [Launching a Run](#2-launching-a-run)
3. [W&B Metrics Reference](#3-wb-metrics-reference)
4. [Step-by-Step Analysis Workflow](#4-step-by-step-analysis-workflow)
5. [Parameter Adjustments by Symptom](#5-parameter-adjustments-by-symptom)
6. [Success Criteria](#6-success-criteria)
7. [TyDi QA — Task-Specific Considerations](#7-tydi-qa--task-specific-considerations)
8. [Pre-Deletion Blend — What It Is and Why It Matters](#8-pre-deletion-blend--what-it-is-and-why-it-matters)
9. [TyDi QA Results and Analysis](#9-tydi-qa-results-and-analysis)

---

## 1. Concepts: Steps vs Epochs

| Term | Definition |
|---|---|
| **Step** | One forward + backward pass on a single batch of 32 examples. Weights are updated once per step. |
| **Epoch** | One full pass through the entire training dataset (~549,367 SNLI examples). |

With `batch_size=32` on SNLI:

```
steps_per_epoch ≈ 549,367 / 32 ≈ 17,168
1 epoch  ≈  17,168 steps
3 epochs ≈  51,500 steps
```

**Important**: the LR scheduler decays linearly from `learning_rate` to 0 over `total_steps`. If `--max-steps` is set lower than the actual number of steps you run, the LR will hit near-zero early and training stalls. Always use `--max-steps -1` for full-epoch runs.

---

## 2. Launching a Run

### Recommended full training command (Modal)

```bash
# MrBERT with delete gate — 3 epochs, 30% deletion target
modal run --detach train_modal.py \
  --model-type MrBERT \
  --max-steps -1 \
  --num-epochs 3 \
  --target-deletion-rate 0.3

# BERT baseline — no delete gate
modal run --detach train_modal.py \
  --model-type BERT \
  --max-steps -1 \
  --num-epochs 3
```

### Key parameters in DEFAULT_ARGS (train_modal.py)

| Parameter | Default | Effect |
|---|---|---|
| `--learning_rate` | `2e-5` | Base LR for BERT weights |
| `--batch_size` | `32` | Examples per step |
| `--target_deletion_rate` | `0.3` | Fraction of non-pad tokens the gate should delete |
| `--deletion_loss_weight` | `0.01` | Initial α₀ for deletion regularizer (PI controller adjusts from here) |
| `--controller_p` | `0.01` | Proportional gain — how fast α responds to deletion rate error |
| `--controller_i` | `0.00001` | Integral gain — slow cumulative correction |
| `--regularizer_delay` | `1000` | Steps before any deletion pressure is applied |
| `--logging_steps` | `50` | How often metrics are logged to W&B |
| `--save_steps` | `1000` | How often checkpoints are saved |

---

## 3. W&B Metrics Reference

### Task performance metrics

| Metric | What it measures | Expected behavior |
|---|---|---|
| `loss` | Combined loss (task + deletion) | Should decrease steadily, then plateau |
| `cross_entropy_loss` | Classification loss only | Primary signal for task learning |
| `accuracy` | Training batch accuracy | Should rise from ~33% (random) to 85%+ |
| `eval/loss` | Validation loss | Should track training loss; divergence = overfitting |
| `eval/accuracy` | Validation accuracy | The definitive performance metric |

### Delete gate metrics

| Metric | What it measures | Expected behavior |
|---|---|---|
| `percent_non_pad_deleted_tokens` | % of non-pad tokens with gate < threshold | Should converge to `target_deletion_rate × 100` |
| `percent_deleted_tokens` | % of all tokens (including pad) deleted | Always lower than the non-pad version |
| `new_seq_len` | Average tokens kept per sequence after deletion | Should stabilize at `avg_seq_len × (1 - target_deletion_rate)` |
| `delete_gate_average` | Mean gate value across all tokens | Near 0 at initialization; rises/falls as gate learns |
| `delete_gate_std` | Spread of gate values within a sequence | Low std = gate treating all tokens similarly (bad); high std = gate is discriminating |
| `delete_gate_max_value` / `delete_gate_min_value` | Range of gate values | Max near 0 (keep), min near -30 (delete) when gate is healthy |
| `delete_gate_loss_coeff` | Current α (PI controller output) | Should stabilize once deletion rate converges |

### Optimization metrics

| Metric | What it measures | Expected behavior |
|---|---|---|
| `learning_rate` | Current LR from linear schedule | Starts at 2e-5, decays to 0 by final step |
| `epoch` | Current epoch (1-indexed) | Increments at ~17,168 step intervals |

---

## 4. Step-by-Step Analysis Workflow

### Phase 1: Steps 0–1,000 (pre-regularizer)

The model is learning the classification task with no deletion pressure.

**What to check:**
- `cross_entropy_loss` should be dropping from ~1.1 toward ~0.7
- `accuracy` should be rising from ~33% (random baseline for 3-class NLI)
- `delete_gate_average` at step 50 should be close to **0** — this confirms the gate bias initialization (bias=10) is working. If it is near -30, all tokens are being deleted from the start, which is a bug.
- `new_seq_len` should be close to the full average non-pad sequence length (~25–35 tokens for SNLI). If it is 5–6 tokens, the gate is already aggressively deleting before training has started, indicating the bias initialization failed.

**Red flags:**
- `accuracy` stuck at 33% after 500 steps → task loss not flowing correctly
- `delete_gate_average` near -30 at step 50 → gate initialization not working

---

### Phase 2: Steps 1,000–5,000 (PI controller engages)

Deletion pressure kicks in at step 1,000. The PI controller begins adjusting α to push the deletion rate toward the 30% target.

**What to check:**
- `percent_non_pad_deleted_tokens` should start rising from near 0% toward 30%
- `delete_gate_loss_coeff` (α) should start increasing from 0.01
- `new_seq_len` will drop as deletion starts — expect a temporary spike or dip as the controller stabilizes. Some oscillation is normal.
- `delete_gate_std` should be increasing — the gate is learning to discriminate between tokens worth keeping and tokens worth deleting.

**Red flags:**
- `percent_non_pad_deleted_tokens` jumps immediately to 70–80% → controller gain too high, gate is collapsing. Reduce `controller_p`.
- `delete_gate_std` stays near 0 → gate is outputting uniform values, not learning to discriminate.

---

### Phase 3: Steps 5,000–17,168 (end of epoch 1)

The gate and task head should be co-adapting.

**What to check:**
- `percent_non_pad_deleted_tokens` should be trending toward the 30% target, possibly still noisy
- `cross_entropy_loss` should still be decreasing
- `accuracy` should be in the 75–85% range by end of epoch 1 for a healthy run
- `new_seq_len` should be stabilizing around `avg_seq_len × 0.70`

---

### Phase 4: Steps 17,168–51,500 (epochs 2–3)

Convergence phase. Both task performance and deletion rate should be settling.

**What to check:**
- `eval/accuracy` (if running `--mode training-and-eval`) is the key metric — compare against BERT baseline
- `percent_non_pad_deleted_tokens` should be within ±5% of the 30% target and no longer trending
- `cross_entropy_loss` curve should be flattening
- `learning_rate` should still be non-zero — if it hits 0 before step 51,500, the scheduler was set for fewer steps (fix: use `--max-steps -1`)

---

## 5. Parameter Adjustments by Symptom

### Deletion rate never reaches target

| Symptom | Likely cause | Adjustment |
|---|---|---|
| Rate plateaus at 10–20% | Controller too slow | Increase `--controller_p` from 0.01 to 0.05 |
| Rate oscillates wildly | Controller overshooting | Decrease `--controller_p` to 0.005 |
| Rate converges slowly over many epochs | Integral term too weak | Increase `--controller_i` from 0.00001 to 0.0001 |
| Rate instantly hits 80%+ at step 1000 | Gate collapse at controller start | Increase `--regularizer_delay` to 2000–3000 |

### Gate not discriminating (low `delete_gate_std`)

The gate outputs similar values for all tokens — it isn't learning what to delete.

| Adjustment | Effect |
|---|---|
| Increase `--deletion_loss_weight` initial value | Stronger deletion pressure from the start |
| Increase `--regularizer_delay` | More time to learn task before deletion pressure — gate has more signal about what's important |
| Try `--use_gumbel_noise` | Adds exploration noise during training to prevent gate from getting stuck |

### Task performance worse than BERT baseline

| Symptom | Likely cause | Adjustment |
|---|---|---|
| `accuracy` lags behind baseline throughout training | Deletion too aggressive, hurting task signal | Reduce `--target_deletion_rate` (e.g. 0.2) |
| `accuracy` starts equal then diverges | Gate deleting informative tokens | Increase `--regularizer_delay` to let task head stabilize first |
| `eval/accuracy` drops after epoch 1 | Overfitting | Reduce `--num_epochs` or add weight decay |

### Learning rate issues

| Symptom | Likely cause | Adjustment |
|---|---|---|
| Loss stops improving mid-run despite no plateau | LR decayed to near zero too early | Use `--max-steps -1` instead of a hardcoded step count |
| Loss unstable, large spikes | LR too high | Reduce `--learning_rate` from 2e-5 to 1e-5 |

---

## 6. Success Criteria

A training run can be considered successful when **all** of the following are met:

### 1. Task performance
- `eval/accuracy` ≥ **89%** on SNLI validation set
- MrBERT `eval/accuracy` is within **1–2%** of the BERT baseline (acceptable cost for compression)
- `eval/accuracy` has **plateaued** — the last 5,000 steps show less than 0.5% improvement

### 2. Delete gate convergence
- `percent_non_pad_deleted_tokens` is within **±3%** of the target deletion rate (e.g. 27–33% for a 30% target)
- `percent_non_pad_deleted_tokens` is **stable** — variance over the last 2,000 steps is less than ±5%
- `delete_gate_std` is **non-trivial** (> 1.0) — the gate is making meaningful distinctions between tokens, not uniformly scoring everything

### 3. Training stability
- `cross_entropy_loss` has **plateaued** (flattened curve, not still trending down steeply)
- `delete_gate_loss_coeff` (α) has **stabilized** — the PI controller is no longer making large adjustments
- `new_seq_len` is **stable** — consistent effective sequence length across batches

### 4. Comparison against baseline
- Run a BERT baseline with identical hyperparameters (`--model-type BERT`) and compare:
  - `eval/accuracy`: MrBERT should be within 2% of BERT
  - Effective compute per forward pass: `new_seq_len / original_seq_len` should be significantly < 1.0 (i.e. MrBERT is actually faster/cheaper per inference step)

### Summary table

| Metric | Success threshold |
|---|---|
| `eval/accuracy` | ≥ 89% |
| Gap vs BERT baseline | ≤ 2% |
| `percent_non_pad_deleted_tokens` | within ±3% of target |
| `delete_gate_std` | > 1.0 |
| `cross_entropy_loss` slope | < 0.001 per 1000 steps |
| `new_seq_len` stability | ± < 1 token over last 2000 steps |

### When to stop early

Stop the run and adjust parameters if by the end of epoch 1 (~17,168 steps):
- `accuracy` < 60% (task not learning)
- `percent_non_pad_deleted_tokens` > 70% consistently (gate has collapsed)
- `delete_gate_std` < 0.1 (gate not discriminating)
- `new_seq_len` < 5 tokens (almost all content deleted)

---

## 7. TyDi QA — Task-Specific Considerations

### Dataset stats

| Property | Value |
|---|---|
| Training examples (English GoldP) | ~3,200 |
| Max sequence length | 384 tokens |
| Batch size | 16 |
| Steps per epoch | ~200 |
| Total steps (3 epochs) | ~600 |

TyDi QA GoldP passages are pre-cropped to answer-containing windows, so the sliding window rarely fires (~1.02× expansion ratio vs ~1.22× for SQuAD). This is why the run completes in ~600 steps rather than the ~850 initially expected.

### Why QA is harder than classification for deletion gates

For classification tasks (SNLI, SST-2, MRPC, IMDB), the task head reads only the `[CLS]` token:

```python
pooled_output = self.pooler(sequence_output[:, 0, :])  # position 0 only
```

`[CLS]` is hardcoded to never be deleted, so the gate cannot hurt classification performance regardless of how aggressively it deletes the rest of the sequence.

For extractive QA, the task head scores **every token position** to find the answer span:

```python
logits = self.qa_outputs(sequence_output)   # (batch, seq, 2) — all positions
start_logits, end_logits = logits.split(1, dim=-1)
```

When the gate soft-deletes an answer span token at layer 3, that token's final-layer (layer 11) representation is corrupted — it has had 8 layers of near-zero incoming attention and carries almost no useful signal. The span head then reads a corrupted value at the answer position and assigns low logits there, causing span prediction to fail even when the correct answer is present in the passage.

**Probabilistic exposure at 30% deletion rate:**
For a 10-token answer span, P(at least one answer token deleted) ≈ 1 − (0.7)^10 ≈ 97%. This means virtually all QA examples in training and evaluation are affected by gate-induced answer corruption when using the default 30% target.

### Gate collapse in QA runs

The original `mrbert-tydiqa-30pct` run exhibited **gate collapse**: instead of converging to the 30% target, `percent_non_pad_deleted_tokens` jumped to ~61% and stayed there. Two mechanisms cause this:

1. **Gumbel noise** (`use_gumbel_noise=True`): Gumbel distribution has a heavy left tail. Occasional large negative noise samples push gate logits below zero, creating a large gradient that locks the gate into high deletion. At step 50, before the deletion loss activates, the gate can collapse from ~0% to 80%+ deletion.

2. **QA task gradient**: In early training, suppressing most context tokens (attention bias −30) reduces the effective attention pool from 384 positions to ~10, making the softmax sharper. This accidentally lowers cross-entropy loss early on — a local minimum. The model "learns" that deleting most tokens is useful, and the gate gets stuck.

| Run | Expected deletion | Actual deletion | Outcome |
|---|---|---|---|
| `mrbert-tydiqa-30pct` (original) | 30% | ~61% | Gate collapsed; EM ≈ 0.10 |
| `mrbert-tydiqa-30pct-predel` (layer 3 + blend) | 30% | ~25% | Near-target; EM ≈ 0.30 |
| `mrbert-tydiqa-30pct-layer9-predel` (layer 9 + blend) | 30% | ~22% | Near-target; EM ≈ 0.35 |

### Step-by-step analysis for TyDi QA

Because TyDi QA only has ~600 total steps (3 epochs), the phases are compressed:

**Steps 0–100 (pre-regularizer)**

- `cross_entropy_loss` should drop from ~4.5 toward ~3.0
- `eval/span_em`, `eval/start_acc`, `eval/end_acc` are logged after each epoch — expect near zero before training properly starts
- `delete_gate_average` near 0 confirms correct initialization (bias=+10 → gate ≈ 0 → keep all tokens)
- `new_seq_len` near 384 — no deletion yet

Red flag: `delete_gate_average` at −25 or below within step 50 → gate is already collapsing via Gumbel noise. Stop and rerun with `--no-use-gumbel-noise`.

**Steps 100–300 (PI controller engages)**

- `percent_non_pad_deleted_tokens` should start rising toward the target
- `delete_gate_std` increasing = healthy (gate learning to discriminate)
- `delete_gate_loss_coeff` (α) rising from initial value

Red flag: `percent_non_pad_deleted_tokens` shoots to 50%+ within 50 steps of the controller engaging → Reduce `--controller-p` or increase `--regularizer-delay`.

**Steps 300–600 (convergence)**

- `percent_non_pad_deleted_tokens` should stabilize within ±5% of target
- `eval/span_em` after epoch 3 is the definitive metric — compare against BERT baseline
- `delete_gate_std` should plateau at 10–13 (healthy) vs staying at 8 (flat/uncalibrated)
- `test/cross_entropy_loss` ≤ 2.0 with pre-deletion blend; ≥ 2.5 without = failure signal

### QA-specific parameters

| Parameter | Classification default | QA recommended | Reason |
|---|---|---|---|
| `--batch-size` | 32 | 16 | 384-token sequences need smaller batch |
| `--regularizer-delay` | 1000 | 100 | Only ~600 total steps; delay must scale proportionally |
| `--delete-gate-layer` | 3 | 9 | Later gate sees deeper question-context interaction |
| `--use-pre-deletion-blend` | True | True | Critical for QA — see Section 8 |
| `--target-deletion-rate` | 0.3 | 0.3 | Layer 9 + blend makes 30% viable; consider 0.1 for safer operation |

### Success criteria for TyDi QA

| Metric | BERT baseline | Acceptable | Good |
|---|---|---|---|
| `test/span_em` | ~0.38–0.40 | ≥ 0.30 | ≥ 0.35 |
| `test/start_acc` | ~0.50–0.56 | ≥ 0.40 | ≥ 0.46 |
| `test/end_acc` | ~0.50 | ≥ 0.40 | ≥ 0.48 |
| `percent_non_pad_deleted_tokens` | — | within ±8% of target | within ±5% of target |
| `delete_gate_std` | — | > 9 | > 11 |
| `test/cross_entropy_loss` | ~1.85 | ≤ 2.1 | ≤ 1.95 |

---

## 8. Pre-Deletion Blend — What It Is and Why It Matters

### The problem

When a token is soft-deleted at layer `delete_gate_layer`, it is still physically present in the sequence (soft deletion adds an attention bias of −30, it does not remove the token). Over the remaining `12 - delete_gate_layer` encoder layers, that token receives near-zero incoming attention from the rest of the sequence. Its final-layer (layer 11) representation degrades — it no longer reflects what the token means in context. For a QA head that scores every position, this produces near-zero logits at the answer span location.

### The fix

Before the gate fires, save a copy of the token representations (the last fully-attended state). For any token the gate deletes, substitute this saved representation into the final output going to the task head.

The substitution is a **soft blend** proportional to the gate's deletion strength:

```
deletion_weight = clamp( -gate_mask / |sigmoid_mask_scale|, 0, 1 )
output = (1 - deletion_weight) × layer_11_hidden + deletion_weight × pre_deletion_hidden
```

| Gate value | `deletion_weight` | What the task head sees |
|---|---|---|
| 0.0 (kept) | 0.0 | Pure layer-11 representation — deep, rich, fully attended |
| −30.0 (fully deleted) | 1.0 | Pure pre-deletion representation — last fully-attended state |
| −15.0 (partially deleted) | 0.5 | Interpolated blend |

This is **structurally identical at training and test time** — no ground truth is needed. The blend weight comes directly from the gate values, which are computed in every forward pass.

### Impact by task type

| Task | Impact | Reason |
|---|---|---|
| Sequence classification (SNLI, SST-2, MRPC, IMDB) | None | Head uses only `[CLS]`; `[CLS]` is never deleted → `deletion_weight` always 0 |
| QA (TyDi QA, SQuAD) | Significant | Head needs all positions; deleted answer tokens get valid representations |
| Token classification (NER) | Same benefit as QA | Head needs all positions |
| MLM | Minor | Averaged over many positions; less critical |

### Configuration

Pre-deletion blend is controlled by `MrBertConfig.use_pre_deletion_blend` (default: `True`).

To run **without** blend (ablation):
```bash
modal run --detach train_modal.py::main \
  --task question_answering --dataset-name local_tydiqa \
  --model-type MrBERT --num-epochs 3 --batch-size 16 \
  --target-deletion-rate 0.3 --regularizer-delay 100 \
  --no-use-pre-deletion-blend \
  --wandb-run-name mrbert-tydiqa-30pct-nopredel --wandb-project mrbert-tydiqa
```

To run **with** blend (default — no flag needed):
```bash
modal run --detach train_modal.py::main \
  --task question_answering --dataset-name local_tydiqa \
  --model-type MrBERT --num-epochs 3 --batch-size 16 \
  --target-deletion-rate 0.3 --regularizer-delay 100 \
  --wandb-run-name mrbert-tydiqa-30pct-predel --wandb-project mrbert-tydiqa
```

For a deeper code-level explanation, see `README_tydiqa_predeletion.md`.

---

## 9. TyDi QA Results and Analysis

### Results table

| Run | `test/span_em` | `test/start_acc` | `test/end_acc` | Deletion Rate | CE Loss |
|---|---|---|---|---|---|
| BERT baseline | ~0.40 | ~0.45 | ~0.50 | — | ~1.85 |
| MrBERT 30% layer3 (no blend) | ~0.10 | ~0.33 | ~0.25 | ~61% (collapsed) | ~2.80 |
| MrBERT 30% layer3 + blend | ~0.30 | ~0.40 | ~0.46 | ~25% | ~1.95 |
| MrBERT 30% layer9 + blend | ~0.35 | ~0.44 | ~0.50 | ~22% | ~1.90 |

### Key findings

**1. Pre-deletion blend is essential for QA**

Without it, the layer-3 gate collapses (61% deletion) and span_em falls to 0.10 — a 4× gap vs the BERT baseline. Enabling the blend with the same 30% target recovers span_em to 0.30 (layer 3) and 0.35 (layer 9). The cross-entropy loss drops from 2.80 to ~1.95, nearly matching BERT's 1.85.

**2. Later gate layer (9 vs 3) further closes the gap**

Moving the gate to layer 9 gives it 9 full BERT layers of question-context interaction before deciding what to delete. At layer 9, deletion decisions at both train and test time are more informed. Result: span_em improves to 0.35 and end_acc exactly matches BERT (0.50). The residual gap in span_em (~0.05) is largely because ~22% of tokens are still deleted, occasionally including answer spans.

**3. Gate learns proper discrimination only when blend is active**

`train/delete_gate_std` tells the story:
- No predel (original): std stays flat at ~8 — gate stuck, no meaningful keep/delete distinction
- Layer 3 + blend: std rises from 8 → 12–13 — bimodal distribution forming (clear keep/delete decisions)
- Layer 9 + blend: std rises from 8 → 10–11 — similar healthy pattern

The blend provides a richer gradient signal back through deleted token positions, enabling the gate to learn structured deletion rather than collapsing to a uniform high-deletion state.

**4. Actual deletion rate tracks target only with blend**

`test/percent_non_pad_deleted_tokens`:
- Original 30% (no blend): ~61% — gate collapse, 2× over-target
- Layer 3 + blend: ~25% — near target, PI controller functioning
- Layer 9 + blend: ~22% — near target, even more calibrated

**5. Throughput gap remains — soft deletion limitation**

`test/samples_per_second`: BERT ~165 vs all MrBERT variants ~107. Soft deletion adds the attention bias computation overhead without removing any FLOPs from the sequence length. The actual efficiency story requires hard deletion (`--hard-delete-train-prob > 0`) to physically reduce sequence length and reduce matrix multiply costs.

### Remaining gap analysis

The ~0.05 span_em gap between MrBERT layer9+blend (0.35) and BERT (0.40) has a clear cause: at 22% deletion, a 3-token answer span is still deleted in ~50% of examples. Reducing the target deletion rate to 10% would cut this to ~27%, likely closing most of the remaining gap. Future runs to try:

```bash
# Conservative deletion — close the remaining gap
modal run --detach train_modal.py::main \
  --task question_answering --dataset-name local_tydiqa \
  --model-type MrBERT --num-epochs 3 --batch-size 16 \
  --delete-gate-layer 9 \
  --target-deletion-rate 0.1 --regularizer-delay 100 \
  --wandb-run-name mrbert-tydiqa-10pct-layer9-predel --wandb-project mrbert-tydiqa

# Hard deletion — actually reduce computation
modal run --detach train_modal.py::main \
  --task question_answering --dataset-name local_tydiqa \
  --model-type MrBERT --num-epochs 3 --batch-size 16 \
  --delete-gate-layer 9 \
  --target-deletion-rate 0.3 --regularizer-delay 100 \
  --hard-delete-train-prob 0.5 \
  --wandb-run-name mrbert-tydiqa-30pct-layer9-predel-hd --wandb-project mrbert-tydiqa
```                                                                                       
