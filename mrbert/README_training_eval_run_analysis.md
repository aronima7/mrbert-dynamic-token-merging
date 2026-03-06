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
-------------
❯ what would be the parameters to tweak to get 0% mrbert to be the same as bert baseline without bypassing the gate?                                                                                       
                                                                                                                     
⏺ The root problem is the gate initialization, not the training hyperparameters. Understanding this makes the levers clear:                                                                                
                                                                                                                                                                                                         
  What's happening:                                                                                                                                                                                        
  bias_init = +10  →  sigmoid(10) ≈ 1.0  →  attention_bias ≈ -30  (max deletion from step 0)
  The gate starts maximally deleting and has to learn its way back to neutral. With deletion_loss_weight=0.1, the pink run moved 85% → 65% over 600 steps — about 20% reduction. To reach 0% needs ~4× more
   signal.        

  Parameters to tweak:

  ┌────────────────────────┬─────────────────────┬───────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │       Parameter        │ Current (0_1wt run) │  Recommended  │                                      Why                                       │
  ├────────────────────────┼─────────────────────┼───────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --deletion-loss-weight │ 0.1                 │ 1.0 – 2.0     │ 10–20× stronger push toward 0% deletion                                        │
  ├────────────────────────┼─────────────────────┼───────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --regularizer-delay    │ 100                 │ 0             │ Don't waste steps — gate correction needs to start immediately                 │
  ├────────────────────────┼─────────────────────┼───────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --no-use-pi-controller │ off                 │ add this flag │ PI controller adapts α based on error; with fixed high α it's more predictable │
  └────────────────────────┴─────────────────────┴───────────────┴────────────────────────────────────────────────────────────────────────────────┘

  modal run --detach train_modal.py::main \
    --model-type MrBERT \
    --task question_answering \
    --dataset-name local_tydiqa \
    --num-epochs 3 --max-steps -1 --batch-size 16 \
    --target-deletion-rate 0.0 \
    --deletion-loss-weight 1.0 \
    --regularizer-delay 0 \
    --no-use-pi-controller \
    --mode training-and-eval \
    --wandb-project mrbert-tydiqa \
    --wandb-run-name mrbert-tydiqa-0pct-1wt

  The honest caveat: there's a fundamental tension — a deletion_loss_weight high enough to overcome the bias=10 initialization in only 600 steps may dominate the task loss early in training, hurting QA
  learning. The gate is fighting against its initialization while simultaneously trying to learn span prediction. You'd need to watch train/cross_entropy_loss vs train/delete_gate_loss to check they're
  balanced.

  The clean fix is a one-line code change in _init_delete_gates() — change bias=10 to bias=-10 when target_deletion_rate=0. That makes sigmoid(-10) ≈ 0 → attention_bias ≈ 0 from step 0, so even a small
  deletion loss weight keeps the gate neutral with no initialization fight. Want me to add a gate_init_bias config parameter for this?

⏺ Do not change the bias. My suggestion was based on a sign error. The math shows the current initialization is already correct:                                                                           
                                                                                                                                                                                                           
  ScaledSigmoid(x) = -30 × sigmoid(-x)                                                                                                                                                                     
                                                                                                                                                                                                           
  bias = +10  →  logit ≈ +10  →  -30 × sigmoid(-10) ≈ -30 × 0.00005 ≈  0.0   → KEEP  ✓                                                                                                                     
  bias = -10  →  logit ≈ -10  →  -30 × sigmoid(+10) ≈ -30 × 0.9999  ≈ -30.0  → DELETE ✗                                                                                                                    
                                                                                                                                                                                                         
  bias=+10 already starts the gate at near-zero attention bias (keep all tokens). Changing to -10 would do the opposite — start at maximum deletion. It would make the 0% run worse, not better.

  ---
  What's actually causing the collapse

  The gate initializes correctly at ≈0, but collapses to 85% deletion within the first 50 steps due to two forces:

  1. Gumbel noise (use_gumbel_noise=True is hardcoded in train_mrbert.py:703).
  Gumbel distribution has a heavy left tail — occasional large negative samples push logit below 0, which causes a large spike in gate value (toward -30). At logit=-5, the sigmoid gradient is large, so
  the task loss gradient locks the gate there permanently.

  2. QA task loss gradient pushes the gate to delete tokens.
  Suppressing most context tokens makes the softmax over 384 positions sharper/lower-variance in early training. The model "learns" to collapse attention — a local minimum that looks good short-term but
  destroys performance.

  Without a deletion loss to oppose these forces, the gate never recovers.

  ---
  Actual fixes without bypass_gate

  ┌──────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
  │               Fix                │                                   How                                   │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Disable Gumbel noise             │ Pass use_gumbel_noise=False in config (currently hardcoded to True)     │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Strong deletion loss from step 0 │ --deletion-loss-weight 1.0 --regularizer-delay 0 --no-use-pi-controller │
  └──────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘

  Disabling Gumbel noise is the more targeted fix — it removes the stochastic kick that starts the collapse. Want me to add use_gumbel_noise as a training arg so it can be controlled from the CLI? Right
  now it's hardcoded to True regardless of target deletion rate.

-------------

