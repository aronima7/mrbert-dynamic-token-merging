# MrDiffusion-SEDD on Modal

## Prerequisites

```bash
pip install modal
modal token set        # one-time auth

# W&B secret (required for training)
modal secret create wandb-secret WANDB_API_KEY=your_key_here
modal secret list      # verify
```

## Local training

Use local training for fast iteration and debugging before committing to a Modal run.
Requires a GPU (or CPU for smoke tests with tiny batch sizes).

```bash
cd mrdiffusion-sedd

# Smoke test (CPU, ~2 min) — verifies imports and forward pass
python train_mrdiffusion.py --max_steps 20 --logging_steps 5 --batch_size 2

# Baseline (no gate) from pretrained weights
python train_mrdiffusion.py \
  --no_delete_gate \
  --pretrained_from louaaron/sedd-small \
  --max_steps 500 --logging_steps 50

# Soft deletion, sigma-conditioned gate, continued pretraining
python train_mrdiffusion.py \
  --pretrained_from louaaron/sedd-small \
  --deletion_mode soft \
  --deletion_type scaled_sigmoid \
  --delete_gate_layer 3 \
  --target_deletion_rate 0.3 \
  --deletion_loss_weight 0.1 \
  --max_steps 500 --logging_steps 50

# Hard deletion
python train_mrdiffusion.py \
  --pretrained_from louaaron/sedd-small \
  --deletion_mode hard \
  --delete_gate_layer 3 \
  --target_deletion_rate 0.3 \
  --deletion_loss_weight 0.1 \
  --max_steps 500 --logging_steps 50

# Sigma-dependent deletion rate schedule
python train_mrdiffusion.py \
  --pretrained_from louaaron/sedd-small \
  --deletion_mode soft \
  --gate_sigma_conditioned \
  --deletion_rate_schedule linear_sigma \
  --r_min 0.05 --r_max 0.5 \
  --max_steps 500 --logging_steps 50

# Disable W&B for quick local runs
python train_mrdiffusion.py \
  --pretrained_from louaaron/sedd-small \
  --max_steps 200 --logging_steps 20 \
  --disable_wandb
```

---

## Local evaluation

Requires a checkpoint on the local filesystem. Download from the volume first
(`modal volume get ...`) or use a locally trained checkpoint.

```bash
cd mrdiffusion-sedd

# Zero-shot NELBO perplexity (all five datasets, hard deletion)
python evaluation/eval_zero_shot.py \
  --checkpoint ./local_final/checkpoint.pt

# Specific datasets only
python evaluation/eval_zero_shot.py \
  --checkpoint ./local_final/checkpoint.pt \
  --datasets wikitext-2 wikitext-103

# MAUVE score (requires: pip install mauve-text)
python evaluation/eval_mauve.py \
  --checkpoint ./local_final/checkpoint.pt \
  --n_gen 200 --n_ref 100 --num_steps 128 \
  --output_dir ./mauve_results

# Gate behavior: deletion rate vs sigma, value distributions
python evaluation/eval_gate_behavior.py \
  --checkpoint ./local_final/checkpoint.pt \
  --output_dir ./gate_analysis

# FLOPs analysis (analytical only, no checkpoint needed)
python evaluation/eval_flops.py \
  --deletion_rate 0.3 --seq_len 1024

# FLOPs + wall-clock profiling (requires checkpoint)
python evaluation/eval_flops.py \
  --checkpoint ./local_final/checkpoint.pt \
  --deletion_rate 0.3 --profile

# Pareto curve: gen perplexity vs FLOPs
python evaluation/eval_pareto.py \
  --mr_checkpoint ./local_mr/checkpoint.pt \
  --baseline_checkpoint ./local_baseline/checkpoint.pt \
  --output_dir ./pareto_results
```

All eval scripts accept `--deletion_mode soft|hard` (default: `hard` for
generation/scoring evals, `soft` for gate behavior analysis).

---

## Modal quick start

```bash
cd mrdiffusion-sedd

# Smoke test: 200 steps, verifies the container runs and gate is active
modal run train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --max-steps 200 \
  --eval-batch-size 16 \
  --disable-wandb
```

Expect `deletion_rate: ~0.11` at step 50 (gate active from the start) rising toward 0.3 as LR warms up.

---

## Memory constraints on A100-40GB

The SEDD model uses `seq_len=1024`. The default `eval.batch_size=512` from the
SEDD config causes OOM on A100-40GB. **`--eval-batch-size 16` is required for
all Modal training runs.**

| Parameter | Value | Notes |
|---|---|---|
| `--batch-size` | `32` | Default; fits fine at seq_len=1024 |
| `--eval-batch-size` | `16` | Required — default 512 OOMs during eval |

---

## Baseline runs (no delete gate)

Trains vanilla SEDD without any gate. Use as the comparison point for all gate experiments.

```bash
# Baseline — verify loss decreases and W&B is logging (~1 hr, ~$4)
modal run --detach train_modal_mrdiffusion.py::main \
  --no-delete-gate \
  --pretrained-from louaaron/sedd-small \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name baseline-pretrained-20k
```

---

## Delete gate experiments (continued pretraining)

All gate experiments start from `louaaron/sedd-small` pretrained weights so only
the gate (and optionally the transformer) needs to learn. The pretrained transformer
blocks are loaded via `--pretrained-from`; the gate starts from random init.

All runs use `--save-steps 1000` so `eval/generative_perplexity` fires at steps
5k, 10k, 15k, and 20k — without this flag it defaults to 25k and never triggers.

### Soft deletion (recommended starting point)

Soft deletion applies a large negative attention bias to deleted tokens — tokens
remain in the sequence but are invisible to subsequent attention layers. Fully
differentiable; no hard decisions during training.

```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name soft-gate-layer3-30pct-20k
```

### Hard deletion

Hard deletion physically removes tokens below `--deletion-threshold` from the
sequence. Surviving tokens are processed by compressed-phase blocks, then restored
to full length before the output layer (via `restore_gate_layer`).

```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode hard \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name hard-gate-layer3-30pct-20k
```

### Hard deletion with position-aware RoPE

By default, after hard deletion the surviving tokens are re-indexed to contiguous
positions `[0, 1, ..., L_kept-1]` before RoPE is recomputed. This is wrong: a token
originally at position 5 gets a position-2 encoding if it is the third survivor,
corrupting relative position information between survivors during the compressed phase.

`--rope-original-positions` fixes this: survivors retain their original position
encodings (`[0, 2, 5, ...]` etc.) by computing RoPE directly from the kept positions
rather than re-indexing. Because FlashAttention's rotary apply only accepts a single
shared position sequence per batch (no per-example positions), the compressed-phase
blocks automatically fall back to SDPA + per-batch rotary. Phase 1 and phase 3
(full-sequence) blocks are unaffected and still use FlashAttention.

```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode hard \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --rope-original-positions \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name hard-gate-rope-original-pos-20k
```

Compare against `hard-gate-layer3-30pct-20k` (contiguous re-indexing). If
`eval/generative_perplexity` improves, position encoding was a meaningful bottleneck
for hard deletion quality.

The gate receives both the token hidden state and the sigma (noise level) embedding,
allowing it to learn noise-level-dependent deletion rates (delete more at high σ,
fewer at low σ). This is the recommended configuration per the guide.

```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --gate-sigma-conditioned \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name soft-gate-sigma-cond-20k
```

### Sigma-dependent deletion rate schedule

Instead of a fixed target rate, the deletion target varies with noise level:
`r_min` at σ≈0 (clean tokens, expensive to delete) up to `r_max` at σ≈σ_max
(fully noised, cheap to delete).

```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --gate-sigma-conditioned \
  --deletion-rate-schedule linear_sigma \
  --r-min 0.05 \
  --r-max 0.5 \
  --deletion-loss-weight 0.1 \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name soft-gate-linear-schedule-20k
```

### Gate at a later layer

Gate layer 3 (of 12) is conservative — only 3 blocks of context before deletion.
Try later layers for richer representations before the gate fires.

```bash
# Gate at layer 6 (middle of the model)
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --delete-gate-layer 6 \
  --target-deletion-rate 0.3 \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name soft-gate-layer6-30pct-20k
```

### Higher deletion rate

```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --target-deletion-rate 0.5 \
  --deletion-loss-weight 0.1 \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name soft-gate-50pct-20k
```

### PI controller for deletion rate tracking

Instead of a fixed `--deletion-loss-weight`, the PI controller dynamically adjusts
it each logging step to close the gap between actual and target deletion rate.
It activates only after gate warmup (`--gate-warmup-steps`) so the ramp phase is
unaffected.

**Hyperparameters** (defined inline in `train_mrdiffusion.py:_make_pi_controller`):
- `kp=0.5` — proportional gain (exponentially smoothed with `gamma=0.9`)
- `ki=1e-5` — integral gain (accumulates error each log step, not each train step)
- Clamped to ≥ 0 to prevent negative loss weights

The live weight is tracked as `train/deletion_loss_weight` in W&B. `--deletion-loss-weight`
sets the initial value and the warmup ramp target; the PI controller takes over after warmup.

**Local sanity check** (verify `train/deletion_loss_weight` adapts):
```bash
python train_mrdiffusion.py \
  --pretrained_from louaaron/sedd-small \
  --deletion_mode soft \
  --deletion_type scaled_sigmoid \
  --delete_gate_layer 3 \
  --target_deletion_rate 0.3 \
  --deletion_loss_weight 0.1 \
  --gate_warmup_steps 200 \
  --use_pi_controller \
  --max_steps 500 --logging_steps 50 \
  --disable_wandb
```

**Modal run** (~1 hr, ~$4):
```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --gate-warmup-steps 2500 \
  --use-pi-controller \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name soft-gate-pi-30pct-20k
```

Compare against `soft-gate-layer3-30pct-20k` (fixed weight) to see whether
adaptive weighting improves deletion rate stability and `eval/loss`.

### Frozen transformer (gate-only training)

The transformer stays frozen for the entire run (`--freeze-transformer-steps` set above
`--max-steps`). Only the gate parameters are updated. Tests whether the pretrained
transformer's representations are already sufficient for the gate to learn useful
deletions, without requiring the model to co-adapt.

```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --freeze-transformer-steps 999999 \
  --save-steps 1000 \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name soft-gate-frozen-transformer-20k
```

Compare against `soft-gate-layer3-30pct-20k` (joint training). If `eval/loss` and
`eval/generative_perplexity` are competitive, the gate is exploiting existing structure
rather than requiring co-adaptation. If quality drops, joint finetuning is load-bearing.

---

## Resuming an interrupted run

Each run auto-saves a meta-checkpoint every `--save-steps` (default 5000). To resume:

```bash
# Re-run the exact same command — the script restores from the latest checkpoint
# and skips pretrained_from initialization if a checkpoint already exists.
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --max-steps 20000 \
  --eval-batch-size 16 \
  --wandb-run-name soft-gate-layer3-30pct-20k   # same run name → same output_dir
```

---

## Downloading checkpoints and results

```bash
# List what's in the volume
modal volume ls mrdiffusion-sedd-checkpoints

# Download a full run directory
modal volume get mrdiffusion-sedd-checkpoints <run-name> ./<run-name>

# Download just the final checkpoint
modal volume get mrdiffusion-sedd-checkpoints <run-name>/final ./local_final

# Download eval results saved to the volume
modal volume get mrdiffusion-sedd-checkpoints <run-name>/mauve_results ./mauve_results
modal volume get mrdiffusion-sedd-checkpoints <run-name>/gate_analysis ./gate_analysis
modal volume get mrdiffusion-sedd-checkpoints <run-name>/pareto_results ./pareto_results
```

---

## Evaluation on Modal

All eval scripts run on an A100 with the volume mounted — no need to download
checkpoints first. Results are printed to stdout and (where applicable) written
back to the volume under `<run-name>/<output-subdir>/`.

All generation/scoring evals default to `--deletion-mode hard` to enforce the
guide's soft-train / hard-inference pattern. Gate behavior analysis defaults to
`--deletion-mode soft` to observe unperturbed gate values.

### Zero-shot NELBO perplexity

Evaluates score entropy loss on PTB, WikiText-2, WikiText-103, LM1B, AG News.
Results are printed to stdout only (no volume write).

```bash
# Default: all five datasets, hard deletion
modal run train_modal_mrdiffusion.py::eval_zero_shot_main \
  --run-name soft-gate-layer3-30pct-20k

# Specific datasets only
modal run train_modal_mrdiffusion.py::eval_zero_shot_main \
  --run-name soft-gate-layer3-30pct-20k \
  --datasets wikitext-2,wikitext-103

# Compare baseline vs gate (run separately, compare stdout)
modal run train_modal_mrdiffusion.py::eval_zero_shot_main \
  --run-name baseline-pretrained-20k
modal run train_modal_mrdiffusion.py::eval_zero_shot_main \
  --run-name soft-gate-layer3-30pct-20k
```

### MAUVE score

Measures distributional similarity between model-generated and reference text.
Saves `mauve_result.json` to the volume.

```bash
# Default: 500 generated vs 200 reference, 128 diffusion steps
modal run train_modal_mrdiffusion.py::eval_mauve_main \
  --run-name soft-gate-layer3-30pct-20k

# More samples for a tighter estimate (~2x longer)
modal run train_modal_mrdiffusion.py::eval_mauve_main \
  --run-name soft-gate-layer3-30pct-20k \
  --n-gen 1000 --n-ref 500

# Download result
modal volume get mrdiffusion-sedd-checkpoints \
  soft-gate-layer3-30pct-20k/mauve_results/mauve_result.json .
```

### Gate behavior analysis

Deletion rate vs σ (overall, masked tokens, clean tokens) and gate value
distributions at multiple noise levels. Saves JSON + `.npy` arrays to the volume.

```bash
# Default: soft mode, wikitext-2
modal run train_modal_mrdiffusion.py::eval_gate_behavior_main \
  --run-name soft-gate-layer3-30pct-20k

# Under actual inference conditions (hard mode)
modal run train_modal_mrdiffusion.py::eval_gate_behavior_main \
  --run-name soft-gate-layer3-30pct-20k \
  --deletion-mode hard

# Download results
modal volume get mrdiffusion-sedd-checkpoints \
  soft-gate-layer3-30pct-20k/gate_analysis .
```

### FLOPs analysis + wall-clock profiling

Prints analytical FLOPs (baseline vs MrSEDD) and optionally runs 100 timed
forward passes to measure actual speedup.

```bash
# Analytical FLOPs + wall-clock profiling (hard deletion = real speedup)
modal run train_modal_mrdiffusion.py::eval_flops_main \
  --run-name soft-gate-layer3-30pct-20k \
  --deletion-rate 0.3

# Compare soft vs hard wall-clock
modal run train_modal_mrdiffusion.py::eval_flops_main \
  --run-name soft-gate-layer3-30pct-20k \
  --deletion-mode soft
modal run train_modal_mrdiffusion.py::eval_flops_main \
  --run-name soft-gate-layer3-30pct-20k \
  --deletion-mode hard
```

### Pareto frontier

Generative perplexity vs total FLOPs at 32/64/128/256/512 diffusion steps,
for both MrSEDD and baseline. Saves `pareto_results.json` to the volume.

```bash
# MrSEDD vs baseline Pareto curve (main result)
modal run train_modal_mrdiffusion.py::eval_pareto_main \
  --mr-run-name soft-gate-layer3-30pct-20k \
  --baseline-run-name baseline-pretrained-20k

# MrSEDD only (no baseline)
modal run train_modal_mrdiffusion.py::eval_pareto_main \
  --mr-run-name soft-gate-layer3-30pct-20k

# Download results
modal volume get mrdiffusion-sedd-checkpoints \
  soft-gate-layer3-30pct-20k/pareto_results/pareto_results.json .
```

---

## W&B metrics

| Metric | Logged at | Description |
|---|---|---|
| `train/loss` | every 50 steps | Score entropy loss on the training batch |
| `train/deletion_rate` | every 50 steps | Mean soft-deletion proxy per token: `gate_output / sigmoid_mask_scale` ∈ [0, 1]. 0 = keeping all tokens, 1 = fully deleting all tokens. Target is `--target-deletion-rate` (e.g. 0.3). Logged as 0 for baseline runs. |
| `train/deletion_loss_weight` | every 50 steps | Current deletion loss weight. Only logged when `--use-pi-controller` is set; otherwise the weight is fixed. Should stabilize near a value that holds `train/deletion_rate` at the target. |
| `train/learning_rate` | every 50 steps | Current learning rate (warmup + constant) |
| `eval/loss` | every 100 steps | Score entropy loss on a WikiText-103 eval batch |
| `eval/generative_perplexity` | every 5,000 steps | GPT-2-Large perplexity of generated samples (lower = better). Requires `--save-steps 1000`; fires at 5k, 10k, 15k, 20k. |
| `samples` | every 25,000 steps | W&B Table of 8 generated text samples |
| `model/n_parameters` | step 0 | Total trainable parameter count |

### What to look for

**Healthy training signals:**
- `train/loss` decreases steadily (model is learning to score)
- `train/deletion_rate` starts at ~0.12 (gate init bias=2) and converges toward `--target-deletion-rate` within ~5k steps (after LR warmup at step 2500)
- `eval/loss` tracks `train/loss` without diverging

**Comparing baseline vs. gate:**
- A well-trained gate should reach similar `eval/loss` as the baseline
  while maintaining the target deletion rate
- `eval/generative_perplexity` is the main quality metric — gate model should match
  or approach baseline perplexity at a fraction of the compute per step

**Warning signs:**
- `train/deletion_rate` stuck near 0.12 and not rising — gate is not learning to delete; try higher `--deletion-loss-weight` (e.g. 0.5), or enable `--use-pi-controller` to let the weight adapt automatically
- `train/deletion_rate` at 1.0 — gate is deleting everything; reduce `--deletion-loss-weight` or lower `--target-deletion-rate`
- `train/loss` diverges vs. baseline — gate is hurting the model; the transformer is frozen for the first 2000 steps by default (`--freeze-transformer-steps`), which should stabilize early training
- `train/deletion_loss_weight` growing unboundedly (PI controller run) — integral term accumulating without recovery; deletion rate is stuck far from target; check gate init and warmup

---

## Estimated costs (A100-40GB)

| Run | Steps | Time | Cost |
|---|---|---|---|
| Smoke test | 200 | ~2 min | <$0.10 |
| All experiments | 20,000 | ~1 hr each | ~$4 each |

---

## Recommended experiment sequence

```
1. baseline-pretrained-20k              — verify W&B is logging, loss decreases
2. soft-gate-layer3-30pct-20k           — verify gate loss decreases, deletion rate converges
3. hard-gate-layer3-30pct-20k           — compare soft vs hard deletion
4. hard-gate-rope-original-pos-20k      — does position-aware RoPE improve hard deletion quality?
5. soft-gate-sigma-cond-20k             — does sigma conditioning help?
6. soft-gate-linear-schedule-20k        — does noise-adaptive rate schedule help?
7. soft-gate-pi-30pct-20k               — does PI-controlled loss weight stabilize deletion rate?
8. soft-gate-frozen-transformer-20k     — is joint finetuning necessary, or is gate-only training sufficient?
```

Runs 1 and 2 can be kicked off in parallel. Runs 3–8 can all run in parallel once
run 2 confirms the gate is learning.

After training, run the full eval suite on the two main runs:

```bash
# Zero-shot perplexity
modal run train_modal_mrdiffusion.py::eval_zero_shot_main --run-name baseline-pretrained-20k
modal run train_modal_mrdiffusion.py::eval_zero_shot_main --run-name soft-gate-layer3-30pct-20k

# MAUVE
modal run train_modal_mrdiffusion.py::eval_mauve_main --run-name baseline-pretrained-20k
modal run train_modal_mrdiffusion.py::eval_mauve_main --run-name soft-gate-layer3-30pct-20k

# Gate behavior (MrSEDD only)
modal run train_modal_mrdiffusion.py::eval_gate_behavior_main --run-name soft-gate-layer3-30pct-20k

# Pareto curve
modal run train_modal_mrdiffusion.py::eval_pareto_main \
  --mr-run-name soft-gate-layer3-30pct-20k \
  --baseline-run-name baseline-pretrained-20k

# FLOPs
modal run train_modal_mrdiffusion.py::eval_flops_main --run-name soft-gate-layer3-30pct-20k
```


