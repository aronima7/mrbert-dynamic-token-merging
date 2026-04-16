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
  --deletion_rate 0.3 --seq_len 128

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

# Smoke test: 50 steps, verifies the container runs
modal run train_modal_mrdiffusion.py::main
```

---

## Baseline runs (no delete gate)

Trains vanilla SEDD without any gate. Use as the comparison point for all gate experiments.

```bash
# Short baseline — verify loss decreases and W&B is logging (~1 hr, ~$4)
modal run --detach train_modal_mrdiffusion.py::main \
  --no-delete-gate \
  --pretrained-from louaaron/sedd-small \
  --max-steps 20000 \
  --wandb-run-name baseline-pretrained-20k

# Longer baseline (~10 hrs, ~$40)
modal run --detach train_modal_mrdiffusion.py::main \
  --no-delete-gate \
  --pretrained-from louaaron/sedd-small \
  --max-steps 200000 \
  --wandb-run-name baseline-pretrained-200k

# Baseline from scratch (expensive, ~$150–300, not recommended)
modal run --detach train_modal_mrdiffusion.py::main \
  --no-delete-gate \
  --wandb-run-name baseline-scratch
```

---

## Delete gate experiments (continued pretraining)

All gate experiments start from `louaaron/sedd-small` pretrained weights so only
the gate (and optionally the transformer) needs to learn. The pretrained transformer
blocks are loaded via `--pretrained-from`; the gate starts from random init.

### Soft deletion (recommended starting point)

Soft deletion applies a large negative attention bias to deleted tokens — tokens
remain in the sequence but are invisible to subsequent attention layers. Fully
differentiable; no hard decisions during training.

```bash
# Short sanity check: does gate loss decrease? (~1 hr, ~$4)
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --max-steps 20000 \
  --wandb-run-name soft-gate-layer3-30pct-20k

# Meaningful soft deletion run (~10 hrs, ~$40)
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --max-steps 200000 \
  --wandb-run-name soft-gate-layer3-30pct-200k
```

### Hard deletion

Hard deletion physically removes tokens below `--deletion-threshold` from the
sequence. Surviving tokens are processed by compressed-phase blocks, then restored
to full length before the output layer (via `restore_gate_layer`).

```bash
# Hard deletion sanity check (~1 hr, ~$4)
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode hard \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --max-steps 20000 \
  --wandb-run-name hard-gate-layer3-30pct-20k

# Hard deletion longer run (~10 hrs, ~$40)
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode hard \
  --deletion-type scaled_sigmoid \
  --delete-gate-layer 3 \
  --target-deletion-rate 0.3 \
  --deletion-loss-weight 0.1 \
  --max-steps 200000 \
  --wandb-run-name hard-gate-layer3-30pct-200k
```

### Sigma-conditioned gate

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
  --max-steps 200000 \
  --wandb-run-name soft-gate-sigma-cond-200k
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
  --max-steps 200000 \
  --wandb-run-name soft-gate-linear-schedule-200k
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
  --max-steps 200000 \
  --wandb-run-name soft-gate-layer6-30pct-200k
```

### Higher deletion rate

```bash
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --target-deletion-rate 0.5 \
  --deletion-loss-weight 0.1 \
  --max-steps 200000 \
  --wandb-run-name soft-gate-50pct-200k
```

---

## Resuming an interrupted run

Each run auto-saves a meta-checkpoint every `--save-steps` (default 5000). To resume:

```bash
# Re-run the exact same command — the script restores from the latest checkpoint
# and skips pretrained_from initialization if a checkpoint already exists.
modal run --detach train_modal_mrdiffusion.py::main \
  --pretrained-from louaaron/sedd-small \
  --deletion-mode soft \
  --max-steps 200000 \
  --wandb-run-name soft-gate-layer3-30pct-200k   # same run name → same output_dir
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
  --run-name soft-gate-layer3-30pct-200k

# Specific datasets only
modal run train_modal_mrdiffusion.py::eval_zero_shot_main \
  --run-name soft-gate-layer3-30pct-200k \
  --datasets wikitext-2,wikitext-103

# Compare baseline vs gate (run separately, compare stdout)
modal run train_modal_mrdiffusion.py::eval_zero_shot_main \
  --run-name baseline-pretrained-200k
modal run train_modal_mrdiffusion.py::eval_zero_shot_main \
  --run-name soft-gate-layer3-30pct-200k
```

### MAUVE score

Measures distributional similarity between model-generated and reference text.
Saves `mauve_result.json` to the volume.

```bash
# Default: 500 generated vs 200 reference, 128 diffusion steps
modal run train_modal_mrdiffusion.py::eval_mauve_main \
  --run-name soft-gate-layer3-30pct-200k

# More samples for a tighter estimate (~2x longer)
modal run train_modal_mrdiffusion.py::eval_mauve_main \
  --run-name soft-gate-layer3-30pct-200k \
  --n-gen 1000 --n-ref 500

# Download result
modal volume get mrdiffusion-sedd-checkpoints \
  soft-gate-layer3-30pct-200k/mauve_results/mauve_result.json .
```

### Gate behavior analysis

Deletion rate vs σ (overall, masked tokens, clean tokens) and gate value
distributions at multiple noise levels. Saves JSON + `.npy` arrays to the volume.

```bash
# Default: soft mode, wikitext-2
modal run train_modal_mrdiffusion.py::eval_gate_behavior_main \
  --run-name soft-gate-layer3-30pct-200k

# Under actual inference conditions (hard mode)
modal run train_modal_mrdiffusion.py::eval_gate_behavior_main \
  --run-name soft-gate-layer3-30pct-200k \
  --deletion-mode hard

# Download results
modal volume get mrdiffusion-sedd-checkpoints \
  soft-gate-layer3-30pct-200k/gate_analysis .
```

### FLOPs analysis + wall-clock profiling

Prints analytical FLOPs (baseline vs MrSEDD) and optionally runs 100 timed
forward passes to measure actual speedup.

```bash
# Analytical FLOPs + wall-clock profiling (hard deletion = real speedup)
modal run train_modal_mrdiffusion.py::eval_flops_main \
  --run-name soft-gate-layer3-30pct-200k \
  --deletion-rate 0.3

# Compare soft vs hard wall-clock
modal run train_modal_mrdiffusion.py::eval_flops_main \
  --run-name soft-gate-layer3-30pct-200k \
  --deletion-mode soft
modal run train_modal_mrdiffusion.py::eval_flops_main \
  --run-name soft-gate-layer3-30pct-200k \
  --deletion-mode hard
```

### Pareto frontier

Generative perplexity vs total FLOPs at 32/64/128/256/512 diffusion steps,
for both MrSEDD and baseline. Saves `pareto_results.json` to the volume.

```bash
# MrSEDD vs baseline Pareto curve (main result)
modal run train_modal_mrdiffusion.py::eval_pareto_main \
  --mr-run-name soft-gate-layer3-30pct-200k \
  --baseline-run-name baseline-pretrained-200k

# MrSEDD only (no baseline)
modal run train_modal_mrdiffusion.py::eval_pareto_main \
  --mr-run-name soft-gate-layer3-30pct-200k

# Download results
modal volume get mrdiffusion-sedd-checkpoints \
  soft-gate-layer3-30pct-200k/pareto_results/pareto_results.json .
```

---

## W&B metrics

| Metric | Logged at | Description |
|---|---|---|
| `train/score_entropy_loss` | every 50 steps | Score entropy loss (lower = model scores better) |
| `train/deletion_loss` | every 50 steps | MSE between actual and target deletion rate |
| `train/total_loss` | every 50 steps | `score_entropy_loss + deletion_loss_weight * deletion_loss` |
| `train/deletion_rate` | every 50 steps | Fraction of tokens actually deleted by the gate |
| `eval/score_entropy_loss` | every 100 steps | Eval score entropy on WikiText-103 |
| `eval/deletion_rate` | every 100 steps | Gate deletion rate on eval set |
| `eval/perplexity` | every 50,000 steps | GPT-2-Large perplexity of generated samples (lower = better) |

### What to look for

**Healthy training signals:**
- `train/score_entropy_loss` decreases steadily (model is learning to score)
- `train/deletion_rate` converges toward `--target-deletion-rate` within ~5k steps
- `train/deletion_loss` decreases as the gate learns to hit the target rate
- `eval/score_entropy_loss` tracks `train/score_entropy_loss` without diverging

**Comparing baseline vs. gate:**
- A well-trained gate should reach similar `eval/score_entropy_loss` as the baseline
  while maintaining the target deletion rate
- `eval/perplexity` is the main quality metric — gate model should match or approach
  baseline perplexity at a fraction of the compute per step

**Warning signs:**
- `train/deletion_rate` stuck at 0 — gate is not learning to delete; try higher `--deletion-loss-weight`
- `train/deletion_rate` stuck at 1 — gate is deleting everything; reduce `--deletion-loss-weight` or lower `--target-deletion-rate`
- `train/score_entropy_loss` diverges vs. baseline — gate is hurting the model; try `--freeze-transformer-steps 2000` to stabilize the gate first

---

## Estimated costs (A100-40GB)

| Run | Steps | Time | Cost |
|---|---|---|---|
| Smoke test | 50 | ~2 min | <$0.10 |
| Sanity check | 20,000 | ~1 hr | ~$4 |
| Meaningful experiment | 200,000 | ~10 hrs | ~$40 |
| Full continued pretraining | 500,000 | ~25 hrs | ~$100 |

---

## Recommended experiment sequence

```
1. baseline-pretrained-20k      — verify W&B is logging, loss decreases
2. soft-gate-layer3-30pct-20k   — verify gate loss decreases, deletion rate converges
3. baseline-pretrained-200k  \
   soft-gate-layer3-30pct-200k  — compare eval perplexity (main result)
4. soft-gate-sigma-cond-200k    — does sigma conditioning help?
5. soft-gate-linear-schedule-200k — does noise-adaptive rate schedule help?
```

After training, run the full eval suite on the two main runs:

```bash
# Zero-shot perplexity
modal run train_modal_mrdiffusion.py::eval_zero_shot_main --run-name baseline-pretrained-200k
modal run train_modal_mrdiffusion.py::eval_zero_shot_main --run-name soft-gate-layer3-30pct-200k

# MAUVE
modal run train_modal_mrdiffusion.py::eval_mauve_main --run-name baseline-pretrained-200k
modal run train_modal_mrdiffusion.py::eval_mauve_main --run-name soft-gate-layer3-30pct-200k

# Gate behavior (MrSEDD only)
modal run train_modal_mrdiffusion.py::eval_gate_behavior_main --run-name soft-gate-layer3-30pct-200k

# Pareto curve
modal run train_modal_mrdiffusion.py::eval_pareto_main \
  --mr-run-name soft-gate-layer3-30pct-200k \
  --baseline-run-name baseline-pretrained-200k

# FLOPs
modal run train_modal_mrdiffusion.py::eval_flops_main --run-name soft-gate-layer3-30pct-200k
```
-------
Note:
* All three root causes were specific to the SEDD codebase:                                                                                                                            
                                                                                                                                                                                                           
  1. Hydra struct errors (wandb_project, load_dir keys in config.yaml, train.py resume fix)                                                                                                                
  → mrdiffusion-sedd uses argparse, not Hydra. No config struct, no overrides, no issue.
                                                                                                                                                                                                           
  2. OOM fixes (batch_size=8, accum=4, eval_batch_size=16, PYTORCH_CUDA_ALLOC_CONF)
  → Root cause was seq_len=1024. mrdiffusion-sedd uses seq_len=128 (8× shorter). The worst-case logits tensor is [batch, 128, 50257] ≈ 0.8 GB at batch=32 — fits easily on an A100-40GB. No OOM risk.
                                                                                                                                                                                                           
  3. Multi-GPU train_8gpu                                                                                                                                                                                  
  → mrdiffusion-sedd is a single-GPU research prototype. The delete gate adds complexity that makes multi-GPU less straightforward, and the shorter seq_len means it doesn't need 8 GPUs to fit standard   
  batch sizes.   
-------