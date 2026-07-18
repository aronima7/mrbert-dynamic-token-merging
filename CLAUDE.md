# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project applying the **delete gate mechanism from MrT5** to multiple architectures. After a specified layer, a learned gate assigns each token a deletion score; low-scoring tokens are soft-deleted (large negative attention bias) or hard-deleted (physically removed), reducing compute while preserving quality.

**Primary current work:** `COLM-Paper-dynamic-token-merging/` — COLM 2026 Workshop on Efficient Reasoning paper submission (MrBERT results only; MrXLMR removed from paper).  
**Core implementation:** Root-level MrBERT (delete gates on BERT, encoder-only) — the subject of the COLM paper.  
**Other explorations:** `mrdiffusion-sedd/`, `mrdiffusion-bd3lms/`, `mrxlmr/` (not included in the current paper).

Reference papers:
- [MrT5: Dynamic Token Merging](https://arxiv.org/abs/2410.20771) (Kallini et al., 2024) — delete gate design
- [SEDD: Score Entropy Discrete Diffusion](https://arxiv.org/abs/2310.16834) (Lou et al., 2023) — diffusion base

---

## COLM Paper Submission

**Location:** `COLM-Paper-dynamic-token-merging/COLM-submission/colm2026_submission.tex`  
**Template:** `COLM-Paper-dynamic-token-merging/COLM-template/`  
**Change log:** `COLM-Paper-dynamic-token-merging/logs.md` — full history of edits, validations, and decisions.

The paper covers MrBERT only (MrXLMR was removed). Page limit: 4–10 pages (main body, excluding references and appendix). Currently at 10 pages.

Key data sources for the paper:
- `mrbert/analysis/wandb_plots/all_runs_summary.csv` — W&B metrics for all runs (ground truth for tables)
- `mrbert/analysis/figures/` — generated CSVs and PDFs for tables/figures
- `util/delta_analysis_results/` — delta-loss correlation analysis outputs

Table/figure generation scripts (in `mrbert/analysis/`):
- `generate_table3_figure3.py` — SNLI results + efficiency frontier
- `generate_table4_classification.py` — SST-2, MRPC, IMDB results
- `generate_table5_tydiqa.py` — TyDi QA results
- `generate_table5_mrxlmr.py` — MrXLMR results (appendix only)
- `generate_table7_correlation.py` — deletion-loss correlation table
- `generate_table8_comparison.py` — MrT5 vs MrBERT comparison
- `generate_figure4_gate_layer_runtime.py` — runtime vs gate layer bar chart
- `generate_figure5_tydiqa_ablation.py` — TyDi QA blending ablation

COLM formatting rules: figures/tables at top/bottom of page ([t]), `\includegraphics[width=0.55\linewidth]` for standalone figures, no font below `\small`, captions below figures/tables, `booktabs` for tables.

---

## Environment Setup

```bash
# MrBERT / general
conda create -n mrbert python=3.11 -y
conda activate mrbert
pip install modal torch transformers datasets accelerate tqdm matplotlib "numpy<2"

# MrDiffusion-SEDD (additional)
pip install einops omegaconf hydra-core wandb
pip install flash-attn==2.5.8 --no-build-isolation  # requires CUDA 11.8+
```

---

## Commands

### MrBERT (root-level)

```bash
# Tests
python test_mrbert.py
python -c "from test_mrbert import test_config_creation; test_config_creation()"

# Train
python train_mrbert.py --task mlm --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
    --output_dir ./mrbert_checkpoints --num_epochs 3 --target_deletion_rate 0.3 --deletion_loss_weight 0.1
python train_mrbert.py --max_steps 100 --logging_steps 10  # smoke test

# Evaluate
python eval_mrbert.py --model_path ./mrbert_checkpoints/final
python eval_mrbert.py --from_pretrained bert-base-uncased

# Compare vs baseline BERT
python run_comparison.py --task mlm --train --num_epochs 10
```

### MrDiffusion-SEDD (`mrdiffusion-sedd/`)

```bash
cd mrdiffusion-sedd

# Tests
python -m pytest tests/test_mrsedd.py -v

# Local training
python train_mrdiffusion.py --output_dir ./runs/smoke --max_steps 50  # smoke test
python train_mrdiffusion.py --output_dir ./runs/baseline --max_steps 5000  # baseline (no gate)
python train_mrdiffusion.py --output_dir ./runs/soft --delete_gate_layer 3 \
    --deletion_type scaled_sigmoid --deletion_mode soft --gate_sigma_conditioned \
    --deletion_rate_schedule linear_sigma --r_min 0.1 --r_max 0.5 --deletion_loss_weight 1.0

# Evaluate
python evaluation/eval_zero_shot.py --model_path ./runs/soft/checkpoints/best
python evaluation/eval_gate_behavior.py --model_path ./runs/soft/checkpoints/best
python evaluation/eval_flops.py --model_path ./runs/soft/checkpoints/best
python evaluation/eval_mauve.py --model_path ./runs/soft/checkpoints/best
```

### Serverless GPU training (Modal)

```bash
# MrBERT
python3 -m modal setup
modal run train_modal.py                   # quick test (20 steps)
modal run train_modal.py --max-steps 500
modal volume get mrbert-checkpoints final ./local_mrbert_final

# MrDiffusion-SEDD
cd mrdiffusion-sedd
modal run train_modal_mrdiffusion.py                         # smoke test
modal run train_modal_mrdiffusion.py --run-name baseline     # no gate
modal run train_modal_mrdiffusion.py --run-name soft-gate    # soft deletion
modal volume ls mrdiffusion-checkpoints
modal volume get mrdiffusion-checkpoints <run-name>/checkpoints ./local_run
```

### MrDiffusion-BD3LM (`mrdiffusion-bd3lms/`)

```bash
cd mrdiffusion-bd3lms

# Tests
python -m pytest tests/test_mrd_bd3lm.py -v

# Training (local / Modal / GCP)
python train_mrd_bd3lm.py --output_dir ./runs/smoke --max_steps 50
modal run train_modal_mrd_bd3lm.py --run-name baseline
python train_gcp_mrd_bd3lm.py --run-name soft-gate

# Evaluate
python evaluation/eval_zero_shot.py --model_path ./runs/soft/checkpoints/best
python evaluation/eval_mauve.py --model_path ./runs/soft/checkpoints/best
python evaluation/eval_flops.py --model_path ./runs/soft/checkpoints/best
python evaluation/eval_pareto.py --model_path ./runs/soft/checkpoints/best
```

### MrXLM-R (`mrxlmr/`)

```bash
cd mrxlmr

# Training (scripts live in training/)
python training/train_mrxlmr.py --output_dir ./runs/smoke --max_steps 50
modal run training/train_modal.py

# Evaluate
python eval/eval_mrxlmr.py --model_path ./runs/best
```

---

## Architecture

### Directory Structure

| Directory | Role |
|-----------|------|
| `mrdiffusion-sedd/` | **Primary**: delete gates on SEDD |
| `mrdiffusion-bd3lms/` | Delete gates on BD3-LM (alternate diffusion baseline; full eval suite, tests, GCP+Modal training) |
| `mrxlmr/` | Delete gates on XLM-R (cross-lingual encoder; SNLI evaluation, Modal training). Note: scripts in `training/` and `eval/` subdirs, model in `models/` |
| `mrbert/` | Delete gates on BERT (encoder-only, foundational; extensive checkpoints and analysis) |
| `mrt5/` | Original MrT5 reference (T5-based); `models/modeling_mrt5.py` is the delete gate reference |
| `diffusion/Score-Entropy-Discrete-Diffusion/` | Official SEDD base implementation (imported by mrdiffusion-sedd via sys.path) |
| `diffusion/bd3lms/` | Official BD3-LM base implementation (imported by mrdiffusion-bd3lms via sys.path) |
| `COLM-Paper-dynamic-token-merging/` | **Active**: COLM 2026 paper submission (LaTeX paper + figures); see logs.md for full edit history |
| `util/` | Modal/GCP setup guides |
| `final-project-report/` | LaTeX/Markdown report with 19 figures (CS224N project report) |

### MrDiffusion-SEDD Core Files

| File | Purpose |
|------|---------|
| `modeling_mrdiffusion.py` | Delete gate + three-phase forward pass; all gate variants |
| `configuration_mrdiffusion.py` | `MrDiffusionConfig` dataclass (35+ parameters) |
| `losses_mrdiffusion.py` | Score entropy loss + sigma-dependent deletion rate loss |
| `train_mrdiffusion.py` | Local training loop with W&B |
| `train_modal_mrdiffusion.py` | Modal serverless training (A100) |
| `evaluation/` | Zero-shot NELBO, gate behavior, FLOPs, MAUVE, Pareto scripts |

### Three-Phase Forward Pass

```
Phase 1 (full L):   Blocks 0 … delete_gate_layer
   ↓
Delete Gate:        LayerNorm + Linear + ScaledSigmoid → values in [sigmoid_mask_scale, 0]
                    Optionally conditioned on noise level σ (gate_sigma_conditioned=True)
   ↓
Phase 2 (compressed):
  Soft deletion:    gate values added as attention bias in subsequent blocks
  Hard deletion:    tokens < deletion_threshold physically removed; RoPE recomputed
   ↓
Restoration at restore_gate_layer (or pre-output if None):
  Soft: clear gate mask
  Hard: scatter compressed states back to full L; fill deleted positions with x_pre_gate
   ↓
Phase 3 (full L):   Remaining blocks → logits [B, L, vocab_size]
```

Output is always full-sequence `[B, L, vocab_size]` — never zero-filled deleted positions.

### Gate Variants (`deletion_type` in config)

| Type | Class | Notes |
|------|-------|-------|
| `scaled_sigmoid` | `SigmoidDeleteGate` / `SigmoidDeleteGateWithSigma` | **Recommended**; σ-conditioned version is essential |
| `log_sigmoid` | `LogSigmoidDeleteGate` / `LogSigmoidDeleteGateWithSigma` | Alternative |
| `random` | `RandomDeleteGate` | Baseline |
| `fixed` | `FixedDeleteGate` | Baseline |

### Key MrDiffusionConfig Parameters

**Gate placement:**
- `delete_gate_layer` (default 3) — which DDiTBlock fires the gate
- `restore_gate_layer` (default None) — where to restore; None = just before output
- `gate_sigma_conditioned` (**True recommended**) — gate sees noise level σ

**Deletion schedule** (use instead of fixed `target_deletion_rate`):
- `deletion_rate_schedule`: `"constant"` | `"linear_sigma"` | `"power_sigma"`
- `r_min` / `r_max`: deletion rate ramps from r_min (clean, σ≈0) to r_max (noisy, σ≈σ_max)
- `deletion_rate_alpha`: exponent for `power_sigma` (1.0 = linear)

**Loss weights:**
- `deletion_loss_weight`: 0.1–5.0 (increase to 1.0+ if gate collapses)
- `gate_logit_reg_weight`: 0.001 (L2 reg on gate logits; set to 0 if gate collapses)

### MrBERT Core Files (root-level)

- `configuration_mrbert.py` — `MrBertConfig` extends `BertConfig`
- `modeling_mrbert.py` — Full implementation; all task heads (MLM, classification, NER, QA, NSP)
- `modeling_bert.py` — Reference BERT for comparison

Delete gate fires after `delete_gate_layer` (default 3). The gate fires BEFORE layer 3's attention/FFN, so layers 0–2 are pre-gate (full sequence) and layers 3–11 are post-gate (compressed) — 9 post-gate layers total. Auxiliary deletion loss pushes the gate toward `--target_deletion_rate`. Use `--delete_gate_lr` for a separate learning rate on gate parameters.

### Import Pattern (sys.path injection)

`mrdiffusion-sedd/` and `mrdiffusion-bd3lms/` import from their respective base implementations via runtime `sys.path` insertion (not installed packages). The modeling files resolve a relative `_SEDD_ROOT` / `_BD3LMS_ROOT` path at import time. This means:
- The base implementation directories (`diffusion/Score-Entropy-Discrete-Diffusion/` and `diffusion/bd3lms/`) must exist at the expected relative path
- Running scripts from a different working directory may break imports
- Always `cd` into the subdirectory before running its scripts

---

## Dependencies

`requirements.txt` pins `transformers==4.39.1` — the MrBERT/MrT5 code depends on this version's internal BERT/T5 APIs. Upgrading may break `modeling_mrbert.py` and `mrt5/models/modeling_mrt5.py`.

---

## Known Issues

### Gate collapse (deletion_rate → 0)

The gate learns to keep all tokens. Fixes (in order of impact):
1. Increase `deletion_loss_weight` to 1.0–5.0
2. Set `gate_logit_reg_weight=0.0`
3. Add `--use_pi_controller` (PI controller adjusts deletion loss weight automatically)
4. Add `--stop_gate_grad` (detach gate from score-entropy gradient)
5. Switch to `deletion_rate_schedule=linear_sigma` with `r_min=0.1, r_max=0.5`

### OOM during perplexity eval (A100-40GB)

Loading GPT-2-Large (3 GB) + MrDiffusion (2.5 GB) causes allocator fragmentation.  
Fix: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the eval script.

### Hard deletion: RoPE position drift

Hard deletion re-indexes surviving tokens to [0, 1, …, L_kept-1], losing original positions.  
Fix: `--rope_original_positions` preserves original indices [0, 2, 5, …]; falls back to SDPA in compressed phase.

---

## Root-Level Utility Scripts

| Script | Purpose |
|--------|---------|
| `plot_wandb_history.py` | Generates PNG plots from W&B history JSON exports |
| `download_wandb_plots.py` | Downloads W&B training data via GraphQL API |
| `launch_hard_deletion_runs.py` | Orchestrates batches of hard deletion training runs |
| `analyze_deletion_correlation.py` | Per-example deletion rate vs. loss correlation analysis |
| `analyze_delta_correlation.py` | Per-example delta-loss (model − baseline) correlation; used for paper Section 5.5 |
| `run_delta_analysis_modal.py` | Runs delta correlation analysis on Modal (A100); downloads results to `util/delta_analysis_results/` |
| `deletion_correlation_analysis.py` | Multi-model deletion analysis for SNLI |
| `run_comparison.py` | MrBERT vs baseline BERT comparison |
| `run_baseline.py` | Run baseline BERT training for comparison |