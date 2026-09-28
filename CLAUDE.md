# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MrBERT applies the **delete gate mechanism from MrT5** (Kallini et al., 2024) to BERT-base for efficient NLU inference. A lightweight learned gate (2,305 params) after encoder layer 3 selectively deletes tokens, achieving 1.89× A100 speedup with 0.27pp accuracy loss on SNLI.

**Primary current work:** COLM 2026 Workshop on Efficient Reasoning paper submission.

Reference: [MrT5: Dynamic Token Merging](https://arxiv.org/abs/2410.20771)

---

## Environment Setup

```bash
conda create -n mrbert python=3.11 -y
conda activate mrbert
pip install modal torch transformers==4.39.1 datasets accelerate tqdm matplotlib "numpy<2" wandb
python3 -m modal setup  # one-time Modal auth for GPU training
```

> `transformers==4.39.1` is pinned — the model code depends on internal BERT APIs from this version. Upgrading will break `mrbert/models/modeling_mrbert.py`.

---

## Commands

```bash
# Tests
cd mrbert && python test/test_mrbert.py

# Smoke test (local, no GPU needed for short runs)
cd mrbert/training
python train_mrbert.py --max_steps 100 --logging_steps 10

# Full training on Modal (A100)
cd mrbert/training
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-snli-30pct --wandb-project mrbert-snli

# Evaluate a checkpoint
cd mrbert
python eval/eval_mrbert.py --model_path ./local_checkpoints/mrbert-snli-30pct/final

# Download checkpoint from Modal volume
modal volume get mrbert-checkpoints mrbert-snli-30pct/final ./mrbert/local_checkpoints/mrbert-snli-30pct/final

# Baseline / comparison runs (top-level entry scripts)
cd mrbert
python run_baseline.py      # evaluate standard BERT (no gate) for baselines
python run_comparison.py    # run BERT vs MrBERT head-to-head on a task

# Generate paper tables/figures (all read from wandb_plots/all_runs_summary.csv)
cd mrbert/analysis
python generate_table3_figure3.py          # SNLI speed/accuracy tradeoff
python generate_table4_classification.py   # classification tasks
python generate_table5_tydiqa.py           # TyDi QA
python generate_table5_mrxlmr.py           # multilingual XLM-R variant
python generate_figure4_gate_layer_runtime.py
python generate_figure5_tydiqa_ablation.py
python generate_table7_correlation.py
python generate_table8_comparison.py
```

---

## Architecture

### MrBERT Forward Pass

```
Layers 0–2 (full sequence):  Standard BERT encoder layers on all tokens
         ↓
Delete Gate (before layer 3): LayerNorm → Linear(768→1) → ScaledSigmoid → values in [-30, 0]
         ↓
Layers 3–11 (compressed):    Gate values added as attention bias (soft deletion)
                              OR tokens below threshold physically removed (hard deletion)
         ↓
Output:  Pre-deletion blend restores original hidden states for deleted tokens
         Full-sequence logits [B, L, ...] — never zero-filled positions
```

**Soft training, hard inference**: Soft deletion (attention bias) during training for differentiability; hard deletion (physical removal) at inference for real speedup.

### Core Files

| Path | Purpose |
|------|---------|
| `mrbert/models/modeling_mrbert.py` | Full model: delete gate + all task heads (classification, NER, QA) |
| `mrbert/models/configuration_mrbert.py` | `MrBertConfig` extends `BertConfig` with gate parameters |
| `mrbert/models/modeling_bert.py` | Reference unmodified BERT for comparison baselines |
| `mrbert/training/train_mrbert.py` | HuggingFace Trainer wrapper with deletion loss, PI controller |
| `mrbert/training/train_modal.py` | Modal serverless training (A100); main entry point for experiments |
| `mrbert/training/pi_controller.py` | PI controller that dynamically adjusts deletion loss weight α |
| `mrbert/eval/eval_mrbert.py` | Evaluation script |
| `mrbert/test/test_mrbert.py` | Unit tests |

### Key Config Parameters

- `delete_gate_layer` (default 3): Where the gate fires. Layers 0 to gate_layer-1 are pre-gate; gate_layer to 11 are post-gate (compressed).
- `deletion_type`: `"scaled_sigmoid"` (learned), `"random"`, `"fixed"` (ablation baselines)
- `sigmoid_mask_scale` (-30.0): Gate output range is [this, 0]; more negative = stronger deletion signal
- `deletion_threshold` (-15.0): Below this = counted as deleted (= sigmoid_mask_scale / 2)
- `use_softmax1` (True): softmax with n+1 denominator (MrT5 recommendation)
- `use_pre_deletion_blend` (True): Blend pre-gate states into deleted positions; critical for extractive QA
- `bypass_gate` (False): Set True to disable gate entirely (for debugging)

### Training Knobs

- `--target_deletion_rate 0.3`: Fraction of tokens the gate should learn to delete
- `--use_pi_controller` / `--no-pi-controller`: PI controller adjusts deletion loss weight dynamically. Without it the gate runs away to ~89% deletion.
- `--delete_gate_lr`: Separate learning rate for gate parameters
- `--hard_delete_train_prob`: Probability of using hard deletion during training (0 = always soft)
- `--deletion_loss_weight`: Static weight if PI controller is off. 0.1–5.0 range.

---

## COLM Paper

**Data source:** `mrbert/analysis/wandb_plots/all_runs_summary.csv` — W&B metrics export, ground truth for all paper tables.

Table/figure generation scripts live in `mrbert/analysis/generate_*.py`. Each reads the CSV and outputs to `mrbert/analysis/figures/`.

COLM formatting: figures/tables at top/bottom of page ([t]), `\includegraphics[width=0.55\linewidth]`, no font below `\small`, captions below, `booktabs` for tables.

---

## Repository Navigation

- **`anonymous-submission/` is a stale, untracked mirror** of the repo (generated by `util/prepare_anonymous_repo.sh`). It is not in git — never edit files there; make changes in the real tree and regenerate. Ignore it when searching.
- **Canonical code lives under `mrbert/`.** The many top-level `README_*.md` and `mrbert/RUNS*.md` / `mrbert/README_*.md` files are experiment scratch notes and run logs, not authoritative docs — treat them as history, not spec. Ground truth for model behavior is the code; ground truth for paper numbers is `mrbert/analysis/wandb_plots/all_runs_summary.csv`.
- `train_modal.py` (Modal/A100) is the primary training entry point; `train_gcp.py` / `train_gce.py` are alternate cloud backends for the same `train_mrbert.py` core.

## Known Issues

### Gate collapse (deletion_rate → 0)

The gate learns to keep all tokens. Fixes in order of impact:
1. Increase `deletion_loss_weight` to 1.0–5.0
2. Set `gate_logit_reg_weight=0.0`
3. Enable PI controller (`--use_pi_controller`)
4. Add `--stop_gate_grad` (detach gate from task loss gradient)

### Anonymization for submission

Run `bash util/prepare_anonymous_repo.sh` to produce a scrubbed copy at `./anonymous-submission/` suitable for anonymous.4open.science. The script removes personal paths, W&B entity names, emails, and metadata from all files.
