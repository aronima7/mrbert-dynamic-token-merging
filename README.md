# MrBERT: Dynamic Token Merging for Encoder-Only Transformers

This repository contains the implementation and experiment code for our COLM 2026 Workshop on Efficient Reasoning paper, which adapts the [MrT5 delete gate](https://arxiv.org/abs/2410.20771) (Kallini et al., 2024) to BERT-base for efficient NLU inference.

MrBERT inserts a lightweight learned gate (2,305 parameters) after encoder layer 3 that selectively deletes tokens, achieving **1.89× A100 inference speedup** with only **0.27pp accuracy loss** on SNLI.

---

## Setup

```bash
conda create -n mrbert python=3.11 -y
conda activate mrbert
pip install -r requirements.txt
python3 -m modal setup  # one-time Modal auth for GPU training
```

> **Important:** `transformers==4.39.1` is pinned — the model code depends on internal BERT APIs from this version.

---

## Repository Structure

```
├── mrbert/
│   ├── models/
│   │   ├── modeling_mrbert.py       # MrBERT model (delete gate + all task heads)
│   │   ├── configuration_mrbert.py  # MrBertConfig (extends BertConfig)
│   │   └── modeling_bert.py         # Reference BERT baseline
│   ├── training/
│   │   ├── train_mrbert.py          # HuggingFace Trainer wrapper
│   │   ├── train_modal.py           # Modal serverless training (A100)
│   │   ├── train_gcp.py             # GCP training
│   │   └── pi_controller.py         # PI controller for deletion rate targeting
│   ├── data/
│   │   └── preprocess_*.py          # Dataset preprocessing (SNLI, SQuAD, SST-2, MRPC, IMDB, TyDi QA)
│   ├── eval/
│   │   └── eval_mrbert.py           # Evaluation script
│   ├── analysis/
│   │   ├── generate_table*.py       # Paper table/figure generation
│   │   ├── generate_figure*.py
│   │   ├── measure_runtime.py       # A100 inference benchmarking
│   │   ├── get_deletion_patterns.py # Extract per-token gate decisions
│   │   ├── deletion_pattern_analysis.py
│   │   ├── analyze_delta_correlation.py  # Per-example delta-loss analysis
│   │   ├── run_delta_analysis_modal.py   # Delta analysis on Modal GPU
│   │   ├── wandb_plots/all_runs_summary.csv  # W&B metrics (ground truth for tables)
│   │   └── figures/                 # Generated CSVs and PDFs
│   └── test/
│       └── test_mrbert.py
└── util/                            # Modal/GCP setup guides
    └── prepare_anonymous_repo.sh    # Generate anonymized copy for submission
```

---

## Reproducing Paper Experiments

All training runs use [Modal](https://modal.com) with NVIDIA A100 GPUs. Checkpoints are stored on the `mrbert-checkpoints` Modal volume.

### Step 1: Data Preprocessing

Preprocess datasets into local NDJSON format (required for Modal training, which uploads them to the container):

```bash
cd mrbert

python data/preprocess_snli.py      # → snli_datasets/
python data/preprocess_sst2.py      # → sst2_datasets/
python data/preprocess_mrpc.py      # → mrpc_datasets/
python data/preprocess_imdb.py      # → imdb_datasets/
python data/preprocess_squad.py     # → squad_datasets/
python data/preprocess_tydiqa.py    # → tydiqa_datasets/
```

### Step 2: Training on Modal (A100)

All commands run from `mrbert/training/`.

#### SNLI (Table 3 — main results)

```bash
# BERT baseline (no gate)
modal run --detach train_modal.py::main \
    --model-type BERT --max-steps 30000 \
    --wandb-run-name bert-snli-baseline --wandb-project mrbert-snli

# MrBERT 30% deletion (main result)
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-snli-30pct --wandb-project mrbert-snli

# MrBERT 50% deletion
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.5 \
    --wandb-run-name mrbert-snli-50pct --wandb-project mrbert-snli

# MrBERT 70% deletion
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.7 \
    --wandb-run-name mrbert-snli-70pct --wandb-project mrbert-snli

# Random deletion baseline (30%)
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --deletion-type random --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-snli-random30 --wandb-project mrbert-snli

# No-PI controller ablation
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 --no-pi-controller \
    --wandb-run-name mrbert-snli-nopi --wandb-project mrbert-snli

# MrBERT 0% deletion control (gate present but target=0)
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.0 --bypass-gate \
    --wandb-run-name mrbert-snli-0pct --wandb-project mrbert-snli
```

#### Gate layer ablation (Table 3 — gate placement)

```bash
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 --delete-gate-layer 1 \
    --wandb-run-name mrbert-snli-layer1 --wandb-project mrbert-snli

modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 --delete-gate-layer 6 \
    --wandb-run-name mrbert-snli-layer6 --wandb-project mrbert-snli

modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 --delete-gate-layer 9 \
    --wandb-run-name mrbert-snli-layer9 --wandb-project mrbert-snli
```

#### Hard deletion training (soft–hard gap experiment)

```bash
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 --hard-delete-train-prob 0.5 \
    --wandb-run-name mrbert-snli-30pct-hd --wandb-project mrbert-snli
```

#### Classification tasks — SST-2, MRPC, IMDB (Table 4)

```bash
# SST-2
modal run --detach train_modal.py::main \
    --model-type BERT --task sst2 --max-steps 10000 \
    --wandb-run-name bert-sst2-baseline --wandb-project mrbert-sst2

modal run --detach train_modal.py::main \
    --model-type MrBERT --task sst2 --max-steps 10000 \
    --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-sst2-30pct --wandb-project mrbert-sst2

# MRPC
modal run --detach train_modal.py::main \
    --model-type BERT --task mrpc --max-steps 10000 \
    --wandb-run-name bert-mrpc-baseline --wandb-project mrbert-mrpc

modal run --detach train_modal.py::main \
    --model-type MrBERT --task mrpc --max-steps 10000 \
    --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-mrpc-30pct --wandb-project mrbert-mrpc

# IMDB (use batch_size=16 due to 512 token sequences)
modal run --detach train_modal.py::main \
    --model-type BERT --task imdb --max-steps 10000 --batch-size 16 \
    --wandb-run-name bert-imdb-baseline --wandb-project mrbert-imdb

modal run --detach train_modal.py::main \
    --model-type MrBERT --task imdb --max-steps 10000 --batch-size 16 \
    --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-imdb-30pct --wandb-project mrbert-imdb
```

#### TyDi QA — extractive QA (Table 5)

```bash
# BERT baseline
modal run --detach train_modal.py::main \
    --model-type BERT --task tydiqa --max-steps 30000 --batch-size 16 \
    --wandb-run-name bert-tydiqa-baseline --wandb-project mrbert-tydiqa

# MrBERT 30% layer 3 (default)
modal run --detach train_modal.py::main \
    --model-type MrBERT --task tydiqa --max-steps 30000 --batch-size 16 \
    --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-tydiqa-30pct --wandb-project mrbert-tydiqa

# MrBERT 30% layer 9 + pre-deletion blending (best TyDi QA config)
modal run --detach train_modal.py::main \
    --model-type MrBERT --task tydiqa --max-steps 30000 --batch-size 16 \
    --target-deletion-rate 0.3 --delete-gate-layer 9 --use-pre-deletion-blend \
    --wandb-run-name mrbert-tydiqa-30pct-layer9-predel --wandb-project mrbert-tydiqa

# Ablation: layer 9 without pre-deletion blending
modal run --detach train_modal.py::main \
    --model-type MrBERT --task tydiqa --max-steps 30000 --batch-size 16 \
    --target-deletion-rate 0.3 --delete-gate-layer 9 --no-use-pre-deletion-blend \
    --wandb-run-name mrbert-tydiqa-30pct-layer9-no-predel --wandb-project mrbert-tydiqa
```

### Step 3: Download Checkpoints

```bash
# List all checkpoints on the volume
modal volume ls mrbert-checkpoints

# Download a specific checkpoint for local evaluation
modal volume get mrbert-checkpoints mrbert-snli-30pct/final \
    ./mrbert/local_checkpoints/mrbert-snli-30pct/final
```

### Step 4: Evaluation

Evaluation runs automatically after training when you pass `--mode training-and-eval` (the Modal default is `training-only`). To evaluate a downloaded checkpoint locally:

```bash
cd mrbert
python eval/eval_mrbert.py \
    --model_path ./local_checkpoints/mrbert-snli-30pct/final \
    --dataset_name local_snli --split test
```

### Step 5: Runtime Benchmarking (A100)

Runtime measurements require an A100 GPU. The benchmarking is integrated into the Modal training script and produces:
- `mrbert/analysis/figures/snli_runtime_table_deletion_percentage.csv`
- `mrbert/analysis/figures/snli_runtime_table_deletion_gate_layer.csv`

For standalone runtime measurement with local checkpoints:

```bash
cd mrbert
python analysis/measure_runtime.py \
    --model_paths local_checkpoints/bert-snli-baseline/final \
                  local_checkpoints/mrbert-snli-30pct/final \
                  local_checkpoints/mrbert-snli-50pct/final \
    --max_seq_length 128 --n_samples 512
```

### Step 6: Delta-Loss Correlation Analysis (Table 7)

This computes per-example correlation between deletion rate and loss degradation relative to baseline BERT:

```bash
# Run on Modal (uses checkpoints already on the volume)
modal run mrbert/analysis/run_delta_analysis_modal.py

# Download results
modal volume get mrbert-checkpoints delta_analysis ./mrbert/delta-analysis-results
```

Or locally with downloaded checkpoints:

```bash
cd mrbert
python analysis/analyze_delta_correlation.py \
    --baseline_path ./local_checkpoints/bert-snli-baseline/final \
    --mrbert_path ./local_checkpoints/mrbert-snli-30pct/final \
    --random_path ./local_checkpoints/mrbert-snli-random30/final \
    --dataset_name snli --split test --max_samples 5000 \
    --output_dir ./delta-analysis-results
```

### Step 7: Deletion Pattern Analysis

Extract per-token gate decisions for qualitative analysis:

```bash
cd mrbert

# Extract gate decisions to JSON
python analysis/get_deletion_patterns.py \
    --model_path ./local_checkpoints/mrbert-snli-30pct/final \
    --output_dir analysis/deletion_patterns \
    --sample_size 1000

# Generate visualizations
python analysis/deletion_pattern_analysis.py \
    --input_file analysis/deletion_patterns/mrbert-snli-30pct_test.json \
    --output_dir analysis/figures
```

### Step 8: Generate Paper Tables and Figures

All generation scripts read from `mrbert/analysis/wandb_plots/all_runs_summary.csv` (W&B metrics export) and/or the runtime CSVs:

```bash
cd mrbert/analysis

python generate_table3_figure3.py          # Table 3 (SNLI) + Figure 3 (efficiency frontier)
python generate_table4_classification.py   # Table 4 (SST-2, MRPC, IMDB)
python generate_table5_tydiqa.py           # Table 5 (TyDi QA + blending ablation)
python generate_table7_correlation.py      # Table 7 (deletion-loss correlations)
python generate_table8_comparison.py       # Table 8 (MrT5 vs MrBERT comparison)
python generate_figure4_gate_layer_runtime.py  # Figure 4 (runtime vs gate layer)
python generate_figure5_tydiqa_ablation.py     # Figure 5 (TyDi QA blending ablation)
```

Outputs go to `mrbert/analysis/figures/` as both CSV (raw data) and PDF (plots).

---

## Local Development (No GPU)

For quick iteration without Modal/GPU:

```bash
cd mrbert/training

# Smoke test (CPU, ~2 min)
python train_mrbert.py \
    --task sequence_classification --dataset_name local_snli \
    --model_type MrBERT --target_deletion_rate 0.3 \
    --max_steps 100 --logging_steps 10 \
    --max_train_samples 800 --max_eval_samples 200 \
    --output_dir ../mrbert_checkpoints/smoke --disable_wandb

# Run tests
cd mrbert && python test/test_mrbert.py
```

---

## Key Design Decisions

- **Gate placement (layer 3)**: 3 pre-gate layers process the full sequence, 9 post-gate layers process the compressed sequence — maximizes compute savings.
- **Soft training, hard inference**: Soft deletion (attention bias) during training preserves differentiability; hard deletion (physical token removal) at inference gives real 1.89× speedup.
- **PI controller**: Dynamically adjusts deletion loss weight α each step so the observed deletion rate tracks the target rate. Without it, the gate collapses (all tokens deleted) or goes to ~0% deletion.
- **Pre-deletion blending**: For deleted tokens, uses their pre-gate hidden states instead of corrupted post-deletion representations. Critical for extractive QA where answer tokens may be deleted.
- **Gate initialization**: Bias = 10.0 ensures near-zero initial deletion, preserving pretrained BERT's accuracy at the start of training.
- **Regularizer delay**: First N steps (default 1000) train on task loss only before enabling the deletion regularizer — lets the model learn the task before being asked to compress.

---

## Training Script Reference

`train_modal.py::main` accepts these key arguments (use `--help` for full list):

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | `MrBERT` | `MrBERT` (with gate) or `BERT` (baseline) |
| `--task` | `sequence_classification` | `sequence_classification`, `sst2`, `mrpc`, `imdb`, `tydiqa` |
| `--mode` | `training-only` | `training-only` or `training-and-eval` (runs eval after training) |
| `--dataset-name` | auto | `local_snli`, `local_sst2`, `local_mrpc`, `local_imdb`, `local_tydiqa` |
| `--max-steps` | 20 | Total training steps |
| `--target-deletion-rate` | 0.3 | Target fraction of tokens to delete |
| `--delete-gate-layer` | 3 | Encoder layer where gate fires (0-indexed) |
| `--deletion-type` | `scaled_sigmoid` | `scaled_sigmoid`, `random`, `fixed` |
| `--use-pi-controller` | True | Enable PI controller for deletion rate |
| `--hard-delete-train-prob` | 0.0 | Fraction of steps using hard deletion |
| `--use-pre-deletion-blend` | True | Blend pre-gate states for deleted tokens |
| `--batch-size` | 32 | Per-device batch size (use 16 for IMDB/TyDi QA) |
| `--regularizer-delay` | 1000 | Steps before enabling deletion loss |
| `--bypass-gate` | False | Disable gate entirely (0% control) |
| `--wandb-run-name` | auto | W&B run name |
| `--wandb-project` | `mrbert` | W&B project name |
| `--extra-args` | | Extra args passed to `train_mrbert.py` (e.g. `"--learning_rate 2e-5"`) |

---

## Tests

```bash
cd mrbert
python test/test_mrbert.py
```

---

## Citation

```bibtex
@inproceedings{mrbert2026,
  title     = {Who Needs Every Token? Adapting Dynamic Token Merging to Subword-level Transformers},
  author    = {Anonymous},
  booktitle = {COLM 2026 Workshop on Efficient Reasoning},
  year      = {2026}
}
```
