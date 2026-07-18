# MrBERT: Dynamic Token Merging for Encoder-Only Transformers

This repository contains the implementation and experiment code for our COLM 2026 paper, which adapts the [MrT5 delete gate](https://arxiv.org/abs/2410.20771) (Kallini et al., 2024) to BERT-base for efficient NLU inference.

MrBERT inserts a lightweight learned gate (2,305 parameters) after encoder layer 3 that selectively deletes tokens, achieving **1.89× A100 inference speedup** with only **0.27pp accuracy loss** on SNLI.

---

## Setup

```bash
conda create -n mrbert python=3.11 -y
conda activate mrbert
pip install modal torch transformers==4.39.1 datasets accelerate tqdm matplotlib "numpy<2" wandb
python3 -m modal setup  # one-time Modal auth for GPU training
```

> `transformers==4.39.1` is pinned — the model code depends on internal BERT APIs from this version.

---

## Repository Structure

```
├── mrbert/                          # Core MrBERT implementation
│   ├── models/
│   │   ├── modeling_mrbert.py       # MrBERT model (all task heads)
│   │   ├── configuration_mrbert.py  # MrBertConfig
│   │   └── modeling_bert.py         # Reference BERT for comparison
│   ├── training/
│   │   ├── train_mrbert.py          # Local training (HuggingFace Trainer)
│   │   ├── train_modal.py           # Modal serverless training (A100)
│   │   └── pi_controller.py         # PI controller for deletion rate
│   ├── data/
│   │   └── preprocess_*.py          # Dataset preprocessing (SNLI, SQuAD, SST-2, MRPC, IMDB, TyDi QA)
│   ├── eval/
│   │   └── eval_mrbert.py           # Evaluation script
│   ├── analysis/                    # Paper figure/table generation + analysis
│   │   ├── generate_table3_figure3.py
│   │   ├── generate_table4_classification.py
│   │   ├── generate_table5_tydiqa.py
│   │   ├── generate_table7_correlation.py
│   │   ├── generate_table8_comparison.py
│   │   ├── generate_figure4_gate_layer_runtime.py
│   │   ├── generate_figure5_tydiqa_ablation.py
│   │   ├── measure_runtime.py       # A100 inference benchmarking
│   │   ├── deletion_pattern_analysis.py
│   │   ├── get_deletion_patterns.py
│   │   ├── hard_deletion_curve.py
│   │   └── figures/                 # Generated CSVs and PDFs
│   └── test/
│       └── test_mrbert.py
├── analyze_delta_correlation.py     # Per-example delta-loss analysis (local)
├── run_delta_analysis_modal.py      # Delta-loss analysis on Modal (A100)
├── mrt5/                            # Original MrT5 reference implementation
├── COLM-Paper-dynamic-token-merging/  # LaTeX paper submission
└── util/                            # Modal/GCP setup guides
```

---

## Reproducing Paper Experiments

All training runs use Modal (NVIDIA A100). Checkpoints are stored on the `mrbert-checkpoints` Modal volume.

### 1. Data preprocessing

```bash
cd mrbert
python data/preprocess_snli.py
python data/preprocess_sst2.py
python data/preprocess_mrpc.py
python data/preprocess_imdb.py
python data/preprocess_squad.py
python data/preprocess_tydiqa.py
```

### 2. Training (Modal, A100)

```bash
cd mrbert/training

# BERT baseline (no gate)
modal run --detach train_modal.py::main \
    --model-type BERT --max-steps 30000 \
    --wandb-run-name bert-snli-baseline --wandb-project mrbert-snli

# MrBERT 30% (main result)
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-snli-30pct --wandb-project mrbert-snli

# MrBERT 50%
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.5 \
    --wandb-run-name mrbert-snli-50pct --wandb-project mrbert-snli

# MrBERT 70%
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.7 \
    --wandb-run-name mrbert-snli-70pct --wandb-project mrbert-snli

# Random deletion baseline
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --deletion-type random --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-snli-random30 --wandb-project mrbert-snli

# No-PI controller ablation
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 --no-pi-controller \
    --wandb-run-name mrbert-snli-nopi --wandb-project mrbert-snli

# Gate layer ablations (layers 1, 6, 9)
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

# Hard-deletion training (soft--hard gap experiment)
modal run --detach train_modal.py::main \
    --model-type MrBERT --max-steps 30000 \
    --target-deletion-rate 0.3 --hard-delete-train-prob 0.5 \
    --wandb-run-name mrbert-snli-30pct-hd --wandb-project mrbert-snli
```

**Classification tasks (SST-2, MRPC, IMDB):**
```bash
# SST-2 baseline + MrBERT
modal run --detach train_modal.py::main \
    --model-type BERT --task sst2 --max-steps 10000 \
    --wandb-run-name bert-sst2-baseline --wandb-project mrbert-sst2

modal run --detach train_modal.py::main \
    --model-type MrBERT --task sst2 --max-steps 10000 \
    --target-deletion-rate 0.3 \
    --wandb-run-name mrbert-sst2-30pct --wandb-project mrbert-sst2

# MRPC and IMDB follow the same pattern with --task mrpc / --task imdb
```

**TyDi QA (extractive QA with pre-deletion blending):**
```bash
# BERT baseline
modal run --detach train_modal.py::main \
    --model-type BERT --task tydiqa --max-steps 30000 \
    --wandb-run-name bert-tydiqa-baseline --wandb-project mrbert-tydiqa

# MrBERT 30% layer 9 + pre-deletion blending (best config)
modal run --detach train_modal.py::main \
    --model-type MrBERT --task tydiqa --max-steps 30000 \
    --target-deletion-rate 0.3 --delete-gate-layer 9 --use-pre-deletion-blend \
    --wandb-run-name mrbert-tydiqa-30pct-layer9-predel --wandb-project mrbert-tydiqa
```

### 3. Download checkpoints

```bash
# List available checkpoints
modal volume ls mrbert-checkpoints

# Download a specific checkpoint
modal volume get mrbert-checkpoints mrbert-snli-30pct/final ./mrbert/local_checkpoints/mrbert-snli-30pct/final
```

### 4. Runtime benchmarking (A100)

Runtime measurements are performed within the Modal training script. Results are saved to:
- `mrbert/analysis/figures/snli_runtime_table_deletion_percentage.csv`
- `mrbert/analysis/figures/snli_runtime_table_deletion_gate_layer.csv`

### 5. Delta-loss correlation analysis

```bash
# Run on Modal (uses checkpoints already on the volume)
modal run run_delta_analysis_modal.py

# Download results
modal volume get mrbert-checkpoints delta_analysis ./util/delta_analysis_results
```

### 6. Deletion pattern analysis

```bash
cd mrbert

# Extract per-token gate decisions (requires local checkpoint + GPU)
python analysis/get_deletion_patterns.py \
    --input_file local_checkpoints/mrbert-snli-30pct/final \
    --output_dir analysis/deletion_patterns

# Generate figures
python analysis/deletion_pattern_analysis.py \
    --input_file analysis/deletion_patterns/mrbert-snli-30pct_test.json \
    --output_dir analysis/figures
```

### 7. Generate paper tables and figures

All scripts read from `mrbert/analysis/wandb_plots/all_runs_summary.csv` (W&B export) and/or the runtime CSVs.

```bash
cd mrbert/analysis

python generate_table3_figure3.py          # Table 3 (SNLI) + Figure 3 (efficiency frontier)
python generate_table4_classification.py   # Table 4 (SST-2, MRPC, IMDB)
python generate_table5_tydiqa.py           # Table 5 (TyDi QA)
python generate_table7_correlation.py      # Table 7 (deletion-loss correlations)
python generate_table8_comparison.py       # Table 8 (MrT5 vs MrBERT comparison)
python generate_figure4_gate_layer_runtime.py  # Figure 4 (runtime vs gate layer)
python generate_figure5_tydiqa_ablation.py     # Figure 5 (TyDi QA blending ablation)
```

---

## Key Design Decisions

- **Gate placement**: Layer 3 (3 pre-gate layers process full sequence, 9 post-gate layers process compressed sequence)
- **Soft training, hard inference**: Soft deletion (attention bias) during training for differentiability; hard deletion (physical removal) at inference for real speedup
- **PI controller**: Dynamically adjusts deletion loss weight to hit the target deletion rate; without it, the gate runs away to ~89% deletion
- **Pre-deletion blending**: Blends pre-gate and post-gate representations for deleted positions; critical for extractive QA where answer tokens must not be corrupted
- **Gate initialization**: Bias = 10.0 ensures near-zero initial deletion (model starts from pretrained baseline)

---

## Tests

```bash
cd mrbert
python test/test_mrbert.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mrbert2026,
  title     = {Who Needs Every Token? Adapting Dynamic Token Merging to Subword-level Transformers},
  author    = {Anonymous},
  booktitle = {COLM 2026 Workshop on Efficient Reasoning},
  year      = {2026}
}
```
