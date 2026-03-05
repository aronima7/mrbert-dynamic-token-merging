# MrBERT — First Milestone Run Plan

**Goal:** Produce a complete set of training runs, metrics, and charts that tell a coherent story at the advisor meeting:
*"MrBERT learns to delete tokens meaningfully. At 30% deletion, accuracy drops minimally while inference gets measurably faster. Random deletion at the same rate performs worse, confirming the gate is learning something non-trivial."*

---

## Run Index

| ID | Model  | Task  | Deletion | Variant                     | W&B Project  | Run Name                | Tier |
|----|--------|-------|----------|-----------------------------|--------------|-------------------------|------|
| A  | BERT   | SNLI  | —        | baseline                    | mrbert-snli  | bert-snli-baseline      | 1    |
| B  | MrBERT | SNLI  | 0%       | no deletion pressure        | mrbert-snli  | mrbert-snli-0pct        | 1    |
| C  | MrBERT | SNLI  | 30%      | main result (soft del only) | mrbert-snli  | mrbert-snli-30pct       | 1    |
| M  | MrBERT | SNLI  | 30%      | hard_delete_train_prob=0.5  | mrbert-snli  | mrbert-snli-30pct-hd    | 1    |
| D  | MrBERT | SNLI  | 50%      | tradeoff curve              | mrbert-snli  | mrbert-snli-50pct       | 1    |
| E  | MrBERT | SNLI  | 70%      | tradeoff curve              | mrbert-snli  | mrbert-snli-70pct       | 1    |
| F  | MrBERT | SNLI  | 30%      | random gate                 | mrbert-snli  | mrbert-snli-random30    | 1    |
| G  | MrBERT | SNLI  | 30%      | no PI controller            | mrbert-snli  | mrbert-snli-nopi        | 2    |
| H  | MrBERT | SNLI  | 30%      | gate at layer 1             | mrbert-snli  | mrbert-snli-layer1      | 2    |
| I  | MrBERT | SNLI  | 30%      | gate at layer 6             | mrbert-snli  | mrbert-snli-layer6      | 2    |
| J  | MrBERT | SNLI  | 30%      | gate at layer 9             | mrbert-snli  | mrbert-snli-layer9      | 2    |
| K  | BERT   | SQuAD   | —        | baseline                    | mrbert-squad  | bert-squad-baseline     | 1    |
| L  | MrBERT | SQuAD   | 30%      | main result                 | mrbert-squad  | mrbert-squad-30pct      | 1    |
| N  | BERT   | SST-2   | —        | baseline                    | mrbert-sst2   | bert-sst2-baseline      | 1    |
| O  | MrBERT | SST-2   | 30%      | main result                 | mrbert-sst2   | mrbert-sst2-30pct       | 1    |
| P  | BERT   | MRPC    | —        | baseline                    | mrbert-mrpc   | bert-mrpc-baseline      | 1    |
| Q  | MrBERT | MRPC    | 30%      | main result                 | mrbert-mrpc   | mrbert-mrpc-30pct       | 1    |
| R  | BERT   | IMDB    | —        | baseline                    | mrbert-imdb   | bert-imdb-baseline      | 1    |
| S  | MrBERT | IMDB    | 30%      | main result                 | mrbert-imdb   | mrbert-imdb-30pct       | 1    |
| T  | BERT   | TyDi QA | —        | baseline                    | mrbert-tydiqa | bert-tydiqa-baseline    | 1    |
| U  | MrBERT | TyDi QA | 30%      | main result                 | mrbert-tydiqa | mrbert-tydiqa-30pct     | 1    |

**Tier 1** = essential for the advisor meeting.
**Tier 2** = adds depth for the gate layer ablation chart and PI controller ablation; launch alongside Tier 1 if bandwidth allows.

---

## Time Estimates

| Phase     | Work                                        | Estimate            |
|-----------|---------------------------------------------|---------------------|
| Phase 1   | Launch all runs (terminal commands)         | ~10 min             |
| Phase 1   | Wait for SNLI runs to complete (A–J)        | ~90–100 min on A100 |
| Phase 1   | Wait for SQuAD runs to complete (K–L)       | ~40 min on A100     |
| Phase 1   | Wait for SST-2 runs to complete (N–O)       | ~15 min on A100     |
| Phase 1   | Wait for MRPC runs to complete (P–Q)        | ~5 min on A100      |
| Phase 1   | Wait for IMDB runs to complete (R–S)        | ~20 min on A100     |
| Phase 1   | Wait for TyDi QA runs to complete (T–U)     | ~10 min on A100     |
| Phase 2   | Download all checkpoints from Modal Volume  | ~15 min             |
| Phase 3   | Capture metrics from W&B                    | ~10 min             |
| Phase 4   | Runtime measurement (measure_runtime.py)    | ~10 min             |
| Phase 4.5 | Hard deletion curve (post-hoc eval)         | ~20 min on A100     |
| Phase 5   | Deletion pattern analysis                   | ~5 min              |
| Phase 6   | Generate all charts                         | ~5 min              |
| **Total** |                                             | **~3.5 hours**      |

---

## Phase 1 — Launch Training Runs

> **All runs in this phase are PARALLEL. Launch all of them from separate terminals (or as a batch) before doing anything else. Each run is independent and Modal handles parallelism.**

### SNLI — Tier 1 Core (Runs A, B, C, D, E, F)

```bash
modal run --detach train_modal.py::main --model-type BERT --task sequence_classification --num-epochs 3 --max-steps -1 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name bert-snli-baseline
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.0 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-0pct
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-30pct
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --hard-delete-train-prob 0.5 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-30pct-hd
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.5 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-50pct
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.7 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-70pct
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --deletion-type random --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-random30
```

### SNLI — Tier 2 Ablations (Runs G, H, I, J)

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --no-use-pi-controller --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-nopi
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --delete-gate-layer 1 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-layer1
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --delete-gate-layer 6 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-layer6
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --delete-gate-layer 9 --mode training-and-eval --wandb-project mrbert-snli --wandb-run-name mrbert-snli-layer9
```

### SQuAD — Tier 1 Core (Runs K, L)

```bash
modal run --detach train_modal.py::main --model-type BERT --task question_answering --dataset-name local_squad --num-epochs 3 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrbert-squad --wandb-run-name bert-squad-baseline
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task question_answering --dataset-name local_squad --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 400 --mode training-and-eval --wandb-project mrbert-squad --wandb-run-name mrbert-squad-30pct
```

### SST-2 — Tier 1 Core (Runs N, O)

```bash
modal run --detach train_modal.py::main --model-type BERT --task sequence_classification --dataset-name local_sst2 --num-epochs 3 --max-steps -1 --batch-size 32 --mode training-and-eval --wandb-project mrbert-sst2 --wandb-run-name bert-sst2-baseline
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --dataset-name local_sst2 --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 32 --regularizer-delay 300 --mode training-and-eval --wandb-project mrbert-sst2 --wandb-run-name mrbert-sst2-30pct
```

### MRPC — Tier 1 Core (Runs P, Q)

```bash
modal run --detach train_modal.py::main --model-type BERT --task sequence_classification --dataset-name local_mrpc --num-epochs 5 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrbert-mrpc --wandb-run-name bert-mrpc-baseline
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --dataset-name local_mrpc --num-epochs 5 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 50 --mode training-and-eval --wandb-project mrbert-mrpc --wandb-run-name mrbert-mrpc-30pct
```

### IMDB — Tier 1 Core (Runs R, S)

```bash
modal run --detach train_modal.py::main --model-type BERT --task sequence_classification --dataset-name local_imdb --num-epochs 3 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrbert-imdb --wandb-run-name bert-imdb-baseline
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task sequence_classification --dataset-name local_imdb --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 300 --mode training-and-eval --wandb-project mrbert-imdb --wandb-run-name mrbert-imdb-30pct
```

### TyDi QA — Tier 1 Core (Runs T, U)

```bash
modal run --detach train_modal.py::main --model-type BERT --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrbert-tydiqa --wandb-run-name bert-tydiqa-baseline
```

```bash
modal run --detach train_modal.py::main --model-type MrBERT --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 100 --mode training-and-eval --wandb-project mrbert-tydiqa --wandb-run-name mrbert-tydiqa-30pct
```

---

## Phase 2 — Download Checkpoints (Optional but Recommended)

> **Sequential. Run after Phase 1 runs complete. Monitor progress at https://modal.com/apps.**

Download all completed checkpoints to local disk. Each command downloads the model files directly into the destination folder, avoiding the nested-directory issue with `modal volume get`:

```bash
mkdir -p ./local_checkpoints/bert-snli/final && modal volume get mrbert-checkpoints bert-snli-baseline/final/config.json ./local_checkpoints/bert-snli/final/config.json
```

The above pattern is cumbersome for full checkpoints. Instead, download each run's `final/` directory by listing its contents first and downloading file by file, or use:

```bash
for run in bert-snli-baseline mrbert-snli-0pct mrbert-snli-30pct mrbert-snli-30pct-hd mrbert-snli-50pct mrbert-snli-70pct mrbert-snli-random30 mrbert-snli-layer1 mrbert-snli-layer6 mrbert-snli-layer9 bert-squad-baseline mrbert-squad-30pct bert-sst2-baseline mrbert-sst2-30pct bert-mrpc-baseline mrbert-mrpc-30pct bert-imdb-baseline mrbert-imdb-30pct bert-tydiqa-baseline mrbert-tydiqa-30pct; do
  mkdir -p ./local_checkpoints/${run}/final
  for f in config.json tokenizer_config.json vocab.txt special_tokens_map.json model.safetensors; do
    modal volume get mrbert-checkpoints ${run}/final/${f} ./local_checkpoints/${run}/final/${f} 2>/dev/null || true
  done
  echo "Downloaded ${run}"
done
```

---

## Phase 3 — Capture Metrics from W&B

> **Sequential. Do this while waiting for downloads or immediately after.**
> **These numbers will be substituted into the chart commands in Phase 5.**

For each completed run, open W&B and copy the following values from the **Summary** tab. Record them in the table below.

### SNLI Results Table (fill in from W&B)

| Run | Model                    | Del Rate Target | test/accuracy | percent_non_pad_deleted_tokens | new_seq_len | seq_len_reduction_pct |
|-----|--------------------------|-----------------|---------------|--------------------------------|-------------|----------------------|
| A   | BERT baseline            | 0%              | `___`         | 0                             | 128         | 0                    |
| B   | MrBERT 0%                | 0%              | `___`         | `___`                         | `___`       | `___`                |
| C   | MrBERT 30%               | 30%             | `___`         | `___`                         | `___`       | `___`                |
| M   | MrBERT 30% hard-train    | 30%             | `___`         | `___`                         | `___`       | `___`                |
| D   | MrBERT 50%               | 50%             | `___`         | `___`                         | `___`       | `___`                |
| E   | MrBERT 70%               | 70%             | `___`         | `___`                         | `___`       | `___`                |
| F   | MrBERT Random 30%        | 30%             | `___`         | `___`                         | `___`       | `___`                |
| G   | MrBERT No-PI 30%         | 30%             | `___`         | `___`                         | `___`       | `___`                |
| H   | MrBERT Layer 1           | 30%             | `___`         | `___`                         | `___`       | `___`                |
| C   | MrBERT Layer 3 (reuse C) | 30%             | `___`         | `___`                         | `___`       | `___`                |
| I   | MrBERT Layer 6           | 30%             | `___`         | `___`                         | `___`       | `___`                |
| J   | MrBERT Layer 9           | 30%             | `___`         | `___`                         | `___`       | `___`                |

### SQuAD Results Table (fill in from W&B)

| Run | Model         | Del Rate | test/squad_em | test/squad_f1 | seq_len_reduction_pct |
|-----|---------------|----------|---------------|---------------|-----------------------|
| K   | BERT baseline | 0%       | `___`         | `___`         | 0                     |
| L   | MrBERT 30%    | 30%      | `___`         | `___`         | `___`                 |

### SST-2 Results Table (fill in from W&B)

| Run | Model         | Del Rate | test/accuracy | seq_len_reduction_pct |
|-----|---------------|----------|---------------|-----------------------|
| N   | BERT baseline | 0%       | `___`         | 0                     |
| O   | MrBERT 30%    | 30%      | `___`         | `___`                 |

### MRPC Results Table (fill in from W&B)

| Run | Model         | Del Rate | test/accuracy | seq_len_reduction_pct |
|-----|---------------|----------|---------------|-----------------------|
| P   | BERT baseline | 0%       | `___`         | 0                     |
| Q   | MrBERT 30%    | 30%      | `___`         | `___`                 |

### IMDB Results Table (fill in from W&B)

| Run | Model         | Del Rate | test/accuracy | seq_len_reduction_pct |
|-----|---------------|----------|---------------|-----------------------|
| R   | BERT baseline | 0%       | `___`         | 0                     |
| S   | MrBERT 30%    | 30%      | `___`         | `___`                 |

### TyDi QA Results Table (fill in from W&B)

| Run | Model         | Del Rate | test/squad_em | test/squad_f1 | seq_len_reduction_pct |
|-----|---------------|----------|---------------|---------------|-----------------------|
| T   | BERT baseline | 0%       | `___`         | `___`         | 0                     |
| U   | MrBERT 30%    | 30%      | `___`         | `___`         | `___`                 |

> **W&B tip:** In each run's Summary tab, search for `test/` to find all final test metrics. The metrics `seq_len_reduction_pct` and `new_seq_len` are under the eval prefix in the last logged step.

---

## Phase 4 — Runtime Measurement

> **Run on A100 via Modal — do not run locally (CPU timing is unreliable for speedup claims).**
> **Requires all Phase 1 training runs to have completed (checkpoints in the Modal volume).**
> **Hard deletion is used by default: MrBERT tokens are physically removed from tensors on GPU.**

Launch the benchmark on A100 (uses all checkpoints already in the Modal volume):

```bash
modal run training/train_modal.py::benchmark_main
```

This benchmarks all SNLI runs and gate-layer ablation runs against each other, then saves results to the volume. Download the results:

```bash
mkdir -p analysis/figures
modal volume get mrbert-checkpoints analysis_figures/snli_runtime_table_deletion_percentage.csv ./analysis/figures/snli_runtime_table_deletion_percentage.csv
modal volume get mrbert-checkpoints analysis_figures/snli_runtime_table_deletion_gate_layer.csv ./analysis/figures/snli_runtime_table_deletion_gate_layer.csv
modal volume get mrbert-checkpoints analysis_figures/snli_runtime_vs_deletion_percentage.pdf ./analysis/figures/snli_runtime_vs_deletion_percentage.pdf
modal volume get mrbert-checkpoints analysis_figures/snli_runtime_vs_deletion_gate_layer.pdf ./analysis/figures/snli_runtime_vs_deletion_gate_layer.pdf
```

**Fill in the table from `analysis/figures/snli_runtime_table_deletion_percentage.csv`:**

| Model            | ms/sample | % decrease vs BERT |
|------------------|-----------|--------------------|
| BERT baseline    | `___`     | —                  |
| MrBERT 0%        | `___`     | `___`              |
| MrBERT 30%       | `___`     | `___`              |
| MrBERT 30% hd    | `___`     | `___`              |
| MrBERT 50%       | `___`     | `___`              |
| MrBERT 70%       | `___`     | `___`              |
| Random 30%       | `___`     | `___`              |
| Layer 1          | `___`     | `___`              |
| Layer 3          | `___`     | `___`              |
| Layer 6          | `___`     | `___`              |
| Layer 9          | `___`     | `___`              |

Saved files: `analysis/figures/snli_runtime_table_deletion_percentage.csv`, `analysis/figures/snli_runtime_vs_deletion_percentage.pdf`, `analysis/figures/snli_runtime_table_deletion_gate_layer.csv`, `analysis/figures/snli_runtime_vs_deletion_gate_layer.pdf`

---

## Phase 4.5 — Hard Deletion Curve

> **Run on A100 via Modal after Phase 1 training completes.**
> **Requires checkpoints in the Modal volume (checkpoint-* subdirs saved during training).**
> **Shows that hard deletion accuracy tracks soft deletion accuracy throughout training — confirms the model does not collapse when tokens are physically removed at inference.**

Run for the main 30% soft-deletion model (run C) and the hard-train variant (run M):

```bash
modal run training/train_modal.py::hard_deletion_curve_main --run-name mrbert-snli-30pct
```

```bash
modal run training/train_modal.py::hard_deletion_curve_main --run-name mrbert-snli-30pct-hd
```

Download results:

```bash
mkdir -p analysis/figures
modal volume get mrbert-checkpoints analysis_figures/mrbert-snli-30pct_hard_deletion_curve.csv ./analysis/figures/mrbert-snli-30pct_hard_deletion_curve.csv
modal volume get mrbert-checkpoints analysis_figures/mrbert-snli-30pct_hard_deletion_curve.pdf ./analysis/figures/mrbert-snli-30pct_hard_deletion_curve.pdf
modal volume get mrbert-checkpoints analysis_figures/mrbert-snli-30pct-hd_hard_deletion_curve.csv ./analysis/figures/mrbert-snli-30pct-hd_hard_deletion_curve.csv
modal volume get mrbert-checkpoints analysis_figures/mrbert-snli-30pct-hd_hard_deletion_curve.pdf ./analysis/figures/mrbert-snli-30pct-hd_hard_deletion_curve.pdf
```

**What to look for:**
- Both soft and hard deletion lines should track closely throughout training
- Run C (soft-only training): expect a small but nonzero soft–hard gap at the end
- Run M (hard_delete_train_prob=0.5): gap should be near zero, since the model was exposed to hard deletion during training
- A large or growing gap = the model relies on deleted tokens still being in the attention context

---

## Phase 5 — Deletion Pattern Analysis

> **Parallel with Phase 4. Requires Phase 2 (MrBERT 30% SNLI checkpoint downloaded).**

```bash
python analysis/get_deletion_patterns.py --model_path ./local_checkpoints/mrbert-snli-30pct/final --local_snli_dir ./snli_datasets --sample_size 1000 --output_dir ./analysis/deletion_patterns --output_file mrbert-snli-30pct_test.json
```

```bash
python analysis/deletion_pattern_analysis.py --input_file ./analysis/deletion_patterns/mrbert-snli-30pct_test.json --output_dir ./analysis/figures
```

Saved files: `analysis/figures/mrbert-snli-30pct_test_by_type.pdf`, `analysis/figures/mrbert-snli-30pct_test_premise_vs_hyp.pdf`

---

## Phase 6 — Generate Charts

> **Sequential. Requires Phase 3 (W&B metrics captured) and Phase 4 (runtime measured).**
> **Replace all `___` placeholders with actual values before running.**
> **Run from the `mrbert/` directory.**

### Theoretical compute savings + accuracy vs compute + accuracy vs seq-length reduction

Replace each `___` with the actual `test/accuracy` value from the W&B table:

```bash
python analysis/compute_savings.py --output_dir analysis/figures --runs "BERT,___,0.0" "MrBERT-0%,___,0.0" "MrBERT-30%,___,0.30" "MrBERT-50%,___,0.50" "MrBERT-70%,___,0.70" "Random-30%,___,0.30"
```

Saved files:
- `analysis/figures/macs_relative.pdf`
- `analysis/figures/macs_by_gate_layer.pdf`
- `analysis/figures/accuracy_vs_compute.pdf`
- `analysis/figures/accuracy_vs_seq_reduction.pdf`

### Gate layer ablation chart

Replace each `___` with accuracy from W&B and ms/sample from Phase 4:

```bash
python analysis/compute_savings.py --output_dir analysis/figures --gate-layer-runs "Layer 1,___,___,1" "Layer 3,___,___,3" "Layer 6,___,___,6" "Layer 9,___,___,9"
```

Saved file: `analysis/figures/gate_layer_ablation.pdf`

---

## Complete List of Output Files

| File                                                                          | What it shows                                              | Advisor relevance                  |
|-------------------------------------------------------------------------------|------------------------------------------------------------|-------------------------------------|
| `analysis/figures/accuracy_vs_seq_reduction.pdf`                             | Accuracy vs sequence length reduction % — core tradeoff    | Primary result                      |
| `analysis/figures/accuracy_vs_compute.pdf`                                   | Accuracy vs relative MACs — efficiency frontier            | Primary result                      |
| `analysis/figures/snli_runtime_vs_deletion_percentage.pdf`                   | Measured ms/sample per model                               | Core claim: actual speedup          |
| `analysis/figures/snli_runtime_table_deletion_percentage.csv`                | Runtime table for paper (mirrors MrT5 Table 3)             | For paper table                     |
| `analysis/figures/gate_layer_ablation.pdf`                                   | Accuracy + runtime vs gate layer — dual axis               | Ablation                            |
| `analysis/figures/macs_relative.pdf`                                         | Theoretical compute savings curve                          | Background                          |
| `analysis/figures/macs_by_gate_layer.pdf`                                    | Theoretical savings by gate layer                          | Background                          |
| `analysis/figures/mrbert-snli-30pct_hard_deletion_curve.pdf`                 | Soft vs hard deletion accuracy vs training step (run C)    | Validates hard deletion robustness  |
| `analysis/figures/mrbert-snli-30pct-hd_hard_deletion_curve.pdf`              | Soft vs hard deletion accuracy vs training step (run M)    | Validates hard deletion robustness  |
| `analysis/figures/mrbert-snli-30pct_test_by_type.pdf`                        | Deletion rate by token type (word/subword/punct)           | Qualitative                         |
| `analysis/figures/mrbert-snli-30pct_test_premise_vs_hyp.pdf`                 | Premise vs hypothesis deletion rate                        | Qualitative                         |
| `analysis/deletion_patterns/mrbert-snli-30pct_test.json`                     | Per-token gate decisions, 1000 examples                    | For colored examples in slides      |

---

## What to Present to Your Advisor

### Structure (30–40 min)

**1. Motivation (2 min)**
- BERT processes all tokens with equal compute regardless of informativeness
- MrT5 showed learned token deletion works for encoder-decoder (T5) models
- Research question: can we apply the same mechanism to BERT for discriminative NLU tasks?

**2. Architecture (5 min)**
- Delete gate inserted after encoder layer 3
- Gate = `LayerNorm → Linear → scaled sigmoid`, 2,305 additional parameters out of 110M
- Soft deletion during training: large negative bias to attention scores of deleted tokens
- PI controller dynamically adjusts deletion pressure α to hit target deletion rate δ
- Show: gate equation + diagram of where it sits in the encoder stack

**3. Core Results — SNLI (5 min)**

Show the filled-in results table + `accuracy_vs_seq_reduction.pdf`:

| Model | Accuracy | Del Rate | Seq Δ | Runtime |
|-------|----------|----------|-------|---------|
| BERT baseline | `___` | 0% | 0% | `___` ms |
| MrBERT 0% (sanity) | `___` | 0% | 0% | `___` ms |
| MrBERT 30% | `___` | 30% | `___`% | `___` ms |
| MrBERT 50% | `___` | 50% | `___`% | `___` ms |
| MrBERT 70% | `___` | 70% | `___`% | `___` ms |
| Random gate 30% | `___` | 30% | 30% | `___` ms |

Key talking points:
- MrBERT 0% should match BERT — validates the implementation is fair
- Compare MrBERT 30% accuracy vs BERT baseline — how small is the gap?
- Compare MrBERT 30% vs Random 30% — the learned gate should do better, proving the gate is learning
- Show `snli_runtime_vs_deletion_percentage.pdf` — the actual measured speedup

**4. Core Results — SQuAD (5 min)**

| Model | EM | F1 | Seq Δ | Runtime |
|-------|----|----|-------|---------|
| BERT baseline | `___` | `___` | 0% | `___` ms |
| MrBERT 30% | `___` | `___` | `___`% | `___` ms |

Key talking point: QA requires extracting the exact answer span. The gate must preserve semantically critical context tokens. Does the accuracy hold up?

**5. Ablations (5 min)**

- `gate_layer_ablation.pdf`: earlier gate → faster but less accurate; layer 3 is the sweet spot
- No PI controller: what happens to actual deletion rate without adaptive control?
- Show `mrbert-snli-30pct_hard_deletion_curve.pdf`: soft vs hard deletion accuracy over training — confirms the model doesn't collapse under hard deletion at inference
- Show `mrbert-snli-30pct_test_by_type.pdf`: which token types get deleted? (function words vs content words)
- Show `mrbert-snli-30pct_test_premise_vs_hyp.pdf`: does the gate treat premise and hypothesis differently?

**6. Discussion (10 min)**

- The efficiency frontier: how far down the tradeoff curve is acceptable for publication?
- Does the gate learn linguistically meaningful patterns?
- What should the next set of experiments be?

---

## Questions for Your Advisor

These questions will directly shape the second milestone experiments.

### Scope and Tasks
1. Is SNLI + SQuAD sufficient as the evaluation suite, or do you recommend adding a third task (e.g., MNLI for generalization, CoNLL NER for token classification, GLUE for a broader benchmark)?
2. MrT5 uses 15 languages to demonstrate language-specific compression rates. For a BERT-based paper, should we include multilingual BERT (mBERT) with multilingual NLI to replicate this finding?

### Architecture and Design
3. We use soft deletion throughout training (attention masking) and could optionally use hard deletion (physically remove tokens) at test time. Is the hard vs soft deletion comparison important to include in the paper?
4. The gate is currently at layer 3 (out of 12). MrT5 uses layer 3 out of 12 for the same reason. Should we justify this choice with a thorough ablation (layers 1–11), or is a sample of 4 layers (1, 3, 6, 9) sufficient?
5. We initialize MrBERT from pretrained BERT weights and only add 2,305 gate parameters. Is this the right framing, or should we also experiment with training the gate from scratch to test whether pretraining helps?

### Baselines
6. MrT5 compares against Boundary Predictor and Convolutional Pooling baselines. For a BERT paper, are there equivalent compression baselines we should include (e.g., SpAtten, TR-BERT, LTP)?
7. How much emphasis should the random deletion baseline receive? Is it sufficient to show MrBERT > random, or do reviewers expect a richer set of non-learned baselines?

### Metrics and Claims
8. The runtime speedup we measure is for forward-pass inference on a single GPU. For a publication claim, should we also measure wall-clock speedup in a batched serving scenario (where hard deletion breaks batch uniformity)?
9. MrT5 reports bits-per-byte (language modeling loss). Our equivalent is accuracy/EM/F1. Is there any value in also pre-training MrBERT on MLM before fine-tuning, to get a language-modeling-style efficiency comparison?

### Paper Framing
10. How should we frame the contribution relative to MrT5? Options: (a) direct adaptation of MrT5 to BERT, (b) independent extension showing the mechanism generalizes across architectures, (c) applied study showing practical efficiency gains for discriminative NLU.
11. What is the target venue? (ACL, EMNLP, NAACL, workshop?) — this affects how thorough the ablations need to be and how many tasks are required.
12. Is there a specific result threshold that would make this "publishable" — e.g., a minimum runtime speedup % at a maximum accuracy drop %?