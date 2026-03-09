# MrXLMR — First Milestone Run Plan

**Goal:** Produce a complete set of training runs, metrics, and charts that tell a coherent story at the advisor meeting:
*"MrXLMR learns to delete tokens meaningfully in a cross-lingual setting. At 30% deletion, accuracy drops minimally while inference gets measurably faster. Random deletion at the same rate performs worse, confirming the gate is learning something non-trivial. The mechanism generalises across six diverse tasks and XLM-R's multilingual pre-training."*

---

## Run Index

| ID | Model  | Task     | Deletion | Variant                      | W&B Project    | Run Name                             | Tier |
|----|--------|----------|----------|------------------------------|----------------|--------------------------------------|------|
| A  | XLMR   | SNLI     | —        | baseline                     | mrxlmr-snli    | xlmr-snli-baseline                   | 1    |
| B  | MrXLMR | SNLI     | 0%       | no deletion pressure         | mrxlmr-snli    | mrxlmr-snli-0pct                     | 1    |
| C  | MrXLMR | SNLI     | 30%      | main result (soft del only)  | mrxlmr-snli    | mrxlmr-snli-30pct                    | 1    |
| D  | MrXLMR | SNLI     | 50%      | tradeoff curve               | mrxlmr-snli    | mrxlmr-snli-50pct                    | 1    |
| E  | MrXLMR | SNLI     | 70%      | tradeoff curve               | mrxlmr-snli    | mrxlmr-snli-70pct                    | 1    |
| F  | MrXLMR | SNLI     | 30%      | random gate                  | mrxlmr-snli    | mrxlmr-snli-random30                 | 1    |
| G  | MrXLMR | SNLI     | 30%      | no PI controller             | mrxlmr-snli    | mrxlmr-snli-nopi                     | 2    |
| H  | MrXLMR | SNLI     | 30%      | gate at layer 1              | mrxlmr-snli    | mrxlmr-snli-layer1                   | 2    |
| I  | MrXLMR | SNLI     | 30%      | gate at layer 6              | mrxlmr-snli    | mrxlmr-snli-layer6                   | 2    |
| J  | MrXLMR | SNLI     | 30%      | gate at layer 9              | mrxlmr-snli    | mrxlmr-snli-layer9                   | 2    |
| K  | XLMR   | SQuAD    | —        | baseline                     | mrxlmr-squad   | xlmr-squad-baseline                  | 1    |
| L  | MrXLMR | SQuAD    | 30%      | main result                  | mrxlmr-squad   | mrxlmr-squad-30pct                   | 1    |
| N  | XLMR   | SST-2    | —        | baseline                     | mrxlmr-sst2    | xlmr-sst2-baseline                   | 1    |
| O  | MrXLMR | SST-2    | 30%      | main result                  | mrxlmr-sst2    | mrxlmr-sst2-30pct                    | 1    |
| P  | XLMR   | MRPC     | —        | baseline                     | mrxlmr-mrpc    | xlmr-mrpc-baseline                   | 1    |
| Q  | MrXLMR | MRPC     | 30%      | main result                  | mrxlmr-mrpc    | mrxlmr-mrpc-30pct                    | 1    |
| R  | XLMR   | IMDB     | —        | baseline                     | mrxlmr-imdb    | xlmr-imdb-baseline                   | 1    |
| S  | MrXLMR | IMDB     | 30%      | main result                  | mrxlmr-imdb    | mrxlmr-imdb-30pct                    | 1    |
| T  | XLMR   | TyDi QA  | —        | baseline                     | mrxlmr-tydiqa  | xlmr-tydiqa-baseline                 | 1    |
| U  | MrXLMR | TyDi QA  | 0%       | sanity check + pre-del blend | mrxlmr-tydiqa  | mrxlmr-tydiqa-0pct-predel            | 1    |
| V  | MrXLMR | TyDi QA  | 30%      | layer 9, no pre-del blend    | mrxlmr-tydiqa  | mrxlmr-tydiqa-30pct-layer9-no-predel | 1    |
| W  | MrXLMR | TyDi QA  | 30%      | layer 3 + pre-del blend      | mrxlmr-tydiqa  | mrxlmr-tydiqa-30pct-predel           | 1    |
| X  | XLMR   | XNLI     | —        | baseline                     | mrxlmr-xnli    | xlmr-xnli-baseline                   | 1    |
| Y  | MrXLMR | XNLI     | 30%      | main result                  | mrxlmr-xnli    | mrxlmr-xnli-30pct                    | 1    |

**Tier 1** = essential for the advisor meeting.
**Tier 2** = adds depth for the gate layer ablation chart and PI controller ablation; launch alongside Tier 1 if bandwidth allows.

---

## Time Estimates

| Phase     | Work                                        | Estimate            |
|-----------|---------------------------------------------|---------------------|
| Phase 1   | Launch all runs (terminal commands)         | ~10 min             |
| Phase 1   | Wait for SNLI runs to complete (A–J, M)     | ~90–100 min on A100 |
| Phase 1   | Wait for SQuAD runs to complete (K–L)       | ~40 min on A100     |
| Phase 1   | Wait for SST-2 runs to complete (N–O)       | ~15 min on A100     |
| Phase 1   | Wait for MRPC runs to complete (P–Q)        | ~5 min on A100      |
| Phase 1   | Wait for IMDB runs to complete (R–S)        | ~20 min on A100     |
| Phase 1   | Wait for TyDi QA runs to complete (T–W)     | ~10 min on A100     |
| Phase 1   | Wait for XNLI runs to complete (X–Y)        | ~20 min on A100     |
| Phase 2   | Download all checkpoints from Modal Volume  | ~15 min             |
| Phase 3   | Capture metrics from W&B                    | ~10 min             |
| Phase 4   | Runtime measurement (measure_runtime.py)    | ~10 min             |
| Phase 4.5 | Hard deletion curve (post-hoc eval)         | ~20 min             |
| Phase 5   | Deletion pattern analysis                   | ~5 min              |
| Phase 6   | Generate all charts                         | ~5 min              |
| **Total** |                                             | **~3.5 hours**      |

---

## Phase 1 — Launch Training Runs

> **All runs in this phase are PARALLEL. Launch all of them from separate terminals (or as a batch) before doing anything else. Each run is independent and Modal handles parallelism.**
> **Run all commands from the `mrxlmr/training/` directory.**

### SNLI — Tier 1 Core (Runs A, B, C, D, E, F, M)

```bash
modal run --detach train_modal.py --model-type XLMR --task sequence_classification --num-epochs 3 --max-steps -1 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name xlmr-snli-baseline
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.0 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-0pct
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-30pct
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.5 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-50pct
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.7 --deletion-loss-weight 0.1 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-70pct
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --deletion-type random --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-random30
```

### SNLI — Tier 2 Ablations (Runs G, H, I, J)

> **Run G requires `use_pi_controller` to be exposed in `main()`.** Add `use_pi_controller: bool = True` to the `main()` parameter list in `training/train_modal.py` and pass it through to `train.remote()`. Without this change, the PI controller is always enabled and `--no-use-pi-controller` has no effect.

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --no-use-pi-controller --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-nopi
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --delete-gate-layer 1 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-layer1
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --delete-gate-layer 6 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-layer6
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --delete-gate-layer 9 --mode training-and-eval --wandb-project mrxlmr-snli --wandb-run-name mrxlmr-snli-layer9
```

### SQuAD — Tier 1 Core (Runs K, L)

```bash
modal run --detach train_modal.py --model-type XLMR --task question_answering --dataset-name local_squad --num-epochs 3 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrxlmr-squad --wandb-run-name xlmr-squad-baseline
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task question_answering --dataset-name local_squad --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 400 --mode training-and-eval --wandb-project mrxlmr-squad --wandb-run-name mrxlmr-squad-30pct
```

### SST-2 — Tier 1 Core (Runs N, O)

```bash
modal run --detach train_modal.py --model-type XLMR --task sequence_classification --dataset-name local_sst2 --num-epochs 3 --max-steps -1 --batch-size 32 --mode training-and-eval --wandb-project mrxlmr-sst2 --wandb-run-name xlmr-sst2-baseline
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --dataset-name local_sst2 --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 32 --regularizer-delay 300 --mode training-and-eval --wandb-project mrxlmr-sst2 --wandb-run-name mrxlmr-sst2-30pct
```

### MRPC — Tier 1 Core (Runs P, Q)

```bash
modal run --detach train_modal.py --model-type XLMR --task sequence_classification --dataset-name local_mrpc --num-epochs 5 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrxlmr-mrpc --wandb-run-name xlmr-mrpc-baseline
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --dataset-name local_mrpc --num-epochs 5 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 50 --mode training-and-eval --wandb-project mrxlmr-mrpc --wandb-run-name mrxlmr-mrpc-30pct
```

### IMDB — Tier 1 Core (Runs R, S)

```bash
modal run --detach train_modal.py --model-type XLMR --task sequence_classification --dataset-name local_imdb --num-epochs 3 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrxlmr-imdb --wandb-run-name xlmr-imdb-baseline
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --dataset-name local_imdb --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 300 --mode training-and-eval --wandb-project mrxlmr-imdb --wandb-run-name mrxlmr-imdb-30pct
```

### TyDi QA — Tier 1 Core (Runs T, U, V, W)

> **TyDi QA requires pre-deletion blending to work correctly at 30% deletion.** Run U (0% sanity check) and V (layer 9 without pre-del blend) to confirm the problem; run W (layer 3 + pre-del blend) as the solution. Runs V and W together confirm that pre-deletion blending is the critical difference.

```bash
modal run --detach train_modal.py --model-type XLMR --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --batch-size 16 --mode training-and-eval --wandb-project mrxlmr-tydiqa --wandb-run-name xlmr-tydiqa-baseline
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --target-deletion-rate 0.0 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 100 --mode training-and-eval --wandb-project mrxlmr-tydiqa --wandb-run-name mrxlmr-tydiqa-0pct-predel
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 100 --delete-gate-layer 9 --no-use-pre-deletion-blend --mode training-and-eval --wandb-project mrxlmr-tydiqa --wandb-run-name mrxlmr-tydiqa-30pct-layer9-no-predel
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task question_answering --dataset-name local_tydiqa --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 16 --regularizer-delay 100 --mode training-and-eval --wandb-project mrxlmr-tydiqa --wandb-run-name mrxlmr-tydiqa-30pct-predel
```

> **Note:** Target deletion rate 30% with layer 9 (no pre-del) and layer 3 (with pre-del) should both achieve results comparable to the XLM-R baseline, confirming that pre-deletion blending is the key enabling factor for extractive QA.

---

### XNLI — Tier 1 Core (Runs X, Y)

> **XNLI is the multilingual-specific contribution of MrXLMR over MrBERT.** Both runs train on the English XNLI split; the cross-lingual test evaluation (zero-shot transfer to zh, de, sw, fr) is run separately after downloading checkpoints (see Phase 5.5).

```bash
modal run --detach train_modal.py --model-type XLMR --task sequence_classification --dataset-name local_xnli --num-epochs 3 --max-steps -1 --batch-size 32 --mode training-and-eval --wandb-project mrxlmr-xnli --wandb-run-name xlmr-xnli-baseline
```

```bash
modal run --detach train_modal.py --model-type MrXLMR --task sequence_classification --dataset-name local_xnli --num-epochs 3 --max-steps -1 --target-deletion-rate 0.3 --deletion-loss-weight 0.1 --batch-size 32 --regularizer-delay 500 --mode training-and-eval --wandb-project mrxlmr-xnli --wandb-run-name mrxlmr-xnli-30pct
```

---

## Phase 2 — Download Checkpoints (Optional but Recommended)

> **Sequential. Run after Phase 1 runs complete. Monitor progress at https://modal.com/apps.**
> **Run from the `mrxlmr/` directory.**

Download all completed checkpoints to local disk:

```bash
for run in xlmr-snli-baseline mrxlmr-snli-0pct mrxlmr-snli-30pct mrxlmr-snli-30pct-hd mrxlmr-snli-50pct mrxlmr-snli-70pct mrxlmr-snli-random30 mrxlmr-snli-layer1 mrxlmr-snli-layer6 mrxlmr-snli-layer9 xlmr-squad-baseline mrxlmr-squad-30pct xlmr-sst2-baseline mrxlmr-sst2-30pct xlmr-mrpc-baseline mrxlmr-mrpc-30pct xlmr-imdb-baseline mrxlmr-imdb-30pct xlmr-tydiqa-baseline mrxlmr-tydiqa-0pct-predel mrxlmr-tydiqa-30pct-layer9-no-predel mrxlmr-tydiqa-30pct-predel xlmr-xnli-baseline mrxlmr-xnli-30pct; do mkdir -p ./local_checkpoints/${run}/final && for f in config.json tokenizer_config.json tokenizer.json sentencepiece.bpe.model special_tokens_map.json model.safetensors; do modal volume get mrxlmr-checkpoints ${run}/final/${f} ./local_checkpoints/${run}/final/${f} 2>/dev/null || true; done && echo "Downloaded ${run}"; done
```

> **XLM-R tokenizer files:** XLM-R uses SentencePiece (`sentencepiece.bpe.model`) and `tokenizer.json` instead of BERT's `vocab.txt`. The loop above downloads both. Verify with `ls ./local_checkpoints/xlmr-snli-baseline/final/`.

---

## Phase 3 — Capture Metrics from W&B

> **Sequential. Do this while waiting for downloads or immediately after.**
> **These numbers will be substituted into the chart commands in Phase 6.**

For each completed run, open W&B and copy the following values from the **Summary** tab. Record them in the tables below.

### SNLI Results Table (fill in from W&B)

| Run | Model                      | Del Rate Target | test/accuracy | percent_non_pad_deleted_tokens | new_seq_len | seq_len_reduction_pct |
|-----|----------------------------|-----------------|---------------|--------------------------------|-------------|----------------------|
| A   | XLMR baseline              | 0%              | `___`         | 0                              | 128         | 0                    |
| B   | MrXLMR 0%                  | 0%              | `___`         | `___`                          | `___`       | `___`                |
| C   | MrXLMR 30%                 | 30%             | `___`         | `___`                          | `___`       | `___`                |
| M   | MrXLMR 30% hard-train      | 30%             | `___`         | `___`                          | `___`       | `___`                |
| D   | MrXLMR 50%                 | 50%             | `___`         | `___`                          | `___`       | `___`                |
| E   | MrXLMR 70%                 | 70%             | `___`         | `___`                          | `___`       | `___`                |
| F   | MrXLMR Random 30%          | 30%             | `___`         | `___`                          | `___`       | `___`                |
| G   | MrXLMR No-PI 30%           | 30%             | `___`         | `___`                          | `___`       | `___`                |
| H   | MrXLMR Layer 1             | 30%             | `___`         | `___`                          | `___`       | `___`                |
| C   | MrXLMR Layer 3 (reuse C)   | 30%             | `___`         | `___`                          | `___`       | `___`                |
| I   | MrXLMR Layer 6             | 30%             | `___`         | `___`                          | `___`       | `___`                |
| J   | MrXLMR Layer 9             | 30%             | `___`         | `___`                          | `___`       | `___`                |

### SQuAD Results Table (fill in from W&B)

| Run | Model          | Del Rate | test/squad_em | test/squad_f1 | seq_len_reduction_pct |
|-----|----------------|----------|---------------|---------------|-----------------------|
| K   | XLMR baseline  | 0%       | `___`         | `___`         | 0                     |
| L   | MrXLMR 30%     | 30%      | `___`         | `___`         | `___`                 |

### SST-2 Results Table (fill in from W&B)

| Run | Model          | Del Rate | test/accuracy | seq_len_reduction_pct |
|-----|----------------|----------|---------------|-----------------------|
| N   | XLMR baseline  | 0%       | `___`         | 0                     |
| O   | MrXLMR 30%     | 30%      | `___`         | `___`                 |

### MRPC Results Table (fill in from W&B)

| Run | Model          | Del Rate | test/accuracy | seq_len_reduction_pct |
|-----|----------------|----------|---------------|-----------------------|
| P   | XLMR baseline  | 0%       | `___`         | 0                     |
| Q   | MrXLMR 30%     | 30%      | `___`         | `___`                 |

### IMDB Results Table (fill in from W&B)

| Run | Model          | Del Rate | test/accuracy | seq_len_reduction_pct |
|-----|----------------|----------|---------------|-----------------------|
| R   | XLMR baseline  | 0%       | `___`         | 0                     |
| S   | MrXLMR 30%     | 30%      | `___`         | `___`                 |

### TyDi QA Results Table (fill in from W&B)

| Run | Model                       | Del Rate | test/squad_em | test/squad_f1 | seq_len_reduction_pct |
|-----|-----------------------------|----------|---------------|---------------|-----------------------|
| T   | XLMR baseline               | 0%       | `___`         | `___`         | 0                     |
| U   | MrXLMR 0% (sanity)         | 0%       | `___`         | `___`         | 0                     |
| V   | MrXLMR 30% layer9 no-predel | 30%      | `___`         | `___`         | `___`                 |
| W   | MrXLMR 30% layer3 predel    | 30%      | `___`         | `___`         | `___`                 |

### XNLI Results Table (fill in from W&B)

| Run | Model          | Del Rate | test/accuracy (en) | seq_len_reduction_pct |
|-----|----------------|----------|--------------------|-----------------------|
| X   | XLMR baseline  | 0%       | `___`              | 0                     |
| Y   | MrXLMR 30%     | 30%      | `___`              | `___`                 |

> **W&B tip:** In each run's Summary tab, search for `test/` to find all final test metrics. The metrics `seq_len_reduction_pct` and `new_seq_len` are logged under the eval prefix at the last step.

---

## Phase 4 — Runtime Measurement

> **Requires Phase 2 (checkpoints downloaded locally).**
> **For reliable GPU timing, run this on a machine with a CUDA GPU. CPU timing is not representative.**
> **Run from the `mrxlmr/` directory.**

Run the runtime benchmark using the downloaded SNLI checkpoints:

```bash
python analysis/measure_runtime.py --model_paths ./local_checkpoints/xlmr-snli-baseline/final ./local_checkpoints/mrxlmr-snli-0pct/final ./local_checkpoints/mrxlmr-snli-30pct/final ./local_checkpoints/mrxlmr-snli-30pct-hd/final ./local_checkpoints/mrxlmr-snli-50pct/final ./local_checkpoints/mrxlmr-snli-70pct/final ./local_checkpoints/mrxlmr-snli-random30/final --snli_dir ./snli_datasets --output_dir ./analysis/figures
```

Gate layer ablation runtime (for the `gate_layer_ablation.pdf` chart):

```bash
python analysis/measure_runtime.py --model_paths ./local_checkpoints/mrxlmr-snli-layer1/final ./local_checkpoints/mrxlmr-snli-30pct/final ./local_checkpoints/mrxlmr-snli-layer6/final ./local_checkpoints/mrxlmr-snli-layer9/final --snli_dir ./snli_datasets --output_dir ./analysis/figures
```

**Fill in the runtime table from `analysis/figures/runtime_table.csv`:**

| Model                 | ms/sample | % decrease vs XLMR |
|-----------------------|-----------|--------------------|
| XLMR baseline         | `___`     | —                  |
| MrXLMR 0%             | `___`     | `___`              |
| MrXLMR 30%            | `___`     | `___`              |
| MrXLMR 30% hard-train | `___`     | `___`              |
| MrXLMR 50%            | `___`     | `___`              |
| MrXLMR 70%            | `___`     | `___`              |
| Random 30%            | `___`     | `___`              |
| Layer 1               | `___`     | `___`              |
| Layer 3               | `___`     | `___`              |
| Layer 6               | `___`     | `___`              |
| Layer 9               | `___`     | `___`              |

Saved files: `analysis/figures/runtime_table.csv`, `analysis/figures/runtime_vs_deletion.pdf`

---

## Phase 4.5 — Hard Deletion Curve

> **Requires Phase 2 (checkpoints downloaded locally, including all intermediate `checkpoint-*` subdirs).**
> **Shows that hard deletion accuracy tracks soft deletion accuracy throughout training — confirms the model does not collapse when tokens are physically removed at inference.**
> **Run from the `mrxlmr/` directory.**

Run for the main 30% soft-deletion model (run C) and the hard-train variant (run M):

```bash
python analysis/hard_deletion_curve.py --checkpoint_dir ./local_checkpoints/mrxlmr-snli-30pct --local_snli_dir ./snli_datasets --run_name mrxlmr-snli-30pct --output_dir ./analysis/figures
```

```bash
python analysis/hard_deletion_curve.py --checkpoint_dir ./local_checkpoints/mrxlmr-snli-30pct-hd --local_snli_dir ./snli_datasets --run_name mrxlmr-snli-30pct-hd --output_dir ./analysis/figures
```

> **Downloading intermediate checkpoints from Modal:** The `hard_deletion_curve.py` script requires `checkpoint-*` subdirs (saved every `save_steps`). Ensure that `--save-steps` was set to a reasonable interval during training (default: 1000). Download intermediate checkpoints with:

```bash
modal volume get mrxlmr-checkpoints mrxlmr-snli-30pct ./local_checkpoints/mrxlmr-snli-30pct
```

```bash
modal volume get mrxlmr-checkpoints mrxlmr-snli-30pct-hd ./local_checkpoints/mrxlmr-snli-30pct-hd
```

**What to look for:**
- Both soft and hard deletion lines should track closely throughout training
- Run C (soft-only training): expect a small but nonzero soft–hard gap at the end
- Run M (hard_delete_train_prob=0.5): gap should be near zero, since the model was exposed to hard deletion during training
- A large or growing gap = the model relies on deleted tokens still being in the attention context (tokens are masked but attention mechanism compensates)

Saved files: `analysis/figures/mrxlmr-snli-30pct_hard_deletion_curve.csv`, `analysis/figures/mrxlmr-snli-30pct_hard_deletion_curve.pdf`, `analysis/figures/mrxlmr-snli-30pct-hd_hard_deletion_curve.csv`, `analysis/figures/mrxlmr-snli-30pct-hd_hard_deletion_curve.pdf`

---

## Phase 5 — Deletion Pattern Analysis

> **Parallel with Phase 4. Requires Phase 2 (MrXLMR 30% SNLI checkpoint downloaded).**
> **Run from the `mrxlmr/` directory.**

```bash
python analysis/get_deletion_patterns.py --model_path ./local_checkpoints/mrxlmr-snli-30pct/final --local_snli_dir ./snli_datasets --sample_size 1000 --output_dir ./analysis/deletion_patterns --output_file mrxlmr-snli-30pct_test.json
```

```bash
python analysis/deletion_pattern_analysis.py --input_file ./analysis/deletion_patterns/mrxlmr-snli-30pct_test.json --output_dir ./analysis/figures
```

Saved files: `analysis/figures/mrxlmr-snli-30pct_test_by_type.pdf`, `analysis/figures/mrxlmr-snli-30pct_test_premise_vs_hyp.pdf`

> **XLM-R tokenization note:** XLM-R uses `▁` (SentencePiece word-boundary marker) on word-initial subwords. The analysis script detects this and categorises tokens into word-initial, continuation subword, punctuation, and special token groups. The SNLI pair separator is `</s></s>` (two consecutive EOS tokens), and the script splits premise from hypothesis at the first `</s>`.

---

## Phase 5.5 — Cross-Lingual XNLI Evaluation

> **Requires Phase 2 (XNLI checkpoints downloaded). Unique to MrXLMR — not possible with MrBERT.**
> **Tests whether deletion-trained models retain zero-shot transfer accuracy across languages.**
> **Run from the `mrxlmr/` directory.**

First preprocess the cross-lingual test files (Chinese, German, Swahili, French):

```bash
python data/preprocess_xnli.py --output_dir ./xnli_datasets --test_languages zh,de,sw,fr
```

Then evaluate each checkpoint on each language test set using `eval_mrxlmr.py` or a direct inference loop:

```bash
python eval/eval_mrxlmr.py --model_path ./local_checkpoints/xlmr-xnli-baseline/final --test_file ./xnli_datasets/xnli-test-zh.json --output_dir ./analysis/figures --run_name xlmr-xnli-baseline-zh
```

```bash
python eval/eval_mrxlmr.py --model_path ./local_checkpoints/mrxlmr-xnli-30pct/final --test_file ./xnli_datasets/xnli-test-zh.json --output_dir ./analysis/figures --run_name mrxlmr-xnli-30pct-zh
```

Repeat for `de`, `sw`, `fr`. **Fill in the cross-lingual results table:**

| Language | XLMR accuracy | MrXLMR-30% accuracy | Drop (pp) |
|----------|---------------|---------------------|-----------|
| en       | `___`         | `___`               | `___`     |
| zh       | `___`         | `___`               | `___`     |
| de       | `___`         | `___`               | `___`     |
| sw       | `___`         | `___`               | `___`     |
| fr       | `___`         | `___`               | `___`     |

**What to look for:**
- Does the accuracy drop from deletion stay consistent across languages, or does it hurt low-resource languages (sw) more than high-resource ones (de, fr, zh)?
- A uniform drop = the gate is language-agnostic. A larger drop for sw = the gate may be over-fitted to English token importance patterns.

---

## Phase 6 — Generate Charts

> **Sequential. Requires Phase 3 (W&B metrics captured) and Phase 4 (runtime measured).**
> **Replace all `___` placeholders with actual values before running.**
> **Run from the `mrxlmr/` directory.**

### Theoretical compute savings + accuracy vs compute + accuracy vs seq-length reduction

Replace each `___` with the actual `test/accuracy` value from the W&B table:

```bash
python analysis/compute_savings.py --output_dir analysis/figures --runs "XLMR,___,0.0" "MrXLMR-0%,___,0.0" "MrXLMR-30%,___,0.30" "MrXLMR-50%,___,0.50" "MrXLMR-70%,___,0.70" "Random-30%,___,0.30"
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

| File                                                                               | What it shows                                              | Advisor relevance                   |
|------------------------------------------------------------------------------------|------------------------------------------------------------|-------------------------------------|
| `analysis/figures/accuracy_vs_seq_reduction.pdf`                                  | Accuracy vs sequence length reduction % — core tradeoff    | Primary result                      |
| `analysis/figures/accuracy_vs_compute.pdf`                                         | Accuracy vs relative MACs — efficiency frontier            | Primary result                      |
| `analysis/figures/runtime_vs_deletion.pdf`                                         | Measured ms/sample per model                               | Core claim: actual speedup          |
| `analysis/figures/runtime_table.csv`                                               | Runtime table for paper (mirrors MrT5 Table 3)             | For paper table                     |
| `analysis/figures/gate_layer_ablation.pdf`                                         | Accuracy + runtime vs gate layer — dual axis               | Ablation                            |
| `analysis/figures/macs_relative.pdf`                                               | Theoretical compute savings curve                          | Background                          |
| `analysis/figures/macs_by_gate_layer.pdf`                                          | Theoretical savings by gate layer                          | Background                          |
| `analysis/figures/mrxlmr-snli-30pct_hard_deletion_curve.pdf`                      | Soft vs hard deletion accuracy vs training step (run C)    | Validates hard deletion robustness  |
| `analysis/figures/mrxlmr-snli-30pct-hd_hard_deletion_curve.pdf`                   | Soft vs hard deletion accuracy vs training step (run M)    | Validates hard deletion robustness  |
| `analysis/figures/mrxlmr-snli-30pct_test_by_type.pdf`                             | Deletion rate by token type (word/subword/punct)           | Qualitative                         |
| `analysis/figures/mrxlmr-snli-30pct_test_premise_vs_hyp.pdf`                      | Premise vs hypothesis deletion rate                        | Qualitative                         |
| `analysis/deletion_patterns/mrxlmr-snli-30pct_test.json`                          | Per-token gate decisions, 1000 examples                    | For colored examples in slides      |
| `analysis/figures/xnli_crosslingual_accuracy.csv` *(manual)*                      | Zero-shot transfer accuracy per language (X vs Y)          | Unique MrXLMR result                |

---

## What to Present to Your Advisor

### Structure (30–40 min)

**1. Motivation (2 min)**
- XLM-R processes all 128 tokens with equal compute regardless of informativeness
- MrT5 showed learned token deletion works for encoder-decoder (T5) models
- MrBERT showed it works for BERT-based discriminative tasks
- Research question: can we apply the same mechanism to XLM-RoBERTa and does multilingual pre-training help or hinder compression?

**2. Architecture (5 min)**
- Delete gate inserted after encoder layer 3 (out of 12)
- Gate = `LayerNorm → Linear(768→1) → scaled sigmoid`, 2,305 additional parameters out of 277M (XLM-R-base)
- Soft deletion during training: large negative bias added to attention scores of deleted tokens
- PI controller dynamically adjusts deletion pressure α to hit target deletion rate δ
- Pre-deletion blending: deleted tokens blend their pre-gate hidden state with the gated output, critical for extractive QA
- Show: gate equation + diagram of where it sits in the XLM-R encoder stack
- Key XLM-R specifics: `<s>`/`</s>` special tokens (not `[CLS]`/`[SEP]`), SentencePiece tokenization, multilingual vocabulary of 250K tokens

**3. Core Results — SNLI (5 min)**

Show the filled-in results table + `accuracy_vs_seq_reduction.pdf`:

| Model            | Accuracy | Del Rate | Seq Δ | Runtime  |
|------------------|----------|----------|-------|----------|
| XLMR baseline    | `___`    | 0%       | 0%    | `___` ms |
| MrXLMR 0%        | `___`    | 0%       | 0%    | `___` ms |
| MrXLMR 30%       | `___`    | 30%      | `___`%| `___` ms |
| MrXLMR 50%       | `___`    | 50%      | `___`%| `___` ms |
| MrXLMR 70%       | `___`    | 70%      | `___`%| `___` ms |
| Random gate 30%  | `___`    | 30%      | 30%   | `___` ms |

Key talking points:
- MrXLMR 0% should match XLMR — validates the implementation is fair
- Compare MrXLMR 30% accuracy vs XLMR baseline — how small is the gap?
- Compare MrXLMR 30% vs Random 30% — the learned gate should outperform random, proving the gate is learning non-trivial structure
- Show `runtime_vs_deletion.pdf` — the actual measured GPU speedup

**4. Core Results — SQuAD and TyDi QA (5 min)**

| Model                       | EM    | F1    | Seq Δ | Runtime  |
|-----------------------------|-------|-------|-------|----------|
| XLMR baseline               | `___` | `___` | 0%    | `___` ms |
| MrXLMR 30% layer3 predel    | `___` | `___` | `___`%| `___` ms |
| MrXLMR 30% layer9 no-predel | `___` | `___` | `___`%| `___` ms |

Key talking point: extractive QA requires the exact answer span to survive deletion. Pre-deletion blending (blending pre-gate hidden states for deleted tokens) is the key mechanism that makes 30% deletion viable. Show that layer 9 without blending degrades sharply while layer 3 with blending does not.

**5. Ablations (5 min)**
- `gate_layer_ablation.pdf`: earlier gate → faster but less accurate; layer 3 is the sweet spot
- No PI controller: what happens to actual deletion rate without adaptive control?
- Show `mrxlmr-snli-30pct_hard_deletion_curve.pdf`: soft vs hard deletion accuracy over training — confirms the model does not collapse under hard deletion at inference
- Show `mrxlmr-snli-30pct_test_by_type.pdf`: which token types get deleted? (function words vs content words; SentencePiece continuation subwords vs word-initial)
- Show `mrxlmr-snli-30pct_test_premise_vs_hyp.pdf`: does the gate treat premise and hypothesis differently?

**6. Discussion (10 min)**
- The efficiency frontier: how far down the tradeoff curve is acceptable?
- Does the gate learn linguistically meaningful patterns that differ from English-only BERT?
- Does multilingual XLM-R pre-training make token deletion harder or easier to learn?
- What should the next set of experiments be?

---

## Questions for Your Advisor

These questions will directly shape the second milestone experiments.

### Scope and Tasks
1. Is SNLI + SQuAD sufficient as the evaluation suite, or do you recommend adding a multilingual NLI task (e.g., XNLI in several languages) to leverage XLM-R's cross-lingual capability?
2. MrT5 uses 15 languages to demonstrate language-specific compression rates. For a cross-lingual paper, should we include XNLI or multilingual SQuAD to test whether the gate compresses different languages at different rates?

### Architecture and Design
3. We use soft deletion throughout training (attention masking) and could optionally use hard deletion (physically remove tokens) at test time. Is the hard vs soft deletion gap comparison important to include in the paper?
4. The gate is currently at layer 3 (out of 12). MrT5 uses layer 3 out of 12 for the same reason. Should we justify this choice with a thorough ablation (layers 1–11), or is a sample of 4 layers (1, 3, 6, 9) sufficient?
5. We initialize MrXLMR from pretrained XLM-R weights and only add 2,305 gate parameters. Is this the right framing, or should we also experiment with training the gate from scratch to test whether multilingual pretraining specifically helps the gate learn language-appropriate compression?

### Baselines
6. MrT5 compares against Boundary Predictor and Convolutional Pooling baselines. For an XLM-R-based paper, are there equivalent compression baselines we should include (e.g., SpAtten, TR-BERT, LTP)?
7. How much emphasis should the random deletion baseline receive? Is it sufficient to show MrXLMR > random, or do reviewers expect a richer set of non-learned baselines?

### Metrics and Claims
8. The runtime speedup we measure is for forward-pass inference on a single GPU. For a publication claim, should we also measure wall-clock speedup in a batched serving scenario (where hard deletion breaks batch uniformity)?
9. MrT5 reports bits-per-byte (language modeling loss). Our equivalent is accuracy/EM/F1. Is there any value in also pre-training MrXLMR on MLM before fine-tuning to get a language-modeling-style efficiency comparison?

### Paper Framing
10. How should we frame the contribution relative to MrT5 and MrBERT? Options: (a) direct adaptation of MrT5 to XLM-R as a cross-lingual study, (b) independent extension showing the mechanism generalises across architectures and language families, (c) applied study showing practical efficiency gains for multilingual discriminative NLU.
11. What is the target venue? (ACL, EMNLP, NAACL, workshop?) — this affects how thorough the ablations need to be and how many tasks and languages are required.
12. Is there a specific result threshold that would make this "publishable" — e.g., a minimum runtime speedup % at a maximum accuracy drop %, or a minimum number of languages evaluated?