# MrBERT Run Commands

cd mrbert/training/

## SST-2
~67k train · 128 tokens · ~6,300 steps/3 epochs

```bash
# MrBERT
modal run --detach train_modal.py::main --task sequence_classification --dataset-name local_sst2 --model-type MrBERT --max-steps -1 --num-epochs 3 --batch-size 32 --regularizer-delay 300 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name mrbert-sst2-30pct --wandb-project mrbert-sst2
```

```bash
# BERT baseline
modal run --detach train_modal.py::main --task sequence_classification --dataset-name local_sst2 --model-type BERT --max-steps -1 --num-epochs 3 --batch-size 32 --mode training-and-eval --wandb-run-name bert-sst2-baseline --wandb-project mrbert-sst2
```

---

## MRPC
~3.7k train · 128 tokens · ~575 steps/3 epochs

```bash
# MrBERT
modal run --detach train_modal.py::main --task sequence_classification --dataset-name local_mrpc --model-type MrBERT --max-steps -1 --num-epochs 5 --batch-size 16 --regularizer-delay 50 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name mrbert-mrpc-30pct --wandb-project mrbert-mrpc
```

```bash
# BERT baseline
modal run --detach train_modal.py::main --task sequence_classification --dataset-name local_mrpc --model-type BERT --max-steps -1 --num-epochs 5 --batch-size 16 --mode training-and-eval --wandb-run-name bert-mrpc-baseline --wandb-project mrbert-mrpc
```

---

## IMDB
~22.5k train · 512 tokens · ~4,200 steps/3 epochs

```bash
# MrBERT
modal run --detach train_modal.py::main --task sequence_classification --dataset-name local_imdb --model-type MrBERT --max-steps -1 --num-epochs 3 --batch-size 16 --regularizer-delay 300 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name mrbert-imdb-30pct --wandb-project mrbert-imdb
```

```bash
# BERT baseline
modal run --detach train_modal.py::main --task sequence_classification --dataset-name local_imdb --model-type BERT --max-steps -1 --num-epochs 3 --batch-size 16 --mode training-and-eval --wandb-run-name bert-imdb-baseline --wandb-project mrbert-imdb
```

---

## TyDi QA (English)
~3.2k English examples → ~3.2k features · 384 tokens · ~600 steps/3 epochs
(GoldP passages are pre-cropped and short; sliding window rarely fires, giving ~1.02x expansion vs the ~1.22x assumed for SQuAD)

```bash
# BERT baseline
modal run --detach train_modal.py::main --task question_answering --dataset-name local_tydiqa --model-type BERT --max-steps -1 --num-epochs 3 --batch-size 16 --mode training-and-eval --wandb-run-name bert-tydiqa-baseline --wandb-project mrbert-tydiqa
```

```bash
# MrBERT 30% — layer 3, no pre-deletion blend (original; exhibits gate collapse ~61%)
modal run --detach train_modal.py::main --task question_answering --dataset-name local_tydiqa --model-type MrBERT --max-steps -1 --num-epochs 3 --batch-size 16 --regularizer-delay 100 --target-deletion-rate 0.3 --no-use-pre-deletion-blend --mode training-and-eval --wandb-run-name mrbert-tydiqa-30pct --wandb-project mrbert-tydiqa
```

```bash
# MrBERT 30% — layer 3 + pre-deletion blend (3x improvement over no-blend; span_em ~0.30)
modal run --detach train_modal.py::main --task question_answering --dataset-name local_tydiqa --model-type MrBERT --max-steps -1 --num-epochs 3 --batch-size 16 --regularizer-delay 100 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name mrbert-tydiqa-30pct-predel --wandb-project mrbert-tydiqa
```

```bash
# MrBERT 30% — layer 9 + pre-deletion blend (best result; span_em ~0.35, end_acc matches BERT)
modal run --detach train_modal.py::main --task question_answering --dataset-name local_tydiqa --model-type MrBERT --max-steps -1 --num-epochs 3 --batch-size 16 --regularizer-delay 100 --target-deletion-rate 0.3 --delete-gate-layer 9 --mode training-and-eval --wandb-run-name mrbert-tydiqa-30pct-layer9-predel --wandb-project mrbert-tydiqa
```

```bash
# MrBERT 0% deletion — bypass gate (clean control; architecturally identical to BERT)
modal run --detach train_modal.py::main --task question_answering --dataset-name local_tydiqa --model-type MrBERT --max-steps -1 --num-epochs 3 --batch-size 16 --regularizer-delay 100 --target-deletion-rate 0.0 --bypass-gate --mode training-and-eval --wandb-run-name mrbert-tydiqa-bypass --wandb-project mrbert-tydiqa
```

---

## SNLI
~549k train · 128 tokens · ~51,500 steps/3 epochs

```bash
# MrBERT
modal run --detach train_modal.py::main --task sequence_classification --dataset-name local_snli --model-type MrBERT --max-steps -1 --num-epochs 3 --batch-size 32 --regularizer-delay 1000 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name mrbert-snli-30pct --wandb-project mrbert-snli
```

```bash
# BERT baseline
modal run --detach train_modal.py::main --task sequence_classification --dataset-name local_snli --model-type BERT --max-steps -1 --num-epochs 3 --batch-size 32 --mode training-and-eval --wandb-run-name bert-snli-baseline --wandb-project mrbert-snli
```

---

## SQuAD
~87k train examples → ~88k features · 384 tokens · ~5,500 steps/3 epochs

```bash
# MrBERT
modal run --detach train_modal.py::main --task question_answering --dataset-name local_squad --model-type MrBERT --max-steps -1 --num-epochs 3 --batch-size 16 --regularizer-delay 400 --target-deletion-rate 0.3 --mode training-and-eval --wandb-run-name mrbert-squad-30pct --wandb-project mrbert-squad
```

```bash
# BERT baseline
modal run --detach train_modal.py::main --task question_answering --dataset-name local_squad --model-type BERT --max-steps -1 --num-epochs 3 --batch-size 16 --mode training-and-eval --wandb-run-name bert-squad-baseline --wandb-project mrbert-squad
```

---

## Download Checkpoints

```bash
modal volume get mrbert-checkpoints mrbert-sst2-30pct/final ./local_checkpoints/mrbert-sst2
modal volume get mrbert-checkpoints bert-sst2-baseline/final ./local_checkpoints/bert-sst2
modal volume get mrbert-checkpoints mrbert-mrpc-30pct/final ./local_checkpoints/mrbert-mrpc
modal volume get mrbert-checkpoints bert-mrpc-baseline/final ./local_checkpoints/bert-mrpc
modal volume get mrbert-checkpoints mrbert-imdb-30pct/final ./local_checkpoints/mrbert-imdb
modal volume get mrbert-checkpoints bert-imdb-baseline/final ./local_checkpoints/bert-imdb
modal volume get mrbert-checkpoints bert-tydiqa-baseline/final ./local_checkpoints/bert-tydiqa
modal volume get mrbert-checkpoints mrbert-tydiqa-30pct/final ./local_checkpoints/mrbert-tydiqa
modal volume get mrbert-checkpoints mrbert-tydiqa-30pct-predel/final ./local_checkpoints/mrbert-tydiqa-predel
modal volume get mrbert-checkpoints mrbert-tydiqa-30pct-layer9-predel/final ./local_checkpoints/mrbert-tydiqa-layer9-predel
modal volume get mrbert-checkpoints mrbert-snli-30pct/final ./local_checkpoints/mrbert-snli
modal volume get mrbert-checkpoints bert-snli-baseline/final ./local_checkpoints/bert-snli
modal volume get mrbert-checkpoints mrbert-squad-30pct/final ./local_checkpoints/mrbert-squad
modal volume get mrbert-checkpoints bert-squad-baseline/final ./local_checkpoints/bert-squad
```

---

## Hyperparameter Reference

| Dataset | batch_size | regularizer_delay | num_epochs | Reason |
|---------|------------|-------------------|------------|--------|
| SST-2   | 32         | 300               | 3          | Standard; 300 ≈ 5% of 6.3k steps |
| MRPC    | 16         | 50                | 5          | Tiny dataset; more epochs to converge; 50 ≈ 8% of 575 steps |
| IMDB    | 16         | 300               | 3          | 512-token sequences need smaller batch; 300 ≈ 7% of 4.2k steps |
| TyDi QA | 16         | 100               | 3          | 384-token QA; 100 ≈ 17% of 600 steps |
| SNLI    | 32         | 1000              | 3          | Large dataset; 1000 ≈ 2% of 51.5k steps |
| SQuAD   | 16         | 400               | 3          | 384-token QA; 400 ≈ 7% of 5.5k steps |