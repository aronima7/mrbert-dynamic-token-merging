# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MrBERT is a CS224N project that adapts the **delete gate mechanism from MrT5** to the BERT architecture. The core idea: after a specified encoder layer, a learned gate assigns each token a scalar score; low-scoring tokens are masked out of subsequent attention layers (soft deletion) or physically removed (hard deletion), reducing computation while preserving performance.

Reference paper: [MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/pdf/2410.20771) (Kallini et al., 2024)

## Environment Setup

```bash
conda create -n mrbert python=3.11 -y
conda activate mrbert
pip install modal torch transformers datasets accelerate tqdm matplotlib "numpy<2"
```

## Commands

### Run tests
```bash
python test_mrbert.py

# Run a single test function
python -c "from test_mrbert import test_config_creation; test_config_creation()"
```

### Train
```bash
# MLM (default task)
python train_mrbert.py --task mlm --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
    --output_dir ./mrbert_checkpoints --num_epochs 3 --target_deletion_rate 0.3 --deletion_loss_weight 0.1

# Sequence classification (SST-2)
python train_mrbert.py --task sequence_classification --dataset_name glue --dataset_config sst2 \
    --output_dir ./mrbert_sst2 --num_epochs 3

# Token classification (CoNLL-2003 NER)
python train_mrbert.py --task token_classification --dataset_name conll2003 --output_dir ./mrbert_ner

# Question answering (SQuAD)
python train_mrbert.py --task question_answering --dataset_name squad --output_dir ./mrbert_squad

# Quick smoke test (100 steps)
python train_mrbert.py --max_steps 100 --logging_steps 10
```

### Evaluate
```bash
python eval_mrbert.py --model_path ./mrbert_checkpoints/final
python eval_mrbert.py --from_pretrained bert-base-uncased  # fresh model, no training
```

### Compare MrBERT vs baseline BERT
```bash
python run_comparison.py --task mlm
python run_comparison.py --task mlm --train --num_epochs 10
python run_baseline.py --task mlm
```

### Serverless GPU training (Modal)
```bash
python3 -m modal setup
modal run train_modal.py                   # quick test (20 steps)
modal run train_modal.py --max-steps 500   # longer run
modal volume get mrbert-checkpoints final ./local_mrbert_final  # download checkpoints
```

## Architecture

### Core files
- **`configuration_mrbert.py`** — `MrBertConfig` extends `BertConfig` with delete gate parameters
- **`modeling_mrbert.py`** — Full MrBERT implementation (1,428 lines); all task heads
- **`modeling_bert.py`** — Reference BERT implementation used for comparison

### Delete gate mechanism (in `modeling_mrbert.py`)
After layer `delete_gate_layer` (default: 3), each token's hidden state is passed through `LayerNorm + Linear → scaled sigmoid` to produce a gate value in [0, 1]:
- **Soft deletion** (default): gate value is added as a large negative bias to attention scores in all subsequent layers — token remains in memory but is ignored
- **Hard deletion**: tokens where gate < `deletion_threshold` are physically removed from the sequence; [CLS] and [SEP] are protected; pad tokens are always deleted

### Gate variants (`deletion_type` config field)
| Type | Class | Description |
|---|---|---|
| `scaled_sigmoid` | `SigmoidDeleteGate` | Main learnable gate; sigmoid scaled by `sigmoid_mask_scale` (default -30.0) |
| `log_sigmoid` | `LogSigmoidDeleteGate` | Alternative learnable gate using log sigmoid |
| `random` | `RandomDeleteGate` | Baseline: delete random tokens at `random_deletion_probability` rate |
| `fixed` | `FixedDeleteGate` | Baseline: delete a fixed fraction `fixed_deletion_amount` of tokens |

### Key `MrBertConfig` parameters
- `delete_gate_layer` (default: 3) — which encoder layer emits the gate
- `deletion_type` (default: `"scaled_sigmoid"`)
- `sigmoid_mask_scale` (default: -30.0) — controls strength of the deletion signal
- `deletion_threshold` (default: 0.5) — threshold for hard deletion
- `use_gumbel_noise` (default: False) — add Gumbel noise during training for exploration
- `random_deletion_probability` / `fixed_deletion_amount` — for non-learnable baselines

### Training details (`train_mrbert.py`)
- Auxiliary **deletion loss** encourages the gate to hit `--target_deletion_rate` (fraction of tokens to delete)
- `--deletion_loss_weight` controls the trade-off between task loss and deletion loss
- `--delete_gate_lr` allows a higher learning rate for gate parameters vs. the base BERT weights

### Task heads
`MrBertForMaskedLM`, `MrBertForSequenceClassification`, `MrBertForTokenClassification`, `MrBertForQuestionAnswering`, `MrBertForMultipleChoice`, `MrBertForNextSentencePrediction` — all in `modeling_mrbert.py`

### Reference implementation
`mrt5/` contains the original MrT5 code (T5-based). Its `models/modeling_mrt5.py` is the primary reference for the delete gate design.
