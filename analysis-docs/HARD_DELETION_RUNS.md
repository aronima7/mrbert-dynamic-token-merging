# Hard Deletion Training Runs

This document describes the hard deletion training runs for MrBERT and provides guidance on launching and monitoring these experiments.

## Overview

**Hard deletion** is a token removal strategy where tokens with low gate values (below a threshold) are physically removed from the sequence, reducing both memory and computation in subsequent layers. This contrasts with **soft deletion** (the default), where low-scoring tokens remain in memory but are masked out via attention bias.

For hard deletion to work effectively at inference time, the model must be trained with hard deletion enabled. The MrT5 paper recommends training with a mixture of hard and soft deletion (e.g., `hard_delete_train_prob=0.5`), meaning 50% of training steps use hard deletion and 50% use soft deletion.

## Hard Deletion Training Configuration

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hard_delete_train_prob` | 0.5 | Probability of using hard deletion on each training step |
| `target_deletion_rate` | 0.3 | Target fraction of tokens to delete (30%) |
| `deletion_type` | "scaled_sigmoid" | Learnable gate with scaled sigmoid activation |
| `delete_gate_layer` | 3 | Gate applied after layer 3 (0-indexed) |
| `mode` | "training-and-eval" | Train model and evaluate on test set |

### Training Details

- **PI Controller**: Dynamically adjusts deletion loss weight `α` to maintain target deletion rate
  - Proportional gain: `k_p = 0.01`
  - Integral gain: `k_i = 0.00001`
- **Gumbel Noise**: Enabled during training for exploration
- **Softmax1**: Uses softmax1 attention normalization (recommended by MrT5 paper)

## Datasets

All runs will be executed on the following datasets:

### 1. SNLI (Natural Language Inference)
- **Task**: Sequence classification
- **Max steps**: 30,000
- **Batch size**: 32
- **Sequence length**: 128
- **Regularizer delay**: 1,000 steps

### 2. SQuAD (Question Answering)
- **Task**: Question answering
- **Max steps**: 30,000
- **Batch size**: 16
- **Sequence length**: 384
- **Regularizer delay**: 1,000 steps

### 3. SST-2 (Sentiment Analysis)
- **Task**: Sequence classification
- **Max steps**: 10,000
- **Batch size**: 32
- **Sequence length**: 128
- **Regularizer delay**: 300 steps

### 4. MRPC (Paraphrase Detection)
- **Task**: Sequence classification
- **Max steps**: 5,000
- **Batch size**: 32
- **Sequence length**: 128
- **Regularizer delay**: 100 steps

### 5. IMDB (Sentiment Analysis - Long)
- **Task**: Sequence classification
- **Max steps**: 20,000
- **Batch size**: 16
- **Sequence length**: 512
- **Regularizer delay**: 300 steps

### 6. TyDiQA (Multilingual Question Answering)
- **Task**: Question answering
- **Max steps**: 30,000
- **Batch size**: 16
- **Sequence length**: 384
- **Regularizer delay**: 1,000 steps

## Launching Runs

### Prerequisites

```bash
pip install modal
modal token set  # one-time authentication
```

### Launch All Runs

```bash
python launch_hard_deletion_runs.py --all
```

This will launch 6 runs in parallel (detached mode) on Modal with A100 GPUs.

### Launch Individual Dataset

```bash
# SNLI
python launch_hard_deletion_runs.py --dataset snli

# SQuAD
python launch_hard_deletion_runs.py --dataset squad

# SST-2
python launch_hard_deletion_runs.py --dataset sst2

# MRPC
python launch_hard_deletion_runs.py --dataset mrpc

# IMDB
python launch_hard_deletion_runs.py --dataset imdb

# TyDiQA
python launch_hard_deletion_runs.py --dataset tydiqa
```

### Test Mode

For quick validation (100 steps only):

```bash
python launch_hard_deletion_runs.py --dataset snli --test
```

### Dry Run

Preview commands without executing:

```bash
python launch_hard_deletion_runs.py --all --dry-run
```

### List Available Datasets

```bash
python launch_hard_deletion_runs.py --list
```

## Monitoring

### Weights & Biases

All runs will be logged to W&B:
- **Project**: `mrbert-hard-deletion`
- **Run names**: `hard-del-{dataset}-0.3`
- **URL**: https://wandb.ai

Key metrics to monitor:
- `train/loss`: Combined task + deletion loss
- `train/task_loss`: Primary task loss (cross-entropy)
- `train/deletion_loss`: Gate regularization loss
- `train/deletion_rate`: Actual deletion rate (should converge to 30%)
- `train/alpha`: PI controller coefficient
- `eval/accuracy` or `eval/f1`: Validation metrics
- `test/accuracy` or `test/f1`: **Final test set performance**

### Modal Dashboard

Check run status:
```bash
modal app list
modal run logs <app-id>
```

## Expected Results

### Test Set Evaluation

After training completes, each run will automatically evaluate on the **test set** with:
1. **Soft deletion** (default): Tokens masked via attention bias
2. **Hard deletion**: Tokens physically removed

The test results will include:
- **Accuracy/F1**: Task performance
- **Deletion rate**: Actual percentage of tokens deleted
- **Speedup**: Inference time improvement (if measured)

### Performance Trade-offs

Based on the MrT5 paper, we expect:
- **Target deletion rate**: 30% of tokens
- **Performance retention**: 95-98% of baseline accuracy
- **Speedup**: ~1.2-1.4x (depending on implementation efficiency)

Different datasets may show varying sensitivity to deletion:
- **NLI tasks** (SNLI): Usually robust to 30% deletion
- **QA tasks** (SQuAD, TyDiQA): May be more sensitive if answer spans are deleted
- **Sentiment** (SST-2, IMDB): Typically robust
- **Paraphrase** (MRPC): Small dataset, may have higher variance

## Downloading Checkpoints

After runs complete:

```bash
# List available checkpoints
modal volume ls mrbert-checkpoints

# Download specific run
modal volume get mrbert-checkpoints hard-del-snli-0.3 ./local_checkpoints/snli

# Download all runs
for dataset in snli squad sst2 mrpc imdb tydiqa; do
    modal volume get mrbert-checkpoints hard-del-${dataset}-0.3 ./local_checkpoints/${dataset}
done
```

## Local Evaluation

To run additional evaluation locally:

```bash
python mrbert/training/train_mrbert.py \
    --model_type MrBERT \
    --model_name ./local_checkpoints/snli/final \
    --mode eval-only \
    --task sequence_classification \
    --dataset_name local_snli \
    --local_snli_dir ./snli_datasets
```

Add `--hard_delete_train_prob 1.0` to force hard deletion during evaluation.

## Troubleshooting

### Run Failed

Check logs:
```bash
modal run logs <run-id>
```

Common issues:
- **Out of memory**: Reduce `batch_size` in `DATASET_CONFIGS`
- **Dataset preprocessing failed**: Check that preprocessing scripts work locally
- **Modal timeout**: Increase timeout in `train_modal.py` if needed

### Poor Performance

If test accuracy is significantly degraded:
1. Check deletion rate converged to target (30%)
2. Verify PI controller is working (α should stabilize)
3. Try lower deletion rate (e.g., 20%)
4. Increase training steps
5. Adjust `regularizer_delay` (too early may hurt learning)

### Hard Deletion Worse Than Soft

This indicates the model wasn't properly trained for hard deletion:
- Verify `hard_delete_train_prob > 0` during training
- Try increasing `hard_delete_train_prob` to 0.7-0.8
- Ensure sufficient training steps after gate starts working

## References

- **MrT5 Paper**: [Kallini et al., 2024 - Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/pdf/2410.20771)
- **MrBERT Architecture**: `mrbert/models/modeling_mrbert.py`
- **Training Script**: `mrbert/training/train_mrbert.py`
- **Modal Launcher**: `mrbert/training/train_modal.py`
