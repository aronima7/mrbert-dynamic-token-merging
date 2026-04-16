# SEDD on Modal

## Prerequisites

```bash
pip install modal
modal token set        # one-time auth
```

## W&B Setup (one-time)

Training logs `train/loss`, `eval/loss`, and `eval/perplexity` to W&B.
Create a Modal secret with your W&B API key before running training:

```bash
# Get your API key from https://wandb.ai/authorize
modal secret create wandb-secret WANDB_API_KEY=your_key_here

# Verify
modal secret list
```

## Usage

```bash
cd diffusion/Score-Entropy-Discrete-Diffusion

# Quick smoke test (2 samples, pretrained small model, 50 steps)
modal run train_modal.py::main

# Train (small model, absorb+loglinear, single A100, default 1.3M iters)
modal run --detach train_modal.py::train_main --batch-size 8 --accum 4 --eval-batch-size 16
# Smaller training run (~1 hr, ~$4)
modal run --detach train_modal.py::train_main --n-iters 20000 --batch-size 8 --accum 4 --eval-batch-size 16

# Train with custom settings
modal run --detach train_modal.py::train_main \
  --run-name my-sedd --model medium --batch-size 8 --accum 4 --eval-batch-size 16 --n-iters 200000

# Train with a custom W&B project name
modal run --detach train_modal.py::train_main \
  --run-name my-sedd --wandb-project my-project --batch-size 8 --accum 4 --eval-batch-size 16

# Resume an interrupted run
modal run --detach train_modal.py::train_main \
  --run-name my-sedd --resume --batch-size 8 --accum 4 --eval-batch-size 16

# Continued pretraining from a pretrained HuggingFace model
# --batch-size 8 --accum 4: avoids OOM during score_entropy on A100-40GB (effective batch = 32)
# --eval-batch-size 16: avoids OOM during eval (config default 512 is too large at seq_len=1024)
modal run --detach train_modal.py::train_main \
  --pretrained-from louaaron/sedd-small \
  --run-name sedd-continued \
  --n-iters 20000 \
  --batch-size 8 --accum 4 \
  --eval-batch-size 16

# Resume an interrupted continued pretraining run
modal run --detach train_modal.py::train_main \
  --pretrained-from louaaron/sedd-small \
  --run-name sedd-continued \
  --n-iters 20000 \
  --batch-size 8 --accum 4 \
  --eval-batch-size 16 \
  --resume

# Sample from pretrained model
modal run train_modal.py::sample_main --model-path louaaron/sedd-medium --steps 256 --batch-size 4


# Sample from locally trained run
modal run train_modal.py::sample_main --model-path my-sedd

# Conditional sampling
modal run train_modal.py::sample_cond_main --prefix "Once upon a time" --suffix "The End."

# Download checkpoints
modal volume get sedd-checkpoints exp_local/my-sedd ./my-sedd
```

## Multi-GPU training (8× A100-40GB)

Uses `train_8gpu_main` — matches the original paper's setup exactly. Each GPU sees 64 samples,
so `batch_size=512` and `eval_batch_size=512` both fit without OOM. ~8× faster wall-clock than
single-GPU at the same cost per step.

```bash
# Sanity check (~8 min, ~$4)
modal run --detach train_modal.py::train_8gpu_main \
  --n-iters 20000 \
  --run-name sedd-8gpu-test

# Continued pretraining from pretrained weights (~1.25 hrs, ~$40)
modal run --detach train_modal.py::train_8gpu_main \
  --pretrained-from louaaron/sedd-small \
  --run-name sedd-continued-8gpu \
  --n-iters 200000
# single gpu (--resume to resume from data checkpoint)
modal run --detach train_modal.py::train_main --pretrained-from louaaron/sedd-small --run-name sedd-continued --n-iters 20000 --eval-batch-size 16 --batch-size 8 --accum 4  --resume

# Full training from scratch (~14 hrs, ~$450)
modal run --detach train_modal.py::train_8gpu_main --run-name sedd-scratch-8gpu
```

| Setup | GPUs | batch_size | eval_batch_size | Time (200k steps) | Cost |
|---|---|---|---|---|---|
| `train_main` | 1× A100-40GB | 8 (accum 4) | 16 | ~10 hrs | ~$40 |
| `train_8gpu_main` | 8× A100-40GB | 512 | 512 | ~1.25 hrs | ~$40 |

Same cost, 8× faster. Use `train_8gpu_main` for any run longer than a quick smoke test.

---



The SEDD small model at `seq_len=1024` creates large intermediate tensors during `score_entropy`
(up to `[B×L, vocab_size]` for the masked positions). On a 40 GB A100 the default `batch_size=32`
OOMs during the training step.

**Required settings for A100-40GB:**

| Parameter | Value | Notes |
|---|---|---|
| `--batch-size` | `8` | Per-step batch; 4× smaller peak tensor |
| `--accum` | `4` | Gradient accumulation → effective batch 32 |
| `--eval-batch-size` | `16` | Eval logits tensor; config default 512 OOMs |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically in `train_modal.py` to
reduce allocator fragmentation.

---



| Metric | Logged at | Description |
|---|---|---|
| `train/loss` | every 50 steps | Score entropy loss on the training batch (averaged across GPUs) |
| `eval/loss` | every 100 steps | Score entropy loss on a held-out WikiText-103 batch |
| `eval/perplexity` | every 50,000 steps | GPT-2-Large perplexity of generated samples (lower = better) |

The run name in W&B matches `--run-name` and the project defaults to `sedd`.

## Training outputs (saved to volume)

Each run saves to `sedd-checkpoints/exp_local/<run-name>/`:

```
<run-name>/
├── .hydra/
│   └── config.yaml          # full Hydra config for the run
├── checkpoints/
│   └── checkpoint_N.pth     # full checkpoint every 50,000 steps
├── checkpoints-meta/
│   └── checkpoint.pth       # latest meta-checkpoint (every 10,000 steps, for resuming)
├── samples/
│   └── iter_N/
│       └── sample_0.txt     # generated text samples at each snapshot
└── logs                     # plain-text training log
```

Download any of these with:
```bash
modal volume get sedd-checkpoints exp_local/<run-name> ./<run-name>
```

## Inference outputs

Sampling (`sample_main`, `sample_cond_main`) prints generated text to stdout, one sample per line separated by `=====...=====`. Output is visible in the modal run terminal.

To capture to a file locally:
```bash
modal run train_modal.py::sample_main --model-path louaaron/sedd-medium --steps 256 --batch-size 4 > samples.txt
```

## Continued pretraining vs training from scratch

**Training from scratch** (`modal run --detach train_modal.py::train_main`) initializes all weights randomly and trains on OpenWebText for 1.3M steps (~$150–$300, multiple resumed runs).

**Continued pretraining** (`--pretrained-from louaaron/sedd-small`) starts from published weights that already produce fluent text, and continues training from there. This is the recommended approach when adapting SEDD (e.g. adding a delete gate) rather than reproducing the original paper.

### What happens under the hood

```
1. init_from_pretrained.py downloads louaaron/sedd-small from HuggingFace
   └── saves checkpoints-meta/checkpoint.pth with:
         model:     pretrained weights
         ema:       shadow params = pretrained weights (not random)
         optimizer: fresh AdamW state (step 0, no accumulated momentum)
         step:      0

2. train.py starts, restore_checkpoint() loads that file
   └── training begins from pretrained weights at step 0

3. On --resume, the init step is skipped (checkpoint already exists)
   └── training picks up from wherever it was interrupted
```

### Recommended continued pretraining settings

Use `train_8gpu_main` for all runs beyond a quick smoke test — same cost as single-GPU, 8× faster.

```bash
# Sanity check: single GPU (~1 hr, ~$4) — cheapest way to verify W&B is logging
modal run --detach train_modal.py::train_main \
  --pretrained-from louaaron/sedd-small \
  --run-name sedd-continued-test \
  --n-iters 20000 \
  --batch-size 8 --accum 4 \
  --eval-batch-size 16 \
  --wandb-project sedd

# Meaningful run: 8 GPUs (~1.25 hrs, ~$40) — use this for all real experiments
modal run --detach train_modal.py::train_8gpu_main \
  --pretrained-from louaaron/sedd-small \
  --run-name sedd-continued-8gpu \
  --n-iters 200000 \
  --wandb-project sedd

# Longer 8-GPU run (~3 hrs, ~$100)
modal run --detach train_modal.py::train_8gpu_main \
  --pretrained-from louaaron/sedd-small \
  --run-name sedd-continued-8gpu-500k \
  --n-iters 500000 \
  --wandb-project sedd

# Medium model, 8 GPUs
modal run --detach train_modal.py::train_8gpu_main \
  --pretrained-from louaaron/sedd-medium \
  --run-name sedd-medium-continued-8gpu \
  --model medium \
  --n-iters 200000
```

### Choosing `--n-iters` for continued pretraining

| Goal | Command | `--n-iters` | Time | Cost |
|---|---|---|---|---|
| Smoke test (verify W&B, loss drops) | `train_main` | 20,000 | ~1 hr | ~$4 |
| Meaningful experiment | `train_8gpu_main` | 200,000 | ~1.25 hrs | ~$40 |
| Longer experiment | `train_8gpu_main` | 500,000 | ~3 hrs | ~$100 |
| Full retraining from scratch | `train_8gpu_main` | 1,300,000 | ~8 hrs | ~$250 |

-------
Note:
* --resume loads checkpoints-meta/checkpoint.pth (the pretrained init saved at step 0), and --eval-batch-size 32 prevents the OOM that killed the run at the first eval step.
-------