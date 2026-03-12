# MrBERT Training on Modal

This guide covers running MrBERT and BERT baseline training on Modal serverless GPUs, including setup, running jobs, monitoring, debugging, and checkpoint management.

---

## Prerequisites

```bash
pip install modal
```

Authenticate (one-time):

```bash
modal setup
```

---

## Weights & Biases Setup

Training logs metrics to W&B. Create a Modal secret with your API key (one-time):

```bash
modal secret create wandb-secret WANDB_API_KEY=your_key_here
```

Verify it exists:

```bash
modal secret list
```

To disable W&B for a run, pass `--disable-wandb` via `extra_args` (see below).

---

## Running Training

All commands are run from `mrbert/training/`:

```bash
cd /path/to/mrbert/training
```

### Quick sanity test (20 steps, attached)

```bash
modal run train_modal.py
```

### MrBERT full training run (detached)

```bash
modal run --detach train_modal.py --model-type MrBERT --max-steps 30000
```

### BERT baseline full training run (detached)

```bash
modal run --detach train_modal.py --model-type BERT --max-steps 30000
```

`--detach` returns immediately and the job continues in the background. Without `--detach`, the terminal stays attached and streams logs live.

---

## Monitoring

### Stream logs from a running job

```bash
modal app logs mrbert-train
```

### List running containers

```bash
modal container list
```

### SSH into a running container

```bash
modal container exec <CONTAINER_ID> /bin/bash
```

Get the container ID from `modal container list`. Once inside:

```bash
nvidia-smi                        # GPU utilization
ps aux | grep python              # check training process
ls /checkpoints                   # inspect volume contents
```

### Monitor via W&B

Live metrics are available at:
```
https://wandb.ai/aronima7-stanford-university/mrbert
```

Key metrics to watch during MrBERT training:

| Metric | What to expect |
|---|---|
| `accuracy` | Should increase steadily |
| `loss` | Should decrease |
| `percent_deleted_tokens` | Should stay near 0 until step 1000, then gradually rise toward 30% |
| `delete_gate_loss_coeff` (α) | Should rise slowly after step 1000 |
| `delete_gate_average` | Should stay near 0 for first 1000 steps, then drift negative slowly |
| `delete_gate_min_value` | Warning sign if this hits -30 — gate collapse |

---

## Stopping a Run

Stop the entire app (all running containers):

```bash
modal app stop mrbert-train
```

Stop a specific container:

```bash
modal container list              # get container ID
modal app stop <APP_ID>           # from modal app list
```

Or stop from the Modal dashboard:
```
https://modal.com/apps
```

---

## Checkpoints

Checkpoints are saved to the Modal Volume `mrbert-checkpoints` at `/checkpoints/` during training (every 1000 steps by default).

### List volume contents

```bash
modal volume ls mrbert-checkpoints
modal volume ls mrbert-checkpoints/checkpoints
```

### Download checkpoints locally

```bash
# Download the final model
modal volume get mrbert-checkpoints checkpoints/final ./local_mrbert_final

# Download a specific step checkpoint
modal volume get mrbert-checkpoints checkpoints/checkpoint-5000 ./local_checkpoint_5000

# Download everything
modal volume get mrbert-checkpoints checkpoints ./local_mrbert_checkpoints
```

### Checkpoint persistence

`volume.commit()` is called after every run (even on failure) to guarantee writes are persisted. Checkpoints written mid-training via `model.save_pretrained()` are also preserved if the job is cancelled, as long as the write completed before cancellation.

---

## Debugging

### Check if the image built correctly

```bash
modal run train_modal.py --max-steps 5
```

A short attached run lets you see build logs and early training output.

### Inspect the container environment

```bash
modal container exec <CONTAINER_ID> /bin/bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
ls /workspace           # verify code was uploaded correctly
ls /checkpoints         # verify volume is mounted
```

### Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `No module named 'configuration_mrbert'` | `PYTHONPATH` not set | Already handled via `sys.path` insert in `train_mrbert.py` |
| `All tokens deleted` | Delete gate collapsed | Reduce `--controller_p` or increase `--regularizer_delay` |
| `ModuleNotFoundError: transformers` | Wrong transformers version | Image uses `transformers>=4.40,<5.0` — don't upgrade to 5.x |
| `File modified during build` | Local checkpoint dirs included in image | Already excluded via `ignore` list in `add_local_dir` |
| Job timed out | Training exceeded timeout | Timeout is set to 24 hours — should not occur for 30k steps |
| `wandb-secret` not found | Secret not created | Run `modal secret create wandb-secret WANDB_API_KEY=...` |

### Force image rebuild

Modal caches the image. If you update `requirements.txt` or the `Dockerfile`, force a rebuild by changing any `pip_install` arg in `train_modal.py` or running:

```bash
modal run train_modal.py --max-steps 1
```

The image rebuilds whenever the pip dependencies or local dir contents change.

---

## Default Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Task | SNLI sequence classification | 3-class NLI |
| Batch size | 32 | |
| Learning rate | 2e-5 | Standard BERT fine-tuning LR |
| Max seq length | 128 | Premise + hypothesis |
| Target deletion rate | 30% | MrBERT only |
| Regularizer delay | 1000 steps | Learn task before deleting |
| PI controller k_p | 0.01 | Low to prevent gate collapse |
| PI controller k_i | 0.00001 | Slow integral ramp |
| Initial deletion loss weight (α) | 0.01 | |
| Save steps | 1000 | |
| Logging steps | 50 | |
| Timeout | 24 hours | |
| GPU | A100 40GB | |

---

## Hardware

Modal runs on an **NVIDIA A100 SXM4 40GB**:

```
GPU  Name                 Persistence-M
  0  NVIDIA A100-SXM4-40GB          On
     5233MiB / 40960MiB   93% GPU-Util
```

At ~3.7 it/s, 30,000 steps takes roughly **2.5 hours**.

---

## File Structure

```
mrbert/training/
├── train_modal.py      # Modal job definition and entrypoint
├── train_mrbert.py     # Main training loop
├── train_gcp.py        # Vertex AI alternative
├── train_gce.py        # GCE VM alternative
├── README_modal.md     # This file
├── README_GCP.md       # Vertex AI instructions
└── README_GCE.md       # GCE VM instructions
```

---

## Comparison: Modal vs GCE vs Vertex AI

| | Modal | GCE VM | Vertex AI |
|---|---|---|---|
| SSH access | Via `container exec` | Full SSH | No |
| Setup complexity | Minimal | Medium | Medium |
| Auto-stop | Yes | Script handles it | Yes |
| Startup time | ~2 min | ~2 min | ~5 min |
| Persistent storage | Modal Volume | GCS / local disk | GCS |
| Cost model | Per second | Per second (manual stop) | Per second |