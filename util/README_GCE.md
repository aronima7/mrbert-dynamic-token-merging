# MrBERT Training on GCE A100

This guide covers running MrBERT and BERT baseline training on a Google Compute Engine VM with an A100 GPU. Unlike Vertex AI, GCE gives you direct SSH access and full control over the environment.

---

## Prerequisites

- GCP project with billing enabled and A100 quota in your target zone
- `gcloud` CLI installed and authenticated
- `gsutil` available (included with `gcloud`)

```bash
gcloud auth login
gcloud auth application-default login
```

Install local Python dependencies:

```bash
pip install google-cloud-compute google-cloud-storage
```

---

## Configuration

All GCP settings are passed as CLI arguments.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--project` | Yes | — | GCP project ID |
| `--bucket` | Yes (training) | — | GCS bucket name for checkpoints (without `gs://`) |
| `--zone` | No | `us-central1-c` | GCP zone with A100 quota |
| `--vm-name` | No | `mrbert-training` | VM instance name |

---

## One-Time Setup

### 1. Create the GCS bucket

```bash
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME
```

### 2. Authenticate Docker (only needed if using Artifact Registry separately)

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## Training

The script handles the full lifecycle automatically:
1. Creates the VM (or starts it if stopped)
2. Uploads the mrbert code via `gcloud scp`
3. Installs Python dependencies
4. Preprocesses SNLI (or downloads from GCS if already done)
5. Runs training and streams logs to your terminal
6. Uploads checkpoints to GCS
7. Stops or deletes the VM

### MrBERT (with delete gate)

```bash
cd /path/to/mrbert/training
python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000
```

### BERT baseline (no delete gate)

```bash
python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type BERT --max-steps 30000
```

### Delete VM after training (avoids idle billing)

```bash
python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000 --delete-vm
```

### Custom run name

```bash
python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000 --run-name mrbert-exp1
```

Checkpoints will be saved to `gs://YOUR_BUCKET/checkpoints/mrbert-exp1/`.

### Extra training args

Any flags after the named args are forwarded to `train_mrbert.py`:

```bash
python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000 -- --batch_size 64 --learning_rate 1e-5
```

### Second run (skip upload and setup)

If the VM already has the code and dependencies installed:

```bash
python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type BERT --max-steps 30000 --skip-upload --skip-setup
```

---

## SSH Access

Open an interactive shell on the VM at any time:

```bash
python train_gce.py --project YOUR_PROJECT --ssh
```

Or directly with gcloud:

```bash
gcloud compute ssh mrbert-training --zone=us-central1-c --project=YOUR_PROJECT
```

Once on the VM, monitor training:

```bash
# Check GPU utilization
nvidia-smi

# Watch training logs (if running in background)
tail -f ~/mrbert/training.log
```

---

## VM Lifecycle

| Command | Effect |
|---|---|
| `--stop-vm` | Stop the VM after training — billing ceases, disk is preserved |
| `--delete-vm` | Delete the VM after training — billing ceases, disk is destroyed |
| `--delete-vm-only` | Delete the VM immediately without training |

**Stop VM manually:**
```bash
python train_gce.py --project YOUR_PROJECT --stop-vm
```

**Delete VM manually:**
```bash
python train_gce.py --project YOUR_PROJECT --delete-vm-only
```

Or via gcloud directly:
```bash
gcloud compute instances stop mrbert-training --zone=us-central1-c --project=YOUR_PROJECT
gcloud compute instances delete mrbert-training --zone=us-central1-c --project=YOUR_PROJECT
```

---

## Downloading Checkpoints

```bash
# Download a specific run
gsutil -m cp -r gs://YOUR_BUCKET/checkpoints/mrbert-mrbert-30000steps ./local_mrbert_checkpoints

# List all checkpoint runs
gsutil ls gs://YOUR_BUCKET/checkpoints/
```

---

## Monitoring

**GPU utilization (on the VM):**
```bash
nvidia-smi
watch -n 1 nvidia-smi   # refresh every second
```

**Weights & Biases:**

Pass your W&B API key and metrics will stream to your wandb dashboard:

```bash
export WANDB_API_KEY=your_key_here
python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000
```

---

## Hardware

| Spec | Value |
|---|---|
| Machine type | `a2-highgpu-1g` |
| GPU | 1× NVIDIA A100 40GB |
| vCPUs | 12 |
| RAM | 85 GB |
| Boot disk | 200 GB |
| OS image | `pytorch-latest-gpu` (CUDA + PyTorch pre-installed) |

At ~3.7 it/s, 30,000 steps takes roughly 2.5 hours.

---

## Default Training Hyperparameters

| Parameter | Value |
|---|---|
| Task | SNLI sequence classification |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Max seq length | 128 |
| Target deletion rate | 30% |
| Regularizer delay | 1000 steps |
| PI controller k_p | 0.01 |
| PI controller k_i | 0.00001 |
| Save steps | 1000 |
| Logging steps | 50 |

---

## Comparison: GCE vs Vertex AI

| | GCE VM | Vertex AI |
|---|---|---|
| SSH access | Yes | No |
| Log streaming | Live in terminal | Via console / `gcloud ai` |
| Auto-stop on completion | No (script handles it) | Yes |
| Startup time | ~2 min | ~5 min |
| Cost control | Manual stop required | Auto-terminates |
| Debugging | Full shell access | Limited |

---

## File Structure

```
mrbert/training/
├── train_gce.py        # GCE VM training script (this file's companion)
├── train_gcp.py        # Vertex AI training script
├── train_modal.py      # Modal serverless GPU training script
├── train_mrbert.py     # Main training loop (used by all three)
├── README_GCE.md       # This file
└── README_GCP.md       # Vertex AI instructions
```