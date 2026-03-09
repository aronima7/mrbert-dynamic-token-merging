# MrBERT Training on GCP Vertex AI

This guide covers running MrBERT and BERT baseline training on GCP Vertex AI using A100 GPUs.

---

## Prerequisites

- GCP project with billing enabled and A100 quota
- Vertex AI API enabled
- Artifact Registry API enabled
- `gcloud` CLI installed and authenticated

```bash
gcloud auth login
gcloud auth configure-docker <REGION>-docker.pkg.dev
```

Install local Python dependencies:

```bash
pip install google-cloud-aiplatform google-cloud-storage
```

---

## Configuration

All GCP settings are passed as CLI arguments — no need to edit the script.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--project` | Yes | — | GCP project ID |
| `--region` | No | `us-central1` | GCP region with A100 quota |
| `--bucket` | Yes | — | GCS bucket name (without `gs://`) |

---

## One-Time Setup

### 1. Create the GCS bucket

```bash
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME
```

### 2. Build and push the Docker image

```bash
cd /path/to/mrbert/training
python train_gcp.py --project YOUR_PROJECT --region us-central1 --bucket YOUR_BUCKET --build-image
```

This builds from `training/Dockerfile`, creates an Artifact Registry repo named `mrbert`, and pushes the image. Only needs to be re-run when dependencies or code changes.

### 3. Preprocess SNLI to GCS (one-time)

```bash
python train_gcp.py --project YOUR_PROJECT --region us-central1 --bucket YOUR_BUCKET --preprocess-snli
```

This submits a short Vertex AI CPU job that downloads SNLI, tokenizes it, and saves the output to `gs://YOUR_BUCKET/snli_datasets/`. Subsequent training jobs skip this step automatically.

---

## Training

### MrBERT (with delete gate)

```bash
python train_gcp.py --project YOUR_PROJECT --region us-central1 --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000
```

### BERT baseline (no delete gate)

```bash
python train_gcp.py --project YOUR_PROJECT --region us-central1 --bucket YOUR_BUCKET --model-type BERT --max-steps 30000
```

Both jobs run detached — the command returns immediately and training continues on Vertex AI.

### Custom run name

```bash
python train_gcp.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000 --run-name mrbert-exp1
```

Checkpoints will be saved to `gs://YOUR_BUCKET/checkpoints/mrbert-exp1/`.

### Extra training args

Any flags after a `--` separator are forwarded to `train_mrbert.py`:

```bash
python train_gcp.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000 -- --batch_size 64 --learning_rate 1e-5
```

---

## Monitoring

**Vertex AI console:**
```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=YOUR_PROJECT
```

**Weights & Biases:**

Pass your W&B API key via environment variable and it will be forwarded to the training container:

```bash
export WANDB_API_KEY=your_key_here
python train_gcp.py --model-type MrBERT --max-steps 30000
```

Or pass it explicitly:
```bash
python train_gcp.py --model-type MrBERT --max-steps 30000 --wandb-api-key YOUR_KEY
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

## Hardware

Jobs run on `a2-highgpu-1g` (1× A100 40GB). At ~3.7 it/s, 30,000 steps takes roughly 2.5 hours.

---

## File Structure

```
mrbert/training/
├── train_gcp.py        # GCP Vertex AI job submission script
├── train_modal.py      # Modal serverless GPU training script
├── train_mrbert.py     # Main training loop (used by both)
├── Dockerfile          # Container image for GCP
└── requirements.txt    # Python dependencies for the container
```