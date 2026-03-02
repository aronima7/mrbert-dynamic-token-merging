# MrBERT: BERT with MrT5-Style Delete Gates

**Author:** Hiva Mohammadzadeh  
**Course:** CS224N - Natural Language Processing with Deep Learning

This repository contains an implementation of **MrBERT**, which adapts the delete gate mechanism from the [MrT5 paper](https://arxiv.org/pdf/2410.20771) to the BERT architecture. The delete gate learns to selectively remove uninformative tokens during encoding, potentially improving computational efficiency while maintaining (or improving) task performance.

---

## Overview

The MrT5 paper introduced a **delete gate** mechanism that learns to remove redundant tokens during encoding. This can:
- **Reduce computational cost** by processing fewer tokens in later layers
- **Improve model focus** by removing uninformative tokens like punctuation or filler words
- **Potentially improve performance** by reducing noise in the representation

MrBERT adapts this idea to the encoder-only BERT architecture, enabling its use for tasks like:
- Masked Language Modeling (MLM)
- Text Classification
- Named Entity Recognition (NER)
- Question Answering
- And more...

### Key Features

- ✅ **Soft Deletion**: Masks attention scores (tokens still present, but ignored)
- ✅ **Hard Deletion**: Physically removes tokens from the sequence
- ✅ **Multiple Delete Gate Types**: Scaled sigmoid, log sigmoid, random, and fixed
- ✅ **Configurable Gate Placement**: Place the delete gate at any encoder layer
- ✅ **Gumbel Noise**: Optional exploration during training
- ✅ **Deletion Loss**: Auxiliary loss to encourage a target deletion rate
- ✅ **Full Task Support**: MLM, classification, token classification, QA, etc.

---

## Architecture

```
Input: [CLS] The quick brown fox [SEP]
         ↓
┌──────────────────────────────────────┐
│         BERT Embedding Layer          │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│         Encoder Layer 0               │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│         Encoder Layer 1               │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   ★ DELETE GATE (default: Layer 2)   │  ← Learns which tokens to delete
│   ┌────────────────────────────────┐ │
│   │ LayerNorm → Linear → Sigmoid   │ │
│   └────────────────────────────────┘ │
│   Output: delete_mask per token      │
└──────────────────────────────────────┘
         ↓
   (Soft: mask attention | Hard: remove tokens)
         ↓
┌──────────────────────────────────────┐
│      Encoder Layers 2-11              │
│      (with delete mask applied)       │
└──────────────────────────────────────┘
         ↓
       Output
```
```

- **Soft Deletion**: `delete_gate_value` is added to attention scores (masking effect)
- **Hard Deletion**: Tokens with `delete_gate_value > threshold` are physically removed

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.39+

### Setup

```bash
# Clone the repository
git clone https://github.com/HivaMohammadzadeh1/CS224N-project.git
cd CS224N-project

# Create conda environment
conda create -n mrbert python=3.11 -y
conda activate mrbert

# Install dependencies
pip install modal torch transformers datasets accelerate tqdm matplotlib "numpy<2"

# For the original mrt5 code
cd mrt5
pip install -r requirements.txt
pip install transformers==4.39.1
```

---

## Configuration

### MrBertConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delete_gate_layer` | int | 2 | Which encoder layer to place the delete gate (0-indexed) |
| `deletion_type` | str | "scaled_sigmoid" | Type of delete gate: `"scaled_sigmoid"`, `"log_sigmoid"`, `"random"`, `"fixed"` |
| `sigmoid_mask_scale` | float | -10.0 | Scale for sigmoid activation (more negative = stronger deletion) |
| `deletion_threshold` | float | None | Threshold for hard deletion. If None, uses soft deletion |
| `gate_layer_norm` | bool | True | Apply LayerNorm before delete gate |
| `use_gumbel_noise` | bool | False | Add Gumbel noise during training for exploration |
| `random_deletion_probability` | float | 0.5 | Deletion probability for random gate type |
| `fixed_deletion_amount` | float | 0.5 | Deletion fraction for fixed gate type |

---

## Training

```bash
# Masked Language Modeling (default)
python train_mrbert.py \
    --task mlm \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./mrbert_checkpoints \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --target_deletion_rate 0.3 \
    --deletion_loss_weight 0.1

# Sequence Classification (e.g., SST-2)
python train_mrbert.py \
    --task sequence_classification \
    --dataset_name glue \
    --dataset_config sst2 \
    --output_dir ./mrbert_sst2 \
    --num_epochs 3

# Token Classification (e.g., NER)
python train_mrbert.py \
    --task token_classification \
    --dataset_name conll2003 \
    --output_dir ./mrbert_ner

# Question Answering (e.g., SQuAD)
python train_mrbert.py \
    --task question_answering \
    --dataset_name squad \
    --output_dir ./mrbert_squad
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | mlm | Task: `mlm`, `sequence_classification`, `token_classification`, `question_answering` |
| `--model_name` | bert-base-uncased | Base BERT model |
| `--delete_gate_layer` | 2 | Delete gate placement |
| `--deletion_type` | scaled_sigmoid | Delete gate type |
| `--target_deletion_rate` | 0.0 | Target deletion rate (0 = no deletion loss) |
| `--deletion_loss_weight` | 0.1 | Weight for auxiliary deletion loss |
| `--delete_gate_lr` | 1e-4 | Learning rate for delete gate (can be higher than base model) |
| `--num_epochs` | 3 | Number of training epochs |
| `--batch_size` | 16 | Batch size |
| `--learning_rate` | 5e-5 | Learning rate |
| `--max_seq_length` | 128 | Maximum sequence length |
---

## Evaluation

```bash
# Evaluate a trained model
python eval_mrbert.py --model_path ./mrbert_checkpoints/final

# Evaluate on fresh MrBERT (no training)
python eval_mrbert.py --from_pretrained bert-base-uncased

# Specify evaluation dataset
python eval_mrbert.py \
    --model_path ./mrbert_checkpoints/final \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_samples 1000
```

### Evaluation Metrics

The evaluation script reports:

1. **MLM Perplexity**: How well the model predicts masked tokens
2. **Deletion Rate**: Average fraction of tokens deleted
3. **Token Analysis**: Which token types are most frequently deleted
4. **Deletion by Position**: Are early/late tokens deleted more often?
5. **Comparison with BERT**: MrBERT vs baseline BERT perplexity

---

## Supported Tasks

| Model Class | Task | Output |
|-------------|------|--------|
| `MrBertModel` | Base model | Hidden states + delete gate outputs |
| `MrBertForMaskedLM` | Masked Language Modeling | Token predictions |
| `MrBertForSequenceClassification` | Text Classification | Class logits |
| `MrBertForTokenClassification` | NER, POS Tagging | Per-token labels |
| `MrBertForQuestionAnswering` | Extractive QA | Start/end positions |
| `MrBertForMultipleChoice` | Multiple Choice | Choice logits |
| `MrBertForNextSentencePrediction` | NSP | Binary classification |
---

## File Structure

```
CS224N-project/
├── modeling_mrbert.py        # Main MrBERT model implementation
├── configuration_mrbert.py   # MrBertConfig class
├── train_mrbert.py           # Training script for multiple tasks
├── eval_mrbert.py            # Evaluation script
├── test_mrbert.py            # Test suite
├── modeling_bert.py          # Reference BERT implementation
├── README.md                 # This documentation
│
├── mrbert_checkpoints/       # Saved model checkpoints
│   └── final/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files...
│
└── mrt5/                     # Original MrT5 reference implementation
    ├── models/
    │   └── modeling_mrt5.py  # Original T5 with delete gates
    ├── data/
    ├── eval/
    └── training/
```

---

## Running on Modal

Train on a serverless GPU (e.g. T4) without managing machines:

```bash
# Install Modal and log in (one-time)
pip install modal
python3 -m modal setup

# Short test run (default: 20 steps)
modal run train_modal.py

# Longer run (e.g. 500 steps)
modal run train_modal.py --max-steps 500
```

Checkpoints are written to the Modal Volume `mrbert-checkpoints`. To download them locally after a run:

```bash
modal volume get mrbert-checkpoints final ./local_mrbert_final
# Or list first: modal volume ls mrbert-checkpoints
```

See `train_modal.py` for defaults and how to customize the training args (e.g. batch size, dataset) via the `train()` function.

---

## Testing

Run the test suite to verify the implementation:

```bash
python test_mrbert.py
```

This runs tests for:
- ✅ Configuration creation
- ✅ Delete gate modules (all types)
- ✅ MrBertModel forward pass (soft deletion)
- ✅ MrBertModel forward pass (hard deletion)
- ✅ All task-specific model heads
- ✅ Gradient flow through delete gate
- ✅ Comparison with baseline BERT
---
# Local Testing

cd mrbert
python data/preprocess_snli.py --output_dir ./snli_datasets --max_samples 1000
python training/train_mrbert.py --model_type BERT --task sequence_classification --dataset_name local_snli --local_snli_dir ./snli_datasets --max_steps 50 --batch_size 8 --logging_steps 10 --output_dir ./bert_snli_test --disable_wandb
---
# metrics

⏺ For BERT (model_type=BERT), the delete gate is absent so delete_gate_output is None. Only task metrics are logged:                                                                                       
                                                                                                                                                                                                         
  ┌────────────────────┬────────────────────────────────────────────┐                                                                                                                                      
  │       Metric       │                Description                 │                                                                                                                                      
  ├────────────────────┼────────────────────────────────────────────┤                                                                                                                                      
  │ loss               │ Total loss (= task loss, no deletion term) │                                                                                                                                      
  ├────────────────────┼────────────────────────────────────────────┤
  │ cross_entropy_loss │ Cross-entropy over 3 SNLI classes          │
  ├────────────────────┼────────────────────────────────────────────┤
  │ total_loss         │ Same as loss                               │
  ├────────────────────┼────────────────────────────────────────────┤
  │ learning_rate      │ Current LR from scheduler                  │
  └────────────────────┴────────────────────────────────────────────┘

  For MrBERT (model_type=MrBERT), all of the above plus gate metrics:

  ┌────────────────────────────────┬───────────────────────────────────────────────────────┐
  │             Metric             │                      Description                      │
  ├────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ delete_gate_loss               │ Mean gate value over non-pad tokens (drives deletion) │
  ├────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ delete_gate_loss_coeff         │ Current α from PI-controller                          │
  ├────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ percent_deleted_tokens         │ % non-pad tokens with gate < threshold                │
  ├────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ percent_non_pad_deleted_tokens │ Same (both are identical in current code)             │
  ├────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ delete_gate_average            │ Mean gate value across batch                          │
  ├────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ delete_gate_std                │ Std of gate values per sequence, averaged over batch  │
  ├────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ delete_gate_max_value          │ Mean of per-sequence max gate values                  │
  ├────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ delete_gate_min_value          │ Mean of per-sequence min gate values                  │
  └────────────────────────────────┴───────────────────────────────────────────────────────┘

  All metrics are printed to stdout every --logging_steps steps. With --disable_wandb they won't be sent anywhere — just console output.

  Two things to watch for in the MrBERT run:
  - percent_deleted_tokens should gradually increase toward 30% (the --target_deletion_rate default) as the PI-controller ramps up α
  - delete_gate_average starting near 0 (from the bias=10 initialization) and drifting negative as the gate learns to delete

  
  ┌────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────┐                                                                               
  │         Metric         │                                       What it measures                                        │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤                                                                               
  │ loss                   │ Total loss (task + α × deletion)                                                              │                                                                             
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤                                                                               
  │ accuracy               │ Fraction of correct predictions in the batch                                                  │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
  │ new_seq_len            │ Mean number of tokens kept per sequence (gate > threshold for MrBERT; non-pad count for BERT) │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
  │ percent_deleted_tokens │ % non-pad tokens deleted by the gate (0 for BERT baseline)                                    │
  └────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘

  All four appear in both stdout and wandb. The progress bar also now shows loss + acc per step.
---
  With k_p=0.01, the proportional term adds at most 0.1 * 0.01 * 0.3 = 0.0003 per step, and the integral term adds 0.00001 * 0.3 = 0.000003 per step — α will rise to ~0.01 over thousands of steps rather 
  than hundreds.
                                                                                                                                                                                                           
  Watch for delete_gate_loss_coeff staying near 0.01 for the first few hundred steps post-delay — that's the sign it's working correctly.
---
 ## Monitoring with Modal

  No, Modal doesn't support SSH into running containers. It's a serverless platform — you don't get direct shell access to the machine running your job.                                                   
                                                                                                                                                                                                         
  What you can do instead:                                                                                                                                                                                 
                                                                                                                                                                                                           
  1. Stream logs in real time (best option):                                                                                                                                                               
  modal app logs mrbert-train                                                                                                                                                                              

  2. Watch a specific run:
  modal container list  # get the container ID
  modal container exec <container-id> /bin/bash  # Modal does support this for running containers
  `modal container exec is the closest equivalent to SSH — it drops you into a shell in the running container. Check if your container ID is listed while the job is running.`

  3. Monitor via wandb — since you have wandb hooked up, the live metrics at https://wandb.ai/aronima7-stanford-university/mrbert are the most practical way to watch training without shell access.

  4. Add more logging to train_mrbert.py if you need to debug something specific — the output streams back through modal app logs.
---
 # nvidia GPU specs

 modal container list
 ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Container ID                  ┃ App ID                    ┃ App Name     ┃ Start Time           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ ta-01KJDGZ2KKTAK60ZR4P2S344AS │ ap-npSr8kooNdWuOqP3EtarIO │ mrbert-train │ 2026-02-26 09:46 PST │
│ ta-01KJDDS2N4R5CK8KTKE89D4BCJ │ ap-JquTj9JSdBlWFOL55k32ia │ mrbert-train │ 2026-02-26 08:51 PST │
└───────────────────────────────┴───────────────────────────┴──────────────┴──────────────────────┘
 modal container exec ta-01KJDGZ2KKTAK60ZR4P2S344AS /bin/bash

 root ~ → nvidia-smi
Thu Feb 26 17:53:14 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  |   00000000:80:00.0 Off |                    0 |
| N/A   61C    P0            295W /  400W |    5233MiB /  40960MiB |     93%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A               1      C   /bin/dumb-init                         5224MiB |
+-----------------------------------------------------------------------------------------+
---
# Stop modal training run

  modal app stop mrbert-train                                                                                                                                                                              
                                                                                                                                                                                                         
  Or if you want to stop a specific run without killing the whole app:                                                                                                                                     
  modal app list                    # get the app ID                                                                                                                                                       
  modal app stop <app-id>                                                                                                                                                                                  
                                                                                                                                                                                                           
  You can also cancel from the Modal dashboard at https://modal.com/apps — find mrbert-train and click Stop.
---
# GCP

 ● Do you have an existing GCP project with A100 quota?             
   → Existing GCP project
 ● How do you want to run training on GCP?                                                                                                                                                                 
   → Vertex AI custom job
 ● Where should checkpoints be saved?                                                                                                                                                                      
   → GCS bucket 

 ● Do you want to hardcode your GCP project ID and region, or use placeholders?                                                                                                                            
   → Use placeholders                                                                                                                                                                                      
 ● How should the training environment be packaged?                                                                                                                                                        
   → Custom Docker image                                                                                                                                                                                   
 ● Do you have an existing GCS bucket for checkpoints?                                                                                                                                                     
   → New bucket 

❯ 1. [✔] gcloud CLI ready                                                                                                                                                                                  
  gcloud is installed and authenticated (gcloud auth login done)                                                                                                                                           
  2. [✔] Vertex AI API enabled                                                                                                                                                                           
  Vertex AI API is enabled in the project
  3. [✔] Artifact Registry API enabled
  Artifact Registry API is enabled

 ---
  GCP Vertex AI

  One-time setup — fill in your values at the top of train_gcp.py:                                                                                                                                         
  GCP_PROJECT = "your-project-id"
  GCP_REGION  = "us-central1"                                                                                                                                                                              
  GCS_BUCKET  = "your-bucket-name"                                                                                                                                                                       

  Create the GCS bucket:
  gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME

  Build and push the Docker image:
  cd /Users/aronimadass/Desktop/projects/stanford/CS224N-project/mrbert/training && python train_gcp.py --build-image

  Preprocess SNLI to GCS (one-time):
  python train_gcp.py --preprocess-snli

  Submit training jobs:
  python train_gcp.py --model-type MrBERT --max-steps 30000
  python train_gcp.py --model-type BERT --max-steps 30000

  Download checkpoints:
  gsutil -m cp -r gs://YOUR_BUCKET/checkpoints/mrbert-mrbert-30000steps ./local_mrbert_checkpoints

  Install the required local deps first:
  pip install google-cloud-aiplatform google-cloud-storage
 ---  
  GCP GCE A100
  Key differences vs Vertex AI:

  ┌─────────────────────────┬────────────┬────────────────────────────────┐
  │                         │ Vertex AI  │             GCE VM             │
  ├─────────────────────────┼────────────┼────────────────────────────────┤
  │ SSH access              │ No         │ Yes                            │
  ├─────────────────────────┼────────────┼────────────────────────────────┤
  │ Startup overhead        │ ~5 min     │ ~2 min                         │
  ├─────────────────────────┼────────────┼────────────────────────────────┤
  │ Auto-stop on completion │ Yes        │ No — you pay until you stop it │
  ├─────────────────────────┼────────────┼────────────────────────────────┤
  │ Checkpoint persistence  │ GCS bucket │ Local disk or GCS              │
  ├─────────────────────────┼────────────┼────────────────────────────────┤
  │ Job queuing             │ Yes        │ No                             │
  └─────────────────────────┴────────────┴────────────────────────────────┘

  The pytorch-latest-gpu image from deeplearning-platform-release comes with CUDA, PyTorch, and conda pre-installed, so you can clone the repo and run train_mrbert.py directly without Docker.
 --- 
⏺ Created train_gce.py. Here's a summary of usage:                                                                                                                                                       
                                                                                                                                                                                                           
  Full training run (VM stops when done):                                                                                                                                                                  
  python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000                                                                                                    
                                                                                                                                                                                                           
  Delete VM after training (cheapest option):                                                                                                                                                            
  python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000 --delete-vm                                                                                      

  SSH into the running VM:
  python train_gce.py --project YOUR_PROJECT --ssh

  Stop or delete VM manually:
  python train_gce.py --project YOUR_PROJECT --stop-vm
  python train_gce.py --project YOUR_PROJECT --delete-vm-only

  Skip re-uploading code on a second run:
  python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type BERT --max-steps 30000 --skip-upload --skip-setup

  The script handles the full lifecycle: creates the VM, uploads code, installs deps, preprocesses SNLI (or downloads from GCS if already done), runs training, uploads checkpoints to GCS, then
  stops/deletes the VM.
 --- 
ARONIMA RUNS

  cd mrbert
  python data/preprocess_snli.py --output_dir ./snli_datasets --max_samples 1000
  python training/train_mrbert.py --model_type MrBERT --task sequence_classification --mode training-and-eval --dataset_name local_snli --local_snli_dir ./snli_datasets --max_steps 50 --batch_size 8 --logging_steps 10 --output_dir ./bert_snli_test --disable_wandb

  python train_gcp.py --model-type MrBERT --max-steps 30000
  python train_gcp.py --model-type BERT --max-steps 30000

  modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --target-deletion-rate 0.3 --mode training-and-eval --delete-gate-layer 3
  modal run --detach train_modal.py --model-type BERT --max-steps -1 --num-epochs 3 --mode training-and-eval

  no PI
  modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --target-deletion-rate 0.3 --mode training-and-eval --no-use-pi-controller
--- 
❯ how many epochs will give me viable results?                                                                                                                                                             
  ⎿  ⧉ Selected 1 lines from mrbert/README.md in PyCharm                                                                                                                                                 
                                                                                                                                                                                                           
⏺ That's a research/empirical question rather than a code question, and the answer depends on what you observe during training. A few guidelines based on standard NLI fine-tuning practice:             
                                                                                                                                                                                                           
  Typical BERT fine-tuning on SNLI:                                                                                                                                                                        
  - 3–5 epochs is the standard range in the literature                                                                                                                                                     
  - BERT-base usually converges on SNLI within 3 epochs (validation accuracy ~90-91%)                                                                                                                      
  - Beyond 5 epochs you risk overfitting

  For MrBERT specifically:
  - You may need more epochs than baseline BERT because the delete gate needs time to learn — the --regularizer_delay 1000 already accounts for this by letting the model learn the task before applying
  deletion pressure
  - The PI controller adjusting deletion_loss_weight also needs steps to stabilize
  - 3 epochs is a reasonable starting point, but you may want to run 5 if the gate hasn't converged

  Practical advice:
  - Watch the validation accuracy and deletion rate![img.png](img.png) in the logs
  - If the deletion rate hasn't hit the 30% target by the end of epoch 1, the gate is still learning and more epochs will help
  - If validation accuracy is still climbing at the end of epoch 3, run more

  There's no substitute for actually looking at your training curves. If you have W&B set up, that'll make it easy to decide whether to run longer.
---
W&B metrics logged by the eval loop: eval/loss, eval/accuracy, eval/percent_deleted_tokens, eval/avg_seq_len.
---
ANALYSIS

⏺ A few observations:                                                                                                                                                                                      
                                                                                                                                                                                                           
  The PI controller is working — the deletion rate is slowly converging toward the 30% target. By step 30k it's sitting around 25-27%, close but not quite there yet.                                      
                                                                                                                                                                                                           
  The early spike (~80% at step ~2k) is expected — this is before the PI controller has stabilized. The gate starts deleting aggressively, then the controller pulls it back.                              
                  
  The two charts are identical — percent_deleted_tokens and percent_non_pad_deleted_tokens show the same curve, which means the deletion rate calculation isn't actually distinguishing between pad and
  non-pad tokens. This is a bug in the logging — both metrics are set to the same value (avg_del_pct) in train_mrbert.py:

  "percent_deleted_tokens": round(avg_del_pct, 4),
  "percent_non_pad_deleted_tokens": round(avg_del_pct, 4),

  The convergence is slow — still trending downward at step 30k, suggesting the model hasn't fully settled. This supports running longer (3 full epochs / ~51k steps) to see if it stabilizes at 30%.

  The high variance throughout (±10-15%) suggests the PI controller gains (controller_p=0.01, controller_i=0.00001) may be too low, causing sluggish correction. If you want faster convergence to 30%, you
   could try increasing controller_p.
---
A few observations:                                                                                                                                                                                      
                                                                                                                                                                                                           
  Loss is healthy                                                                                                                                                                                          
  - Total loss drops cleanly from ~1.1 to ~0.7, still trending down at step 30k. The model hasn't converged — it would benefit from more steps.                                                            
                                                                                                                                                                                                           
  Deletion rate is not converging to 30%
  - Both percent_non_pad_deleted_tokens and percent_deleted_tokens are nearly identical (~25-27% at step 30k), which is suspicious — with padding in SNLI sequences the all-token rate should be noticeably
   lower than the non-pad rate. Worth double-checking the fix took effect.
  - More importantly: the target is 30% but it's plateauing around 25-27%. The PI controller isn't pushing it up enough, likely because controller_p=0.01 and controller_i=0.00001 are too conservative.

  new_seq_len trend is concerning
  - Effective sequence length starts at ~5-6, spikes to ~25, then slowly climbs to ~20. The early low values (steps 1k-5k) suggest the gate was deleting almost everything before the PI controller kicked
  in. By step 30k it's stabilizing around 19-20 tokens — that's very aggressive deletion given SNLI sequences are typically 20-40 tokens after tokenization.

  Learning rate is nearly exhausted
  - The linear decay has brought LR almost to zero by step 30k. This means running more steps without adjusting the schedule won't help much — the LR scheduler was configured for 30k steps. If you train
  for 51k steps, you should set --max-steps -1 so the scheduler recalculates total_steps from num_epochs rather than being set to 30k.

  Summary: the model is learning but the gate hasn't reached target. Running 3 full epochs with a corrected LR schedule is the right next step.
---
❯   Loss is healthy                                                                                                                                                                                        
                                                                                                                                                                                                       
    - Total loss drops cleanly from ~1.1 to ~0.7, still trending down at step 30k. The model hasn't converged — it would benefit from more steps.                                                          
                                                                                                                                                                                                       
  do you expect to see the loss plateau once the model has converged instead of trending downwards?                                                                                                        
                                                                                                                                                                                                           
⏺ Yes — for a well-behaved training run you'd expect the training loss to flatten out as the model saturates its capacity on the data. The fact that it's still trending down at step 30k just means it    
  hasn't reached that plateau yet.                                                                                                                                                                         
                                                                                                                                                                                                           
  That said, training loss alone isn't the right convergence signal here. What you actually want to watch is validation accuracy — training loss can keep slowly decreasing due to mild overfitting even   
  after validation accuracy has peaked. That's why the eval loop matters: you could be at 91% val accuracy already and the loss curve wouldn't tell you that.  
---
❯   new_seq_len trend is concerning                                                                                                                                                                        
    - Effective sequence length starts at ~5-6, spikes to ~25, then slowly climbs to ~20. The early low values (steps 1k-5k) suggest the gate was deleting almost everything before the PI controller  
  kicked                                                                                                                                                                                                   
    in. By step 30k it's stabilizing around 19-20 tokens — that's very aggressive deletion given SNLI sequences are typically 20-40 tokens after tokenization.                                             
                                                                                                                                                                                                           
                                                                                                                                                                                                           
  do we need to adjust anything to account for this concern? 
  
  The gate initialization is already designed to prevent this — _init_delete_gates() sets bias=10 so initial gate values are ≈ 0, well above the -15 threshold, meaning all tokens should be kept at the
  start. The early low new_seq_len of 5-6 is inconsistent with that initialization working correctly.

  The most useful diagnostic: look at delete_gate_average in W&B at the very first logging step (step 50). If it's close to 0, the initialization is fine and the early seq_len drop has another cause. If
  it's close to -30, the initialization isn't taking effect.

  That said, two adjustments worth considering regardless:

  1. Increase regularizer_delay — currently 1000 steps. Bumping to 2000-3000 gives the model more time to learn the classification task before deletion pressure engages. This is a safe change with no
  downside.
  2. The PI controller gains may need tuning — the current controller_p=0.01 is deliberately conservative (to prevent gate collapse), but it means the controller responds slowly when the deletion rate
  drifts far from target. The early spike to ~80% then slow recovery to 25-27% is a symptom of this. Increasing controller_p to 0.05 would make it respond faster, but risks overshooting in the other
  direction.

  Neither of these is urgent — the model IS learning and the gate IS broadly working. The early instability settles out. The more important issue remains the LR schedule: if you run 3 epochs (~51k steps)
   the scheduler needs to be recalculated for that budget, otherwise LR decays to near zero by step 30k and the remaining 21k steps train with almost no gradient signal.
---
 ❯ explain steps and epochs in the context of training runs                                                                                                                                                 
                                                           
⏺ Step: one forward + backward pass on a single batch. Each step the model sees batch_size examples and updates its weights once.                                                                          
                                                                                                                                                                                                           
  Epoch: one full pass through the entire training dataset. With 549,367 SNLI examples and batch size 32, one epoch = ~17,168 steps.                                                                       
                  
  The relationship:                                                                                                                                                                                        
  steps_per_epoch = dataset_size / batch_size                                                                                                                                                              
  total_steps     = steps_per_epoch × num_epochs

  So for this training setup:
  1 epoch  ≈  17,168 steps
  3 epochs ≈  51,500 steps

  Why the distinction matters here: the LR scheduler decays linearly from learning_rate to 0 over total_steps. If you set --max-steps 30000 but train for 51k steps, the LR hits near-zero at step 30k and
  the last 21k steps are nearly useless. If you use --max-steps -1, the scheduler correctly sets total_steps = 51,500 and the LR decays smoothly over all 3 epochs.
---
⏺ Yes. Running without the PI controller means α (the deletion loss weight) stays fixed at whatever --deletion_loss_weight is set to, rather than being dynamically adjusted to hit the target deletion  
  rate.                                                                                                                                                                                                    
                                                                                                                                                                                                           
  What it signifies: You're applying a constant, fixed pressure on the gate to delete tokens, regardless of whether the actual deletion rate is above or below the target. The model has to find its own   
  equilibrium between task loss and deletion loss — there's no feedback mechanism correcting it toward 30%. In practice this means:                                                                        
                  
  - The deletion rate may settle anywhere, not necessarily near the target
  - If --deletion_loss_weight is too high, the gate collapses (deletes too much)
  - If too low, the gate learns to delete nothing
  - It's a useful ablation to show the PI controller is necessary for stable, targeted deletion

  The PI controller is already configurable — --use_pi_controller is a flag in train_mrbert.py (line 249, default True). You just need to expose it in train_modal.py. Want me to add it?
---
  modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --mode training-and-eval --no-use-pi-controller

  With --no-use-pi-controller, α stays fixed at --deletion_loss_weight (default 0.01) for the entire run. If you want stronger fixed deletion pressure, pair it with a higher value, e.g. pass
  --deletion_loss_weight 0.1 via extra_args.
---
change location of deletion gate (default is 3)
  # Run 6: gate at layer 1                                                                                                                                                                                 
  modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --target-deletion-rate 0.3 --mode training-and-eval --delete-gate-layer 1
                                                                                                                                                                                                           
  # Run 7: gate at layer 6                                                                                                                                                                               
  modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --target-deletion-rate 0.3 --mode training-and-eval --delete-gate-layer 6
---
run training + eval in one go
run for 3 epochs

vary deletion percentage (20, 30, 50)
run with/without pi controller
run with gate at layer 1, 3, 6
---
 For the modal run: "modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --target-deletion-rate 0.3 --mode training-and-eval --delete-gate-layer 3 --wandb-run-name       
  run1-mrbert-deletion_rate_30pct", following is the eval result:                                                                                                                                          
    eval/loss: 0.5483                                                                                                                                                                                      
    eval/accuracy: 0.78                                                                                                                                                                                    
    eval/percent_deleted_tokens: 33.6186                                                                                                                                                                   
    eval/avg_seq_len: 18.22                                                                                                                                                                                
  What is your analysis and recommendation for next run?                                                                                                                                                   
  ⎿ [Image #9]                                                                                                                                                                                            
  ⎿ [Image #10]                                                                                                                                                                                           
  ⎿ [Image #11]                                                                                                                                                                                           
  ⎿ [Image #12]
---
⏺ This is because percent_deleted_tokens (the W&B training metric) and eval/percent_deleted_tokens (the eval metric) are measuring different things:                                                       
                                                                                                                                                                                                           
  - Training chart (percent_deleted_tokens): fraction of all tokens deleted, including padding. With max_seq_length=128 and SNLI sequences averaging ~26 non-pad tokens, ~80% of each sequence is padding. 
  Padding is always forced to gate value -30 (always deleted). So ~80% padding deletion + ~30% of non-pad deleted = ~86% all-token deletion. This is expected.                                             
  - Eval result (eval/percent_deleted_tokens: 33.6%): this is using percent_deleted from compute_deletion_loss, which only counts non-pad tokens. So 33.6% means 33.6% of actual content tokens were       
  deleted — right on the 30% target.

  So the two numbers are consistent — they're just measuring different denominators. The 86% figure is dominated by padding and is misleading as a headline metric. The 33.6% non-pad figure is the one
  that actually matters.

  This also confirms the earlier observation: the two W&B training metrics percent_deleted_tokens and percent_non_pad_deleted_tokens should be showing ~86% and ~30% respectively — if they're both showing
   ~86%, the fix we made to compute_deletion_loss to return separate values may not have taken effect in this run (it was run before that fix). Worth checking on the next run.
---
 --controller-p

 To run with faster PI controller convergence:

  modal run --detach train_modal.py --model-type MrBERT --max-steps -1 --num-epochs 3 --target-deletion-rate 0.3 --mode training-and-eval --controller-p 0.05 --wandb-run-name                             
  run3-mrbert-controller_p_0.05
---
❯ explain cross entropy loss and training accuracy to me in the context of your analysis:                                                                                                                  
    1. The model hasn't converged — training accuracy is still rising at step 51k and cross_entropy_loss is still declining. The model needs more epochs.                                                 
                                                                                                                                                                                                           
⏺ Cross-entropy loss measures how confident and correct the model's predictions are. For a 3-class problem (entailment/neutral/contradiction), it outputs a probability distribution over the 3 classes.   
  Cross-entropy penalizes the model when it assigns low probability to the correct class:                                                                                                                  
                                                                                                                                                                                                           
  - Loss = 1.1 → model is essentially guessing (random = log(3) ≈ 1.1)                                                                                                                                     
  - Loss = 0.5 → model is moderately confident and mostly correct                                                                                                                                          
  - Loss = 0.0 → model is perfectly confident on every example

  Training accuracy is simpler — just the fraction of training batches where argmax(logits) == true_label.

  Why both matter for convergence:

  They're measuring the same thing from different angles. In a converged model you expect both to plateau — accuracy stops rising and loss stops falling because the model has extracted all the signal it
  can from the data.

  In Run 1 at step 51k:
  - cross_entropy_loss was still declining (hadn't flattened)
  - accuracy was still rising (hadn't flattened)

  Both curves still had positive slope, meaning the model was still actively learning. A converged model's curves look flat for the last 5,000–10,000 steps. The fact that both were still moving at the
  end of epoch 3 means epoch 3 was not the ceiling — more epochs would have continued improving the model.

  This is why 0.78 eval/accuracy is likely an underestimate of what MrBERT can achieve — the model simply hadn't finished training.
---
 use_softmax1 default changed from False → True
To disable it explicitly:                                                                                                                                                                              
  modal run --detach train_modal.py --model-type MrBERT ... --no-use-softmax1
---
 Yes, it does impact the analysis in a few specific ways:  

  Run 1 used use_softmax1=False                                                                                                                                                                            
   
  The Run 1 results (eval/accuracy: 0.78, ~33% deletion rate) were produced with standard softmax — meaning when tokens were soft-deleted (large negative attention bias), their attention mass            
  redistributed to remaining tokens. This is architecturally incorrect for a deletion model: you're suppressing a token but inadvertently boosting attention on others.                                  

  What softmax1 changes

  With use_softmax1=True, attention weights can sum to < 1. When a token is suppressed by the gate, its attention "falls off" rather than flowing to neighbors. This is the behavior the MrT5 paper
  designed for — the gate and attention variants are meant to work together.

  Concrete implications for your analysis:

  1. Run 1 accuracy (0.78) is a slightly pessimistic baseline for MrBERT. Future runs with softmax1=True may see a small accuracy improvement simply from the attention fix, independent of any other
  changes.
  2. Fair comparison is now compromised. Run 1 (softmax1=False) vs. Run 2+ (softmax1=True) conflates two variables. If accuracy improves, you can't attribute it solely to hyperparameter changes.
  3. The BERT baseline is also affected. Even without deletion, softmax1 changes attention behavior. For a fair comparison, the BERT baseline run should use consistent use_softmax1 settings relative to
  whichever MrBERT runs it's being compared against.

  Recommendation:

  Re-run the BERT baseline and at least one MrBERT run (your primary 30% deletion run) with use_softmax1=True before drawing conclusions. The softmax variant should be held constant across all
  comparisons you report. Since use_softmax1=True is now the default and aligns with the MrT5 paper, treat it as the correct setting going forward and treat Run 1 as a preliminary/exploratory run rather
  than a reportable result.
---
Modal logs:
look for 'Starting training...' to see the initial params for training run
look for ''
---
Analysis:

⏺ Three scripts created under mrbert/analysis/. Here's what each does and how they connect:

  ---
  Workflow

  # Step 1 — run model over SNLI test, save gate decisions
  python analysis/get_deletion_patterns.py \
      --model_path ./mrbert_checkpoints/final \
      --local_snli_dir ./snli_datasets \
      --sample_size 1000 \
      --output_dir ./analysis/deletion_patterns

  # Step 2 — analyse patterns from step 1
  python analysis/deletion_pattern_analysis.py \
      --input_file ./analysis/deletion_patterns/final_test.json \
      --output_dir ./analysis/figures

  # Step 3 — compute/plot theoretical MACs savings (no model needed)
  python analysis/compute_savings.py

  ---
  Script breakdown

  Script: get_deletion_patterns.py
  MrT5 equivalent: get_deletion_patterns.py
  What it does: Runs MrBertForSequenceClassification over SNLI; saves per-token gate_values, deletion_mask, prediction, label to JSON
  ────────────────────────────────────────
  Script: compute_savings.py
  MrT5 equivalent: compute_savings.ipynb
  What it does: BERT-specific MACs formula (encoder-only, no decoder); plots relative compute vs deletion ratio; two figures: gate@layer3 and gate@layer 1/3/6/9
  ────────────────────────────────────────
  Script: deletion_pattern_analysis.py
  MrT5 equivalent: deletion_pattern_analysis.ipynb
  What it does: Loads JSON from step 1; colored terminal output; deletion rates by token type; top most/least deleted tokens; SNLI-specific premise vs hypothesis deletion rates; saves bar charts

  Key SNLI-specific addition not in MrT5: the premise vs hypothesis breakdown — whether MrBERT preferentially deletes more from the premise or hypothesis segment.

✻ Churned for 4m 31s

❯ how and when do you recommend running analysis, ie, do I need to run this for each run or only the final run?                                                                                           
                                                                                                                
⏺ Run analysis on selected runs only, not every run. Here's the reasoning and a concrete recommendation:                                                                                                  
                                                                                                                                                                                                        
  ---                                                                                                                                                                                                     
  What analysis tells you vs what W&B already tells you
                                                                                                                                                                                                          
  W&B already gives you the key training signals during a run: accuracy, loss, deletion rate, gate average/std, alpha. You don't need the analysis scripts to decide whether a run is good.               

  The analysis scripts answer different questions:
  - Which tokens is the gate choosing to delete? (semantic interpretation)
  - Is the gate deleting more from premise or hypothesis? (SNLI-specific behaviour)
  - What is the actual compute saving at a given deletion rate? (paper numbers)

  These are paper-writing questions, not training decisions.

  ---
  Recommended timing

  compute_savings.py — run once, now. It requires no model, just the deletion rate numbers from your runs. You can already plug in 30% and 50% to get the theoretical savings figures for your paper.

  python analysis/compute_savings.py

  get_deletion_patterns.py + deletion_pattern_analysis.py — run on two or three specific checkpoints at the end:

  ┌──────────────────────────────────────┬──────────────────┬─────────────────────────────────────────────────────────────────────────┐
  │                 When                 │ Which checkpoint │                                   Why                                   │
  ├──────────────────────────────────────┼──────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ After your best MrBERT run converges │ Final checkpoint │ Primary results for the paper — what does the trained gate actually do? │
  ├──────────────────────────────────────┼──────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ After the 50% deletion run           │ Final checkpoint │ Compare deletion patterns at different rates                            │
  ├──────────────────────────────────────┼──────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Optionally: no-PI-controller run     │ Final checkpoint │ Does the gate learn a different deletion strategy without PI pressure?  │
  └──────────────────────────────────────┴──────────────────┴─────────────────────────────────────────────────────────────────────────┘

  **Do not run per run. The patterns won't be interpretable until the model has converged. A partially trained gate (e.g. step 10k) will show near-random deletion and tell you nothing meaningful.**

  ---
  **Concrete recommendation**

  Right now you have three runs in flight (30% softmax1, 50% softmax1, no-PI). Once they finish:

  1. Run compute_savings.py immediately — gives you the efficiency numbers
  2. Pick the best-accuracy run as your primary and run get_deletion_patterns.py → deletion_pattern_analysis.py on it
  3. If the 50% run performs acceptably, run analysis on that too to show the accuracy-efficiency curve has an interpretable gate at each operating point
  4. Skip analysis on the no-PI run unless its accuracy is competitive — it's mainly an ablation data point, not a primary result
---
⏺ Now you have three figures. Here's the usage flow:

  Right now (no run data needed):                                                                                                                                                                         
  python analysis/compute_savings.py
  # → macs_relative.pdf, macs_by_gate_layer.pdf                                                                                                                                                           
                                                                                                                                                                                                        
  After your runs finish (plug in real eval numbers):
  python analysis/compute_savings.py \
    --runs "BERT,0.9055,0.0" \
           "MrBERT-30%,<acc>,0.32" \
           "MrBERT-50%,<acc>,0.50"
  # → all three figures including accuracy_vs_compute.pdf

  What each figure shows:

  ┌─────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Figure          │                                              What it answers for the paper                                               │
  ├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ macs_relative.pdf       │ "How much compute does MrBERT save at a given deletion rate?" — the theoretical efficiency curve                         │
  ├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ macs_by_gate_layer.pdf  │ "Does it matter which layer the gate is at?" — motivation for the layer 3 choice                                         │
  ├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ accuracy_vs_compute.pdf │ Core result — "What accuracy do you give up for each unit of compute saved?" — this is the figure that goes in the paper │
  └─────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
---