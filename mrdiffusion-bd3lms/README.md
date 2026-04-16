# MrBD3LM

MrBD3LM adapts the **delete gate mechanism from MrT5/MrBERT** to the **BD3-LM (Block Denoising Discrete Diffusion Language Model)** architecture. After a specified transformer block, a learned gate assigns each *noisy* token a scalar score; low-scoring tokens are either soft-deleted (large negative attention bias) or hard-deleted (physically removed), reducing per-step computation across thousands of diffusion steps.

Reference papers:
- [BD3-LMs: Block Denoising Discrete Diffusion Language Models](https://arxiv.org/abs/2406.11524) (Arriola et al., 2024)
- [MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/abs/2410.20771) (Kallini et al., 2024)

---

## Why BD3-LM Is Better Than SEDD for Token Deletion

> **From the advisor guide:** *"BD3-LM is probably the better starting point for this work because of its masking-based noise process. When many tokens in a block are masked, those masked tokens are all identical `[MASK]`, which means they carry redundant information in intermediate layers — a perfect target for merging."*

| Property | SEDD | BD3-LM |
|---|---|---|
| Noise type | Uniform (random vocabulary values) | Absorbing (all masked tokens = `[MASK]`) |
| Redundancy of corrupted tokens | Low — each is a different random token | **High** — all `[MASK]` tokens are identical |
| Natural deletion target | No clear signal | Masked tokens → trivially redundant |
| Deletion schedule | Needs learned σ-dependent schedule | Natural: `r(t) = p(t)` (mask fraction) |

The gate learns: **delete masked tokens, keep clean tokens**. This creates a built-in curriculum: at high noise (many `[MASK]` tokens), delete aggressively; at low noise (mostly clean tokens), delete conservatively.

---

## Architecture

### Model Diagram

```
Input [B, L]         Noise level t [B]
  x_t (noisy)              │
  x_0 (clean)         ─────┤
       │                   │
       ▼                   ▼
EmbeddingLayer        TimestepEmbedder
       │
concat([x_t; x_0]) → [B, 2L, D]
       │
RoPE precomputed for x_t positions
Block-diff mask (M_BD ⊕ M_OBC ⊕ M_BC)  [2L × 2L]
       │
       ▼
┌─────────────────────────────────┐
│  MrDDiTBlock 0                  │ ← adaLN from c(t), full 2L attention
└─────────────────────────────────┘
       │
      ...          Phase 1 (full 2L)
       │
       ▼
┌─────────────────────────────────┐
│  MrDDiTBlock [delete_gate_layer]│ ← full 2L attention
└──────────────┬──────────────────┘
               │
               ▼
     ┌─────────────────────────────────────────────┐
     │  Delete Gate  (fires on x_t portion ONLY)   │
     │                                             │
     │  LayerNorm(x_t)                             │
     │       │                                     │
     │  Linear  ←── c(t)  (if sigma_conditioned)  │
     │       │                                     │
     │  ScaledSigmoid: s·sigmoid(−x), s=−30        │
     └─────────────────┬───────────────────────────┘
                       │
                gate [B, L, 1] in [−30, 0]
                  0 = keep, −30 = delete
                       │
         ┌─────────────┴─────────────────────────────────┐
         │  Soft deletion          Hard deletion          │
         │                                               │
         │  gate values added      x_t tokens below      │
         │  as additive bias on    threshold removed;     │
         │  x_t KEY positions;     x_0 stays intact;     │
         │  SDPA fallback          RoPE recomputed for    │
         │                         compressed x_t         │
         └─────────────┬───────────────────────────────────┘
                       │
                       │       Phase 2
                       │   (compressed / biased)
       ▼                                            ▼
┌─────────────────────────────────┐
│  MrDDiTBlock [gate_layer + 1]   │ ← soft: SDPA + gate bias; hard: SDPA on compressed
└─────────────────────────────────┘
       │
      ...
       │
       ▼
┌─────────────────────────────────┐
│  MrDDiTBlock [restore_gate_     │ ← restore full 2L before this block
│               layer]            │
└──────────────┬──────────────────┘
               │
               ▼  ── RESTORATION ──
     Soft: clear gate mask
     Hard: scatter compressed x_t back to full L;
           fill deleted positions with x_pre_gate
           (NOT zero logits — critical!)
               │
               │        Phase 3 (full 2L again)
       ▼
┌─────────────────────────────────┐
│  MrDDiTBlock [N − 1]            │ ← full 2L attention
└─────────────────────────────────┘
       │
       ▼
DDitFinalLayer  (adaLN + Linear)
       │
       │  logits [B, 2L, V] → first L logits returned
       │
       ▼
MDLM loss: −loss_scale · log p_θ(x_0 | x_t, x_0)  (masked positions only)
  +
Deletion Rate Loss: MSE(actual_rate, target_rate_at_t)
  where target_rate = p(t) = fraction of masked tokens  [noise_fraction schedule]
```

**Small model:** 12 blocks, D=768, 12 heads — default gate fires after block 3 (25% depth)
**Medium model:** 24 blocks, D=1024, 16 heads — recommended gate layer 6 (25% depth)

> **Why restoration matters:** Without it, hard-deleted positions receive zero logits from the output layer, maximizing loss at those positions and poisoning gradients. With `_restore_hidden_states`, deleted positions use their pre-gate hidden states, producing valid (if low-confidence) predictions everywhere.

---

### Attention Mask Modification

BD3-LM uses a special 3-component block-diff mask:
- **M_BD** (block-diagonal): Self-attention within noisy blocks
- **M_OBC** (offset-block-causal): Cross-attention from x_t to previous x_0 blocks
- **M_BC** (block-causal): Update x_0 tokens from clean context

MrBD3LM extends this with a **per-sample delete gate bias** added to x_t KEY positions:
```
attn_bias[:, :, :, :L] += delete_gate_mask  # [B, 1, 1, L] broadcast
```
Tokens with gate ≈ −30 become unreachable keys — effectively invisible to all queries.
This requires fallback from FlexAttention to SDPA (which supports additive biases).

---

### Delete Gate Mechanism

After DDiTBlock `delete_gate_layer` (default: 3), each **x_t token's** hidden state is passed through `LayerNorm → Linear → ScaledSigmoid`:

| Gate value | Meaning |
|---|---|
| ≈ 0 | Keep this token |
| ≈ −30 | Delete this token |

**Soft deletion** (default): Gate value added as large negative bias to attention key scores for x_t positions. All tokens remain in memory but deleted ones are ignored. Falls back to SDPA.

**Hard deletion**: Tokens where `gate < deletion_threshold` (default −15) are physically removed from x_t. x_0 stays intact. RoPE recomputed for shortened x_t. At `restore_gate_layer`, `_restore_hidden_states()` places pre-gate states back at deleted positions.

---

### Deletion Rate Schedule

BD3-LM enables a natural deletion rate target:

| Schedule | Formula | Description |
|---|---|---|
| `noise_fraction` **(recommended)** | `r(t) = p(t)` | Delete exactly the fraction of masked tokens at step t. Perfect alignment with BD3-LM noise |
| `constant` | `r = target_deletion_rate` | Fixed target (MrBERT baseline approach) |
| `linear_sigma` | `r = r_min + (r_max - r_min) · p(t)` | Linear ramp |
| `power_sigma` | `r = r_min + (r_max - r_min) · p(t)^α` | Convex/concave schedule |

The `noise_fraction` schedule is the key advantage of BD3-LM over SEDD:
since the gate **should** delete `[MASK]` tokens (all identical), the natural target rate
is exactly `p(t)` — the probability that a given token was masked at time t.

---

### Key Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `delete_gate_layer` | 3 | DDiTBlock index after which gate fires |
| `restore_gate_layer` | `None` | Block index to restore full L. `None` = just before output |
| `deletion_type` | `scaled_sigmoid` | Gate variant (see table below) |
| `deletion_mode` | `soft` | `soft` or `hard` |
| `sigmoid_mask_scale` | −30.0 | Gate output range lower bound |
| `deletion_threshold` | −15.0 | Threshold for hard deletion + rate metrics |
| `gate_sigma_conditioned` | **True** | Condition gate on noise level (guide-recommended) |
| `deletion_rate_schedule` | `noise_fraction` | Schedule for target deletion rate |
| `target_deletion_rate` | 0.3 | Used for `constant` schedule only |
| `r_min` / `r_max` | 0.05 / 0.5 | For `linear_sigma` / `power_sigma` |
| `deletion_rate_alpha` | 1.0 | Exponent for `power_sigma` |
| `deletion_loss_weight` | 0.1 | Weight of auxiliary deletion rate loss |
| `block_size` | 1 | BD3-LM block size (1=MDLM, 16/64=BD3-LM) |
| `cross_attn` | True | Use [x_t; x_0] cross-attention during training |

### Gate Variants

| `deletion_type` | Conditioning | Description |
|---|---|---|
| `scaled_sigmoid` + sigma | token + t (default) | Main learnable gate |
| `scaled_sigmoid` + `--no_gate_sigma_conditioned` | token only | Noise-blind (ablation) |
| `random` | — | Delete random x_t tokens (baseline) |
| `fixed` | — | Delete last `fixed_deletion_amount` fraction (baseline) |

---

## File Structure

```
mrdiffusion-bd3lms/
├── __init__.py                  # Package exports
├── configuration_mrd_bd3lm.py  # MrBD3LMConfig (extends BD3LMConfig)
├── modeling_mrd_bd3lm.py        # MrBD3LM model — gate classes + MrDDiTBlock + MrDITBackbone
├── train_mrd_bd3lm.py           # Local training script (argparse, W&B, single GPU)
├── train_modal_mrd_bd3lm.py     # Modal serverless GPU training
├── train_gcp_mrd_bd3lm.py       # GCP Vertex AI training (DDP + torchrun)
└── README.md                    # This file
```

The base BD3-LM code lives in `../diffusion/bd3lms/`. All `sys.path` manipulation is handled automatically — no environment variables needed.

---

## Environment Setup

```bash
# Option A: use the existing BD3-LM conda env
conda env create -f ../diffusion/bd3lms/requirements.txt
conda activate bd3lm

# Option B: create fresh env
conda create -n mrd-bd3lm python=3.10 -y
conda activate mrd-bd3lm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install einops transformers datasets accelerate tokenizers wandb tqdm numpy
# Optional: for FlexAttention (requires PyTorch 2.5+)
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
```

---

## W&B Setup

```bash
wandb login
# Or:
export WANDB_API_KEY=<your_key>
```

Metrics logged during training:

| Metric | Description |
|---|---|
| `train/loss` | Combined MDLM + deletion rate loss |
| `train/score_loss` | MDLM masked cross-entropy loss |
| `train/del_loss` | Deletion rate MSE auxiliary loss |
| `train/deletion_rate` | Actual fraction of x_t tokens deleted (hard threshold) |
| `train/learning_rate` | Current LR |
| `eval/loss` | Validation MDLM loss |
| `eval/deletion_rate` | Validation deletion rate |
| `eval/generative_perplexity` | GPT2-Large perplexity on generated samples |

---

## Local Training (Single GPU or CPU)

All commands run from the project root (`CS224N-project/`).

### Smoke test (CPU / single GPU, 200 steps, tiny model)

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --model_size tiny \
    --max_steps 200 \
    --logging_steps 20 \
    --eval_steps 100 \
    --batch_size 4 \
    --seq_len 128 \
    --dataset wikitext-2-v1 \
    --output_dir ./mrd_bd3lm_smoke \
    --disable_wandb
```

### Baseline BD3-LM (no gate)

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --no_delete_gate \
    --dataset wikitext-103-v1 \
    --output_dir ./mrd_bd3lm_baseline \
    --wandb_project mrd_bd3lm \
    --wandb_run_name bd3lm-baseline
```

### MrBD3LM — Soft deletion, noise_fraction schedule (recommended)

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --delete_gate_layer 3 \
    --deletion_mode soft \
    --deletion_rate_schedule noise_fraction \
    --deletion_loss_weight 0.1 \
    --dataset wikitext-103-v1 \
    --output_dir ./mrd_bd3lm_soft_nf \
    --wandb_project mrd_bd3lm \
    --wandb_run_name mrd-soft-layer3-nf
```

### MrBD3LM — Hard deletion

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --delete_gate_layer 3 \
    --deletion_mode hard \
    --deletion_rate_schedule noise_fraction \
    --output_dir ./mrd_bd3lm_hard \
    --wandb_run_name mrd-hard-layer3
```

### MrBD3LM — No sigma conditioning (ablation)

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --no_gate_sigma_conditioned \
    --deletion_rate_schedule constant \
    --target_deletion_rate 0.3 \
    --output_dir ./mrd_bd3lm_no_sigma \
    --wandb_run_name mrd-no-sigma-cond
```

### MrBD3LM — Random deletion baseline (ablation)

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --deletion_type random \
    --random_deletion_probability 0.3 \
    --deletion_loss_weight 0.0 \
    --output_dir ./mrd_bd3lm_random \
    --wandb_run_name mrd-random-30pct
```

### BD3-LM with block size > 1 (proper BD3-LM, not MDLM)

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --block_size 16 \
    --delete_gate_layer 3 \
    --deletion_mode soft \
    --output_dir ./mrd_bd3lm_block16 \
    --wandb_run_name mrd-soft-block16
```

### Medium model

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --model_size medium \
    --batch_size 16 \
    --delete_gate_layer 6 \
    --output_dir ./mrd_bd3lm_medium \
    --wandb_run_name mrd-medium-layer6
```

### Separate gate learning rate

```bash
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \
    --delete_gate_layer 3 \
    --delete_gate_lr 3e-3 \
    --output_dir ./mrd_bd3lm_gate_lr \
    --wandb_run_name mrd-gate-lr-3e3
```

### Resume from checkpoint

Training automatically resumes from `<output_dir>/checkpoints/checkpoint.pt` if it exists. Just re-run the same command.

---

## Modal (Serverless A100)

### Prerequisites

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>
modal secret create wandb-secret WANDB_API_KEY=<your_wandb_api_key>
```

### Smoke test (100 steps, wikitext-2)

```bash
modal run mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py
```

### Baseline BD3-LM

```bash
modal run --detach mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py::main \
    --no-delete-gate \
    --dataset wikitext-103-v1 \
    --max-steps 500000 \
    --wandb-run-name bd3lm-baseline
```

### MrBD3LM — Soft deletion (noise_fraction)

```bash
modal run --detach mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py::main \
    --deletion-mode soft \
    --deletion-rate-schedule noise_fraction \
    --deletion-loss-weight 0.1 \
    --max-steps 500000 \
    --wandb-project mrd_bd3lm \
    --wandb-run-name mrd-soft-layer3-nf
```

### MrBD3LM — Hard deletion

```bash
modal run --detach mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py::main \
    --deletion-mode hard \
    --deletion-rate-schedule noise_fraction \
    --max-steps 500000 \
    --wandb-run-name mrd-hard-layer3
```

### MrBD3LM — Medium model

```bash
modal run --detach mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py::main \
    --model-size medium \
    --batch-size 16 \
    --delete-gate-layer 6 \
    --max-steps 1000000 \
    --wandb-run-name mrd-medium-layer6
```

### Download checkpoints

```bash
# Download final checkpoint of a named run
modal volume get mrd-bd3lm-checkpoints \
    mrd-soft-layer3-nf/final \
    ./local_mrd_final

# List all stored runs
modal volume ls mrd-bd3lm-checkpoints
```

---

## GCP Vertex AI

### Prerequisites

```bash
pip install google-cloud-aiplatform gcsfs
gcloud auth application-default login
```

### Multi-GPU DDP (local torchrun, 4 GPUs)

```bash
torchrun --nproc_per_node=4 mrdiffusion-sedd-bd3lms/train_gcp_mrd_bd3lm.py \
    --max_steps 500000 \
    --batch_size 32 \
    --dataset openwebtext \
    --output_dir ./mrd_bd3lm_output \
    --wandb_project mrd_bd3lm \
    --wandb_run_name mrd-4gpu
```

### Submit Vertex AI job

```bash
python mrdiffusion-sedd-bd3lms/train_gcp_mrd_bd3lm.py --launch_gcp \
    --gcp_project <your-project-id> \
    --gcp_region us-central1 \
    --gcp_machine_type a2-highgpu-4g \
    --wandb_run_name mrd-gcp-a100 \
    --max_steps 500000 \
    --dataset openwebtext \
    --output_dir gs://your-bucket/mrd-bd3lm/run1
```

---

## Evaluation

Generative perplexity is logged automatically during training. For standalone evaluation:

```bash
# Generate samples and score with GPT2-Large
python -c "
import torch, sys
sys.path.insert(0, 'mrdiffusion-bd3lms')
sys.path.insert(0, 'diffusion/bd3lms')
from configuration_mrd_bd3lm import MrBD3LMConfig
from modeling_mrd_bd3lm import MrBD3LM
from train_mrd_bd3lm import compute_generative_perplexity
from noise_schedule import LogLinearNoise
from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('./mrd_bd3lm_output/final/checkpoint.pt', map_location=device)
config = MrBD3LMConfig(**ckpt['config'])
model = MrBD3LM(config).to(device)
model.load_state_dict(ckpt['model'])
tokenizer = AutoTokenizer.from_pretrained('gpt2')
noise = LogLinearNoise().to(device)
ppl = compute_generative_perplexity(model, tokenizer, noise, config.vocab_size-1, config, device, n_samples=32)
print(f'Generative perplexity: {ppl:.1f}')
"
```

---

## Experiment Suggestions

| Experiment | Key flags |
|---|---|
| Ablation: noise_fraction vs constant schedule | `--deletion_rate_schedule noise_fraction` vs `constant` |
| Ablation: sigma conditioning | default (`--gate_sigma_conditioned`) vs `--no_gate_sigma_conditioned` |
| Ablation: gate layer depth | `--delete_gate_layer 1 / 3 / 6 / 9` |
| Ablation: soft vs hard | `--deletion_mode soft` vs `hard` |
| Ablation: restore layer | `--restore_gate_layer -1` vs `6` vs `9` |
| Ablation: block size | `--block_size 1` (MDLM) vs `16` vs `64` (BD3-LM) |
| Baseline: no gate | `--no_delete_gate` |
| Baseline: random deletion | `--deletion_type random --random_deletion_probability 0.3` |
| Deletion rate sensitivity | `--target_deletion_rate 0.1 / 0.3 / 0.5` (constant schedule) |
| r_min / r_max sweep | `--deletion_rate_schedule linear_sigma --r_min 0.0 --r_max 0.3` |
| Gate learning rate | `--delete_gate_lr 3e-3` vs default joint LR |

Track all runs in W&B under the `mrd_bd3lm` project to compare perplexity curves and deletion rate trajectories.

---

## Key Design Decisions (Guide Compliance)

1. **Positional restoration** — `_restore_hidden_states()` fills deleted x_t positions with pre-gate hidden states. NOT zero logits. This prevents gradient poisoning from deleted positions.

2. **Sigma-dependent deletion schedule** — `noise_fraction` schedule makes target rate = `p(t)` (expected mask fraction at time t). Naturally aligns with BD3-LM's masking noise.

3. **`gate_sigma_conditioned=True` default** — Gate is noise-level-aware. At high t (mostly masked), gate can be aggressive. At low t (mostly clean), gate preserves more.

4. **Delete gate fires on x_t only** — Clean context tokens (x_0) are never deleted. They serve as crucial conditioning for denoising and should always be available.

5. **SDPA fallback during phase 2** — FlexAttention does not support per-sample additive key biases. Phase 2 uses `F.scaled_dot_product_attention` with the combined structural + delete gate bias. Phase 1 and 3 can use either.

6. **Speedup compounds** — In diffusion, total cost = (# steps) × (cost/step). Token merging reduces cost/step. With BD3-LM using T=5K steps, even 20% per-step savings → 20% total speedup.