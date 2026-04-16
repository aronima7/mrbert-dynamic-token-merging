# MrDiffusion

MrDiffusion adapts the **delete gate mechanism from MrT5/MrBERT** to the **SEDD (Score Entropy Discrete Diffusion)** architecture. After a specified transformer block, a learned gate assigns each token a scalar score; low-scoring tokens are either soft-deleted (large negative attention bias) or hard-deleted (physically removed), reducing computation while preserving generation quality.

Reference papers:
- [SEDD: Discrete Diffusion Modeling by Estimating the Ratio of the Data Distribution](https://arxiv.org/abs/2310.16834) (Lou et al., 2023)
- [MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/abs/2410.20771) (Kallini et al., 2024)

---

## Architecture

### Model Diagram

```
Input token ids  [B, L]          Noise level σ  [B]
       │                                │
       ▼                                ▼
 EmbeddingLayer                 TimestepEmbedder
       │                                │
       │  x [B, L, D]          c [B, cond_dim]
       │                                │
       ├──── RoPE precomputed ──────────┤
       │     (cos, sin)                 │
       │                                │
       ▼                                │
 ┌─────────────────────────────────┐    │
 │  DDiTBlock 0                    │◄───┤  adaLN modulation
 │  (adaLN + FlashAttn + MLP)      │    │  from c at every block
 └─────────────────────────────────┘    │
       │                                │
      ...          Phase 1              │
       │          (full L)              │
       ▼                                │
 ┌─────────────────────────────────┐    │
 │  DDiTBlock [delete_gate_layer]  │◄───┤
 └──────────────┬──────────────────┘    │
                │                       │
                ▼                       │
        ┌───────────────────────┐       │
        │      Delete Gate      │       │
        │                       │       │
        │  LayerNorm            │       │
        │      │                │       │
        │  Linear               │◄──── c  (if gate_sigma_conditioned=True)
        │      │                │       │
        │  ScaledSigmoid        │       │
        │  s·sigmoid(−x) s=−30 │       │
        └──────────┬────────────┘       │
                   │                    │
             gate [B, L, 1]             │
             values in [−30, 0]         │
                   │                    │
        ┌──────────┴──────────────────────────────────────┐
        │  Soft deletion              Hard deletion        │
        │                                                  │
        │  gate_mask [B,L,1] used     Tokens where         │
        │  as additive attn bias      gate ≤ threshold     │
        │  in subsequent blocks       physically removed;  │
        │  (SDPA fallback)            RoPE recomputed      │
        └──────────┬──────────────────────────────────────┘
                   │
                   │         Phase 2
                   │     (compressed / biased)
                   │
       ▼                                ▼
 ┌─────────────────────────────────┐    │
 │  MrDDiTBlock [gate_layer + 1]   │◄───┤
 └─────────────────────────────────┘    │
       │                                │
      ...                               │
       │                                │
       ▼                                │
 ┌─────────────────────────────────┐    │
 │  MrDDiTBlock [restore_gate_     │◄───┤
 │               layer]            │    │
 └──────────────┬──────────────────┘    │
                │                       │
                ▼  ── RESTORATION ──    │
        Soft: clear gate mask           │
        Hard: scatter compressed        │
              states back to L;         │
              fill deleted positions    │
              with x_pre_gate           │
              (NOT zero logits)         │
                │                       │
                │         Phase 3       │
                │       (full L again)  │
       ▼                                │
 ┌─────────────────────────────────┐    │
 │  MrDDiTBlock [restore + 1]      │◄───┤
 └─────────────────────────────────┘    │
       │                                │
      ...                               │
       ▼                                │
 ┌─────────────────────────────────┐    │
 │  MrDDiTBlock [n_blocks − 1]     │◄───┘
 └─────────────────────────────────┘
       │
       ▼
 DDitFinalLayer  (adaLN + Linear)
       │
       │  logits [B, L, vocab_size]  ← always full L; no zero-logit scatter
       │
       ▼
 Scale by σ  +  zero observed token logits
       │
       ▼
 Score Entropy Loss  +  Deletion Rate Loss r(σ)  [sigma-dependent target]
```

**SEDD small:** 12 blocks, D = 768, 12 heads — default gate fires after block 3 (25% depth)
**SEDD medium:** 24 blocks, D = 1024, 16 heads — recommended gate layer 6 (25% depth)

> **Restoration matters.** Without it, hard-deleted positions receive zero logits at the output, maximising the score-entropy loss for those positions and poisoning gradients.  With `_restore_hidden_states`, deleted positions use their pre-gate hidden states so the output layer produces valid (if low-confidence) scores everywhere.

---

### Delete Gate Mechanism

After DDiTBlock layer `delete_gate_layer` (default: 3), each token's hidden state is passed through `LayerNorm → Linear → ScaledSigmoid` to produce a gate value in `[sigmoid_mask_scale, 0]` (default `[-30, 0]`):

| Gate value | Meaning |
|---|---|
| ≈ 0 | Keep this token |
| ≈ −30 | Delete this token |

**Soft deletion** (default): gate value is added as a large negative bias to attention key scores in the *compressed phase* blocks (gate_layer+1 .. restore_gate_layer). Falls back from FlashAttention to `F.scaled_dot_product_attention` when the mask is active. After `restore_gate_layer` the mask is cleared and FlashAttention resumes on all L tokens.

**Hard deletion**: tokens where `gate < deletion_threshold` (default −15) are physically removed before downstream blocks. RoPE is recomputed for surviving positions. At `restore_gate_layer` (or just before the output layer if unset), `_restore_hidden_states` places the pre-gate hidden states back at deleted positions so the output layer produces valid scores for every position — **not zero logits**.

### Gate Variants

| `deletion_type` | Conditioning | Description |
|---|---|---|
| `scaled_sigmoid` | token + σ (default) | Main learnable gate — sigmoid scaled by `sigmoid_mask_scale` |
| `scaled_sigmoid` + `--no_gate_sigma_conditioned` | token only | Gate blind to noise level (ablation) |
| `log_sigmoid` | token + σ | Alternative learnable gate using log sigmoid |
| `random` | — | Baseline: delete random tokens at `random_deletion_probability` |
| `fixed` | — | Baseline: delete a fixed fraction `fixed_deletion_amount` of tokens |

### Key Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `delete_gate_layer` | 3 | DDiTBlock index after which the gate fires |
| `restore_gate_layer` | `None` | DDiTBlock index at which to restore full L. `None` = just before output layer |
| `deletion_type` | `scaled_sigmoid` | Gate variant (see table above) |
| `deletion_mode` | `soft` | `soft` or `hard` |
| `sigmoid_mask_scale` | −30.0 | Controls strength of the deletion signal |
| `deletion_threshold` | −15.0 | Threshold for hard deletion and deletion rate metrics |
| `gate_sigma_conditioned` | **True** | Condition gate on σ — essential for noise-level-aware deletion |
| `use_gumbel_noise` | False | Add Gumbel noise to gate logits during training |
| `deletion_loss_weight` | 0.1 | Weight of auxiliary deletion rate loss |
| `deletion_rate_schedule` | `constant` | Schedule for target deletion rate vs σ: `constant`, `linear_sigma`, `power_sigma` |
| `target_deletion_rate` | 0.3 | Target rate for `constant` schedule |
| `r_min` | 0.05 | Minimum deletion rate at σ≈0 (nearly clean tokens) |
| `r_max` | 0.5 | Maximum deletion rate at σ≈σ\_max (fully noised) |
| `deletion_rate_alpha` | 1.0 | Exponent for `power_sigma` schedule (1.0 = linear) |
| `sigma_max` | 20.0 | Must match SEDD `noise.sigma_max` |

---

## File Structure

```
mrdiffusion/
├── configuration_mrdiffusion.py   # MrDiffusionConfig dataclass
├── modeling_mrdiffusion.py        # MrSEDD model — delete gate classes + MrDDiTBlock
├── losses_mrdiffusion.py          # Score entropy loss + deletion rate regularisation
├── train_mrdiffusion.py           # Local training script (argparse, W&B, single GPU)
├── train_modal_mrdiffusion.py     # Modal serverless GPU training script
└── README.md
```

The base SEDD code lives in `../diffusion/Score-Entropy-Discrete-Diffusion/`. All `sys.path` manipulation is handled automatically — you do not need to set any environment variables.

---

## Environment Setup

MrDiffusion requires the SEDD environment (Python 3.9, CUDA 11.8, flash-attn).

```bash
# Create the environment from the SEDD spec
conda env create -f ../diffusion/Score-Entropy-Discrete-Diffusion/environment.yml
conda activate sedd

# Install wandb (not in original SEDD env)
pip install wandb
```

> The `mrbert` conda environment is missing `einops` and `flash-attn` and **cannot** be used for MrDiffusion.

---

## W&B Setup

```bash
# One-time login
wandb login

# Or set the API key as an environment variable
export WANDB_API_KEY=<your_key>
```

Metrics logged during training:

| Metric | Description |
|---|---|
| `train/loss` | Combined score entropy + deletion loss |
| `train/deletion_rate` | Fraction of tokens deleted at σ = 0.5 |
| `train/learning_rate` | Current LR (after warmup) |
| `eval/loss` | Eval score entropy loss |
| `eval/generative_perplexity` | GPT2-Large perplexity on generated samples |
| `samples` | W&B Table of generated text (at snapshot intervals) |
| `model/n_parameters` | Total parameter count |

---

## Local Training

All commands are run from the project root (`CS224N-project/`).

### Smoke test (CPU or single GPU, 200 steps)

```bash
conda activate sedd
python mrdiffusion-sedd/train_mrdiffusion.py \
    --max_steps 200 \
    --logging_steps 20 \
    --eval_steps 50 \
    --batch_size 4 \
    --output_dir ./mrdiffusion_smoke_test \
    --disable_wandb
```

### Baseline SEDD (no gate)

```bash
python mrdiffusion-sedd/train_mrdiffusion.py \
    --no_delete_gate \
    --output_dir ./mrdiffusion_baseline \
    --wandb_project mrdiffusion-sedd \
    --wandb_run_name sedd-baseline
```

### MrSEDD — Soft deletion (default)

```bash
python mrdiffusion-sedd/train_mrdiffusion.py \
    --delete_gate_layer 3 \
    --deletion_type scaled_sigmoid \
    --deletion_mode soft \
    --target_deletion_rate 0.3 \
    --deletion_loss_weight 0.1 \
    --output_dir ./mrdiffusion_soft_30pct \
    --wandb_project mrdiffusion-sedd \
    --wandb_run_name mrsedd-soft-layer3-30pct
```

### MrSEDD — Sigma-conditioned gate

```bash
python mrdiffusion-sedd/train_mrdiffusion.py \
    --deletion_mode soft \
    --gate_sigma_conditioned \
    --target_deletion_rate 0.3 \
    --output_dir ./mrdiffusion_sigma_conditioned \
    --wandb_run_name mrsedd-sigma-cond-soft-30pct
```

### MrSEDD — Hard deletion

```bash
python mrdiffusion-sedd/train_mrdiffusion.py \
    --deletion_mode hard \
    --delete_gate_layer 3 \
    --target_deletion_rate 0.3 \
    --output_dir ./mrdiffusion_hard_30pct \
    --wandb_run_name mrsedd-hard-layer3-30pct
```

### MrSEDD — Random deletion baseline (ablation)

```bash
python mrdiffusion-sedd/train_mrdiffusion.py \
    --deletion_type random \
    --random_deletion_probability 0.3 \
    --deletion_loss_weight 0.0 \
    --output_dir ./mrdiffusion_random_baseline \
    --wandb_run_name mrsedd-random-30pct
```

### Medium model

```bash
python mrdiffusion-sedd/train_mrdiffusion.py \
    --model_size medium \
    --batch_size 16 \
    --delete_gate_layer 6 \
    --target_deletion_rate 0.3 \
    --output_dir ./mrdiffusion_medium \
    --wandb_run_name mrsedd-medium-soft-layer6-30pct
```

### Separate gate learning rate

```bash
python mrdiffusion-sedd/train_mrdiffusion.py \
    --delete_gate_layer 3 \
    --delete_gate_lr 3e-3 \
    --target_deletion_rate 0.3 \
    --output_dir ./mrdiffusion_gate_lr \
    --wandb_run_name mrsedd-gate-lr-3e3
```

### Resume from checkpoint

Training automatically resumes from `<output_dir>/checkpoints-meta/checkpoint.pth` if it exists. Just re-run the same command.

---

## Modal (Serverless GPU)

### Prerequisites

```bash
pip install modal
modal token set --token-id <your_token_id> --token-secret <your_token_secret>

# Store your W&B API key as a Modal secret
modal secret create wandb-secret WANDB_API_KEY=<your_wandb_api_key>
```

### Smoke test (50 steps, A100)

```bash
modal run mrdiffusion-sedd/train_modal_mrdiffusion.py
```

### Baseline SEDD

```bash
modal run --detach mrdiffusion-sedd/train_modal_mrdiffusion.py::main \
    --no-delete-gate \
    --max-steps 500000
```

### MrSEDD — Soft deletion

```bash
modal run --detach mrdiffusion-sedd/train_modal_mrdiffusion.py::main \
    --deletion-type scaled_sigmoid \
    --deletion-mode soft \
    --target-deletion-rate 0.3 \
    --max-steps 500000 \
    --wandb-project mrdiffusion-sedd \
    --wandb-run-name mrsedd-soft-layer3-30pct
```

### MrSEDD — Hard deletion

```bash
modal run --detach mrdiffusion-sedd/train_modal_mrdiffusion.py::main \
    --deletion-mode hard \
    --target-deletion-rate 0.3 \
    --max-steps 500000 \
    --wandb-run-name mrsedd-hard-layer3-30pct
```

### MrSEDD — Sigma-conditioned gate

```bash
modal run --detach mrdiffusion-sedd/train_modal_mrdiffusion.py::main \
    --gate-sigma-conditioned \
    --target-deletion-rate 0.3 \
    --max-steps 500000 \
    --wandb-run-name mrsedd-sigma-cond-30pct
```

### Medium model

```bash
modal run --detach mrdiffusion-sedd/train_modal_mrdiffusion.py::main \
    --model-size medium \
    --batch-size 16 \
    --delete-gate-layer 6 \
    --target-deletion-rate 0.3 \
    --max-steps 1000000 \
    --wandb-run-name mrsedd-medium-soft-layer6-30pct
```

### Download checkpoints

```bash
# Download the final checkpoint of a named run
modal volume get mrdiffusion-sedd-checkpoints \
    mrsedd-soft-layer3-30pct/final \
    ./local_mrdiffusion_final

# Download all snapshots
modal volume get mrdiffusion-sedd-checkpoints \
    mrsedd-soft-layer3-30pct/checkpoints \
    ./local_checkpoints

# List all runs stored in the volume
modal volume ls mrdiffusion-sedd-checkpoints
```

---

## Evaluation

MrSEDD is evaluated identically to SEDD: generate sequences (Euler predictor, 128 steps) and measure perplexity with a frozen GPT2-Large model.

Generative perplexity is logged automatically to W&B at every snapshot. To run a standalone evaluation on a saved checkpoint, use the original SEDD sampling infrastructure:

```bash
cd ../diffusion/Score-Entropy-Discrete-Diffusion
python run_sample.py \
    --model_path ../../mrdiffusion_soft_30pct/final/checkpoint.pth \
    --steps 128
```

---

## Experiment Suggestions

| Experiment | Key flags |
|---|---|
| Ablation: constant vs scheduled deletion rate | `--deletion_rate_schedule constant` vs `linear_sigma` vs `power_sigma` |
| Ablation: deletion rate (constant schedule) | `--target_deletion_rate 0.1 / 0.3 / 0.5` |
| Ablation: r_min / r_max (scheduled) | `--r_min 0.0 --r_max 0.3` vs `--r_min 0.05 --r_max 0.5` |
| Ablation: restore layer | `--restore_gate_layer -1` (output only) vs `6` (mid-network) vs `9` (late) |
| Ablation: gate layer | `--delete_gate_layer 1 / 3 / 6 / 9` |
| Ablation: soft vs hard | `--deletion_mode soft` vs `--deletion_mode hard` |
| Ablation: sigma conditioning | default (`--gate_sigma_conditioned`) vs `--no_gate_sigma_conditioned` |
| Ablation: gate type | `--deletion_type scaled_sigmoid / random / fixed` |
| Baseline comparison | `--no_delete_gate` (vanilla SEDD) |

Track all runs in W&B under the `mrdiffusion` project to compare perplexity curves and deletion rate trajectories side-by-side.

---

## Why BD3-LM Is a Better Long-term Target

The guide identifies a fundamental asymmetry between SEDD and BD3-LM that makes BD3-LM more amenable to token deletion:

**SEDD (uniform / absorb noise):** Corrupted tokens take *random vocabulary values*.  Because each corrupted token is a different random token, they carry non-trivially different hidden states and are not obviously redundant to the gate — the gate must learn from scratch which tokens are worth keeping.

**BD3-LM (masking noise):** Corrupted tokens are all replaced with the single `[MASK]` token.  Within any noised block, all masked positions produce *identical embeddings* and carry no differential information.  A delete gate can trivially learn to collapse all masked tokens to a single representative, keeping only the unmasked (clean) context tokens at full resolution.  This creates a natural curriculum: at high mask rates (early reverse process) delete aggressively; at low mask rates (late reverse process) delete minimally — exactly the behaviour that `linear_sigma` / `power_sigma` schedules are designed to encourage.

**Recommended roadmap:**
1. Validate the architecture on SEDD (simpler codebase, no block-causal constraints).
2. Port to BD3-LM once the gate is stable: apply deletion *within* each BD3-LM block of L′ tokens, keeping cross-block attention at full resolution to preserve the KV-caching advantage.
3. The `gate_sigma_conditioned=True` default and sigma-dependent schedule already encode the noise-level awareness required for both models.

--------
Notes:

⏺ The pre-deletion mechanism is already correctly implemented. Here's the mapping:

  What the guide means by "pre-deletion QA setup":

  In MrBERT's QA task, hard deletion creates a problem — the span extraction head needs predictions for all L original token positions, but after hard deletion the sequence is only L′ tokens. The QA fix
  is to not zero-fill deleted positions — instead, use the hidden state each deleted token had just before the gate fired. This gives the output head something meaningful to work with at those positions.

  How mrdiffusion-sedd implements exactly this:

  line 729:  x_pre_gate = x           ← save hidden states just before gate fires
  ...
  line 756:  x = self._restore_hidden_states(x, x_pre_gate, keep_mask)

  _restore_hidden_states (line 618):
  - Kept positions → get the updated hidden state from the compressed-phase blocks
  - Deleted positions → get x_pre_gate (their representation before the gate)

  Then self.output_layer(x, c) at line 760 always sees a full-length [B, L, D] tensor, so every token position gets a valid score — exactly what the diffusion loss requires.

  The only difference from MrBERT's QA case is that in SEDD the motivation is the score entropy loss needing valid logits for all L positions, but the mechanism is identical. The implementation is
  correct as-is.

* the concrete scores in SEDD are mainly an interpretation of the network outputs for the loss computation.

  What the network actually outputs:
                                                                                                                                                                                                           
  MrSEDD.forward() → self.output_layer(x, c) → [B, L, vocab_size] logits                                                                                                                                   
                                                                                                                                                                                                           
  This is identical in shape and type to what a BERT MLM head produces. The transformer has no awareness of "scores" — it's just a linear projection from hidden states to vocab.                          
                                                                                                                                                                                                         
  Where the "score" interpretation happens — outside the model:                                                                                                                                            
                                                                                                                                                                                                         
  # model/utils.py — get_score_fn wraps the model output                                                                                                                                                   
  score = model_fn(x, sigma)          # raw [B, L, vocab_size] logits                                                                                                                                      
                                                                                                                                                                                                           
  if sampling:                                                                                                                                                                                             
      return score.exp()              # interpret as ratio p_t(x̂)/p_t(x) for sampling                                                                                                                     
  return score                        # interpret as log-ratio for loss                                                                                                                                    
                                                                                                                                                                                                           
  # losses.py                                                                                                                                                                                              
  log_score = log_score_fn(perturbed_batch, sigma)        # raw model output                                                                                                                               
  loss = graph.score_entropy(log_score, sigma, ...)       # interprets as log-scores                                                                                                                       
                                                                                                                                                                                                           
  The score_entropy loss and the Euler/τ-leaping sampler are what give the logits their "score" meaning. The model itself just produces numbers.                                                           
                                                                                                                                                                                                           
  Implication for the delete gate:                                                                                                                                                                         
                                                                                                                                                                                                         
  The guide's point is that this fact makes the delete gate straightforward — there's no special output structure to preserve. As long as MrSEDD.forward() returns [B, L, vocab_size] logits covering all L
   positions (which _restore_hidden_states ensures), the SEDD loss and sampling code work identically. The gate only touches intermediate hidden states; the output contract is unchanged.               
                                                                                                                                                                                                           
  Our implementation respects this exactly. 
--------