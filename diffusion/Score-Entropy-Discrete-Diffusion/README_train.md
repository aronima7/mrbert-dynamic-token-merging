# SEDD: Architecture, Training, and Inference

Score Entropy Discrete Diffusion (SEDD) is a discrete diffusion model for text. It learns to reverse a corruption process (forward diffusion) that gradually destroys a token sequence, and at inference time generates text by iteratively denoising from pure noise.

---

## Table of Contents

1. [Forward Diffusion (Corruption)](#1-forward-diffusion-corruption)
2. [Noise Schedule](#2-noise-schedule)
3. [Model Architecture (DDIT)](#3-model-architecture-ddit)
4. [Model Configs](#4-model-configs)
5. [Loss Function](#5-loss-function)
6. [Training Loop (run_train.py)](#6-training-loop-run_trainpy)
7. [Data: Train vs Eval Split](#7-data-train-vs-eval-split)
8. [Inference (Sampling)](#8-inference-sampling)
9. [Perplexity Metric](#9-perplexity-metric)
10. [Interpreting W&B Metrics](#10-interpreting-wb-metrics)
11. [Continued Pretraining](#11-continued-pretraining)

---

## 1. Forward Diffusion (Corruption)

SEDD defines a continuous-time Markov process over token sequences. Each token transitions independently according to a rate matrix **Q**, controlled by a noise level σ.

### Graph types (`graph_lib.py`)

Two corruption graphs are supported:

**Absorbing** (`graph.type=absorb`) — tokens are replaced by a single `[MASK]` token:
```
                  rate = 1
   token_i  ─────────────────►  [MASK]   (token vocab+1)
             prob = 1 - e^(-σ)

  [MASK] is absorbing: once masked, stays masked
```

**Uniform** (`graph.type=uniform`) — tokens are replaced uniformly at random:
```
               rate = 1/V
   token_i  ──────────────►  any token in vocab (size V)
             prob = 1 - e^(-σ)
```

At `σ=0` the sequence is clean; at `σ→∞` it is fully corrupted (all masked / fully random).

---

## 2. Noise Schedule

The noise schedule maps a continuous time `t ∈ [0, 1]` to a noise level σ(t) (`noise_lib.py`).

### LogLinear (`noise.type=loglinear`) — default, used with absorb

```
σ(t)  =  -log(1 - (1-ε)·t)         ε = 1e-3

dσ/dt =  (1-ε) / (1 - (1-ε)·t)
```

- `t=0`: σ≈0, sequence is clean
- `t=1`: σ→large, almost all tokens masked
- Chosen so that the masking probability `1 - e^(-σ) ≈ (1-ε)·t` grows approximately linearly in time

### Geometric (`noise.type=geometric`) — used with uniform

```
σ(t)  =  σ_min^(1-t) · σ_max^t       (σ_min=1e-4, σ_max=20)

dσ/dt =  σ(t) · log(σ_max / σ_min)
```

- Exponential interpolation between σ_min and σ_max

---

## 3. Model Architecture (DDIT)

The score model is a **D**iscrete **Di**ffusion **T**ransformer (DDIT) — a bidirectional transformer conditioned on the noise level σ. It outputs a log-score (log ratio) for each token position over the vocabulary.

```
Input: noisy token sequence  x_t  [B, L]  (integer token ids)
       noise level           σ    [B]

┌──────────────────────────────────────────────────────────────────┐
│                          SEDD forward()                          │
│                                                                  │
│   x_t [B, L] ──► EmbeddingLayer ──► x [B, L, H]                │
│                   (vocab_size → H)                               │
│                                                                  │
│   σ [B] ──► TimestepEmbedder ──► SiLU ──► c [B, cond_dim]      │
│              (sinusoidal → H → cond_dim)                         │
│                                                                  │
│   x ──► RotaryEmbedder ──► (cos, sin) positional encoding       │
│                                                                  │
│   ┌─────────────────────────────────────────────────────┐       │
│   │  DDiTBlock  ×  n_blocks                             │       │
│   │                                                     │       │
│   │  c ──► adaLN_modulation ──► shift, scale, gate      │       │
│   │         (cond_dim → 6H)        ×2 (attn + mlp)      │       │
│   │                                                     │       │
│   │  x ──► LayerNorm ──► modulate(shift,scale)          │       │
│   │     ──► QKV proj ──► Rotary ──► FlashAttn (bidir)   │       │
│   │     ──► gate_msa ──► residual add ──► x             │       │
│   │                                                     │       │
│   │  x ──► LayerNorm ──► modulate(shift,scale)          │       │
│   │     ──► MLP (H→4H, GELU, 4H→H)                     │       │
│   │     ──► gate_mlp ──► residual add ──► x             │       │
│   └─────────────────────────────────────────────────────┘       │
│                                                                  │
│   x ──► DDitFinalLayer ──► log_score [B, L, vocab_size]         │
│          (adaLN → LayerNorm → Linear)                            │
│                                                                  │
│   [absorb only] log_score -= log(e^σ - 1) + log(V)   (scale)   │
│   log_score[:, :, x_t] = 0   (zero out self-transitions)        │
└──────────────────────────────────────────────────────────────────┘

Output: log_score [B, L, vocab_size]
        log_score[b, i, j] ≈ log p(x0_i = j | x_t_i, σ) - log p_σ(x_t_i | x0_i=j)
```

### Key design choices

| Component | Detail |
|---|---|
| **Attention** | Bidirectional (non-causal, `causal=False`); uses FlashAttention for efficiency |
| **Positional encoding** | Rotary (RoPE), applied to Q and K inside every block |
| **Timestep conditioning** | Adaptive LayerNorm (adaLN) — σ controls shift, scale, and gate of every block |
| **adaLN init** | Modulation weights zero-initialized so each block starts as identity |
| **Final layer init** | Output projection zero-initialized so log_score starts at 0 |
| **Precision** | Transformer blocks run in bfloat16; LayerNorm in float32 |
| **σ scaling** | For absorb: `log_score -= log(e^σ - 1) + log(V-1)` keeps scores near 0 at init |

### DDiTBlock in detail

```
       ┌──────── c (noise condition) ────────────────────────┐
       │                                                     │
       │   adaLN_modulation (Linear: cond_dim → 6H)         │
       │   └─► shift_msa, scale_msa, gate_msa               │
       │       shift_mlp, scale_mlp, gate_mlp                │
       │                                                     │
x ─────┼──────────────────────────────────────────── x_skip │
       │   LayerNorm → modulate(shift_msa, scale_msa)        │
       │   → QKV Linear (H → 3H)                             │
       │   → Rotary positional encoding on Q, K              │
       │   → FlashAttention (bidirectional)                  │
       │   → attn_out Linear (H → H)                         │
       │   → gate_msa * output + x_skip ─────────────────► x│
       │                                                     │
x ─────┼──────────────────────────────────────────── x_skip │
       │   LayerNorm → modulate(shift_mlp, scale_mlp)        │
       │   → MLP: Linear(H→4H) → GELU → Linear(4H→H)        │
       │   → gate_mlp * output + x_skip ────────────────── x│
       └─────────────────────────────────────────────────────┘
```

---

## 4. Model Configs

Defined in `configs/model/`. Select with `model=small` or `model=medium`.

| Parameter | Small | Medium | Description |
|---|---|---|---|
| `hidden_size` (H) | 768 | 1024 | Token embedding / hidden dimension |
| `cond_dim` | 128 | 128 | Noise conditioning dimension |
| `length` (L) | 1024 | 1024 | Sequence length in tokens |
| `n_blocks` | 12 | 24 | Number of transformer blocks |
| `n_heads` | 12 | 16 | Attention heads (head_dim = H/n_heads = 64) |
| `scale_by_sigma` | True | True | Apply σ-scaling to output (absorb only) |
| `dropout` | 0.1 | 0.1 | Dropout rate in attention and MLP |
| **~Parameters** | ~110M | ~400M | Approximate parameter count |

Vocab size: 50,257 (GPT-2 BPE) + 1 `[MASK]` token for absorb = **50,258** total.

---

## 5. Loss Function

### Score entropy (`losses.py`, `graph_lib.py`)

SEDD trains to predict the **score**: the log ratio of the data distribution to the noisy marginal,

```
s_θ(x_t, σ)[j]  ≈  log p(x₀ = j | x_t, σ)  –  log p_σ(x_t | x₀ = j)
```

This is the log-likelihood ratio of each clean token j given the noisy observation.

The training objective is a **score entropy** loss, an f-divergence-based bound on the reverse KL:

```
L = E_{t ~ U[ε,1]}  [  dσ/dt  ·  Σ_i  H_SE( s_θ(x_t,σ), x₀_i, x_t_i )  ]
```

- `t` is sampled uniformly in `[ε, 1]`; σ and dσ/dt are computed from t
- `x_t` is sampled from the forward process: `graph.sample_transition(x₀, σ)`
- `H_SE` is the score entropy at position i — its exact form depends on the graph:

**Absorb graph** — loss is nonzero only at masked positions (`x_t = [MASK]`):
```
H_SE = exp(s[j≠MASK]).sum()            (positive term: pushes scores for wrong tokens down)
     - (1/esigm1) · s[x₀]             (negative term: pushes score for true token up)
     + (1/esigm1) · (log(1/esigm1)-1) (constant)

where esigm1 = e^σ - 1
```

**Uniform graph** — loss computed at all positions, comparing the predicted distribution over the whole vocabulary.

### At a glance: training vs inference loss

| | Training | Inference |
|---|---|---|
| **Loss computed?** | Yes — `get_step_fn(..., train=True)` | No — model runs `@torch.no_grad()` |
| **Gradients** | Backprop through score entropy loss | None |
| **Model weights** | Trained weights + gradient updates | EMA weights (copied to model) |
| **σ source** | Sampled uniformly: `t ~ U[ε, 1]`, σ = noise(t) | Scheduled: linspace from σ(1)→σ(ε) |
| **x_t source** | Sampled from forward process `p(x_t | x₀)` | Carried forward from previous step |
| **Objective** | Minimize score entropy | Use score to compute reverse transition rates |
| **Eval loss** | Also computed (EMA weights, no grad) | Not computed |

### Gradient accumulation

When `accum > 1`, the step function accumulates gradients over `accum` mini-batches before calling the optimizer. The effective batch size is `training.batch_size` (the total across all GPUs and accumulation steps):

```
per_GPU_batch = training.batch_size / (ngpus × accum)
```

The step counter (`state['step']`) increments only after a full accumulated update.

---

## 6. Training Loop (`run_train.py`)

```
  Training loop (while step < n_iters)
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  batch = next(train_iter)          # [B/ngpus, L] token ids   │
  │                                                                │
  │  ┌─ loss_fn(model, batch) ──────────────────────────────────┐ │
  │  │  t ~ U[ε, 1]                    # random timestep        │ │
  │  │  σ, dσ/dt = noise(t)            # noise level + rate     │ │
  │  │  x_t = graph.sample_transition(x₀, σ)  # corrupt         │ │
  │  │  log_score = model(x_t, σ)      # DDIT forward pass      │ │
  │  │  loss = (dσ/dt · score_entropy(log_score, σ, x_t, x₀))  │ │
  │  │          .sum(dim=-1).mean()                              │ │
  │  └──────────────────────────────────────────────────────────┘ │
  │                                                                │
  │  scaler.scale(loss).backward()     # mixed precision          │
  │                                                                │
  │  [every accum steps:]                                          │
  │    unscale → clip gradients (max_norm=1.0)                    │
  │    optimizer.step() → optimizer.zero_grad()                   │
  │    ema.update()                    # EMA of model weights     │
  │    state['step'] += 1                                          │
  │                                                                │
  │  [every log_freq=50 steps, rank 0:]                            │
  │    all_reduce(loss) → log to file + W&B: train/loss           │
  │                                                                │
  │  [every snapshot_freq_for_preemption=10k steps, rank 0:]      │
  │    save checkpoints-meta/checkpoint.pth   (for resuming)      │
  │                                                                │
  │  [every eval_freq=100 steps:]                                  │
  │    eval_batch = next(eval_iter)                                │
  │    eval_loss = loss_fn(EMA model, eval_batch)  # no grad      │
  │    all_reduce(eval_loss) → log: eval/loss                     │
  │                                                                │
  │  [every snapshot_freq=50k steps:]                              │
  │    save checkpoints/checkpoint_N.pth                          │
  │    generate samples with EMA model → write to samples/iter_N/ │
  │    compute GPT-2-Large perplexity on samples → log: eval/ppl  │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

### Optimizer

AdamW with linear warmup and gradient clipping:

```
lr(step) = lr_max × min(step / warmup, 1.0)

lr_max   = 3e-4
warmup   = 2500 steps
grad_clip = 1.0
β₁, β₂  = 0.9, 0.999
ε        = 1e-8
```

### EMA

An exponential moving average of model weights (`decay=0.9999`) is maintained throughout training. EMA weights are used for eval loss and all sample generation — they are not trained directly.

### Metrics summary

| Metric | Source | Frequency | What it measures |
|---|---|---|---|
| `train/loss` | Score entropy on training batch | every 50 steps | How well the model recovers corrupted training tokens |
| `eval/loss` | Score entropy on WikiText-103 batch | every 100 steps | Generalization of the score to unseen text |
| `eval/perplexity` | GPT-2-Large on generated samples | every 50,000 steps | Fluency of text the model generates |

---

## 7. Data: Train vs Eval Split

Only **two** splits are used — there is no test set. The task is unconditional generation, so there are no labelled examples to score against. `eval/perplexity` is computed on *generated* text judged by GPT-2-Large, not on held-out ground-truth sequences.

| | Training | Evaluation |
|---|---|---|
| **Dataset** | OpenWebText (`data.train=openwebtext`) | WikiText-103 (`data.valid=wikitext103`) |
| **HuggingFace split** | `"train"` | `"validation"` |
| **Tokenizer** | GPT-2 BPE (vocab 50,257) | same |
| **Preprocessing** | Tokenize → append EOS → concatenate → chunk to 1024 tokens | same |
| **Batch size** | `training.batch_size / (ngpus × accum)` per GPU | `eval.batch_size / (ngpus × accum)` per GPU |
| **Sampler** | DistributedSampler (shuffled each epoch) | DistributedSampler |
| **Used for** | Computing training loss, backprop | Computing eval loss (no grad, EMA weights) |

### OpenWebText (training)

OpenWebText is an open replication of OpenAI's WebText dataset (the corpus GPT-2 was trained on).

| Property | Value |
|---|---|
| **Source** | Web pages linked from Reddit posts with ≥3 upvotes |
| **Size** | ~38GB raw text, ~8M documents |
| **Tokens** | ~9B GPT-2 BPE tokens |
| **Domain** | Broad English web text — news, forums, blogs, Wikipedia, etc. |
| **HuggingFace ID** | `Skylion007/openwebtext` |
| **Split used** | `train` only |
| **License** | Unspecified (scraped web content) |

Preprocessing in `data.py`:
1. Tokenize each document with GPT-2 BPE
2. Append EOS token (`<|endoftext|>`, id 50256) to each document
3. Concatenate all token sequences into one long stream
4. Chunk into non-overlapping windows of 1024 tokens (remainder dropped)

### WikiText-103 (evaluation)

WikiText-103 is a standard language modelling benchmark derived from verified Wikipedia articles.

| Property | Value |
|---|---|
| **Source** | English Wikipedia (Good and Featured articles only) |
| **Size** | ~500MB, ~28K articles |
| **Tokens** | ~103M GPT-2 BPE tokens (train) / ~218K (validation) / ~246K (test) |
| **Domain** | Encyclopedic text — structured, formal prose |
| **HuggingFace ID** | `wikitext`, config `wikitext-103-raw-v1` |
| **Split used** | `validation` (for eval loss during training) |
| **License** | Creative Commons Attribution-ShareAlike |

A WikiText-specific detokenizer (`wt_detokenizer` in `data.py`) cleans spacing artefacts from the raw dataset (e.g. ` @-@ ` → `-`) before tokenization.

**Why different datasets for train and eval?** Using a held-out domain (Wikipedia) to evaluate a model trained on web text gives a cleaner measure of generalization — the eval loss cannot be inflated by memorization of training documents.

### Alternative datasets

| `data.train` / `data.valid` | Source |
|---|---|
| `openwebtext` | ~8GB web text (default train) |
| `wikitext103` | WikiText-103 (~100M tokens, default eval) |
| `wikitext2` | WikiText-2 (~2M tokens) |
| `ptb` | Penn Treebank |
| `lm1b` | One Billion Words |
| `text8` | Character-level (special handling in data loader) |

---

## 8. Inference (Sampling)

At inference, the model generates text by reversing the forward diffusion process via a **predictor-corrector** sampler (`sampling.py`).

```
  Inference: N denoising steps (t: 1 → ε)
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  x = graph.sample_limit(B, L)                                 │
  │      absorb:  all [MASK]                                       │
  │      uniform: all random tokens                               │
  │                                                                │
  │  timesteps = linspace(1, ε, steps+1)                          │
  │                                                                │
  │  for i in range(steps):                                        │
  │    t = timesteps[i]                                            │
  │    [optional: proj_fun(x) to enforce fixed tokens]            │
  │                                                                │
  │    ┌─ Predictor step ──────────────────────────────────────┐  │
  │    │  σ, dσ/dt = noise(t)                                  │  │
  │    │  log_score = EMA_model(x, σ)                          │  │
  │    │                                                        │  │
  │    │  Euler:    rev_rate = dt·dσ/dt·reverse_rate(x, score) │  │
  │    │            x = sample_rate(x, rev_rate)               │  │
  │    │                                                        │  │
  │    │  Analytic: stag_score = staggered_score(score, Δσ)    │  │
  │    │            probs = stag_score · transp_transition(x,Δσ)│ │
  │    │            x = sample_categorical(probs)              │  │
  │    └───────────────────────────────────────────────────────┘  │
  │                                                                │
  │  [if denoise=True (default):]                                  │
  │    final denoiser step at t=ε                                  │
  │    probs = stag_score(σ_ε) · transp_transition(x, σ_ε)        │
  │    x = sample_categorical(probs[..., :-1])  # exclude [MASK]  │
  │                                                                │
  │  return x  [B, L]  integer token ids                          │
  └────────────────────────────────────────────────────────────────┘
```

### Predictor types

| `sampling.predictor` | Description |
|---|---|
| `euler` | Discrete Euler method: samples a token transition from the reverse rate |
| `analytic` | Uses the analytic staggered score to compute exact transition probabilities (default in snapshots) |
| `none` | No predictor step (corrector only) |

### Conditional generation (`run_sample_cond.py`)

A `proj_fun` is applied at every step to pin specified token positions:

```
input_locs = [0, 1, 2, ..., len(prefix)-1] + [L-len(suffix), ..., L-1]
proj_fun(x): x[:, input_locs] = input_ids   # overwrite those positions
```

This constrains the sampler to produce text consistent with the prefix/suffix — an infilling task.

### Key sampling hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `steps` | 128 (training snapshots), 1024 (run_sample.py) | More steps = slower but higher quality |
| `sampling.predictor` | `euler` | Predictor algorithm |
| `sampling.noise_removal` | `True` | Apply denoising step at end |
| `--batch_size` | 1 | Sequences generated in parallel |

---

## 9. Perplexity Metric

### What it measures

Perplexity (PPL) measures how fluent the model's *generated* text is, as judged by an external language model (GPT-2-Large). It is **not** a measure of reconstruction accuracy — it evaluates whether the samples look like natural language.

```
PPL = exp( (1/(L-1)) · Σ_{i=1}^{L-1}  -log p_GPT2(x_{i+1} | x_1, ..., x_i) )
```

A lower perplexity means GPT-2-Large assigns higher probability to the generated tokens — i.e. the text is more fluent and natural. A perfectly fluent sentence drawn from the GPT-2 training distribution would have PPL near GPT-2-Large's own test perplexity (~15–20 on WikiText-103). Very high PPL (>1000) indicates incoherent or repetitive text.

### How it is computed (`run_train.py`)

At every `snapshot_freq` steps (default: 50,000):

```
1. Generate a batch of samples with the EMA model
   shape: (batch_size // (ngpus × accum), 1024)

2. Load GPT-2-Large (774M params) onto the same GPU

3. For each sub-batch of eval.perplexity_batch_size=32 sequences:
   loss, logits = gpt2_large(samples, labels=samples)
   logits = logits.transpose(-1, -2)          # [B, V, L]
   per_seq_ppl = cross_entropy(logits[..., :-1], samples[..., 1:],
                               reduction="none")
                 .mean(dim=-1).exp()           # [B]
   batch_ppl = per_seq_ppl.mean()

4. Average over all sub-batches → total_perplexity

5. all_reduce across GPUs → log to W&B: eval/perplexity
   (GPT-2-Large is then deleted to free GPU memory)
```

### Why GPT-2-Large and not the SEDD model itself

SEDD is a **score model**, not an autoregressive model — it does not directly define `p(x)` as a product of conditionals. Computing the exact likelihood of a sequence under SEDD requires numerical integration over the diffusion trajectory, which is expensive. Using an external autoregressive model (GPT-2-Large) as a proxy is a standard approach in discrete diffusion literature (used in the original SEDD paper).

### Relationship to train/eval loss

| Metric | What it reflects | When it improves |
|---|---|---|
| `train/loss` | Score estimation accuracy on training data | Model learns to denoise training tokens |
| `eval/loss` | Score estimation accuracy on WikiText-103 | Model generalizes to unseen text |
| `eval/perplexity` | Fluency of *generated* text | Score estimation translates into coherent generation |

The three metrics are related but not identical. A model can have low train/eval loss but still produce repetitive or incoherent samples if the sampling procedure amplifies small errors in the score. Perplexity is the most direct measure of end-to-end generation quality.

### Typical values

From the SEDD paper (medium model, absorb+loglinear, trained on OpenWebText):

| Model | Generative PPL (GPT-2-Large) |
|---|---|
| GPT-2 (117M, AR baseline) | ~18 |
| SEDD-small | ~50–80 |
| SEDD-medium | ~25–35 |
| Random text | >10,000 |

---

## 10. Interpreting W&B Metrics

### The three metrics and what healthy training looks like

```
  Steps →
  0        2.5k      50k       100k      500k      1.3M
  |warmup  |         |         |         |         |

  train/loss
  ████▇▇▆▆▅▅▄▄▄▃▃▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂   ← should decrease steadily

  eval/loss
  ████▇▇▆▆▅▅▄▄▄▃▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂   ← should track train/loss closely

  eval/perplexity  (logged every 50k steps only)
  ·         ·██   ·▇▇   ·▅▅   ·▄▄   ·▃▃       ← should fall as training progresses
```

### `train/loss` — score entropy on training batches

Logged every 50 steps. This is the primary learning signal.

| What you see | Likely cause |
|---|---|
| Steady decrease over first ~50k steps | Normal — model is learning the score |
| Loss spikes then recovers | Normal — large learning rate or hard batch; gradient clipping handles this |
| Loss drops then plateaus for many steps | Normal after initial learning; ensure LR warmup completed |
| Loss increases or diverges | LR too high, gradient explosion, or data issue — check grad norms |
| Loss is constant from step 0 | Model not training — check optimizer, frozen params, or zero gradients |

Early in training (steps < 2,500), the learning rate is ramping up linearly (`lr = lr_max × step/warmup`), so the loss may decrease slowly at first. After warmup, loss should drop more rapidly.

### `eval/loss` — score entropy on WikiText-103 validation

Logged every 100 steps using EMA weights on the validation set (never used for gradient updates).

| What you see | Likely cause |
|---|---|
| `eval/loss` ≈ `train/loss` throughout | Good generalization — model is not overfitting |
| `eval/loss` > `train/loss` by a large margin | Overfitting to OpenWebText or domain gap (expected to some degree — different corpus) |
| `eval/loss` decreasing while `train/loss` plateaus | Normal — EMA weights often generalize better than live weights |
| `eval/loss` increasing while `train/loss` decreasing | Overfitting — consider reducing model size or adding regularization |

A small gap between train and eval loss is expected because the datasets are different (web text vs Wikipedia). What matters most is that eval/loss is also trending down over time.

### `eval/perplexity` — GPT-2-Large on generated samples

Logged every 50,000 steps. This is the end-to-end quality signal.

| What you see | Likely cause |
|---|---|
| PPL > 5,000 early in training | Normal — model generates near-random text before learning the score |
| PPL falling from >1,000 to <200 over first 200k steps | Normal convergence trajectory |
| PPL plateaus above ~200 | May need more training steps, or sampling `--steps` is too low (use 1024 for best quality) |
| PPL plateaus around 50–80 (small) / 25–35 (medium) | Expected final performance per the paper |
| PPL increases after previously decreasing | Rare — could indicate training instability; check `train/loss` for spikes |

**Note:** perplexity is sensitive to the number of sampling steps. The training loop uses `sampling.steps=128` for speed; running `run_sample.py` with `--steps 1024` will give lower (better) PPL on the same checkpoint.

### How the three metrics relate

```
  train/loss ──► measures: does the model learn to estimate scores?
       │
       └──► if yes: eval/loss ──► measures: does it generalize to unseen text?
                        │
                        └──► if yes: eval/perplexity ──► measures: does good score
                                                          estimation produce fluent text?
```

All three should decrease together during healthy training. If `train/loss` and `eval/loss` are low but `eval/perplexity` remains high, the bottleneck is in the sampling procedure (too few steps, wrong predictor), not the model.

### Common patterns to watch for

| Pattern | Interpretation |
|---|---|
| `eval/loss` diverges upward after step ~100k | Possible overfitting or LR schedule issue |
| `train/loss` drops fast in first 10k steps then slows | Normal — easy structure learned first, fine-grained patterns take longer |
| Large gap between `eval/perplexity` at 128 vs 1024 steps | Normal — more denoising steps always helps; 128 is a training-time proxy |
| `eval/perplexity` not logged for many checkpoints | `eval.perplexity=True` must be set and `snapshot_sampling=True`; GPT-2-Large must fit in GPU memory alongside SEDD |

---

## 11. Continued Pretraining

### What it is

Instead of training from random initialization on OpenWebText for 1.3M steps, **continued pretraining** starts from published pretrained weights (`louaaron/sedd-small` or `louaaron/sedd-medium`) and continues training from there. The model already generates fluent text; continued pretraining teaches it new behavior (e.g. a delete gate or modified architecture) at a fraction of the cost.

### Training from scratch vs continued pretraining

| | From scratch | Continued pretraining |
|---|---|---|
| **Starting weights** | Random (Kaiming uniform) | Pretrained (fluent text generation) |
| **Starting EMA** | Initialized to random weights | Initialized to pretrained weights |
| **Optimizer state** | Fresh (step 0) | Fresh (step 0) |
| **Steps needed** | ~1,300,000 | ~20,000–500,000 |
| **Estimated cost** | ~$150–$300 | ~$4–$100 |
| **Initial `eval/perplexity`** | >10,000 | ~50–80 (small) |
| **Use case** | Reproducing the paper | Adapting SEDD with new components |

### How the checkpoint is initialized (`init_from_pretrained.py`)

```
louaaron/sedd-small  (HuggingFace)
         │
         ▼
SEDD.from_pretrained()          ← loads model weights
         │
         ├──► model.state_dict()          saved to checkpoint['model']
         │
         ├──► ExponentialMovingAverage(   saved to checkpoint['ema']
         │        model.parameters(),     shadow_params = pretrained weights
         │        decay=0.9999)
         │
         ├──► fresh AdamW(               saved to checkpoint['optimizer']
         │        model + noise params,   state = {} (no accumulated momentum)
         │        lr=3e-4)
         │
         └──► step = 0                   saved to checkpoint['step']
                  │
                  ▼
         checkpoints-meta/checkpoint.pth
                  │
                  ▼
         restore_checkpoint() in run_train.py picks it up automatically
```

The EMA shadow parameters are initialized to the pretrained weights, so the EMA model also starts from a good state — eval loss and sample quality are good from step 0.

### What to watch in W&B during continued pretraining

| Metric | Expected behavior |
|---|---|
| `train/loss` | Should start at a reasonable value (not near zero, not >10) and decrease further |
| `eval/loss` | Should start close to `train/loss` and track it |
| `eval/perplexity` (first snapshot at step 50k) | Should start around the pretrained model's PPL (~50–80 for small) |

If `train/loss` starts very high (>5) something went wrong with weight loading. If it starts near zero and doesn't move, the learning rate may be too low or the weights are frozen.

----------------
Some important points:

⏺ From data.py and config.yaml, the default training setup is:

  - Task: unconditional language modelling (no labels, no classification — the model learns to denoise corrupted token sequences back to fluent text)
  - Training data: https://huggingface.co/datasets/Skylion007/openwebtext — a large English web text corpus (~8GB, ~8M documents scraped from URLs shared on Reddit with ≥3 upvotes)
  - Eval data: WikiText-103 (validation split) — used only to compute eval/loss during training; never seen during gradient updates
  - Tokenizer: GPT-2 BPE, vocab size 50,257
  - Sequence length: 1024 tokens (documents are concatenated and chunked)

⏺ In the context of SEDD, sampling means generating new text from the model.

  It's the inference-time process of starting from a fully corrupted sequence (all [MASK] tokens for absorb, or all random tokens for uniform) and iteratively denoising it over N steps until you have a  
  coherent 1024-token text sequence.
                                                                                                                                                                                                           
  [MASK][MASK][MASK]...[MASK]    ← start (pure noise, t=1)                                                                                                                                                 
         ↓  ↓  ↓   (N denoising steps)                                                                                                                                                                     
  "The quick brown fox jumps..."  ← output (t≈0)                                                                                                                                                           
                                                                                                                                                                                                           
  It's called "sampling" (rather than "generation" or "decoding") because each denoising step is stochastic — the model outputs a probability distribution over tokens at each position, and a token is    
  randomly drawn from that distribution. Running the same model twice with the same input will produce different outputs.                                                                                  
                                                                                                                                                                                                           
  This contrasts with autoregressive models like GPT-2, which generate token-by-token left-to-right. SEDD refines the entire sequence in parallel across all positions simultaneously at each step.  

⏺ From configs/model/small.yaml and configs/model/medium.yaml:

  ┌────────────────────┬───────┬────────┐                                                                                                                                                                  
  │                    │ Small │ Medium │
  ├────────────────────┼───────┼────────┤                                                                                                                                                                  
  │ Hidden size        │ 768   │ 1024   │                                                                                                                                                                
  ├────────────────────┼───────┼────────┤                                                                                                                                                                  
  │ Transformer blocks │ 12    │ 24     │                                                                                                                                                                  
  ├────────────────────┼───────┼────────┤                                                                                                                                                                  
  │ Attention heads    │ 12    │ 16     │                                                                                                                                                                  
  ├────────────────────┼───────┼────────┤                                                                                                                                                                  
  │ Head dimension     │ 64    │ 64     │                                                                                                                                                                
  ├────────────────────┼───────┼────────┤                                                                                                                                                                  
  │ Conditioning dim   │ 128   │ 128    │                                                                                                                                                                
  ├────────────────────┼───────┼────────┤                                                                                                                                                                  
  │ Sequence length    │ 1024  │ 1024   │                                                                                                                                                                
  ├────────────────────┼───────┼────────┤                                                                                                                                                                  
  │ Dropout            │ 0.1   │ 0.1    │                                                                                                                                                                
  ├────────────────────┼───────┼────────┤                                                                                                                                                                  
  │ ~Parameters        │ ~110M │ ~400M  │                                                                                                                                                                
  └────────────────────┴───────┴────────┘                                                                                                                                                                  
                                                                                                                                                                                                         
  Both have the same architecture (DDIT) — medium is just wider (768→1024 hidden) and twice as deep (12→24 blocks).                                                                                        
                                                                                                                                                                                                         
  Practical differences on Modal:                                                                                                                                                                          
  - Small fits comfortably on a single A100 80GB with batch_size=32–64                                                                                                                                   
  - Medium needs accum=2 per the README (training.accum=2) to fit at the same effective batch size                                                                                                         
  - Medium trains ~3–4× slower per step                                                                                                                                                                  
  - Medium produces noticeably better text (PPL ~25–35 vs ~50–80 for small)                                                                                                                                
                                                                                                                                                                                                           
  The pretrained HuggingFace models are louaaron/sedd-small and louaaron/sedd-medium.

⏺ Perplexity measures how "surprised" a language model is by a piece of text — lower means the text looks more natural and fluent to the model.

  Intuition                                                                                                                                                                                                
   
  Given a sequence of tokens, a language model assigns a probability to each token given the previous ones. Perplexity is the geometric mean of the inverse of those probabilities:                        
                  
  PPL = exp( average negative log-likelihood per token )                                                                                                                                                   
      = exp( -(1/L) · Σ log p(token_i | token_1...token_{i-1}) )                                                                                                                                           
                                                                                                                                                                                                           
  - A model that always predicts the next token perfectly → PPL = 1                                                                                                                                        
  - A model that is completely random over 50,257 tokens → PPL = 50,257                                                                                                                                    
  - GPT-2-Large on WikiText-103 → PPL ≈ 15–20                                                                                                                                                              
                                                                                                                                                                                                           
  How it's computed in SEDD (run_train.py)                                                                                                                                                                 
                                                                                                                                                                                                           
  SEDD can't compute its own perplexity directly (it's not autoregressive — it has no p(token | previous tokens)). Instead, GPT-2-Large is used as an external judge:                                      
                  
  1. SEDD generates a batch of text samples (1024 tokens each)                                                                                                                                             
                                                                                                                                                                                                           
  2. GPT-2-Large scores each sample:                                                                                                                                                                       
     loss, logits = gpt2_large(sample, labels=sample)                                                                                                                                                      
     logits = logits.transpose(-1, -2)          # [B, vocab, L]                                                                                                                                            
                                                                                                                                                                                                           
  3. Per-token cross-entropy → per-sequence PPL:                                                                                                                                                           
     ppl = cross_entropy(logits[..., :-1], sample[..., 1:],                                                                                                                                                
                         reduction="none")                                                                                                                                                                 
           .mean(dim=-1).exp()                  # one PPL per sequence                                                                                                                                     
                                                                                                                                                                                                           
  4. Average over all sequences and GPUs → eval/perplexity                                                                                                                                                 
                                                                                                                                                                                                           
  So the question being answered is: does GPT-2-Large find SEDD's generated text plausible?                                                                                                                
                  
  Reference values                                                                                                                                                                                         
                  
  ┌───────────────────────┬─────────────────┐                                                                                                                                                              
  │      Text source      │ GPT-2-Large PPL │
  ├───────────────────────┼─────────────────┤                                                                                                                                                              
  │ Random tokens         │ >10,000         │
  ├───────────────────────┼─────────────────┤                                                                                                                                                              
  │ SEDD-small (trained)  │ ~50–80          │                                                                                                                                                              
  ├───────────────────────┼─────────────────┤                                                                                                                                                              
  │ SEDD-medium (trained) │ ~25–35          │                                                                                                                                                              
  ├───────────────────────┼─────────────────┤                                                                                                                                                              
  │ GPT-2 generated text  │ ~18             │
  ├───────────────────────┼─────────────────┤                                                                                                                                                              
  │ Real Wikipedia text   │ ~15–20          │
  └───────────────────────┴─────────────────┘      

❯ what is the estimated training cost for modal run --detach train_modal.py::train_main?                                                                                                                   

⏺ Rough estimate, with caveats:

  Modal A100 80GB pricing

  ~$3.70/hour (check current pricing at modal.com/pricing as this changes).

  Estimated training time (small model, single A100, default settings)

  ┌───────────────────────────────────────────────┬─────────────────────────────┐
  │                    Config                     │          Estimate           │
  ├───────────────────────────────────────────────┼─────────────────────────────┤
  │ Steps                                         │ 1,300,000 (default n_iters) │
  ├───────────────────────────────────────────────┼─────────────────────────────┤
  │ Batch size                                    │ 32 (Modal default)          │
  ├───────────────────────────────────────────────┼─────────────────────────────┤
  │ Step time (small model, bfloat16, flash-attn) │ ~0.1–0.2 sec/step           │
  ├───────────────────────────────────────────────┼─────────────────────────────┤
  │ Total wall-clock time                         │ ~36–72 hours                │
  ├───────────────────────────────────────────────┼─────────────────────────────┤
  │ Estimated cost                                │ ~$133–$266                  │
  └───────────────────────────────────────────────┴─────────────────────────────┘

  Important caveats

  The 24-hour timeout will interrupt the run. The Modal function has timeout=3600 * 24. A full 1.3M step run needs 2–3 resumed runs:
  modal run --detach train_modal.py::train_main --run-name sedd-run --resume
  Each resumed run picks up from checkpoints-meta/checkpoint.pth (saved every 10k steps).

  For a cheaper test run, reduce n_iters:
  # ~1 hour, ~$4
  modal run --detach train_modal.py::train_main --n-iters 20000

  # ~6 hours, ~$22 — enough to see loss converging
  modal run --detach train_modal.py::train_main --n-iters 100000

  Medium model would be roughly 3–4× more expensive (~$400–$800 for a full run).

⏺ 3 containers because train_modal.py defines 3 separate @app.function decorated functions — Modal registers all of them when the app is deployed, even if you only called one:                            
                                                                                                                                                                                                           
  - train — the one actually running (green dot, 1 container active)                                                                                                                                       
  - sample — unconditional text generation (idle, grey)                                                                                                                                                    
  - sample_cond — conditional text generation (idle, grey)                                                                                                                                                 
                                                                                                                                                                                                           
  sample_cond is prefix/suffix infilling: you give it a fixed start and/or end, and it generates the middle. Example from the code:                                                                        
  modal run train_modal.py::sample_cond_main \                                                                                                                                                             
    --prefix "Once upon a time" \                                                                                                                                                                          
    --suffix "The End."                                                                                                                                                                                    
  SEDD can do this naturally because the absorb (mask) graph allows conditioning on observed tokens — you fix the prefix/suffix tokens and only denoise the masked middle region.                          
                                                                                                                                                                                                           
  The sample and sample_cond containers show as idle (grey circle) because Modal pre-warms/registers them but they have no active inputs right now. They won't incur GPU costs unless you actually call    
  them. 
----------------