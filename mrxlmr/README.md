# MrXLMR — XLM-RoBERTa with MrT5-Style Delete Gates

MrXLMR adapts the **token deletion gate** from the [MrT5 paper](https://arxiv.org/pdf/2410.20771) (Kallini et al., 2024) to the XLM-RoBERTa architecture. A lightweight learned gate fires after a fixed encoder layer and assigns each token a scalar score; low-scoring tokens are suppressed from all subsequent attention computation (**soft deletion**) or physically removed from the sequence (**hard deletion**), trading a small accuracy cost for substantial compute reduction.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Delete Gate Mechanics](#2-delete-gate-mechanics)
3. [XLM-R vs BERT Specifics](#3-xlm-r-vs-bert-specifics)
4. [Soft vs Hard Deletion](#4-soft-vs-hard-deletion)
5. [Pre-Deletion Blending](#5-pre-deletion-blending)
6. [Training Objective and PI Controller](#6-training-objective-and-pi-controller)
7. [Code Structure and Walkthrough](#7-code-structure-and-walkthrough)
8. [Token-Level Examples by Task](#8-token-level-examples-by-task)
9. [Configuration Reference](#9-configuration-reference)
10. [Usage](#10-usage)

---

## 1. Architecture Overview

### Block Diagram

```
Input Text
    │
    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  XLM-R Tokenizer (SentencePiece, vocab=250K)                          │
│  "The cat sat" → [<s>, ▁The, ▁cat, ▁sat, </s>]                      │
│                  [  0,    6,  1234,  432,    2 ]  ← token IDs        │
└───────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  RobertaEmbeddings                                                    │
│  word_embeddings + position_embeddings + token_type_embeddings        │
│  (batch, seq_len, 768)                                                │
└───────────────────────────────────────────────────────────────────────┘
    │
    ▼  Layers 0, 1, 2  (standard XLM-R attention + FFN)
┌───────────────────────────────────────────────────────────────────────┐
│  MrXLMRLayer × 3      [no gate]                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  MrXLMRSelfAttention  (softmax1 optional)                       │  │
│  │  RobertaSelfOutput                                              │  │
│  │  RobertaIntermediate + RobertaOutput  (FFN)                     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
    │  hidden_states: (batch, seq_len, 768)
    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║  MrXLMRLayer 3  ◄── DELETE GATE FIRES HERE                          ║
║                                                                       ║
║  ① Standard attention + FFN (same as above)                          ║
║                                                                       ║
║  ② SigmoidDeleteGate                                                 ║
║     hidden_states  → LayerNorm → Linear(768→1) → ScaledSigmoid(-30) ║
║     ┌──────────────────────────────────────────────────────────────┐ ║
║     │  gate_values ∈ [-30, 0]  per token                          │ ║
║     │  gate ≈  0.0  →  KEEP   (threshold > -15)                   │ ║
║     │  gate ≈ -30.0 →  DELETE (threshold < -15)                   │ ║
║     │                                                              │ ║
║     │  Forced rules:                                               │ ║
║     │    position 0 (<s>)    → gate = 0.0  always KEPT            │ ║
║     │    token_id == 2 (</s>) → gate = 0.0  always KEPT           │ ║
║     │    token_id == 1 (<pad>)→ gate = -30.0  always DELETED      │ ║
║     └──────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════╝
    │  delete_gate_mask: (batch, seq_len, 1)
    │  [stored, propagated to all layers 4–11]
    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  MrXLMRLayer × 8   (layers 4–11)     [gate mask applied]             │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Soft mode:  gate_mask added to attention scores as bias        │  │
│  │    attention_scores += gate_mask  (≈ -30 for deleted tokens)    │  │
│  │    → deleted tokens receive near-zero attention weight          │  │
│  │                                                                 │  │
│  │  Hard mode:  tokens physically removed by scatter-gather        │  │
│  │    sequence_length shrinks from seq_len to kept_len             │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
    │  sequence_output: (batch, kept_len or seq_len, 768)
    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Optional: Pre-Deletion Blend                                         │
│  Deleted tokens receive interpolated hidden state (pre-gate + final)  │
└───────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────┬──────────────────┬──────────────┬──────────────────────────┐
│  MLM     │ Seq. Class.      │ Token Class. │ Question Answering        │
│ RobertaL │ Linear(768→num_  │ Linear(768→  │ Linear(768→2)            │
│ MHead    │ labels) on <s>   │ num_labels)  │ (start/end logits)        │
│          │ pooled output    │ per token    │                           │
└──────────┴──────────────────┴──────────────┴──────────────────────────┘
```

### Key Dimensions (xlm-roberta-base)

| Parameter | Value |
|---|---|
| Vocabulary size | 250,002 |
| Hidden size (`d_model`) | 768 |
| FFN size (`d_ff`) | 3,072 |
| Attention heads | 12 |
| Encoder layers | 12 |
| Delete gate at layer | 3 (default, 0-indexed) |
| Gate output range | [-30, 0] |
| Deletion threshold | -15.0 |
| Total parameters | ~278M (with classification head) |

---

## 2. Delete Gate Mechanics

### Mathematical Formulation

The delete gate is a single linear layer followed by a scaled sigmoid:

```
h_L  = hidden state at gate layer L    (batch, seq, 768)
h_LN = LayerNorm(h_L)                  (batch, seq, 768)
z    = Linear(h_LN)                    (batch, seq, 1)     ← learned logit
g    = sigmoid_mask_scale × sigmoid(-z)                    ← gate value

where sigmoid_mask_scale = -30.0 (default)
```

Substituting the sigmoid:

```
g = -30 × σ(-z) = -30 / (1 + exp(z))
```

This maps any logit `z` to a gate value in **(-30, 0)**:

| Logit `z` | gate `g` | Interpretation |
|---|---|---|
| +∞ | ≈ 0.0 | strongly KEPT |
| 10.0 | ≈ -0.001 | KEPT (initial bias) |
| 0.0 | -15.0 | boundary (threshold) |
| -10.0 | ≈ -29.9 | DELETED |
| -∞ | -30.0 | fully DELETED |

**Initialization**: bias = +10.0, weight_std = 0.001 — so all tokens start with `g ≈ -0.001 ≈ 0` (kept). The model must learn to push logits negative to delete tokens. This initialization is applied **after** `post_init()` to avoid being overwritten by HuggingFace's standard initialization.

### Gate Code Path

```
modeling_mrxlmr.py : SigmoidDeleteGate.forward()          ← lines 164–200
                       ScaledSigmoid.forward()             ← lines 131–132
                         → g = sigmoid_mask_scale × sigmoid(-logit)
                       Force <s> position 0: g[:, 0, :] = 0.0
                       Force </s> token_id=2: g = 0.0
                       Force <pad> token_id=1: g = -30.0
```

### Optional Gumbel Noise (Training)

When `use_gumbel_noise=True`, Gumbel-distributed noise is added to logits during training:

```python
# modeling_mrxlmr.py : SigmoidDeleteGate.forward(), line 175–176
if self.training and self.use_gumbel_noise:
    gumbel_noise = gumbel_noise_like(delete_gate_logits)
    delete_gate_logits = delete_gate_logits + gumbel_noise
```

This provides stochastic exploration so the gate doesn't collapse to always deleting or keeping the same subset of tokens.

---

## 3. XLM-R vs BERT Specifics

XLM-R uses SentencePiece tokenization (trained on 100 languages, 250K vocabulary) instead of WordPiece. The subword convention is **inverted** compared to BERT:

| | BERT (WordPiece) | XLM-R (SentencePiece) |
|---|---|---|
| Word-initial token | `the`, `cat` (no prefix) | `▁the`, `▁cat` (▁ prefix) |
| Continuation token | `##ing`, `##tion` | `ing`, `tion` (no prefix) |
| CLS token | `[CLS]` (id=101) | `<s>` (id=0) |
| SEP token | `[SEP]` (id=102) | `</s>` (id=2) |
| PAD token | `[PAD]` (id=0) | `<pad>` (id=1) |
| MASK token | `[MASK]` (id=103) | `<mask>` (id=250001) |

The gate protects CLS/SEP and always deletes PAD using these XLM-R-specific IDs:

```python
# modeling_mrxlmr.py : lines 46–48
XLM_R_CLS_TOKEN_ID = 0   # <s>
XLM_R_PAD_TOKEN_ID = 1   # <pad>
XLM_R_SEP_TOKEN_ID = 2   # </s>
```

**Weight loading**: The base model attribute is named `roberta` (not `bert`) to match the pretrained XLM-R weight key layout (`roberta.embeddings.*`, `roberta.encoder.layer.*.attention.*`), enabling correct `from_pretrained()` loading:

```python
# modeling_mrxlmr.py : MrXLMRForSequenceClassification.__init__()
self.roberta = MrXLMRModel(config)    # ← must be "roberta", not "bert"
```

**SNLI pair format**: XLM-R encodes sentence pairs with two separator tokens:

```
<s> premise </s> </s> hypothesis </s>
```

(BERT uses `[CLS] premise [SEP] hypothesis [SEP]` with one separator between segments.)

---

## 4. Soft vs Hard Deletion

### Soft Deletion (default, used during training)

The gate mask is added as a **bias to attention logits** in every layer after the gate:

```python
# modeling_mrxlmr.py : MrXLMRSelfAttention.forward(), lines 360–362
if delete_gate_mask is not None:
    delete_gate_mask_expanded = delete_gate_mask.squeeze(-1).unsqueeze(1).unsqueeze(1)
    attention_scores = attention_scores + delete_gate_mask_expanded
```

For a deleted token with `gate ≈ -30`, after softmax the attention weight becomes:

```
softmax(..., -30, ...) = exp(-30) / sum(exp) ≈ 9e-14  (near-zero)
```

Deleted tokens are **invisible to all subsequent attention** but still occupy memory. The sequence length does not change. This makes the operation differentiable, enabling end-to-end gradient flow through the gate.

### Hard Deletion (inference, optional training)

Tokens physically removed from the tensor via scatter-gather:

```python
# modeling_mrxlmr.py : MrXLMRLayer._get_new_positions_and_mask(), lines 448–476
keep_this = delete_gate_mask > threshold           # boolean keep mask
target_pos = cumsum(keep_this) - 1                # new positions for kept tokens
new_len    = max(target_pos) + 1                  # shortened sequence
hidden_states = gather(hidden_states, target_pos) # (batch, new_len, 768)
```

Hard deletion produces **real runtime speedup** (fewer tokens → fewer attention MACs in layers 4–11) but is non-differentiable, so it is not used as the default training mode.

During training it can be applied stochastically with `hard_delete_train_prob` (default 0.0) to regularise the model for inference-time hard deletion.

**Compute savings formula** (layers after gate L, with keep ratio `k`):

```
MACs_saved = (n_after_layers / n_total_layers) × (1 - k²)

Example: gate at layer 3, 30% deletion (k=0.70), 12 layers total:
  n_after = 8,  n_total = 12
  Attention MACs_saved = (8/12) × (1 - 0.49) = 0.34  →  34% savings
```

---

## 5. Pre-Deletion Blending

When `use_pre_deletion_blend=True` (default), deleted tokens receive a **weighted blend** of their pre-gate hidden state and their corrupted final-layer representation:

```python
# modeling_mrxlmr.py : MrXLMRModel._blend_pre_deletion(), lines 828–831
deletion_weight = clamp(-gate_mask / abs(sigmoid_mask_scale), 0.0, 1.0)
output = (1 - deletion_weight) × final_hidden + deletion_weight × pre_deletion_hidden
```

The weight schedule:

| gate value | deletion_weight | Hidden state used |
|---|---|---|
| 0.0 (kept) | 0.0 | final-layer representation |
| -15.0 (boundary) | 0.5 | 50/50 blend |
| -30.0 (deleted) | 1.0 | pre-gate representation |

**Why this matters for QA**: In question answering, the answer span tokens may be deleted by the gate. Without blending, those token positions would carry corrupted representations (they attended to almost nothing for 8 layers). Blending recovers a meaningful hidden state for the span head to decode. The encoder saves compute by treating the span as deleted, but the head still sees a useful representation.

The pre-deletion hidden state is saved just before layer 3 fires:

```python
# modeling_mrxlmr.py : MrXLMREncoder.forward(), lines 609–612
if layer_module.has_delete_gate and use_pre_deletion_blend:
    pre_deletion_hidden = hidden_states   # saved before gate layer runs
```

---

## 6. Training Objective and PI Controller

### Combined Loss

```
L_total = L_task  +  α × L_deletion
```

where:

- `L_task` = task loss (cross-entropy for classification/MLM, span CE for QA)
- `L_deletion` = mean gate value over non-pad tokens (more negative = more deletion)
- `α` = deletion loss weight, dynamically adjusted by the PI controller

**Deletion loss** computed in `train_mrxlmr.py : compute_deletion_loss()`:

```python
deletion_loss = delete_gate_output[non_pad_mask].mean()
# More deletion → more negative gate values → smaller (more negative) deletion_loss
# Minimizing L_total with positive α drives gate values lower → more deletion
```

### Proportional-Integral (PI) Controller

The PI controller (`pi_controller.py`) adjusts `α` each step to track a target deletion rate `δ`:

```
error  = δ - actual_deletion_rate           # positive if deleting too little
p_acc  = γ × p_acc + (1 - γ) × kp × error  # EMA-smoothed proportional term
i_acc  = i_acc + ki × error                 # integral term (eliminates steady-state offset)
α      = max(0, p_acc + i_acc)
```

| Parameter | Default | Role |
|---|---|---|
| `target_deletion_rate` δ | 0.3 | Fraction of tokens to delete |
| `controller_p` kp | 0.01 | Proportional gain |
| `controller_i` ki | 1e-5 | Integral gain |
| `gamma` γ | 0.9 | EMA smoothing for proportional term |

**Dynamics**: If the model is deleting 15% when the target is 30%, `error = 0.15 > 0`, so `α` increases, strengthening the deletion pressure. If it overshoots to 40%, `error = -0.10 < 0`, and `α` decreases. The integral term prevents persistent under/over-deletion.

### Regularizer Delay

```python
# train_mrxlmr.py : MrXLMRTrainer.compute_loss()
if global_step < args.regularizer_delay:
    loss = task_loss   # pure task loss for first N steps
else:
    loss = task_loss + alpha * deletion_loss
```

This lets the task head converge before deletion pressure is applied, preventing the gate from collapsing early in training.

---

## 7. Code Structure and Walkthrough

```
mrxlmr/
├── models/
│   ├── configuration_mrxlmr.py     ← MrXLMRConfig (extends XLMRobertaConfig)
│   ├── modeling_mrxlmr.py          ← full model implementation (~900 lines)
│   └── diagnostics.py              ← token drop analysis, parameter summaries
│
├── training/
│   ├── train_mrxlmr.py             ← HuggingFace Trainer-based fine-tuning
│   ├── train_modal.py              ← serverless GPU training on Modal (A100)
│   └── pi_controller.py            ← PI controller for deletion rate targeting
│
├── data/
│   ├── preprocess_snli.py          ← SNLI → NDJSON with XLM-R tokens
│   ├── preprocess_squad.py         ← SQuAD 2.0 → sliding-window NDJSON
│   ├── preprocess_tydiqa.py        ← TyDi QA (English) → NDJSON
│   ├── preprocess_sst2.py          ← GLUE SST-2 → NDJSON
│   ├── preprocess_mrpc.py          ← GLUE MRPC → NDJSON
│   └── preprocess_imdb.py          ← IMDB sentiment → NDJSON
│
├── eval/
│   └── eval_mrxlmr.py              ← evaluation script (accuracy, deletion rate)
│
└── analysis/
    ├── get_deletion_patterns.py     ← per-token gate JSON for all SNLI test examples
    ├── deletion_pattern_analysis.py ← token-type rates, premise/hypothesis split, plots
    ├── measure_runtime.py           ← runtime benchmarking vs XLM-R baseline
    ├── hard_deletion_curve.py       ← soft vs hard accuracy across checkpoints
    └── compute_savings.py           ← theoretical MACs savings plots
```

### Key Classes and Their Roles

#### `models/configuration_mrxlmr.py`

```python
class MrXLMRConfig(XLMRobertaConfig):
    model_type = "mrxlmr"
    # Gate parameters on top of standard XLMRobertaConfig
    sigmoid_mask_scale = -30.0    # output range for gate: (-30, 0)
    deletion_threshold = -15.0    # below this → token counted as deleted
    delete_gate_layer  = 3        # which encoder layer holds the gate (0-indexed)
    use_softmax1       = True     # attention with n+1 denominator
    deletion_type      = "scaled_sigmoid"  # or "random"/"fixed" for ablations
    use_gumbel_noise   = False    # stochastic gate exploration
    use_pre_deletion_blend = True # blend pre-gate states for deleted tokens
```

#### `models/modeling_mrxlmr.py`

**Module hierarchy**:

```
MrXLMRForSequenceClassification
  └── MrXLMRModel
        ├── RobertaEmbeddings         ← from transformers (unchanged)
        ├── MrXLMREncoder
        │     └── MrXLMRLayer × 12
        │           ├── MrXLMRAttention
        │           │     ├── MrXLMRSelfAttention  ← softmax1 + gate mask
        │           │     └── RobertaSelfOutput    ← from transformers (unchanged)
        │           ├── RobertaIntermediate        ← from transformers (unchanged)
        │           ├── RobertaOutput              ← from transformers (unchanged)
        │           └── SigmoidDeleteGate          ← only in layer 3
        └── RobertaPooler             ← from transformers (unchanged)
```

**Execution flow through `MrXLMREncoder.forward()`**:

```
Layer 0: standard → hidden (batch, seq, 768)
Layer 1: standard → hidden (batch, seq, 768)
Layer 2: standard → hidden (batch, seq, 768)
                                                   ← save pre_deletion_hidden here
Layer 3: standard attention + FFN
         + SigmoidDeleteGate fires
           → delete_gate_mask (batch, seq, 1)
           → propagated to all remaining layers
Layer 4–11: attention_scores += delete_gate_mask   ← soft deletion
            (or: physically remove tokens)          ← hard deletion
```

**Gate initialization timing** (`modeling_mrxlmr.py:696–713`):

```python
def __init__(self, config):
    super().__init__(config)
    self.embeddings = RobertaEmbeddings(config)
    self.encoder    = MrXLMREncoder(config)
    self.pooler     = RobertaPooler(config)
    self.post_init()          # ← HuggingFace standard init (overwrites all weights)
    self._init_delete_gates() # ← must come AFTER post_init() to set bias=10, std=0.001
```

Without the post-`post_init()` call, HuggingFace would reset the gate's bias to ~0.0, causing the gate to immediately delete ~50% of tokens before any learning has occurred.

#### `training/train_mrxlmr.py`

The trainer extends HuggingFace `Trainer` with a custom `compute_loss()`:

```python
class MrXLMRTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, hard_delete=use_hard_delete)
        task_loss = outputs.loss

        # Deletion loss from gate values
        deletion_loss, del_rate, _ = compute_deletion_loss(
            outputs.delete_gate_output,
            inputs["input_ids"],
            deletion_threshold,
            sigmoid_mask_scale,
        )

        # PI controller updates alpha
        if use_pi_controller:
            alpha = pi_controller.update(del_rate / 100.0)

        # Combine losses (skip deletion loss before regularizer_delay steps)
        if global_step >= regularizer_delay:
            loss = task_loss + alpha * deletion_loss
        else:
            loss = task_loss
```

---

## 8. Token-Level Examples by Task

### Notation

```
[KEPT]   = gate value > -15.0 (token attends normally)
[DEL]    = gate value < -15.0 (token suppressed in subsequent layers)
<s>      = CLS token, always KEPT (position 0)
</s>     = SEP token, always KEPT (token_id=2)
<pad>    = always DELETED (token_id=1)
```

---

### 8.1 Sequence Classification — SNLI (Entailment)

**Task**: Classify (premise, hypothesis) pair as entailment / neutral / contradiction.

**Input** (premise: "A man is playing guitar", hypothesis: "A person is making music"):

```
XLM-R tokenization:
  <s>  ▁A  ▁man  ▁is  ▁playing  ▁guitar  </s>  </s>  ▁A  ▁person  ▁is  ▁making  ▁music  </s>  <pad>…
   0    1    2    3       4         5        6     7    8      9      10     11       12      13   14…
```

**After gate fires (layer 3), trained model at 30% deletion rate**:

```
Token:   <s>   ▁A  ▁man   ▁is  ▁playing ▁guitar </s> </s>  ▁A ▁person  ▁is ▁making ▁music </s>  <pad>
Gate:     0.0 -0.1 -0.3  -28.5   -0.4    -0.8    0.0  0.0 -29.1  -0.2  -27.8  -0.6   -0.5  0.0   -30.0
Status:  KEPT KEPT KEPT  [DEL]   KEPT    KEPT   KEPT KEPT [DEL]  KEPT  [DEL]  KEPT   KEPT KEPT   [DEL]
```

The gate learns that the **copula "is"** carries little semantic content for entailment — the predicates ("playing guitar" ↔ "making music") and the nominals ("man"/"person") are what matter.

**Training**: Total loss = CE(logits, label=0) + α × mean(gate_values[non-pad])

**Inference**:
- Soft: layers 4–11 see `attention_scores[:, :, "is" col] += -28.5` → near-zero weight
- Hard: layers 4–11 operate on shortened sequence (11 → 8 tokens), genuine speedup

**Classification**: `<s>` pooled output → Linear(768→3) → argmax → "entailment"

---

### 8.2 Sequence Classification — SST-2 (Sentiment)

**Task**: Binary sentiment classification (negative=0, positive=1).

**Input**: "The film is a masterpiece"

```
Token:  <s>  ▁The  ▁film   ▁is    ▁a  ▁master  piece  </s>
ID:      0     6    312     22    20     4521    3892     2
```

**After gate (30% deletion target)**:

```
Token:  <s>  ▁The  ▁film   ▁is    ▁a  ▁master  piece  </s>
Gate:   0.0  -27.2   -0.3  -29.1 -28.8   -0.5   -0.4   0.0
Status: KEPT [DEL]  KEPT  [DEL] [DEL]   KEPT   KEPT  KEPT
```

The gate deletes the article ("The"), the copula ("is"), and the determiner ("a") — function words with low sentiment signal. The compound "masterpiece" ("▁master" + "piece") is kept, as it carries the positive sentiment.

---

### 8.3 Sequence Classification — MRPC (Paraphrase Detection)

**Task**: Binary classification — are two sentences paraphrases?

**Input**:
- S1: "He said the policy was not a good idea"
- S2: "He said the idea was not a good policy"

```
XLM-R pair encoding:
  <s> ▁He ▁said ▁the ▁policy ▁was ▁not ▁a ▁good ▁idea </s> </s> ▁He ▁said ▁the ▁idea ▁was ▁not ▁a ▁good ▁policy </s>
```

**After gate**:

```
Token:   <s>  ▁He ▁said  ▁the ▁policy ▁was  ▁not  ▁a ▁good ▁idea </s> </s>  ▁He ▁said  ▁the ▁idea  ▁was  ▁not   ▁a ▁good ▁policy </s>
Gate:    0.0 -29.5  -0.4 -28.1   -0.6  -28.3  -0.8 -27.9  -0.3  -0.5  0.0  0.0  -29.2  -0.3  -28.0  -0.6  -28.4  -0.7  -28.1  -0.2   -0.4  0.0
Status: KEPT [DEL] KEPT [DEL]   KEPT  [DEL] KEPT [DEL] KEPT KEPT KEPT KEPT [DEL] KEPT  [DEL] KEPT  [DEL]  KEPT  [DEL] KEPT   KEPT  KEPT
```

The gate deletes subject pronouns, "the", and "was" — common across both sentences. The key swapped content words ("policy"/"idea") are preserved, allowing the model to detect the paraphrase relationship.

---

### 8.4 Sequence Classification — IMDB (Sentiment)

**Task**: Binary sentiment on long movie reviews (max_length=256).

**Input**: "A brilliant piece of cinema. The acting is superb, though the pacing drags in places."

```
Token:  <s>  ▁A ▁bri lli ant ▁piece ▁of ▁cinema  .  ▁The ▁acting ▁is ▁super b  , ▁though ▁the ▁pacing ▁drag s  ▁in ▁places  .  </s>
```

**After gate** (longer sequences → higher absolute deletion count):

```
Token:   <s>   ▁A  ▁bri  lli  ant  ▁piece   ▁of ▁cinema    .  ▁The ▁acting   ▁is  ▁super     b    ,  ▁though  ▁the  ▁pacing  ▁drag    s   ▁in ▁places    .  </s>
Gate:    0.0 -28.3  -0.5 -0.4 -0.6   -0.4  -28.1   -0.5  -27.3 -27.9   -0.4 -29.1   -0.4  -0.5 -28.2   -0.4  -28.7   -0.6   -0.5 -0.4 -28.5   -0.5 -27.8   0.0
Status: KEPT [DEL] KEPT KEPT KEPT   KEPT  [DEL]   KEPT  [DEL] [DEL]  KEPT [DEL]   KEPT  KEPT [DEL]   KEPT  [DEL]   KEPT   KEPT KEPT [DEL]  KEPT [DEL]  KEPT
```

Function words, punctuation, and "is" are deleted. The multi-token word "brilliant" (▁bri + lli + ant) is fully kept because all pieces contribute to the sentiment signal.

---

### 8.5 Token Classification — CoNLL-2003 NER

**Task**: Per-token label (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).

**Input**: "Barack Obama visited Google in London"

```
Token:  <s>  ▁Bar  ack  ▁Obama  ▁visited  ▁Google   ▁in  ▁London  </s>
```

**After gate**:

```
Token:   <s>   ▁Bar  ack  ▁Obama  ▁visited  ▁Google   ▁in  ▁London  </s>
Gate:    0.0   -0.4 -0.5    -0.6    -28.4     -0.7   -29.1    -0.5   0.0
Status: KEPT  KEPT KEPT   KEPT    [DEL]     KEPT   [DEL]   KEPT  KEPT
```

The gate deletes "visited" and "in" (non-entity function tokens), while preserving the named entity pieces (BarACK, Obama, Google, London).

**Critical difference from classification**: The token classification head reads **every token position**, not just `<s>`. Deleted tokens still receive predictions but through corrupted (near-zero attention) representations:

```python
# modeling_mrxlmr.py : MrXLMRForTokenClassification.forward()
sequence_output = blended_output    # pre-deletion blend applied
logits = self.classifier(dropout(sequence_output))
# shape: (batch, seq_len, num_labels) — all positions predicted, including deleted ones
```

The pre-deletion blend ensures deleted tokens still carry useful positional context for their labels.

**Training loss** excludes positions with `labels == -100` (special tokens and padding):

```python
loss_fct = CrossEntropyLoss()
loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
# labels = -100 for <s>, </s>, <pad>, and subword continuations (only first subword is labeled)
```

---

### 8.6 Question Answering — SQuAD / TyDi QA

**Task**: Extract start/end span of answer from context given a question.

**Input**:
- Question: "Who founded Microsoft?"
- Context: "Bill Gates and Paul Allen founded Microsoft in 1975."

**XLM-R encoding** (question first, then context, max_length=384):

```
<s> ▁Who ▁founded ▁Microsoft ? </s> </s> ▁Bill ▁Gates ▁and ▁Paul ▁Allen ▁founded ▁Microsoft ▁in ▁19 75 . </s>
 0    1       2         3      4   5    6     7      8     9    10    11       12         13    14  15  16  17  18
```

**After gate** (gate fires after question + beginning of context have been encoded):

```
Token:   <s> ▁Who ▁founded ▁Microsoft    ?  </s> </s>  ▁Bill  ▁Gates   ▁and   ▁Paul  ▁Allen  ▁founded ▁Microsoft  ▁in  ▁19  75   .  </s>
Gate:    0.0  -0.5   -0.4      -0.5    -28.2  0.0  0.0   -0.6   -0.5  -28.4   -0.5    -0.6    -28.7      -0.5   -28.3 -0.5 -0.4 -28.1  0.0
Status: KEPT KEPT  KEPT      KEPT    [DEL] KEPT KEPT   KEPT   KEPT  [DEL]   KEPT    KEPT    [DEL]     KEPT   [DEL] KEPT KEPT [DEL] KEPT
```

The gate deletes "?", "and", "founded" (repeated from question, less informative in context), and "in" — keeping the named entities and the year.

**Pre-deletion blend becomes critical here**: The answer is "Bill Gates", both of which are KEPT. But if the answer happened to be a deleted token (e.g., "1975" which maps to "▁19" + "75"), the blend ensures those positions still produce valid span logits:

```
deleted_weight = clamp(-gate_val / 30, 0, 1) = clamp(0.5/30, 0, 1) = 0.017 for gate=-0.5  (kept)
deleted_weight = clamp(28.1/30, 0, 1) = 0.937 for gate=-28.1  (deleted ".")
```

**Span head**:

```python
# modeling_mrxlmr.py : MrXLMRForQuestionAnswering.forward()
sequence_output = MrXLMRModel.forward(...).last_hidden_state   # (batch, seq, 768)
sequence_output = blend(sequence_output, pre_deletion_hidden, gate_mask)
logits = self.qa_outputs(sequence_output)   # Linear(768→2)
start_logits, end_logits = logits.split(1, dim=-1)
# Answer: argmax(start_logits) = 7 ("▁Bill"), argmax(end_logits) = 8 ("▁Gates")
```

**Pre-deletion format for TyDi QA** uses English-only examples (filtered by `id.startswith("english-")`), with answer spans extracted identically to SQuAD. The gold "yes/no" annotation type in TyDi is treated as a regular span QA task (mapped to the question position as the answer span when there is no text span).

**Sliding window** (for long contexts, `preprocess_squad.py`): Long documents are split into overlapping windows of `max_length=384` with stride 128, and the answer is predicted in the window where it appears.

---

### 8.7 MLM Pre-training

**Task**: Predict masked tokens (15% masking, replace with `<mask>` id=250001).

**Input** (masked): "▁The ▁cat [MASK] ▁on ▁the ▁mat"

**Token sequence**:

```
<s>  ▁The  ▁cat [MASK]   ▁on   ▁the   ▁mat   </s>
 0      6  1234  250001   432    6      3421     2
```

**After gate**:

```
Token:  <s>  ▁The  ▁cat [MASK]   ▁on  ▁the  ▁mat  </s>
Gate:   0.0  -28.1  -0.4   -0.3  -28.5 -28.9  -0.4  0.0
Status: KEPT [DEL] KEPT   KEPT  [DEL] [DEL]  KEPT KEPT
```

The model deletes "The", "on", "the" (function words) while keeping "cat" and "mat" (content words) and the masked target token. This means layers 4–11 compute over a shorter sequence while still attending to the information needed to predict the mask.

**MLM head** (uses `RobertaLMHead` — different from BERT's `BertOnlyMLMHead`):

```python
# modeling_mrxlmr.py : MrXLMRForMaskedLM
self.lm_head = RobertaLMHead(config)
# RobertaLMHead: Linear(768→768) → GELU → LayerNorm → Linear(768→vocab_size)
# weight tied to embedding table: lm_head.decoder.weight == embeddings.word_embeddings.weight
```

---

## 9. Configuration Reference

### `MrXLMRConfig` Parameters

| Parameter | Default | Description |
|---|---|---|
| `delete_gate_layer` | 3 | Encoder layer index where gate fires (0-indexed) |
| `deletion_type` | `"scaled_sigmoid"` | Gate variant: `scaled_sigmoid`, `log_sigmoid`, `random`, `fixed` |
| `sigmoid_mask_scale` | -30.0 | Output range of gate: gate ∈ (sigmoid_mask_scale, 0) |
| `deletion_threshold` | -15.0 | Gate value below which token is counted as deleted (= scale/2) |
| `gate_layer_norm` | True | Apply LayerNorm before the gate linear layer |
| `use_softmax1` | True | Attention with n+1 denominator (MrT5 recommendation) |
| `use_gumbel_noise` | False | Add Gumbel noise to gate logits during training |
| `use_pre_deletion_blend` | True | Blend pre-gate hidden states for deleted token positions |
| `bypass_gate` | False | Disable gate entirely (0%-deletion ablation baseline) |

### Gate Type Summary

| `deletion_type` | Learnable? | Use case |
|---|---|---|
| `scaled_sigmoid` | Yes | Main gate: `g = -30 × σ(-z)` |
| `log_sigmoid` | Yes | Alternative: `g = log(σ(z))` |
| `random` | No | Ablation: randomly delete `random_deletion_probability` fraction |
| `fixed` | No | Ablation: always delete exactly `fixed_deletion_amount` fraction |

---

## 10. Usage

### Environment Setup

```bash
conda create -n mrxlmr python=3.11 -y
conda activate mrxlmr
pip install torch transformers>=4.40.0 datasets accelerate tqdm matplotlib "numpy<2" sentencepiece wandb modal
```

### Data Preprocessing

```bash
cd mrxlmr/data/

# SNLI (3-class NLI)
python preprocess_snli.py --output_dir ./snli_datasets

# SQuAD (QA, sliding window)
python preprocess_squad.py --output_dir ./squad_datasets

# TyDi QA (English-only QA)
python preprocess_tydiqa.py --output_dir ./tydiqa_datasets

# SST-2 (sentiment)
python preprocess_sst2.py --output_dir ./sst2_datasets

# MRPC (paraphrase)
python preprocess_mrpc.py --output_dir ./mrpc_datasets

# IMDB (sentiment, long documents)
python preprocess_imdb.py --output_dir ./imdb_datasets --max_length 256
```

### Local Training

```bash
cd mrxlmr/training/

# SNLI sequence classification, 30% deletion target
python train_mrxlmr.py \
    --task sequence_classification \
    --dataset_name local_snli \
    --local_snli_dir ../data/snli_datasets \
    --model_type MrXLMR \
    --target_deletion_rate 0.3 \
    --delete_gate_layer 3 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --output_dir ./mrxlmr_snli

# XLM-R baseline (no delete gate)
python train_mrxlmr.py \
    --task sequence_classification \
    --dataset_name local_snli \
    --local_snli_dir ../data/snli_datasets \
    --model_type XLMR \
    --output_dir ./xlmr_snli_baseline

# SQuAD QA
python train_mrxlmr.py \
    --task question_answering \
    --dataset_name local_squad \
    --local_squad_dir ../data/squad_datasets \
    --max_seq_length 384 \
    --batch_size 16 \
    --target_deletion_rate 0.3 \
    --output_dir ./mrxlmr_squad

# Quick smoke test (100 steps)
python train_mrxlmr.py --max_steps 100 --logging_steps 10 --output_dir ./test_run
```

### Modal (Serverless GPU, A100)

```bash
cd mrxlmr/training/

# Quick test (20 steps)
modal run train_modal.py

# Full SNLI run with delete gate
modal run --detach train_modal.py \
    --model-type MrXLMR \
    --max-steps -1 \
    --target-deletion-rate 0.3

# XLM-R baseline
modal run --detach train_modal.py --model-type XLMR --max-steps -1

# SQuAD
modal run --detach train_modal.py \
    --task question_answering \
    --dataset-name local_squad \
    --batch-size 16 \
    --max-steps -1

# Download checkpoints
modal volume get mrxlmr-checkpoints checkpoints ./local_mrxlmr_checkpoints
```

### Evaluation

```bash
cd mrxlmr/eval/

# SNLI accuracy + deletion rate
python eval_mrxlmr.py \
    --model_path ./mrxlmr_snli/final \
    --dataset_name local_snli \
    --local_snli_dir ../data/snli_datasets

# Hard deletion evaluation
python eval_mrxlmr.py \
    --model_path ./mrxlmr_snli/final \
    --hard_delete

# Compare MrXLMR vs XLM-R baseline
python eval_mrxlmr.py \
    --model_path ./mrxlmr_snli/final \
    --compare_xlmr

# Show deletion examples
python eval_mrxlmr.py \
    --model_path ./mrxlmr_snli/final \
    --show_examples
```

### Analysis

```bash
cd mrxlmr/

# Generate per-token gate decisions JSON
python analysis/get_deletion_patterns.py \
    --model_path ./mrxlmr_snli/final \
    --sample_size 1000 \
    --output_dir ./analysis/deletion_patterns

# Analyze deletion patterns (token types, premise vs hypothesis)
python analysis/deletion_pattern_analysis.py \
    --input_file ./analysis/deletion_patterns/final_test.json \
    --output_dir ./analysis/figures

# Runtime benchmarking
python analysis/measure_runtime.py \
    --models "XLM-R,./xlmr_snli/final" "MrXLMR-30%,./mrxlmr_snli/final" \
    --local_snli_dir ./data/snli_datasets

# Soft vs hard accuracy across checkpoints
python analysis/hard_deletion_curve.py \
    --checkpoint_dir ./mrxlmr_snli \
    --local_snli_dir ./data/snli_datasets

# Theoretical MACs savings (no model needed)
python analysis/compute_savings.py \
    --runs "XLM-R,0.9055,0.0" "MrXLMR-30%,0.87,0.30" "MrXLMR-50%,0.84,0.50"
```

---

## References

- Kallini et al. (2024). [MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/pdf/2410.20771)
- Conneau et al. (2020). [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-RoBERTa
- Liu et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)