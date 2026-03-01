# MrBERT Explainer: What We Changed and Why

This document explains every modification made to standard BERT to create MrBERT, with the intuition behind each change and pointers to the relevant code.

**Reference paper**: [MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/pdf/2410.20771) (Kallini et al., 2024). MrBERT adapts the delete gate mechanism from MrT5 to the BERT architecture.

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Code and File Structure](#3-code-and-file-structure)
4. [Config Changes](#4-config-changes)
5. [The Delete Gate](#5-the-delete-gate)
6. [Soft Deletion: Masking Attention](#6-soft-deletion-masking-attention)
7. [Hard Deletion: Physically Removing Tokens](#7-hard-deletion-physically-removing-tokens)
8. [Protecting Special Tokens](#8-protecting-special-tokens)
9. [Where the Gate Sits in the Encoder](#9-where-the-gate-sits-in-the-encoder)
10. [Training: The Deletion Loss and PI Controller](#10-training-the-deletion-loss-and-pi-controller)
11. [Gate Variants](#11-gate-variants)
12. [What Stays the Same](#12-what-stays-the-same)
13. [End-to-End Flow](#13-end-to-end-flow)

---

## 1. The Core Idea

Standard BERT processes every token through all 12 encoder layers, even tokens that carry little semantic information (e.g. punctuation, filler words, repeated content). This is computationally wasteful.

**The MrBERT idea**: after a chosen encoder layer, assign each token a score. Low-scoring tokens are either ignored by subsequent attention layers (soft deletion) or physically removed from the sequence (hard deletion). The rest of the encoder only processes the surviving tokens.

This is a learned compression: the model is trained to identify and discard tokens that don't contribute to the task, reducing the effective sequence length and therefore the compute required for the remaining layers.

---

## 2. Architecture Diagram

### MrBERT vs standard BERT

```
STANDARD BERT                          MrBERT
─────────────────────────────────────  ──────────────────────────────────────────────────────
Input tokens (seq_len = L)             Input tokens (seq_len = L)
        │                                      │
  BertEmbeddings                         BertEmbeddings
        │                                      │
  Encoder Layer 0  ─┐                    Encoder Layer 0  ─┐
  Encoder Layer 1   │ standard           Encoder Layer 1   │ standard BERT layers
  Encoder Layer 2   │ BERT               Encoder Layer 2  ─┘  (no gate)
  Encoder Layer 3   │ layers                    │
  Encoder Layer 4   │                    Encoder Layer 3  ← DELETE GATE HERE
  Encoder Layer 5   │                           │
  Encoder Layer 6   │                    ┌──────┴────────────────────────────┐
  Encoder Layer 7   │                    │  SigmoidDeleteGate                │
  Encoder Layer 8   │                    │  LayerNorm → Linear(768→1)        │
  Encoder Layer 9   │                    │    → ScaledSigmoid → [-30, 0]     │
  Encoder Layer 10  │                    │                                    │
  Encoder Layer 11 ─┘                    │  Per-token gate score:            │
        │                                │    ≈  0  → KEEP                   │
  Pooler / Task Head                     │    ≈ -30 → DELETE                 │
        │                                └──────┬────────────────────────────┘
  [CLS] → classification                        │ gate_mask (batch, seq, 1)
                                                │ passed to layers 4-11
                                         Encoder Layer 4  ┐
                                         Encoder Layer 5  │ attention scores
                                         Encoder Layer 6  │   += gate_mask
                                         Encoder Layer 7  │ (deleted tokens
                                         Encoder Layer 8  │  receive ~0
                                         Encoder Layer 9  │  attention)
                                         Encoder Layer 10 │
                                         Encoder Layer 11 ┘
                                                │
                                         Pooler / Task Head
                                                │
                                         [CLS] → classification
```

### The delete gate in detail

```
Hidden state at layer 3  (batch, seq_len, 768)
            │
       ┌────┴────────────────────────────────────────────────────┐
       │                    SigmoidDeleteGate                     │
       │                                                          │
       │   hidden_states                                          │
       │       │                                                  │
       │   LayerNorm                                              │
       │       │                                                  │
       │   Linear(768 → 1)   ← learns which features = deletable │
       │       │                                                  │
       │   [+ Gumbel noise]  ← optional, encourages exploration  │
       │       │                                                  │
       │   ScaledSigmoid:  -30 × σ(-logit)                       │
       │       │                                                  │
       │   gate_values  (batch, seq_len, 1)  in [-30, 0]         │
       │       │                                                  │
       │   Force overrides:                                       │
       │     [CLS]  position 0  →  0.0   (always keep)           │
       │     [SEP]  token_id 102 →  0.0   (always keep)          │
       │     [PAD]  token_id 0   → -30.0  (always delete)        │
       │                                                          │
       └────┬─────────────────────────────────────────────────── ┘
            │
       gate_mask  (used as additive bias in attention layers 4-11)
            │
            ▼
   attention_scores += gate_mask   (before softmax, in every layer 4-11)
   → tokens with gate ≈ -30 receive ≈ 0 attention from all other tokens
```

### Training objective

```
Total loss = Task loss  +  α × Deletion loss
                               ▲
                    PI Controller adjusts α each step
                    to keep actual_deletion_rate ≈ target_deletion_rate

Deletion loss = mean(gate_values)  over non-pad tokens
             → minimizing pushes gate values toward -30 (more deletion)
             → task loss resists (keep tokens that help the task)
```

---

## 3. Code and File Structure

```
mrbert/
│
├── models/                          ← model architecture
│   ├── configuration_mrbert.py      MrBertConfig: BertConfig + gate hyperparameters
│   ├── modeling_mrbert.py           Full MrBERT implementation (~1,400 lines)
│   │     ScaledSigmoid              activation: -30 × σ(-x), output in [-30, 0]
│   │     SigmoidDeleteGate          learnable gate: LayerNorm → Linear → ScaledSigmoid
│   │     LogSigmoidDeleteGate       alternative gate variant
│   │     RandomDeleteGate           ablation: random token deletion
│   │     FixedDeleteGate            ablation: fixed-rate token deletion
│   │     MrBertSelfAttention        BERT attention + additive gate mask
│   │     MrBertLayer                single encoder layer, optionally has delete gate
│   │     MrBertEncoder              12-layer encoder, propagates gate mask
│   │     MrBertModel                full model (embeddings + encoder + pooler)
│   │     MrBertForSequenceClassification   ← used for SNLI
│   │     MrBertForMaskedLM
│   │     MrBertForTokenClassification
│   │     MrBertForQuestionAnswering
│   │     MrBertForMultipleChoice
│   │     MrBertForNextSentencePrediction
│   ├── modeling_bert.py             Standard BERT reference implementation (baseline)
│   └── diagnostics.py              Compute savings, dropped-token analysis, gate demo
│
├── training/                        ← training infrastructure
│   ├── train_mrbert.py              Main training script (all tasks, all CLI args)
│   ├── pi_controller.py             PIController: adjusts deletion loss weight α
│   ├── train_modal.py               Modal (serverless GPU) launcher
│   ├── train_gcp.py                 GCP launcher
│   ├── train_gce.py                 GCE launcher
│   └── requirements.txt
│
├── eval/
│   └── eval_mrbert.py               Standalone evaluation script
│
├── data/
│   └── preprocess_snli.py           Downloads and preprocesses SNLI → JSON files
│
├── test/
│   └── test_mrbert.py               Unit tests for model components
│
├── snli_datasets/                   Preprocessed SNLI (generated, not committed)
│   ├── snli-train.json
│   ├── snli-validation.json
│   └── snli-test.json
│
├── run_comparison.py                Runs MrBERT vs BERT comparison experiment
├── run_baseline.py                  Runs BERT baseline only
│
└── README_*.md                      Documentation
```

### Key relationships between files

```
train_mrbert.py
    imports ──► configuration_mrbert.py   (MrBertConfig)
    imports ──► modeling_mrbert.py        (MrBertFor* task heads)
    imports ──► pi_controller.py          (PIController)
    imports ──► diagnostics.py            (log_parameter_summary, etc.)

train_modal.py
    launches ──► train_mrbert.py          (via subprocess on Modal A100)

modeling_mrbert.py
    imports ──► configuration_mrbert.py   (MrBertConfig)
    extends  ──► BertPreTrainedModel      (HuggingFace transformers)
```

---

## 4. Config Changes

**File**: `mrbert/models/configuration_mrbert.py`, lines 48–71

`MrBertConfig` extends the standard `BertConfig` with these new fields:

| Parameter | Default | Purpose |
|---|---|---|
| `delete_gate_layer` | `3` | Which encoder layer emits the gate (0-indexed). Layers after this one only see surviving tokens. |
| `deletion_type` | `"scaled_sigmoid"` | Which gate variant to use. |
| `sigmoid_mask_scale` | `-30.0` | Controls the strength of the deletion signal added to attention scores. |
| `deletion_threshold` | `-15.0` | Gate value below which a token is counted as deleted (`sigmoid_mask_scale / 2`). |
| `gate_layer_norm` | `True` | Apply LayerNorm to hidden states before computing the gate. |
| `use_softmax1` | `True` | Add 1 to softmax denominator, allowing attention weights to sum < 1. Recommended by the MrT5 paper. |
| `use_gumbel_noise` | `False` | Add Gumbel noise to gate logits during training to encourage exploration. |
| `random_deletion_probability` | `0.5` | Used only by the random gate ablation. |
| `fixed_deletion_amount` | `0.5` | Used only by the fixed gate ablation. |

None of these exist in standard `BertConfig`. They expose the gate as a set of tunable hyperparameters without touching the core BERT architecture.

---

## 5. The Delete Gate

**File**: `mrbert/models/modeling_mrbert.py`, lines 160–249

### The ScaledSigmoid activation (lines 160–169)

```python
def forward(self, input):
    return self.sigmoid_mask_scale * torch.sigmoid(-input)
```

This maps the gate's raw logit to a value in the range `[-30, 0]`:

- **Large positive logit** → `sigmoid(-large) ≈ 0` → gate value ≈ `0` → **keep the token**
- **Large negative logit** → `sigmoid(-large_neg) ≈ 1` → gate value ≈ `-30` → **delete the token**

The output is designed to be added directly to attention scores (see Section 4). A value of 0 has no effect; a value of -30 makes the softmax assign essentially zero attention weight to that token.

### The SigmoidDeleteGate module (lines 178–249)

The gate is a small neural network sitting on top of one encoder layer:

```
hidden_state (768-dim)
     ↓
[LayerNorm]           ← normalizes the input for stable training
     ↓
Linear(768 → 1)       ← learns which features predict "this token is dispensable"
     ↓
ScaledSigmoid(-30)    ← maps the scalar logit to [-30, 0]
     ↓
gate_value per token
```

**Initialization** (lines 236–248): The linear layer is initialized with very small weights (`std=0.001`) and a large positive bias (`10.0`). This is critical. With `bias=10`:

```
gate_value = -30 × sigmoid(-10) ≈ -30 × 0.000045 ≈ 0
```

All tokens start with a gate value near 0 (keep everything). The model has to actively learn to push tokens toward -30. This prevents catastrophic deletion at the start of training before the model has learned anything useful.

---

## 6. Soft Deletion: Masking Attention

**File**: `mrbert/models/modeling_mrbert.py`, lines 393–441 (`MrBertSelfAttention.forward`)

Soft deletion is the default mode. Tokens are not removed from memory — they just become invisible to attention in subsequent layers.

### How it works (lines 414–419)

```python
if delete_gate_mask is not None:
    delete_gate_mask_expanded = delete_gate_mask.squeeze(-1).unsqueeze(1).unsqueeze(1)
    attention_scores = attention_scores + delete_gate_mask_expanded
```

In standard BERT, attention scores are computed as `QK^T / sqrt(d)` and then passed to softmax. In MrBERT, before the softmax, we add the gate values to the attention scores along the key dimension.

**Intuition**: If token A has gate value -30, then every other token's attention score for attending to token A is reduced by 30. After softmax, token A receives essentially zero attention weight from every other token. It is still in the sequence but is completely ignored by all subsequent layers.

This is an elegant approach because:
- It requires no architectural surgery — just an additive bias before softmax
- It is fully differentiable, so the gate can be trained end-to-end
- The original BERT attention mechanism is untouched; the gate is purely additive

### Softmax1 variant (lines 422–425)

Optionally enabled via `use_softmax1=True`. Adds 1 to the softmax denominator, allowing attention weights to sum to less than 1. This lets the model express "I don't want to attend to anything strongly" rather than being forced to distribute all attention mass across available tokens.

---

## 7. Hard Deletion: Physically Removing Tokens

**File**: `mrbert/models/modeling_mrbert.py`, lines 509–553 (`MrBertLayer`)

Hard deletion is an optional mode (`hard_delete=True`). Instead of masking attention, tokens below the deletion threshold are physically removed from the sequence tensor.

### Step 1: Decide which tokens survive (lines 509–541, `_get_new_positions_and_mask`)

```python
keep_this = delete_gate_mask > deletion_threshold  # True = keep, False = delete
```

A cumulative sum maps each surviving token to its new position in the shorter sequence. A new attention mask is computed for the shortened sequence.

### Step 2: Gather surviving tokens (lines 543–553, `_hard_delete_hidden_states`)

```python
new_hidden_states = torch.gather(hidden_states, 1, gather_indices)
```

`torch.gather` physically selects only the rows corresponding to kept tokens, reducing the sequence dimension from `seq_len` to `num_kept_tokens`.

**Intuition**: After hard deletion, layers 4–11 operate on a shorter sequence. Since attention is `O(n²)` in sequence length, removing 30% of tokens reduces attention compute in those layers by roughly `0.7² ≈ 49%`. This is the actual efficiency gain MrBERT is designed to achieve.

**Why soft deletion is the default**: Soft deletion is fully differentiable and easier to train. Hard deletion involves discrete decisions (keep/delete) which complicate gradients. In practice, soft deletion is used during training; hard deletion can be applied at inference time.

---

## 8. Protecting Special Tokens

**File**: `mrbert/models/modeling_mrbert.py`, lines 213–232 (`SigmoidDeleteGate.forward`)

BERT uses three special tokens that must never be deleted:

### [CLS] (line 215)
```python
gate_values[:, 0, :] = 0.0
```
The [CLS] token at position 0 is hardcoded to gate value 0 (keep). For classification tasks, the [CLS] hidden state is the model's representation of the entire input. Deleting it would destroy the task head's input.

### [SEP] (lines 217–223)
```python
sep_mask = (input_ids == 102).unsqueeze(-1)
gate_values = torch.where(sep_mask, torch.zeros_like(gate_values), gate_values)
```
[SEP] tokens (token ID 102) mark sentence boundaries. In SNLI, which uses two sentences (premise + hypothesis), [SEP] tells the model where one ends and the other begins. Deleting it would conflate the two sentences.

### [PAD] (lines 225–232)
```python
pad_mask = (input_ids == self.pad_token_id).unsqueeze(-1)
gate_values = torch.where(pad_mask, torch.tensor(sigmoid_mask_scale), gate_values)
```
Padding tokens are forced to the most negative gate value (`-30`), meaning they are always deleted. This is correct behavior — padding carries no information — and it means the deletion rate metrics measure real content deletion, not just padding removal.

---

## 9. Where the Gate Sits in the Encoder

**File**: `mrbert/models/modeling_mrbert.py`, lines 631–723 (`MrBertEncoder`)

### Layer construction (lines 644–647)

```python
self.layer = nn.ModuleList([
    MrBertLayer(config, has_delete_gate=(i == config.delete_gate_layer))
    for i in range(config.num_hidden_layers)
])
```

Only one layer — layer 3 by default — has `has_delete_gate=True`. All other layers are functionally identical to standard BERT layers.

**Why layer 3?** After 3 layers, the model has built up enough contextual representation for the gate to make meaningful keep/delete decisions. Earlier layers may not have enough context; later layers have already done most of the work and there is less compute to save.

### Forward pass gate propagation (lines 670–707)

During the forward pass, the encoder tracks the `delete_gate_mask` produced at layer 3. This mask is passed as input to every subsequent attention layer (layers 4–11), which apply the soft deletion bias. The gate is computed once and reused for the rest of the encoder.

---

## 10. Training: The Deletion Loss and PI Controller

**Files**: `mrbert/training/train_mrbert.py`, `mrbert/training/pi_controller.py`

Training MrBERT requires solving two objectives simultaneously:
1. Minimize the task loss (get good NLI accuracy)
2. Delete approximately the target fraction of tokens

These two objectives can conflict — the easiest way to minimize task loss is to delete nothing. The deletion loss and PI controller are the mechanism that enforces the deletion target.

### The deletion loss (lines 347–387, `compute_deletion_loss`)

```python
deletion_loss = delete_gate_output[non_pad_mask].mean()
```

This is simply the mean gate value across all non-pad tokens. Since gate values are in `[-30, 0]`, minimizing this loss pushes gate values toward -30 (more deletion). The task loss pushes the gate toward 0 (keep everything useful). The balance between these two forces determines the actual deletion rate.

### The PI controller (`mrbert/training/pi_controller.py`)

A fixed `deletion_loss_weight` is fragile — too high and the gate collapses (deletes everything), too low and nothing gets deleted. Instead, the weight `α` is adjusted dynamically by a PI controller:

```python
error = target_rate - actual_deletion_rate
p_acc = gamma * p_acc + (1 - gamma) * kp * error   # proportional term (EMA smoothed)
i_acc = i_acc + ki * error                           # integral term (accumulates over time)
alpha = max(0.0, p_acc + i_acc)                      # final weight
```

`gamma` (default 0.9) controls EMA smoothing of the proportional term. The controller is a standalone module in `pi_controller.py` and is imported by `train_mrbert.py`.

**Intuition**: If the actual deletion rate is below the target (not deleting enough), `error` is positive, so `α` increases, applying more deletion pressure. If the rate overshoots, `error` is negative, reducing `α`. The integral term prevents the rate from perpetually sitting slightly below target — it accumulates the error over time and applies a slow corrective push.

### The regularizer delay (line 977–985)

```python
if global_step >= args.regularizer_delay:
    loss = task_loss + current_alpha * deletion_loss
else:
    loss = task_loss
```

For the first `regularizer_delay` steps (default: 1000), only the task loss is optimized. This gives the model time to learn what makes a good NLI prediction before deletion pressure is applied. If deletion pressure starts immediately, the gate may learn to delete tokens before the model knows which ones are important.

### Separate learning rate for the gate (lines 850–866)

```python
optimizer = AdamW([
    {"params": other_params, "lr": args.learning_rate},       # 2e-5
    {"params": delete_gate_params, "lr": args.delete_gate_lr}, # 1e-4
])
```

The gate is a randomly initialized small network sitting on top of pretrained BERT weights. It needs to learn faster than the pretrained weights, which only need fine-tuning. A higher learning rate for gate parameters allows the gate to adapt quickly without destabilizing the pretrained representations.

---

## 11. Gate Variants

**File**: `mrbert/models/modeling_mrbert.py`, lines 178–346

Four gate implementations are available via `deletion_type` in the config:

| Type | Class | Lines | Description |
|---|---|---|---|
| `scaled_sigmoid` | `SigmoidDeleteGate` | 178–249 | Main learnable gate. Trained end-to-end. |
| `log_sigmoid` | `LogSigmoidDeleteGate` | 251–256 | Same architecture, uses log sigmoid instead of scaled sigmoid. |
| `random` | `RandomDeleteGate` | 259–299 | Ablation: randomly deletes tokens at a configured rate. Tests whether the learned gate actually outperforms chance. |
| `fixed` | `FixedDeleteGate` | 301–346 | Ablation: always deletes the same fixed fraction of non-special tokens. Tests whether learned deletion is better than uniform deletion. |

The random and fixed gates exist to answer the question: *does learning which tokens to delete actually matter, or is any deletion at the right rate equally good?*

---

## 12. What Stays the Same

Everything else in the BERT architecture is unchanged:

- The tokenizer and vocabulary
- The embedding layer
- The 12-layer transformer encoder structure
- Multi-head self-attention (except the additive mask)
- Feed-forward layers within each encoder layer
- Layer normalization and residual connections
- All task-specific heads (`MrBertForSequenceClassification`, `MrBertForMaskedLM`, etc.) — these are direct ports of the standard BERT task heads

The baseline for comparison is `modeling_bert.py`, which is a standard BERT implementation. Running with `--model-type BERT` uses this file directly, with no gate and no deletion, so results are directly comparable.

---

## 13. End-to-End Flow

Here is what happens to a single SNLI example (premise + hypothesis) during a MrBERT forward pass:

```
Input tokens: [CLS] The cat sat . [SEP] The animal rested . [SEP] [PAD] [PAD] ...
                ↓
         BertEmbeddings (unchanged)
                ↓
         Encoder Layer 0  (standard BERT layer)
         Encoder Layer 1  (standard BERT layer)
         Encoder Layer 2  (standard BERT layer)
                ↓
         Encoder Layer 3  ← has delete gate
           ├─ Standard BERT self-attention + FFN (produces hidden states)
           └─ SigmoidDeleteGate:
                LayerNorm → Linear(768→1) → ScaledSigmoid
                ┌─────────────────────────────────────────┐
                │ [CLS]  →  0.0        (protected)         │
                │ The    →  -2.1       (keep)               │
                │ cat    →  -0.3       (keep)               │
                │ sat    →  -18.4      (delete)             │
                │ .      →  -24.1      (delete)             │
                │ [SEP]  →  0.0        (protected)          │
                │ The    →  -19.2      (delete)             │
                │ animal →  -1.1       (keep)               │
                │ rested →  -0.8       (keep)               │
                │ .      →  -22.3      (delete)             │
                │ [SEP]  →  0.0        (protected)          │
                │ [PAD]  →  -30.0      (always deleted)     │
                └─────────────────────────────────────────┘
                ↓ delete_gate_mask (passed to layers 4-11)
         Encoder Layer 4  ← receives delete_gate_mask
           self-attention scores += delete_gate_mask
           → deleted tokens receive ~0 attention weight from all other tokens
         Encoder Layer 5  (same)
         ...
         Encoder Layer 11 (same)
                ↓
         [CLS] hidden state → classifier head → entailment / neutral / contradiction
```

The key insight: layers 4–11 never "see" the deleted tokens in any meaningful sense. The gate has effectively compressed the sequence, and all subsequent computation operates on the surviving tokens only.

---

## Key File Reference

| Component | File | Lines |
|---|---|---|
| Config additions | `mrbert/models/configuration_mrbert.py` | 48–71 |
| ScaledSigmoid | `mrbert/models/modeling_mrbert.py` | 160–169 |
| SigmoidDeleteGate | `mrbert/models/modeling_mrbert.py` | 178–249 |
| Special token protection | `mrbert/models/modeling_mrbert.py` | 213–232 |
| Soft deletion (attention mask) | `mrbert/models/modeling_mrbert.py` | 414–419 |
| Hard deletion | `mrbert/models/modeling_mrbert.py` | 509–553 |
| Gate in encoder | `mrbert/models/modeling_mrbert.py` | 644–647 |
| PI controller | `mrbert/training/pi_controller.py` | — |
| Deletion loss | `mrbert/training/train_mrbert.py` | — |
| Regularizer delay + combined loss | `mrbert/training/train_mrbert.py` | — |
| Gate learning rate | `mrbert/training/train_mrbert.py` | — |
| Diagnostics (compute savings, dropped tokens) | `mrbert/models/diagnostics.py` | — |
| Baseline BERT (for comparison) | `mrbert/models/modeling_bert.py` | — |