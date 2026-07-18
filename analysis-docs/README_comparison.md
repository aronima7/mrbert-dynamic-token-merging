# MrT5 vs MrBERT vs MrXLMR — Delete Gate Comparison

This document compares the delete gate mechanism as implemented in the three models:
**MrT5** (original paper, seq2seq), **MrBERT** (encoder-only, WordPiece), and **MrXLMR**
(encoder-only, SentencePiece multilingual). All three share the same high-level idea, but
differ meaningfully in architecture, token conventions, initialization, hard-deletion
mechanics, and training objectives.

---

## Table of Contents

1. [Side-by-Side Overview](#1-side-by-side-overview)
2. [Architecture Diagrams](#2-architecture-diagrams)
3. [Gate Module: What Is Shared](#3-gate-module-what-is-shared)
4. [Gate Placement and Firing Order](#4-gate-placement-and-firing-order)
5. [Special Token Handling](#5-special-token-handling)
6. [Gate Initialization](#6-gate-initialization)
7. [Soft Deletion: Attention Mask Application](#7-soft-deletion-attention-mask-application)
8. [Hard Deletion: Physical Token Removal](#8-hard-deletion-physical-token-removal)
9. [Pre-Deletion Blending](#9-pre-deletion-blending)
10. [Decoder Propagation (MrT5 Only)](#10-decoder-propagation-mrt5-only)
11. [Training Objective and Loss Functions](#11-training-objective-and-loss-functions)
12. [PI Controller](#12-pi-controller)
13. [Layer Normalization Inside the Gate](#13-layer-normalization-inside-the-gate)
14. [Softmax1](#14-softmax1)
15. [Configuration Differences](#15-configuration-differences)
16. [Token-Level Walkthrough: Same Input, Three Models](#16-token-level-walkthrough-same-input-three-models)
17. [Implementation Inheritance and Code Reuse](#17-implementation-inheritance-and-code-reuse)
18. [Key Decisions and Rationale](#18-key-decisions-and-rationale)

---

## 1. Side-by-Side Overview

| Property | MrT5 | MrBERT | MrXLMR |
|---|---|---|---|
| **Base model** | T5 (encoder-decoder) | BERT (encoder-only) | XLM-RoBERTa (encoder-only) |
| **Model type** | seq2seq | discriminative | discriminative |
| **Tokenizer** | SentencePiece (T5) | WordPiece (30K) | SentencePiece (250K) |
| **Subword prefix** | `▁` (word-initial) | `##` (continuation) | `▁` (word-initial) |
| **Vocab size** | 32,128 | 30,522 | 250,002 |
| **PAD token ID** | 0 | 0 | **1** |
| **CLS token** | None (no CLS) | `[CLS]` id=101 | `<s>` id=0 |
| **SEP token** | (punctuation list) | `[SEP]` id=102 | `</s>` id=2 |
| **MASK token** | `<extra_id_0>` | `[MASK]` id=103 | `<mask>` id=250001 |
| **Default gate layer** | 2 | 3 | 3 |
| **sigmoid_mask_scale** | **-10.0** | -30.0 | -30.0 |
| **deletion_threshold** | **None** | -15.0 | -15.0 |
| **Gate output range** | (-10, 0) | (-30, 0) | (-30, 0) |
| **Gate fires** | **BEFORE** layer's attention | AFTER layer's attention | AFTER layer's attention |
| **use_softmax1 default** | **False** | True | True |
| **Gate init (weight)** | `xavier_uniform_` | `normal_(std=0.01)` | `normal_(std=0.001)` |
| **Gate init (bias)** | **1.0** | 10.0 | 10.0 |
| **Post-init re-init** | No | No | **Yes** (explicit `_init_delete_gates()`) |
| **Pre-deletion blend** | **No** | Yes | Yes |
| **bypass_gate flag** | **No** | Yes | Yes |
| **Relative pos. bias** | Yes (T5 learned) | No (absolute) | No (absolute) |
| **Hard delete pos. bias** | **Yes (4D gather)** | N/A | N/A |
| **Decoder gate propagation** | **Yes (cross-attn)** | No | No |
| **Loss functions** | Multiple options | Mean gate only | Mean gate only |
| **Entropy regularization** | **Yes** | No | No |
| **Scores loss (attention norm)** | **Yes** | No | No |
| **Task heads** | Span corruption, XNLI, QA, char tasks | MLM, SeqClass, TokenClass, QA, MC, NSP | MLM, SeqClass, TokenClass, QA, MC |
| **Primary training task** | Span corruption (T5 pretraining) | MLM or fine-tuning | MLM or fine-tuning |

---

## 2. Architecture Diagrams

### MrT5: Encoder-Decoder with Gate in Encoder

```
Input bytes/tokens
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  T5 ENCODER                                                       │
│  Block 0 (SelfAttn + FFN)         [no gate]                       │
│  Block 1 (SelfAttn + FFN)         [no gate]                       │
│  ─────────────────────────────────────────────────────────────    │
│  Block 2 ◄── GATE FIRES FIRST BEFORE SELF-ATTENTION              │
│    ┌─ SigmoidDeleteGate(hidden) ──► delete_gate_mask              │
│    └─ MrT5SelfAttention(hidden, mask=delete_gate_mask)            │
│       MrT5CrossAttention(hidden, encoder_hidden_states=...)       │
│       T5LayerFF                                                   │
│  ─────────────────────────────────────────────────────────────    │
│  Block 3–11: SelfAttn with gate mask propagated                   │
└───────────────────────────────────────────────────────────────────┘
    │  encoder_hidden_states (possibly hard-deleted)
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  T5 DECODER                                                       │
│  Block 0–11: SelfAttn + CrossAttn(encoder_hidden, gate_mask)      │
│                              ▲                                    │
│              gate_mask also passed to cross-attention             │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
LM Head (vocab projection, seq2seq output)
```

### MrBERT: Encoder-Only with Gate After Layer

```
Input tokens (WordPiece)
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  BERT ENCODER                                                     │
│  Layer 0 (MrBertSelfAttn + BertFFN)    [no gate]                  │
│  Layer 1                               [no gate]                  │
│  Layer 2                               [no gate]                  │
│  ─────────────────────────────────────────────────────────────    │
│  Layer 3                                                          │
│    ┌─ MrBertSelfAttention(hidden)                                 │
│    │   BertSelfOutput (residual)                                  │
│    │   BertIntermediate + BertOutput (FFN)                        │
│    └─ SigmoidDeleteGate(hidden) ──► delete_gate_mask  ← fires    │
│                                          AFTER FFN               │
│  ─────────────────────────────────────────────────────────────    │
│  Layer 4–11: SelfAttn receives delete_gate_mask as additive bias  │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼  optional: _blend_pre_deletion()
    │
    ▼
Task head (MLM / SeqClass / TokenClass / QA / MC / NSP)
```

### MrXLMR: Encoder-Only with Gate After Layer (XLM-R)

```
(identical structure to MrBERT but with XLM-R components)

Input tokens (SentencePiece, 250K vocab)
    │
    ▼
RobertaEmbeddings (word + position, no token_type in practice)
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  XLM-R ENCODER                                                    │
│  Layer 0–2: MrXLMRSelfAttention + RobertaSelfOutput + Roberta FFN │
│  ─────────────────────────────────────────────────────────────    │
│  Layer 3                                                          │
│    ┌─ MrXLMRSelfAttention(hidden)                                 │
│    │   RobertaSelfOutput                                          │
│    │   RobertaIntermediate + RobertaOutput                        │
│    └─ SigmoidDeleteGate(hidden) ──► delete_gate_mask  ← fires    │
│                                          AFTER FFN               │
│  ─────────────────────────────────────────────────────────────    │
│  Layer 4–11: same as MrBERT pattern                               │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼  optional: _blend_pre_deletion()
    │
    ▼
Task head (MLM / SeqClass / TokenClass / QA / MC)
```

---

## 3. Gate Module: What Is Shared

All three models share an **identical core gate formula** in `SigmoidDeleteGate`:

```python
# Identical across MrT5 / MrBERT / MrXLMR
class ScaledSigmoid(nn.Module):
    def forward(self, input):
        return self.sigmoid_mask_scale * torch.sigmoid(-input)

class SigmoidDeleteGate(nn.Module):
    def forward(self, hidden_states, input_ids):
        h = LayerNorm(hidden_states)          # optional
        logits = Linear(768→1)(h)             # (batch, seq, 1)
        if training and use_gumbel_noise:
            logits += GumbelNoise(logits)
        gate_values = ScaledSigmoid(logits)   # maps to (scale, 0)
        # ... protect/delete special tokens ...
        return gate_values, logits
```

The `LogSigmoidDeleteGate`, `RandomDeleteGate`, and `FixedDeleteGate` ablation variants are also present in all three models with identical logic.

**Gumbel noise** is identical across all three:

```python
# modeling_mrt5.py : line 91 / modeling_mrbert.py : line 172 / modeling_mrxlmr.py : line 135
def gumbel_noise_like(x):
    eps = 3e-4 if x.dtype == torch.float16 else 1e-10
    uniform = torch.empty_like(x).uniform_(eps, 1 - eps)
    return -((-uniform.log()).log())
```

---

## 4. Gate Placement and Firing Order

This is the **most architecturally significant difference** between MrT5 and the two encoder-only models.

### MrT5: Gate Fires BEFORE Self-Attention (in the same block)

```python
# mrt5/models/modeling_mrt5.py : MrT5Block.forward(), lines 639–672
if self.has_delete_gate:
    delete_gate_values, delete_gate_logits = self.delete_gate(hidden_states, input_ids)
    delete_gate_mask = delete_gate_values

    if hard_delete:
        new_positions, delete_gate_mask = self.__get_new_positions_and_mask(...)
        hidden_states = self.__hard_delete_hidden_states(hidden_states, new_positions)

# THEN self-attention runs with the gate mask
self_attention_outputs = self.layer[0](
    hidden_states,
    ...
    delete_gate_mask=delete_gate_mask,   # ← used by THIS block's self-attention
)
```

The consequence: **the gate layer's own self-attention already sees the deletion mask**. Tokens scheduled for deletion cannot attend to anything in this same layer. The gate fires on the output of the *previous* block and is applied to the *current* block's attention.

### MrBERT / MrXLMR: Gate Fires AFTER Self-Attention and FFN

```python
# modeling_mrbert.py / modeling_mrxlmr.py : Layer.forward()
# 1. Full self-attention + FFN runs on all tokens (no mask yet)
attention_outputs = self.attention(hidden_states, ...)
layer_output = self.feed_forward_chunk(attention_output)

# 2. THEN gate fires on the post-FFN output
if self.has_delete_gate:
    delete_gate_values, delete_gate_logits = self.delete_gate(layer_output, input_ids)
    # gate_mask is passed to ALL SUBSEQUENT layers (4–11), NOT to this layer itself
```

The consequence: **the gate layer's own attention processes all tokens normally**. Only layers 4 and beyond see the deletion mask. This means layer 3 acts as a "last full-information" layer before deletion pressure begins.

### Visual Timeline

```
          MrT5 (gate at block 2)          MrBERT/MrXLMR (gate at layer 3)
          ──────────────────────          ────────────────────────────────
Block 0:  full attention, full FFN        Layer 0:  full attention, full FFN
Block 1:  full attention, full FFN        Layer 1:  full attention, full FFN
Block 2:  [GATE FIRES]                    Layer 2:  full attention, full FFN
          hard_delete (optional)          Layer 3:  full attention, full FFN
          self-attention with gate mask             [GATE FIRES after FFN]
          cross-attention with gate mask            hard_delete (optional)
Block 3:  self-attention with gate mask   Layer 4:  attention with gate mask
Block 4:  self-attention with gate mask   Layer 5:  attention with gate mask
 ...                                       ...
```

---

## 5. Special Token Handling

This is where the three models diverge most visibly.

### MrT5: PAD Only

```python
# mrt5/models/modeling_mrt5.py : SigmoidDeleteGate.forward(), line 120–123
if (input_ids == 0).any():     # T5 PAD token = 0
    pad_mask = (input_ids == 0).unsqueeze(-1)
    gate_values = torch.where(pad_mask, sigmoid_mask_scale, gate_values)
# No CLS protection (T5 has no CLS)
# No SEP protection (T5 uses relative attention biases, not segment separators)
```

T5 has no CLS/SEP tokens in the BERT sense. Segment boundaries are encoded via the sequence structure and relative position biases, so no explicit protection is needed.

### MrBERT: CLS + SEP + PAD

```python
# mrbert/models/modeling_mrbert.py : SigmoidDeleteGate.forward(), lines 214–233
gate_values[:, 0, :] = 0.0                           # [CLS] always at position 0

sep_mask = (input_ids == 102).unsqueeze(-1)           # [SEP] token_id = 102
gate_values = torch.where(sep_mask, zeros, gate_values)

if (input_ids == pad_token_id).any():                 # [PAD] token_id = 0
    pad_mask = (input_ids == 0).unsqueeze(-1)
    gate_values = torch.where(pad_mask, sigmoid_mask_scale_tensor, gate_values)
```

### MrXLMR: `<s>` + `</s>` + `<pad>` (different IDs)

```python
# mrxlmr/models/modeling_mrxlmr.py : SigmoidDeleteGate.forward(), lines 181–198
gate_values[:, 0, :] = 0.0                            # <s> always at position 0

sep_mask = (input_ids == 2).unsqueeze(-1)             # </s> token_id = 2 (not 102!)
gate_values = torch.where(sep_mask, zeros, gate_values)

if (input_ids == 1).any():                            # <pad> token_id = 1 (not 0!)
    pad_mask = (input_ids == 1).unsqueeze(-1)
    gate_values = torch.where(pad_mask, sigmoid_mask_scale_tensor, gate_values)
```

### Token ID Summary

| Token | MrT5 | MrBERT | MrXLMR |
|---|---|---|---|
| CLS | N/A | 101 | 0 |
| SEP | Protected via sep_tokens list (FixedGate) | 102 | **2** |
| PAD | **0** | **0** | **1** |
| MASK | `<extra_id_N>` | 103 | 250001 |

The MrXLMR shift of PAD from id=0 to id=1 is subtle but consequential: any code that checks `input_ids == 0` to find PAD tokens (as MrT5 and BERT do) would **silently fail** to detect padding in XLM-R sequences. The non-pad mask used in the deletion loss is also affected:

```python
# MrT5 / MrBERT trainer: non_pad_mask = input_ids != 0
# MrXLMR trainer:         non_pad_mask = input_ids != 1  ← must be 1, not 0
```

### FixedDeleteGate: Separator Token Lists

The `FixedDeleteGate` (ablation) also shows divergence in how separator positions are found:

```python
# MrT5 FixedDeleteGate (modeling_mrt5.py:182–183):
# Byte-level representation — separators are ASCII punctuation byte values
self.sep_tokens = torch.tensor([12, 13, 35, 36, 37, ..., 1])   # ~35 values

# MrBERT FixedDeleteGate (modeling_mrbert.py:314–316):
# WordPiece token IDs for BERT special tokens and common punctuation
self.sep_tokens = torch.tensor([101, 102, 103, 1012, 1010, 1029, 1000, 1001])

# MrXLMR FixedDeleteGate (modeling_mrxlmr.py):
# SentencePiece IDs for XLM-R separators (uses is_sep from input_ids == SEP_ID)
```

---

## 6. Gate Initialization

All three models face the same problem: HuggingFace's `from_pretrained()` / `post_init()` will call `_init_weights()` on **all** modules, overwriting any carefully set gate bias. Each model handles this differently.

### MrT5: Xavier weight + bias=1 at construction time (no fix)

```python
# mrt5/models/modeling_mrt5.py : SigmoidDeleteGate._init_weights(), lines 127–132
def _init_weights(self, m, init_func="xavier_uniform_"):
    if isinstance(m, nn.Linear):
        TORCH_INIT_FUNCTIONS[init_func](m.weight)   # xavier_uniform_
        m.bias.data.fill_(1)                        # bias = 1.0
```

With `sigmoid_mask_scale=-10` and `bias=1`:

```
gate = -10 × σ(-1) = -10 × 0.269 = -2.69   (not near 0, model starts with ~15% deletion)
```

The lower bias means the gate starts with a non-negligible deletion signal. Combined with `sigmoid_mask_scale=-10` (shallower range), the gate is more "active" from the start.

### MrBERT: Small normal weight + bias=10 at construction time (no post-init fix)

```python
# mrbert/models/modeling_mrbert.py : SigmoidDeleteGate._init_weights(), lines 246–249
def _init_weights(self, m, init_func: str = "xavier_uniform_"):  # init_func unused!
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)  # ← always normal_, ignores init_func
        m.bias.data.fill_(10)                          # bias = 10.0
```

With `sigmoid_mask_scale=-30` and `bias=10`:

```
gate = -30 × σ(-10) = -30 × 4.5e-5 ≈ -0.0014   (near 0, almost no deletion)
```

**Problem**: `_init_weights` is called during `__init__`, then HuggingFace's `post_init()` → `init_weights()` → `_init_weights()` is called again, potentially overwriting bias=10 with a standard init (typically bias≈0). MrBERT does not guard against this.

### MrXLMR: Even smaller normal weight + bias=10, explicitly re-applied AFTER post_init

```python
# mrxlmr/models/modeling_mrxlmr.py : MrXLMRModel.__init__(), lines 691–694
def __init__(self, config):
    super().__init__(config)
    ...
    self.post_init()           # ← HuggingFace standard init runs here
    self._init_delete_gates()  # ← explicitly overrides gate weights AFTER post_init
```

```python
# modeling_mrxlmr.py : _init_delete_gates(), lines 703–710
def _init_delete_gates(self):
    for layer in self.encoder.layer:
        if layer.has_delete_gate and hasattr(layer.delete_gate, 'feed_forward'):
            nn.init.normal_(layer.delete_gate.feed_forward.weight, mean=0.0, std=0.001)
            layer.delete_gate.feed_forward.bias.data.fill_(10.0)
```

This is the **most robust** initialization pattern. Additionally, MrXLMR uses `std=0.001` (10× smaller than MrBERT's `std=0.01`), meaning weight variation starts even smaller.

```
gate = -30 × σ(-10) ≈ -0.0014   (near 0, almost no deletion initially)
```

### Comparison: Starting Deletion Rate

| Model | σ(−bias) | gate value | Starts with |
|---|---|---|---|
| MrT5 | σ(−1) = 0.269 | -10 × 0.269 = **-2.69** | ~10–15% deletion |
| MrBERT | σ(−10) ≈ 4.5e-5 | -30 × 4.5e-5 ≈ **-0.001** | ~0% deletion (if post_init doesn't clobber) |
| MrXLMR | σ(−10) ≈ 4.5e-5 | -30 × 4.5e-5 ≈ **-0.001** | ~0% deletion (guaranteed by post_init fix) |

---

## 7. Soft Deletion: Attention Mask Application

The mechanism for applying the gate mask to attention scores is functionally identical but differs in tensor layout between T5 and the BERT-family models.

### MrT5: Squeeze(-1), then unsqueeze on the key dimension

```python
# mrt5/models/modeling_mrt5.py : MrT5Attention.forward(), lines 368–369
if delete_gate_mask is not None:
    scores = scores + delete_gate_mask.squeeze(-1).unsqueeze(-2).unsqueeze(-2)
    #  gate: (batch, seq, 1) → (batch, seq) → (batch, 1, seq) → (batch, 1, 1, seq)
    #  scores shape: (batch, n_heads, seq_query, seq_key)
    #  broadcasting: gate applied over all queries (key dimension)
```

### MrBERT / MrXLMR: Same shape transformation

```python
# modeling_mrbert.py / modeling_mrxlmr.py : SelfAttention.forward()
if delete_gate_mask is not None:
    delete_gate_mask_expanded = delete_gate_mask.squeeze(-1).unsqueeze(1).unsqueeze(1)
    #  gate: (batch, seq, 1) → (batch, seq) → (batch, 1, seq) → (batch, 1, 1, seq)
    attention_scores = attention_scores + delete_gate_mask_expanded
```

These are numerically equivalent — the dimensional operations are the same, just written differently (`unsqueeze(-2)` vs `unsqueeze(1)`).

The **effect** is identical in all three: for a deleted token at position `j`, all queries at any position `i` have their raw attention score to key `j` reduced by ~30 (or ~10 in MrT5), pushing the softmax weight for that key toward zero.

---

## 8. Hard Deletion: Physical Token Removal

Hard deletion physically removes tokens from the tensor via scatter-gather. The core algorithm is shared, but T5 requires an additional step to update the **relative position bias**.

### Shared Algorithm (all three)

```python
keep_this = delete_gate_mask > deletion_threshold         # (batch, seq) bool
target_pos = cumsum(keep_this) - 1                        # new position in shortened seq
new_len    = target_pos[:, -1].max() + 1
positions  = arange(seq_len).expand(batch_size, -1) × keep_this
src_side_pos = zeros(batch_size, new_len)
src_side_pos.scatter_add_(1, target_pos, positions)        # gather indices
hidden_states = gather(hidden_states, src_side_pos)        # (batch, new_len, d)
```

### MrT5: Also gathers position_bias (T5 relative attention, 4D)

```python
# mrt5/models/modeling_mrt5.py : MrT5Block.forward(), lines 656–668
if position_bias is not None:
    # position_bias: (batch, n_heads, seq, seq) → must be gathered in both seq dims
    new_position_bias = self.__hard_delete_4_dimensions(
        position_bias.permute(0, 2, 3, 1), new_positions)   # gather rows
    new_position_bias = self.__hard_delete_4_dimensions(
        new_position_bias.permute(0, 2, 1, 3), new_positions) # gather cols
    position_bias = new_position_bias.permute(0, 3, 2, 1)

# Also gathers attention_mask (4D in T5, shape: batch, seq, seq, 1)
new_attention_mask = self.__hard_delete_4_dimensions(
    attention_mask.permute(0, 3, 1, 2), new_positions)
attention_mask = new_attention_mask.permute(0, 2, 3, 1)
```

### MrBERT / MrXLMR: Only gather hidden states + 2D attention mask

```python
# modeling_mrbert.py / modeling_mrxlmr.py : Layer.forward()
hidden_states = self._hard_delete_hidden_states(hidden_states, new_positions)
# new_attention_mask rebuilt from scratch (1D → broadcast), not gathered
new_mask = (arange(new_len) <= target_pos[:, -1:]).float()
new_mask = (~new_mask) * -1e9
new_mask = new_mask.unsqueeze(1).unsqueeze(1)   # (batch, 1, 1, new_len)
```

The difference arises because BERT/XLM-R use **absolute positional embeddings** (baked into the embedding layer, not recomputed per layer), so there is no per-layer position tensor to gather. T5's relative position biases are computed inside each attention layer and must be explicitly shortened.

### T5-Specific: `__hard_delete_4_dimensions`

```python
# mrt5/models/modeling_mrt5.py : lines 589–591
def __hard_delete_4_dimensions(self, position_bias, positions):
    return torch.gather(
        position_bias, 1,
        positions.unsqueeze(2).unsqueeze(3).expand(
            -1, -1, position_bias.size(2), position_bias.size(3)
        )
    )
```

This operation has no equivalent in MrBERT or MrXLMR. It is the primary reason the hard deletion code in MrT5 is more complex.

---

## 9. Pre-Deletion Blending

**MrT5 does not implement pre-deletion blending.** MrBERT and MrXLMR both do, with identical code.

### Motivation

Without blending, tokens that are "deleted" by the gate still exist in the final hidden state tensor (soft deletion leaves them in memory), but their representations are corrupted: for 8 layers they attended to almost nothing. This is acceptable for CLS-based tasks (classification), but harmful for **token-level tasks** (NER, QA) where the head must decode from every position.

### Implementation in MrBERT and MrXLMR

```python
# modeling_mrbert.py / modeling_mrxlmr.py : MrBertModel._blend_pre_deletion()
@staticmethod
def _blend_pre_deletion(sequence_output, pre_deletion_hidden, delete_gate_mask, sigmoid_mask_scale):
    deletion_weight = torch.clamp(
        -delete_gate_mask / abs(sigmoid_mask_scale), 0.0, 1.0
    )
    # deletion_weight = 0 for kept tokens (gate ≈ 0)
    # deletion_weight = 1 for deleted tokens (gate ≈ sigmoid_mask_scale)
    return (1.0 - deletion_weight) * sequence_output + deletion_weight * pre_deletion_hidden
```

The pre-deletion hidden state is captured just before the gate layer runs:

```python
# modeling_mrbert.py / modeling_mrxlmr.py : Encoder.forward()
if layer_module.has_delete_gate and use_pre_deletion_blend:
    pre_deletion_hidden = hidden_states   # save before gate fires
```

### Why MrT5 Doesn't Need It

MrT5's primary task is span corruption (seq2seq). The decoder generates output tokens autoregressively from cross-attention over encoder states. Even if some encoder positions are "corrupted" by deletion, the decoder's cross-attention can attend selectively to the useful remaining tokens. The decoder never needs to decode a specific *position* of the encoder output, only its content. In contrast, QA span prediction in BERT requires the `start_logits[j]` at encoder position `j` to be meaningful even if token `j` was deleted.

---

## 10. Decoder Propagation (MrT5 Only)

MrT5's encoder gate mask propagates into the **decoder's cross-attention**:

```python
# mrt5/models/modeling_mrt5.py : MrT5Block.forward(), lines 713–726
# In the DECODER block:
if self.is_decoder:
    cross_attention_outputs = self.layer[1](
        hidden_states,
        key_value_states=encoder_hidden_states,
        ...
        delete_gate_mask=delete_gate_mask,   # ← encoder gate passed to cross-attn
    )
```

```python
# and in MrT5Block.forward(), the encoder's self-attention:
self_attention_outputs = self.layer[0](
    hidden_states,
    ...
    delete_gate_mask=None if self.is_decoder else delete_gate_mask,
    # ← gate mask ONLY for encoder self-attention, NOT decoder self-attention
)
```

This means:
- **Encoder self-attention**: sees the gate mask (deleted tokens invisible to other encoder tokens)
- **Decoder self-attention**: does NOT see the gate mask (decoder attends freely to all its own tokens)
- **Decoder cross-attention**: sees the gate mask (decoder cannot attend to deleted encoder tokens)

MrBERT and MrXLMR have no decoder, so this distinction does not apply.

---

## 11. Training Objective and Loss Functions

### MrT5: Multiple Loss Functions + Entropy + Scores Loss

```python
# mrt5/training/trainer.py : MrT5Trainer.__compute_loss(), lines 279–340
if "gate_mean" in self.loss_function:
    delete_gate_loss = delete_gate_output[non_pad_mask].mean()
elif "clamped_logits_mean" in self.loss_function:
    delete_gate_loss = torch.clamp(delete_gate_logits[non_pad_mask],
                                   min=self.deletion_threshold * 2).mean()
elif self.loss_function == "logits_mean":
    delete_gate_loss = delete_gate_logits[non_pad_mask].mean()
elif self.loss_function == "gate_var_loss":
    delete_gate_loss = -delete_gate_output[non_pad_mask].var(dim=0).mean()
```

Additionally, an optional **entropy regularization** term:

```python
# mrt5/training/trainer.py : compute_entropy_reg_loss(), lines 244–255
# H_1: per-token entropy (pushes gate toward binary decisions)
# H_2: sequence-level entropy (encourages diversity across positions)
if "entropy_reg" in self.loss_function:
    delete_gate_loss += entropy_reg_coeff_1 * H_1 + entropy_reg_coeff_2 * H_2
```

And an optional **attention score norm loss** (to prevent attention scores from exploding after deletion):

```python
# Penalizes large attention scores in layers after the gate
loss += scores_loss_coeff * __scores_loss(outputs)
```

### MrBERT / MrXLMR: Single Loss Function

```python
# train_mrbert.py / train_mrxlmr.py : compute_deletion_loss()
deletion_loss = delete_gate_output[non_pad_mask].mean()
# Only "gate_mean" — the simplest and most stable option
```

No entropy regularization. No scores loss. The simplification reflects the narrower task scope (single fine-tuning objective) and reduces hyperparameter sensitivity.

### Loss Combination

All three models use:

```
L_total = L_task + α × L_deletion
```

Where `L_task` is:
- MrT5: Seq2seq cross-entropy (span corruption)
- MrBERT/MrXLMR: Task-specific cross-entropy (MLM, seq class, token class, QA span)

---

## 12. PI Controller

All three implement a PI controller to track `target_deletion_rate`. The implementations are functionally identical but MrT5 inlines it in the trainer while MrBERT/MrXLMR extract it to a shared module.

### MrT5: Inlined in `MrT5Trainer`

```python
# mrt5/training/trainer.py : MrT5Trainer.pi_controller(), lines 262–266
def pi_controller(self, target_deletion, current_deletion):
    err = target_deletion - current_deletion
    self.p_acc = 0.9 * self.p_acc + 0.1 * self.p * err
    self.i_acc = self.i_acc + self.i * err
    return max(0.0, self.p_acc + self.i_acc)
```

Also has a simpler `i_controller` (integral-only) as an alternative.

### MrBERT / MrXLMR: Extracted to `pi_controller.py`

```python
# mrbert/training/pi_controller.py / mrxlmr/training/pi_controller.py
class PIController:
    def update(self, actual_rate):
        error = self.target_rate - actual_rate
        self.p_acc = gamma * self.p_acc + (1 - gamma) * kp * error
        self.i_acc = self.i_acc + ki * error
        return max(0.0, self.p_acc + self.i_acc)
```

The formulas are identical (γ=0.9, EMA on proportional term). The MrBERT/MrXLMR version adds a `reset()` method and a `return_state` diagnostic option.

### Comparison

| | MrT5 | MrBERT | MrXLMR |
|---|---|---|---|
| Implementation | Inlined in trainer | Shared `PIController` class | Shared `PIController` class |
| Integral-only fallback | Yes (`i_controller`) | No | No |
| `reset()` method | No | Yes | Yes |
| Default kp | 0.5 | 0.5 | 0.01 |
| Default ki | 1e-5 | 5e-5 | 1e-5 |
| `controller_step` | Yes (update every N steps) | No (every step) | No (every step) |

---

## 13. Layer Normalization Inside the Gate

All three use a layer norm before the gate linear projection when `gate_layer_norm=True`, but the norm type differs:

### MrT5: `T5LayerNorm` (RMS Norm, no bias)

```python
# mrt5/models/modeling_mrt5.py : SigmoidDeleteGate.__init__(), line 101
self.layer_norm = T5LayerNorm(config.hidden_size)
# T5LayerNorm: weight-only RMS normalization, no additive bias
# output = x / rms(x) * weight
```

T5 uses RMS normalization throughout. The gate's layer norm is consistent with this.

### MrBERT: `nn.LayerNorm` (standard, with bias)

```python
# mrbert/models/modeling_mrbert.py : SigmoidDeleteGate.__init__()
self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
# Standard LN: (x - mean) / std * weight + bias
```

### MrXLMR: `nn.LayerNorm` (standard, with bias) — same as MrBERT

```python
# mrxlmr/models/modeling_mrxlmr.py : SigmoidDeleteGate.__init__()
self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
```

XLM-R uses standard LayerNorm throughout (same as BERT), so this is consistent.

---

## 14. Softmax1

`softmax1` adds 1 to the softmax denominator, allowing attention weights to sum to less than 1. This is especially helpful after deletion, where fewer tokens are available and the model should not be forced to redistribute full attention weight.

```python
# Identical implementation in all three
def softmax1(x, dim=-1):
    maxes = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - maxes)
    return x_exp / (x_exp.sum(dim=dim, keepdim=True) + torch.exp(-maxes))
```

### Default Settings

| | MrT5 | MrBERT | MrXLMR |
|---|---|---|---|
| `use_softmax1` default | **False** | **True** | **True** |
| Source | Imported from `modeling_t5.py` | Defined locally | Defined locally |

MrT5 defaults to standard softmax, consistent with the original T5. MrBERT and MrXLMR default to softmax1, following the MrT5 paper's recommendation.

---

## 15. Configuration Differences

### `deletion_threshold` Default

```python
# MrT5Config (configuration_mrt5.py : line 10)
deletion_threshold = None      # ← must be set explicitly or defaults to None

# MrBertConfig / MrXLMRConfig
deletion_threshold = -15.0     # ← explicitly set = sigmoid_mask_scale / 2
```

A `None` threshold in MrT5 means the hard-deletion logic skips the keep-mask computation unless `deletion_threshold` is passed at inference time. In MrBERT/MrXLMR it is always defined.

### `sigmoid_mask_scale`

```python
# MrT5Config : default -10.0
# MrBertConfig / MrXLMRConfig : default -30.0
```

The deeper range in MrBERT/MrXLMR (-30 vs -10) makes the gate signal stronger: a deleted token at -30 is 3× more suppressed in attention scores than at -10. The threshold-to-scale ratio is preserved (0.5 × scale in both cases: -5 for MrT5, -15 for MrBERT/MrXLMR).

### `bypass_gate`

```python
# MrT5Config : no bypass_gate field
# MrBertConfig / MrXLMRConfig : bypass_gate = False (optional, for ablations)
```

MrBERT and MrXLMR support `bypass_gate=True` to completely disable the gate at inference time, allowing a zero-deletion ablation without retraining. MrT5 achieves the same by using `deletion_type=None`.

### `use_pre_deletion_blend`

```python
# MrT5Config : not present
# MrBertConfig / MrXLMRConfig : use_pre_deletion_blend = True
```

### `delete_gate_layer` Default

```python
# MrT5Config : 2   (0-indexed, among 12 encoder blocks)
# MrBertConfig : 3
# MrXLMRConfig : 3
```

### Full Config Comparison

| Config field | MrT5 default | MrBERT default | MrXLMR default |
|---|---|---|---|
| `sigmoid_mask_scale` | **-10.0** | -30.0 | -30.0 |
| `deletion_threshold` | **None** | -15.0 | -15.0 |
| `delete_gate_layer` | **2** | 3 | 3 |
| `use_softmax1` | **False** | True | True |
| `use_gumbel_noise` | False | False | False |
| `use_pre_deletion_blend` | N/A | True | True |
| `bypass_gate` | N/A | False | False |
| `gate_layer_norm` | True | True | True |
| `deletion_type` | **None** | scaled_sigmoid | scaled_sigmoid |

---

## 16. Token-Level Walkthrough: Same Input, Three Models

Input sentence pair: "The cat sat. A dog ran."

### MrT5 Tokenization (SentencePiece, T5 byte-level)

```
Input IDs (T5 tokenizer, no CLS/SEP):
  ▁The   ▁cat   ▁sat  .    ▁A   ▁dog   ▁ran  .   <eos>
   209    1515   3224  5    71   1782    6501   5    1
```

Gate protection:
- `<eos>` (id=1) → treated as PAD → forced to gate=-10
- No CLS or SEP protection
- Gate fires at block 2, **before** self-attention in that block

### MrBERT Tokenization (WordPiece)

```
Input IDs (BERT tokenizer):
  [CLS]  The   cat   sat    .     A   dog  ran    .  [SEP]  <PAD>…
   101    1996  4937  3507  1012  1037 3899 2743  1012  102    0…
```

Gate protection:
- `[CLS]` at position 0 → forced gate=0 (KEPT)
- `[SEP]` id=102 → forced gate=0 (KEPT)
- `[PAD]` id=0 → forced gate=-30 (DELETED)

### MrXLMR Tokenization (SentencePiece, XLM-R 250K)

```
Input IDs (XLM-R tokenizer):
  <s>    ▁The   ▁cat   ▁sat    .    ▁A    ▁dog   ▁ran    .   </s>  <pad>…
   0       83   1234   4421    5    183   2156   4876     5    2      1…
```

Gate protection:
- `<s>` at position 0 → forced gate=0 (KEPT)
- `</s>` id=2 → forced gate=0 (KEPT)
- `<pad>` id=1 → forced gate=-30 (DELETED)

### Gate Values After Training (illustrative, 30% deletion target)

```
             MrT5          MrBERT          MrXLMR
Token        gate∈(-10,0)  gate∈(-30,0)    gate∈(-30,0)
──────────   ──────────    ────────────    ────────────
[start]      N/A            0.0 (forced)    0.0 (forced)
▁The/The     -8.5  [DEL]   -27.1 [DEL]    -26.8 [DEL]
▁cat/cat     -0.3  [KEEP]  -0.8  [KEEP]   -0.7  [KEEP]
▁sat/sat     -0.4  [KEEP]  -1.1  [KEEP]   -1.2  [KEEP]
./.          -8.8  [DEL]   -28.4 [DEL]    -27.9 [DEL]
▁A/A         -9.1  [DEL]   -28.9 [DEL]    -28.6 [DEL]
▁dog/dog     -0.3  [KEEP]  -0.9  [KEEP]   -0.6  [KEEP]
▁ran/ran     -0.5  [KEEP]  -1.4  [KEEP]   -0.8  [KEEP]
./.          -8.8  [DEL]   -28.4 [DEL]    -27.8 [DEL]
[end]        -10.0 (PAD)    0.0 (forced)    0.0 (forced)

Threshold:   > -5.0         > -15.0         > -15.0
```

All three agree on what to delete (articles, punctuation) and keep (content words). The absolute values differ due to `sigmoid_mask_scale` (-10 vs -30).

---

## 17. Implementation Inheritance and Code Reuse

```
Inheritance chain:
──────────────────────────────────────────────────────────
MrT5          MrBERT                   MrXLMR
──────────────────────────────────────────────────────────
T5ForCondGen  BertPreTrainedModel       XLMRobertaPreTrainedModel
  T5Stack       BertModel                 MrXLMRModel
    MrT5Block     MrBertLayer               MrXLMRLayer
      MrT5          MrBertAttention           MrXLMRAttention
      SelfAttn        MrBertSelf                MrXLMRSelf
                      BertSelfOutput ←reuse     RobertaSelfOutput ←reuse
                      BertIntermediate ←reuse   RobertaIntermediate ←reuse
                      BertOutput ←reuse         RobertaOutput ←reuse
                    BertEmbeddings ←reuse     RobertaEmbeddings ←reuse
                    BertPooler ←reuse         RobertaPooler ←reuse
                    BertOnlyMLMHead ←reuse    RobertaLMHead ←reuse
```

### Modules Reused Unchanged

| Module | MrBERT source | MrXLMR source |
|---|---|---|
| Embeddings | `BertEmbeddings` | `RobertaEmbeddings` |
| Self output projection | `BertSelfOutput` | `RobertaSelfOutput` |
| FFN first layer | `BertIntermediate` | `RobertaIntermediate` |
| FFN second layer | `BertOutput` | `RobertaOutput` |
| Pooler | `BertPooler` | `RobertaPooler` |
| MLM head | `BertOnlyMLMHead` | `RobertaLMHead` |

Both MrBERT and MrXLMR **replace** only the self-attention module (to add gate mask support) and the encoder (to manage gate propagation). Everything else is reused directly from HuggingFace.

### Weight Loading Convention

A subtle but important difference:

```python
# MrBERT task heads:
self.bert = MrBertModel(config)   # ← named "bert" to match BERT weight keys
                                   #   e.g. bert.encoder.layer.0.attention.self.query.weight

# MrXLMR task heads:
self.roberta = MrXLMRModel(config) # ← named "roberta" to match XLM-R weight keys
                                    #   e.g. roberta.encoder.layer.0.attention.self.query.weight

# MrT5 inherits from T5ForConditionalGeneration directly — no rename needed
```

---

## 18. Key Decisions and Rationale

### Why does MrXLMR re-initialize gates after `post_init()`?

HuggingFace's `from_pretrained()` calls `init_weights()` → `_init_weights()` on all modules after loading the pretrained checkpoint. If `_init_weights` does not explicitly skip the gate (or if the gate weights are not in the checkpoint), it will reset the gate to standard init, overwriting the carefully chosen `bias=10.0`. MrBERT silently has this bug; MrXLMR explicitly fixes it by calling `_init_delete_gates()` after `post_init()`.

### Why is MrXLMR's `std=0.001` vs MrBERT's `std=0.01`?

Smaller weight std means the gate's logit variation is tighter initially, producing gate values even closer to 0 (completely kept). This gives the PI controller more room to gradually ramp up deletion pressure without the gate immediately jumping to high deletion rates in the first few steps.

### Why does MrT5 fire the gate before attention, but MrBERT/MrXLMR fire it after?

MrT5 is the **original** design from the paper. MrBERT/MrXLMR made a deliberate change: by firing the gate after the complete layer (attention + FFN), the gate has access to richer contextual information for the deletion decision. The gate has "seen" the intra-layer attention patterns before deciding what to delete, which may produce higher-quality deletion decisions.

### Why does MrXLMR have pre-deletion blending but MrT5 does not?

The tasks differ. MrT5's primary task (span corruption, XNLI) uses either the decoder's output or the encoder's CLS-like final state — neither requires per-position accuracy at deleted positions. MrBERT/MrXLMR support QA and NER, where every token position in the final hidden state must be decodable. Blending recovers meaningful representations at deleted positions without preventing the efficiency gain.

### Why does MrT5's `sigmoid_mask_scale=-10` vs `-30` in MrBERT/MrXLMR?

The MrT5 paper uses -10. MrBERT and MrXLMR use -30 (with threshold at -15) to produce a harder decision boundary — at -30 the deleted token's attention weight is `exp(-30) ≈ 9e-14`, essentially zero. At -10 it is `exp(-10) ≈ 4.5e-5`, still non-negligible. The larger scale may be beneficial for encoder-only models where there is no decoder to compensate for imperfect deletion.