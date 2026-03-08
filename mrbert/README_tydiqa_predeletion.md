# Pre-Deletion Blending for QA Tasks in MrBERT

## Background: The Delete Gate

MrBERT adapts the MrT5 delete gate mechanism to BERT. After a configurable encoder layer
(`delete_gate_layer`, default: 3), a learned gate scores every token in the sequence. Low-scoring
tokens are **soft-deleted** — their gate value is added as a large negative attention bias to all
subsequent encoder layers, causing them to be effectively ignored by later self-attention.

The gate is implemented in `modeling_mrbert.py:SigmoidDeleteGate` (line 179). The core activation
is `ScaledSigmoid` (line 161):

```python
# modeling_mrbert.py:168
def forward(self, input: torch.Tensor) -> torch.Tensor:
    return self.sigmoid_mask_scale * torch.sigmoid(-input)
```

With `sigmoid_mask_scale = -30.0` (the default), gate values fall in **[−30, 0]**:
- `gate ≈ 0` → token is **kept**; no bias is added to attention
- `gate ≈ −30` → token is **deleted**; attention scores for that token are shifted by −30
  (effectively −∞ after softmax), making the token invisible to subsequent layers

---

## The Problem: Deleted Answer Span Tokens in QA

In extractive QA (TyDi QA, SQuAD), the model must predict `start_position` and `end_position`
as indices into the final encoder hidden states. The span prediction head reads the
**layer-11 (final layer) representation** of every token and scores it:

```python
# modeling_mrbert.py:1254–1260
sequence_output = outputs.last_hidden_state   # (batch, seq, hidden)
logits = self.qa_outputs(sequence_output)
start_logits, end_logits = logits.split(1, dim=-1)
```

When a token is soft-deleted at layer 3, it still occupies a position in the sequence (soft deletion
does not physically remove tokens — that is hard deletion). However, after layer 3, all subsequent
attention layers receive that token's attention bias of −30. This means:

1. **No other token attends to the deleted token** — the deleted token never contributes to
   other tokens' representations through attention.
2. **The deleted token's own self-attention is also impaired** — it can still attend to others,
   but its representation is never reinforced by incoming attention from the rest of the sequence.
3. **The layer-11 representation of a deleted token is corrupted** — it has been computed with
   8 layers of near-zero incoming attention, making it an unreliable signal for span prediction.

If the gate deletes a token that is part of the ground-truth answer span (e.g., the start or end
token), the QA head reads a corrupted representation at that position and produces low logits
even when the correct answer is present. This is the primary driver of the large EM/F1 gap
observed for MrBERT vs. BERT on TyDi QA:

| Model | span_em | start_acc | end_acc | Deletion Rate |
|-------|---------|-----------|---------|---------------|
| BERT baseline | 0.38 | 0.56 | 0.50 | — |
| MrBERT 30% layer3 (no blend) | 0.10 | 0.20 | 0.18 | ~61% (collapsed) |
| MrBERT 30% layer3 + blend | 0.30 | 0.46 | 0.39 | ~25% |
| MrBERT 30% layer9 + blend | 0.35 | 0.49 | 0.50 | ~22% |

---

## The Fix: Pre-Deletion Blending

### Core Idea

Before the gate fires, the token representations are **fully attended** — every token has had
3 unobstructed self-attention layers. These layer-3 representations are saved just before the
gate is applied. For any token that gets deleted, we substitute its final-layer (corrupted)
representation with this saved pre-gate representation.

The substitution is **soft** and proportional to the gate's deletion strength:

```
deletion_weight = clamp( -gate_mask / |sigmoid_mask_scale|, 0, 1 )
output = (1 - deletion_weight) × final_layer_hidden + deletion_weight × pre_deletion_hidden
```

| Gate value | `deletion_weight` | What the QA head sees |
|---|---|---|
| 0.0 (kept) | 0.0 | Pure final-layer (layer 11) representation |
| −30.0 (fully deleted) | 1.0 | Pure pre-deletion (layer `gate_layer − 1`) representation |
| −15.0 (half deleted) | 0.5 | 50/50 blend |

### Why This Works at Both Training and Test Time

No ground-truth span positions are used in the blending calculation. The blend is driven
entirely by the gate values themselves, which are available whenever the model runs a forward
pass. This means:

- **Training**: the blend helps the loss signal reach the gate through well-formed span
  representations, giving the gate a cleaner gradient.
- **Inference**: deleted answer tokens are still readable by the span head, recovering
  accuracy without changing the model architecture or requiring any post-processing.

### Why It Has No Effect on Sequence Classification

`[CLS]` (position 0) is always explicitly protected from deletion:

```python
# modeling_mrbert.py:216
gate_values[:, 0, :] = 0.0   # CLS is never deleted
```

Since `deletion_weight` for `[CLS]` is always 0, the blend reduces to the identity for
classification tasks — `sequence_output` is unchanged.

---

## Code Walkthrough

### 1. Capture: Pre-Deletion Hidden States (`MrBertEncoder.forward`, line 684)

```python
# modeling_mrbert.py:684–687
# Save hidden states just before the gate layer — these are the last
# fully-attended representations before any token is deleted.
if layer_module.has_delete_gate and not layer_module.bypass_gate \
        and getattr(self.config, "use_pre_deletion_blend", True):
    pre_deletion_hidden = hidden_states
```

This runs at the top of the encoder loop, *before* `layer_module(hidden_states, ...)` is called
for the gate layer. So `pre_deletion_hidden` holds the hidden states from the previous layer
(i.e., layer `delete_gate_layer − 1`), which are still fully unmasked.

### 2. Propagation: Output Dataclass (`MrBertBaseModelOutput`, line 55)

```python
# modeling_mrbert.py:62
pre_deletion_hidden: Optional[torch.FloatTensor] = None
```

The saved tensor is returned as part of the model output alongside `delete_gate_mask`,
making it accessible to every task head.

### 3. Blend Function (`MrBertModel._blend_pre_deletion`, line 893)

```python
# modeling_mrbert.py:893–918
@staticmethod
def _blend_pre_deletion(
    sequence_output: torch.Tensor,
    pre_deletion_hidden: Optional[torch.FloatTensor],
    delete_gate_mask: Optional[torch.FloatTensor],
    sigmoid_mask_scale: float,
) -> torch.Tensor:
    if pre_deletion_hidden is None or delete_gate_mask is None:
        return sequence_output
    deletion_weight = torch.clamp(
        -delete_gate_mask / abs(sigmoid_mask_scale), 0.0, 1.0
    )  # (batch, seq, 1)
    return (1.0 - deletion_weight) * sequence_output + deletion_weight * pre_deletion_hidden
```

### 4. Application: QA Head (`MrBertForQuestionAnswering.forward`, line 1254)

```python
# modeling_mrbert.py:1254–1260
sequence_output = outputs.last_hidden_state
if getattr(self.config, "use_pre_deletion_blend", True):
    sequence_output = self.bert._blend_pre_deletion(
        sequence_output, outputs.pre_deletion_hidden,
        outputs.delete_gate_mask, self.config.sigmoid_mask_scale,
    )
logits = self.qa_outputs(sequence_output)
```

The same pattern is applied in `MrBertForMaskedLM` (line 982) and
`MrBertForTokenClassification` (line 1172).

---

## Configuration

Pre-deletion blending is controlled by `MrBertConfig.use_pre_deletion_blend` (default: `True`).

```python
# configuration_mrbert.py:65
use_pre_deletion_blend: bool = True
```

### Modal training flag

```bash
# Enable (default — omit flag entirely, or pass explicitly)
modal run --detach train_modal.py::main \
  --task question_answering --dataset-name local_tydiqa \
  --model-type MrBERT --num-epochs 3 --batch-size 16 \
  --target-deletion-rate 0.3 --regularizer-delay 100 \
  --wandb-run-name mrbert-tydiqa-30pct-predel --wandb-project mrbert-tydiqa

# Disable (ablation: compare with and without blend)
modal run --detach train_modal.py::main \
  --task question_answering --dataset-name local_tydiqa \
  --model-type MrBERT --num-epochs 3 --batch-size 16 \
  --target-deletion-rate 0.3 --regularizer-delay 100 \
  --no-use-pre-deletion-blend \
  --wandb-run-name mrbert-tydiqa-30pct-nopredel --wandb-project mrbert-tydiqa
```

---

## Limitations and Notes

- **Pre-deletion representations are shallower.** Tokens saved at layer `delete_gate_layer − 1`
  have had fewer self-attention passes than the final-layer representations. For a gate at layer 3
  this means only 3 layers of context; a gate at layer 9 saves richer representations (9 layers).
  This is one reason the layer-9 variant performs better on TyDi QA.

- **Blend is soft, not gated.** Partially-deleted tokens (gate between −30 and 0) get an
  interpolated representation. This is intentional — it provides a smooth gradient during
  training rather than a hard switch.

- **No effect on hard deletion.** In hard-deletion mode, tokens are physically removed from
  the sequence before reaching the QA head. The blend does not apply because deleted tokens
  are no longer present in `sequence_output`. Hard deletion is not used in the default training
  configuration (`hard_delete_train_prob = 0.0`).