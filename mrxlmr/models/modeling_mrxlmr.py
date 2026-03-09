# modeling_mrxlmr.py
# Description: XLM-RoBERTa with MrT5-style delete gates.
# Adapted from modeling_mrbert.py (which in turn is adapted from HuggingFace modeling_bert.py).
# XLM-R special token IDs: <s>=0 (CLS), <pad>=1, </s>=2 (SEP), <mask>=250001
"""
MrXLMR model - XLM-RoBERTa with MrT5-style delete gates.

This implementation adapts the delete gate mechanism from the MrT5 paper
to the XLM-RoBERTa architecture for encoder-only tasks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import XLMRobertaPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
    MultipleChoiceModelOutput,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaPooler,
    RobertaSelfOutput,
    RobertaIntermediate,
    RobertaOutput,
    RobertaLMHead,
)
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import logging

from configuration_mrxlmr import MrXLMRConfig


logger = logging.get_logger(__name__)

# XLM-R special token IDs
XLM_R_CLS_TOKEN_ID = 0   # <s>
XLM_R_PAD_TOKEN_ID = 1   # <pad>
XLM_R_SEP_TOKEN_ID = 2   # </s>


# =============================================================================
# Custom Output Dataclasses
# =============================================================================

@dataclass
class MrXLMRBaseModelOutput(BaseModelOutputWithPoolingAndCrossAttentions):
    """Output type for MrXLMRModel with delete gate information."""
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None
    pre_deletion_hidden: Optional[torch.FloatTensor] = None


@dataclass
class MrXLMRMaskedLMOutput(MaskedLMOutput):
    """Output type for MrXLMRForMaskedLM with delete gate information."""
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrXLMRSequenceClassifierOutput(SequenceClassifierOutput):
    """Output type for MrXLMRForSequenceClassification with delete gate information."""
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrXLMRTokenClassifierOutput(TokenClassifierOutput):
    """Output type for MrXLMRForTokenClassification with delete gate information."""
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrXLMRQuestionAnsweringOutput(QuestionAnsweringModelOutput):
    """Output type for MrXLMRForQuestionAnswering with delete gate information."""
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrXLMRMultipleChoiceOutput(MultipleChoiceModelOutput):
    """Output type for MrXLMRForMultipleChoice with delete gate information."""
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


# =============================================================================
# Softmax1 Function (from MrT5)
# =============================================================================

def softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax with n+1 in the denominator for smoother attention distributions.
    This allows attention weights to sum to less than 1.
    """
    maxes = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    # Add 1 to denominator (equivalent to adding exp(0) = 1)
    return x_exp / (x_exp_sum + torch.exp(-maxes))


# =============================================================================
# Delete Gate Modules
# =============================================================================

class ScaledSigmoid(nn.Module):
    """Sigmoid activation scaled by a constant factor."""

    def __init__(self, sigmoid_mask_scale: float):
        super().__init__()
        self.sigmoid_mask_scale = sigmoid_mask_scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_mask_scale * torch.sigmoid(-input)


def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    """Generate Gumbel noise with the same shape as input tensor."""
    eps = 3e-4 if x.dtype == torch.float16 else 1e-10
    uniform = torch.empty_like(x).uniform_(eps, 1 - eps)
    return -((-uniform.log()).log())


class SigmoidDeleteGate(nn.Module):
    """
    Delete gate using scaled sigmoid activation (adapted for XLM-R).

    Protects:
      - <s>  (position 0, CLS equivalent, token_id=0)
      - </s> (SEP equivalent, token_id=2)
    Always deletes:
      - <pad> (token_id=1)
    """

    def __init__(self, config: MrXLMRConfig):
        super().__init__()
        self.has_layer_norm = config.gate_layer_norm
        if self.has_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.feed_forward)
        self.activation = ScaledSigmoid(config.sigmoid_mask_scale)
        self.use_gumbel_noise = config.use_gumbel_noise
        self.pad_token_id = config.pad_token_id  # 1 for XLM-R

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.has_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        delete_gate_logits = self.feed_forward(hidden_states)

        # Add Gumbel noise during training for exploration
        if self.training and self.use_gumbel_noise:
            gumbel_noise = gumbel_noise_like(delete_gate_logits)
            delete_gate_logits = delete_gate_logits + gumbel_noise

        gate_values = self.activation(delete_gate_logits)

        # CRITICAL: Never delete <s> token (position 0) - it's the CLS equivalent
        gate_values[:, 0, :] = 0.0

        # Protect </s> tokens (token_id=2 for XLM-R, equivalent to [SEP] in BERT)
        sep_mask = (input_ids == XLM_R_SEP_TOKEN_ID).unsqueeze(-1)
        gate_values = torch.where(
            sep_mask,
            torch.zeros_like(gate_values),
            gate_values
        )

        # Always fully delete <pad> tokens (token_id=1 for XLM-R)
        if self.pad_token_id is not None and (input_ids == self.pad_token_id).any():
            pad_mask = (input_ids == self.pad_token_id).unsqueeze(-1)
            gate_values = torch.where(
                pad_mask,
                torch.tensor(self.activation.sigmoid_mask_scale, device=gate_values.device, dtype=gate_values.dtype),
                gate_values
            )

        return gate_values, delete_gate_logits

    def _init_weights(self, m: nn.Module):
        """Initialize weights. Bias is set high to strongly bias toward keeping tokens initially.

        With ScaledSigmoid(-30) and bias=10:
        gate_value = -30 * sigmoid(-10) ≈ -30 * 0.000045 ≈ -0.00135 (very close to 0 = keep)

        With deletion_threshold = -15, gate_value > -15 means KEEP the token.
        The model must learn to delete tokens, not start by deleting them.
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            m.bias.data.fill_(10)  # Large positive bias to ensure tokens are kept initially


class LogSigmoidDeleteGate(SigmoidDeleteGate):
    """Delete gate using log sigmoid activation."""

    def __init__(self, config: MrXLMRConfig):
        super().__init__(config)
        self.activation = nn.LogSigmoid()


class RandomDeleteGate(nn.Module):
    """Random delete gate for ablation studies."""

    def __init__(self, config: MrXLMRConfig):
        super().__init__()
        self.sigmoid_mask_scale = config.sigmoid_mask_scale
        self.random_deletion_probability = config.random_deletion_probability

    def _random_mask_tensor(self, x: torch.Tensor, n: int) -> torch.Tensor:
        target_shape = (x.shape[0], x.shape[1], 1)
        total_elements = x.shape[0] * x.shape[1]

        flat_tensor = torch.zeros(total_elements, dtype=torch.float32, device=x.device)
        indices = torch.randperm(total_elements, device=x.device)[:n]
        flat_tensor[indices] = 1.0

        return flat_tensor.view(target_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        deletion_percentage = np.random.normal(
            loc=self.random_deletion_probability,
            scale=0.05
        )
        deletion_percentage = np.clip(deletion_percentage, 0.0, 1.0)
        n_deletions = int(deletion_percentage * hidden_states.shape[0] * hidden_states.shape[1])

        random_mask = self._random_mask_tensor(hidden_states, n_deletions)
        delete_gate_mask = random_mask * self.sigmoid_mask_scale

        return delete_gate_mask, delete_gate_mask


class FixedDeleteGate(nn.Module):
    """Fixed delete gate for ablation studies. Deletes a fixed fraction of tokens."""

    def __init__(self, config: MrXLMRConfig):
        super().__init__()
        self.sigmoid_mask_scale = config.sigmoid_mask_scale
        self.fixed_deletion_amount = config.fixed_deletion_amount
        # XLM-R special token IDs to protect: <s>=0, <pad>=1, </s>=2, <unk>=3, <mask>=250001
        self.sep_tokens = torch.tensor([0, 1, 2, 3, 250001])

    def _create_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        self.sep_tokens = self.sep_tokens.to(device)

        mask = torch.zeros((batch_size, seq_len), device=device)

        is_sep = torch.isin(input_ids, self.sep_tokens)

        for b in range(batch_size):
            non_sep_indices = (~is_sep[b]).nonzero(as_tuple=True)[0]
            n_to_delete = int(len(non_sep_indices) * self.fixed_deletion_amount)
            if n_to_delete > 0:
                perm = torch.randperm(len(non_sep_indices), device=device)[:n_to_delete]
                delete_indices = non_sep_indices[perm]
                mask[b, delete_indices] = 1.0

        return mask * self.sigmoid_mask_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delete_gate_mask = self._create_mask(input_ids).unsqueeze(-1)
        return delete_gate_mask, delete_gate_mask


# =============================================================================
# Modified Attention with Delete Gate Support
# =============================================================================

class MrXLMRSelfAttention(nn.Module):
    """
    XLM-R self-attention with delete gate mask support.

    Extends standard self-attention to apply delete gate masks to attention scores,
    effectively "deleting" certain tokens by preventing other tokens from attending to them.
    """

    def __init__(self, config: MrXLMRConfig, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size ** -0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        # MrT5 addition
        self.use_softmax1 = getattr(config, "use_softmax1", False)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape from (batch, seq, hidden) to (batch, heads, seq, head_size)."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        delete_gate_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.scaling

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply delete gate mask (shape: batch, seq, 1 -> batch, 1, 1, seq)
        if delete_gate_mask is not None:
            delete_gate_mask_expanded = delete_gate_mask.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores + delete_gate_mask_expanded

        if self.use_softmax1:
            attention_probs = softmax1(attention_scores, dim=-1)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class MrXLMRAttention(nn.Module):
    """XLM-R attention block with delete gate support."""

    def __init__(self, config: MrXLMRConfig):
        super().__init__()
        self.self = MrXLMRSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        delete_gate_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            delete_gate_mask=delete_gate_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# =============================================================================
# Modified Layer with Delete Gate
# =============================================================================

class MrXLMRLayer(nn.Module):
    """
    XLM-R layer with optional delete gate.

    The delete gate is applied after the layer's computations to determine
    which tokens should be deleted for subsequent layers.
    """

    def __init__(self, config: MrXLMRConfig, has_delete_gate: bool = False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MrXLMRAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

        self.has_delete_gate = has_delete_gate
        self.bypass_gate = getattr(config, "bypass_gate", False)
        if has_delete_gate:
            if config.deletion_type == "scaled_sigmoid":
                self.delete_gate = SigmoidDeleteGate(config)
            elif config.deletion_type == "log_sigmoid":
                self.delete_gate = LogSigmoidDeleteGate(config)
            elif config.deletion_type == "random":
                self.delete_gate = RandomDeleteGate(config)
            elif config.deletion_type == "fixed":
                self.delete_gate = FixedDeleteGate(config)
            else:
                raise ValueError(f"Invalid deletion type: {config.deletion_type}")

        self.sigmoid_mask_scale = config.sigmoid_mask_scale
        self.deletion_threshold = config.deletion_threshold

    def _get_new_positions_and_mask(
        self,
        batch_size: int,
        seq_len: int,
        delete_gate_mask: torch.Tensor,
        deletion_threshold: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute new token positions after hard deletion."""
        delete_gate_mask = delete_gate_mask.squeeze(-1)

        threshold = deletion_threshold if deletion_threshold is not None else self.deletion_threshold
        keep_this = delete_gate_mask > threshold

        target_pos = torch.cumsum(keep_this.int(), dim=1) - 1
        new_len = max(target_pos[:, -1].max().item() + 1, 1)
        target_pos = target_pos.clamp(min=0)

        positions = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
        positions = positions * keep_this.int()

        src_side_pos = torch.zeros(batch_size, new_len, device=device, dtype=torch.int32)
        src_side_pos.scatter_add_(1, target_pos.long(), positions)

        new_mask = torch.arange(new_len, device=device).unsqueeze(0).expand(batch_size, -1) <= target_pos[:, -1:]
        new_mask = (~new_mask).float() * -1e9
        new_mask = new_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, new_len)

        return src_side_pos.long(), new_mask

    def _hard_delete_hidden_states(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Remove tokens from hidden states based on positions."""
        return torch.gather(
            hidden_states,
            1,
            positions.unsqueeze(2).expand(-1, -1, hidden_states.size(2))
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        delete_gate_mask: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, ...]:
        delete_gate_values = None
        delete_gate_logits = None
        new_attention_mask = attention_mask

        if self.has_delete_gate and input_ids is not None and not self.bypass_gate:
            delete_gate_values, delete_gate_logits = self.delete_gate(hidden_states, input_ids)
            delete_gate_mask = delete_gate_values

            threshold = deletion_threshold if deletion_threshold is not None else self.deletion_threshold
            if threshold is not None and (delete_gate_values < threshold).all():
                raise ValueError(
                    "All tokens would be deleted. Adjust deletion_threshold or sigmoid_mask_scale."
                )

            if hard_delete and threshold is not None:
                new_positions, new_attention_mask = self._get_new_positions_and_mask(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    delete_gate_mask,
                    threshold,
                    hidden_states.device,
                )
                hidden_states = self._hard_delete_hidden_states(hidden_states, new_positions)
                attention_mask = new_attention_mask

        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            delete_gate_mask=delete_gate_mask if not hard_delete else None,
        )
        attention_output = attention_outputs[0]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output
        )

        outputs = (layer_output,) + attention_outputs[1:]

        # Gate outputs always appended as last 4 items when has_delete_gate=True:
        #   [-4] delete_gate_values
        #   [-3] delete_gate_logits
        #   [-2] delete_gate_mask
        #   [-1] new_attention_mask
        if self.has_delete_gate:
            outputs = outputs + (delete_gate_values, delete_gate_logits, delete_gate_mask, new_attention_mask)

        return outputs

    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# =============================================================================
# Modified Encoder
# =============================================================================

class MrXLMREncoder(nn.Module):
    """
    XLM-R encoder with delete gate support.

    One layer (specified by config.delete_gate_layer) contains a delete gate
    that learns which tokens to remove from the sequence.
    """

    def __init__(self, config: MrXLMRConfig):
        super().__init__()
        self.config = config

        self.layer = nn.ModuleList([
            MrXLMRLayer(config, has_delete_gate=(i == config.delete_gate_layer))
            for i in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, ...]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        delete_gate_mask = None
        delete_gate_output = None
        delete_gate_logits = None
        final_attention_mask = attention_mask
        pre_deletion_hidden = None  # hidden states saved just before the gate fires

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            # Save hidden states just before the gate layer fires
            if (layer_module.has_delete_gate
                    and not layer_module.bypass_gate
                    and getattr(self.config, "use_pre_deletion_blend", True)):
                pre_deletion_hidden = hidden_states

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    delete_gate_mask,
                    input_ids,
                    hard_delete,
                    deletion_threshold,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                    delete_gate_mask=delete_gate_mask,
                    input_ids=input_ids,
                    hard_delete=hard_delete,
                    deletion_threshold=deletion_threshold,
                )

            hidden_states = layer_outputs[0]

            if layer_module.has_delete_gate:
                delete_gate_output, delete_gate_logits, delete_gate_mask, new_attention_mask = (
                    layer_outputs[-4], layer_outputs[-3], layer_outputs[-2], layer_outputs[-1]
                )
                if hard_delete:
                    attention_mask = new_attention_mask
                    final_attention_mask = attention_mask

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return (
            hidden_states,
            all_hidden_states,
            all_self_attentions,
            delete_gate_mask,
            delete_gate_output,
            delete_gate_logits,
            final_attention_mask,
            pre_deletion_hidden,
        )


# =============================================================================
# MrXLMR Model
# =============================================================================

class MrXLMRModel(XLMRobertaPreTrainedModel):
    """
    MrXLMR model with delete gate mechanism.

    XLM-RoBERTa with a learnable delete gate to selectively remove tokens
    during encoding, reducing computational cost.

    Attribute is named `roberta` to match XLM-R pretrained weight layout,
    ensuring correct weight loading via from_pretrained.
    """

    config_class = MrXLMRConfig

    def __init__(self, config: MrXLMRConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = MrXLMREncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.post_init()

        # CRITICAL: Re-initialize delete gate AFTER post_init() to ensure correct initialization
        self._init_delete_gates()

    def _init_delete_gates(self):
        """Initialize delete gate weights to strongly favor keeping tokens.

        Must be called AFTER post_init() because post_init() overwrites
        all module weights with the standard initialization.
        """
        found_gate = False
        for i, layer in enumerate(self.encoder.layer):
            if layer.has_delete_gate:
                gate = layer.delete_gate
                if hasattr(gate, 'feed_forward'):
                    nn.init.normal_(gate.feed_forward.weight, mean=0.0, std=0.001)
                    gate.feed_forward.bias.data.fill_(10.0)
                    found_gate = True
                    print(f"[MrXLMR] Initialized delete gate at layer {i} with bias=10.0, weight_std=0.001")

        if not found_gate:
            print(f"[MrXLMR] WARNING: No delete gate found! delete_gate_layer={self.config.delete_gate_layer}")

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _convert_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert attention mask to additive mask for attention scores."""
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            extended_attention_mask = attention_mask

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

        return extended_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
    ) -> MrXLMRBaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        extended_attention_mask = self._convert_attention_mask(
            attention_mask, input_shape, self.dtype
        )

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            input_ids=input_ids,
            hard_delete=hard_delete,
            deletion_threshold=deletion_threshold,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return MrXLMRBaseModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs[1],
            attentions=encoder_outputs[2],
            delete_gate_mask=encoder_outputs[3],
            delete_gate_output=encoder_outputs[4],
            delete_gate_logits=encoder_outputs[5],
            pre_deletion_hidden=encoder_outputs[7],
        )

    @staticmethod
    def _blend_pre_deletion(
        sequence_output: torch.Tensor,
        pre_deletion_hidden: Optional[torch.FloatTensor],
        delete_gate_mask: Optional[torch.FloatTensor],
        sigmoid_mask_scale: float,
    ) -> torch.Tensor:
        """
        For deleted tokens, replace their corrupted final-layer representation
        with the pre-deletion hidden state saved just before the gate fired.

        deletion_weight = clamp(-gate_mask / |sigmoid_mask_scale|, 0, 1)
          = 0.0  for kept tokens  (gate≈0)     -> use final-layer representation
          = 1.0  for deleted tokens (gate≈-30)  -> use pre-deletion representation

        Has zero effect on sequence classification (CLS is never deleted, weight=0).
        """
        if pre_deletion_hidden is None or delete_gate_mask is None:
            return sequence_output
        deletion_weight = torch.clamp(
            -delete_gate_mask / abs(sigmoid_mask_scale), 0.0, 1.0
        )  # (batch, seq, 1)
        return (1.0 - deletion_weight) * sequence_output + deletion_weight * pre_deletion_hidden


# =============================================================================
# MrXLMR for Masked Language Modeling
# =============================================================================

class MrXLMRForMaskedLM(XLMRobertaPreTrainedModel):
    """MrXLMR model with a masked language modeling head."""

    config_class = MrXLMRConfig
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config: MrXLMRConfig):
        super().__init__(config)

        self.roberta = MrXLMRModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        self.post_init()
        # CRITICAL: Re-initialize delete gate AFTER post_init()
        self.roberta._init_delete_gates()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
    ) -> MrXLMRMaskedLMOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            hard_delete=hard_delete,
            deletion_threshold=deletion_threshold,
        )

        sequence_output = outputs.last_hidden_state
        if getattr(self.config, "use_pre_deletion_blend", True):
            sequence_output = self.roberta._blend_pre_deletion(
                sequence_output, outputs.pre_deletion_hidden,
                outputs.delete_gate_mask, self.config.sigmoid_mask_scale,
            )
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        return MrXLMRMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# MrXLMR for Sequence Classification
# =============================================================================

class MrXLMRForSequenceClassification(XLMRobertaPreTrainedModel):
    """MrXLMR model with a sequence classification head (linear on top of CLS token)."""

    config_class = MrXLMRConfig

    def __init__(self, config: MrXLMRConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = MrXLMRModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()
        self.roberta._init_delete_gates()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
    ) -> MrXLMRSequenceClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            hard_delete=hard_delete,
            deletion_threshold=deletion_threshold,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return MrXLMRSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# MrXLMR for Token Classification
# =============================================================================

class MrXLMRForTokenClassification(XLMRobertaPreTrainedModel):
    """MrXLMR model with a token classification head (NER, POS tagging, etc.)."""

    config_class = MrXLMRConfig

    def __init__(self, config: MrXLMRConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = MrXLMRModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()
        self.roberta._init_delete_gates()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
    ) -> MrXLMRTokenClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            hard_delete=hard_delete,
            deletion_threshold=deletion_threshold,
        )

        sequence_output = outputs.last_hidden_state
        if getattr(self.config, "use_pre_deletion_blend", True):
            sequence_output = self.roberta._blend_pre_deletion(
                sequence_output, outputs.pre_deletion_hidden,
                outputs.delete_gate_mask, self.config.sigmoid_mask_scale,
            )
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return MrXLMRTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# MrXLMR for Question Answering
# =============================================================================

class MrXLMRForQuestionAnswering(XLMRobertaPreTrainedModel):
    """MrXLMR model with an extractive question answering head."""

    config_class = MrXLMRConfig

    def __init__(self, config: MrXLMRConfig):
        super().__init__(config)
        self.num_labels = 2  # start and end logits

        self.roberta = MrXLMRModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.post_init()
        self.roberta._init_delete_gates()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
    ) -> MrXLMRQuestionAnsweringOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            hard_delete=hard_delete,
            deletion_threshold=deletion_threshold,
        )

        sequence_output = outputs.last_hidden_state
        if getattr(self.config, "use_pre_deletion_blend", True):
            sequence_output = self.roberta._blend_pre_deletion(
                sequence_output, outputs.pre_deletion_hidden,
                outputs.delete_gate_mask, self.config.sigmoid_mask_scale,
            )

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return MrXLMRQuestionAnsweringOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# MrXLMR for Multiple Choice
# =============================================================================

class MrXLMRForMultipleChoice(XLMRobertaPreTrainedModel):
    """MrXLMR model for multiple-choice tasks."""

    config_class = MrXLMRConfig

    def __init__(self, config: MrXLMRConfig):
        super().__init__(config)

        self.roberta = MrXLMRModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.post_init()
        self.roberta._init_delete_gates()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
    ) -> MrXLMRMultipleChoiceOutput:
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None else None
        )

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            hard_delete=hard_delete,
            deletion_threshold=deletion_threshold,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return MrXLMRMultipleChoiceOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )