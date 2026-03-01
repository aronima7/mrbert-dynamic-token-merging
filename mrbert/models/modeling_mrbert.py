# modeling_mrt5.py
# Author: Hiva Mohammadzadeh
# Description: This file contains the implementation of the MrBERT model.
# The code is adapted from HuggingFace's modeling_bert.py. New code sequences
# are labeled with comments.
"""
MrBERT model - BERT with MrT5-style delete gates.

This implementation adapts the delete gate mechanism from the MrT5 paper
to the BERT architecture for encoder-only tasks like Masked Language Modeling.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertPooler,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertLMPredictionHead,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
)
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import logging

from configuration_mrbert import MrBertConfig


logger = logging.get_logger(__name__)


# =============================================================================
# Custom Output Dataclasses
# =============================================================================

@dataclass
class MrBertBaseModelOutput(BaseModelOutputWithPoolingAndCrossAttentions):
    """
    Output type for MrBertModel with delete gate information.
    """
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrBertMaskedLMOutput(MaskedLMOutput):
    """
    Output type for MrBertForMaskedLM with delete gate information.
    """
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrBertSequenceClassifierOutput(SequenceClassifierOutput):
    """
    Output type for MrBertForSequenceClassification with delete gate information.
    """
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrBertTokenClassifierOutput(TokenClassifierOutput):
    """
    Output type for MrBertForTokenClassification with delete gate information.
    """
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrBertQuestionAnsweringOutput(QuestionAnsweringModelOutput):
    """
    Output type for MrBertForQuestionAnswering with delete gate information.
    """
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrBertMultipleChoiceOutput(MultipleChoiceModelOutput):
    """
    Output type for MrBertForMultipleChoice with delete gate information.
    """
    delete_gate_mask: Optional[torch.FloatTensor] = None
    delete_gate_output: Optional[torch.FloatTensor] = None
    delete_gate_logits: Optional[torch.FloatTensor] = None


@dataclass
class MrBertNextSentencePredictorOutput(NextSentencePredictorOutput):
    """
    Output type for MrBertForNextSentencePrediction with delete gate information.
    """
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
# Initialization Functions
# =============================================================================

TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
}


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
    Delete gate using scaled sigmoid activation.
    
    This is the main delete gate from the MrT5 paper. It learns to predict
    which tokens should be deleted based on hidden states.
    """
    
    def __init__(self, config: MrBertConfig):
        super().__init__()
        self.has_layer_norm = config.gate_layer_norm
        if self.has_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.feed_forward)
        self.activation = ScaledSigmoid(config.sigmoid_mask_scale)
        self.use_gumbel_noise = config.use_gumbel_noise
        self.pad_token_id = config.pad_token_id

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.has_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        delete_gate_logits = self.feed_forward(hidden_states)

        # Add Gumbel noise during training for exploration
        if self.training and self.use_gumbel_noise:
            gumbel_noise = gumbel_noise_like(delete_gate_logits)
            delete_gate_logits = delete_gate_logits + gumbel_noise

        gate_values = self.activation(delete_gate_logits)

        # CRITICAL: Never delete [CLS] token (position 0) - it's needed for classification
        # Set gate value to 0 (meaning "keep this token")
        gate_values[:, 0, :] = 0.0
        
        # Also protect [SEP] tokens (token_id = 102 for BERT)
        sep_mask = (input_ids == 102).unsqueeze(-1)
        gate_values = torch.where(
            sep_mask,
            torch.zeros_like(gate_values),
            gate_values
        )

        # Set gate values for pad tokens to sigmoid_mask_scale (fully delete padding)
        if self.pad_token_id is not None and (input_ids == self.pad_token_id).any():
            pad_mask = (input_ids == self.pad_token_id).unsqueeze(-1)
            gate_values = torch.where(
                pad_mask, 
                torch.tensor(self.activation.sigmoid_mask_scale, device=gate_values.device, dtype=gate_values.dtype), 
                gate_values
            )

        return gate_values, delete_gate_logits

    def _init_weights(self, m: nn.Module, init_func: str = "xavier_uniform_"):
        """Initialize weights. Bias is set high to strongly bias toward keeping tokens initially.
        
        With ScaledSigmoid(-30) and bias=10:
        gate_value = -30 * sigmoid(-10) ≈ -30 * 0.000045 ≈ -0.00135 (very close to 0 = keep)
        
        With deletion_threshold = -15, gate_value > -15 means KEEP the token.
        The model must learn to delete tokens, not start by deleting them.
        """
        if isinstance(m, nn.Linear):
            # Use small weights to avoid large negative outputs
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            m.bias.data.fill_(10)  # Large positive bias to ensure tokens are kept initially


class LogSigmoidDeleteGate(SigmoidDeleteGate):
    """Delete gate using log sigmoid activation."""
    
    def __init__(self, config: MrBertConfig):
        super().__init__(config)
        self.activation = nn.LogSigmoid()


class RandomDeleteGate(nn.Module):
    """
    Random delete gate for ablation studies.
    
    Randomly deletes tokens based on a configured probability.
    """
    
    def __init__(self, config: MrBertConfig):
        super().__init__()
        self.sigmoid_mask_scale = config.sigmoid_mask_scale
        self.random_deletion_probability = config.random_deletion_probability

    def _random_mask_tensor(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Create a random mask tensor with n positions marked for deletion."""
        target_shape = (x.shape[0], x.shape[1], 1)
        total_elements = x.shape[0] * x.shape[1]
        
        flat_tensor = torch.zeros(total_elements, dtype=torch.float32, device=x.device)
        indices = torch.randperm(total_elements, device=x.device)[:n]
        flat_tensor[indices] = 1.0
        
        return flat_tensor.view(target_shape)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample deletion percentage from Gaussian
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
    """
    Fixed delete gate for ablation studies.
    
    Deletes a fixed fraction of tokens in each segment (between separators).
    """
    
    def __init__(self, config: MrBertConfig):
        super().__init__()
        self.sigmoid_mask_scale = config.sigmoid_mask_scale
        self.fixed_deletion_amount = config.fixed_deletion_amount
        # Common separator token IDs (punctuation, special tokens)
        self.sep_tokens = torch.tensor([
            101, 102, 103,  # [CLS], [SEP], [MASK] for BERT
            1012, 1010, 1029, 1000, 1001,  # Common punctuation
        ])

    def _create_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        self.sep_tokens = self.sep_tokens.to(device)
        
        # Create initial mask filled with 0 (keep all tokens)
        mask = torch.zeros((batch_size, seq_len), device=device)
        
        # Find separator positions
        is_sep = torch.isin(input_ids, self.sep_tokens)
        
        # For each sequence, delete fixed_deletion_amount of non-separator tokens
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
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delete_gate_mask = self._create_mask(input_ids).unsqueeze(-1)
        return delete_gate_mask, delete_gate_mask


# =============================================================================
# Modified Attention with Delete Gate Support
# =============================================================================

class MrBertSelfAttention(nn.Module):
    """
    BERT self-attention with delete gate mask support.
    
    This extends the standard BERT self-attention to apply delete gate
    masks to attention scores, effectively "deleting" certain tokens
    by preventing other tokens from attending to them.
    """
    
    def __init__(self, config: MrBertConfig, position_embedding_type=None):
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
        
        # MrT5 additions
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
        # Project to query, key, value
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.scaling

        # Apply attention mask (additive mask where masked positions have large negative values)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply delete gate mask
        # delete_gate_mask shape: (batch, seq, 1) -> expand to (batch, 1, 1, seq)
        if delete_gate_mask is not None:
            # The mask is applied to the key dimension (last dim of attention scores)
            delete_gate_mask_expanded = delete_gate_mask.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores + delete_gate_mask_expanded

        # Normalize attention scores
        if self.use_softmax1:
            attention_probs = softmax1(attention_scores, dim=-1)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout
        attention_probs = self.dropout(attention_probs)

        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Compute context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class MrBertAttention(nn.Module):
    """BERT attention block with delete gate support."""
    
    def __init__(self, config: MrBertConfig):
        super().__init__()
        self.self = MrBertSelfAttention(config)
        self.output = BertSelfOutput(config)

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

class MrBertLayer(nn.Module):
    """
    BERT layer with optional delete gate.
    
    The delete gate is applied after the layer's computations to determine
    which tokens should be deleted for subsequent layers.
    """
    
    def __init__(self, config: MrBertConfig, has_delete_gate: bool = False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MrBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        
        # Delete gate
        self.has_delete_gate = has_delete_gate
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
        
        # Determine which tokens to keep
        threshold = deletion_threshold if deletion_threshold is not None else self.deletion_threshold
        keep_this = delete_gate_mask > threshold
        
        # Calculate target positions for remaining tokens
        target_pos = torch.cumsum(keep_this.int(), dim=1) - 1
        new_len = max(target_pos[:, -1].max().item() + 1, 1)
        target_pos = target_pos.clamp(min=0)
        
        # Map positions
        positions = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
        positions = positions * keep_this.int()
        
        src_side_pos = torch.zeros(batch_size, new_len, device=device, dtype=torch.int32)
        src_side_pos.scatter_add_(1, target_pos.long(), positions)
        
        # Create new mask
        new_mask = torch.arange(new_len, device=device).unsqueeze(0).expand(batch_size, -1) <= target_pos[:, -1:]
        new_mask = (~new_mask).float() * -1e9
        new_mask = new_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, new_len)
        
        return src_side_pos.long(), new_mask

    def _hard_delete_hidden_states(
        self, 
        hidden_states: torch.Tensor, 
        positions: torch.Tensor
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
        # Initialize delete gate outputs
        delete_gate_values = None
        delete_gate_logits = None
        new_attention_mask = attention_mask
        
        # Apply delete gate if this layer has one
        if self.has_delete_gate and input_ids is not None:
            delete_gate_values, delete_gate_logits = self.delete_gate(hidden_states, input_ids)
            delete_gate_mask = delete_gate_values
            
            # Check if all tokens would be deleted (gate_values < threshold means delete)
            threshold = deletion_threshold if deletion_threshold is not None else self.deletion_threshold
            if threshold is not None and (delete_gate_values < threshold).all():
                raise ValueError(
                    "All tokens would be deleted. Adjust deletion_threshold or sigmoid_mask_scale."
                )
            
            # Apply hard deletion if requested
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
        
        # Self-attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            delete_gate_mask=delete_gate_mask if not hard_delete else None,
        )
        attention_output = attention_outputs[0]
        
        # Feed-forward
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, 
            self.chunk_size_feed_forward, 
            self.seq_len_dim, 
            attention_output
        )
        
        outputs = (layer_output,) + attention_outputs[1:]

        # Add delete gate outputs if present.
        # When has_delete_gate=True the tuple always ends with these 4 items (in order):
        #   [-4] delete_gate_values   — gate scores (batch, seq, 1)
        #   [-3] delete_gate_logits   — pre-activation logits (batch, seq, 1)
        #   [-2] delete_gate_mask     — same as delete_gate_values, used as attention bias
        #   [-1] new_attention_mask   — updated mask after hard deletion (or original if soft)
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

class MrBertEncoder(nn.Module):
    """
    BERT encoder with delete gate support.
    
    One layer (specified by config.delete_gate_layer) contains a delete gate
    that learns which tokens to remove from the sequence.
    """
    
    def __init__(self, config: MrBertConfig):
        super().__init__()
        self.config = config
        
        # Create layers with delete gate at specified layer
        self.layer = nn.ModuleList([
            MrBertLayer(config, has_delete_gate=(i == config.delete_gate_layer))
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
        
        # Delete gate tracking
        delete_gate_mask = None
        delete_gate_output = None
        delete_gate_logits = None
        final_attention_mask = attention_mask
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
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

            # Update delete gate info if this layer has a gate.
            # Gate outputs are always the last 4 items — see MrBertLayer.forward() for layout.
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
        )


# =============================================================================
# MrBERT Model
# =============================================================================

class MrBertModel(BertPreTrainedModel):
    """
    MrBERT model with delete gate mechanism.
    
    This is a BERT model that incorporates a learnable delete gate to
    selectively remove tokens during encoding, reducing computational cost.
    """
    
    config_class = MrBertConfig
    
    def __init__(self, config: MrBertConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = MrBertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        
        # Initialize weights (this will call _init_weights on all modules)
        self.post_init()
        
        # CRITICAL: Re-initialize delete gate AFTER post_init() to ensure correct initialization
        self._init_delete_gates()

    def _init_delete_gates(self):
        """Initialize delete gate weights to strongly favor keeping tokens.
        
        This must be called AFTER post_init() because post_init() overwrites
        all module weights with the standard BERT initialization.
        """
        found_gate = False
        for i, layer in enumerate(self.encoder.layer):
            if layer.has_delete_gate:
                gate = layer.delete_gate
                if hasattr(gate, 'feed_forward'):
                    # Initialize with small weights and large positive bias
                    # This ensures initial gate values are close to 0 (keep tokens)
                    nn.init.normal_(gate.feed_forward.weight, mean=0.0, std=0.001)
                    gate.feed_forward.bias.data.fill_(10.0)
                    found_gate = True
                    print(f"[MrBERT] Initialized delete gate at layer {i} with bias=10.0, weight_std=0.001")
        
        if not found_gate:
            print(f"[MrBERT] WARNING: No delete gate found! delete_gate_layer={self.config.delete_gate_layer}")

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
        
        # Convert from 0/1 mask to additive mask
        # 1 -> 0.0 (attend), 0 -> -10000.0 (don't attend)
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
    ) -> MrBertBaseModelOutput:
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
        
        # Default attention mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        # Convert attention mask
        extended_attention_mask = self._convert_attention_mask(
            attention_mask, input_shape, self.dtype
        )
        
        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # Embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        # Encoder
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
        
        return MrBertBaseModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs[1],
            attentions=encoder_outputs[2],
            delete_gate_mask=encoder_outputs[3],
            delete_gate_output=encoder_outputs[4],
            delete_gate_logits=encoder_outputs[5],
        )


# =============================================================================
# MrBERT for Masked Language Modeling
# =============================================================================

class MrBertForMaskedLM(BertPreTrainedModel):
    """
    MrBERT model with a masked language modeling head.
    
    This model includes the delete gate mechanism and can be used for
    pre-training or fine-tuning on MLM tasks.
    """
    
    config_class = MrBertConfig
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    
    def __init__(self, config: MrBertConfig):
        super().__init__(config)
        
        self.bert = MrBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        
        # Initialize weights
        self.post_init()
        
        # CRITICAL: Re-initialize delete gate AFTER post_init()
        self.bert._init_delete_gates()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

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
    ) -> MrBertMaskedLMOutput:
        outputs = self.bert(
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
        prediction_scores = self.cls(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), 
                labels.view(-1)
            )
        
        return MrBertMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# MrBERT for Sequence Classification
# =============================================================================

class MrBertForSequenceClassification(BertPreTrainedModel):
    """
    MrBERT model with a sequence classification head.
    
    This can be used for tasks like sentiment analysis, text classification,
    or any task that requires classifying an entire sequence.
    """
    
    config_class = MrBertConfig
    
    def __init__(self, config: MrBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.bert = MrBertModel(config)
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()
        
        # CRITICAL: Re-initialize delete gate AFTER post_init()
        self.bert._init_delete_gates()

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
    ) -> MrBertSequenceClassifierOutput:
        outputs = self.bert(
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
        
        return MrBertSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# MrBERT for Token Classification
# =============================================================================

class MrBertForTokenClassification(BertPreTrainedModel):
    """
    MrBERT model with a token classification head.
    
    This can be used for tasks like Named Entity Recognition (NER),
    Part-of-Speech tagging, or any token-level classification task.
    
    Note: When using hard deletion, the output sequence length may not match
    the input sequence length, which can complicate token-level predictions.
    Consider using soft deletion for token classification tasks.
    """
    
    config_class = MrBertConfig
    
    def __init__(self, config: MrBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = MrBertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()
        
        # CRITICAL: Re-initialize delete gate AFTER post_init()
        self.bert._init_delete_gates()

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
    ) -> MrBertTokenClassifierOutput:
        outputs = self.bert(
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
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return MrBertTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# MrBERT for Question Answering
# =============================================================================

class MrBertForQuestionAnswering(BertPreTrainedModel):
    """
    MrBERT model with a question answering head (extractive QA).
    
    This can be used for tasks like SQuAD where the model needs to find
    the start and end positions of the answer span in the context.
    
    Note: When using hard deletion, answer positions may shift. Consider
    using soft deletion or tracking position mappings for QA tasks.
    """
    
    config_class = MrBertConfig
    
    def __init__(self, config: MrBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = MrBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()
        
        # CRITICAL: Re-initialize delete gate AFTER post_init()
        self.bert._init_delete_gates()

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
    ) -> MrBertQuestionAnsweringOutput:
        outputs = self.bert(
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
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # Clamp positions to valid range
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        return MrBertQuestionAnsweringOutput(
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
# MrBERT for Multiple Choice
# =============================================================================

class MrBertForMultipleChoice(BertPreTrainedModel):
    """
    MrBERT model with a multiple choice classification head.
    
    This can be used for tasks like SWAG or other multiple choice QA tasks.
    """
    
    config_class = MrBertConfig
    
    def __init__(self, config: MrBertConfig):
        super().__init__(config)
        
        self.bert = MrBertModel(config)
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self.post_init()
        
        # CRITICAL: Re-initialize delete gate AFTER post_init()
        self.bert._init_delete_gates()

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
    ) -> MrBertMultipleChoiceOutput:
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        
        # Flatten for processing
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        outputs = self.bert(
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
        
        return MrBertMultipleChoiceOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# MrBERT for Next Sentence Prediction
# =============================================================================

class MrBertForNextSentencePrediction(BertPreTrainedModel):
    """
    MrBERT model with a next sentence prediction head.
    
    This can be used for sentence pair classification tasks or
    as part of BERT pre-training.
    """
    
    config_class = MrBertConfig
    
    def __init__(self, config: MrBertConfig):
        super().__init__(config)
        
        self.bert = MrBertModel(config)
        self.cls = BertOnlyNSPHead(config)
        
        # Initialize weights
        self.post_init()
        
        # CRITICAL: Re-initialize delete gate AFTER post_init()
        self.bert._init_delete_gates()

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
    ) -> MrBertNextSentencePredictorOutput:
        outputs = self.bert(
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
        seq_relationship_scores = self.cls(pooled_output)
        
        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))
        
        return MrBertNextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            delete_gate_mask=outputs.delete_gate_mask,
            delete_gate_output=outputs.delete_gate_output,
            delete_gate_logits=outputs.delete_gate_logits,
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Config
    "MrBertConfig",
    # Base Model
    "MrBertModel",
    # Task-specific Models
    "MrBertForMaskedLM",
    "MrBertForSequenceClassification",
    "MrBertForTokenClassification",
    "MrBertForQuestionAnswering",
    "MrBertForMultipleChoice",
    "MrBertForNextSentencePrediction",
    # Output Classes
    "MrBertBaseModelOutput",
    "MrBertMaskedLMOutput",
    "MrBertSequenceClassifierOutput",
    "MrBertTokenClassifierOutput",
    "MrBertQuestionAnsweringOutput",
    "MrBertMultipleChoiceOutput",
    "MrBertNextSentencePredictorOutput",
    # Delete Gate Modules
    "SigmoidDeleteGate",
    "LogSigmoidDeleteGate",
    "RandomDeleteGate",
    "FixedDeleteGate",
]
