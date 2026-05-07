"""
MrDiffusion: SEDD with MrT5-style delete gates.

Adapts the delete gate mechanism from MrBERT/MrT5 to Score Entropy Discrete
Diffusion (SEDD). After a specified DDiTBlock layer a learned gate assigns each
token a scalar score; low-scoring tokens are either soft-deleted (large negative
attention bias applied in subsequent blocks) or hard-deleted (physically removed
from the sequence).

Reference:
  - SEDD: Lou et al., 2023 (https://arxiv.org/abs/2310.16834)
  - MrT5: Kallini et al., 2024 (https://arxiv.org/abs/2410.20771)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# ---------------------------------------------------------------------------
# Ensure SEDD root is on sys.path so that `model`, `graph_lib`, etc. resolve.
# ---------------------------------------------------------------------------
import sys as _sys
import os as _os

_MRDIFF_DIR = _os.path.dirname(_os.path.abspath(__file__))
_SEDD_ROOT = _os.path.join(_os.path.dirname(_MRDIFF_DIR), "diffusion", "Score-Entropy-Discrete-Diffusion")
if _SEDD_ROOT not in _sys.path:
    _sys.path.insert(0, _SEDD_ROOT)

from model.transformer import (
    LayerNorm,
    EmbeddingLayer,
    TimestepEmbedder,
    LabelEmbedder,
    DDitFinalLayer,
    modulate,
)
from model import rotary
from model.fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    get_bias_dropout_add_scale,
    modulate_fused,
)
from omegaconf import OmegaConf

from configuration_mrdiffusion import MrDiffusionConfig


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MrSEDDOutput:
    """Output of MrSEDD.forward()."""
    logits: torch.Tensor                               # [B, L, vocab_size] — may be shorter for hard deletion
    delete_gate_mask: Optional[torch.Tensor] = None   # [B, L, 1] gate bias values
    delete_gate_logits: Optional[torch.Tensor] = None # [B, L, 1] raw gate logits (pre-activation)
    delete_gate_output: Optional[torch.Tensor] = None # [B, L, 1] gate values (sigmoid output)
    # For hard deletion: mapping from compressed → original positions
    keep_indices: Optional[torch.Tensor] = None       # [B, L_kept] original token positions kept

    def exp(self) -> torch.Tensor:
        """Delegate to logits.exp() so SEDD's score_fn(sampling=True) works unchanged."""
        return self.logits.exp()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    """Gumbel noise with same shape/device/dtype as x."""
    eps = 3e-4 if x.dtype == torch.float16 else 1e-10
    uniform = torch.empty_like(x).uniform_(eps, 1 - eps)
    return -((-uniform.log()).log())


class ScaledSigmoid(nn.Module):
    """sigmoid_mask_scale * sigmoid(-x)  →  values in [sigmoid_mask_scale, 0]."""
    def __init__(self, sigmoid_mask_scale: float):
        super().__init__()
        self.sigmoid_mask_scale = sigmoid_mask_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_mask_scale * torch.sigmoid(-x)


# ---------------------------------------------------------------------------
# Delete Gate variants
# ---------------------------------------------------------------------------

class SigmoidDeleteGate(nn.Module):
    """
    Token-only delete gate (same design as MrBERT).

    Input:  hidden_states [B, L, hidden_size]
    Output: gate_values [B, L, 1], gate_logits [B, L, 1]
    """

    def __init__(self, hidden_size: int, mr_config: MrDiffusionConfig):
        super().__init__()
        self.has_layer_norm = mr_config.gate_layer_norm
        if self.has_layer_norm:
            self.layer_norm = LayerNorm(hidden_size)
        self.feed_forward = nn.Linear(hidden_size, 1)
        # Initialise: positive bias → gate starts near 0 (keep all tokens).
        # Value 2.0 keeps gate_output ≈ -3.6 (above deletion threshold -15) while
        # providing ~1000× better gradient flow than bias=10.
        nn.init.normal_(self.feed_forward.weight, mean=0.0, std=0.01)
        self.feed_forward.bias.data.fill_(2.0)
        self.activation = ScaledSigmoid(mr_config.sigmoid_mask_scale)
        self.use_gumbel_noise = mr_config.use_gumbel_noise

    def forward(
        self,
        hidden_states: torch.Tensor,
        c: Optional[torch.Tensor] = None,  # unused, accepted for API symmetry
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layer_norm(hidden_states) if self.has_layer_norm else hidden_states
        logits = self.feed_forward(h)  # [B, L, 1]
        if self.training and self.use_gumbel_noise:
            logits = logits + gumbel_noise_like(logits)
        gate_values = self.activation(logits)
        return gate_values, logits


class SigmoidDeleteGateWithSigma(nn.Module):
    """
    Delete gate conditioned on token hidden state AND sigma (timestep) embedding.

    Allows the gate to learn different deletion patterns at different noise levels.
    Input:  hidden_states [B, L, hidden_size], c [B, cond_dim]
    Output: gate_values [B, L, 1], gate_logits [B, L, 1]
    """

    def __init__(self, hidden_size: int, cond_dim: int, mr_config: MrDiffusionConfig):
        super().__init__()
        self.has_layer_norm = mr_config.gate_layer_norm
        if self.has_layer_norm:
            self.layer_norm = LayerNorm(hidden_size)
        self.feed_forward = nn.Linear(hidden_size + cond_dim, 1)
        nn.init.normal_(self.feed_forward.weight, mean=0.0, std=0.01)
        self.feed_forward.bias.data.fill_(2.0)
        self.activation = ScaledSigmoid(mr_config.sigmoid_mask_scale)
        self.use_gumbel_noise = mr_config.use_gumbel_noise

    def forward(
        self,
        hidden_states: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layer_norm(hidden_states) if self.has_layer_norm else hidden_states
        # Expand condition to sequence length
        c_exp = c.unsqueeze(1).expand(-1, h.shape[1], -1)  # [B, L, cond_dim]
        gate_input = torch.cat([h, c_exp], dim=-1)          # [B, L, hidden+cond]
        logits = self.feed_forward(gate_input)
        if self.training and self.use_gumbel_noise:
            logits = logits + gumbel_noise_like(logits)
        gate_values = self.activation(logits)
        return gate_values, logits


class LogSigmoidDeleteGate(SigmoidDeleteGate):
    """Token-only delete gate using log sigmoid activation."""
    def __init__(self, hidden_size: int, mr_config: MrDiffusionConfig):
        super().__init__(hidden_size, mr_config)
        self.activation = nn.LogSigmoid()


class LogSigmoidDeleteGateWithSigma(SigmoidDeleteGateWithSigma):
    """Sigma-conditioned delete gate using log sigmoid activation."""
    def __init__(self, hidden_size: int, cond_dim: int, mr_config: MrDiffusionConfig):
        super().__init__(hidden_size, cond_dim, mr_config)
        self.activation = nn.LogSigmoid()


class RandomDeleteGate(nn.Module):
    """Random deletion baseline (no learned parameters)."""

    def __init__(self, mr_config: MrDiffusionConfig):
        super().__init__()
        self.sigmoid_mask_scale = mr_config.sigmoid_mask_scale
        self.prob = mr_config.random_deletion_probability

    def forward(
        self,
        hidden_states: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prob = float(np.clip(np.random.normal(self.prob, 0.05), 0.0, 1.0))
        # Bernoulli mask: 1 = delete
        mask = (torch.rand(
            hidden_states.shape[0], hidden_states.shape[1], 1,
            device=hidden_states.device
        ) < prob).float()
        gate_values = mask * self.sigmoid_mask_scale
        return gate_values, gate_values


class FixedDeleteGate(nn.Module):
    """Fixed-fraction deletion baseline (no learned parameters)."""

    def __init__(self, mr_config: MrDiffusionConfig):
        super().__init__()
        self.sigmoid_mask_scale = mr_config.sigmoid_mask_scale
        self.frac = mr_config.fixed_deletion_amount

    def forward(
        self,
        hidden_states: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = hidden_states.shape
        n_del = int(self.frac * L)
        gate_values = torch.zeros(B, L, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        if n_del > 0:
            for b in range(B):
                idx = torch.randperm(L, device=hidden_states.device)[:n_del]
                gate_values[b, idx, 0] = self.sigmoid_mask_scale
        return gate_values, gate_values


class BottleneckDeleteGate(nn.Module):
    """
    Plan-specified 2-layer bottleneck MLP gate (plan Section 2.2).

    Architecture:
        noise_proj: Linear(cond_dim → hidden_size)
        gate_mlp:   Linear(2·hidden_size → bottleneck_dim) → GELU → Linear(bottleneck_dim → 1)

    Uses Gumbel-sigmoid with temperature annealing during training.
    Hard decisions via straight-through estimator (STE).

    Returns (gate_values in [0,1], raw logits) where 1=keep, 0=delete.
    For soft deletion: convert to bias = sigmoid_mask_scale * (1 - gate_values).
    """

    def __init__(self, hidden_size: int, cond_dim: int,
                 bottleneck_dim: int = 128,
                 sigmoid_mask_scale: float = -30.0,
                 use_layer_norm: bool = True):
        super().__init__()
        self.sigmoid_mask_scale = sigmoid_mask_scale
        self.layer_norm = LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.noise_proj = nn.Linear(cond_dim, hidden_size)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
        )
        self.gate_mlp[-1].bias.data.fill_(3.0)
        nn.init.normal_(self.gate_mlp[0].weight, std=0.01)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)
        self._temperature = 1.0  # set externally by training loop

    def forward(
        self,
        hidden_states: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layer_norm(hidden_states)
        if c is not None:
            noise_expanded = self.noise_proj(c).unsqueeze(1).expand_as(h)
            gate_input = torch.cat([h, noise_expanded], dim=-1)
        else:
            gate_input = torch.cat([h, torch.zeros_like(h)], dim=-1)

        logits = self.gate_mlp(gate_input)

        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            gate_values = torch.sigmoid((logits + gumbel) / self._temperature)
        else:
            gate_values = torch.sigmoid(logits)

        # Convert [0,1] gate to additive bias in [sigmoid_mask_scale, 0]
        gate_bias = self.sigmoid_mask_scale * (1.0 - gate_values)
        return gate_bias, logits


class PreDeletionBlend(nn.Module):
    """
    Blend each deleted token into its next surviving neighbor before removal.

    h'_i = g_i * h_i + (1 - g_i) * h_{next}

    where g_i ∈ [0,1] (1=keep, 0=delete).
    """
    def forward(self, hidden_states: torch.Tensor,
                gate_values: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, L, D]
        gate_values:   [B, L, 1] in [0,1] where 1=keep
        """
        h_next = torch.roll(hidden_states, -1, dims=1)
        h_next[:, -1, :] = hidden_states[:, -1, :]
        return gate_values * hidden_states + (1 - gate_values) * h_next


class TokenMerge(nn.Module):
    """Remove deleted tokens from sequence (hard deletion)."""

    def forward(self, hidden_states: torch.Tensor,
                gate_hard: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        hidden_states: [B, L, D]
        gate_hard:     [B, L, 1] binary (1=keep)

        Returns (merged [B, L', D], merge_info dict).
        """
        B, L, D = hidden_states.shape
        keep_mask = gate_hard.squeeze(-1).bool()
        keep_counts = keep_mask.sum(dim=1)
        L_prime = max(int(keep_counts.max().item()), 1)

        merged = torch.zeros(B, L_prime, D,
                             device=hidden_states.device, dtype=hidden_states.dtype)
        for b in range(B):
            kept = keep_mask[b].nonzero(as_tuple=True)[0]
            n = kept.shape[0]
            if n > 0:
                merged[b, :n] = hidden_states[b, kept]

        merge_info = {"keep_mask": keep_mask, "original_length": L}
        return merged, merge_info


class TokenRestore(nn.Module):
    """Scatter merged tokens back to full-length, filling deleted slots from pre-gate states."""

    def forward(self, merged: torch.Tensor, merge_info: dict,
                pre_gate_states: torch.Tensor) -> torch.Tensor:
        """
        merged:          [B, L', D]
        merge_info:      from TokenMerge
        pre_gate_states: [B, L, D]
        """
        keep_mask = merge_info["keep_mask"]
        restored = pre_gate_states.clone()
        for b in range(merged.shape[0]):
            kept = keep_mask[b].nonzero(as_tuple=True)[0]
            n = kept.shape[0]
            if n > 0:
                restored[b, kept] = merged[b, :n]
        return restored


def build_delete_gate(
    mr_config: MrDiffusionConfig,
    hidden_size: int,
    cond_dim: int,
) -> nn.Module:
    """Factory: instantiate the correct delete gate based on config."""
    gate_type = getattr(mr_config, "gate_type", mr_config.deletion_type)

    if gate_type == "bottleneck_mlp":
        return BottleneckDeleteGate(
            hidden_size, cond_dim,
            getattr(mr_config, "gate_bottleneck_dim", 128),
            mr_config.sigmoid_mask_scale,
            mr_config.gate_layer_norm,
        )

    dtype = mr_config.deletion_type
    sigma_cond = mr_config.gate_sigma_conditioned

    if dtype == "scaled_sigmoid":
        if sigma_cond:
            return SigmoidDeleteGateWithSigma(hidden_size, cond_dim, mr_config)
        return SigmoidDeleteGate(hidden_size, mr_config)
    elif dtype == "log_sigmoid":
        if sigma_cond:
            return LogSigmoidDeleteGateWithSigma(hidden_size, cond_dim, mr_config)
        return LogSigmoidDeleteGate(hidden_size, mr_config)
    elif dtype == "random":
        return RandomDeleteGate(mr_config)
    elif dtype == "fixed":
        return FixedDeleteGate(mr_config)
    else:
        raise ValueError(f"Unknown deletion_type: {dtype!r}")


# ---------------------------------------------------------------------------
# Modified DDiTBlock with soft-deletion support
# ---------------------------------------------------------------------------

class MrDDiTBlock(nn.Module):
    """
    DDiTBlock extended to accept an optional delete_gate_mask for soft deletion.

    When delete_gate_mask is provided (soft deletion mode), the block falls back
    from FlashAttention to F.scaled_dot_product_attention so that an additive
    per-key attention bias can be applied.  All other computations are identical
    to the original DDiTBlock.
    """

    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def _attn_with_mask(
        self,
        x: torch.Tensor,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        delete_gate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard (non-Flash) attention path used when delete_gate_mask is present.
        delete_gate_mask: [B, L, 1]  (gate bias for each key position)
        """
        B, L, D = x.shape
        qkv = self.attn_qkv(x)  # [B, L, 3*D]
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        # qkv: [B, L, 3, n_heads, head_dim] → q/k/v: [B, n_heads, L, head_dim]
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Build additive attention bias from gate mask
        # delete_gate_mask [B, L, 1] → [B, 1, 1, L] (broadcast over heads and queries)
        attn_bias = delete_gate_mask.squeeze(-1).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        attn_bias = attn_bias.to(q.dtype)

        with torch.cuda.amp.autocast(enabled=False):
            x_attn = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                attn_mask=attn_bias.float(),
                dropout_p=self.dropout if self.training else 0.0,
            ).to(q.dtype)

        x_attn = x_attn.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return x_attn

    def _attn_with_positional_rotary(
        self,
        x: torch.Tensor,
        positional_rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """SDPA attention path with per-batch per-position RoPE.

        Used during the compressed phase of hard deletion when
        rope_original_positions=True.  FlashAttention's rotary apply only supports
        a single shared position sequence across the batch; after hard deletion each
        batch item keeps different original positions, so we fall back to
        _apply_rotary_pos_emb_torchscript (which handles per-batch cos/sin) + SDPA.

        positional_rotary_cos_sin: (cos, sin) each [B, L_kept, 3, 1, D_rot]
        """
        B, L, D = x.shape
        qkv = self.attn_qkv(x)  # [B, L, 3*D]
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = positional_rotary_cos_sin
            qkv = rotary._apply_rotary_pos_emb_torchscript(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        with torch.cuda.amp.autocast(enabled=False):
            x_attn = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                dropout_p=self.dropout if self.training else 0.0,
            ).to(q.dtype)
        x_attn = x_attn.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return x_attn

    def _attn_flash(
        self,
        x: torch.Tensor,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        seqlens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Original FlashAttention path."""
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
        B, L, D = x.shape
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (B + 1) * L, step=L, dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x_attn = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, L, 0.0, causal=False)
        x_attn = rearrange(x_attn, "(b s) h d -> b s (h d)", b=B)
        return x_attn

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        c: torch.Tensor,
        seqlens: Optional[torch.Tensor] = None,
        delete_gate_mask: Optional[torch.Tensor] = None,
        positional_rotary_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        x_norm = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        if positional_rotary_cos_sin is not None:
            x_attn = self._attn_with_positional_rotary(x_norm, positional_rotary_cos_sin)
        elif delete_gate_mask is not None:
            x_attn = self._attn_with_mask(x_norm, rotary_cos_sin, delete_gate_mask)
        else:
            x_attn = self._attn_flash(x_norm, rotary_cos_sin, seqlens)

        x = bias_dropout_scale_fn(self.attn_out(x_attn), None, gate_msa, x_skip, self.dropout)
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout
        )
        return x


# ---------------------------------------------------------------------------
# MrSEDD — main model
# ---------------------------------------------------------------------------

class MrSEDD(nn.Module):
    """
    SEDD with an MrT5-style delete gate injected after `mr_config.delete_gate_layer`.

    The delete gate assigns each token a scalar score in [sigmoid_mask_scale, 0]:
      - Soft deletion (default): score is added as a key-dimension attention bias
        in all subsequent DDiTBlocks, effectively hiding the token from attention.
      - Hard deletion: tokens below deletion_threshold are physically removed;
        RoPE is recomputed for the surviving positions.

    Configuration is a union of:
      - `config` (OmegaConf): original SEDD model/training/graph/noise config.
      - `mr_config` (MrDiffusionConfig): delete gate parameters.
    """

    def __init__(self, config, mr_config: MrDiffusionConfig):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        self.mr_config = mr_config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)

        hidden_size = config.model.hidden_size
        cond_dim = config.model.cond_dim
        n_blocks = config.model.n_blocks

        # Standard SEDD components
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = rotary.Rotary(hidden_size // config.model.n_heads)

        # Transformer blocks — all use MrDDiTBlock (superset of DDiTBlock)
        self.blocks = nn.ModuleList([
            MrDDiTBlock(
                hidden_size,
                config.model.n_heads,
                cond_dim,
                dropout=config.model.dropout,
            )
            for _ in range(n_blocks)
        ])

        self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma

        # Delete gate + blend/merge/restore
        self.delete_gate = build_delete_gate(mr_config, hidden_size, cond_dim)
        self.pre_blend = PreDeletionBlend()
        self.token_merge = TokenMerge()
        self.token_restore = TokenRestore()

    # ------------------------------------------------------------------
    # Hard deletion helpers
    # ------------------------------------------------------------------

    def _apply_hard_deletion(
        self,
        x: torch.Tensor,
        gate_values: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Physically remove tokens where gate_value <= deletion_threshold.

        Returns:
            x_kept:         [B, L_kept, hidden_size]
            keep_mask:      [B, L] bool — which original positions survived
            idx_kept:       [B, L_kept] token ids for surviving positions
            kept_positions: [B, L_kept] original integer positions of survivors
        """
        threshold = self.mr_config.deletion_threshold
        keep_mask = gate_values.squeeze(-1) > threshold  # [B, L] bool

        # Pad to equal length across batch (minimum kept length)
        min_kept = max(int(keep_mask.sum(dim=1).min().item()), 1)

        B, L, D = x.shape
        x_kept = torch.zeros(B, min_kept, D, device=x.device, dtype=x.dtype)
        idx_kept = torch.zeros(B, min_kept, dtype=indices.dtype, device=x.device)
        kept_positions = torch.zeros(B, min_kept, dtype=torch.long, device=x.device)

        for b in range(B):
            pos = keep_mask[b].nonzero(as_tuple=False).squeeze(1)
            n = min(len(pos), min_kept)
            x_kept[b, :n] = x[b, pos[:n]]
            idx_kept[b, :n] = indices[b, pos[:n]]
            kept_positions[b, :n] = pos[:n]

        return x_kept, keep_mask, idx_kept, kept_positions

    def _restore_hidden_states(
        self,
        x_compressed: torch.Tensor,
        x_pre_gate: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scatter compressed hidden states back to original sequence length.

        Kept positions   → updated hidden state from compressed-phase processing.
        Deleted positions → their hidden state from just before the gate fired
                            (x_pre_gate), so the output layer can produce a
                            meaningful score for those positions rather than zeros.

        Args:
            x_compressed: [B, L_kept, D]  hidden states after compressed blocks
            x_pre_gate:   [B, L, D]       hidden states recorded before gate fired
            keep_mask:    [B, L] bool     which original positions were kept

        Returns: [B, L, D]
        """
        # Start from pre-gate states (fills deleted positions automatically)
        x_restored = x_pre_gate.clone()
        for b in range(x_compressed.shape[0]):
            pos = keep_mask[b].nonzero(as_tuple=False).squeeze(1)
            n = min(len(pos), x_compressed.shape[1])
            x_restored[b, pos[:n]] = x_compressed[b, :n]
        return x_restored

    def _recompute_rotary(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recompute RoPE embeddings for the current sequence length."""
        return self.rotary_emb(x)

    def _compute_rotary_for_positions(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-batch RoPE cos/sin for arbitrary (non-contiguous) positions.

        Used after hard deletion when rope_original_positions=True: survivors at
        original positions [0,2,5,...] get their original encodings rather than
        being re-indexed to [0,1,2,...].

        FlashAttention's apply_rotary_emb_qkv_ only accepts a single shared
        position sequence for the whole batch, so the output here must be used
        with _apply_rotary_pos_emb_torchscript + SDPA (see _attn_with_positional_rotary).

        positions: [B, L_kept]  original integer positions of surviving tokens.
        Returns:   (cos, sin) each [B, L_kept, 3, 1, D_rot]
        """
        inv_freq = self.rotary_emb.inv_freq                          # [D_rot/2]
        freqs = positions.float().unsqueeze(-1) * inv_freq           # [B, L_kept, D_rot/2]
        emb = torch.cat([freqs, freqs], dim=-1)                      # [B, L_kept, D_rot]
        cos = emb.cos().unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 1, 1)
        sin = emb.sin().unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 1, 1)
        cos[:, :, 2, :, :] = 1.0  # v-component identity
        sin[:, :, 2, :, :] = 0.0
        return cos, sin

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        indices: torch.Tensor,
        sigma: torch.Tensor,
        gate_temperature: float = 1.0,
    ) -> MrSEDDOutput:
        """
        Args:
            indices: [B, L] corrupted token ids
            sigma:   [B]    noise level per example

        Forward phases
        --------------
        Phase 1 — blocks 0 .. delete_gate_layer
            Full-length FlashAttention on all L tokens.

        Gate fires after delete_gate_layer:
            Soft: compute gate bias mask [B, L, 1]; tokens stay in sequence.
            Hard: physically remove tokens below threshold; store x_pre_gate for
                  restoration.

        Phase 2 — blocks delete_gate_layer+1 .. restore_gate_layer (or last block)
            Soft: MrDDiTBlock with additive attention bias (SDPA fallback).
            Hard: FlashAttention on shortened L′ sequence.

        Restoration at restore_gate_layer (or just before output_layer if None):
            Soft: clear the gate mask; subsequent blocks return to FlashAttention.
            Hard: _restore_hidden_states fills deleted positions with x_pre_gate
                  so the output layer produces valid scores for all L positions.

        Phase 3 — blocks restore_gate_layer+1 .. N  (only if restore_gate_layer set)
            Full-length FlashAttention on restored L tokens.

        Output layer sees full L in all cases — no zero-logit scatter needed.
        """
        original_indices = indices
        original_len = indices.shape[1]

        x = self.vocab_embed(indices)       # [B, L, D]
        c = F.silu(self.sigma_map(sigma))   # [B, cond_dim]
        rotary_cos_sin = self.rotary_emb(x)

        delete_gate_mask = None
        gate_output = None
        gate_logits = None
        keep_mask = None
        x_pre_gate = None
        positional_rotary_cos_sin = None
        restored = False

        restore_at = self.mr_config.restore_gate_layer  # int or None

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i, block in enumerate(self.blocks):

                # ---- Restore at the configured layer ----
                if restore_at is not None and i == restore_at and not restored:
                    if self.mr_config.deletion_mode == "hard" and keep_mask is not None:
                        x = self._restore_hidden_states(x, x_pre_gate, keep_mask)
                        rotary_cos_sin = self._recompute_rotary(x)
                        positional_rotary_cos_sin = None  # phase 3: back to full contiguous sequence
                        indices = original_indices
                        keep_mask = None
                    # Soft: just stop applying the mask; tokens were never removed
                    delete_gate_mask = None
                    restored = True

                # ---- Forward through block ----
                if positional_rotary_cos_sin is not None:
                    x = block(x, rotary_cos_sin, c, seqlens=None,
                              positional_rotary_cos_sin=positional_rotary_cos_sin)
                elif delete_gate_mask is not None:
                    x = block(x, rotary_cos_sin, c, seqlens=None,
                              delete_gate_mask=delete_gate_mask)
                else:
                    x = block(x, rotary_cos_sin, c, seqlens=None)

                # ---- Gate fires after delete_gate_layer ----
                if i == self.mr_config.delete_gate_layer:
                    x_pre_gate = x  # keep gradient; used for restoration
                    # Set temperature on BottleneckDeleteGate if applicable
                    if hasattr(self.delete_gate, "_temperature"):
                        self.delete_gate._temperature = gate_temperature
                    gate_output, gate_logits = self.delete_gate(x, c)

                    if self.mr_config.deletion_mode == "soft":
                        # Optionally stop score-entropy gradients from reaching the gate.
                        # When True, the gate is trained only by the gate loss, not by
                        # the diffusion objective pushing it toward "keep everything".
                        mask = gate_output.detach() if getattr(self.mr_config, "stop_gate_grad", False) else gate_output
                        delete_gate_mask = mask
                    else:  # hard
                        # For BottleneckDeleteGate: gate_output is bias (scale*(1-prob))
                        # Convert to keep_mask: keep if gate > threshold
                        gate_hard = (gate_output > self.mr_config.deletion_threshold).float()
                        # PreDeletionBlend before removal
                        if isinstance(self.delete_gate, BottleneckDeleteGate):
                            gate_prob = torch.sigmoid(gate_logits)  # [B, L, 1] in [0,1]
                            x = self.pre_blend(x, gate_prob)
                        x, keep_mask, indices, kept_positions = self._apply_hard_deletion(
                            x, gate_output, indices
                        )
                        if self.mr_config.rope_original_positions:
                            positional_rotary_cos_sin = self._compute_rotary_for_positions(
                                kept_positions
                            )
                        else:
                            rotary_cos_sin = self._recompute_rotary(x)

            # ---- Restore just before output if restore_gate_layer is None ----
            if not restored:
                if self.mr_config.deletion_mode == "hard" and keep_mask is not None:
                    # Critical fix: use pre-gate hidden states for deleted positions
                    # instead of zero logits, so the score network produces valid
                    # predictions for every position required by the diffusion loss.
                    x = self._restore_hidden_states(x, x_pre_gate, keep_mask)
                    indices = original_indices
                # Soft: no action needed — all L tokens present throughout

            logits = self.output_layer(x, c)  # always [B, L, vocab_size]

        # Sigma scaling (same as base SEDD)
        if self.scale_by_sigma:
            assert self.absorb, "scale_by_sigma requires absorb graph."
            esigm1_log = torch.where(
                sigma < 0.5,
                torch.expm1(sigma),
                sigma.exp() - 1,
            ).log().to(logits.dtype)[:, None, None]
            logits = logits - esigm1_log - np.log(logits.shape[-1] - 1)

        # Zero out logits for observed (non-noisy) token positions
        # indices is always original_indices here (restored above if needed)
        logits = torch.scatter(
            logits, -1, indices[..., None], torch.zeros_like(logits[..., :1])
        )

        return MrSEDDOutput(
            logits=logits,
            delete_gate_mask=delete_gate_mask,
            delete_gate_logits=gate_logits,
            delete_gate_output=gate_output,
            keep_indices=keep_mask,
        )