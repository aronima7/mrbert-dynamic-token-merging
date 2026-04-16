"""
MrBD3LM Model — BD3-LM with a delete gate mechanism.

Architecture overview
---------------------
Input: [x_t; x_0] concatenated (2L tokens) when cross_attn=True.
       During sampling, only x_t (L tokens) is used.

Phase 1 (blocks 0..delete_gate_layer):
    Full attention over 2L tokens using the block-diff mask.

Delete Gate:
    Fires on the x_t portion (first L tokens) after block delete_gate_layer.
    LayerNorm → Linear → ScaledSigmoid → gate [B, L, 1] in [sigmoid_mask_scale, 0].
    Optionally conditioned on sigma (noise level) for noise-level-aware deletion.

    Key insight: At high noise, many x_t tokens are [MASK] = identical embeddings.
    The gate can learn to aggressively collapse these while keeping unmasked tokens.

Phase 2 (blocks delete_gate_layer+1..restore_gate_layer):
    Soft: gate values added as additive attention bias on x_t keys → SDPA fallback.
    Hard: x_t tokens below threshold physically removed; x_0 kept intact.

Restoration (at restore_gate_layer, or just before output if None):
    Soft: clear gate mask → FlexAttention/SDPA resumes on full 2L.
    Hard: scatter compressed x_t states back; deleted positions get x_pre_gate.

Phase 3 (blocks restore_gate_layer+1..N):
    Full attention over 2L tokens again.

Output layer always sees full 2L → full L logits returned (no zero-logit scatter).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BD3LMS_ROOT = os.path.join(os.path.dirname(_THIS_DIR), "diffusion", "bd3lms")
if _BD3LMS_ROOT not in sys.path:
    sys.path.insert(0, _BD3LMS_ROOT)

from models.hf.modeling_bd3lm import (
    DDiTBlock,
    DDitFinalLayer,
    DITBackbone,
    EmbeddingLayer,
    LayerNorm,
    Rotary,
    TimestepEmbedder,
    apply_rotary_pos_emb_torchscript,
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    block_diff_mask,
    modulate_fused,
)

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False

from configuration_mrd_bd3lm import MrBD3LMConfig


# ══════════════════════════════════════════════════════════════════════════════
# Delete Gate Classes
# ══════════════════════════════════════════════════════════════════════════════

class ScaledSigmoid(nn.Module):
    """
    ScaledSigmoid(x) = scale * sigmoid(-x).

    With scale = -30 and bias initialised to 10 (easy-keep):
      - Large positive input → sigmoid(-x)≈0 → output ≈ 0     (keep)
      - Large negative input → sigmoid(-x)≈1 → output ≈ scale (delete)
    """
    def __init__(self, scale: float = -30.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.sigmoid(-x)


class SigmoidDeleteGate(nn.Module):
    """
    Learnable gate conditioned on token hidden state only.

    Linear(hidden_dim → 1) → ScaledSigmoid
    Bias initialised to +10 → gate starts near 0 (keep all tokens).
    Returns values in [sigmoid_mask_scale, 0].
    """
    def __init__(self, hidden_dim: int, sigmoid_mask_scale: float = -30.0,
                 use_layer_norm: bool = True):
        super().__init__()
        self.layer_norm = LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        self.linear.bias.data.fill_(10.0)
        nn.init.normal_(self.linear.weight, std=0.001)
        self.activation = ScaledSigmoid(sigmoid_mask_scale)

    def forward(self, x: torch.Tensor, sigma=None) -> torch.Tensor:
        """x: [B, L, D] → gate [B, L, 1] in [sigmoid_mask_scale, 0]."""
        return self.activation(self.linear(self.layer_norm(x)))


class SigmoidDeleteGateWithSigma(nn.Module):
    """
    Learnable gate conditioned on token hidden state AND sigma (noise level).

    The sigma conditioning is injected by expanding the sigma embedding to
    match sequence length and concatenating with token hidden states.

    This is the guide-recommended default: gate must be noise-level-aware.
    At high noise (many [MASK] tokens), delete aggressively.
    At low noise (mostly clean tokens), delete conservatively.
    """
    def __init__(self, hidden_dim: int, cond_dim: int,
                 sigmoid_mask_scale: float = -30.0, use_layer_norm: bool = True):
        super().__init__()
        self.layer_norm = LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(hidden_dim + cond_dim, 1, bias=True)
        self.linear.bias.data.fill_(10.0)
        nn.init.normal_(self.linear.weight, std=0.001)
        self.activation = ScaledSigmoid(sigmoid_mask_scale)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, L, D]
        sigma: [B, cond_dim]   (pre-computed by TimestepEmbedder)
        → gate [B, L, 1] in [sigmoid_mask_scale, 0]
        """
        L = x.shape[1]
        sigma_expanded = sigma.unsqueeze(1).expand(-1, L, -1)  # [B, L, cond_dim]
        h = torch.cat([self.layer_norm(x), sigma_expanded], dim=-1)  # [B, L, D+cond_dim]
        return self.activation(self.linear(h))


class RandomDeleteGate(nn.Module):
    """Baseline: delete each token independently with fixed probability."""
    def __init__(self, probability: float = 0.3, sigmoid_mask_scale: float = -30.0):
        super().__init__()
        self.probability = probability
        self.sigmoid_mask_scale = sigmoid_mask_scale

    def forward(self, x: torch.Tensor, sigma=None) -> torch.Tensor:
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device)
        return torch.where(mask < self.probability,
                           torch.full_like(mask, self.sigmoid_mask_scale),
                           torch.zeros_like(mask))


class FixedDeleteGate(nn.Module):
    """Baseline: always delete the last `fraction` tokens by position."""
    def __init__(self, fraction: float = 0.3, sigmoid_mask_scale: float = -30.0):
        super().__init__()
        self.fraction = fraction
        self.sigmoid_mask_scale = sigmoid_mask_scale

    def forward(self, x: torch.Tensor, sigma=None,
                temperature: float = 1.0) -> tuple:
        B, L, _ = x.shape
        n_delete = int(L * self.fraction)
        gate = torch.zeros(B, L, 1, device=x.device)
        gate[:, L - n_delete:] = self.sigmoid_mask_scale
        return gate, gate  # (gate_values, logits)


class BottleneckDeleteGate(nn.Module):
    """
    Plan-specified 2-layer bottleneck MLP gate (plan Section 2.2).

    Architecture:
        noise_proj: Linear(D → D)
        gate_mlp:   Linear(2D → bottleneck_dim) → GELU → Linear(bottleneck_dim → 1)

    Uses Gumbel-sigmoid with temperature annealing during training.
    Hard decisions via straight-through estimator (STE).

    Output is a gate in [0, 1] where 1 = keep, 0 = delete.
    For soft deletion we convert to additive attention bias via:
        attn_bias = sigmoid_mask_scale * (1 - gate)
    """

    def __init__(self, hidden_dim: int, cond_dim: int,
                 bottleneck_dim: int = 128,
                 sigmoid_mask_scale: float = -30.0,
                 use_layer_norm: bool = True):
        super().__init__()
        self.sigmoid_mask_scale = sigmoid_mask_scale
        self.layer_norm = LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        # Project noise embedding to match hidden_dim for concatenation
        self.noise_proj = nn.Linear(cond_dim, hidden_dim)
        # 2-layer bottleneck MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
        )
        # Init: large positive bias → gate starts near 1 (keep all tokens)
        self.gate_mlp[-1].bias.data.fill_(3.0)
        nn.init.normal_(self.gate_mlp[0].weight, std=0.01)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor, sigma: Optional[torch.Tensor],
                temperature: float = 1.0) -> tuple:
        """
        x:    [B, L, D]
        sigma: [B, cond_dim]  noise embedding
        temperature: Gumbel-sigmoid temperature (anneal 2.0 → 0.5)

        Returns:
            gate_values: [B, L, 1]  soft gate in [0, 1]  (1=keep, 0=delete)
            logits:      [B, L, 1]  raw gate logits before sigmoid
        """
        h = self.layer_norm(x)
        if sigma is not None:
            noise_expanded = self.noise_proj(sigma).unsqueeze(1).expand_as(h)
            gate_input = torch.cat([h, noise_expanded], dim=-1)
        else:
            gate_input = torch.cat([h, torch.zeros_like(h)], dim=-1)

        logits = self.gate_mlp(gate_input)  # [B, L, 1]

        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            gate_values = torch.sigmoid((logits + gumbel) / temperature)
        else:
            gate_values = torch.sigmoid(logits)

        return gate_values, logits


# ── Updated existing gates to return (gate_values, logits) ─────────────────

class SigmoidDeleteGate(nn.Module):
    """
    Learnable gate conditioned on token hidden state only.

    Linear(hidden_dim → 1) → ScaledSigmoid
    Bias initialised to +10 → gate starts near 0 (keep all tokens).
    Returns values in [sigmoid_mask_scale, 0].
    """
    def __init__(self, hidden_dim: int, sigmoid_mask_scale: float = -30.0,
                 use_layer_norm: bool = True):
        super().__init__()
        self.layer_norm = LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        self.linear.bias.data.fill_(10.0)
        nn.init.normal_(self.linear.weight, std=0.001)
        self.activation = ScaledSigmoid(sigmoid_mask_scale)

    def forward(self, x: torch.Tensor, sigma=None,
                temperature: float = 1.0) -> tuple:
        """Returns (gate_values, logits)."""
        logits = self.linear(self.layer_norm(x))
        return self.activation(logits), logits


class SigmoidDeleteGateWithSigma(nn.Module):
    """
    Learnable gate conditioned on token hidden state AND sigma (noise level).
    Returns values in [sigmoid_mask_scale, 0].
    """
    def __init__(self, hidden_dim: int, cond_dim: int,
                 sigmoid_mask_scale: float = -30.0, use_layer_norm: bool = True):
        super().__init__()
        self.layer_norm = LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(hidden_dim + cond_dim, 1, bias=True)
        self.linear.bias.data.fill_(10.0)
        nn.init.normal_(self.linear.weight, std=0.001)
        self.activation = ScaledSigmoid(sigmoid_mask_scale)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor,
                temperature: float = 1.0) -> tuple:
        L = x.shape[1]
        sigma_expanded = sigma.unsqueeze(1).expand(-1, L, -1)
        h = torch.cat([self.layer_norm(x), sigma_expanded], dim=-1)
        logits = self.linear(h)
        return self.activation(logits), logits


class PreDeletionBlend(nn.Module):
    """
    Before deletion, blend each token with its next surviving neighbor.

    h'_i = g_i * h_i + (1 - g_i) * h_{next}

    where g_i ∈ [0,1] (1=keep, 0=delete).  For BottleneckDeleteGate, gate_values
    are already in [0,1].  For ScaledSigmoid gates, convert first.
    """

    def forward(self, hidden_states: torch.Tensor,
                gate_values: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, L, D]
        gate_values:   [B, L, 1]  values in [0, 1] where 1=keep

        Returns blended: [B, L, D]
        """
        # Shift right: h_next[i] = h[i+1]; last position keeps itself
        h_next = torch.roll(hidden_states, -1, dims=1)
        h_next[:, -1, :] = hidden_states[:, -1, :]
        return gate_values * hidden_states + (1 - gate_values) * h_next


class TokenMerge(nn.Module):
    """
    Remove deleted tokens from the sequence (hard deletion).

    Returns compressed sequence and merge_info dict for restore.
    """

    def forward(self, hidden_states: torch.Tensor,
                gate_hard: torch.Tensor) -> tuple:
        """
        hidden_states: [B, L, D]
        gate_hard:     [B, L, 1]  binary keep decisions (1=keep)

        Returns:
            merged:     [B, L_max', D]  padded compressed sequence
            merge_info: dict with keys needed for TokenRestore
        """
        B, L, D = hidden_states.shape
        keep_mask = gate_hard.squeeze(-1).bool()  # [B, L]
        keep_counts = keep_mask.sum(dim=1)  # [B]
        L_prime = int(keep_counts.max().item())
        L_prime = max(L_prime, 1)

        merged = torch.zeros(B, L_prime, D,
                             device=hidden_states.device, dtype=hidden_states.dtype)
        merged_positions = torch.zeros(B, L_prime, dtype=torch.long,
                                       device=hidden_states.device)
        valid_mask = torch.zeros(B, L_prime, dtype=torch.bool,
                                 device=hidden_states.device)

        for b in range(B):
            kept = keep_mask[b].nonzero(as_tuple=True)[0]
            n = kept.shape[0]
            if n > 0:
                merged[b, :n] = hidden_states[b, kept]
                merged_positions[b, :n] = kept
                valid_mask[b, :n] = True

        merge_info = {
            "keep_mask": keep_mask,      # [B, L] bool
            "keep_counts": keep_counts,  # [B] int
            "original_length": L,
            "merged_positions": merged_positions,  # [B, L'] original indices
            "valid_mask": valid_mask,    # [B, L'] which slots are real
        }
        return merged, merge_info


class TokenRestore(nn.Module):
    """
    Scatter merged tokens back to full-length sequence.

    Deleted positions receive their pre-gate hidden states (not zeros),
    matching the guide requirement: "deleted positions get pre-gate hidden states
    via _restore_hidden_states, NOT zero logits."
    """

    def forward(self, merged: torch.Tensor, merge_info: dict,
                pre_gate_states: torch.Tensor) -> torch.Tensor:
        """
        merged:          [B, L', D]
        merge_info:      dict from TokenMerge
        pre_gate_states: [B, L, D]  fallback for deleted positions

        Returns restored: [B, L, D]
        """
        B = merged.shape[0]
        L = merge_info["original_length"]
        keep_mask = merge_info["keep_mask"]

        # Start from pre-gate states (deleted positions keep pre-gate values)
        restored = pre_gate_states.clone()

        for b in range(B):
            kept = keep_mask[b].nonzero(as_tuple=True)[0]
            n = kept.shape[0]
            if n > 0:
                restored[b, kept] = merged[b, :n]

        return restored


def build_delete_gate(config: MrBD3LMConfig) -> nn.Module:
    """Factory: construct the delete gate specified by config."""
    gate_type = getattr(config, "gate_type", config.deletion_type)

    if gate_type == "bottleneck_mlp":
        return BottleneckDeleteGate(
            config.hidden_dim,
            config.cond_dim,
            getattr(config, "gate_bottleneck_dim", 128),
            config.sigmoid_mask_scale,
            config.gate_layer_norm,
        )
    elif config.deletion_type == "scaled_sigmoid":
        if config.gate_sigma_conditioned:
            return SigmoidDeleteGateWithSigma(
                config.hidden_dim,
                config.cond_dim,
                config.sigmoid_mask_scale,
                config.gate_layer_norm,
            )
        return SigmoidDeleteGate(
            config.hidden_dim,
            config.sigmoid_mask_scale,
            config.gate_layer_norm,
        )
    elif config.deletion_type == "random":
        return RandomDeleteGate(config.random_deletion_probability,
                                config.sigmoid_mask_scale)
    elif config.deletion_type == "fixed":
        return FixedDeleteGate(config.fixed_deletion_amount,
                               config.sigmoid_mask_scale)
    else:
        raise ValueError(f"Unknown deletion_type/gate_type: {config.deletion_type}")


# ══════════════════════════════════════════════════════════════════════════════
# MrDDiTBlock — DDiTBlock extended with per-sample delete gate mask support
# ══════════════════════════════════════════════════════════════════════════════

class MrDDiTBlock(DDiTBlock):
    """
    DDiTBlock with optional delete gate mask.

    When delete_gate_mask is None:  standard forward (FlexAttention or SDPA).
    When delete_gate_mask is given: SDPA fallback with gate bias on x_t keys.

    The gate mask is applied ONLY to the x_t portion (first n_xt keys).
    The x_0 portion (clean context) is never masked.
    """

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, mask=None,
                delete_gate_mask=None, sample_mode=False, store_kv=False):
        if delete_gate_mask is not None:
            return self._forward_with_delete_mask(
                x, rotary_cos_sin, c, mask, delete_gate_mask)
        return super().forward(x, rotary_cos_sin, c, mask=mask,
                               sample_mode=sample_mode, store_kv=store_kv)

    def _forward_with_delete_mask(self, x, rotary_cos_sin, c, struct_mask,
                                   delete_gate_mask):
        """
        SDPA forward with combined structural + delete gate bias.

        delete_gate_mask : [B, n_xt, 1]  gate values in [sigmoid_mask_scale, 0]
            Added to attention scores for x_t KEY positions only.
        struct_mask      : [2n, 2n] bool (or BlockMask)  structural block-diff mask.
        """
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        if self.adaln:
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = (
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2))

        x_skip = x
        x_norm = modulate_fused(self.norm1(x), shift_msa, scale_msa) if self.adaln else self.norm1(x)

        # ── build QKV for the full [x_t; x_0] sequence ───────────────────────
        n_xt = delete_gate_mask.shape[1]  # L (noisy token count)
        qkv_xt = self.get_qkv(x_norm[:, :n_xt], rotary_cos_sin)     # [B, n_xt, 3, H, dh]
        qkv_x0 = self.get_qkv(x_norm[:, n_xt:], rotary_cos_sin)     # [B, n_x0, 3, H, dh]
        qkv = torch.cat([qkv_xt, qkv_x0], dim=1)                     # [B, 2L, 3, H, dh]

        B, seq_len, _, H, dh = qkv.shape
        scale = 1.0 / math.sqrt(dh)

        # [B, H, seq_len, dh]
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        # ── build additive attention bias [B, H, seq_len, seq_len] ───────────
        # Start from structural mask (block-diff pattern)
        if struct_mask is not None:
            # struct_mask: [2L, 2L] bool (True=attend, False=mask)
            bool_mask = struct_mask.bool() if not isinstance(struct_mask, bool) else struct_mask
            # Convert to [1, 1, 2L, 2L] additive bias
            struct_bias = torch.zeros(
                1, 1, seq_len, seq_len, device=x.device, dtype=x.dtype)
            struct_bias.masked_fill_(~bool_mask.unsqueeze(0).unsqueeze(0), -1e9)
        else:
            struct_bias = torch.zeros(
                1, 1, seq_len, seq_len, device=x.device, dtype=x.dtype)

        # Add delete gate bias to x_t KEY columns (first n_xt columns)
        # delete_gate_mask: [B, n_xt, 1] → [B, 1, 1, n_xt] → broadcast to all Q rows
        gate_key_bias = delete_gate_mask.squeeze(-1).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, n_xt]
        # Full bias: [B, H, seq_len, seq_len]
        attn_bias = struct_bias.expand(B, H, -1, -1).clone()
        attn_bias[:, :, :, :n_xt] = attn_bias[:, :, :, :n_xt] + gate_key_bias

        # ── SDPA ──────────────────────────────────────────────────────────────
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=scale)  # [B, H, seq_len, dh]
        x = einops.rearrange(attn_out, 'b h s d -> b s (h d)')

        # ── MLP residual ──────────────────────────────────────────────────────
        if self.adaln:
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, gate_msa, x_skip, self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None, gate_mlp, x, self.dropout)
        else:
            ones = torch.ones_like(x)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, ones, x_skip, self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, ones, x, self.dropout)

        return x


# ══════════════════════════════════════════════════════════════════════════════
# MrDITBackbone — DITBackbone with 3-phase delete gate forward pass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MrBD3LMOutput:
    """Return type from MrBD3LM.forward()."""
    logits: torch.Tensor            # [B, L, vocab_size]
    delete_gate_output: Optional[torch.Tensor]  # [B, L, 1] gate values or None
    delete_gate_mask: Optional[torch.Tensor]    # same as delete_gate_output during phase 2
    deletion_rate: Optional[torch.Tensor]       # [B] actual deletion fraction


class MrDITBackbone(nn.Module):
    """
    BD3-LM DITBackbone extended with 3-phase delete gate mechanism.

    Phase 1 : blocks[0 .. delete_gate_layer]          — full 2L attention
    Gate    : fires on x_t portion (first L tokens)
    Phase 2 : blocks[delete_gate_layer+1 .. restore]  — compressed/biased
    Restore : expand L' → L before phase 3 (hard) or clear mask (soft)
    Phase 3 : blocks[restore+1 .. N]                  — full 2L attention
    Output  : DDitFinalLayer on full 2L → first L logits returned
    """

    def __init__(self, config: MrBD3LMConfig):
        super().__init__()
        self.config = config
        self.cross_attn = config.cross_attn
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.n = config.model_length  # L

        self.vocab_embed = EmbeddingLayer(config.hidden_dim, config.vocab_size)

        self.adaln = config.adaln
        if self.adaln:
            self.sigma_map = TimestepEmbedder(config.cond_dim)

        self.rotary_emb = Rotary(config.hidden_dim // config.n_heads)

        # Replace DDiTBlock with MrDDiTBlock
        self.blocks = nn.ModuleList([
            MrDDiTBlock(
                n=self.n,
                block_size=config.block_size,
                dim=config.hidden_dim,
                n_heads=config.n_heads,
                cond_dim=config.cond_dim,
                causal=config.causal,
                dropout=config.dropout,
                adaln=config.adaln,
                attn_backend="sdpa",  # always SDPA (flex not compatible with delete bias)
            )
            for _ in range(config.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(
            config.hidden_dim, config.vocab_size, config.cond_dim, adaln=config.adaln)

        if self.cross_attn:
            self.gen_mask(config.model_length, config.block_size, attn_backend="sdpa")

        # Delete gate + blend/merge/restore modules
        self.delete_gate = build_delete_gate(config)
        self.pre_blend = PreDeletionBlend()
        self.token_merge = TokenMerge()
        self.token_restore = TokenRestore()
        self.precision = torch.float32

    def gen_mask(self, seqlen, block_size, attn_backend="sdpa"):
        """Generate structural block-diff attention mask."""
        self.mask = block_diff_mask(
            b=None, h=None,
            q_idx=torch.arange(seqlen * 2)[:, None],
            kv_idx=torch.arange(seqlen * 2)[None, :],
            block_size=block_size, n=seqlen)

    def _apply_delete_gate(self, x: torch.Tensor, c: Optional[torch.Tensor],
                           use_gumbel: bool = False,
                           temperature: float = 1.0) -> tuple:
        """
        Compute gate values for the x_t portion (first L tokens).

        Returns (gate [B, L, 1], gate_logits [B, L, 1]).
        """
        x_xt = x[:, :self.n]  # [B, L, D]

        gate_out = self.delete_gate(x_xt, c)
        if isinstance(gate_out, tuple):
            gate, logits = gate_out
        else:
            gate, logits = gate_out, gate_out  # legacy gates return single tensor

        if use_gumbel and self.training and not isinstance(self.delete_gate, BottleneckDeleteGate):
            # BottleneckDeleteGate handles gumbel internally via temperature
            gumbel = -torch.empty_like(gate).exponential_().log()
            gate = gate + gumbel * 0.1

        return gate, logits

    def _restore_hidden_states(
        self,
        x_compressed_xt: torch.Tensor,  # [B, L_max', D]
        x_pre_gate_xt: torch.Tensor,    # [B, L, D]
        keep_mask: torch.Tensor,        # [B, L] bool
    ) -> torch.Tensor:
        """
        Scatter compressed x_t states back to full-length positions.
        Deleted positions receive their pre-gate hidden states (not zeros).

        Delegates to TokenRestore.
        """
        L_prime = x_compressed_xt.shape[1]
        merge_info = {
            "keep_mask": keep_mask,
            "original_length": x_pre_gate_xt.shape[1],
        }
        return self.token_restore(x_compressed_xt, merge_info, x_pre_gate_xt)

    def _recompute_rotary(self, x: torch.Tensor) -> tuple:
        """Recompute RoPE for a (possibly shorter) sequence."""
        return self.rotary_emb(x)

    def forward(
        self,
        indices: torch.Tensor,              # [B, 2L] (x_t; x_0) or [B, L] when sample_mode
        sigma: Optional[torch.Tensor],      # [B] noise level
        sample_mode: bool = False,
        store_kv: bool = False,
        output_hidden_states: bool = False,
        gate_temperature: float = 1.0,      # Gumbel-sigmoid temperature (annealed during training)
    ):
        """
        3-phase forward pass.

        Returns (logits [B, L, V], all_hidden_states list).
        Also attaches `_last_gate_output` attribute for loss access.
        """
        if not self.config.time_conditioning and self.adaln:
            sigma = torch.zeros_like(sigma) if sigma is not None else sigma

        x = self.vocab_embed(indices)  # [B, 2L, D] or [B, L, D]
        all_hidden_states = [x] if output_hidden_states else []

        c = None
        if self.adaln and sigma is not None:
            c = F.silu(self.sigma_map(sigma))  # [B, cond_dim]

        n_xt = self.n  # L

        if self.cross_attn and not sample_mode:
            mask = self.mask.to(x.device)
            rotary_cos_sin = self.rotary_emb(x[:, :n_xt])
        else:
            mask = None
            rotary_cos_sin = self.rotary_emb(x)

        gate_output = None
        gate_logits = None           # raw logits for regularization
        delete_gate_mask = None  # active gate values during phase 2
        x_pre_gate_xt = None
        keep_mask = None
        x_xt_compressed = None

        delete_gate_layer = self.config.delete_gate_layer
        restore_gate_layer = self.config.restore_gate_layer  # int or None
        n_blocks = len(self.blocks)

        # Effective restore index: None → after last block (just before output)
        restore_idx = restore_gate_layer if restore_gate_layer is not None else n_blocks

        with torch.cuda.amp.autocast(dtype=self.precision):
            for i, block in enumerate(self.blocks):

                # ── Phase 1: full attention ───────────────────────────────────
                if i <= delete_gate_layer:
                    x = block(x, rotary_cos_sin, c, mask=mask,
                              delete_gate_mask=None,
                              sample_mode=sample_mode, store_kv=store_kv)

                    # Gate fires immediately after delete_gate_layer
                    if i == delete_gate_layer and not sample_mode:
                        gate_output, gate_logits = self._apply_delete_gate(
                            x, c, self.config.use_gumbel_noise, gate_temperature)

                        if self.config.deletion_mode == "soft":
                            delete_gate_mask = gate_output  # [B, L, 1]

                        elif self.config.deletion_mode == "hard":
                            # Use STE: gate_hard = round(gate), gradients flow through gate
                            if isinstance(self.delete_gate, BottleneckDeleteGate):
                                # gate_output in [0,1]; 1=keep
                                gate_hard = (gate_output > 0.5).float()
                                gate_hard = gate_output + (gate_hard - gate_output).detach()
                                keep_mask = gate_hard.squeeze(-1).bool()
                                # PreDeletionBlend before merging
                                x_xt_portion = self.pre_blend(x[:, :n_xt], gate_hard[:, :, :1])
                                x = torch.cat([x_xt_portion, x[:, n_xt:]], dim=1)
                            else:
                                # Legacy ScaledSigmoid gates: threshold in [scale, 0]
                                keep_mask = (gate_output.squeeze(-1)
                                             >= self.config.deletion_threshold)
                            keep_mask[:, 0] = True  # always keep first token
                            x_pre_gate_xt = x[:, :n_xt].clone()
                            x_xt_compressed, x_for_phase2 = self._compress_xt(x, keep_mask)
                            x = x_for_phase2

                # ── Phase 2: compressed/biased attention ──────────────────────
                elif i > delete_gate_layer and i < restore_idx:
                    if self.config.deletion_mode == "soft":
                        # Pass delete gate mask → SDPA fallback, gate biases xt keys
                        x = block(x, rotary_cos_sin, c, mask=mask,
                                  delete_gate_mask=delete_gate_mask)
                    elif self.config.deletion_mode == "hard":
                        # x is now [B, L'+L, D], no block-diff mask
                        rotary_cs_compressed = self._recompute_rotary(x)
                        x = block(x, rotary_cs_compressed, c, mask=None,
                                  delete_gate_mask=None)

                # ── Restoration ───────────────────────────────────────────────
                if i == restore_idx - 1 and not sample_mode:
                    if self.config.deletion_mode == "soft":
                        delete_gate_mask = None  # clear mask → back to full attn
                    elif self.config.deletion_mode == "hard" and keep_mask is not None:
                        x_xt_restored = self._restore_hidden_states(
                            x[:, :x_xt_compressed.shape[1]],
                            x_pre_gate_xt,
                            keep_mask)
                        x = torch.cat([x_xt_restored, x[:, -n_xt:]], dim=1)
                        # Restore RoPE
                        rotary_cos_sin = self._recompute_rotary(x[:, :n_xt])

                # ── Phase 3: full attention ────────────────────────────────────
                # (handled above; blocks after restore_idx use mask=mask, no gate)
                if i >= restore_idx:
                    x = block(x, rotary_cos_sin, c, mask=mask,
                              delete_gate_mask=None)

                if output_hidden_states:
                    all_hidden_states.append(x)

            logits = self.output_layer(x, c)

        if self.cross_attn and not sample_mode:
            logits = logits[:, :n_xt]
            if output_hidden_states:
                all_hidden_states = [h[:, :n_xt] for h in all_hidden_states]

        # Store gate outputs for loss computation (accessed by training loop)
        self._last_gate_output = gate_output   # [B, L, 1] or None
        self._last_gate_logits = gate_logits   # [B, L, 1] or None (raw logits)

        return logits, all_hidden_states

    def _compress_xt(self, x: torch.Tensor, keep_mask: torch.Tensor):
        """
        Physically remove x_t tokens below threshold.

        keep_mask : [B, L] bool
        x         : [B, 2L, D]

        Returns:
            x_xt_compressed : [B, L_max', D]  (compressed x_t, padded to L_max')
            x_for_phase2    : [B, L_max'+L, D] cat([compressed_x_t, x_0])
        """
        B, _, D = x.shape
        n_xt = keep_mask.shape[1]
        x_xt = x[:, :n_xt]   # [B, L, D]
        x_x0 = x[:, n_xt:]   # [B, L, D]  clean context — always kept

        L_prime = keep_mask.sum(dim=1).max().item()  # max kept across batch
        x_xt_compressed = torch.zeros(B, int(L_prime), D,
                                      device=x.device, dtype=x.dtype)
        for b in range(B):
            kept = keep_mask[b].nonzero(as_tuple=True)[0]
            x_xt_compressed[b, :kept.shape[0]] = x_xt[b, kept]

        x_for_phase2 = torch.cat([x_xt_compressed, x_x0], dim=1)
        return x_xt_compressed, x_for_phase2

    def forward_block(
        self,
        x_block: torch.Tensor,      # [B, block_size] noised block token ids
        kv_cache: Optional[torch.Tensor],  # [B, L_past, D*3] or None
        sigma: torch.Tensor,        # [B] noise level
        gate_temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Within-block denoising with delete gate operating only on the current block.

        Used when block_size > 1 to achieve within-block merging while keeping
        cross-block attention (KV cache) at full resolution.

        Args:
            x_block: [B, block_size] — noised tokens for current block
            kv_cache: [B, L_past, D*3] — keys+values from previous clean blocks
            sigma: [B] — noise level for this block
            gate_temperature: float — for Gumbel annealing

        Returns:
            logits: [B, block_size, vocab_size]
        """
        B, Lb = x_block.shape
        D = self.config.hidden_dim

        # Embed block
        h = self.vocab_embed(x_block)  # [B, Lb, D]

        c = None
        if self.adaln and sigma is not None:
            c = F.silu(self.sigma_map(sigma))

        rotary_cs = self.rotary_emb(h)

        delete_gate_layer = self.config.delete_gate_layer
        restore_gate_layer = self.config.restore_gate_layer
        n_blocks = len(self.blocks)
        restore_idx = restore_gate_layer if restore_gate_layer is not None else n_blocks

        gate_output = None
        gate_logits_out = None
        keep_mask = None
        h_pre_gate = None
        h_compressed = None

        for i, block in enumerate(self.blocks):
            h = block(h, rotary_cs, c, mask=None, delete_gate_mask=None)

            if i == delete_gate_layer:
                gate_out = self.delete_gate(h, c)
                if isinstance(gate_out, tuple):
                    gate_output, gate_logits_out = gate_out
                else:
                    gate_output, gate_logits_out = gate_out, gate_out

                if isinstance(self.delete_gate, BottleneckDeleteGate):
                    gate_hard = (gate_output > 0.5).float()
                    gate_hard = gate_output + (gate_hard - gate_output).detach()
                    keep_mask = gate_hard.squeeze(-1).bool()
                    keep_mask[:, 0] = True
                    h = self.pre_blend(h, gate_hard)
                    h_pre_gate = h.clone()
                    h_compressed, merge_info = self.token_merge(h, gate_hard)
                    h = h_compressed
                    rotary_cs = self.rotary_emb(h)

            elif i == restore_idx - 1 and keep_mask is not None:
                h = self.token_restore(h, merge_info, h_pre_gate)
                rotary_cs = self.rotary_emb(h)

        self._last_gate_output = gate_output
        self._last_gate_logits = gate_logits_out

        logits = self.output_layer(h, c)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# MrBD3LM — HuggingFace-compatible model
# ══════════════════════════════════════════════════════════════════════════════

import transformers
from transformers import modeling_outputs


class MrBD3LM(transformers.PreTrainedModel):
    """
    HuggingFace-compatible MrBD3LM.

    Wraps MrDITBackbone.  Use forward() for training; reset_kv_cache() for
    block-by-block sampling.

    Delete gate output is available via model.backbone._last_gate_output
    after each forward pass.
    """
    config_class = MrBD3LMConfig
    base_model_prefix = "mr_bd3lm"

    def __init__(self, config: MrBD3LMConfig):
        super().__init__(config)
        self.config = config
        self.backbone = MrDITBackbone(config)

    def reset_kv_cache(self, eval_batch_size: int = 1):
        for block in self.backbone.blocks:
            block.kv_cache = torch.zeros(
                eval_batch_size,
                self.config.model_length,
                self.config.hidden_dim * 3,
                device="cuda",
                dtype=torch.bfloat16)
            block.cache_idx = 0

    def forward(
        self,
        input_ids: torch.LongTensor,
        timesteps: Optional[torch.FloatTensor] = None,
        sample_mode: Optional[bool] = False,
        store_kv: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        gate_temperature: float = 1.0,
    ):
        """HF-compatible forward.  Returns MaskedLMOutput or raw logits."""
        return_dict = (return_dict
                       if return_dict is not None
                       else self.config.use_return_dict)

        logits, all_hidden_states = self.backbone(
            indices=input_ids,
            sigma=timesteps,
            sample_mode=sample_mode,
            store_kv=store_kv,
            output_hidden_states=output_hidden_states,
            gate_temperature=gate_temperature,
        )

        if return_dict:
            return modeling_outputs.MaskedLMOutput(
                logits=logits,
                hidden_states=all_hidden_states if output_hidden_states else None,
                loss=None,
            )
        if output_hidden_states:
            return logits, all_hidden_states
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# Deletion rate helpers
# ══════════════════════════════════════════════════════════════════════════════

def target_rate_at_timestep(
    move_chance: torch.Tensor,  # [B] or [B, L]  p(t) from noise schedule
    config: MrBD3LMConfig,
) -> torch.Tensor:
    """
    Compute per-sample target deletion rate as a function of move_chance p(t).

    Schedules
    ---------
    "noise_fraction" (BD3-LM-specific):
        target = p(t) — delete exactly the fraction of tokens that are masked.
        This is the natural target: gate should delete masked tokens (all [MASK]).
    "constant":
        Returns config.target_deletion_rate for all t.
    "linear_sigma":
        r(t) = r_min + (r_max - r_min) * clamp(p(t) / p_max, 0, 1)
    "power_sigma":
        r(t) = r_min + (r_max - r_min) * clamp(p(t), 0, 1) ^ alpha
    """
    schedule = config.deletion_rate_schedule

    if schedule == "constant":
        return torch.full(
            (move_chance.shape[0],), config.target_deletion_rate,
            device=move_chance.device, dtype=move_chance.dtype)

    # Reduce to [B] if needed
    p = move_chance.view(move_chance.shape[0], -1).mean(-1)  # [B]

    if schedule == "noise_fraction":
        return p.clamp(0.0, 1.0)

    frac = p.clamp(0.0, 1.0)
    if schedule == "power_sigma":
        frac = frac ** config.deletion_rate_alpha
    # linear_sigma: alpha=1
    return config.r_min + (config.r_max - config.r_min) * frac


def deletion_rate_loss(
    gate_output: torch.Tensor,    # [B, L, 1]
    target_rates: torch.Tensor,   # [B] or scalar
    deletion_threshold: float,
) -> torch.Tensor:
    """
    MSE loss between actual soft deletion rate and target rate.

    Soft proxy: soft_delete_i = sigmoid(-gate_i / |threshold|)
    actual_rate = mean_i(soft_delete_i)  per sample.
    """
    soft_del = torch.sigmoid(-gate_output / (abs(deletion_threshold) + 1e-8))
    actual_rates = soft_del.mean(dim=1).squeeze(-1)  # [B]

    if isinstance(target_rates, torch.Tensor):
        targets = target_rates.to(actual_rates.dtype).to(actual_rates.device)
    else:
        targets = torch.full_like(actual_rates, float(target_rates))

    return F.mse_loss(actual_rates, targets)