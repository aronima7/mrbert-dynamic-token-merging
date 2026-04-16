"""
MrBD3LM Configuration.

Extends BD3LMConfig (HuggingFace) with delete gate parameters.
The gate fires after `delete_gate_layer` transformer blocks and operates on
the noisy token (x_t) positions only.  Clean context (x_0) is never deleted.

Key insight from the guide (README_guide.md):
  BD3-LM uses masking-based noise, so corrupted tokens are all identical
  [MASK] tokens — perfect deletion targets.  A learned gate can aggressively
  collapse masked tokens while preserving unmasked (informative) ones.  This
  creates a natural curriculum: at high noise (many masks) delete more; at low
  noise (few masks) delete less — exactly what sigma-conditioned gates implement.
"""

from __future__ import annotations

import sys
import os

# BD3LM HF config lives in diffusion/bd3lms/models/hf/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BD3LMS_ROOT = os.path.join(os.path.dirname(_THIS_DIR), "diffusion", "bd3lms")
if _BD3LMS_ROOT not in sys.path:
    sys.path.insert(0, _BD3LMS_ROOT)

from models.hf.configuration_bd3lm import BD3LMConfig


class MrBD3LMConfig(BD3LMConfig):
    """
    BD3LMConfig + delete gate parameters.

    All BD3LMConfig fields are inherited.  New fields begin with ``delete_``
    or ``gate_`` or ``deletion_`` to avoid namespace collisions.

    Delete Gate Parameters
    ----------------------
    delete_gate_layer : int
        DDiTBlock index (0-indexed) after which the gate fires.
        Default 3 (25% depth for small model with 12 blocks).

    restore_gate_layer : int or None
        DDiTBlock index at which to restore the full-length representation.
        None (default) means restore just before the output layer.
        Must be > delete_gate_layer.

    deletion_type : str
        Gate variant: ``"scaled_sigmoid"`` (learnable), ``"random"``, ``"fixed"``.

    deletion_mode : str
        ``"soft"`` (additive attention bias, default) or ``"hard"`` (physical removal).

    sigmoid_mask_scale : float
        Scale factor for the ScaledSigmoid gate.  Gate output is in
        [sigmoid_mask_scale, 0].  Default -30.0.

    deletion_threshold : float
        Threshold for hard deletion and soft deletion rate metrics.
        Tokens with gate value <= deletion_threshold are counted as "deleted".

    gate_sigma_conditioned : bool
        Condition the gate on sigma (noise level).  Default True (guide-recommended).

    gate_layer_norm : bool
        Apply LayerNorm before the gate linear.  Default True.

    use_gumbel_noise : bool
        Add Gumbel noise to gate logits during training (exploration).

    deletion_rate_schedule : str
        ``"constant"`` | ``"linear_sigma"`` | ``"power_sigma"`` | ``"noise_fraction"``.
        For BD3-LM, ``"noise_fraction"`` is special: target rate = move_chance p(t),
        i.e. the expected fraction of masked tokens at timestep t.

    target_deletion_rate : float
        Used when deletion_rate_schedule == "constant".  Default 0.3.

    r_min : float
        Min deletion rate at sigma ≈ 0.  Used for ``"linear_sigma"`` / ``"power_sigma"``.

    r_max : float
        Max deletion rate at sigma ≈ sigma_max.

    deletion_rate_alpha : float
        Exponent for ``"power_sigma"`` schedule.  1.0 == linear.

    sigma_max : float
        Must match the noise schedule sigma_max.  Default 1.0 (LogLinear at t=1).

    deletion_loss_weight : float
        Weight of auxiliary deletion rate loss.  Default 0.1.

    random_deletion_probability : float
        Used for ``deletion_type == "random"``.

    fixed_deletion_amount : float
        Fraction to delete for ``deletion_type == "fixed"``.
    """

    model_type = "mr_bd3lm"

    def __init__(
        self,
        # BD3LM base params (with BD3LM defaults)
        block_size: int = 1,
        vocab_size: int = 50258,
        model_length: int = 1024,
        cross_attn: bool = True,
        adaln: bool = True,
        attn_backend: str = "sdpa",  # changed default from flex to sdpa for compatibility
        causal: bool = False,
        hidden_dim: int = 768,
        cond_dim: int = 128,
        n_blocks: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        time_conditioning: bool = True,
        var_min: bool = False,
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 0.999,
        # Delete gate params
        delete_gate_layer: int = 3,
        restore_gate_layer=None,
        deletion_type: str = "scaled_sigmoid",
        gate_type: str = "scaled_sigmoid",  # "scaled_sigmoid" | "bottleneck_mlp"
        gate_bottleneck_dim: int = 128,
        gate_logit_reg_weight: float = 0.001,
        deletion_mode: str = "soft",
        sigmoid_mask_scale: float = -30.0,
        deletion_threshold: float = -15.0,
        gate_sigma_conditioned: bool = True,
        gate_layer_norm: bool = True,
        use_gumbel_noise: bool = False,
        deletion_rate_schedule: str = "noise_fraction",
        target_deletion_rate: float = 0.3,
        r_min: float = 0.05,
        r_max: float = 0.5,
        deletion_rate_alpha: float = 1.0,
        sigma_max: float = 1.0,
        deletion_loss_weight: float = 0.1,
        random_deletion_probability: float = 0.3,
        fixed_deletion_amount: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            block_size=block_size,
            vocab_size=vocab_size,
            model_length=model_length,
            cross_attn=cross_attn,
            adaln=adaln,
            attn_backend=attn_backend,
            causal=causal,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            time_conditioning=time_conditioning,
            var_min=var_min,
            sampling_eps_min=sampling_eps_min,
            sampling_eps_max=sampling_eps_max,
            **kwargs,
        )
        self.delete_gate_layer = delete_gate_layer
        self.restore_gate_layer = restore_gate_layer
        self.deletion_type = deletion_type
        self.gate_type = gate_type
        self.gate_bottleneck_dim = gate_bottleneck_dim
        self.gate_logit_reg_weight = gate_logit_reg_weight
        self.deletion_mode = deletion_mode
        self.sigmoid_mask_scale = sigmoid_mask_scale
        self.deletion_threshold = deletion_threshold
        self.gate_sigma_conditioned = gate_sigma_conditioned
        self.gate_layer_norm = gate_layer_norm
        self.use_gumbel_noise = use_gumbel_noise
        self.deletion_rate_schedule = deletion_rate_schedule
        self.target_deletion_rate = target_deletion_rate
        self.r_min = r_min
        self.r_max = r_max
        self.deletion_rate_alpha = deletion_rate_alpha
        self.sigma_max = sigma_max
        self.deletion_loss_weight = deletion_loss_weight
        self.random_deletion_probability = random_deletion_probability
        self.fixed_deletion_amount = fixed_deletion_amount

    @property
    def _restore_layer_index(self):
        """Concrete block index for restoration (None → len(blocks)-1 + 1)."""
        return self.restore_gate_layer  # None means "just before output layer"