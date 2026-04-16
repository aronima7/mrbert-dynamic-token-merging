"""MrDiffusion configuration - SEDD with MrT5-style delete gates."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MrDiffusionConfig:
    """
    Delete gate configuration for MrSEDD.

    These parameters extend the base SEDD OmegaConf config with delete gate
    parameters inspired by the MrT5/MrBERT approach, updated to address the
    unique challenges of discrete diffusion (iterative sampling, per-position
    score output, noise-level-dependent information content).

    Gate placement
    --------------
    delete_gate_layer (int):
        DDiTBlock index after which the gate fires.  Default is 3, matching
        MrT5/MrBERT convention (25% depth for the small 12-block model).
    restore_gate_layer (Optional[int]):
        DDiTBlock index at which deleted/suppressed tokens are restored to full
        sequence length.  Blocks in [delete_gate_layer+1, restore_gate_layer]
        run with the reduced / biased sequence; blocks after restore_gate_layer
        run full FlashAttention on the complete L-length sequence again.

        Setting this to a layer before the final block (e.g. delete_gate_layer+3)
        lets the outer transformer layers reconstruct coherent representations
        for all L positions before the output projection, which is critical for
        producing valid per-position scores needed by the diffusion loss.

        None (default): restore just before the output layer.  For *soft*
        deletion this is a no-op (tokens were never removed); for *hard*
        deletion the pre-gate hidden states are used for deleted positions
        instead of zero logits.

    Gate variant
    ------------
    deletion_type (str):
        "scaled_sigmoid" | "log_sigmoid" | "random" | "fixed".
    sigmoid_mask_scale (float):
        Negative scale applied to sigmoid output.  More negative → stronger
        deletion signal.  Default -30.0 (MrT5).
    deletion_threshold (float):
        Gate value at or below which a token is counted as deleted for metrics
        and hard-deletion logic.  Default -15.0 (sigmoid_mask_scale / 2).
    gate_layer_norm (bool):
        Apply LayerNorm to hidden state before gate linear projection.
    gate_sigma_conditioned (bool):
        Whether the gate also receives the sigma (noise level) embedding.
        Defaults to True — the guide explicitly identifies noise-level awareness
        as essential: at high σ most tokens are corrupted and cheap to drop; at
        low σ tokens carry critical signal and should be preserved.
    deletion_mode (str):
        "soft": gate values are added as large negative attention key-biases in
        subsequent blocks.  Tokens remain but are invisible to attention.
        "hard": tokens below deletion_threshold are physically removed before
        the compressed-phase blocks and restored at restore_gate_layer.
    use_gumbel_noise (bool):
        Add Gumbel noise to gate logits during training for exploration.

    Deletion rate schedule
    ----------------------
    The guide notes that a fixed target deletion rate is wrong for diffusion:
    at high noise most tokens are corrupted/random and deletion is cheap; at low
    noise tokens carry critical signal.  The schedule maps σ → target_rate.

    deletion_rate_schedule (str):
        "constant"     — fixed target_deletion_rate regardless of σ.
        "linear_sigma" — rate scales linearly with σ: r(σ) = r_min + (r_max-r_min)·(σ/σ_max).
        "power_sigma"  — r(σ) = r_min + (r_max-r_min)·(σ/σ_max)^α, where α is
                         deletion_rate_alpha.  α<1 → concave (aggressive early),
                         α>1 → convex (conservative early).
    target_deletion_rate (float):
        Used only when deletion_rate_schedule="constant".
    r_min (float):
        Minimum deletion rate, applied at σ ≈ 0 (nearly clean tokens).
    r_max (float):
        Maximum deletion rate, applied at σ ≈ σ_max (fully noised).
    deletion_rate_alpha (float):
        Exponent for "power_sigma" schedule.  1.0 = linear.
    sigma_max (float):
        Maximum σ in the noise schedule (must match SEDD config; default 20.0).

    Loss weight
    -----------
    deletion_loss_weight (float):
        Weight of auxiliary deletion rate MSE loss relative to score entropy.

    Baselines
    ---------
    random_deletion_probability (float):
        Mean deletion probability for "random" gate type.
    fixed_deletion_amount (float):
        Fixed fraction to delete for "fixed" gate type.
    """

    # Gate placement
    delete_gate_layer: int = 3
    restore_gate_layer: Optional[int] = None

    # Gate variant
    deletion_type: str = "scaled_sigmoid"
    gate_type: str = "scaled_sigmoid"        # "scaled_sigmoid" | "bottleneck_mlp"
    gate_bottleneck_dim: int = 128
    gate_logit_reg_weight: float = 0.001    # L2 reg on gate logits (plan Section 3.4)
    sigmoid_mask_scale: float = -30.0
    deletion_threshold: float = -15.0
    gate_layer_norm: bool = True
    gate_sigma_conditioned: bool = True
    deletion_mode: str = "soft"
    use_gumbel_noise: bool = False

    # Deletion rate schedule
    deletion_rate_schedule: str = "constant"
    target_deletion_rate: float = 0.3     # used when schedule="constant"
    r_min: float = 0.05                   # rate at σ → 0
    r_max: float = 0.5                    # rate at σ → σ_max
    deletion_rate_alpha: float = 1.0      # schedule curvature exponent
    sigma_max: float = 20.0              # must match SEDD noise.sigma_max

    # Loss weight
    deletion_loss_weight: float = 0.1

    # Baselines
    random_deletion_probability: float = 0.5
    fixed_deletion_amount: float = 0.5

    def to_dict(self) -> dict:
        """Convert to plain dict for OmegaConf merging."""
        return {
            "delete_gate_layer": self.delete_gate_layer,
            "restore_gate_layer": self.restore_gate_layer,
            "deletion_type": self.deletion_type,
            "sigmoid_mask_scale": self.sigmoid_mask_scale,
            "deletion_threshold": self.deletion_threshold,
            "gate_layer_norm": self.gate_layer_norm,
            "gate_sigma_conditioned": self.gate_sigma_conditioned,
            "deletion_mode": self.deletion_mode,
            "use_gumbel_noise": self.use_gumbel_noise,
            "deletion_rate_schedule": self.deletion_rate_schedule,
            "target_deletion_rate": self.target_deletion_rate,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "deletion_rate_alpha": self.deletion_rate_alpha,
            "sigma_max": self.sigma_max,
            "deletion_loss_weight": self.deletion_loss_weight,
            "random_deletion_probability": self.random_deletion_probability,
            "fixed_deletion_amount": self.fixed_deletion_amount,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MrDiffusionConfig":
        """Construct from a plain dict (e.g. from OmegaConf.to_container)."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)