"""
MrBD3LM package.

Apply the MrT5/MrBERT delete gate mechanism to BD3-LM (Block Denoising
Discrete Diffusion Language Models).

Key insight: BD3-LM corrupts tokens to identical [MASK] tokens → perfectly
redundant representations that a gate can learn to collapse, preserving
computation for unmasked (informative) tokens.
"""

from .configuration_mrd_bd3lm import MrBD3LMConfig
from .modeling_mrd_bd3lm import (
    MrBD3LM,
    MrBD3LMOutput,
    MrDITBackbone,
    SigmoidDeleteGate,
    SigmoidDeleteGateWithSigma,
    RandomDeleteGate,
    FixedDeleteGate,
    BottleneckDeleteGate,
    PreDeletionBlend,
    TokenMerge,
    TokenRestore,
    deletion_rate_loss,
    target_rate_at_timestep,
)

__all__ = [
    "MrBD3LMConfig",
    "MrBD3LM",
    "MrBD3LMOutput",
    "MrDITBackbone",
    "SigmoidDeleteGate",
    "SigmoidDeleteGateWithSigma",
    "RandomDeleteGate",
    "FixedDeleteGate",
    "BottleneckDeleteGate",
    "PreDeletionBlend",
    "TokenMerge",
    "TokenRestore",
    "deletion_rate_loss",
    "target_rate_at_timestep",
]