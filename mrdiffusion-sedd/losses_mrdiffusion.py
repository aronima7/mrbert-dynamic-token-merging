"""
Loss functions for MrDiffusion training.

Extends the base SEDD score-entropy loss with an auxiliary deletion rate loss.
The key improvement over a fixed-rate target is a *sigma-dependent deletion
schedule* r(σ): at high noise levels most tokens are corrupted and carry little
signal, so aggressive deletion is cheap; at low noise levels tokens carry
critical information, so the gate should preserve more.
"""

import sys as _sys
import os as _os

_MRDIFF_DIR = _os.path.dirname(_os.path.abspath(__file__))
_SEDD_ROOT = _os.path.join(_os.path.dirname(_MRDIFF_DIR), "diffusion", "Score-Entropy-Discrete-Diffusion")
if _SEDD_ROOT not in _sys.path:
    _sys.path.insert(0, _SEDD_ROOT)

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import graph_lib
from model import utils as mutils

from configuration_mrdiffusion import MrDiffusionConfig


# ---------------------------------------------------------------------------
# Sigma-dependent deletion rate schedule
# ---------------------------------------------------------------------------

def target_rate_at_sigma(
    sigma: torch.Tensor,
    mr_config: MrDiffusionConfig,
) -> torch.Tensor:
    """
    Compute the per-sample target deletion rate as a function of noise level σ.

    Schedules
    ---------
    "constant"
        Returns mr_config.target_deletion_rate for all σ.
    "linear_sigma"
        r(σ) = r_min + (r_max − r_min) · clamp(σ / σ_max, 0, 1)
        Linear ramp from r_min (σ≈0, nearly clean) to r_max (σ≈σ_max, fully noised).
    "power_sigma"
        r(σ) = r_min + (r_max − r_min) · clamp(σ / σ_max, 0, 1)^α
        Convex (α>1) or concave (α<1) version of linear_sigma.

    Returns: [B] tensor of target rates, or a Python float for "constant".
    """
    schedule = mr_config.deletion_rate_schedule
    if schedule == "constant":
        return mr_config.target_deletion_rate  # scalar, broadcast later

    frac = (sigma / mr_config.sigma_max).clamp(0.0, 1.0)  # [B]
    if schedule == "power_sigma":
        frac = frac ** mr_config.deletion_rate_alpha
    # linear_sigma is power_sigma with alpha=1.0
    return mr_config.r_min + (mr_config.r_max - mr_config.r_min) * frac  # [B]


# ---------------------------------------------------------------------------
# Deletion rate loss
# ---------------------------------------------------------------------------

def deletion_rate_loss(
    gate_output: torch.Tensor,
    target_rates,
    deletion_threshold: float,
    sigmoid_mask_scale: float = -30.0,
) -> torch.Tensor:
    """
    Per-sample MSE loss encouraging the deletion fraction to match target_rates.

    gate_output:  [B, L, 1]  gate values (in [sigmoid_mask_scale, 0])
    target_rates: [B] tensor of per-sample targets, or a scalar float.
    deletion_threshold: gate value at/below which a token is "deleted" (unused in proxy).
    sigmoid_mask_scale: the minimum gate value (default -30.0).

    Uses a linear differentiable proxy for the deletion fraction:
        soft_delete_i = gate_output_i / sigmoid_mask_scale
    which is 0 when gate_output=0 (no deletion) and 1 when gate_output=sigmoid_mask_scale
    (full deletion).  Unlike the previous sigmoid proxy whose range was [0.5, 0.88],
    this proxy covers [0, 1] so targets like 0.3 are reachable and the gradient
    correctly pushes toward more deletion when actual_rate < target.
    """
    soft_delete = gate_output / (sigmoid_mask_scale - 1e-8)  # [0, 1]
    actual_rates = soft_delete.mean(dim=1).squeeze(-1)  # [B]

    if isinstance(target_rates, torch.Tensor):
        targets = target_rates.to(actual_rates.dtype).to(actual_rates.device)
    else:
        targets = torch.full_like(actual_rates, float(target_rates))

    return F.mse_loss(actual_rates, targets)


# ---------------------------------------------------------------------------
# Bimodality loss — penalizes gate logits in the undecided zone
# ---------------------------------------------------------------------------

def bimodality_loss(
    gate_logits: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """
    Penalizes gate logits that fall in the undecided zone [-margin, margin].

    Encourages a clean bimodal split: logits should be clearly positive (delete)
    or clearly negative (keep), not hovering near 0 where sigmoid is ~0.5.

    Uses a soft hinge: penalty = max(0, margin - |logit|)^2, averaged over all tokens.
    """
    abs_logits = gate_logits.abs()
    penalty = F.relu(margin - abs_logits).pow(2)
    return penalty.mean()


# ---------------------------------------------------------------------------
# Combined loss function
# ---------------------------------------------------------------------------

def get_loss_fn(
    noise,
    graph,
    train: bool,
    mr_config: MrDiffusionConfig,
    sampling_eps: float = 1e-3,
):
    """
    Returns a loss_fn(model, batch) → per-sample losses [B].

    Score entropy loss (unchanged from SEDD) plus an auxiliary deletion rate
    loss whose per-sample target is determined by the sigma-dependent schedule.
    """

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        if t is None:
            t = (1 - sampling_eps) * torch.rand(
                batch.shape[0], device=batch.device
            ) + sampling_eps

        sigma, dsigma = noise(t)  # [B], [B]

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        output = model(perturbed_batch, sigma)

        # Support MrSEDD (MrSEDDOutput) and vanilla SEDD (raw tensor)
        if hasattr(output, "logits"):
            log_score = output.logits
            gate_output = output.delete_gate_output      # [B, L, 1] or None
            gate_logits_out = output.delete_gate_logits  # [B, L, 1] or None
        else:
            log_score = output
            gate_output = None
            gate_logits_out = None

        # Score entropy loss [B]
        score_loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        score_loss = (dsigma[:, None] * score_loss).sum(dim=-1)  # [B]

        total_loss = score_loss

        # Auxiliary deletion rate loss (sigma-dependent target)
        if gate_output is not None and mr_config.deletion_loss_weight > 0.0:
            target_rates = target_rate_at_sigma(sigma, mr_config)
            del_loss = deletion_rate_loss(
                gate_output, target_rates, mr_config.deletion_threshold,
                mr_config.sigmoid_mask_scale,
            )
            total_loss = total_loss + mr_config.deletion_loss_weight * del_loss

        # Gate logit regularization (plan Section 3.4: λ=0.001)
        gate_logit_reg_weight = getattr(mr_config, "gate_logit_reg_weight", 0.001)
        if gate_logits_out is not None and gate_logit_reg_weight > 0.0:
            logit_reg = gate_logit_reg_weight * gate_logits_out.pow(2).mean()
            total_loss = total_loss + logit_reg

        # Bimodality loss: penalizes logits in the undecided zone [-2, 2]
        gate_bimodality_weight = getattr(mr_config, "gate_bimodality_weight", 0.0)
        if gate_logits_out is not None and gate_bimodality_weight > 0.0:
            bimod_loss = gate_bimodality_weight * bimodality_loss(gate_logits_out)
            total_loss = total_loss + bimod_loss

        return total_loss

    return loss_fn


# ---------------------------------------------------------------------------
# Loss component breakdown (for logging only, not used during training steps)
# ---------------------------------------------------------------------------

def compute_loss_components(
    model,
    batch: torch.Tensor,
    noise,
    graph,
    mr_config: MrDiffusionConfig,
    sampling_eps: float = 1e-3,
) -> dict:
    """
    Run a single forward pass and return each loss term separately.
    Called at logging steps only — never during the training gradient update.

    Returns dict with keys:
        score_entropy_loss, gate_loss, logit_reg_loss, total_loss  (all Python floats)
    """
    t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
    sigma, dsigma = noise(t)
    perturbed_batch = graph.sample_transition(batch, sigma[:, None])

    with torch.no_grad():
        output = model(perturbed_batch, sigma)

    if hasattr(output, "logits"):
        log_score = output.logits
        gate_output = output.delete_gate_output
        gate_logits_out = output.delete_gate_logits
    else:
        log_score = output
        gate_output = None
        gate_logits_out = None

    score_loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
    score_entropy = (dsigma[:, None] * score_loss).sum(dim=-1).mean().item()

    gate_loss_val = 0.0
    if gate_output is not None and mr_config.deletion_loss_weight > 0.0:
        target_rates = target_rate_at_sigma(sigma, mr_config)
        gate_loss_val = deletion_rate_loss(
            gate_output, target_rates, mr_config.deletion_threshold,
            mr_config.sigmoid_mask_scale,
        ).item()

    logit_reg_val = 0.0
    gate_logit_reg_weight = getattr(mr_config, "gate_logit_reg_weight", 0.001)
    if gate_logits_out is not None and gate_logit_reg_weight > 0.0:
        logit_reg_val = (gate_logit_reg_weight * gate_logits_out.pow(2).mean()).item()

    bimodality_val = 0.0
    gate_bimodality_weight = getattr(mr_config, "gate_bimodality_weight", 0.0)
    if gate_logits_out is not None and gate_bimodality_weight > 0.0:
        bimodality_val = (gate_bimodality_weight * bimodality_loss(gate_logits_out)).item()

    total = score_entropy + mr_config.deletion_loss_weight * gate_loss_val + logit_reg_val + bimodality_val

    return {
        "score_entropy_loss": score_entropy,
        "gate_loss": gate_loss_val,
        "logit_reg_loss": logit_reg_val,
        "bimodality_loss": bimodality_val,
        "total_loss": total,
    }


# ---------------------------------------------------------------------------

def get_optimizer(config, params):
    if config.optim.optimizer == "Adam":
        return optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    elif config.optim.optimizer == "AdamW":
        return optim.AdamW(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not supported.")


def optimization_manager(config):
    """Returns an optimize_fn based on config."""

    def optimize_fn(
        optimizer,
        scaler,
        params,
        step,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
    ):
        scaler.unscale_(optimizer)
        if warmup > 0:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum, mr_config: MrDiffusionConfig):
    """
    Returns a step_fn(state, batch) for one (possibly accumulated) update.
    Mirrors SEDD's get_step_fn but uses the MrDiffusion-aware loss.
    """
    loss_fn = get_loss_fn(noise, graph, train, mr_config)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter, total_loss

        model = state["model"]

        if train:
            optimizer = state["optimizer"]
            scaler = state["scaler"]
            loss = loss_fn(model, batch, cond=cond).mean() / accum

            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()

            if accum_iter == accum:
                accum_iter = 0
                state["step"] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state["step"])
                state["ema"].update(model.parameters())
                optimizer.zero_grad()
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state["ema"]
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn