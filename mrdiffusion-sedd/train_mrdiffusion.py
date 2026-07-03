"""
Training script for MrDiffusion (MrSEDD).

Usage:
    # Quick smoke test (200 steps, single GPU)
    python train_mrdiffusion.py --max_steps 200 --logging_steps 20

    # Full run with soft deletion (default)
    python train_mrdiffusion.py \\
        --sedd_config ../diffusion/Score-Entropy-Discrete-Diffusion/configs/config.yaml \\
        --output_dir ./mrdiffusion_checkpoints \\
        --delete_gate_layer 3 \\
        --deletion_type scaled_sigmoid \\
        --deletion_mode soft \\
        --target_deletion_rate 0.3 \\
        --deletion_loss_weight 0.1

    # Hard deletion with sigma-conditioned gate
    python train_mrdiffusion.py \\
        --deletion_mode hard \\
        --gate_sigma_conditioned \\
        --delete_gate_layer 3

    # Compare against baseline SEDD (no gate)
    python train_mrdiffusion.py --no_delete_gate

    # W&B run with custom project/name
    python train_mrdiffusion.py \\
        --wandb_project mrdiffusion-sedd \\
        --wandb_run_name soft-gate-layer3-30pct

This script is self-contained and does NOT require Hydra.  It loads the SEDD
OmegaConf config from YAML, merges in MrDiffusion gate parameters, and uses
the MrDiffusion-aware loss function.
"""

import argparse
import os
import sys
import logging
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Path setup — add SEDD root so its modules are importable
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SEDD_ROOT = os.path.join(_PROJECT_ROOT, "diffusion", "Score-Entropy-Discrete-Diffusion")

if _SEDD_ROOT not in sys.path:
    sys.path.insert(0, _SEDD_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# SEDD modules
import data as sedd_data
import graph_lib
import noise_lib
import sampling as sedd_sampling
import utils as sedd_utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# MrDiffusion modules
from configuration_mrdiffusion import MrDiffusionConfig
from modeling_mrdiffusion import MrSEDD
import losses_mrdiffusion as mr_losses


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MrSEDD (SEDD + delete gate)")

    # --- SEDD base config ---
    p.add_argument(
        "--sedd_config",
        default=os.path.join(_SEDD_ROOT, "configs", "config.yaml"),
        help="Path to SEDD OmegaConf config YAML.",
    )
    p.add_argument(
        "--model_size",
        default="small",
        choices=["small", "medium"],
        help="Model size overlay (small.yaml or medium.yaml).",
    )

    # --- Training ---
    p.add_argument("--output_dir", default="./mrdiffusion_checkpoints")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override n_iters from config for quick tests.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch_size from config.")
    p.add_argument("--eval_batch_size", type=int, default=None,
                   help="Override eval.batch_size from config. Default 512 OOMs at seq_len=1024.")
    p.add_argument("--data_cache_dir", type=str, default=None,
                   help="Override data.cache_dir. Set to a persistent path to avoid re-downloading datasets.")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="e.g. 'cuda:0' or 'cpu'. Defaults to cuda if available.")

    # --- Delete gate ---
    p.add_argument("--no_delete_gate", action="store_true",
                   help="Train vanilla SEDD (baseline, no gate).")
    p.add_argument("--gate_type", default="scaled_sigmoid",
                   choices=["scaled_sigmoid", "bottleneck_mlp"],
                   help="Gate architecture: scaled_sigmoid (current) or bottleneck_mlp (plan)")
    p.add_argument("--gate_bottleneck_dim", type=int, default=128)
    p.add_argument("--gate_logit_reg_weight", type=float, default=0.001,
                   help="L2 regularization on gate logits (plan Section 3.4)")
    p.add_argument("--gate_init_bias", type=float, default=0.0,
                   help="Gate linear layer bias initialization. 0=balanced start (recommended), "
                        "positive=biased toward keep. Previous default was 2.0.")
    p.add_argument("--gate_bimodality_weight", type=float, default=0.01,
                   help="Weight for bimodality loss that penalizes gate logits in undecided zone [-2,2]. "
                        "Encourages clean keep/delete separation.")
    p.add_argument("--delete_gate_layer", type=int, default=3,
                   help="DDiTBlock index after which the gate fires.")
    p.add_argument("--restore_gate_layer", type=int, default=-1,
                   help="DDiTBlock index at which to restore full sequence length. "
                        "-1 (default) = restore just before output layer.")
    p.add_argument("--deletion_type", default="scaled_sigmoid",
                   choices=["scaled_sigmoid", "log_sigmoid", "random", "fixed"])
    p.add_argument("--deletion_mode", default="soft", choices=["soft", "hard"])
    p.add_argument("--sigmoid_mask_scale", type=float, default=-30.0)
    p.add_argument("--deletion_threshold", type=float, default=-15.0)
    p.add_argument("--gate_layer_norm", action="store_true", default=True)
    p.add_argument("--no_gate_layer_norm", dest="gate_layer_norm", action="store_false")
    p.add_argument("--gate_sigma_conditioned", action="store_true", default=True,
                   help="Condition gate on sigma (noise level). Default True.")
    p.add_argument("--no_gate_sigma_conditioned", dest="gate_sigma_conditioned",
                   action="store_false",
                   help="Disable sigma conditioning on the gate.")
    p.add_argument("--use_gumbel_noise", action="store_true", default=False)
    p.add_argument("--stop_gate_grad", action="store_true", default=False,
                   help="Detach gate output from score-entropy gradient; gate is trained only by gate loss.")

    # --- Deletion rate schedule ---
    p.add_argument("--deletion_rate_schedule", default="linear_sigma",
                   choices=["constant", "linear_sigma", "power_sigma"],
                   help="How the target deletion rate varies with noise level σ. "
                        "'constant' uses --target_deletion_rate for all σ. "
                        "'linear_sigma'/'power_sigma' ramp from --r_min (σ≈0) "
                        "to --r_max (σ≈σ_max).")
    p.add_argument("--target_deletion_rate", type=float, default=0.3,
                   help="Target deletion rate for 'constant' schedule.")
    p.add_argument("--r_min", type=float, default=0.05,
                   help="Minimum deletion rate at σ≈0 for non-constant schedules.")
    p.add_argument("--r_max", type=float, default=0.5,
                   help="Maximum deletion rate at σ≈σ_max for non-constant schedules.")
    p.add_argument("--deletion_rate_alpha", type=float, default=1.0,
                   help="Exponent for 'power_sigma' schedule. 1.0=linear, "
                        ">1 convex (conservative at low noise), <1 concave.")
    p.add_argument("--sigma_max", type=float, default=20.0,
                   help="σ_max from SEDD noise schedule (default 20.0).")
    p.add_argument("--deletion_loss_weight", type=float, default=0.1)
    p.add_argument("--random_deletion_probability", type=float, default=0.5)
    p.add_argument("--fixed_deletion_amount", type=float, default=0.5)

    # --- Gate LR (separate LR for gate params, like MrBERT) ---
    p.add_argument("--delete_gate_lr", type=float, default=None,
                   help="Optional higher LR for delete gate parameters only.")

    # --- Delete gate training phases (plan Section 3.2) ---
    p.add_argument("--pretrained_from", type=str, default=None,
                   help="HuggingFace model ID to warm-start from, e.g. 'louaaron/sedd-small'. "
                        "Loads pretrained SEDD weights into MrSEDD (strict=False) so the "
                        "transformer blocks start from pretrained weights and only the gate "
                        "is trained from random init.")
    p.add_argument("--from_pretrained_checkpoint", type=str, default=None,
                   help="Phase 2: warm-start from a Phase 1 baseline checkpoint")
    p.add_argument("--gate_warmup_steps", type=int, default=5_000,
                   help="Ramp deletion_loss_weight 0→target over this many steps")
    p.add_argument("--freeze_transformer_steps", type=int, default=2_000,
                   help="Freeze transformer (non-gate) params for first N steps")
    p.add_argument("--rope_original_positions", action="store_true", default=False,
                   help="Hard deletion: apply RoPE at original token positions rather "
                        "than re-indexing survivors to [0..L_kept-1]. Requires SDPA "
                        "fallback for compressed-phase blocks (no FlashAttn).")
    p.add_argument("--gumbel_temp_start", type=float, default=2.0,
                   help="Initial Gumbel-sigmoid temperature (plan: 2.0)")
    p.add_argument("--gumbel_temp_end", type=float, default=0.5,
                   help="Final Gumbel-sigmoid temperature (plan: 0.5)")
    p.add_argument("--gumbel_temp_steps", type=int, default=50_000,
                   help="Anneal temperature over this many steps (plan: 50K)")
    p.add_argument("--use_pi_controller", action="store_true", default=False,
                   help="Use PI controller to track target deletion rate")
    p.add_argument("--pi_min_weight", type=float, default=0.1,
                   help="Floor for PI controller output — prevents deletion_loss_weight from "
                        "collapsing to 0 which kills gate learning signal.")

    # --- W&B ---
    p.add_argument("--wandb_project", default="mrdiffusion-sedd",
                   help="W&B project name.")
    p.add_argument("--wandb_entity", default="aronima7-stanford-university",
                   help="W&B entity (team or username).")
    p.add_argument("--wandb_run_name", default="",
                   help="W&B run name. Auto-generated if empty.")
    p.add_argument("--disable_wandb", action="store_true",
                   help="Disable W&B logging entirely.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(work_dir: str) -> logging.Logger:
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "train.log")
    logger = logging.getLogger("mrdiffusion-sedd")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Deletion rate helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_deletion_rate(gate_output, deletion_threshold: float, sigmoid_mask_scale: float = -30.0) -> float:
    """Fraction of tokens being deleted by the gate (linear proxy, range [0, 1]).

    Uses the same proxy as deletion_rate_loss: gate_output / sigmoid_mask_scale.
    Maps gate_output=0 (no deletion) → 0 and gate_output=sigmoid_mask_scale → 1.
    """
    if gate_output is None:
        return 0.0
    return (gate_output / sigmoid_mask_scale).squeeze(-1).mean().item()


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def init_wandb(args, cfg, mr_config):
    """Initialise W&B run. Returns the run object (or None if disabled)."""
    if args.disable_wandb:
        return None
    try:
        import wandb
    except ImportError:
        return None

    model_tag = "sedd-baseline" if args.no_delete_gate else "mrsedd"
    run_name = args.wandb_run_name or (
        f"{model_tag}-{args.deletion_type}-layer{args.delete_gate_layer}"
        f"-{args.deletion_mode}-{int(args.target_deletion_rate * 100)}pct"
    )

    wandb_config = {
        # model
        "model_size": args.model_size,
        "n_blocks": cfg.model.n_blocks,
        "hidden_size": cfg.model.hidden_size,
        "n_heads": cfg.model.n_heads,
        "seq_length": cfg.model.length,
        # training
        "batch_size": cfg.training.batch_size,
        "lr": cfg.optim.lr,
        "warmup": cfg.optim.warmup,
        "n_iters": cfg.training.n_iters,
        "ema": cfg.training.ema,
        # gate
        "no_delete_gate": args.no_delete_gate,
        **mr_config.to_dict(),
        "delete_gate_lr": args.delete_gate_lr,
        # noise / graph
        "noise_type": cfg.noise.type,
        "graph_type": cfg.graph.type,
    }

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        config=wandb_config,
        resume="allow",
    )
    return run


def wandb_log(metrics: dict, step: int):
    """Log metrics to W&B if the library is available and a run is active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def wandb_finish():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


def _make_pi_controller(target_rate: float, initial_weight: float = 0.0, min_weight: float = 0.1):
    """Create an inline PI controller (plan Section 2.5).

    initial_weight: seed p_acc so the first output matches the warmup-end weight,
    avoiding the cliff where the controller resets to near-zero on handoff.
    min_weight: floor for the output — prevents the controller from driving
    deletion_loss_weight to 0, which kills gate learning signal.
    ki=1e-3 (was 1e-5) so the integral accumulates meaningfully within 20k steps.
    """
    class _PIController:
        def __init__(self, target_rate, kp=0.5, ki=1e-3, gamma=0.9):
            self.target_rate = target_rate
            self.kp = kp; self.ki = ki; self.gamma = gamma
            self.p_acc = initial_weight; self.i_acc = 0.0
            self.min_weight = min_weight
        def update(self, actual_rate):
            error = self.target_rate - actual_rate
            self.p_acc = self.gamma * self.p_acc + (1-self.gamma) * self.kp * error
            self.i_acc += self.ki * error
            return max(self.min_weight, self.p_acc + self.i_acc)
    return _PIController(target_rate)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Output directories
    work_dir = args.output_dir
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_path = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_path), exist_ok=True)

    logger = get_logger(work_dir)

    # Load SEDD config
    base_cfg = OmegaConf.load(args.sedd_config)
    model_cfg_path = os.path.join(
        os.path.dirname(args.sedd_config), "model", f"{args.model_size}.yaml"
    )
    model_cfg = OmegaConf.load(model_cfg_path)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create({"model": model_cfg}))

    # Override training params from CLI
    if args.max_steps is not None:
        cfg.training.n_iters = args.max_steps
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        cfg.eval.batch_size = args.eval_batch_size
    if args.data_cache_dir is not None:
        cfg.data.cache_dir = args.data_cache_dir
    cfg.training.log_freq = args.logging_steps
    cfg.training.eval_freq = args.eval_steps
    cfg.training.snapshot_freq_for_preemption = args.save_steps
    cfg.training.snapshot_freq = args.save_steps * 5
    cfg.ngpus = 1
    cfg.work_dir = work_dir

    logger.info(f"SEDD config:\n{OmegaConf.to_yaml(cfg)}")

    # Build MrDiffusion config
    mr_config = MrDiffusionConfig(
        delete_gate_layer=args.delete_gate_layer,
        restore_gate_layer=None if args.restore_gate_layer == -1 else args.restore_gate_layer,
        deletion_type=args.deletion_type,
        gate_type=args.gate_type,
        gate_bottleneck_dim=args.gate_bottleneck_dim,
        gate_logit_reg_weight=args.gate_logit_reg_weight,
        gate_init_bias=args.gate_init_bias,
        gate_bimodality_weight=args.gate_bimodality_weight,
        sigmoid_mask_scale=args.sigmoid_mask_scale,
        deletion_threshold=args.deletion_threshold,
        gate_layer_norm=args.gate_layer_norm,
        gate_sigma_conditioned=args.gate_sigma_conditioned,
        deletion_mode=args.deletion_mode,
        use_gumbel_noise=args.use_gumbel_noise,
        stop_gate_grad=args.stop_gate_grad,
        deletion_rate_schedule=args.deletion_rate_schedule,
        target_deletion_rate=args.target_deletion_rate,
        r_min=args.r_min,
        r_max=args.r_max,
        deletion_rate_alpha=args.deletion_rate_alpha,
        sigma_max=args.sigma_max,
        deletion_loss_weight=args.deletion_loss_weight,
        pi_min_weight=args.pi_min_weight,
        rope_original_positions=args.rope_original_positions,
        random_deletion_probability=args.random_deletion_probability,
        fixed_deletion_amount=args.fixed_deletion_amount,
    )
    logger.info(f"MrDiffusion config: {mr_config}")

    # W&B
    init_wandb(args, cfg, mr_config)

    # Graph + noise
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    sampling_eps = 1e-5

    # Build model
    if args.no_delete_gate:
        logger.info("Building baseline SEDD (no delete gate).")
        score_model = SEDD(cfg).to(device)
        mr_config_for_loss = MrDiffusionConfig(deletion_loss_weight=0.0)
    else:
        logger.info("Building MrSEDD with delete gate.")
        score_model = MrSEDD(cfg, mr_config).to(device)
        mr_config_for_loss = mr_config

    n_params = sum(p.numel() for p in score_model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    wandb_log({"model/n_parameters": n_params}, step=0)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    # Optimizer — optionally use higher LR for gate params
    if args.delete_gate_lr is not None and not args.no_delete_gate:
        gate_params = list(score_model.delete_gate.parameters())
        gate_ids = {id(p) for p in gate_params}
        base_params = [p for p in score_model.parameters() if id(p) not in gate_ids]
        param_groups = [
            {"params": base_params, "lr": cfg.optim.lr},
            {"params": gate_params, "lr": args.delete_gate_lr},
        ]
        optimizer = mr_losses.get_optimizer(cfg, param_groups)
    else:
        optimizer = mr_losses.get_optimizer(
            cfg, chain(score_model.parameters(), noise.parameters())
        )

    scaler = torch.cuda.amp.GradScaler()
    state = dict(
        optimizer=optimizer,
        scaler=scaler,
        model=score_model,
        noise=noise,
        ema=ema,
        step=0,
    )

    # Restore from checkpoint if exists
    state = sedd_utils.restore_checkpoint(checkpoint_meta_path, state, device)
    initial_step = int(state["step"])

    # Phase 2: warm-start from pretrained baseline
    if args.from_pretrained_checkpoint and initial_step == 0:
        ckpt = torch.load(args.from_pretrained_checkpoint, map_location=device)
        if "model" in ckpt:
            missing, unexpected = score_model.load_state_dict(ckpt["model"], strict=False)
        else:
            missing, unexpected = score_model.load_state_dict(ckpt, strict=False)
        logger.info(f"[warmstart] loaded from {args.from_pretrained_checkpoint} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # Continued pretraining: load from HuggingFace pretrained SEDD weights
    if args.pretrained_from and initial_step == 0:
        logger.info(f"[pretrained_from] loading {args.pretrained_from} from HuggingFace ...")
        pretrained_sedd = SEDD.from_pretrained(args.pretrained_from)
        missing, unexpected = score_model.load_state_dict(
            pretrained_sedd.state_dict(), strict=False
        )
        del pretrained_sedd
        logger.info(
            f"[pretrained_from] loaded. missing (gate params): {len(missing)}, "
            f"unexpected: {len(unexpected)}"
        )
        if missing:
            logger.info(f"  missing keys (new gate params, will train from init): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.info(f"  unexpected keys (ignored): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    logger.info(f"Starting from step {initial_step}.")

    # Freeze transformer if at start of phase 2
    if args.freeze_transformer_steps > 0 and initial_step < args.freeze_transformer_steps:
        if not args.no_delete_gate:
            for name, p in score_model.named_parameters():
                if "delete_gate" not in name:
                    p.requires_grad_(False)
            logger.info(f"Transformer frozen for first {args.freeze_transformer_steps} steps")
            # Re-initialize EMA to only shadow currently-trainable params.
            # EMA.__init__ and update() both filter by requires_grad; if shadow_params
            # was built before freezing (all params) but update() only sees gate params,
            # the zip misaligns and causes a shape mismatch.
            ema = ExponentialMovingAverage(
                [p for p in score_model.parameters() if p.requires_grad],
                decay=cfg.training.ema,
            )
            state["ema"] = ema

    # Data
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_ds, eval_ds = sedd_data.get_dataloaders(cfg, distributed=False)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # PI controller for deletion rate targeting (plan Section 2.5)
    pi_controller = None
    if args.use_pi_controller and not args.no_delete_gate:
        pi_controller = _make_pi_controller(
            mr_config.target_deletion_rate,
            initial_weight=args.deletion_loss_weight,
            min_weight=args.pi_min_weight,
        )

    # Sampling function for generation evals
    sampling_shape = (min(16, cfg.training.batch_size), cfg.model.length)
    sampling_fn = sedd_sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    # Step functions
    optimize_fn = mr_losses.optimization_manager(cfg)
    train_step_fn = mr_losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum, mr_config_for_loss)
    eval_step_fn = mr_losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum, mr_config_for_loss)

    num_train_steps = cfg.training.n_iters
    logger.info(f"Training for {num_train_steps} steps.")

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    while state["step"] < num_train_steps + 1:
        step = state["step"]

        # ── Unfreeze transformer at freeze_transformer_steps ──────────────────
        if (args.freeze_transformer_steps > 0
                and step == args.freeze_transformer_steps
                and not args.no_delete_gate):
            for p in score_model.parameters():
                p.requires_grad_(True)
            logger.info(f"Unfreezing transformer at step {step}")
            # Re-initialize EMA to track all params again now that the full model
            # is trainable.  The frozen-phase EMA only shadowed gate params, so we
            # must rebuild shadow_params to cover the newly-unfrozen transformer.
            ema = ExponentialMovingAverage(
                score_model.parameters(), decay=cfg.training.ema
            )
            state["ema"] = ema

        # ── Gate warmup: ramp deletion_loss_weight ─────────────────────────────
        if args.gate_warmup_steps > 0 and step <= args.gate_warmup_steps:
            warmup_frac = min(1.0, step / max(1, args.gate_warmup_steps))
            mr_config_for_loss.deletion_loss_weight = (
                warmup_frac * args.deletion_loss_weight)

        # ── Gumbel temperature annealing (2.0 → 0.5 over gumbel_temp_steps) ───
        if args.gumbel_temp_steps > 0:
            temp_frac = min(1.0, step / max(1, args.gumbel_temp_steps))
            gate_temp = args.gumbel_temp_start + temp_frac * (
                args.gumbel_temp_end - args.gumbel_temp_start)
        else:
            gate_temp = 1.0
        # Propagate temperature to BottleneckDeleteGate if used
        if not args.no_delete_gate and hasattr(score_model, "delete_gate"):
            if hasattr(score_model.delete_gate, "_temperature"):
                score_model.delete_gate._temperature = gate_temp

        if cfg.data.train != "text8":
            batch = next(train_iter)["input_ids"].to(device)
        else:
            batch = next(train_iter).to(device)

        loss = train_step_fn(state, batch)

        if step != state["step"]:
            new_step = state["step"]

            # ------ Logging (train loss + deletion rate) ------
            if new_step % cfg.training.log_freq == 0:
                deletion_rate = 0.0
                gate_mean = 0.0
                gate_std = 0.0
                score_entropy_loss = loss.item()
                gate_loss_val = 0.0
                logit_reg_loss = 0.0
                bimodality_loss_val = 0.0
                time_per_forward_ms = 0.0
                avg_sequence_length = float(cfg.model.length)

                if not args.no_delete_gate:
                    with torch.no_grad():
                        score_model.eval()
                        t_dummy = torch.full((1,), 0.5, device=device)
                        sigma_dummy, _ = noise(t_dummy)
                        perturbed = graph.sample_transition(batch[:1], sigma_dummy[:, None])

                        # Time the forward pass
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                        import time as _time
                        if t0 is not None:
                            t0.record()
                        else:
                            _t_start = _time.perf_counter()

                        out = score_model(perturbed, sigma_dummy)

                        if t1 is not None:
                            t1.record()
                            torch.cuda.synchronize()
                            time_per_forward_ms = t0.elapsed_time(t1)
                        else:
                            time_per_forward_ms = (_time.perf_counter() - _t_start) * 1000.0

                        score_model.train()

                    if hasattr(out, "delete_gate_output") and out.delete_gate_output is not None:
                        gate_out = out.delete_gate_output  # [1, L, 1]
                        deletion_rate = compute_deletion_rate(
                            gate_out,
                            mr_config.deletion_threshold,
                            mr_config.sigmoid_mask_scale,
                        )
                        gate_mean = gate_out.mean().item()
                        gate_std = gate_out.std().item()
                        avg_sequence_length = cfg.model.length * (1.0 - deletion_rate)

                    # Loss component breakdown (separate score entropy vs gate loss)
                    components = mr_losses.compute_loss_components(
                        score_model, batch[:1], noise, graph, mr_config_for_loss
                    )
                    score_entropy_loss = components["score_entropy_loss"]
                    gate_loss_val = components["gate_loss"]
                    logit_reg_loss = components["logit_reg_loss"]
                    bimodality_loss_val = components.get("bimodality_loss", 0.0)

                # Theoretical attention speedup: compressed-phase blocks see (1-r)²
                # fraction of attention cost; full-sequence blocks are unchanged.
                n_blocks = len(score_model.blocks) if hasattr(score_model, "blocks") else 12
                gate_layer = mr_config.delete_gate_layer if not args.no_delete_gate else 0
                compressed_blocks = n_blocks - gate_layer - 1
                kept_ratio = 1.0 - deletion_rate
                full_cost = n_blocks
                mr_cost = (gate_layer + 1) + compressed_blocks * (kept_ratio ** 2)
                speedup_vs_baseline = full_cost / mr_cost if mr_cost > 0 else 1.0

                # PI controller update
                if (pi_controller is not None
                        and new_step > args.gate_warmup_steps
                        and deletion_rate > 0.0):
                    mr_config_for_loss.deletion_loss_weight = pi_controller.update(deletion_rate)

                # Compute current LR (after warmup schedule)
                current_lr = optimizer.param_groups[0]["lr"]

                logger.info(
                    f"step: {new_step}, train_loss: {loss.item():.5e}"
                    + (f", deletion_rate: {deletion_rate:.3f}" if not args.no_delete_gate else "")
                    + (f", del_loss_w: {mr_config_for_loss.deletion_loss_weight:.4f}" if pi_controller is not None else "")
                    + f", lr: {current_lr:.2e}"
                )

                metrics = {
                    "train/loss": loss.item(),
                    "train/score_entropy_loss": score_entropy_loss,
                    "train/learning_rate": current_lr,
                }
                if not args.no_delete_gate:
                    metrics.update({
                        "train/gate_loss": gate_loss_val,
                        "train/logit_reg_loss": logit_reg_loss,
                        "train/bimodality_loss": bimodality_loss_val,
                        "train/deletion_rate": deletion_rate,
                        "train/gate_mean": gate_mean,
                        "train/gate_std": gate_std,
                        "efficiency/time_per_forward_ms": time_per_forward_ms,
                        "efficiency/avg_sequence_length": avg_sequence_length,
                        "efficiency/speedup_vs_baseline": speedup_vs_baseline,
                    })
                if pi_controller is not None:
                    metrics["train/deletion_loss_weight"] = mr_config_for_loss.deletion_loss_weight
                wandb_log(metrics, step=new_step)

            # ------ Meta checkpoint ------
            if new_step % cfg.training.snapshot_freq_for_preemption == 0:
                sedd_utils.save_checkpoint(checkpoint_meta_path, state)
                logger.info(f"Saved meta checkpoint at step {new_step}.")

            # ------ Eval loss ------
            if new_step % cfg.training.eval_freq == 0:
                if cfg.data.valid != "text8":
                    eval_batch = next(eval_iter)["input_ids"].to(device)
                else:
                    eval_batch = next(train_iter).to(device)
                eval_loss = eval_step_fn(state, eval_batch)
                logger.info(f"step: {new_step}, eval_loss: {eval_loss.item():.5e}")
                wandb_log({"eval/loss": eval_loss.item()}, step=new_step)

            # ------ Snapshot: save + sample + perplexity ------
            if new_step > 0 and (
                new_step % cfg.training.snapshot_freq == 0 or new_step == num_train_steps
            ):
                save_idx = new_step // cfg.training.snapshot_freq
                sedd_utils.save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_{save_idx}.pth"), state
                )
                logger.info(f"Saved snapshot checkpoint_{save_idx}.pth")

                if cfg.training.snapshot_sampling:
                    logger.info(f"Generating samples at step {new_step}...")
                    this_sample_dir = os.path.join(sample_dir, f"iter_{new_step}")
                    os.makedirs(this_sample_dir, exist_ok=True)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())
                    score_model.train()  # sampling sets eval(); restore train mode before next step

                    sentences = tokenizer.batch_decode(sample)
                    sample_file = os.path.join(this_sample_dir, "samples.txt")
                    with open(sample_file, "w") as f:
                        for s in sentences:
                            f.write(s + "\n")
                            f.write("=" * 80 + "\n")
                    logger.info(f"Saved samples to {sample_file}")

                    # Log sample text to W&B
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log(
                                {"samples": wandb.Table(
                                    columns=["text"],
                                    data=[[s] for s in sentences[:8]],
                                )},
                                step=new_step,
                            )
                    except ImportError:
                        pass

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            bs = min(cfg.eval.perplexity_batch_size, sample.shape[0])
                            n_batches = sample.shape[0] // bs
                            total_ppl = 0.0
                            for i in range(n_batches):
                                s = sample[i * bs : (i + 1) * bs]
                                loss_gpt, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                ppl = F.cross_entropy(
                                    logits[..., :-1], s[..., 1:], reduction="none"
                                ).mean(dim=-1).exp().mean()
                                total_ppl += ppl.item()
                            total_ppl /= max(n_batches, 1)
                            logger.info(f"Generative perplexity at step {new_step}: {total_ppl:.3f}")
                            wandb_log({"eval/generative_perplexity": total_ppl}, step=new_step)
                            del eval_model
                            torch.cuda.empty_cache()

    # Save final model
    final_path = os.path.join(work_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    sedd_utils.save_checkpoint(os.path.join(final_path, "checkpoint.pth"), state)
    logger.info(f"Training complete. Final checkpoint saved to {final_path}.")
    wandb_finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)