"""
Modal serverless GPU training for MrBD3LM.

Prerequisites
-------------
    pip install modal
    modal token set --token-id <id> --token-secret <secret>
    modal secret create wandb-secret WANDB_API_KEY=<your_key>

Usage
-----
# Quick smoke test (100 steps, A100)
modal run mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py

# Baseline BD3-LM (no gate)
modal run --detach mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py::main \\
    --no-delete-gate --max-steps 500000 \\
    --wandb-run-name bd3lm-baseline

# MrBD3LM soft deletion (noise_fraction schedule — recommended)
modal run --detach mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py::main \\
    --delete-gate-layer 3 --deletion-mode soft \\
    --deletion-rate-schedule noise_fraction \\
    --deletion-loss-weight 0.1 \\
    --max-steps 500000 \\
    --wandb-run-name mrd-soft-layer3-nf

# MrBD3LM hard deletion
modal run --detach mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py::main \\
    --deletion-mode hard --delete-gate-layer 3 \\
    --deletion-rate-schedule noise_fraction \\
    --max-steps 500000 \\
    --wandb-run-name mrd-hard-layer3

# Medium model
modal run --detach mrdiffusion-sedd-bd3lms/train_modal_mrd_bd3lm.py::main \\
    --model-size medium --batch-size 16 \\
    --delete-gate-layer 6 --deletion-mode soft \\
    --max-steps 1000000 \\
    --wandb-run-name mrd-medium-soft-layer6

# Download checkpoints
modal volume get mrd-bd3lm-checkpoints <run_name>/final ./local_mrd_bd3lm_final

# List all stored runs
modal volume ls mrd-bd3lm-checkpoints
"""

from __future__ import annotations
import os
import sys

import modal

# ══════════════════════════════════════════════════════════════════════════════
# Modal App + infrastructure
# ══════════════════════════════════════════════════════════════════════════════

app = modal.App("mrd-bd3lm-train")

# Persistent volume for checkpoints (survives across runs)
volume = modal.Volume.from_name("mrd-bd3lm-checkpoints", create_if_missing=True)
VOLUME_MOUNT = "/checkpoints"

# CUDA 12 + PyTorch image with all required packages
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:23.10-py3")
    .pip_install(
        "einops>=0.6.0",
        "transformers>=4.40.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "tokenizers>=0.15.0",
        "wandb>=0.16.0",
        "tqdm>=4.65.0",
        "numpy<2",
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# Training function
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 24,          # 24 hours max
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    # Mount the whole project (so we can import bd3lms code)
    mounts=[modal.Mount.from_local_dir(
        os.path.join(os.path.dirname(os.path.dirname(__file__))),
        remote_path="/project",
    )],
)
def train_remote(
    # Training control
    max_steps: int = 500_000,
    logging_steps: int = 1_000,
    eval_steps: int = 10_000,
    ppl_steps: int = 50_000,
    save_steps: int = 10_000,
    ppl_n_samples: int = 16,
    # Data
    dataset: str = "wikitext-103-v1",
    seq_len: int = 1024,
    batch_size: int = 8,
    eval_batch_size: int = 8,
    block_size: int = 1,
    no_cross_attn: bool = False,
    # Model
    model_size: str = "small",
    # Optimizer
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    warmup_steps: int = 2_500,
    delete_gate_lr: float = None,
    # Gate
    no_delete_gate: bool = False,
    delete_gate_layer: int = 3,
    restore_gate_layer: int = -1,
    deletion_type: str = "scaled_sigmoid",
    deletion_mode: str = "soft",
    sigmoid_mask_scale: float = -30.0,
    deletion_threshold: float = -15.0,
    gate_sigma_conditioned: bool = True,
    use_gumbel_noise: bool = False,
    deletion_rate_schedule: str = "noise_fraction",
    target_deletion_rate: float = 0.3,
    r_min: float = 0.05,
    r_max: float = 0.5,
    deletion_rate_alpha: float = 1.0,
    deletion_loss_weight: float = 0.1,
    random_deletion_probability: float = 0.3,
    # W&B
    wandb_project: str = "mrd_bd3lm",
    wandb_run_name: str = None,
):
    import sys
    sys.path.insert(0, "/project")
    sys.path.insert(0, "/project/mrdiffusion-sedd-bd3lms")
    sys.path.insert(0, "/project/diffusion/bd3lms")

    run_name = wandb_run_name or f"mrd-{deletion_mode}-layer{delete_gate_layer}"
    output_dir = os.path.join(VOLUME_MOUNT, run_name)

    import argparse
    from train_mrd_bd3lm import train

    # Build a Namespace that matches train()'s arg interface
    args = argparse.Namespace(
        max_steps=max_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        ppl_steps=ppl_steps,
        save_steps=save_steps,
        ppl_n_samples=ppl_n_samples,
        output_dir=output_dir,
        dataset=dataset,
        seq_len=seq_len,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=4,
        block_size=block_size,
        no_cross_attn=no_cross_attn,
        model_size=model_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        delete_gate_lr=delete_gate_lr,
        no_delete_gate=no_delete_gate,
        delete_gate_layer=delete_gate_layer,
        restore_gate_layer=restore_gate_layer,
        deletion_type=deletion_type,
        deletion_mode=deletion_mode,
        sigmoid_mask_scale=sigmoid_mask_scale,
        deletion_threshold=deletion_threshold,
        gate_sigma_conditioned=gate_sigma_conditioned,
        use_gumbel_noise=use_gumbel_noise,
        deletion_rate_schedule=deletion_rate_schedule,
        target_deletion_rate=target_deletion_rate,
        r_min=r_min,
        r_max=r_max,
        deletion_rate_alpha=deletion_rate_alpha,
        deletion_loss_weight=deletion_loss_weight,
        random_deletion_probability=random_deletion_probability,
        disable_wandb=False,
        wandb_project=wandb_project,
        wandb_run_name=run_name,
    )

    try:
        train(args)
    finally:
        # Always commit the volume so checkpoints are persisted
        volume.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    max_steps: int = 100,           # smoke test default
    logging_steps: int = 10,
    eval_steps: int = 50,
    ppl_steps: int = 100,
    save_steps: int = 100,
    ppl_n_samples: int = 4,
    dataset: str = "wikitext-2-v1",
    seq_len: int = 256,
    batch_size: int = 4,
    eval_batch_size: int = 4,
    block_size: int = 1,
    no_cross_attn: bool = False,
    model_size: str = "small",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    warmup_steps: int = 100,
    delete_gate_lr: float = None,
    no_delete_gate: bool = False,
    delete_gate_layer: int = 3,
    restore_gate_layer: int = -1,
    deletion_type: str = "scaled_sigmoid",
    deletion_mode: str = "soft",
    sigmoid_mask_scale: float = -30.0,
    deletion_threshold: float = -15.0,
    gate_sigma_conditioned: bool = True,
    use_gumbel_noise: bool = False,
    deletion_rate_schedule: str = "noise_fraction",
    target_deletion_rate: float = 0.3,
    r_min: float = 0.05,
    r_max: float = 0.5,
    deletion_rate_alpha: float = 1.0,
    deletion_loss_weight: float = 0.1,
    random_deletion_probability: float = 0.3,
    wandb_project: str = "mrd_bd3lm",
    wandb_run_name: str = "modal-smoke-test",
):
    """Local entrypoint: dispatches to the remote A100 function."""
    train_remote.remote(
        max_steps=max_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        ppl_steps=ppl_steps,
        save_steps=save_steps,
        ppl_n_samples=ppl_n_samples,
        dataset=dataset,
        seq_len=seq_len,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        block_size=block_size,
        no_cross_attn=no_cross_attn,
        model_size=model_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        delete_gate_lr=delete_gate_lr,
        no_delete_gate=no_delete_gate,
        delete_gate_layer=delete_gate_layer,
        restore_gate_layer=restore_gate_layer,
        deletion_type=deletion_type,
        deletion_mode=deletion_mode,
        sigmoid_mask_scale=sigmoid_mask_scale,
        deletion_threshold=deletion_threshold,
        gate_sigma_conditioned=gate_sigma_conditioned,
        use_gumbel_noise=use_gumbel_noise,
        deletion_rate_schedule=deletion_rate_schedule,
        target_deletion_rate=target_deletion_rate,
        r_min=r_min,
        r_max=r_max,
        deletion_rate_alpha=deletion_rate_alpha,
        deletion_loss_weight=deletion_loss_weight,
        random_deletion_probability=random_deletion_probability,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )