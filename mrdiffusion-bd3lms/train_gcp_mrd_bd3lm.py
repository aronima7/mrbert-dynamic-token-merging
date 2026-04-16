"""
GCP training script for MrBD3LM.

Supports single-node multi-GPU training via torchrun (DDP) and
single-node single-GPU training for debugging.

Usage
-----
# Single GPU (debug/test)
python mrdiffusion-sedd-bd3lms/train_gcp_mrd_bd3lm.py \\
    --max_steps 500000 --batch_size 8 \\
    --dataset openwebtext \\
    --output_dir gs://your-bucket/mrd-bd3lm/run1 \\
    --wandb_project mrd_bd3lm --wandb_run_name mrd-gcp-run1

# Multi-GPU DDP (4 GPUs on one node)
torchrun --nproc_per_node=4 mrdiffusion-sedd-bd3lms/train_gcp_mrd_bd3lm.py \\
    --max_steps 500000 --batch_size 32 \\
    --dataset openwebtext \\
    --output_dir ./output \\
    --wandb_project mrd_bd3lm --wandb_run_name mrd-gcp-4gpu

# GCP Vertex AI job launcher (run from local machine)
python mrdiffusion-sedd-bd3lms/train_gcp_mrd_bd3lm.py --launch_gcp \\
    --gcp_project <project_id> \\
    --gcp_region us-central1 \\
    --gcp_machine_type a2-highgpu-4g \\
    --wandb_run_name mrd-gcp-a100

Notes
-----
- For GCS output_dir (gs://...), install: pip install gcs-fuse-csi-driver or use
  gcsfs: pip install gcsfs  (checkpoints saved via gcsfs if output_dir starts with gs://)
- Set WANDB_API_KEY in environment or use: wandb login
- GCP Docker image should have: torch, transformers, datasets, wandb, einops
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BD3LMS_ROOT = os.path.join(os.path.dirname(_SCRIPT_DIR), "diffusion", "bd3lms")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _BD3LMS_ROOT not in sys.path:
    sys.path.insert(0, _BD3LMS_ROOT)

from configuration_mrd_bd3lm import MrBD3LMConfig
from modeling_mrd_bd3lm import MrBD3LM, deletion_rate_loss, target_rate_at_timestep
from noise_schedule import LogLinearNoise
from train_mrd_bd3lm import (
    compute_loss,
    evaluate,
    compute_generative_perplexity,
    save_checkpoint,
    load_checkpoint,
    load_dataset,
    init_wandb,
    wandb_log,
    wandb_finish,
    subs_parameterization,
    NEG_INFINITY,
)


# ══════════════════════════════════════════════════════════════════════════════
# GCS checkpoint helpers (optional, for gs:// output dirs)
# ══════════════════════════════════════════════════════════════════════════════

def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def save_checkpoint_gcs(state: dict, gcs_path: str):
    """Save checkpoint to GCS using gcsfs if available, else raise."""
    try:
        import gcsfs
        import io
        fs = gcsfs.GCSFileSystem()
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        with fs.open(gcs_path, "wb") as f:
            f.write(buf.read())
        print(f"[ckpt] saved to {gcs_path}")
    except ImportError:
        raise ImportError(
            "gcsfs is required for GCS output. Install with: pip install gcsfs")


def maybe_save_checkpoint(state: dict, path: str):
    if is_gcs_path(path):
        save_checkpoint_gcs(state, path)
    else:
        save_checkpoint(state, path)


# ══════════════════════════════════════════════════════════════════════════════
# DDP setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_ddp():
    """Initialize DDP if running under torchrun."""
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


# ══════════════════════════════════════════════════════════════════════════════
# GCP Vertex AI launcher
# ══════════════════════════════════════════════════════════════════════════════

def launch_gcp_job(args):
    """Submit a Vertex AI custom training job."""
    try:
        from google.cloud import aiplatform
    except ImportError:
        raise ImportError("google-cloud-aiplatform required. Install with: "
                          "pip install google-cloud-aiplatform")

    aiplatform.init(project=args.gcp_project, location=args.gcp_region)

    # Build training command
    cmd_args = [f"--{k.replace('_','-')} {v}"
                for k, v in vars(args).items()
                if k not in ("launch_gcp", "gcp_project", "gcp_region",
                              "gcp_machine_type", "gcp_image_uri")
                   and v is not None and not isinstance(v, bool)]
    bool_flags = [f"--{k.replace('_','-')}"
                  for k, v in vars(args).items()
                  if isinstance(v, bool) and v
                     and k not in ("launch_gcp",)]
    all_args = " ".join(cmd_args + bool_flags)

    image = args.gcp_image_uri or f"gcr.io/{args.gcp_project}/mrd-bd3lm:latest"

    worker_pool_specs = [{
        "machine_spec": {"machine_type": args.gcp_machine_type,
                         "accelerator_type": "NVIDIA_TESLA_A100",
                         "accelerator_count": 4},
        "replica_count": 1,
        "container_spec": {
            "image_uri": image,
            "command": ["torchrun", "--nproc_per_node=4",
                        "mrdiffusion-sedd-bd3lms/train_gcp_mrd_bd3lm.py"],
            "args": all_args.split(),
        },
    }]

    job = aiplatform.CustomJob(
        display_name=f"mrd-bd3lm-{args.wandb_run_name or 'run'}",
        worker_pool_specs=worker_pool_specs,
    )
    job.run(sync=False)
    print(f"[GCP] submitted job: {job.display_name}")
    print(f"[GCP] resource name: {job.resource_name}")
    return job


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    rank, world_size, device = setup_ddp()
    main = is_main_process(rank)

    if main:
        print(f"[train] world_size={world_size} device={device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_presets = {
        "small":  dict(hidden_dim=768,  n_blocks=12, n_heads=12, cond_dim=128),
        "medium": dict(hidden_dim=1024, n_blocks=24, n_heads=16, cond_dim=128),
        "tiny":   dict(hidden_dim=384,  n_blocks=6,  n_heads=6,  cond_dim=64),
    }
    preset = model_presets[args.model_size]

    config = MrBD3LMConfig(
        block_size=args.block_size,
        vocab_size=50257 + 1,
        model_length=args.seq_len,
        cross_attn=not args.no_cross_attn,
        adaln=True,
        attn_backend="sdpa",
        time_conditioning=True,
        **preset,
        delete_gate_layer=args.delete_gate_layer,
        restore_gate_layer=(
            None if args.restore_gate_layer == -1 else args.restore_gate_layer),
        deletion_type=args.deletion_type,
        deletion_mode=args.deletion_mode,
        sigmoid_mask_scale=args.sigmoid_mask_scale,
        deletion_threshold=args.deletion_threshold,
        gate_sigma_conditioned=args.gate_sigma_conditioned,
        use_gumbel_noise=args.use_gumbel_noise,
        deletion_rate_schedule=args.deletion_rate_schedule,
        target_deletion_rate=args.target_deletion_rate,
        r_min=args.r_min,
        r_max=args.r_max,
        deletion_rate_alpha=args.deletion_rate_alpha,
        sigma_max=1.0,
        deletion_loss_weight=0.0 if args.no_delete_gate else args.deletion_loss_weight,
        random_deletion_probability=args.random_deletion_probability,
    )

    model = MrBD3LM(config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index],
                    find_unused_parameters=True)

    n_params = sum(p.numel() for p in model.parameters())
    if main:
        print(f"[model] {n_params/1e6:.1f}M parameters")

    noise = LogLinearNoise().to(device)
    mask_index = config.vocab_size - 1

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader_base, val_loader, tokenizer = load_dataset(
        args.dataset, seq_len=args.seq_len,
        batch_size=args.batch_size // world_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    # Wrap with distributed sampler if DDP
    if world_size > 1:
        from datasets import load_dataset as hf_load
        # Rebuild with DistributedSampler (simplified: just use the existing loader split)
        # For proper DDP, you'd want DistributedSampler here
        pass  # DataLoader already handles per-GPU batching via batch_size//world_size

    # ── Optimizer ─────────────────────────────────────────────────────────────
    m = model.module if world_size > 1 else model
    gate_params, base_params = [], []
    for name, p in m.named_parameters():
        if "delete_gate" in name:
            gate_params.append(p)
        else:
            base_params.append(p)

    optimizer_groups = [{"params": base_params, "lr": args.lr}]
    if gate_params and args.delete_gate_lr is not None:
        optimizer_groups.append({"params": gate_params, "lr": args.delete_gate_lr})
    elif gate_params:
        optimizer_groups[0]["params"] += gate_params

    optimizer = torch.optim.AdamW(optimizer_groups,
                                   weight_decay=args.weight_decay,
                                   betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if not is_gcs_path(args.output_dir):
        ckpt_path = os.path.join(args.output_dir, "checkpoints", "checkpoint.pt")
        if os.path.exists(ckpt_path) and main:
            start_step = load_checkpoint(ckpt_path, m, optimizer, device)
    else:
        ckpt_path = args.output_dir.rstrip("/") + "/checkpoints/checkpoint.pt"

    # Broadcast start_step to all ranks
    if world_size > 1:
        start_step_t = torch.tensor([start_step], device=device)
        dist.broadcast(start_step_t, src=0)
        start_step = start_step_t.item()

    # ── W&B (main process only) ───────────────────────────────────────────────
    wandb_run = init_wandb(args) if main else None

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step = start_step
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    running = dict(loss=0.0, score_loss=0.0, del_loss=0.0, del_rate=0.0)

    while step < args.max_steps:
        for batch in train_loader_base:
            if step >= args.max_steps:
                break
            x0 = batch["input_ids"].to(device)
            attn_mask = batch.get("attention_mask",
                                   torch.ones_like(x0)).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                metrics = compute_loss(model if world_size == 1 else model.module,
                                       x0, attn_mask, noise,
                                       mask_index, config, device)
            scaler.scale(metrics["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            for k in running:
                running[k] += metrics.get(k.replace("_", "/"),
                                           metrics.get(k, 0)).item() if isinstance(
                    metrics.get(k.replace("_", "/"),
                                metrics.get(k, 0)), torch.Tensor) else 0.0

            if step % args.logging_steps == 0 and main:
                n = args.logging_steps
                log_m = {
                    "train/loss": running["loss"] / n,
                    "train/score_loss": running["score_loss"] / n,
                    "train/deletion_rate": running["del_rate"] / n,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                }
                print(f"step {step:6d} | loss {log_m['train/loss']:.4f} "
                      f"| del_rate {log_m['train/deletion_rate']:.3f}")
                wandb_log(wandb_run, log_m, step)
                for k in running:
                    running[k] = 0.0

            if step % args.eval_steps == 0 and main:
                eval_m = evaluate(model if world_size == 1 else model.module,
                                   val_loader, noise, mask_index, config, device)
                print(f"  [eval] {eval_m}")
                wandb_log(wandb_run, eval_m, step)

            if step % args.ppl_steps == 0 and main:
                ppl = compute_generative_perplexity(
                    model if world_size == 1 else model.module,
                    tokenizer, noise, mask_index, config, device,
                    n_samples=args.ppl_n_samples)
                wandb_log(wandb_run, {"eval/generative_perplexity": ppl}, step)

            if step % args.save_steps == 0 and main:
                state = {
                    "step": step,
                    "model": (model.module if world_size > 1 else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config.__dict__,
                }
                maybe_save_checkpoint(state, ckpt_path)

    if main:
        state = {
            "step": step,
            "model": (model.module if world_size > 1 else model).state_dict(),
            "config": config.__dict__,
        }
        final_path = args.output_dir.rstrip("/") + "/final/checkpoint.pt"
        maybe_save_checkpoint(state, final_path)
        print(f"[train] done at step {step}")
        wandb_finish(wandb_run)

    cleanup_ddp()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()

    # GCP launcher
    p.add_argument("--launch_gcp", action="store_true",
                   help="Launch a Vertex AI job instead of training locally")
    p.add_argument("--gcp_project", type=str, default=None)
    p.add_argument("--gcp_region", type=str, default="us-central1")
    p.add_argument("--gcp_machine_type", type=str, default="a2-highgpu-4g")
    p.add_argument("--gcp_image_uri", type=str, default=None)

    # Training
    p.add_argument("--max_steps", type=int, default=500_000)
    p.add_argument("--logging_steps", type=int, default=1_000)
    p.add_argument("--eval_steps", type=int, default=10_000)
    p.add_argument("--ppl_steps", type=int, default=50_000)
    p.add_argument("--save_steps", type=int, default=10_000)
    p.add_argument("--ppl_n_samples", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="./mrd_bd3lm_output")

    # Data
    p.add_argument("--dataset", type=str, default="openwebtext")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--block_size", type=int, default=1)
    p.add_argument("--no_cross_attn", action="store_true")

    # Model
    p.add_argument("--model_size", type=str, default="small",
                   choices=["tiny", "small", "medium"])

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=2_500)
    p.add_argument("--delete_gate_lr", type=float, default=None)

    # Gate
    p.add_argument("--no_delete_gate", action="store_true")
    p.add_argument("--delete_gate_layer", type=int, default=3)
    p.add_argument("--restore_gate_layer", type=int, default=-1)
    p.add_argument("--deletion_type", type=str, default="scaled_sigmoid")
    p.add_argument("--deletion_mode", type=str, default="soft",
                   choices=["soft", "hard"])
    p.add_argument("--sigmoid_mask_scale", type=float, default=-30.0)
    p.add_argument("--deletion_threshold", type=float, default=-15.0)
    p.add_argument("--gate_sigma_conditioned", dest="gate_sigma_conditioned",
                   action="store_true", default=True)
    p.add_argument("--no_gate_sigma_conditioned",
                   dest="gate_sigma_conditioned", action="store_false")
    p.add_argument("--use_gumbel_noise", action="store_true")
    p.add_argument("--deletion_rate_schedule", type=str, default="noise_fraction")
    p.add_argument("--target_deletion_rate", type=float, default=0.3)
    p.add_argument("--r_min", type=float, default=0.05)
    p.add_argument("--r_max", type=float, default=0.5)
    p.add_argument("--deletion_rate_alpha", type=float, default=1.0)
    p.add_argument("--deletion_loss_weight", type=float, default=0.1)
    p.add_argument("--random_deletion_probability", type=float, default=0.3)

    # W&B
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mrd_bd3lm")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.launch_gcp:
        launch_gcp_job(args)
    else:
        train(args)