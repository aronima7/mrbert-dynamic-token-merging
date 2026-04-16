"""
Initialize a training checkpoint from a pretrained HuggingFace SEDD model.

Creates checkpoints-meta/checkpoint.pth in the format expected by run_train.py,
with pretrained model weights and a fresh optimizer/EMA at step 0.

Usage (called automatically by train_modal.py when --pretrained-from is set):
  python init_from_pretrained.py \
      --pretrained_from louaaron/sedd-small \
      --work_dir /checkpoints/exp_local/my-run \
      --noise_type loglinear \
      --ema_decay 0.9999 \
      --lr 3e-4
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from itertools import chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_from", required=True,
                        help="HuggingFace model ID or local path, e.g. louaaron/sedd-small")
    parser.add_argument("--work_dir", required=True,
                        help="Training work dir; checkpoint saved to <work_dir>/checkpoints-meta/checkpoint.pth")
    parser.add_argument("--noise_type", default="loglinear", choices=["loglinear", "geometric"])
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.work_dir, "checkpoints-meta")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.path.exists(ckpt_path):
        print(f"Checkpoint already exists at {ckpt_path} — skipping initialization.")
        return

    print(f"Loading pretrained model from {args.pretrained_from} ...")
    from model import SEDD
    from model.ema import ExponentialMovingAverage
    import noise_lib
    from omegaconf import OmegaConf

    model = SEDD.from_pretrained(args.pretrained_from)

    # Build a minimal config stub so noise_lib.get_noise() works
    noise_cfg = OmegaConf.create({
        "noise": {
            "type": args.noise_type,
            "sigma_min": 1e-4,
            "sigma_max": 20,
        }
    })
    noise = noise_lib.get_noise(noise_cfg)

    # EMA shadow params initialized to the current (pretrained) model weights
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    # Fresh AdamW optimizer over model + noise params (matches run_train.py structure)
    optimizer = optim.AdamW(
        chain(model.parameters(), noise.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),  # empty state (step 0)
        "step": 0,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Saved initialization checkpoint to {ckpt_path}")
    print(f"  model: {args.pretrained_from}")
    print(f"  step:  0")
    print(f"  EMA shadow params: {len(ema.shadow_params)} tensors")


if __name__ == "__main__":
    main()