"""
MAUVE score evaluation for MrSEDD (plan Section 4.1, Experiment 5).

Computes MAUVE score between:
  - model-generated sequences
  - reference sequences from the dataset

Requires: pip install mauve-text

Usage
-----
python mrdiffusion-sedd/evaluation/eval_mauve.py \
    --checkpoint ./mrsedd_output/final/checkpoint.pt \
    --dataset wikitext-103-v1 --seq_len 128 \
    --n_gen 500 --n_ref 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_SEDD_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "Score-Entropy-Discrete-Diffusion")
for p in [_MODULE_DIR, _SEDD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configuration_mrdiffusion import MrDiffusionConfig
from modeling_mrdiffusion import MrSEDD


def generate_texts(model, tokenizer, config, device,
                   n_samples=500, num_steps=128, seq_len=128):
    """Generate text via SEDD ancestral sampling."""
    sys.path.insert(0, _MODULE_DIR)
    from train_mrdiffusion import _generate_samples
    samples = []
    batch_size = 16
    for start in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - start)
        texts = _generate_samples(model, tokenizer, config, device,
                                   n=bs, num_steps=num_steps, seq_len=seq_len)
        samples.extend(texts)
    return samples[:n_samples]


def get_reference_texts(dataset_name, seq_len, n_ref, tokenizer_name="gpt2"):
    """Load reference texts from dataset."""
    from datasets import load_dataset as hf_load
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    if dataset_name == "lm1b":
        ds = hf_load("lm1b", split="test", trust_remote_code=True)
        texts = [row["text"] for row in ds.select(range(min(n_ref, len(ds))))]
    elif dataset_name.startswith("wikitext"):
        ds = hf_load("wikitext", dataset_name, split="validation")
        texts = [row["text"] for row in ds if len(row["text"].strip()) > 20][:n_ref]
    else:
        ds = hf_load(dataset_name, split="train")
        texts = [row["text"] for row in ds.select(range(n_ref))]

    return texts[:n_ref]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="wikitext-103-v1")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_gen", type=int, default=500)
    p.add_argument("--n_ref", type=int, default=200)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--output_dir", default="./mauve_results_sedd")
    p.add_argument("--deletion_mode", default="hard", choices=["soft", "hard"],
                   help="Deletion mode at inference. 'hard' (default) physically removes tokens "
                        "as per guide: soft during training, hard at inference.")
    args = p.parse_args()

    try:
        import mauve
    except ImportError:
        print("Install mauve-text: pip install mauve-text")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt.get("config", {})
    try:
        config = MrDiffusionConfig(**cfg_dict)
    except Exception:
        config = MrDiffusionConfig()

    model = MrSEDD(config).to(device)
    state = ckpt.get("model", ckpt.get("ema", ckpt))
    model.load_state_dict(state, strict=False)
    model.mr_config.deletion_mode = args.deletion_mode
    model.eval()
    print(f"[inference] deletion_mode={args.deletion_mode}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Generate
    print(f"Generating {args.n_gen} samples...")
    gen_texts = generate_texts(model, tokenizer, config, device,
                                n_samples=args.n_gen, num_steps=args.num_steps,
                                seq_len=args.seq_len)

    # Reference
    print(f"Loading {args.n_ref} reference texts...")
    ref_texts = get_reference_texts(args.dataset, args.seq_len, args.n_ref)

    if len(gen_texts) == 0 or len(ref_texts) == 0:
        print("ERROR: no texts generated or loaded.")
        return

    # Compute MAUVE
    print(f"Computing MAUVE ({len(gen_texts)} gen, {len(ref_texts)} ref)...")
    result = mauve.compute_mauve(
        p_text=gen_texts[:args.n_gen],
        q_text=ref_texts[:args.n_ref],
        device_id=0 if torch.cuda.is_available() else -1,
        max_text_length=args.seq_len,
        verbose=True,
    )

    print(f"\nMAUVE score: {result.mauve:.4f}")

    with open(os.path.join(args.output_dir, "mauve_result.json"), "w") as f:
        json.dump({
            "mauve": result.mauve,
            "n_gen": len(gen_texts),
            "n_ref": len(ref_texts),
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "num_steps": args.num_steps,
        }, f, indent=2)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()