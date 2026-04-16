"""
Zero-shot transfer evaluation for MrSEDD (plan Section 4.2, Experiment 4).

Evaluates a model trained on OWT on held-out datasets:
  PTB, WikiText-2, WikiText-103, LM1B, AG News

Usage
-----
python mrdiffusion-sedd/evaluation/eval_zero_shot.py \
    --checkpoint ./mrsedd_output/final/checkpoint.pt \
    --seq_len 128 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import math
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

EVAL_DATASETS = {
    "ptb":          ("ptb_text_only", "ptb_text_only", "test"),
    "wikitext-2":   ("wikitext", "wikitext-2-raw-v1", "test"),
    "wikitext-103": ("wikitext", "wikitext-103-raw-v1", "test"),
    "lm1b":         ("lm1b", None, "test"),
    "ag_news":      ("ag_news", None, "test"),
}


def eval_nelbo(model, data_loader, config, device):
    """Compute NELBO (negative ELBO) perplexity on a dataset."""
    sys.path.insert(0, _MODULE_DIR)
    from losses_mrdiffusion import get_loss_fn
    from train_mrdiffusion import load_graph, load_noise

    graph = load_graph(config)
    noise = load_noise(config)
    loss_fn = get_loss_fn(config, graph, noise, model, train=False)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            loss = loss_fn(input_ids)
            if isinstance(loss, dict):
                loss = loss.get("loss", loss.get("score_loss", list(loss.values())[0]))
            n_tokens = attn_mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    model.train()
    nll = total_loss / max(total_tokens, 1)
    return math.exp(nll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=200)
    p.add_argument("--datasets", nargs="+", default=list(EVAL_DATASETS.keys()))
    p.add_argument("--deletion_mode", default="hard", choices=["soft", "hard"],
                   help="Deletion mode at inference. 'hard' (default) physically removes tokens "
                        "as per guide: soft during training, hard at inference.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"[inference] deletion_mode={args.deletion_mode}")

    from datasets import load_dataset as hf_load
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\n{'Dataset':<20} | {'NELBO PPL':>10}")
    print("-" * 35)

    results = {}
    for ds_name in args.datasets:
        if ds_name not in EVAL_DATASETS:
            print(f"  Skipping unknown dataset: {ds_name}")
            continue

        hf_name, config_name, split = EVAL_DATASETS[ds_name]
        try:
            if config_name:
                ds = hf_load(hf_name, config_name, split=split, trust_remote_code=True)
            else:
                ds = hf_load(hf_name, split=split, trust_remote_code=True)

            text_col = "text" if "text" in ds.column_names else ds.column_names[0]

            def tok(batch):
                return tokenizer(batch[text_col], truncation=True,
                                 max_length=args.seq_len, padding="max_length")

            ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
            ds.set_format("torch")
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

            class LimitedLoader:
                def __init__(self, loader, n):
                    self.loader = loader
                    self.n = n
                def __iter__(self):
                    for i, batch in enumerate(self.loader):
                        if i >= self.n:
                            break
                        yield batch

            ppl = eval_nelbo(model, LimitedLoader(loader, args.max_batches),
                             config, device)
            results[ds_name] = ppl
            print(f"{ds_name:<20} | {ppl:>10.2f}")
        except Exception as e:
            print(f"{ds_name:<20} | ERROR: {e}")

    out = {"checkpoint": args.checkpoint, "results": results}
    print(f"\nZero-shot results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()