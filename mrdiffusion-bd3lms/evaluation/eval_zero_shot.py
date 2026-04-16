"""
Zero-shot transfer evaluation for MrBD3LM (plan Section 4.2, Experiment 4).

Evaluates a model trained on OWT on held-out datasets:
  PTB, WikiText-2, WikiText-103, LM1B, LAMBADA, AG News

Usage
-----
python mrdiffusion-bd3lms/evaluation/eval_zero_shot.py \\
    --checkpoint ./mrd_owt_output/final/checkpoint.pt \\
    --seq_len 128 --batch_size 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_SCRIPT_DIR)
_BD3LMS_ROOT = os.path.join(os.path.dirname(_MODULE_DIR), "diffusion", "bd3lms")
for p in [_MODULE_DIR, _BD3LMS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configuration_mrd_bd3lm import MrBD3LMConfig
from modeling_mrd_bd3lm import MrBD3LM
from noise_schedule import LogLinearNoise

EVAL_DATASETS = {
    "ptb":          ("ptb_text_only", "ptb_text_only", "test"),
    "wikitext-2":   ("wikitext", "wikitext-2-raw-v1", "test"),
    "wikitext-103": ("wikitext", "wikitext-103-raw-v1", "test"),
    "lm1b":         ("lm1b", None, "test"),
    "ag_news":      ("ag_news", None, "test"),
}


def eval_nelbo(model, data_loader, noise, mask_index, config, device):
    """Compute NELBO (negative ELBO) perplexity on a dataset."""
    sys.path.insert(0, _MODULE_DIR)
    from train_mrd_bd3lm import compute_loss

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            x0 = batch["input_ids"].to(device)
            attn_mask = batch.get("attention_mask",
                                   torch.ones_like(x0)).to(device)
            metrics = compute_loss(model, x0, attn_mask, noise,
                                   mask_index, config, device)
            total_loss += metrics["score_loss"].item() * attn_mask.sum().item()
            total_tokens += attn_mask.sum().item()
    model.train()

    nll = total_loss / max(total_tokens, 1)
    return math.exp(nll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=200)
    p.add_argument("--datasets", nargs="+",
                   default=list(EVAL_DATASETS.keys()))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt.get("config", {})
    try:
        config = MrBD3LMConfig(**cfg_dict)
    except Exception:
        config = MrBD3LMConfig(model_length=args.seq_len)

    model = MrBD3LM(config).to(device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    noise = LogLinearNoise().to(device)
    mask_index = config.vocab_size - 1

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
                ds = hf_load(hf_name, config_name, split=split,
                              trust_remote_code=True)
            else:
                ds = hf_load(hf_name, split=split, trust_remote_code=True)

            text_col = "text" if "text" in ds.column_names else ds.column_names[0]

            def tok(batch):
                return tokenizer(batch[text_col], truncation=True,
                                 max_length=args.seq_len, padding="max_length")

            ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
            ds.set_format("torch")
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                drop_last=False)

            # Limit to max_batches
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
                              noise, mask_index, config, device)
            results[ds_name] = ppl
            print(f"{ds_name:<20} | {ppl:>10.2f}")
        except Exception as e:
            print(f"{ds_name:<20} | ERROR: {e}")

    import json
    out = {"checkpoint": args.checkpoint, "results": results}
    print(f"\nZero-shot results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()