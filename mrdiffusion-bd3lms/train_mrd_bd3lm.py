"""
Standalone training script for MrBD3LM.

Usage
-----
# Smoke test (CPU or single GPU, 200 steps, wikitext)
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \\
    --max_steps 200 --logging_steps 20 --eval_steps 100 \\
    --batch_size 4 --seq_len 128 --disable_wandb

# Baseline BD3LM (no gate)
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \\
    --no_delete_gate --dataset openwebtext \\
    --wandb_project mrd_bd3lm --wandb_run_name bd3lm-baseline

# MrBD3LM soft deletion (default)
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \\
    --delete_gate_layer 3 --deletion_mode soft \\
    --deletion_rate_schedule noise_fraction \\
    --deletion_loss_weight 0.1 \\
    --wandb_project mrd_bd3lm --wandb_run_name mrd-soft-layer3

# MrBD3LM hard deletion
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \\
    --delete_gate_layer 3 --deletion_mode hard \\
    --target_deletion_rate 0.3 \\
    --wandb_run_name mrd-hard-layer3

# MrBD3LM no sigma conditioning (ablation)
python mrdiffusion-sedd-bd3lms/train_mrd_bd3lm.py \\
    --no_gate_sigma_conditioned \\
    --wandb_run_name mrd-no-sigma-cond

All commands are run from the project root (CS224N-project/).
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


# ══════════════════════════════════════════════════════════════════════════════
# PI Controller (from mrbert/training/pi_controller.py)
# ══════════════════════════════════════════════════════════════════════════════

class PIController:
    """
    Proportional-Integral controller that adjusts deletion loss weight α.
    Tracks a target deletion rate r(t) = 0.05 + 0.65 * t (plan Section 2.5).
    """
    def __init__(self, target_rate: float, kp: float = 0.5,
                 ki: float = 1e-5, gamma: float = 0.9):
        self.target_rate = target_rate
        self.kp = kp
        self.ki = ki
        self.gamma = gamma
        self.p_acc = 0.0
        self.i_acc = 0.0

    def update(self, actual_rate: float) -> float:
        error = self.target_rate - actual_rate
        self.p_acc = self.gamma * self.p_acc + (1 - self.gamma) * self.kp * error
        self.i_acc = self.i_acc + self.ki * error
        return max(0.0, self.p_acc + self.i_acc)

    def reset(self):
        self.p_acc = 0.0
        self.i_acc = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# EMA helper
# ══════════════════════════════════════════════════════════════════════════════

class ExponentialMovingAverage:
    """EMA of model parameters (decay 0.9999, plan requirement)."""
    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in parameters}

    @classmethod
    def from_named_parameters(cls, named_params, decay: float = 0.9999):
        inst = cls.__new__(cls)
        inst.decay = decay
        inst.shadow = {n: p.data.clone() for n, p in named_params}
        return inst

    def update(self, named_params):
        with torch.no_grad():
            for n, p in named_params:
                if n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def store(self, named_params):
        self._backup = {n: p.data.clone() for n, p in named_params}

    def copy_to(self, named_params):
        for n, p in named_params:
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, named_params):
        for n, p in named_params:
            if n in self._backup:
                p.data.copy_(self._backup[n])


# ══════════════════════════════════════════════════════════════════════════════
# W&B helpers
# ══════════════════════════════════════════════════════════════════════════════

def init_wandb(args):
    if args.disable_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        print(f"[W&B] run: {run.url}")
        return run
    except ImportError:
        print("[W&B] wandb not installed — disabling W&B logging.")
        return None


def wandb_log(run, metrics: dict, step: int):
    if run is None:
        return
    import wandb
    wandb.log(metrics, step=step)


def wandb_finish(run):
    if run is None:
        return
    import wandb
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_name: str, seq_len: int, batch_size: int,
                 eval_batch_size: int, num_workers: int = 4,
                 mask_token_id: int = 50256):
    """
    Load and tokenize a HuggingFace text dataset.

    Supported:
        "lm1b"          — LM1B (bert-base-uncased tokenizer, seq_len=128 by default)
        "openwebtext"   — GPT-2 tokenizer
        "wikitext-103-v1", "wikitext-2-v1"
    """
    from datasets import load_dataset as hf_load
    from transformers import AutoTokenizer

    # LM1B uses bert-base-uncased tokenizer (plan requirement)
    if dataset_name == "lm1b":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        print(f"[data] loading lm1b ...")
        ds = hf_load("lm1b", trust_remote_code=True)
        text_col = "text"
        train_split, val_split = "train", "test"
        val_frac = None
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        print(f"[data] loading {dataset_name} ...")
        if dataset_name == "openwebtext":
            ds = hf_load("openwebtext", trust_remote_code=True)
            text_col = "text"
            train_split, val_split = "train", "train"
            val_frac = 0.001
        elif dataset_name.startswith("wikitext"):
            config_name = dataset_name
            ds = hf_load("wikitext", config_name)
            text_col = "text"
            train_split, val_split = "train", "validation"
            val_frac = None
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize(batch):
        tokens = tokenizer(batch[text_col], truncation=True,
                           max_length=seq_len, padding="max_length")
        return tokens

    train_ds = ds[train_split].map(tokenize, batched=True,
                                   remove_columns=ds[train_split].column_names,
                                   num_proc=min(num_workers, 4))
    if val_frac is not None:
        n_val = max(100, int(len(train_ds) * val_frac))
        val_dataset = train_ds.select(range(n_val))
        train_dataset = train_ds.select(range(n_val, len(train_ds)))
    else:
        val_dataset = ds[val_split].map(tokenize, batched=True,
                                        remove_columns=ds[val_split].column_names,
                                        num_proc=min(num_workers, 4))
        train_dataset = train_ds

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

    return train_loader, val_loader, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Training step
# ══════════════════════════════════════════════════════════════════════════════

NEG_INFINITY = -1e6
MASK_INDEX_OFFSET = 1  # GPT2 vocab_size=50257, mask_index=50257


def subs_parameterization(logits: torch.Tensor, xt: torch.Tensor,
                           mask_index: int) -> torch.Tensor:
    """MDLM/subs parameterization: log probs over vocabulary."""
    logits[:, :, mask_index] += NEG_INFINITY
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked = xt != mask_index
    logits[unmasked] = NEG_INFINITY
    logits[unmasked, xt[unmasked]] = 0
    return logits


def compute_loss(model: MrBD3LM,
                 x0: torch.Tensor,
                 attention_mask: torch.Tensor,
                 noise: LogLinearNoise,
                 mask_index: int,
                 mr_config: MrBD3LMConfig,
                 device: torch.device,
                 gate_temperature: float = 1.0) -> dict:
    """
    Compute combined BD3-LM + deletion rate loss + gate logit regularization.

    BD3-LM loss: MDLM / masked cross-entropy over masked positions.
    Deletion loss: MSE between actual gate deletion rate and target.
    Gate logit regularization: 0.001 * gate_logits.pow(2).mean()

    Returns dict with 'loss', 'score_loss', 'del_loss', 'deletion_rate'.
    """
    B, L = x0.shape
    # Sample timesteps uniformly
    t = torch.rand(B, device=device)  # [B]
    loss_scale, move_chance = noise(t)  # [B], [B]
    p = move_chance.unsqueeze(-1)  # [B, 1]

    # Mask tokens: xt[i,j] = mask_index if rand < p[i], else x0[i,j]
    move_indices = torch.rand(B, L, device=device) < p  # [B, L]
    xt = torch.where(move_indices, mask_index, x0)     # [B, L]

    # BD3-LM: concatenate [xt; x0] for cross-attention
    if mr_config.cross_attn:
        x_input = torch.cat([xt, x0], dim=-1)  # [B, 2L]
    else:
        x_input = xt

    logits = model(x_input, timesteps=t,
                   gate_temperature=gate_temperature)  # [B, L, V] or MaskedLMOutput
    if hasattr(logits, "logits"):
        logits = logits.logits

    # MDLM: log-softmax, zero out mask token, set unmasked probs to identity
    log_p = subs_parameterization(logits.clone(), xt, mask_index)

    # NLL loss over masked positions only
    log_p_theta = torch.gather(log_p, -1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    loss_scale_expanded = loss_scale.unsqueeze(-1).expand_as(log_p_theta)

    nll = -(loss_scale_expanded * log_p_theta * attention_mask)
    score_loss = nll.sum() / attention_mask.sum()

    # Deletion rate loss
    gate_output = model.backbone._last_gate_output  # [B, L, 1] or None
    gate_logits = getattr(model.backbone, "_last_gate_logits", None)  # [B, L, 1] or None
    del_loss = torch.tensor(0.0, device=device)
    actual_deletion_rate = torch.tensor(0.0, device=device)

    if gate_output is not None and mr_config.deletion_loss_weight > 0.0:
        target_rates = target_rate_at_timestep(move_chance, mr_config)
        del_loss = deletion_rate_loss(
            gate_output, target_rates, mr_config.deletion_threshold)
        # Actual deletion rate (hard threshold)
        actual_deletion_rate = (
            gate_output.squeeze(-1) <= mr_config.deletion_threshold
        ).float().mean()

    total_loss = score_loss + mr_config.deletion_loss_weight * del_loss

    # Gate logit regularization (plan Section 3.4: λ=0.001)
    gate_logit_reg_weight = getattr(mr_config, "gate_logit_reg_weight", 0.001)
    if gate_logits is not None and gate_logit_reg_weight > 0.0:
        logit_reg = gate_logit_reg_weight * gate_logits.pow(2).mean()
        total_loss = total_loss + logit_reg

    return {
        "loss": total_loss,
        "score_loss": score_loss.detach(),
        "del_loss": del_loss.detach(),
        "deletion_rate": actual_deletion_rate.detach(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: MrBD3LM, val_loader: DataLoader, noise: LogLinearNoise,
             mask_index: int, mr_config: MrBD3LMConfig,
             device: torch.device, max_batches: int = 50) -> dict:
    model.eval()
    total_loss = 0.0
    total_del_rate = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x0 = batch["input_ids"].to(device)
        attn_mask = batch.get("attention_mask",
                               torch.ones_like(x0)).to(device)
        metrics = compute_loss(model, x0, attn_mask, noise,
                               mask_index, mr_config, device)
        total_loss += metrics["loss"].item()
        total_del_rate += metrics["deletion_rate"].item()
        n += 1
    model.train()
    return {
        "eval/loss": total_loss / max(n, 1),
        "eval/deletion_rate": total_del_rate / max(n, 1),
    }


def compute_generative_perplexity(model: MrBD3LM,
                                  tokenizer,
                                  noise: LogLinearNoise,
                                  mask_index: int,
                                  mr_config: MrBD3LMConfig,
                                  device: torch.device,
                                  n_samples: int = 8) -> float:
    """
    Generate n_samples sequences via MDLM reverse diffusion, then score with GPT2-Large.
    Returns generative perplexity (lower is better).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer as AT
        eval_model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)
        eval_tokenizer = AT.from_pretrained("gpt2-large")
        eval_model.eval()
    except Exception as e:
        print(f"[ppl] could not load gpt2-large: {e}")
        return float("nan")

    samples = _generate_samples(model, tokenizer, noise, mask_index,
                                 mr_config, device, n=n_samples,
                                 num_steps=128)
    if not samples:
        return float("nan")

    total_nll = 0.0
    total_toks = 0
    with torch.no_grad():
        for text in samples:
            enc = eval_tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=1024)
            ids = enc["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            out = eval_model(ids, labels=ids)
            n_tok = ids.shape[1] - 1
            total_nll += out.loss.item() * n_tok
            total_toks += n_tok

    del eval_model
    if total_toks == 0:
        return float("nan")
    return math.exp(total_nll / total_toks)


@torch.no_grad()
def _generate_samples(model: MrBD3LM, tokenizer, noise, mask_index, mr_config,
                       device, n: int = 4, num_steps: int = 64) -> list[str]:
    """Simple MDLM reverse diffusion sampler (analytic update)."""
    model.eval()
    L = mr_config.model_length
    # Start from all-mask
    x = torch.full((n, L), mask_index, dtype=torch.long, device=device)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t = torch.full((n,), t_val, device=device)
        _, move_chance_t = noise(t)
        t_s = torch.clamp(t - dt, min=1e-4)
        _, move_chance_s = noise(t_s)

        if mr_config.cross_attn:
            # During sampling, we don't have x0; use x itself as a proxy
            x_input = torch.cat([x, x], dim=-1)
        else:
            x_input = x

        logits = model(x_input, timesteps=t)
        if hasattr(logits, "logits"):
            logits = logits.logits

        # MDLM: pick the most likely non-mask token for each masked position
        log_probs = subs_parameterization(logits.clone(), x, mask_index)
        probs = log_probs.exp()  # [B, L, V]

        # Probability of unmasking: (p_s / p_t)
        unmask_prob = (1 - move_chance_s / move_chance_t).unsqueeze(-1)  # [B, 1, 1]
        # For masked positions: sample from denoised probs with probability unmask_prob
        masked = (x == mask_index)
        rand = torch.rand(n, L, device=device)
        do_unmask = masked & (rand < unmask_prob.squeeze(-1))

        if do_unmask.any():
            # Sample from predicted distribution
            flat_probs = probs[do_unmask]  # [N, V]
            sampled = torch.multinomial(flat_probs + 1e-9, num_samples=1).squeeze(-1)
            x = x.clone()
            x[do_unmask] = sampled

    model.train()
    # Decode
    texts = tokenizer.batch_decode(x, skip_special_tokens=True)
    return [t for t in texts if len(t.strip()) > 10]


def _phase3_search(model, val_loader, noise, mask_index, config, device, step):
    """
    BD3-LM Phase 3: grid search over (r_max, deletion_rate_schedule) every 5K steps.

    Plan Section 3.2: grid over β ∈ {0.0,0.15,0.3,0.45}, ω, r_max ∈ {0.5,0.6,0.7,0.8}.
    Here we sweep r_max and update config if a better value is found.
    """
    r_max_grid = [0.5, 0.6, 0.7, 0.8]
    best_loss = float("inf")
    best_r_max = config.r_max
    original_r_max = config.r_max

    model.eval()
    with torch.no_grad():
        for r_max in r_max_grid:
            config.r_max = r_max
            total_loss = 0.0
            n_batches = 0
            for batch in val_loader:
                if n_batches >= 10:
                    break
                x0 = batch["input_ids"].to(device)
                attn_mask = batch.get("attention_mask",
                                       torch.ones_like(x0)).to(device)
                metrics = compute_loss(model, x0, attn_mask, noise,
                                       mask_index, config, device)
                total_loss += metrics["loss"].item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_r_max = r_max

    config.r_max = best_r_max
    model.train()
    print(f"[phase3] step={step}: best r_max={best_r_max:.2f} (loss={best_loss:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[ckpt] saved to {path}")


def load_checkpoint(path: str, model: MrBD3LM, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # ── Model config ──────────────────────────────────────────────────────────
    model_presets = {
        "small":  dict(hidden_dim=768,  n_blocks=12, n_heads=12, cond_dim=128),
        "medium": dict(hidden_dim=1024, n_blocks=24, n_heads=16, cond_dim=128),
        "tiny":   dict(hidden_dim=384,  n_blocks=6,  n_heads=6,  cond_dim=64),
    }
    preset = model_presets[args.model_size]

    # If no_delete_gate: build a standard BD3LM config with dummy gate params
    config = MrBD3LMConfig(
        block_size=args.block_size,
        vocab_size=50257 + 1,  # +1 for mask token
        model_length=args.seq_len,
        cross_attn=not args.no_cross_attn,
        adaln=True,
        attn_backend="sdpa",
        time_conditioning=True,
        var_min=False,
        **preset,
        # Gate params
        gate_type=args.gate_type,
        gate_bottleneck_dim=args.gate_bottleneck_dim,
        gate_logit_reg_weight=args.gate_logit_reg_weight,
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

    # Disable gate if requested
    if args.no_delete_gate:
        config.deletion_loss_weight = 0.0
        config.deletion_type = "fixed"
        config.fixed_deletion_amount = 0.0

    model = MrBD3LM(config).to(device)

    # ── Warm-start from pretrained checkpoint ─────────────────────────────────
    if args.from_pretrained_checkpoint:
        ckpt = torch.load(args.from_pretrained_checkpoint, map_location=device)
        model_state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"[warmstart] loaded from {args.from_pretrained_checkpoint}")
        if missing:
            print(f"[warmstart] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.1f}M parameters")

    # ── Noise schedule ────────────────────────────────────────────────────────
    noise = LogLinearNoise().to(device)
    mask_index = config.vocab_size - 1  # last token in vocab

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, tokenizer = load_dataset(
        args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    print(f"[data] train batches: {len(train_loader)}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    param_groups = [{"params": [], "lr": args.lr}]
    gate_params = []
    base_params = []

    for name, p in model.named_parameters():
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

    # ── LR scheduler (linear warmup) ──────────────────────────────────────────
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    ckpt_path = os.path.join(args.output_dir, "checkpoints", "checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"[ckpt] resuming from {ckpt_path}")
        start_step = load_checkpoint(ckpt_path, model, optimizer, device)
        print(f"[ckpt] resumed at step {start_step}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = init_wandb(args)
    if wandb_run:
        import wandb
        wandb.log({"model/n_parameters": n_params}, step=0)

    # ── EMA (decay 0.9999, plan requirement) ──────────────────────────────────
    ema = ExponentialMovingAverage.from_named_parameters(
        list(model.named_parameters()), decay=0.9999)

    # ── PI controller (plan Section 2.5) ──────────────────────────────────────
    pi_controller = PIController(
        target_rate=args.target_deletion_rate,
        kp=0.5, ki=1e-5, gamma=0.9,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step = start_step
    epoch = 0
    running_loss = 0.0
    running_score_loss = 0.0
    running_del_loss = 0.0
    running_del_rate = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Freeze transformer parameters for first freeze_transformer_steps
    if args.freeze_transformer_steps > 0 and step < args.freeze_transformer_steps:
        for name, p in model.named_parameters():
            if "delete_gate" not in name:
                p.requires_grad_(False)
        print(f"[train] transformer frozen for first {args.freeze_transformer_steps} steps")

    print(f"[train] starting at step {step}, max_steps {args.max_steps}")

    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break

            # ── Unfreeze transformer at freeze_transformer_steps ──────────────
            if args.freeze_transformer_steps > 0 and step == args.freeze_transformer_steps:
                for p in model.parameters():
                    p.requires_grad_(True)
                print(f"[train] unfreezing transformer at step {step}")

            # ── Gate warmup: ramp deletion_loss_weight 0→target over 5K-15K steps
            if args.gate_warmup_steps > 0 and step <= args.gate_warmup_steps:
                warmup_frac = min(1.0, step / max(1, args.gate_warmup_steps))
                config.deletion_loss_weight = warmup_frac * args.deletion_loss_weight
            else:
                config.deletion_loss_weight = args.deletion_loss_weight

            # ── Gumbel temperature annealing (2.0 → 0.5 over gumbel_temp_steps)
            if args.gumbel_temp_steps > 0:
                temp_frac = min(1.0, step / max(1, args.gumbel_temp_steps))
                gate_temp = args.gumbel_temp_start + temp_frac * (
                    args.gumbel_temp_end - args.gumbel_temp_start)
            else:
                gate_temp = 1.0

            x0 = batch["input_ids"].to(device)
            attn_mask = batch.get("attention_mask",
                                   torch.ones_like(x0)).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                metrics = compute_loss(model, x0, attn_mask, noise,
                                       mask_index, config, device,
                                       gate_temperature=gate_temp)

            scaler.scale(metrics["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ── EMA update ────────────────────────────────────────────────────
            ema.update(model.named_parameters())

            # ── PI controller update ──────────────────────────────────────────
            del_rate = metrics["deletion_rate"].item()
            pi_alpha = pi_controller.update(del_rate)
            # Apply PI-adjusted weight (bounded)
            if not args.no_delete_gate and args.use_pi_controller:
                config.deletion_loss_weight = min(
                    args.deletion_loss_weight * 3,
                    max(0.0, pi_alpha))

            step += 1
            running_loss += metrics["loss"].item()
            running_score_loss += metrics["score_loss"].item()
            running_del_loss += metrics["del_loss"].item()
            running_del_rate += del_rate

            # ── BD3-LM Phase 3: joint schedule search every 5K steps ──────────
            if (step % 5000 == 0 and not args.no_delete_gate
                    and args.phase3_search and step > args.gate_warmup_steps):
                _phase3_search(model, val_loader, noise, mask_index, config, device, step)

            # ── Logging ───────────────────────────────────────────────────────
            if step % args.logging_steps == 0:
                avg = lambda v: v / args.logging_steps
                log_metrics = {
                    "train/loss": avg(running_loss),
                    "train/score_loss": avg(running_score_loss),
                    "train/del_loss": avg(running_del_loss),
                    "train/deletion_rate": avg(running_del_rate),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/gate_temperature": gate_temp,
                    "train/deletion_loss_weight": config.deletion_loss_weight,
                    "train/pi_alpha": pi_alpha,
                    "train/step": step,
                }
                print(
                    f"step {step:6d} | loss {avg(running_loss):.4f} "
                    f"| del_rate {avg(running_del_rate):.3f} "
                    f"| temp {gate_temp:.2f} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e}")
                wandb_log(wandb_run, log_metrics, step)
                running_loss = running_score_loss = running_del_loss = running_del_rate = 0.0

            # ── Evaluation ────────────────────────────────────────────────────
            if step % args.eval_steps == 0:
                eval_metrics = evaluate(model, val_loader, noise,
                                        mask_index, config, device)
                print(f"  [eval] loss={eval_metrics['eval/loss']:.4f}  "
                      f"del_rate={eval_metrics['eval/deletion_rate']:.3f}")
                wandb_log(wandb_run, eval_metrics, step)

            # ── Generative perplexity ─────────────────────────────────────────
            if step % args.ppl_steps == 0:
                # Use EMA weights for evaluation
                ema.store(model.named_parameters())
                ema.copy_to(model.named_parameters())
                ppl = compute_generative_perplexity(
                    model, tokenizer, noise, mask_index, config, device,
                    n_samples=args.ppl_n_samples)
                ema.restore(model.named_parameters())
                print(f"  [ppl] generative perplexity={ppl:.1f}")
                wandb_log(wandb_run, {"eval/generative_perplexity": ppl}, step)

            # ── Checkpoint ────────────────────────────────────────────────────
            if step % args.save_steps == 0 or step == args.max_steps:
                save_checkpoint({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema_shadow": ema.shadow,
                    "config": config.__dict__,
                }, ckpt_path)
                snap_path = os.path.join(
                    args.output_dir, "snapshots", f"step_{step}.pt")
                save_checkpoint({
                    "step": step,
                    "model": model.state_dict(),
                    "ema_shadow": ema.shadow,
                    "config": config.__dict__,
                }, snap_path)

    # Final save
    save_checkpoint({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema_shadow": ema.shadow,
        "config": config.__dict__,
    }, os.path.join(args.output_dir, "final", "checkpoint.pt"))
    print(f"[train] done at step {step}")
    wandb_finish(wandb_run)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train MrBD3LM")

    # Training control
    p.add_argument("--max_steps", type=int, default=500_000)
    p.add_argument("--logging_steps", type=int, default=1_000)
    p.add_argument("--eval_steps", type=int, default=10_000)
    p.add_argument("--ppl_steps", type=int, default=50_000)
    p.add_argument("--save_steps", type=int, default=10_000)
    p.add_argument("--ppl_n_samples", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="./mrd_bd3lm_output")

    # Warm-start / training phases (plan Section 3.2)
    p.add_argument("--from_pretrained_checkpoint", type=str, default=None,
                   help="Phase 2: warm-start from a Phase 1 checkpoint path")
    p.add_argument("--gate_warmup_steps", type=int, default=5_000,
                   help="Ramp deletion_loss_weight 0→target over this many steps")
    p.add_argument("--freeze_transformer_steps", type=int, default=2_000,
                   help="Freeze transformer (non-gate) params for first N steps")
    p.add_argument("--gumbel_temp_start", type=float, default=2.0,
                   help="Initial Gumbel-sigmoid temperature (plan: 2.0)")
    p.add_argument("--gumbel_temp_end", type=float, default=0.5,
                   help="Final Gumbel-sigmoid temperature (plan: 0.5)")
    p.add_argument("--gumbel_temp_steps", type=int, default=50_000,
                   help="Anneal temperature over this many steps (plan: 50K)")
    p.add_argument("--use_pi_controller", action="store_true", default=False,
                   help="Use PI controller to track target deletion rate")
    p.add_argument("--phase3_search", action="store_true", default=False,
                   help="BD3-LM Phase 3: grid search over r_max every 5K steps")

    # Data
    p.add_argument("--dataset", type=str, default="wikitext-103-v1",
                   choices=["lm1b", "openwebtext", "wikitext-103-v1", "wikitext-2-v1"])
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--block_size", type=int, default=1,
                   help="BD3-LM block size (1=MDLM, 16=BD3-LM small, 64=BD3-LM large)")
    p.add_argument("--no_cross_attn", action="store_true",
                   help="Disable [xt;x0] cross-attention (MDLM baseline)")

    # Model
    p.add_argument("--model_size", type=str, default="small",
                   choices=["tiny", "small", "medium"])

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=2_500)
    p.add_argument("--delete_gate_lr", type=float, default=None,
                   help="Separate LR for gate parameters (default: same as --lr). "
                        "Plan recommends 3x base lr for gate.")

    # Delete gate
    p.add_argument("--no_delete_gate", action="store_true",
                   help="Disable the delete gate (vanilla BD3-LM baseline)")
    p.add_argument("--gate_type", type=str, default="scaled_sigmoid",
                   choices=["scaled_sigmoid", "bottleneck_mlp"],
                   help="Gate architecture: scaled_sigmoid (current) or bottleneck_mlp (plan)")
    p.add_argument("--gate_bottleneck_dim", type=int, default=128)
    p.add_argument("--gate_logit_reg_weight", type=float, default=0.001,
                   help="L2 regularization on gate logits (plan Section 3.4: 0.001)")
    p.add_argument("--delete_gate_layer", type=int, default=3)
    p.add_argument("--restore_gate_layer", type=int, default=-1,
                   help="-1 = just before output layer")
    p.add_argument("--deletion_type", type=str, default="scaled_sigmoid",
                   choices=["scaled_sigmoid", "random", "fixed"])
    p.add_argument("--deletion_mode", type=str, default="soft",
                   choices=["soft", "hard"])
    p.add_argument("--sigmoid_mask_scale", type=float, default=-30.0)
    p.add_argument("--deletion_threshold", type=float, default=-15.0)
    p.add_argument("--gate_sigma_conditioned", dest="gate_sigma_conditioned",
                   action="store_true", default=True)
    p.add_argument("--no_gate_sigma_conditioned",
                   dest="gate_sigma_conditioned", action="store_false")
    p.add_argument("--use_gumbel_noise", action="store_true", default=False)
    p.add_argument("--deletion_rate_schedule", type=str, default="noise_fraction",
                   choices=["constant", "linear_sigma", "power_sigma", "noise_fraction"])
    p.add_argument("--target_deletion_rate", type=float, default=0.3,
                   help="Used when deletion_rate_schedule=constant")
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
    train(args)