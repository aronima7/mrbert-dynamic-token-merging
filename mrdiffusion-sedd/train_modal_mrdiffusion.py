"""
Run MrDiffusion training on Modal (serverless GPU).

Prerequisites:
    pip install modal
    modal token set --token-id <id> --token-secret <secret>
    modal secret create wandb-secret WANDB_API_KEY=<your_key>

Quick smoke test (50 steps):
    modal run train_modal_mrdiffusion.py

Soft deletion, 30% target rate:
    modal run --detach train_modal_mrdiffusion.py::main \\
        --deletion-type scaled_sigmoid \\
        --deletion-mode soft \\
        --target-deletion-rate 0.3 \\
        --max-steps 500000

Hard deletion, sigma-conditioned gate:
    modal run --detach train_modal_mrdiffusion.py::main \\
        --deletion-mode hard \\
        --gate-sigma-conditioned \\
        --max-steps 500000

Baseline SEDD (no gate):
    modal run --detach train_modal_mrdiffusion.py::main --no-delete-gate

Download checkpoints when done:
    modal volume get mrdiffusion-sedd-checkpoints <run_label>/final ./local_mrdiffusion_final
    modal volume get mrdiffusion-sedd-checkpoints <run_label>/checkpoints ./local_checkpoints
"""

import os
import subprocess
import sys

import modal

# ---------------------------------------------------------------------------
# Modal app + persistent volume
# ---------------------------------------------------------------------------

app = modal.App("mrdiffusion-sedd-train")
volume = modal.Volume.from_name("mrdiffusion-sedd-checkpoints", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# debian_slim + explicit torch + pre-built flash-attn wheel avoids the
# "No module named 'torch'" build error that occurs when flash-attn is
# installed from source in an isolated pip build environment.

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Pre-built wheel for torch 2.2 + CUDA 12.2 + Python 3.10
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/"
        "flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .pip_install(
        "einops==0.7.0",
        "omegaconf==2.3.0",
        "hydra-core==1.3.2",
        "hydra-submitit-launcher==1.2.0",
        "datasets==2.17.1",
        "transformers==4.38.1",
        "wandb",
        "huggingface_hub",
        "accelerate",
        "numpy<2",
        "tqdm",
    )
    .add_local_dir(
        # Copy the entire CS224N-project root into the container
        os.path.join(os.path.dirname(__file__), ".."),
        remote_path="/workspace",
        ignore=[
            # checkpoints and caches
            "mrdiffusion_checkpoints",
            "mrbert_checkpoints",
            "mrbert",
            "mrbert-alina",
            "mrxlmr",
            "wandb",
            # large data
            "data",
            "*.pth",
            "*.bin",
            # git / IDE
            ".git",
            ".idea",
            "__pycache__",
            "*.pyc",
        ],
    )
)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600 * 24,  # 24 hours max
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    # Model
    model_size: str = "small",
    # Training
    max_steps: int = 50,
    batch_size: int = 32,
    output_dir: str = "/checkpoints",
    logging_steps: int = 50,
    eval_steps: int = 100,
    save_steps: int = 5000,
    seed: int = 42,
    # Delete gate
    no_delete_gate: bool = False,
    delete_gate_layer: int = 3,
    deletion_type: str = "scaled_sigmoid",
    deletion_mode: str = "soft",
    sigmoid_mask_scale: float = -30.0,
    deletion_threshold: float = -15.0,
    gate_sigma_conditioned: bool = False,
    use_gumbel_noise: bool = False,
    deletion_loss_weight: float = 0.1,
    target_deletion_rate: float = 0.3,
    delete_gate_lr: float = 0.0,
    # Continued pretraining
    pretrained_from: str = "",
    # W&B
    wandb_project: str = "mrdiffusion-sedd",
    wandb_run_name: str = "",
    disable_wandb: bool = False,
    # Extras
    extra_args: list[str] | None = None,
):
    """
    Run train_mrdiffusion.py on an A100 via Modal.

    Args:
        model_size: "small" (768-dim, 12 blocks) or "medium" (1024-dim, 24 blocks).
        max_steps: Total training steps. Set to -1 to use the default from config.yaml (1.3M).
        batch_size: Per-GPU batch size.
        output_dir: Root dir for checkpoints inside the Modal volume.
        logging_steps: Log train loss every N gradient steps.
        eval_steps: Run eval loss every N gradient steps.
        save_steps: Save a meta checkpoint every N gradient steps.
        seed: Random seed.
        no_delete_gate: If True, train vanilla SEDD without any delete gate (baseline).
        delete_gate_layer: DDiTBlock index (0-indexed) after which the gate fires.
        deletion_type: "scaled_sigmoid" | "log_sigmoid" | "random" | "fixed".
        deletion_mode: "soft" (attention bias) | "hard" (physical removal).
        sigmoid_mask_scale: Negative scale for soft deletion bias (default -30.0).
        deletion_threshold: Gate threshold for counting a token as deleted.
        gate_sigma_conditioned: Condition gate on sigma (noise level) in addition to hidden state.
        use_gumbel_noise: Add Gumbel noise to gate logits during training.
        deletion_loss_weight: Weight of auxiliary deletion rate loss.
        target_deletion_rate: Target fraction of tokens to delete (0.0 – 1.0).
        delete_gate_lr: If > 0, use a separate higher LR for gate parameters only.
        pretrained_from: HuggingFace model ID to warm-start from, e.g. 'louaaron/sedd-small'.
            Loads pretrained SEDD weights into MrSEDD (strict=False) so the transformer blocks
            start from pretrained weights and only the gate is trained from random init.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name. Auto-generated if empty.
        disable_wandb: Disable W&B logging.
        extra_args: Optional extra CLI arguments passed verbatim to train_mrdiffusion.py.
    """
    os.chdir("/workspace/mrdiffusion-sedd")

    model_tag = "sedd-baseline" if no_delete_gate else "mrsedd"
    run_label = wandb_run_name or (
        f"{model_tag}-{model_size}-{deletion_type}-layer{delete_gate_layer}"
        f"-{deletion_mode}-{int(target_deletion_rate * 100)}pct"
    )
    run_output_dir = f"{output_dir}/{run_label}"

    print("=" * 70)
    print(f"RUN: {run_label}")
    print(f"  output_dir={run_output_dir}")
    print(f"  model_size={model_size}, max_steps={max_steps}, batch_size={batch_size}")
    print(f"  no_delete_gate={no_delete_gate}")
    if not no_delete_gate:
        print(f"  deletion_type={deletion_type}, deletion_mode={deletion_mode}")
        print(f"  delete_gate_layer={delete_gate_layer}, target_deletion_rate={target_deletion_rate}")
        print(f"  gate_sigma_conditioned={gate_sigma_conditioned}")
    print("=" * 70)

    cmd = [
        sys.executable,
        "train_mrdiffusion.py",
        "--model_size", model_size,
        "--output_dir", run_output_dir,
        "--batch_size", str(batch_size),
        "--logging_steps", str(logging_steps),
        "--eval_steps", str(eval_steps),
        "--save_steps", str(save_steps),
        "--seed", str(seed),
        "--delete_gate_layer", str(delete_gate_layer),
        "--deletion_type", deletion_type,
        "--deletion_mode", deletion_mode,
        "--sigmoid_mask_scale", str(sigmoid_mask_scale),
        "--deletion_threshold", str(deletion_threshold),
        "--deletion_loss_weight", str(deletion_loss_weight),
        "--target_deletion_rate", str(target_deletion_rate),
        "--wandb_project", wandb_project,
    ]

    if max_steps > 0:
        cmd.extend(["--max_steps", str(max_steps)])
    if no_delete_gate:
        cmd.append("--no_delete_gate")
    if gate_sigma_conditioned:
        cmd.append("--gate_sigma_conditioned")
    if use_gumbel_noise:
        cmd.append("--use_gumbel_noise")
    if delete_gate_lr > 0.0:
        cmd.extend(["--delete_gate_lr", str(delete_gate_lr)])
    if pretrained_from:
        cmd.extend(["--pretrained_from", pretrained_from])
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])
    if disable_wandb:
        cmd.append("--disable_wandb")
    if extra_args:
        cmd.extend(extra_args)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/workspace/mrdiffusion-sedd"})

    # Always commit volume so partial checkpoints are not lost
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"train_mrdiffusion.py exited with code {result.returncode}")

    print(f"Done. Checkpoints at: {run_output_dir}")
    print("Download with:")
    print(f"  modal volume get mrdiffusion-sedd-checkpoints {run_label}/final ./local_mrdiffusion_final")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model_size: str = "small",
    max_steps: int = 50,
    batch_size: int = 32,
    no_delete_gate: bool = False,
    delete_gate_layer: int = 3,
    deletion_type: str = "scaled_sigmoid",
    deletion_mode: str = "soft",
    gate_sigma_conditioned: bool = False,
    use_gumbel_noise: bool = False,
    deletion_loss_weight: float = 0.1,
    target_deletion_rate: float = 0.3,
    delete_gate_lr: float = 0.0,
    pretrained_from: str = "",
    wandb_project: str = "mrdiffusion-sedd",
    wandb_run_name: str = "",
    disable_wandb: bool = False,
):
    """Local entrypoint: parses CLI flags and invokes the remote train() function."""
    train.remote(
        model_size=model_size,
        max_steps=max_steps,
        batch_size=batch_size,
        no_delete_gate=no_delete_gate,
        delete_gate_layer=delete_gate_layer,
        deletion_type=deletion_type,
        deletion_mode=deletion_mode,
        gate_sigma_conditioned=gate_sigma_conditioned,
        use_gumbel_noise=use_gumbel_noise,
        deletion_loss_weight=deletion_loss_weight,
        target_deletion_rate=target_deletion_rate,
        delete_gate_lr=delete_gate_lr,
        pretrained_from=pretrained_from,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        disable_wandb=disable_wandb,
    )


# =============================================================================
# Evaluation functions
# =============================================================================

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600 * 2,
)
def eval_zero_shot(
    run_name: str,
    seq_len: int = 128,
    batch_size: int = 8,
    max_batches: int = 200,
    deletion_mode: str = "hard",
    datasets: str = "",  # comma-separated, empty = all defaults
):
    """
    Zero-shot NELBO perplexity on PTB, WikiText-2/103, LM1B, AG News.

    run_name resolves to /checkpoints/<run_name>/final/checkpoint.pt.
    deletion_mode='hard' enforces the guide's soft-train/hard-inference pattern.
    """
    os.chdir("/workspace/mrdiffusion-sedd")
    checkpoint = f"/checkpoints/{run_name}/final/checkpoint.pt"

    cmd = [
        sys.executable, "evaluation/eval_zero_shot.py",
        "--checkpoint", checkpoint,
        "--seq_len", str(seq_len),
        "--batch_size", str(batch_size),
        "--max_batches", str(max_batches),
        "--deletion_mode", deletion_mode,
    ]
    if datasets:
        cmd.extend(["--datasets"] + datasets.split(","))

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/workspace/mrdiffusion-sedd"})
    if result.returncode != 0:
        raise RuntimeError(f"eval_zero_shot.py exited with code {result.returncode}")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600 * 4,
)
def eval_mauve(
    run_name: str,
    n_gen: int = 500,
    n_ref: int = 200,
    num_steps: int = 128,
    seq_len: int = 128,
    dataset: str = "wikitext-103-v1",
    deletion_mode: str = "hard",
    output_subdir: str = "mauve_results",
):
    """
    MAUVE score: model-generated vs. reference text distribution.

    run_name resolves to /checkpoints/<run_name>/final/checkpoint.pt.
    Results are saved back to /checkpoints/<run_name>/<output_subdir>/.
    """
    os.chdir("/workspace/mrdiffusion-sedd")
    checkpoint = f"/checkpoints/{run_name}/final/checkpoint.pt"
    output_dir = f"/checkpoints/{run_name}/{output_subdir}"

    cmd = [
        sys.executable, "evaluation/eval_mauve.py",
        "--checkpoint", checkpoint,
        "--dataset", dataset,
        "--seq_len", str(seq_len),
        "--n_gen", str(n_gen),
        "--n_ref", str(n_ref),
        "--num_steps", str(num_steps),
        "--output_dir", output_dir,
        "--deletion_mode", deletion_mode,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/workspace/mrdiffusion-sedd"})
    volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"eval_mauve.py exited with code {result.returncode}")
    print(f"Results saved to volume at {run_name}/{output_subdir}/mauve_result.json")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def eval_gate_behavior(
    run_name: str,
    seq_len: int = 128,
    batch_size: int = 8,
    n_batches: int = 50,
    dataset: str = "wikitext-2",
    deletion_mode: str = "soft",
    output_subdir: str = "gate_analysis",
):
    """
    Gate behavior analysis: deletion rate vs sigma, gate value distributions.

    deletion_mode='soft' (default) preserves the unperturbed gate values for
    diagnostic purposes. Pass deletion_mode='hard' to analyze behavior under
    actual inference conditions.
    Results are saved to /checkpoints/<run_name>/<output_subdir>/.
    """
    os.chdir("/workspace/mrdiffusion-sedd")
    checkpoint = f"/checkpoints/{run_name}/final/checkpoint.pt"
    output_dir = f"/checkpoints/{run_name}/{output_subdir}"

    cmd = [
        sys.executable, "evaluation/eval_gate_behavior.py",
        "--checkpoint", checkpoint,
        "--dataset", dataset,
        "--seq_len", str(seq_len),
        "--batch_size", str(batch_size),
        "--n_batches", str(n_batches),
        "--output_dir", output_dir,
        "--deletion_mode", deletion_mode,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/workspace/mrdiffusion-sedd"})
    volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"eval_gate_behavior.py exited with code {result.returncode}")
    print(f"Results saved to volume at {run_name}/{output_subdir}/")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def eval_flops(
    run_name: str,
    seq_len: int = 128,
    batch_size: int = 8,
    deletion_rate: float = 0.3,
    deletion_mode: str = "hard",
    profile: bool = True,
):
    """
    Analytical FLOPs estimate + optional wall-clock profiling (100 forward passes).

    deletion_mode='hard' (default) is required for wall-clock speedup to be real.
    """
    os.chdir("/workspace/mrdiffusion-sedd")
    checkpoint = f"/checkpoints/{run_name}/final/checkpoint.pt"

    cmd = [
        sys.executable, "evaluation/eval_flops.py",
        "--checkpoint", checkpoint,
        "--seq_len", str(seq_len),
        "--batch_size", str(batch_size),
        "--deletion_rate", str(deletion_rate),
        "--deletion_mode", deletion_mode,
    ]
    if profile:
        cmd.append("--profile")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/workspace/mrdiffusion-sedd"})
    if result.returncode != 0:
        raise RuntimeError(f"eval_flops.py exited with code {result.returncode}")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600 * 6,
)
def eval_pareto(
    mr_run_name: str,
    baseline_run_name: str = "",
    seq_len: int = 128,
    batch_size: int = 8,
    n_samples: int = 32,
    deletion_rate: float = 0.3,
    deletion_mode: str = "hard",
    step_counts: str = "32,64,128,256,512",
    output_subdir: str = "pareto_results",
):
    """
    Pareto frontier: generative perplexity vs total FLOPs at multiple step counts.

    mr_run_name resolves to /checkpoints/<mr_run_name>/final/checkpoint.pt.
    baseline_run_name (optional) resolves the same way; omit to skip baseline curve.
    Results are saved to /checkpoints/<mr_run_name>/<output_subdir>/.
    """
    os.chdir("/workspace/mrdiffusion-sedd")
    mr_checkpoint = f"/checkpoints/{mr_run_name}/final/checkpoint.pt"
    output_dir = f"/checkpoints/{mr_run_name}/{output_subdir}"

    cmd = [
        sys.executable, "evaluation/eval_pareto.py",
        "--mr_checkpoint", mr_checkpoint,
        "--seq_len", str(seq_len),
        "--batch_size", str(batch_size),
        "--n_samples", str(n_samples),
        "--deletion_rate", str(deletion_rate),
        "--deletion_mode", deletion_mode,
        "--output_dir", output_dir,
        "--step_counts",
    ] + step_counts.split(",")

    if baseline_run_name:
        cmd.extend(["--baseline_checkpoint",
                     f"/checkpoints/{baseline_run_name}/final/checkpoint.pt"])

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/workspace/mrdiffusion-sedd"})
    volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"eval_pareto.py exited with code {result.returncode}")
    print(f"Results saved to volume at {mr_run_name}/{output_subdir}/pareto_results.json")


# =============================================================================
# Evaluation local entrypoints
# =============================================================================

@app.local_entrypoint()
def eval_zero_shot_main(
    run_name: str,
    seq_len: int = 128,
    batch_size: int = 8,
    max_batches: int = 200,
    deletion_mode: str = "hard",
    datasets: str = "",
):
    """
    modal run train_modal_mrdiffusion.py::eval_zero_shot_main --run-name <run-name>
    """
    eval_zero_shot.remote(
        run_name=run_name,
        seq_len=seq_len,
        batch_size=batch_size,
        max_batches=max_batches,
        deletion_mode=deletion_mode,
        datasets=datasets,
    )


@app.local_entrypoint()
def eval_mauve_main(
    run_name: str,
    n_gen: int = 500,
    n_ref: int = 200,
    num_steps: int = 128,
    seq_len: int = 128,
    dataset: str = "wikitext-103-v1",
    deletion_mode: str = "hard",
    output_subdir: str = "mauve_results",
):
    """
    modal run train_modal_mrdiffusion.py::eval_mauve_main --run-name <run-name>
    """
    eval_mauve.remote(
        run_name=run_name,
        n_gen=n_gen,
        n_ref=n_ref,
        num_steps=num_steps,
        seq_len=seq_len,
        dataset=dataset,
        deletion_mode=deletion_mode,
        output_subdir=output_subdir,
    )


@app.local_entrypoint()
def eval_gate_behavior_main(
    run_name: str,
    seq_len: int = 128,
    batch_size: int = 8,
    n_batches: int = 50,
    dataset: str = "wikitext-2",
    deletion_mode: str = "soft",
    output_subdir: str = "gate_analysis",
):
    """
    modal run train_modal_mrdiffusion.py::eval_gate_behavior_main --run-name <run-name>
    """
    eval_gate_behavior.remote(
        run_name=run_name,
        seq_len=seq_len,
        batch_size=batch_size,
        n_batches=n_batches,
        dataset=dataset,
        deletion_mode=deletion_mode,
        output_subdir=output_subdir,
    )


@app.local_entrypoint()
def eval_flops_main(
    run_name: str,
    seq_len: int = 128,
    batch_size: int = 8,
    deletion_rate: float = 0.3,
    deletion_mode: str = "hard",
    profile: bool = True,
):
    """
    modal run train_modal_mrdiffusion.py::eval_flops_main --run-name <run-name>
    """
    eval_flops.remote(
        run_name=run_name,
        seq_len=seq_len,
        batch_size=batch_size,
        deletion_rate=deletion_rate,
        deletion_mode=deletion_mode,
        profile=profile,
    )


@app.local_entrypoint()
def eval_pareto_main(
    mr_run_name: str,
    baseline_run_name: str = "",
    seq_len: int = 128,
    batch_size: int = 8,
    n_samples: int = 32,
    deletion_rate: float = 0.3,
    deletion_mode: str = "hard",
    step_counts: str = "32,64,128,256,512",
    output_subdir: str = "pareto_results",
):
    """
    modal run train_modal_mrdiffusion.py::eval_pareto_main \\
        --mr-run-name <run-name> --baseline-run-name <baseline-run-name>
    """
    eval_pareto.remote(
        mr_run_name=mr_run_name,
        baseline_run_name=baseline_run_name,
        seq_len=seq_len,
        batch_size=batch_size,
        n_samples=n_samples,
        deletion_rate=deletion_rate,
        deletion_mode=deletion_mode,
        step_counts=step_counts,
        output_subdir=output_subdir,
    )