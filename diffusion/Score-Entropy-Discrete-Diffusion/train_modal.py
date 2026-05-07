"""
Run SEDD (Score Entropy Discrete Diffusion) training and inference on Modal.

Prerequisites:
  pip install modal
  modal token set  # one-time auth

Quick test (sample from pretrained model, 50 steps):
  modal run train_modal.py::main

Train SEDD (absorb graph, loglinear noise, small model, single A100):
  modal run --detach train_modal.py::train_main

Train SEDD uniform (geometric noise, uniform graph):
  modal run --detach train_modal.py::train_main --graph-type uniform --noise-type geometric

Resume training from a previous run:
  modal run --detach train_modal.py::train_main --run-name my-run --resume

Sample unconditionally from a pretrained model:
  modal run train_modal.py::sample_main --model-path louaaron/sedd-medium --steps 256 --batch-size 4

Sample from a locally trained checkpoint (run name resolves to /checkpoints/exp_local/<run-name>):
  modal run train_modal.py::sample_main --model-path my-run

Sample conditionally with prefix/suffix:
  modal run train_modal.py::sample_cond_main --prefix "Once upon a time" --suffix "The End."

Download checkpoints:
  modal volume get sedd-checkpoints exp_local/<run-name> ./local_sedd_run
"""

import os
import subprocess
import sys

import modal

SEDD_DIR = os.path.dirname(__file__)

app = modal.App("sedd-train")

# Persistent volume for checkpoints and cached datasets
volume = modal.Volume.from_name("sedd-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "ninja-build")
    # PyTorch for CUDA 12.1 (Modal A100 instances run CUDA 12.x)
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Pre-built flash-attn wheel — avoids needing nvcc at image build time.
    # Built for torch 2.2 + CUDA 12.2, Python 3.10 (compatible with cu121 at runtime).
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/"
        "flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .pip_install(
        "transformers==4.38.1",
        "datasets==2.17.1",
        "hydra-core==1.3.2",
        "omegaconf",
        "accelerate",
        "einops",
        "jaxtyping",
        "fancy-einsum",
        "tqdm",
        "numpy<2",
        "wandb",
    )
    .add_local_dir(
        SEDD_DIR,
        remote_path="/sedd",
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            "data",
            "exp_local",
            "exp",
        ],
    )
)


def _resolve_model_path(model_path: str) -> str:
    """
    Resolve a model path for run_sample.py / run_sample_cond.py:
      - Absolute paths are used as-is.
      - HuggingFace model IDs (e.g. 'louaaron/sedd-medium') are used as-is.
      - Plain run names (e.g. 'my-run') resolve to /checkpoints/exp_local/<name>.
    """
    if model_path.startswith("/"):
        return model_path
    if "/" in model_path:
        # Looks like a HuggingFace owner/model ID
        return model_path
    return f"/checkpoints/exp_local/{model_path}"


# =============================================================================
# Training
# =============================================================================

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600 * 24,  # 24 hours max
    secrets=[modal.Secret.from_name("wandb-secret")],
    # Reduce CUDA allocator fragmentation — helps when score_entropy creates
    # large intermediate tensors (e.g. [B*L, vocab_size].exp()).
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)
def train(
    run_name: str = "sedd-run",
    model: str = "small",
    noise_type: str = "loglinear",
    graph_type: str = "absorb",
    n_iters: int = 1300001,
    # batch_size is the *total* batch size; per-GPU = batch_size / (ngpus * accum)
    # Default 32 is conservative for a single A100 80GB with the small model.
    # Increase to 64 or 128 with accum to match the paper's effective batch of 512.
    batch_size: int = 32,
    accum: int = 1,
    ngpus: int = 1,
    resume: bool = False,
    wandb_project: str = "sedd",
    pretrained_from: str = "",
    eval_batch_size: int = 32,
):
    """
    Train SEDD on a single A100 via DDP (ngpus=1 by default).

    Checkpoints and samples are saved to the 'sedd-checkpoints' volume under
    exp_local/<run_name>/. Pass resume=True to continue from the latest meta-checkpoint.

    Effective batch size = batch_size. To match the paper's 512-sample batches
    on a single GPU, set batch_size=64 and accum=8 (or batch_size=128 accum=4).

    Metrics are logged to W&B under wandb_project. Requires a Modal secret named
    'wandb-secret' with WANDB_API_KEY set.

    pretrained_from: HuggingFace model ID to use as weight initialization for
    continued pretraining, e.g. 'louaaron/sedd-small'. If set, the pretrained
    weights are saved as the initial checkpoint before training begins, so
    training starts from those weights at step 0 with a fresh optimizer.

    eval_batch_size: batch size for the periodic eval step (default 32). The
    config default of 512 causes OOM at seq_len=1024 on a 40 GB A100.
    """
    os.chdir("/sedd")

    work_dir = f"/checkpoints/exp_local/{run_name}"
    os.makedirs(work_dir, exist_ok=True)

    # If pretrained_from is set and no checkpoint exists yet, initialize from HF weights
    ckpt_path = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if pretrained_from and not os.path.exists(ckpt_path):
        print(f"Initializing from pretrained model: {pretrained_from}")
        init_cmd = [
            sys.executable, "init_from_pretrained.py",
            "--pretrained_from", pretrained_from,
            "--work_dir", work_dir,
            "--noise_type", noise_type,
        ]
        result = subprocess.run(init_cmd)
        if result.returncode != 0:
            raise RuntimeError("init_from_pretrained.py failed")

    cmd = [
        sys.executable, "train.py",
        f"ngpus={ngpus}",
        f"model={model}",
        f"noise.type={noise_type}",
        f"graph.type={graph_type}",
        f"training.n_iters={n_iters}",
        f"training.batch_size={batch_size}",
        f"training.accum={accum}",
        # Persist downloaded datasets in the volume so reruns skip the download
        "data.cache_dir=/checkpoints/data",
        # Set absolute output directory and disable Hydra's CWD change
        f"hydra.run.dir={work_dir}",
        "hydra.job.chdir=false",
        f"wandb_project={wandb_project}",
        f"wandb_name={run_name}",
        f"eval.batch_size={eval_batch_size}",
        # perplexity_batch_size must be <= sampling batch (batch_size // (ngpus * accum))
        # to avoid ZeroDivisionError in run_train.py:228
        f"eval.perplexity_batch_size={batch_size // (ngpus * accum)}",
    ]

    # uniform graph requires scale_by_sigma=False (not yet configured in defaults)
    if graph_type == "uniform":
        cmd.append("model.scale_by_sigma=False")

    # Resume from a previous run's meta-checkpoint.
    # No load_dir needed: restore_checkpoint() in run_train.py always loads
    # work_dir/checkpoints-meta/checkpoint.pth unconditionally. Since work_dir
    # is already set via hydra.run.dir above, the checkpoint is found automatically.
    # (load_dir is only needed to reload a *different* run's config.)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)

    # Commit volume regardless of outcome to preserve any partial checkpoints
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")

    print(f"Checkpoints saved to volume under exp_local/{run_name}/")
    print(f"Download with: modal volume get sedd-checkpoints exp_local/{run_name} ./{run_name}")


@app.local_entrypoint()
def train_main(
    run_name: str = "sedd-run",
    model: str = "small",
    noise_type: str = "loglinear",
    graph_type: str = "absorb",
    n_iters: int = 1300001,
    batch_size: int = 32,
    accum: int = 1,
    ngpus: int = 1,
    resume: bool = False,
    wandb_project: str = "sedd",
    pretrained_from: str = "",
    eval_batch_size: int = 32,
):
    """
    Entrypoint for `modal run train_modal.py::train_main [options]`.

    Common overrides:
      --run-name my-experiment
      --model medium
      --batch-size 64 --accum 8      # effective batch 512 on single GPU
      --n-iters 100000               # shorter run
      --resume                       # continue from checkpoints-meta/checkpoint.pth
      --wandb-project my-project
      --pretrained-from louaaron/sedd-small   # continued pretraining from HF weights
      --eval-batch-size 32           # reduce from default 512 to avoid OOM on A100-40GB
    """
    train.remote(
        run_name=run_name,
        model=model,
        noise_type=noise_type,
        graph_type=graph_type,
        n_iters=n_iters,
        batch_size=batch_size,
        accum=accum,
        ngpus=ngpus,
        resume=resume,
        wandb_project=wandb_project,
        pretrained_from=pretrained_from,
        eval_batch_size=eval_batch_size,
    )


# =============================================================================
# Multi-GPU training (8× A100-40GB)
# Matches the original SEDD paper setup: batch_size=512 across 8 GPUs (64/GPU),
# eval_batch_size=512 (64/GPU), accum=1. Uses the same training body as train().
# =============================================================================

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=8),
    volumes={"/checkpoints": volume},
    timeout=3600 * 24,
    secrets=[modal.Secret.from_name("wandb-secret")],
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)
def train_8gpu(
    run_name: str = "sedd-run-8gpu",
    model: str = "small",
    noise_type: str = "loglinear",
    graph_type: str = "absorb",
    n_iters: int = 1300001,
    batch_size: int = 512,
    accum: int = 1,
    eval_batch_size: int = 512,
    wandb_project: str = "sedd",
    pretrained_from: str = "",
):
    """
    Train SEDD on 8× A100-40GB — matches the original paper's setup.

    With 8 GPUs: per-GPU batch = 512/8 = 64, eval per-GPU = 64. Both fit in 40 GB.
    ~8× faster wall-clock than single-GPU at the same cost per step.

    pretrained_from: HuggingFace model ID for continued pretraining (e.g. 'louaaron/sedd-small').
    """
    os.chdir("/sedd")

    work_dir = f"/checkpoints/exp_local/{run_name}"
    os.makedirs(work_dir, exist_ok=True)

    ckpt_path = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if pretrained_from and not os.path.exists(ckpt_path):
        print(f"Initializing from pretrained model: {pretrained_from}")
        init_cmd = [
            sys.executable, "init_from_pretrained.py",
            "--pretrained_from", pretrained_from,
            "--work_dir", work_dir,
            "--noise_type", noise_type,
        ]
        result = subprocess.run(init_cmd)
        if result.returncode != 0:
            raise RuntimeError("init_from_pretrained.py failed")

    cmd = [
        sys.executable, "train.py",
        "ngpus=8",
        f"model={model}",
        f"noise.type={noise_type}",
        f"graph.type={graph_type}",
        f"training.n_iters={n_iters}",
        f"training.batch_size={batch_size}",
        f"training.accum={accum}",
        "data.cache_dir=/checkpoints/data",
        f"hydra.run.dir={work_dir}",
        "hydra.job.chdir=false",
        f"wandb_project={wandb_project}",
        f"wandb_name={run_name}",
        f"eval.batch_size={eval_batch_size}",
        f"eval.perplexity_batch_size={batch_size // (8 * accum)}",
    ]

    if graph_type == "uniform":
        cmd.append("model.scale_by_sigma=False")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)

    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")

    print(f"Checkpoints saved to volume under exp_local/{run_name}/")
    print(f"Download with: modal volume get sedd-checkpoints exp_local/{run_name} ./{run_name}")


@app.local_entrypoint()
def train_8gpu_main(
    run_name: str = "sedd-run-8gpu",
    model: str = "small",
    noise_type: str = "loglinear",
    graph_type: str = "absorb",
    n_iters: int = 1300001,
    batch_size: int = 512,
    accum: int = 1,
    eval_batch_size: int = 512,
    wandb_project: str = "sedd",
    pretrained_from: str = "",
):
    """
    Entrypoint for `modal run train_modal.py::train_8gpu_main [options]`.

    Runs on 8× A100-40GB — original paper setup. ~8× faster wall-clock than train_main.

    Examples:
      # Sanity check (~8 min, ~$4)
      modal run --detach train_modal.py::train_8gpu_main --n-iters 20000 --run-name sedd-8gpu-test

      # Continued pretraining (~1.25 hrs, ~$40)
      modal run --detach train_modal.py::train_8gpu_main \\
        --pretrained-from louaaron/sedd-small \\
        --run-name sedd-continued-8gpu \\
        --n-iters 200000

      # Full training from scratch (~14 hrs, ~$450)
      modal run --detach train_modal.py::train_8gpu_main --run-name sedd-scratch-8gpu
    """
    train_8gpu.remote(
        run_name=run_name,
        model=model,
        noise_type=noise_type,
        graph_type=graph_type,
        n_iters=n_iters,
        batch_size=batch_size,
        accum=accum,
        eval_batch_size=eval_batch_size,
        wandb_project=wandb_project,
        pretrained_from=pretrained_from,
    )


# =============================================================================
# Unconditional sampling
# =============================================================================

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def sample(
    model_path: str = "louaaron/sedd-medium",
    batch_size: int = 4,
    steps: int = 1024,
):
    """
    Generate unconditional text samples.

    model_path can be:
      - A HuggingFace model ID: 'louaaron/sedd-small', 'louaaron/sedd-medium'
      - A local run name: 'my-run'  (resolves to /checkpoints/exp_local/my-run)
      - An absolute path: '/checkpoints/exp_local/my-run'
    """
    os.chdir("/sedd")

    resolved = _resolve_model_path(model_path)
    cmd = [
        sys.executable, "run_sample.py",
        "--model_path", resolved,
        "--batch_size", str(batch_size),
        "--steps", str(steps),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def sample_main(
    model_path: str = "louaaron/sedd-medium",
    batch_size: int = 4,
    steps: int = 1024,
):
    """
    Entrypoint for `modal run train_modal.py::sample_main [options]`.

    Examples:
      modal run train_modal.py::sample_main --model-path louaaron/sedd-small --steps 256
      modal run train_modal.py::sample_main --model-path my-run --steps 512 --batch-size 8
    """
    sample.remote(model_path=model_path, batch_size=batch_size, steps=steps)


# =============================================================================
# Conditional sampling (prefix / suffix infilling)
# =============================================================================

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def sample_cond(
    model_path: str = "louaaron/sedd-medium",
    batch_size: int = 4,
    steps: int = 1024,
    prefix: str = "Hi, my name is",
    suffix: str = " and that's why I'm late.",
):
    """
    Generate text conditioned on a fixed prefix and/or suffix.

    model_path follows the same resolution rules as `sample`.
    """
    os.chdir("/sedd")

    resolved = _resolve_model_path(model_path)
    cmd = [
        sys.executable, "run_sample_cond.py",
        "--model_path", resolved,
        "--batch_size", str(batch_size),
        "--steps", str(steps),
        "--prefix", prefix,
        "--suffix", suffix,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def sample_cond_main(
    model_path: str = "louaaron/sedd-medium",
    batch_size: int = 4,
    steps: int = 1024,
    prefix: str = "Hi, my name is",
    suffix: str = " and that's why I'm late.",
):
    """
    Entrypoint for `modal run train_modal.py::sample_cond_main [options]`.

    Examples:
      modal run train_modal.py::sample_cond_main --prefix "Once upon a time" --suffix "The End."
      modal run train_modal.py::sample_cond_main --model-path my-run --steps 512
    """
    sample_cond.remote(
        model_path=model_path,
        batch_size=batch_size,
        steps=steps,
        prefix=prefix,
        suffix=suffix,
    )


# =============================================================================
# Default entrypoint: quick smoke test (sample from pretrained model)
# =============================================================================

@app.local_entrypoint()
def main():
    """
    Quick smoke test: sample 2 sequences from the pretrained small model in 50 steps.
    Run with: modal run train_modal.py
    """
    sample.remote(model_path="louaaron/sedd-small", batch_size=2, steps=50)