"""
Run MrBERT training on Modal (serverless GPU).

Prerequisites:
  pip install modal
  modal token set  # one-time auth

Quick test (short run):
  modal run train_modal.py

Train MrBERT on SNLI (with delete gate):
  modal run --detach train_modal.py --model-type MrBERT --max-steps 30000

Train BERT baseline on SNLI (no delete gate):
  modal run --detach train_modal.py --model-type BERT --max-steps 30000

Download checkpoints when done:
  modal volume get mrbert-checkpoints checkpoints ./local_mrbert_checkpoints
"""

import os
import subprocess
import sys

import modal

# Training defaults
DEFAULT_ARGS = [
    "--task", "sequence_classification",
    "--dataset_name", "local_snli",
    "--local_snli_dir", "/checkpoints/snli_datasets",
    "--num_epochs", "3",
    "--batch_size", "32",
    "--learning_rate", "2e-5",
    "--target_deletion_rate", "0.3",
    "--deletion_loss_weight", "0.01",   # weak initial α — PI controller adjusts from here
    "--controller_p", "0.01",           # low proportional gain to prevent rapid α ramp-up
    "--controller_i", "0.00001",        # slow integral ramp to prevent gate collapse
    "--max_seq_length", "128",
    "--logging_steps", "50",
    "--save_steps", "1000",
    "--regularizer_delay", "1000",      # learn task first before applying deletion pressure
]

# Modal app and image
app = modal.App("mrbert-train")

# Volume for persisting checkpoints (survives after the run)
volume = modal.Volume.from_name("mrbert-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0,<5.0.0",
        "datasets",
        "tqdm",
        "accelerate",
        "numpy<2",
        "wandb",
    )
    .add_local_dir(
        "..",
        remote_path="/workspace",
        ignore=[
            "snli_datasets",
            "mrbert_checkpoints",
            "mrbert_sst2",
            "__pycache__",
            "*.pyc",
        ],
    )
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600 * 24,  # 24 hours max
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    output_dir: str = "/checkpoints",
    model_type: str = "MrBERT",
    max_steps: int = 20,
    num_epochs: int = 3,
    target_deletion_rate: float = 0.3,
    mode: str = "training-only",
    use_pi_controller: bool = True,
    delete_gate_layer: int = 3,
    wandb_run_name: str = "",
    controller_p: float = 0.01,
    use_softmax1: bool = True,
    extra_args: list[str] | None = None,
):
    """
    Run train_mrbert.py on Modal with GPU.

    Args:
        output_dir: Where to save checkpoints (default: /checkpoints, persisted to Volume).
        model_type: Model architecture — "MrBERT" (with delete gate) or "BERT" (baseline).
        max_steps: Override default max_steps (use -1 for full training).
        num_epochs: Number of training epochs (default: 3).
        target_deletion_rate: Fraction of tokens to delete (default: 0.3).
        mode: One of "training-only", "eval-only", "training-and-eval" (default: training-only).
        use_pi_controller: Dynamically adjust deletion loss weight to hit target rate (default: True).
        delete_gate_layer: Which encoder layer (0-indexed) gets the delete gate (default: 3).
        wandb_run_name: W&B run name (default: auto-generated).
        controller_p: Proportional gain for PI controller (default: 0.01). Increase for faster convergence.
        use_softmax1: Use softmax1 (n+1 denominator) for attention, as recommended by MrT5 paper (default: True).
        extra_args: Optional list of extra CLI args, e.g. ["--batch_size", "32"].
    """
    os.chdir("/workspace/training")

    run_label = wandb_run_name or f"mrbert-{model_type.lower()}-{mode}"
    print("=" * 60)
    print(f"RUN: {run_label}")
    print(f"  model_type={model_type}, epochs={num_epochs}, max_steps={max_steps}")
    print(f"  target_deletion_rate={target_deletion_rate}, delete_gate_layer={delete_gate_layer}")
    print(f"  mode={mode}, pi_controller={use_pi_controller}")
    print("=" * 60)
    # SNLI requires a preprocessed dataset. Generate it if missing.
    snli_train_path = "/checkpoints/snli_datasets/snli-train.json"
    if not os.path.exists(snli_train_path):
        print("Preprocessing SNLI dataset...")
        os.makedirs("/checkpoints/snli_datasets", exist_ok=True)
        preprocess_cmd = [
            sys.executable, "/workspace/data/preprocess_snli.py",
            "--output_dir", "/checkpoints/snli_datasets",
        ]
        env = {**os.environ, "PYTHONPATH": "/workspace"}
        result = subprocess.run(preprocess_cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"SNLI preprocessing exited with code {result.returncode}")
    else:
        print(f"SNLI dataset already exists at {snli_train_path}, skipping preprocessing.")

    cmd = [
        sys.executable,
        "train_mrbert.py",
        *DEFAULT_ARGS,
        "--model_type", model_type,
        "--output_dir", output_dir,
        "--max_steps", str(max_steps),
        "--num_epochs", str(num_epochs),
        "--target_deletion_rate", str(target_deletion_rate),
        "--mode", mode,
        "--delete_gate_layer", str(delete_gate_layer),
    ]
    if not use_pi_controller:
        cmd.append("--no_use_pi_controller")
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])
    cmd.extend(["--controller_p", str(controller_p)])
    if not use_softmax1:
        cmd.append("--no_use_softmax1")
    if extra_args:
        cmd.extend(extra_args)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)

    # Persist volume regardless of outcome to preserve any partial checkpoints
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")

    print("Checkpoints saved to Volume 'mrbert-checkpoints'.")
    print("Download with: modal volume get mrbert-checkpoints <remote_path> <local_path>")


@app.local_entrypoint()
def main(
    model_type: str = "MrBERT",
    max_steps: int = 20,
    num_epochs: int = 3,
    target_deletion_rate: float = 0.3,
    mode: str = "training-only",
    use_pi_controller: bool = True,
    delete_gate_layer: int = 3,
    wandb_run_name: str = "",
    controller_p: float = 0.01,
    use_softmax1: bool = True,
):
    """
    Entrypoint for `modal run train_modal.py [--model-type MrBERT|BERT] [--max-steps N] [--num-epochs N] [--target-deletion-rate F] [--mode ...] [--no-use-pi-controller] [--delete-gate-layer N] [--wandb-run-name NAME] [--controller-p F] [--no-use-softmax1]`.
    """
    train.remote(output_dir="/checkpoints", model_type=model_type, max_steps=max_steps, num_epochs=num_epochs, target_deletion_rate=target_deletion_rate, mode=mode, use_pi_controller=use_pi_controller, delete_gate_layer=delete_gate_layer, wandb_run_name=wandb_run_name, controller_p=controller_p, use_softmax1=use_softmax1)
