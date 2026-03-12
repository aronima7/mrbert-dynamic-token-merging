"""
Run mrt5/training/train.py on Modal (serverless GPU).

Prerequisites:
  pip install modal
  modal token set  # one-time auth

Quick test (20 steps, vowel_removal, MrT5):
  cd mrt5/training
  modal run train_modal.py

Custom run:
  modal run train_modal.py --training-task vowel_removal --model-type T5 --max-steps 500

Download checkpoints when done:
  modal volume get mrt5-checkpoints models ./local_mrt5_checkpoints

Checkpoints are saved under:
  /checkpoints/models/<training_task>/<model_type>/<run_name>/checkpoints/
"""

import modal
import os
import subprocess
import sys

app = modal.App("mrt5-train")

# Persists checkpoints after the container exits
volume = modal.Volume.from_name("mrt5-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "transformers==4.39.1",
        "datasets==2.18.0",
        "accelerate==0.28.0",
        "nltk==3.8.1",
        "pandas",
        "scipy",
        "seaborn",
        "matplotlib",
        "tqdm",
        "numpy>=1.21,<2",
        "wandb",
    )
    # Upload the mrt5/ source tree into /workspace. Exclude generated data dirs
    # and caches to avoid Modal's "file modified during build" error.
    .add_local_dir(
        "..",
        remote_path="/workspace",
        ignore=[
            "diagnostic_datasets",
            "lm_datasets",
            "finetune_datasets",
            "training/local_t5_checkpoints",
            "training/local_mrt5_checkpoints",
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
    env={
        "MRT5_BASE_PATH": "/checkpoints",
        "TORCHDYNAMO_DISABLE": "1",
    },
)
def train(
    training_task: str = "vowel_removal",
    model_type: str = "MrT5",
    max_steps: int = 20,
    extra_args: list[str] | None = None,
):
    """
    Run train.py on Modal with GPU.

    Args:
        training_task: Positional task arg (e.g. vowel_removal, span_corruption, xnli).
        model_type: Model architecture (T5, MrT5, RandomT5, FixedT5, BPT5, CanineT5).
        max_steps: Stop after this many steps (-1 for full training).
        extra_args: Additional CLI flags, e.g. ["--learning_rate", "1e-4"].
    """
    DIAGNOSTIC_TASKS = ['copy', 'vowel_removal', 'contextual_vowel_removal', 'merge_ABC']

    # Diagnostic tasks require a preprocessed dataset file. Generate it if missing.
    if training_task in DIAGNOSTIC_TASKS:
        dataset_path = f"/checkpoints/diagnostic_datasets/diagnostic-{training_task}-train.json"
        if not os.path.exists(dataset_path):
            print(f"Preprocessing {training_task} dataset...")
            os.makedirs("/checkpoints/diagnostic_datasets", exist_ok=True)
            preprocess_cmd = [
                sys.executable, "data/preprocess_diagnostic_dataset.py",
                training_task,
                "--train_n", "1280000",
                "--eval_n", "16000",
            ]
            env = {**os.environ, "PYTHONPATH": "/workspace"}
            result = subprocess.run(preprocess_cmd, stderr=subprocess.STDOUT, cwd="/workspace", env=env)
            if result.returncode != 0:
                raise RuntimeError(f"Preprocessing exited with code {result.returncode}")
        else:
            print(f"Dataset already exists at {dataset_path}, skipping preprocessing.")

    os.chdir("/workspace/training")

    cmd = [
        sys.executable, "train.py",
        training_task,
        "--model_type", model_type,
        "--max_steps", str(max_steps),
        "--save_steps", str(min(1000, max_steps)),
        "--eval_steps", str(min(1000, max_steps)),
        "--logging_steps", str(min(50, max_steps)),
        "--delete_gate_loss_coeff", "0.01",
        "--target_deletion_rate", "0.3",
        "--controller_i", "0.00001",    # reduced to prevent gate collapse
        "--regularizer_delay", "10000", # paper recommendation: 10k steps for diagnostic tasks
    ]
    if extra_args:
        cmd.extend(extra_args)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")

    volume.commit()
    print("Checkpoints saved to Volume 'mrt5-checkpoints'.")
    print("Download with: modal volume get mrt5-checkpoints models ./local_mrt5_checkpoints")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
    env={
        "MRT5_BASE_PATH": "/checkpoints",
        "TORCHDYNAMO_DISABLE": "1",
    },
)
def evaluate(
    training_task: str = "vowel_removal",
    model_type: str = "MrT5",
    model_name: str = "google/byt5-small",
    checkpoint: int = 20,
    num_batches: int = 250,
    random_seed: int = 42,
):
    """
    Run diagnostic_task_eval.py on Modal against a checkpoint stored in the volume.

    Args:
        training_task: Task the model was trained on (e.g. vowel_removal).
        model_type: Model architecture (T5, MrT5, RandomT5, FixedT5).
        model_name: HuggingFace model name used during training (e.g. google/byt5-small).
        checkpoint: Step number of the checkpoint to evaluate.
        num_batches: Number of eval batches to run.
        random_seed: Must match the seed used during training.
    """
    os.chdir("/workspace/eval")

    # Eval script also needs the test dataset
    DIAGNOSTIC_TASKS = ['copy', 'vowel_removal', 'contextual_vowel_removal', 'merge_ABC']
    if training_task in DIAGNOSTIC_TASKS:
        dataset_path = f"/checkpoints/diagnostic_datasets/diagnostic-{training_task}-test.json"
        if not os.path.exists(dataset_path):
            print(f"Preprocessing {training_task} dataset for eval...")
            env = {**os.environ, "PYTHONPATH": "/workspace"}
            subprocess.run(
                [sys.executable, "data/preprocess_diagnostic_dataset.py", training_task,
                 "--train_n", "1280000", "--eval_n", "16000"],
                cwd="/workspace", env=env, stderr=subprocess.STDOUT, check=True,
            )

    short_model_name = model_name.split("/")[-1]
    # train.py builds run_name as "{model_type}_{short_model_name}_seed{seed}" and then
    # appends "_seed{seed}" again in result_dir, producing a doubled seed suffix.
    # load_model_from_path expects model_name such that "{model_name}_seed{seed}" matches
    # the directory name, so we must pass the run_name (with one _seed already included).
    eval_model_name = f"{model_type}_{short_model_name}_seed{random_seed}"
    cmd = [
        sys.executable, "diagnostic_task_eval.py",
        training_task,
        eval_model_name,
        model_type,
        "--checkpoint", str(checkpoint),
        "--num_batches", str(num_batches),
        "--random_seed", str(random_seed),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Eval exited with code {result.returncode}")


@app.local_entrypoint()
def main(
    training_task: str = "vowel_removal",
    model_type: str = "MrT5",
    max_steps: int = 20,
    checkpoint: int = 0,
    eval_only: bool = False,
):
    """
    Entrypoint for modal run train_modal.py.

    Train:   modal run train_modal.py [--training-task X] [--model-type Y] [--max-steps N]
    Eval:    modal run train_modal.py --eval-only --checkpoint N [--training-task X] [--model-type Y]
    """
    if eval_only:
        ckpt = checkpoint if checkpoint > 0 else max_steps
        evaluate.remote(
            training_task=training_task,
            model_type=model_type,
            checkpoint=ckpt,
        )
    else:
        train.remote(
            training_task=training_task,
            model_type=model_type,
            max_steps=max_steps,
        )