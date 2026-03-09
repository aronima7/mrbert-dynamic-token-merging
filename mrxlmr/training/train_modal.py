"""
Run MrXLMR training on Modal (serverless GPU).

Prerequisites:
  pip install modal
  modal token set  # one-time auth

Quick test (short run):
  modal run train_modal.py::main

Train MrXLMR on SNLI (with delete gate):
  modal run --detach train_modal.py::main --model-type MrXLMR --max-steps 30000

Train XLM-R baseline on SNLI (no delete gate):
  modal run --detach train_modal.py::main --model-type XLMR --max-steps 30000

Train random deletion baseline:
  modal run --detach train_modal.py::main --model-type MrXLMR --deletion-type random --target-deletion-rate 0.3 --max-steps 30000

Download checkpoints when done:
  modal volume get mrxlmr-checkpoints checkpoints ./local_mrxlmr_checkpoints
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
    "--learning_rate", "2e-5",
    "--target_deletion_rate", "0.3",
    "--controller_p", "0.01",
    "--controller_i", "0.00001",
    "--max_seq_length", "128",
    "--logging_steps", "50",
    "--eval_steps", "500",
    "--save_steps", "1000",
]

app = modal.App("mrxlmr-train")

volume = modal.Volume.from_name("mrxlmr-checkpoints", create_if_missing=True)

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
        "matplotlib",
        "sentencepiece",  # required for XLM-R tokenizer
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), ".."),
        remote_path="/workspace",
        ignore=[
            "snli_datasets",
            "squad_datasets",
            "sst2_datasets",
            "mrpc_datasets",
            "imdb_datasets",
            "tydiqa_datasets",
            "xnli_datasets",
            "local_checkpoints",
            "wandb",
            ".idea",
            ".git",
            "__pycache__",
            "*.pyc",
            "test_snli_mrxlmr",
            "test_squad_mrxlmr",
            "test_sst2_mrxlmr",
            "test_mrpc_mrxlmr",
            "test_imdb_mrxlmr",
            "test_tydiqa_mrxlmr",
            "test_xnli_mrxlmr",
            "*.safetensors",
            "*.bin",
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
    model_type: str = "MrXLMR",
    max_steps: int = 20,
    num_epochs: int = 3,
    target_deletion_rate: float = 0.3,
    mode: str = "training-only",
    use_pi_controller: bool = True,
    delete_gate_layer: int = 3,
    wandb_run_name: str = "",
    wandb_project: str = "",
    controller_p: float = 0.01,
    use_softmax1: bool = True,
    task: str = "sequence_classification",
    dataset_name: str = "",
    deletion_type: str = "scaled_sigmoid",
    hard_delete_train_prob: float = 0.0,
    deletion_loss_weight: float = 0.1,
    batch_size: int = 32,
    regularizer_delay: int = 1000,
    bypass_gate: bool = False,
    use_gumbel_noise: bool = True,
    use_pre_deletion_blend: bool = True,
    extra_args: list[str] | None = None,
):
    """
    Run train_mrxlmr.py on Modal with GPU.

    Args:
        output_dir: Where to save checkpoints (default: /checkpoints, persisted to Volume).
        model_type: "MrXLMR" (with delete gate) or "XLMR" (baseline).
        max_steps: Override default max_steps (use -1 for full training).
        num_epochs: Number of training epochs (default: 3).
        target_deletion_rate: Fraction of tokens to delete (default: 0.3).
        mode: "training-only", "eval-only", or "training-and-eval".
        use_pi_controller: Dynamically adjust deletion loss weight to hit target rate.
        delete_gate_layer: Which encoder layer (0-indexed) gets the delete gate.
        wandb_run_name: W&B run name (default: auto-generated).
        wandb_project: W&B project name (default: "mrxlmr").
        controller_p: Proportional gain for PI controller.
        use_softmax1: Use softmax1 attention (recommended by MrT5 paper).
        task: "sequence_classification" or "question_answering".
        dataset_name: Dataset to use. Empty -> task default (local_snli / local_squad).
        deletion_type: Gate type — "scaled_sigmoid", "random", or "fixed".
        hard_delete_train_prob: Probability of hard deletion per training step.
        deletion_loss_weight: Initial α₀ for deletion loss.
        batch_size: Training batch size.
        regularizer_delay: Steps before deletion pressure is applied.
        bypass_gate: Skip the delete gate entirely (0%-deletion control).
        use_gumbel_noise: Add Gumbel noise to gate logits.
        use_pre_deletion_blend: Blend pre-deletion hidden states for deleted tokens.
        extra_args: Optional list of extra CLI args.
    """
    os.chdir("/workspace/training")

    if not dataset_name:
        dataset_name = "local_snli" if task == "sequence_classification" else "local_squad"

    run_label = wandb_run_name or f"mrxlmr-{model_type.lower()}-{dataset_name}"
    run_output_dir = f"{output_dir}/{run_label}"
    print("=" * 60)
    print(f"RUN: {run_label}")
    print(f"  output_dir={run_output_dir}")
    print(f"  model_type={model_type}, task={task}, dataset={dataset_name}")
    print(f"  epochs={num_epochs}, max_steps={max_steps}")
    print(f"  target_deletion_rate={target_deletion_rate}, delete_gate_layer={delete_gate_layer}")
    print(f"  deletion_type={deletion_type}, hard_delete_train_prob={hard_delete_train_prob}")
    print(f"  mode={mode}, pi_controller={use_pi_controller}")
    print("=" * 60)

    DATASET_CONFIGS = {
        "local_snli":   ("/checkpoints/snli_datasets",   "snli-validation.json",   "preprocess_snli.py",   []),
        "local_squad":  ("/checkpoints/squad_datasets",  "squad-validation.json",  "preprocess_squad.py",  []),
        "local_sst2":   ("/checkpoints/sst2_datasets",   "sst2-validation.json",   "preprocess_sst2.py",   []),
        "local_mrpc":   ("/checkpoints/mrpc_datasets",   "mrpc-validation.json",   "preprocess_mrpc.py",   []),
        "local_imdb":   ("/checkpoints/imdb_datasets",   "imdb-validation.json",   "preprocess_imdb.py",   []),
        "local_tydiqa": ("/checkpoints/tydiqa_datasets", "tydiqa-validation.json", "preprocess_tydiqa.py", []),
        "local_xnli":   ("/checkpoints/xnli_datasets",   "xnli-validation.json",   "preprocess_xnli.py",   []),
    }

    if dataset_name in DATASET_CONFIGS:
        dataset_dir, sentinel, preprocess_script, extra_preprocess = DATASET_CONFIGS[dataset_name]
        check_path = f"{dataset_dir}/{sentinel}"
        if not os.path.exists(check_path):
            print(f"Preprocessing {dataset_name} dataset...")
            os.makedirs(dataset_dir, exist_ok=True)
            preprocess_cmd = [
                sys.executable, f"/workspace/data/{preprocess_script}",
                "--output_dir", dataset_dir,
                *extra_preprocess,
            ]
            env = {**os.environ, "PYTHONPATH": "/workspace"}
            result = subprocess.run(preprocess_cmd, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"{preprocess_script} exited with code {result.returncode}")
        else:
            print(f"Dataset already exists at {check_path}, skipping preprocessing.")

    DATASET_DIR_ARGS = {
        "local_snli":   ("--local_snli_dir",   "/checkpoints/snli_datasets"),
        "local_squad":  ("--local_squad_dir",  "/checkpoints/squad_datasets"),
        "local_sst2":   ("--local_sst2_dir",   "/checkpoints/sst2_datasets"),
        "local_mrpc":   ("--local_mrpc_dir",   "/checkpoints/mrpc_datasets"),
        "local_imdb":   ("--local_imdb_dir",   "/checkpoints/imdb_datasets"),
        "local_tydiqa": ("--local_tydiqa_dir", "/checkpoints/tydiqa_datasets"),
        "local_xnli":   ("--local_xnli_dir",   "/checkpoints/xnli_datasets"),
    }
    dir_flag, dir_val = DATASET_DIR_ARGS.get(dataset_name, (None, None))
    task_args = ["--task", task, "--dataset_name", dataset_name]
    if dir_flag:
        task_args += [dir_flag, dir_val]
    if task == "question_answering" or dataset_name in ("local_squad", "local_tydiqa"):
        task_args += ["--max_seq_length", "384"]

    # Filter DEFAULT_ARGS to remove task/dataset keys (replaced by task_args)
    default_args_filtered = []
    skip_next = False
    task_keys = {
        "--task", "--dataset_name", "--max_seq_length",
        "--batch_size", "--regularizer_delay",
        "--local_snli_dir", "--local_squad_dir",
        "--local_sst2_dir", "--local_mrpc_dir", "--local_imdb_dir", "--local_tydiqa_dir",
        "--local_xnli_dir",
    }
    for arg in DEFAULT_ARGS:
        if skip_next:
            skip_next = False
            continue
        if arg in task_keys:
            skip_next = True
            continue
        default_args_filtered.append(arg)

    cmd = [
        sys.executable,
        "train_mrxlmr.py",
        *default_args_filtered,
        *task_args,
        "--model_type", model_type,
        "--output_dir", run_output_dir,
        "--max_steps", str(max_steps),
        "--num_epochs", str(num_epochs),
        "--target_deletion_rate", str(target_deletion_rate),
        "--mode", mode,
        "--delete_gate_layer", str(delete_gate_layer),
        "--deletion_type", deletion_type,
        "--deletion_loss_weight", str(deletion_loss_weight),
        "--batch_size", str(batch_size),
        "--regularizer_delay", str(regularizer_delay),
    ]
    if hard_delete_train_prob > 0.0:
        cmd.extend(["--hard_delete_train_prob", str(hard_delete_train_prob)])
    if not use_pi_controller:
        cmd.append("--no_use_pi_controller")
    if bypass_gate:
        cmd.append("--bypass_gate")
    if not use_gumbel_noise:
        cmd.append("--no_use_gumbel_noise")
    if not use_pre_deletion_blend:
        cmd.append("--no_use_pre_deletion_blend")
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])
    if wandb_project:
        cmd.extend(["--wandb_project", wandb_project])
    cmd.extend(["--controller_p", str(controller_p)])
    if not use_softmax1:
        cmd.append("--no_use_softmax1")
    if extra_args:
        cmd.extend(extra_args)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)

    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")

    print("Checkpoints saved to Volume 'mrxlmr-checkpoints'.")
    print("Download with: modal volume get mrxlmr-checkpoints <remote_path> <local_path>")


@app.local_entrypoint()
def main(
    model_type: str = "MrXLMR",
    max_steps: int = 20,
    num_epochs: int = 3,
    target_deletion_rate: float = 0.3,
    mode: str = "training-only",
    task: str = "sequence_classification",
    dataset_name: str = "",
    deletion_type: str = "scaled_sigmoid",
    hard_delete_train_prob: float = 0.0,
    deletion_loss_weight: float = 0.1,
    batch_size: int = 32,
    regularizer_delay: int = 1000,
    delete_gate_layer: int = 3,
    wandb_run_name: str = "",
    wandb_project: str = "mrxlmr",
    controller_p: float = 0.01,
    bypass_gate: bool = False,
    use_gumbel_noise: bool = True,
    use_pre_deletion_blend: bool = True,
    use_softmax1: bool = True,
):
    """
    Main entrypoint for: modal run train_modal.py

    Quick test (20 steps):
      modal run train_modal.py

    Full SNLI run:
      modal run --detach train_modal.py --max-steps -1 --model-type MrXLMR --target-deletion-rate 0.3

    XLMR baseline:
      modal run --detach train_modal.py --model-type XLMR --max-steps -1

    SQuAD:
      modal run --detach train_modal.py --task question_answering --dataset-name local_squad \
          --batch-size 16 --regularizer-delay 500 --max-steps -1
    """
    train.remote(
        model_type=model_type,
        max_steps=max_steps,
        num_epochs=num_epochs,
        target_deletion_rate=target_deletion_rate,
        mode=mode,
        task=task,
        dataset_name=dataset_name,
        deletion_type=deletion_type,
        hard_delete_train_prob=hard_delete_train_prob,
        deletion_loss_weight=deletion_loss_weight,
        batch_size=batch_size,
        regularizer_delay=regularizer_delay,
        delete_gate_layer=delete_gate_layer,
        wandb_run_name=wandb_run_name,
        wandb_project=wandb_project,
        controller_p=controller_p,
        bypass_gate=bypass_gate,
        use_gumbel_noise=use_gumbel_noise,
        use_pre_deletion_blend=use_pre_deletion_blend,
        use_softmax1=use_softmax1,
    )