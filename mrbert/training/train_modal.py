"""
Run MrBERT training on Modal (serverless GPU).

Prerequisites:
  pip install modal
  modal token set  # one-time auth

Quick test (short run):
  modal run train_modal.py::main

Train MrBERT on SNLI (with delete gate):
  modal run --detach train_modal.py::main --model-type MrBERT --max-steps 30000

Train BERT baseline on SNLI (no delete gate):
  modal run --detach train_modal.py::main --model-type BERT --max-steps 30000

Train random deletion baseline (same rate, no learned gate):
  modal run --detach train_modal.py::main --model-type MrBERT --deletion-type random --target-deletion-rate 0.3 --max-steps 30000

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
    "--learning_rate", "2e-5",
    "--target_deletion_rate", "0.3",
    "--controller_p", "0.01",           # low proportional gain to prevent rapid α ramp-up
    "--controller_i", "0.00001",        # slow integral ramp to prevent gate collapse
    "--max_seq_length", "128",
    "--logging_steps", "50",
    "--save_steps", "1000",
    # batch_size and regularizer_delay are NOT here — they are passed explicitly
    # per dataset via the batch_size / regularizer_delay function parameters.
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
        "matplotlib",
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
            "mrbert_checkpoints",
            "mrbert_sst2",
            "mrbert_squad_test",
            "local_checkpoints",
            "test_imdb_mrbert",
            "test_mrpc_mrbert",
            "test_sst2_mrbert",
            "test_tydiqa_mrbert",
            "wandb",
            ".idea",
            ".git",
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
        wandb_project: W&B project name (default: "mrbert"). Use to separate runs by dataset.
        controller_p: Proportional gain for PI controller (default: 0.01).
        use_softmax1: Use softmax1 attention, as recommended by MrT5 paper (default: True).
        task: Training task — "sequence_classification" or "question_answering".
        dataset_name: Dataset to use. Empty string → task default (local_snli / local_squad).
            Supported values: local_snli, local_squad, local_sst2, local_mrpc, local_imdb, local_tydiqa.
        deletion_type: Gate type — "scaled_sigmoid" (learned), "random", or "fixed".
        hard_delete_train_prob: Probability of hard deletion per training step (default: 0.0).
        deletion_loss_weight: Initial α₀ for deletion loss (default: 0.1).
        batch_size: Training (and eval) batch size (default: 32). Use 16 for 384–512 token tasks.
        regularizer_delay: Steps before deletion pressure is applied (default: 1000). Scale down
            proportionally for small datasets (e.g. 100 for MRPC, 300 for SST-2/IMDB).
        extra_args: Optional list of extra CLI args, e.g. ["--learning_rate", "3e-5"].
    """
    os.chdir("/workspace/training")

    # Resolve dataset_name: empty string → task default
    if not dataset_name:
        dataset_name = "local_snli" if task == "sequence_classification" else "local_squad"

    run_label = wandb_run_name or f"mrbert-{model_type.lower()}-{dataset_name}"
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

    # Dataset configs: (checkpoint_dir, sentinel_file, preprocess_script, extra_preprocess_args)
    DATASET_CONFIGS = {
        "local_snli":   ("/checkpoints/snli_datasets",   "snli-train.json",   "preprocess_snli.py",   []),
        "local_squad":  ("/checkpoints/squad_datasets",  "squad-train.json",  "preprocess_squad.py",  []),
        "local_sst2":   ("/checkpoints/sst2_datasets",   "sst2-train.json",   "preprocess_sst2.py",   []),
        "local_mrpc":   ("/checkpoints/mrpc_datasets",   "mrpc-train.json",   "preprocess_mrpc.py",   []),
        "local_imdb":   ("/checkpoints/imdb_datasets",   "imdb-train.json",   "preprocess_imdb.py",   []),
        "local_tydiqa": ("/checkpoints/tydiqa_datasets", "tydiqa-train.json", "preprocess_tydiqa.py", []),
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

    # Map dataset_name → the --local_*_dir CLI flag and its value
    DATASET_DIR_ARGS = {
        "local_snli":   ("--local_snli_dir",   "/checkpoints/snli_datasets"),
        "local_squad":  ("--local_squad_dir",  "/checkpoints/squad_datasets"),
        "local_sst2":   ("--local_sst2_dir",   "/checkpoints/sst2_datasets"),
        "local_mrpc":   ("--local_mrpc_dir",   "/checkpoints/mrpc_datasets"),
        "local_imdb":   ("--local_imdb_dir",   "/checkpoints/imdb_datasets"),
        "local_tydiqa": ("--local_tydiqa_dir", "/checkpoints/tydiqa_datasets"),
    }
    dir_flag, dir_val = DATASET_DIR_ARGS.get(dataset_name, (None, None))
    task_args = ["--task", task, "--dataset_name", dataset_name]
    if dir_flag:
        task_args += [dir_flag, dir_val]
    # QA tasks need a longer max_seq_length for context passages
    if task == "question_answering" or dataset_name == "local_tydiqa":
        task_args += ["--max_seq_length", "384"]

    # Strip task/dataset keys from DEFAULT_ARGS (replaced by task_args above)
    task_keys = {
        "--task", "--dataset_name", "--max_seq_length",
        "--batch_size", "--regularizer_delay",
        "--local_snli_dir", "--local_squad_dir",
        "--local_sst2_dir", "--local_mrpc_dir", "--local_imdb_dir", "--local_tydiqa_dir",
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
        "train_mrbert.py",
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

    # Persist volume regardless of outcome to preserve any partial checkpoints
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")

    print("Checkpoints saved to Volume 'mrbert-checkpoints'.")
    print("Download with: modal volume get mrbert-checkpoints <remote_path> <local_path>")


# =============================================================================
# Runtime benchmark (runs measure_runtime.py on A100 against saved checkpoints)
# =============================================================================

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def benchmark(
    deletion_percentage_models: str = "",
    deletion_gate_layer_models: str = "",
    n_warmup: int = 10,
    n_timed: int = 100,
    batch_size: int = 32,
    output_subdir: str = "analysis_figures",
):
    """
    Run measure_runtime.py on the A100 against checkpoints stored in the volume.

    deletion_percentage_models / deletion_gate_layer_models are comma-separated LABEL:run_name pairs, e.g.:
      "BERT:bert-snli-baseline,MrBERT-30%:mrbert-snli-30pct"

    Checkpoint paths are resolved as /checkpoints/<run_name>/final.
    Results (runtime_table.csv, runtime_vs_deletion.pdf) are saved to
    /checkpoints/<output_subdir>/ and committed to the volume for download.
    """
    import sys
    sys.path.insert(0, "/workspace/models")
    sys.path.insert(0, "/workspace/analysis")

    output_dir = f"/checkpoints/{output_subdir}"
    os.makedirs(output_dir, exist_ok=True)

    def parse_model_specs(spec_str):
        """Parse 'LABEL:run_name,...' into [('LABEL', '/checkpoints/run_name/final'), ...]."""
        if not spec_str.strip():
            return []
        entries = []
        for part in spec_str.split(","):
            part = part.strip()
            if ":" not in part:
                print(f"Warning: skipping malformed model spec {part!r} (expected LABEL:run_name)")
                continue
            label, run_name = part.split(":", 1)
            path = f"/checkpoints/{run_name.strip()}/final"
            if not os.path.exists(path):
                print(f"Warning: checkpoint not found at {path} — skipping {label}")
                continue
            entries.append((label.strip(), path))
        return entries

    deletion_percentage_entries  = parse_model_specs(deletion_percentage_models)
    deletion_gate_layer_entries  = parse_model_specs(deletion_gate_layer_models)

    def run_benchmark(model_entries, label, csv_name, pdf_name):
        if not model_entries:
            print(f"No valid models for {label} benchmark — skipping.")
            return
        print(f"\n{'='*60}")
        print(f"Benchmarking {label} ({len(model_entries)} models) on A100")
        print(f"{'='*60}")

        # Import here so sys.path insertions above are in effect
        from measure_runtime import load_model, load_snli_batch, measure_runtime, print_and_save_table, plot_runtime_vs_deletion
        import torch
        from transformers import BertTokenizer

        device = "cuda"
        max_seq_length = 128

        tokenizer = BertTokenizer.from_pretrained(model_entries[0][1])
        dataset = load_snli_batch(
            local_snli_dir="/checkpoints/snli_datasets",
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            n_samples=512,
        )

        all_results = []
        for model_label, model_path in model_entries:
            print(f"\nLoading {model_label} from {model_path} ...")
            model = load_model(model_path, device)
            stats = measure_runtime(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                n_warmup=n_warmup,
                n_timed=n_timed,
                device=device,
                hard_delete=True,
            )
            ms   = stats["mean_ms_per_sample"]
            std  = stats["std_ms_per_sample"]
            seq  = stats["mean_seq_len_after"]
            orig = stats["original_seq_len"]
            print(f"  {model_label}: {ms:.2f} ± {std:.2f} ms/sample", end="")
            if seq is not None:
                print(f"  |  seq reduction: {(1 - seq/orig)*100:.1f}%", end="")
            print()
            all_results.append({
                "label": model_label,
                "mean_ms": ms,
                "std_ms": std,
                "mean_new_seq_len": seq,
            })
            del model
            torch.cuda.empty_cache()

        baseline_ms = all_results[0]["mean_ms"]
        print_and_save_table(all_results, baseline_ms, max_seq_length, output_dir, filename=csv_name)
        plot_runtime_vs_deletion(all_results, baseline_ms, output_dir, filename=pdf_name)

    run_benchmark(deletion_percentage_entries, "SNLI deletion percentage ablation",
                  csv_name="snli_runtime_table_deletion_percentage.csv",
                  pdf_name="snli_runtime_vs_deletion_percentage.pdf")
    run_benchmark(deletion_gate_layer_entries, "SNLI deletion gate layer ablation",
                  csv_name="snli_runtime_table_deletion_gate_layer.csv",
                  pdf_name="snli_runtime_vs_deletion_gate_layer.pdf")

    volume.commit()
    print(f"\nResults saved to volume at '{output_subdir}/'")
    print(f"Download with: modal volume get mrbert-checkpoints {output_subdir} ./analysis/figures")


@app.local_entrypoint()
def benchmark_main(
    deletion_percentage_models: str = "BERT:bert-snli-baseline,MrBERT-0%:mrbert-snli-0pct,MrBERT-30%:mrbert-snli-30pct,MrBERT-30%-HD:mrbert-snli-30pct-hd,MrBERT-50%:mrbert-snli-50pct,MrBERT-70%:mrbert-snli-70pct,Random-30%:mrbert-snli-random30",
    deletion_gate_layer_models: str = "Layer-1:mrbert-snli-layer1,Layer-3:mrbert-snli-30pct,Layer-6:mrbert-snli-layer6,Layer-9:mrbert-snli-layer9",
    n_warmup: int = 10,
    n_timed: int = 100,
    batch_size: int = 32,
    output_subdir: str = "analysis_figures",
):
    """
    Entrypoint for `modal run train_modal.py::benchmark_main`.
    Runs inference timing on A100 against all trained checkpoints in the volume.
    Defaults benchmark all SNLI runs; override with --deletion-percentage-models and --deletion-gate-layer-models.
    """
    benchmark.remote(
        deletion_percentage_models=deletion_percentage_models,
        deletion_gate_layer_models=deletion_gate_layer_models,
        n_warmup=n_warmup,
        n_timed=n_timed,
        batch_size=batch_size,
        output_subdir=output_subdir,
    )


@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600 * 4,
)
def hard_deletion_curve(
    run_name: str = "mrbert-snli-30pct",
    output_subdir: str = "analysis_figures",
    n_samples: int = 9824,
    batch_size: int = 64,
):
    """
    Post-hoc evaluation of soft vs hard deletion accuracy across training checkpoints.
    Loads each checkpoint-* and final/ from /checkpoints/<run_name>/,
    evaluates on the SNLI test set under both soft and hard deletion,
    and saves a CSV + PDF to /checkpoints/<output_subdir>/.

    Download results:
      modal volume get mrbert-checkpoints <output_subdir>/<run_name>_hard_deletion_curve.csv ./analysis/figures/<run_name>_hard_deletion_curve.csv
      modal volume get mrbert-checkpoints <output_subdir>/<run_name>_hard_deletion_curve.pdf ./analysis/figures/<run_name>_hard_deletion_curve.pdf
    """
    import sys
    sys.path.insert(0, "/workspace/models")
    sys.path.insert(0, "/workspace/analysis")

    checkpoint_dir = f"/checkpoints/{run_name}"
    output_dir = f"/checkpoints/{output_subdir}"
    os.makedirs(output_dir, exist_ok=True)

    from hard_deletion_curve import find_checkpoints, load_snli_test, evaluate_accuracy, save_csv, plot_curve
    import torch
    from transformers import BertTokenizer

    device = "cuda"
    max_seq_length = 128

    print(f"Checkpoint dir: {checkpoint_dir}")
    checkpoints = find_checkpoints(checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    tokenizer = BertTokenizer.from_pretrained(checkpoints[0][1])
    dataset = load_snli_test(
        local_snli_dir="/checkpoints/snli_datasets",
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        n_samples=n_samples,
    )

    import json
    from configuration_mrbert import MrBertConfig
    from modeling_mrbert import MrBertForSequenceClassification

    results = []
    n = len(checkpoints)
    for i, (step, ckpt_path) in enumerate(checkpoints):
        step_label = "final" if i == n - 1 and os.path.basename(ckpt_path) == "final" else str(step)
        print(f"\n[{i+1}/{n}] {step_label}  ({ckpt_path})")

        with open(os.path.join(ckpt_path, "config.json")) as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "mrbert":
            config = MrBertConfig.from_pretrained(ckpt_path)
            model = MrBertForSequenceClassification.from_pretrained(ckpt_path, config=config)
        else:
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(ckpt_path)
        model.to(device)

        soft_acc = evaluate_accuracy(model, dataset, batch_size, device, hard_delete=False)
        hard_acc = evaluate_accuracy(model, dataset, batch_size, device, hard_delete=True)
        print(f"  soft={soft_acc:.4f}  hard={hard_acc:.4f}  gap={abs(soft_acc-hard_acc)*100:.2f}pp")

        results.append({
            "step": step,
            "step_label": step_label,
            "soft_acc": soft_acc,
            "hard_acc": hard_acc,
        })
        del model
        torch.cuda.empty_cache()

    csv_path = os.path.join(output_dir, f"{run_name}_hard_deletion_curve.csv")
    pdf_path = os.path.join(output_dir, f"{run_name}_hard_deletion_curve.pdf")
    save_csv(results, csv_path)
    plot_curve(results, pdf_path, run_name)

    volume.commit()
    print(f"\nResults saved to volume at '{output_subdir}/'")
    print(f"Download CSV: modal volume get mrbert-checkpoints {output_subdir}/{run_name}_hard_deletion_curve.csv ./analysis/figures/{run_name}_hard_deletion_curve.csv")
    print(f"Download PDF: modal volume get mrbert-checkpoints {output_subdir}/{run_name}_hard_deletion_curve.pdf ./analysis/figures/{run_name}_hard_deletion_curve.pdf")


@app.local_entrypoint()
def hard_deletion_curve_main(
    run_name: str = "mrbert-snli-30pct",
    output_subdir: str = "analysis_figures",
    n_samples: int = 9824,
    batch_size: int = 64,
):
    """
    Entrypoint for `modal run training/train_modal.py::hard_deletion_curve_main`.
    Evaluates soft vs hard deletion accuracy across all saved checkpoints for a run.

    Example:
      modal run training/train_modal.py::hard_deletion_curve_main --run-name mrbert-snli-30pct
    """
    hard_deletion_curve.remote(
        run_name=run_name,
        output_subdir=output_subdir,
        n_samples=n_samples,
        batch_size=batch_size,
    )


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
):
    """
    Entrypoint for `modal run train_modal.py [--task ...] [--dataset-name ...] [--model-type MrBERT|BERT] ...`.

    --dataset-name values: local_snli (default for classification), local_squad (default for QA),
                           local_sst2, local_mrpc, local_imdb, local_tydiqa
    --batch-size: use 16 for IMDB (512 tokens) and TyDi QA (384 tokens); 32 for others
    --regularizer-delay: scale to ~5-10% of total training steps per dataset
    """
    train.remote(
        output_dir="/checkpoints",
        model_type=model_type,
        max_steps=max_steps,
        num_epochs=num_epochs,
        target_deletion_rate=target_deletion_rate,
        mode=mode,
        use_pi_controller=use_pi_controller,
        delete_gate_layer=delete_gate_layer,
        wandb_run_name=wandb_run_name,
        wandb_project=wandb_project,
        controller_p=controller_p,
        use_softmax1=use_softmax1,
        task=task,
        dataset_name=dataset_name,
        deletion_type=deletion_type,
        hard_delete_train_prob=hard_delete_train_prob,
        deletion_loss_weight=deletion_loss_weight,
        batch_size=batch_size,
        regularizer_delay=regularizer_delay,
    )
