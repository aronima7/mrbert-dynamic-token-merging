"""
Run delta-loss correlation analysis on Modal (serverless GPU).

This runs analyze_delta_correlation.py against checkpoints stored in the
Modal volume 'mrbert-checkpoints'.

Usage:
    # Run with defaults (SNLI, 1000 examples, all three models):
    modal run run_delta_analysis_modal.py

    # Custom settings:
    modal run run_delta_analysis_modal.py \
        --dataset snli --split validation --max-samples 2000

    # Download results when done:
    modal volume get mrbert-checkpoints delta_analysis ./delta_analysis_results

Checkpoint layout on volume (at /checkpoints/):
    bert-snli-baseline/final/     — baseline BERT (no gate)
    mrbert-snli-30pct/final/      — MrBERT with learned gate, 30% target
    mrbert-snli-random30/final/   — MrBERT with random gate, 30% deletion
"""

import os
import subprocess
import sys

import modal

app = modal.App("mrbert-delta-analysis")

volume = modal.Volume.from_name("mrbert-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0,<5.0.0",
        "datasets",
        "tqdm",
        "numpy<2",
        "matplotlib",
        "scipy",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__)),
        remote_path="/workspace",
        ignore=[
            "mrbert/local_checkpoints",
            "mrbert/mrbert_checkpoints",
            "mrbert/mrbert_squad_test",
            "mrbert/test_*",
            "mrxlmr/test_*",
            "mrdiffusion-sedd",
            "mrdiffusion-bd3lms",
            "diffusion",
            "mrt5",
            "wandb",
            ".git",
            "__pycache__",
            "*.pyc",
            "COLM-Paper-dynamic-token-merging",
            "final-project-report",
        ],
    )
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def run_analysis(
    baseline_run: str = "bert-snli-baseline",
    mrbert_run: str = "mrbert-snli-30pct",
    random_run: str = "mrbert-snli-random30",
    dataset: str = "snli",
    dataset_config: str = "",
    split: str = "validation",
    max_samples: int = 1000,
    output_name: str = "delta_analysis",
):
    """
    Run the delta-loss correlation analysis on Modal.

    Args:
        baseline_run: Name of baseline BERT run on the volume.
        mrbert_run: Name of learned-gate MrBERT run on the volume.
        random_run: Name of random-gate MrBERT run (set to "" to skip).
        dataset: HuggingFace dataset name.
        dataset_config: Dataset config (e.g., "sst2" for glue).
        split: Dataset split to evaluate.
        max_samples: Number of examples to analyze.
        output_name: Output directory name (saved to volume).
    """
    os.chdir("/workspace")

    # Resolve checkpoint paths
    baseline_path = f"/checkpoints/{baseline_run}/final"
    mrbert_path = f"/checkpoints/{mrbert_run}/final"
    random_path = f"/checkpoints/{random_run}/final" if random_run else ""

    # Verify checkpoints exist
    print("=" * 60)
    print("CHECKING CHECKPOINTS")
    print("=" * 60)

    for label, path in [("Baseline", baseline_path), ("MrBERT", mrbert_path)]:
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"  ✓ {label}: {path} ({len(files)} files)")
        else:
            raise FileNotFoundError(
                f"  ✗ {label} checkpoint not found at {path}\n"
                f"    Available runs: {os.listdir('/checkpoints/')}"
            )

    if random_path:
        if os.path.exists(random_path):
            files = os.listdir(random_path)
            print(f"  ✓ Random: {random_path} ({len(files)} files)")
        else:
            print(f"  ⚠ Random checkpoint not found at {random_path}, skipping.")
            random_path = ""

    # Build command
    output_dir = f"/checkpoints/{output_name}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "analyze_delta_correlation.py",
        "--baseline_path", baseline_path,
        "--mrbert_path", mrbert_path,
        "--dataset_name", dataset,
        "--split", split,
        "--max_samples", str(max_samples),
        "--output_dir", output_dir,
        "--batch_size", "32",
    ]

    if dataset_config:
        cmd.extend(["--dataset_config", dataset_config])

    if random_path:
        cmd.extend(["--random_path", random_path])

    print("\n" + "=" * 60)
    print("RUNNING ANALYSIS")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}\n")

    env = {**os.environ, "PYTHONPATH": "/workspace"}
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Analysis script exited with code {result.returncode}")

    # Commit results to volume
    volume.commit()

    print("\n" + "=" * 60)
    print("DONE — Results saved to volume")
    print("=" * 60)
    print(f"\nDownload results with:")
    print(f"  modal volume get mrbert-checkpoints {output_name} ./{output_name}")
    print(f"\nFiles produced:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f} ({size/1024:.1f} KB)")


@app.local_entrypoint()
def main(
    baseline_run: str = "bert-snli-baseline",
    mrbert_run: str = "mrbert-snli-30pct",
    random_run: str = "mrbert-snli-random30",
    dataset: str = "snli",
    dataset_config: str = "",
    split: str = "validation",
    max_samples: int = 1000,
    output_name: str = "delta_analysis",
):
    """Local entrypoint — dispatches to Modal GPU."""
    run_analysis.remote(
        baseline_run=baseline_run,
        mrbert_run=mrbert_run,
        random_run=random_run,
        dataset=dataset,
        dataset_config=dataset_config,
        split=split,
        max_samples=max_samples,
        output_name=output_name,
    )
