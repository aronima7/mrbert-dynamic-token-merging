"""
Run MrBERT training on GCP Vertex AI with A100 GPUs.

Prerequisites:
  pip install google-cloud-aiplatform google-cloud-storage
  gcloud auth login
  gcloud auth configure-docker <REGION>-docker.pkg.dev

One-time setup:
  1. Create the GCS bucket:
       gsutil mb -l <REGION> gs://<BUCKET_NAME>
  2. Build and push the Docker image:
       python train_gcp.py --project PROJECT --region REGION --bucket BUCKET --build-image

Train MrBERT on SNLI (with delete gate):
  python train_gcp.py --project PROJECT --region REGION --bucket BUCKET --model-type MrBERT --max-steps 30000

Train BERT baseline (no delete gate):
  python train_gcp.py --project PROJECT --region REGION --bucket BUCKET --model-type BERT --max-steps 30000

Download checkpoints when done:
  gsutil -m cp -r gs://<BUCKET>/checkpoints/<RUN_NAME> ./local_mrbert_checkpoints
"""

import argparse
import os
import subprocess
import sys

# =============================================================================
# Constants
# =============================================================================
IMAGE_REPO = "mrbert"       # Artifact Registry repo name
IMAGE_NAME = "mrbert-train"

# Training defaults (mirrors train_modal.py DEFAULT_ARGS)
DEFAULT_TRAIN_ARGS = [
    "--task", "sequence_classification",
    "--dataset_name", "local_snli",
    "--local_snli_dir", "/gcs/snli_datasets",
    "--num_epochs", "3",
    "--batch_size", "32",
    "--learning_rate", "2e-5",
    "--target_deletion_rate", "0.3",
    "--deletion_loss_weight", "0.01",
    "--controller_p", "0.01",
    "--controller_i", "0.00001",
    "--max_seq_length", "128",
    "--logging_steps", "50",
    "--save_steps", "1000",
    "--regularizer_delay", "1000",
]


# =============================================================================
# Image build & push
# =============================================================================

def build_and_push_image(project: str, region: str, repo: str, image_uri: str):
    """Build Docker image from mrbert/training/Dockerfile and push to Artifact Registry."""
    training_dir = os.path.dirname(os.path.abspath(__file__))
    mrbert_dir = os.path.dirname(training_dir)

    # Create Artifact Registry repo if it doesn't exist
    subprocess.run([
        "gcloud", "artifacts", "repositories", "create", repo,
        "--repository-format=docker",
        f"--location={region}",
        f"--project={project}",
        "--quiet",
    ], check=False)  # ignore error if repo already exists

    print(f"Building image: {image_uri}")
    subprocess.run([
        "docker", "build",
        "-f", os.path.join(training_dir, "Dockerfile"),
        "-t", image_uri,
        mrbert_dir,   # build context is mrbert/ root
    ], check=True)

    print(f"Pushing image: {image_uri}")
    subprocess.run(["docker", "push", image_uri], check=True)
    print("Image pushed successfully.")


# =============================================================================
# SNLI preprocessing on GCS
# =============================================================================

def preprocess_snli_to_gcs(project: str, region: str, image_uri: str, gcs_bucket: str):
    """Run SNLI preprocessing as a short Vertex AI job, saving output to GCS."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    gcs_output = f"gs://{gcs_bucket}/snli_datasets"
    job = aiplatform.CustomContainerTrainingJob(
        display_name="mrbert-preprocess-snli",
        container_uri=image_uri,
    )
    job.run(
        args=[
            "python", "/workspace/data/preprocess_snli.py",
            "--output_dir", "/gcs/snli_datasets",
        ],
        environment_variables={
            "PYTHONPATH": "/workspace",
            "AIP_MODEL_DIR": gcs_output,
        },
        base_output_dir=gcs_output,
        machine_type="n1-standard-4",
        accelerator_type=None,
        sync=True,
    )
    print(f"SNLI data saved to {gcs_output}")


# =============================================================================
# Submit training job
# =============================================================================

def submit_training_job(
    project: str,
    region: str,
    image_uri: str,
    gcs_bucket: str,
    model_type: str,
    max_steps: int,
    run_name: str,
    extra_args: list[str],
    wandb_api_key: str | None,
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    gcs_output_dir = f"gs://{gcs_bucket}/checkpoints/{run_name}"
    snli_dir = f"gs://{gcs_bucket}/snli_datasets"

    train_cmd = [
        "bash", "-c",
        # Mount GCS bucket via gcsfuse, then run training
        f"mkdir -p /gcs && gcsfuse --implicit-dirs {gcs_bucket} /gcs && "
        f"cd /workspace/training && "
        f"python train_mrbert.py "
        + " ".join([
            *DEFAULT_TRAIN_ARGS,
            "--model_type", model_type,
            "--output_dir", f"/gcs/checkpoints/{run_name}",
            "--max_steps", str(max_steps),
            "--local_snli_dir", "/gcs/snli_datasets",
            *extra_args,
        ])
    ]

    env_vars = {"PYTHONPATH": "/workspace"}
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key

    print(f"Submitting Vertex AI job: {run_name}")
    print(f"  Model type: {model_type}")
    print(f"  Max steps:  {max_steps}")
    print(f"  Checkpoints: {gcs_output_dir}")

    job = aiplatform.CustomContainerTrainingJob(
        display_name=run_name,
        container_uri=image_uri,
    )
    job.run(
        args=train_cmd,
        environment_variables=env_vars,
        base_output_dir=gcs_output_dir,
        machine_type="a2-highgpu-1g",   # 1x A100 40GB
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
        timeout=86400,   # 24 hours
        sync=False,      # detached — returns immediately
    )
    print(f"\nJob submitted. Monitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project}")
    print(f"\nDownload checkpoints when done:")
    print(f"  gsutil -m cp -r {gcs_output_dir} ./local_mrbert_checkpoints")


# =============================================================================
# Entrypoint
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Submit MrBERT training to Vertex AI.")

    # Required GCP config
    parser.add_argument("--project", type=str, required=True, help="GCP project ID.")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region (default: us-central1).")
    parser.add_argument("--bucket", type=str, required=True, help="GCS bucket name for checkpoints (without gs://).")

    parser.add_argument("--image-repo", type=str, default=IMAGE_REPO, help="Artifact Registry repo name (default: mrbert).")
    parser.add_argument(
        "--model-type", type=str, default="MrBERT", choices=["MrBERT", "BERT"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=30000,
        help="Maximum training steps.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Job display name and checkpoint subdirectory. Auto-generated if not set.",
    )
    parser.add_argument(
        "--build-image", action="store_true",
        help="Build and push the Docker image before submitting the job.",
    )
    parser.add_argument(
        "--preprocess-snli", action="store_true",
        help="Run SNLI preprocessing job on Vertex AI before training.",
    )
    parser.add_argument(
        "--wandb-api-key", type=str, default=os.environ.get("WANDB_API_KEY"),
        help="W&B API key (defaults to $WANDB_API_KEY env var).",
    )
    parser.add_argument(
        "extra_args", nargs=argparse.REMAINDER,
        help="Extra args forwarded to train_mrbert.py, e.g. --batch_size 64",
    )
    args = parser.parse_args()

    # Derive image URI from CLI args
    image_uri = f"{args.region}-docker.pkg.dev/{args.project}/{args.image_repo}/{IMAGE_NAME}:latest"

    if args.build_image:
        build_and_push_image(args.project, args.region, args.image_repo, image_uri)

    if args.preprocess_snli:
        preprocess_snli_to_gcs(args.project, args.region, image_uri, args.bucket)

    run_name = args.run_name or f"mrbert-{args.model_type.lower()}-{args.max_steps}steps"

    submit_training_job(
        project=args.project,
        region=args.region,
        image_uri=image_uri,
        gcs_bucket=args.bucket,
        model_type=args.model_type,
        max_steps=args.max_steps,
        run_name=run_name,
        extra_args=args.extra_args,
        wandb_api_key=args.wandb_api_key,
    )


if __name__ == "__main__":
    main()
