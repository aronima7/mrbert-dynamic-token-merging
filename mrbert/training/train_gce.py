"""
Run MrBERT training on a GCE VM with an A100 GPU.

This script creates a GCE VM, uploads the mrbert package, runs training,
downloads checkpoints, and optionally stops/deletes the VM when done.

Prerequisites:
  pip install google-cloud-compute google-cloud-storage
  gcloud auth login
  gcloud auth application-default login

Usage:
  # Create VM, train, download checkpoints (VM left stopped when done)
  python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000

  # Same but delete VM after training (saves cost)
  python train_gce.py --project YOUR_PROJECT --bucket YOUR_BUCKET --model-type MrBERT --max-steps 30000 --delete-vm

  # SSH into an existing VM
  python train_gce.py --project YOUR_PROJECT --ssh

  # Stop/delete an existing VM
  python train_gce.py --project YOUR_PROJECT --stop-vm
  python train_gce.py --project YOUR_PROJECT --delete-vm-only

Download checkpoints separately:
  gsutil -m cp -r gs://YOUR_BUCKET/checkpoints/RUN_NAME ./local_mrbert_checkpoints
"""

import argparse
import os
import subprocess
import sys
import time

# =============================================================================
# Constants
# =============================================================================
VM_NAME = "mrbert-training"
IMAGE_FAMILY = "pytorch-latest-gpu"
IMAGE_PROJECT = "deeplearning-platform-release"
MACHINE_TYPE = "a2-highgpu-1g"       # 1× A100 40GB
ACCELERATOR = "type=nvidia-tesla-a100,count=1"
BOOT_DISK_SIZE = "200GB"
REMOTE_WORKSPACE = "/home/mrbert"

# Training defaults (mirrors train_modal.py DEFAULT_ARGS)
DEFAULT_TRAIN_ARGS = [
    "--task", "sequence_classification",
    "--dataset_name", "local_snli",
    "--local_snli_dir", f"{REMOTE_WORKSPACE}/snli_datasets",
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
# Helpers
# =============================================================================

def gcloud(*args, check=True, capture=False):
    """Run a gcloud command."""
    cmd = ["gcloud", *args]
    print("$", " ".join(cmd))
    if capture:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result.stdout.strip()
    subprocess.run(cmd, check=check)


def gsutil(*args, check=True):
    subprocess.run(["gsutil", *args], check=check)


def ssh_run(project, zone, vm_name, remote_cmd, check=True):
    """Run a command on the VM over SSH."""
    subprocess.run([
        "gcloud", "compute", "ssh", vm_name,
        f"--project={project}",
        f"--zone={zone}",
        "--command", remote_cmd,
    ], check=check)


# =============================================================================
# VM lifecycle
# =============================================================================

def create_vm(project, zone, vm_name):
    """Create the GCE VM if it doesn't already exist."""
    # Check if VM already exists
    result = subprocess.run([
        "gcloud", "compute", "instances", "describe", vm_name,
        f"--project={project}", f"--zone={zone}", "--format=value(status)",
    ], capture_output=True, text=True)

    if result.returncode == 0:
        status = result.stdout.strip()
        if status == "RUNNING":
            print(f"VM '{vm_name}' already running.")
            return
        elif status == "TERMINATED":
            print(f"VM '{vm_name}' exists but is stopped — starting it.")
            gcloud("compute", "instances", "start", vm_name,
                   f"--project={project}", f"--zone={zone}")
            _wait_for_ssh(project, zone, vm_name)
            return

    print(f"Creating VM '{vm_name}'...")
    gcloud(
        "compute", "instances", "create", vm_name,
        f"--project={project}",
        f"--zone={zone}",
        f"--machine-type={MACHINE_TYPE}",
        f"--accelerator={ACCELERATOR}",
        f"--image-family={IMAGE_FAMILY}",
        f"--image-project={IMAGE_PROJECT}",
        f"--boot-disk-size={BOOT_DISK_SIZE}",
        "--maintenance-policy=TERMINATE",
        "--metadata=install-nvidia-driver=True",
    )
    print("Waiting for VM to be ready...")
    _wait_for_ssh(project, zone, vm_name)


def _wait_for_ssh(project, zone, vm_name, max_wait=300):
    """Poll until SSH is available."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        result = subprocess.run([
            "gcloud", "compute", "ssh", vm_name,
            f"--project={project}", f"--zone={zone}",
            "--command", "echo ready",
            "--ssh-flag=-o ConnectTimeout=5",
        ], capture_output=True)
        if result.returncode == 0:
            print("VM is ready.")
            return
        print("  SSH not ready yet, retrying...")
        time.sleep(10)
    raise TimeoutError(f"VM '{vm_name}' did not become SSH-accessible within {max_wait}s")


def stop_vm(project, zone, vm_name):
    print(f"Stopping VM '{vm_name}'...")
    gcloud("compute", "instances", "stop", vm_name,
           f"--project={project}", f"--zone={zone}")
    print("VM stopped. Billing has ceased.")


def delete_vm(project, zone, vm_name):
    print(f"Deleting VM '{vm_name}'...")
    gcloud("compute", "instances", "delete", vm_name,
           f"--project={project}", f"--zone={zone}", "--quiet")
    print("VM deleted.")


# =============================================================================
# Code upload
# =============================================================================

def upload_code(project, zone, vm_name):
    """Rsync the mrbert package to the VM."""
    training_dir = os.path.dirname(os.path.abspath(__file__))
    mrbert_dir = os.path.dirname(training_dir)

    print("Uploading mrbert code to VM...")
    # Create remote workspace
    ssh_run(project, zone, vm_name, f"mkdir -p {REMOTE_WORKSPACE}")

    subprocess.run([
        "gcloud", "compute", "scp",
        "--recurse",
        mrbert_dir,
        f"{vm_name}:{REMOTE_WORKSPACE}",
        f"--project={project}",
        f"--zone={zone}",
        "--compress",
    ], check=True)
    print("Code uploaded.")


# =============================================================================
# Environment setup
# =============================================================================

def setup_environment(project, zone, vm_name):
    """Install Python dependencies on the VM."""
    print("Installing dependencies on VM...")
    ssh_run(project, zone, vm_name,
        f"pip install --quiet "
        f"'transformers>=4.40.0,<5.0.0' datasets tqdm accelerate 'numpy<2' wandb google-cloud-storage"
    )
    print("Dependencies installed.")


# =============================================================================
# SNLI preprocessing
# =============================================================================

def preprocess_snli(project, zone, vm_name, gcs_bucket):
    """Download and preprocess SNLI on VM, upload to GCS."""
    snli_gcs = f"gs://{gcs_bucket}/snli_datasets/snli-train.json"

    # Check if already on GCS
    result = subprocess.run(["gsutil", "-q", "stat", snli_gcs], capture_output=True)
    if result.returncode == 0:
        print("SNLI already preprocessed on GCS, skipping.")
        # Download to VM
        ssh_run(project, zone, vm_name,
            f"mkdir -p {REMOTE_WORKSPACE}/snli_datasets && "
            f"gsutil -m cp gs://{gcs_bucket}/snli_datasets/* {REMOTE_WORKSPACE}/snli_datasets/"
        )
        return

    print("Preprocessing SNLI on VM...")
    ssh_run(project, zone, vm_name,
        f"PYTHONPATH={REMOTE_WORKSPACE} python {REMOTE_WORKSPACE}/data/preprocess_snli.py "
        f"--output_dir {REMOTE_WORKSPACE}/snli_datasets"
    )
    print("Uploading SNLI data to GCS...")
    ssh_run(project, zone, vm_name,
        f"gsutil -m cp {REMOTE_WORKSPACE}/snli_datasets/* gs://{gcs_bucket}/snli_datasets/"
    )
    print("SNLI data uploaded to GCS.")


# =============================================================================
# Training
# =============================================================================

def run_training(
    project, zone, vm_name, gcs_bucket,
    model_type, max_steps, run_name,
    extra_args, wandb_api_key,
):
    """Launch training on the VM, saving checkpoints to GCS on completion."""
    checkpoint_dir = f"{REMOTE_WORKSPACE}/checkpoints/{run_name}"

    train_cmd = " ".join([
        f"PYTHONPATH={REMOTE_WORKSPACE}",
        f"WANDB_API_KEY={wandb_api_key}" if wandb_api_key else "",
        f"python {REMOTE_WORKSPACE}/training/train_mrbert.py",
        *DEFAULT_TRAIN_ARGS,
        "--model_type", model_type,
        "--output_dir", checkpoint_dir,
        "--max_steps", str(max_steps),
        *extra_args,
    ])

    # Upload checkpoints to GCS after training
    upload_cmd = (
        f"gsutil -m cp -r {checkpoint_dir} gs://{gcs_bucket}/checkpoints/"
    )

    print(f"Starting training: {model_type}, {max_steps} steps")
    print(f"Checkpoints will be saved to gs://{gcs_bucket}/checkpoints/{run_name}")

    # Run training + upload in a single SSH call so it survives disconnection
    ssh_run(project, zone, vm_name,
        f"{train_cmd} && echo 'Training complete. Uploading checkpoints...' && {upload_cmd}"
    )
    print(f"\nCheckpoints uploaded to gs://{gcs_bucket}/checkpoints/{run_name}")
    print(f"Download with: gsutil -m cp -r gs://{gcs_bucket}/checkpoints/{run_name} ./local_mrbert_checkpoints")


# =============================================================================
# Entrypoint
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run MrBERT training on a GCE A100 VM.")

    # Required
    parser.add_argument("--project", type=str, required=True, help="GCP project ID.")
    parser.add_argument("--bucket", type=str, default=None, help="GCS bucket for checkpoints (without gs://). Required for training.")

    # Optional GCP config
    parser.add_argument("--zone", type=str, default="us-central1-c", help="GCP zone (default: us-central1-c).")
    parser.add_argument("--vm-name", type=str, default=VM_NAME, help=f"VM instance name (default: {VM_NAME}).")

    # Training config
    parser.add_argument("--model-type", type=str, default="MrBERT", choices=["MrBERT", "BERT"])
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Checkpoint subdirectory name. Auto-generated if not set.")
    parser.add_argument("--wandb-api-key", type=str, default=os.environ.get("WANDB_API_KEY"))

    # VM lifecycle flags
    parser.add_argument("--ssh", action="store_true", help="Open an interactive SSH session to the VM.")
    parser.add_argument("--stop-vm", action="store_true", help="Stop the VM after training (default).")
    parser.add_argument("--delete-vm", action="store_true", help="Delete the VM after training.")
    parser.add_argument("--delete-vm-only", action="store_true", help="Just delete the VM, don't train.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip code upload (use if code is already on VM).")
    parser.add_argument("--skip-setup", action="store_true", help="Skip pip install (use if deps are already installed).")

    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to train_mrbert.py.")
    args = parser.parse_args()

    # Handle VM-only operations
    if args.delete_vm_only:
        delete_vm(args.project, args.zone, args.vm_name)
        return

    if args.ssh:
        subprocess.run([
            "gcloud", "compute", "ssh", args.vm_name,
            f"--project={args.project}", f"--zone={args.zone}",
        ])
        return

    if args.bucket is None:
        parser.error("--bucket is required for training.")

    run_name = args.run_name or f"mrbert-{args.model_type.lower()}-{args.max_steps}steps"

    # Full training workflow
    create_vm(args.project, args.zone, args.vm_name)

    if not args.skip_upload:
        upload_code(args.project, args.zone, args.vm_name)

    if not args.skip_setup:
        setup_environment(args.project, args.zone, args.vm_name)

    preprocess_snli(args.project, args.zone, args.vm_name, args.bucket)

    run_training(
        project=args.project,
        zone=args.zone,
        vm_name=args.vm_name,
        gcs_bucket=args.bucket,
        model_type=args.model_type,
        max_steps=args.max_steps,
        run_name=run_name,
        extra_args=args.extra_args,
        wandb_api_key=args.wandb_api_key,
    )

    if args.delete_vm:
        delete_vm(args.project, args.zone, args.vm_name)
    else:
        stop_vm(args.project, args.zone, args.vm_name)


if __name__ == "__main__":
    main()