#!/usr/bin/env python3
"""
Launch hard deletion training runs for all datasets.

This script launches MrBERT training runs with hard deletion enabled (hard_delete_train_prob=0.5)
for all supported datasets. Hard deletion training is required for the model to work well with
hard deletion at inference time.

IMPORTANT: All runs will evaluate on the TEST SET after training completes (mode=training-and-eval).
This ensures we get final test performance for each dataset with hard deletion.

Usage:
    # Launch all runs (trains + evaluates on test set)
    python launch_hard_deletion_runs.py --all

    # Launch specific dataset (trains + evaluates on test set)
    python launch_hard_deletion_runs.py --dataset snli

    # Quick test run (100 steps only)
    python launch_hard_deletion_runs.py --dataset snli --test

    # Dry run (see commands without executing)
    python launch_hard_deletion_runs.py --all --dry-run

    # List all available datasets
    python launch_hard_deletion_runs.py --list

Prerequisites:
    pip install modal
    modal token set  # one-time auth
"""

import argparse
import subprocess
import sys
import time
from typing import Dict, List


# Dataset configurations with optimized hyperparameters
DATASET_CONFIGS = {
    "snli": {
        "task": "sequence_classification",
        "dataset_name": "local_snli",
        "batch_size": 32,
        "max_steps": 30000,
        "regularizer_delay": 1000,
        "learning_rate": 2e-5,
        "max_seq_length": 128,
        "description": "SNLI - Natural Language Inference",
    },
    "squad": {
        "task": "question_answering",
        "dataset_name": "local_squad",
        "batch_size": 16,
        "max_steps": 30000,
        "regularizer_delay": 1000,
        "learning_rate": 3e-5,
        "max_seq_length": 384,
        "description": "SQuAD - Question Answering",
    },
    "sst2": {
        "task": "sequence_classification",
        "dataset_name": "local_sst2",
        "batch_size": 32,
        "max_steps": 10000,
        "regularizer_delay": 300,
        "learning_rate": 2e-5,
        "max_seq_length": 128,
        "description": "SST-2 - Sentiment Analysis",
    },
    "mrpc": {
        "task": "sequence_classification",
        "dataset_name": "local_mrpc",
        "batch_size": 32,
        "max_steps": 5000,
        "regularizer_delay": 100,
        "learning_rate": 2e-5,
        "max_seq_length": 128,
        "description": "MRPC - Paraphrase Detection",
    },
    "imdb": {
        "task": "sequence_classification",
        "dataset_name": "local_imdb",
        "batch_size": 16,
        "max_steps": 20000,
        "regularizer_delay": 300,
        "learning_rate": 2e-5,
        "max_seq_length": 512,
        "description": "IMDB - Sentiment Analysis",
    },
    "tydiqa": {
        "task": "question_answering",
        "dataset_name": "local_tydiqa",
        "batch_size": 16,
        "max_steps": 30000,
        "regularizer_delay": 1000,
        "learning_rate": 3e-5,
        "max_seq_length": 384,
        "description": "TyDiQA - Multilingual Question Answering",
    },
}


def build_modal_command(
    dataset_key: str,
    config: Dict,
    hard_delete_train_prob: float = 0.0,
    target_deletion_rate: float = 0.3,
    test_mode: bool = False,
    detach: bool = True,
) -> List[str]:
    """Build modal run command for a specific dataset."""

    cmd = [
        "modal",
        "run",
    ]

    if detach:
        cmd.append("--detach")

    cmd.extend([
        "mrbert/training/train_modal.py::main",
        "--model-type", "MrBERT",
        "--task", config["task"],
        "--dataset-name", config["dataset_name"],
        "--batch-size", str(config["batch_size"]),
        "--regularizer-delay", str(config["regularizer_delay"]),
        "--hard-delete-train-prob", str(hard_delete_train_prob),
        "--target-deletion-rate", str(target_deletion_rate),
        "--mode", "training-and-eval",  # Ensures test set evaluation after training
        "--wandb-project", "aronima",
        "--wandb-run-name", f"hard-del-{dataset_key}-{target_deletion_rate}",
        "--use-pi-controller",
        "--use-gumbel-noise",
        "--controller-p", "0.01",
        "--deletion-type", "scaled_sigmoid",
        "--delete-gate-layer", "3",
    ])

    # Add max_steps
    if test_mode:
        cmd.extend(["--max-steps", "100"])
    else:
        cmd.extend(["--max-steps", str(config["max_steps"])])

    # Add extra args if needed
    if "learning_rate" in config:
        cmd.extend(["--extra-args", f"--learning_rate {config['learning_rate']}"])

    if "max_seq_length" in config:
        if "--extra-args" in cmd:
            # Append to existing extra-args
            idx = cmd.index("--extra-args") + 1
            cmd[idx] += f" --max_seq_length {config['max_seq_length']}"
        else:
            cmd.extend(["--extra-args", f"--max_seq_length {config['max_seq_length']}"])

    return cmd


def launch_run(dataset_key: str, test_mode: bool = False, detach: bool = True, dry_run: bool = False):
    """Launch a single training run."""

    if dataset_key not in DATASET_CONFIGS:
        print(f"ERROR: Unknown dataset '{dataset_key}'")
        print(f"Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
        return False

    config = DATASET_CONFIGS[dataset_key]

    print("=" * 80)
    print(f"Launching: {config['description']}")
    print(f"Dataset: {dataset_key}")
    print(f"Hard deletion training probability: 0.0 (soft deletion only during training)")
    print(f"Target deletion rate: 0.3 (30% hard deletion on test set)")
    print(f"Mode: training-and-eval (will evaluate on test set after training)")
    print(f"Max steps: {100 if test_mode else config['max_steps']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Regularizer delay: {config['regularizer_delay']}")
    print("=" * 80)

    cmd = build_modal_command(
        dataset_key=dataset_key,
        config=config,
        test_mode=test_mode,
        detach=detach,
    )

    print(f"\nCommand: {' '.join(cmd)}\n")

    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return True

    try:
        # Use run() and wait for Modal to confirm the job is queued (not just process spawned).
        # Popen returns immediately before Modal finishes scheduling, leaving idle containers.
        result = subprocess.run(cmd, timeout=120)
        if result.returncode == 0:
            print(f"\n✓ Successfully queued run for {dataset_key}")
            return True
        else:
            print(f"\n✗ Modal returned non-zero exit code {result.returncode} for {dataset_key}")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n✗ Timed out waiting for Modal to queue run for {dataset_key}")
        return False
    except Exception as e:
        print(f"\n✗ Failed to launch run for {dataset_key}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Launch hard deletion training runs for MrBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Launch runs for all datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Launch run for a specific dataset",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test run (100 steps only)",
    )
    parser.add_argument(
        "--no-detach",
        action="store_true",
        help="Run in foreground (don't detach)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )

    args = parser.parse_args()

    # List datasets
    if args.list:
        print("\nAvailable datasets:")
        print("-" * 80)
        for key, config in DATASET_CONFIGS.items():
            print(f"  {key:10s} - {config['description']}")
            print(f"             Task: {config['task']}, Steps: {config['max_steps']}, Batch: {config['batch_size']}")
        print("-" * 80)
        return

    # Check that either --all or --dataset is specified
    if not args.all and not args.dataset:
        parser.print_help()
        print("\nERROR: Must specify either --all or --dataset")
        sys.exit(1)

    # Determine which datasets to run
    if args.all:
        datasets_to_run = list(DATASET_CONFIGS.keys())
        print(f"\n🚀 Launching hard deletion runs for ALL datasets ({len(datasets_to_run)} total)")
    else:
        datasets_to_run = [args.dataset]
        print(f"\n🚀 Launching hard deletion run for {args.dataset}")

    if args.test:
        print("⚠️  TEST MODE: Running with 100 steps only")

    if args.dry_run:
        print("🔍 DRY RUN: Commands will be printed but not executed")

    print()

    # Launch runs
    success_count = 0
    failed_datasets = []

    for i, dataset_key in enumerate(datasets_to_run):
        success = launch_run(
            dataset_key=dataset_key,
            test_mode=args.test,
            detach=not args.no_detach,
            dry_run=args.dry_run,
        )

        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_key)

        print()  # Spacing between runs

        # Small delay between launches to avoid overwhelming Modal API
        if i < len(datasets_to_run) - 1 and not args.dry_run:
            time.sleep(2)

    # Summary
    print("=" * 80)
    print(f"SUMMARY: {success_count}/{len(datasets_to_run)} runs launched successfully")

    if failed_datasets:
        print(f"\nFailed datasets: {', '.join(failed_datasets)}")
        sys.exit(1)
    else:
        print("\n✓ All runs launched successfully in parallel!")

        if not args.dry_run and not args.test:
            print(f"\n🚀 {success_count} training runs are now executing in parallel on Modal")
            print("\nTo monitor runs:")
            print("  - Modal dashboard: https://modal.com/logs/hiva/main")
            print("  - Weights & Biases: https://wandb.ai")
            print("  - Project: aronima")
            print("\nTo download checkpoints after completion:")
            print("  modal volume get mrbert-checkpoints <run-name> ./local_checkpoints")


if __name__ == "__main__":
    main()
