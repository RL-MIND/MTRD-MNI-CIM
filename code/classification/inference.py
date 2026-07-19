"""Portable CIFAR checkpoint loading and clean-test inference helpers.

This module deliberately runs the ordinary PyTorch model without a CIM
simulator.  It is intended for checking a mounted CIFAR-10/CIFAR-100 weight
file, exporting reproducible top-1 predictions, and separating that basic
checkpoint check from the formal simulator-backed CIM evaluation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from .model import PaperVGG16, load_checkpoint_strict
from .repro import (
    CODE_ROOT,
    build_cifar_loaders,
    checkpoint_normalization_identity,
    cifar_identity,
    environment_identity,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)


PREDICTION_FIELDS = (
    "dataset",
    "sample_index",
    "target",
    "prediction",
    "correct",
    "top1_confidence",
)


def _num_classes(dataset: str) -> int:
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    raise ValueError(f"unsupported dataset: {dataset}")


def _prepare_output_directory(
    output_directory: Path,
    checkpoint: Path,
    *,
    overwrite: bool,
) -> None:
    """Create an empty result directory without risking the input checkpoint."""
    output = output_directory.resolve()
    weight = checkpoint.resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"checkpoint-test output is not a directory: {output}")
        if any(output.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "checkpoint-test output exists; pass --overwrite to replace it: "
                    f"{output}"
                )
            if weight.is_relative_to(output):
                raise ValueError(
                    "refusing to overwrite an output directory that contains the input "
                    f"checkpoint: {output}"
                )
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def test_checkpoint(args) -> tuple[Path, Path]:
    """Strict-load one VGG16 checkpoint and test it on the full CIFAR test set.

    The output contains one prediction row per official test image and a JSON
    manifest that binds the result to the exact weight and data identities.
    """
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint is missing or empty: {checkpoint}")

    normalization = checkpoint_normalization_identity(
        checkpoint,
        dataset=args.dataset,
        requested_profile=args.normalization_profile,
    )
    if args.require_checkpoint_manifest and not normalization["verified"]:
        raise RuntimeError(
            "--require-checkpoint-manifest was requested, but the adjacent checkpoint "
            "training manifest could not verify dataset and normalization provenance"
        )

    output = Path(args.output_dir).expanduser().resolve()
    _prepare_output_directory(output, checkpoint, overwrite=args.overwrite)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    _, test_loader, _ = build_cifar_loaders(
        args.dataset,
        Path(args.data_root),
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        download=args.download,
        normalization_profile=args.normalization_profile,
    )

    model = PaperVGG16(_num_classes(args.dataset))
    load_checkpoint_strict(model, str(checkpoint), "cpu")
    model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        logits = model(inputs.to(device, non_blocking=True))
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        predictions_cpu = predictions.cpu()
        confidence_cpu = confidence.cpu()
        targets_cpu = targets.cpu()
        for target, prediction, score in zip(
            targets_cpu.tolist(), predictions_cpu.tolist(), confidence_cpu.tolist()
        ):
            matched = int(prediction == target)
            rows.append(
                {
                    "dataset": args.dataset,
                    "sample_index": total,
                    "target": int(target),
                    "prediction": int(prediction),
                    "correct": matched,
                    "top1_confidence": float(score),
                }
            )
            correct += matched
            total += 1
    if total == 0:
        raise RuntimeError("official CIFAR test loader is empty")

    predictions_path = output / "classification_clean_test_predictions.csv"
    manifest_path = output / "classification_clean_test_manifest.json"
    write_csv(predictions_path, rows, PREDICTION_FIELDS)
    manifest = {
        "schema": "mtrd.classification.clean-checkpoint-test.v1",
        "created_utc": utc_now(),
        "execution_engine": "PyTorch clean digital inference; no CIM simulator",
        "dataset": cifar_identity(
            args.dataset,
            Path(args.data_root),
            include_hash=True,
            normalization_profile=args.normalization_profile,
        ),
        "evaluation_split": "official CIFAR test split",
        "architecture": "legacy-compatible-vgg16-bn-13conv-3fc",
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "strict_load_valid": True,
            "normalization_provenance": normalization,
            "checkpoint_manifest_required": bool(args.require_checkpoint_manifest),
        },
        "result": {
            "correct": correct,
            "total": total,
            "accuracy_fraction": correct / total,
            "accuracy_percent": 100.0 * correct / total,
        },
        "predictions_csv": str(predictions_path),
        "predictions_csv_sha256": sha256_file(predictions_path),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "requested_device": args.device,
        "resolved_device": str(device),
        "external_checkpoint_provenance_warning": (
            "A strict architecture load proves compatibility only. A missing or unverified "
            "adjacent training manifest does not establish the checkpoint's training role or "
            "paper-curve provenance."
        ),
        "environment": environment_identity(),
        "source": git_identity(CODE_ROOT),
    }
    write_json(manifest_path, manifest)
    return predictions_path, manifest_path
