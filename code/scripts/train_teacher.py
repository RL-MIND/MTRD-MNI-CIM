#!/usr/bin/env python3
"""Train and test the VGG-16 teachers used by the PyTorch classification workflow.

This entry point intentionally preserves the released classification behavior:
multiclass cross entropy is applied to logits, and the official CIFAR test split
is evaluated after every epoch to select the inference checkpoint. The adjacent
JSON manifest records that this differs from the manuscript's sigmoid/BCE text
and that the checkpoint-selection split is not an independent holdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models import get_model
from utils.data import get_classification_loaders
from utils.helpers import get_device, progress_bar, set_seed
from utils.paths import data_root as default_data_root
from utils.paths import results_root
from utils.training_state import load_resume_state, save_training_state


MANIFEST_SCHEMA = "mtrd.classification.teacher-training.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and test a PyTorch VGG-16 teacher on CIFAR."
    )
    parser.add_argument("--task", required=True, choices=("classification",))
    parser.add_argument("--noise_type", default="rram", choices=("rram", "pcm", "none"))
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save_dir", default=str(results_root() / "checkpoints")
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume from a *.train-state.pt sidecar, or initialize from a "
            "trusted legacy inference checkpoint."
        ),
    )
    parser.add_argument("--model", default="vgg16", choices=("vgg16",))
    parser.add_argument(
        "--dataset", default="cifar10", choices=("cifar10", "cifar100")
    )
    parser.add_argument("--data_root", default=str(default_data_root()))
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.lr <= 0 or not math.isfinite(args.lr):
        parser.error("--lr must be finite and positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    if args.noise_std < 0 or not math.isfinite(args.noise_std):
        parser.error("--noise_std must be finite and non-negative")
    return args


def is_clean_teacher(args: argparse.Namespace) -> bool:
    return args.noise_type == "none" or abs(float(args.noise_std)) < 1e-12


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(get_device())
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device


def protocol_record(args: argparse.Namespace) -> dict[str, object]:
    return {
        "task": "classification",
        "task": "single-label CIFAR classification",
        "architecture": "project VGG-16",
        "implemented_objective": "torch.nn.CrossEntropyLoss applied to logits",
        "manuscript_objective_text": "sigmoid activation with binary cross entropy",
        "objective_discrepancy": (
            "The public runner preserves the released source behavior. Author "
            "confirmation is required before using this objective for a numerical "
            "release."
        ),
        "optimizer": "SGD",
        "scheduler": "CosineAnnealingLR",
        "checkpoint_selection_split": "official CIFAR test split",
        "independent_holdout_used": False,
        "selection_disclosure": (
            "The official test split is evaluated after every epoch and reused for "
            "final paper evaluation; reported accuracy is therefore not from an "
            "independent holdout."
        ),
        "noise_equation": {
            "none": "W_noisy = W",
            "rram": "W_noisy = W * exp(N(0, sigma^2))",
            "pcm": "W_noisy = W + N(0, (eta * layer_max(W))^2)",
        }[args.noise_type],
        "noise_realization": "fresh per perturbable layer and forward pass",
        "noise_type": args.noise_type,
        "noise_std": float(args.noise_std),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    os.replace(temporary, path)


def _checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    directory = Path(args.save_dir) / args.dataset / args.model
    checkpoint = directory / f"ckpt_{args.noise_type}_{args.noise_std}.pth"
    manifest = checkpoint.with_suffix(".manifest.json")
    clean_alias = directory / "ckpt_clean.pth"
    return checkpoint, manifest, clean_alias


@torch.no_grad()
def test_classification(
    model: nn.Module,
    loader,
    device: torch.device,
    noise_type: str,
    noise_std: float,
) -> tuple[float, int, int]:
    """Evaluate one stochastic realization sequence on the official test split."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, lamda=noise_std, noise_type=noise_type)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    if total == 0:
        raise RuntimeError("the CIFAR test loader produced zero samples")
    return 100.0 * correct / total, int(correct), int(total)


def train_classification(args: argparse.Namespace) -> Path:
    """Run checkpoint-compatible training and per-epoch test evaluation."""
    device = resolve_device(args.device)
    set_seed(args.seed)
    train_loader, test_loader, num_classes = get_classification_loaders(
        args.dataset,
        args.data_root,
        args.batch_size,
        args.num_workers,
        seed=args.seed,
    )

    model = get_model(args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    checkpoint_path, manifest_path, clean_alias_path = _checkpoint_paths(args)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    protocol = protocol_record(args)
    print("Training protocol:")
    print(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=True))
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "arguments": vars(args),
        "protocol": protocol,
        "checkpoint": str(checkpoint_path.resolve()),
        "training_state": str(
            checkpoint_path.with_suffix(".train-state.pt").resolve()
        ),
        "best_checkpoint_selection_test_accuracy_percent": None,
        "completed_epochs": 0,
    }
    start_epoch = 0
    best_accuracy = float("-inf")
    if args.resume:
        resume = load_resume_state(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            map_location=device,
            expected_task="classification",
            current_args=args,
        )
        start_epoch = resume["start_epoch"]
        best_accuracy = resume["best_metrics"].get("accuracy", best_accuracy)
        mode = "full training state" if resume["full_state"] else "legacy weights"
        print(
            f"Resumed {mode} from {args.resume}; next epoch={start_epoch + 1}"
        )
        manifest["resume_mode"] = mode
        manifest["completed_epochs"] = start_epoch
        if math.isfinite(best_accuracy):
            manifest["best_checkpoint_selection_test_accuracy_percent"] = (
                best_accuracy
            )
    _atomic_json(manifest_path, manifest)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(
                inputs, lamda=args.noise_std, noise_type=args.noise_type
            )
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)
            progress_bar(
                batch_index,
                len(train_loader),
                f"Loss:{loss_sum / (batch_index + 1):.3f} "
                f"Acc:{100.0 * correct / total:.2f}%",
            )

        test_accuracy, test_correct, test_total = test_classification(
            model,
            test_loader,
            device,
            args.noise_type,
            args.noise_std,
        )
        previous_best = (
            f"{best_accuracy:.2f}%" if math.isfinite(best_accuracy) else "not available"
        )
        print(
            f"Epoch {epoch + 1}: official-test accuracy={test_accuracy:.2f}% "
            f"({test_correct}/{test_total}); previous best={previous_best}"
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            if is_clean_teacher(args):
                torch.save(model.state_dict(), clean_alias_path)
                print(f"Clean-teacher alias: {clean_alias_path}")

        scheduler.step()
        state_path = save_training_state(
            checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            task="classification",
            best_metrics={"accuracy": best_accuracy},
            train_loader=train_loader,
            extra={
                "inference_checkpoint": str(checkpoint_path),
                "protocol": protocol,
                "checkpoint_selection_test_correct": test_correct,
                "checkpoint_selection_test_total": test_total,
            },
        )
        manifest.update(
            {
                "completed_epochs": epoch + 1,
                "latest_checkpoint_selection_test_accuracy_percent": test_accuracy,
                "latest_checkpoint_selection_test_correct": test_correct,
                "latest_checkpoint_selection_test_total": test_total,
                "best_checkpoint_selection_test_accuracy_percent": best_accuracy,
                "training_state": str(state_path.resolve()),
            }
        )
        _atomic_json(manifest_path, manifest)
        print(f"Training state: {state_path}")

    manifest["status"] = "complete"
    manifest["best_checkpoint_selection_test_accuracy_percent"] = best_accuracy
    _atomic_json(manifest_path, manifest)
    print(
        "Training complete. The selected checkpoint used the official test split; "
        "see the adjacent manifest for the protocol disclosure."
    )
    print(f"Best official-test selection accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Manifest: {manifest_path}")
    return checkpoint_path


def main() -> int:
    args = parse_args()
    train_classification(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
