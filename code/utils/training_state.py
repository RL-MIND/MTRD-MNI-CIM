"""Versioned state sidecars for deterministic teacher-training resume."""

from __future__ import annotations

import os
import platform
import random
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.checkpoints import extract_state_dict, load_checkpoint


TRAINING_STATE_FORMAT = "mtrd.training-state.v1"


def training_state_path(checkpoint_path: str | Path) -> Path:
    """Return the sidecar path associated with an inference checkpoint."""
    return Path(checkpoint_path).with_suffix(".train-state.pt")


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG used by the classification training loop."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )


def _loader_generator_state(loader: Any) -> torch.Tensor | None:
    generator = getattr(loader, "generator", None)
    return generator.get_state() if generator is not None else None


def _restore_loader_generator(loader: Any, state: torch.Tensor | None) -> None:
    generator = getattr(loader, "generator", None)
    if generator is not None and state is not None:
        generator.set_state(state.cpu())


def _provenance() -> dict[str, Any]:
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "cwd": os.getcwd(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pytorch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        provenance["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return provenance


def save_training_state(
    checkpoint_path: str | Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: Any,
    task: str,
    scheduler: Any = None,
    best_metrics: Mapping[str, float] | None = None,
    train_loader: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save the state required to resume after an epoch."""
    path = training_state_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": TRAINING_STATE_FORMAT,
        "task": task,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "args": dict(vars(args)) if hasattr(args, "__dict__") else dict(args),
        "best_metrics": dict(best_metrics or {}),
        "rng_state": capture_rng_state(),
        "train_loader_generator_state": _loader_generator_state(train_loader),
        "provenance": _provenance(),
        "extra": dict(extra or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_resume_state(
    resume_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    train_loader: Any = None,
    map_location: str | torch.device = "cpu",
    expected_task: str | None = None,
    current_args: Any = None,
) -> dict[str, Any]:
    """Load a complete sidecar or initialize from inference-only weights."""
    checkpoint = load_checkpoint(resume_path, map_location=map_location)
    is_training_state = (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("format") == TRAINING_STATE_FORMAT
    )
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    if not is_training_state:
        return {
            "full_state": False,
            "start_epoch": 0,
            "best_metrics": {},
            "resume_path": str(resume_path),
        }

    if expected_task is not None and checkpoint.get("task") != expected_task:
        raise ValueError(
            f"resume task mismatch: saved={checkpoint.get('task')!r}, "
            f"current={expected_task!r}"
        )
    saved_args = dict(checkpoint.get("args") or {})
    if current_args is not None:
        current = (
            dict(vars(current_args))
            if hasattr(current_args, "__dict__")
            else dict(current_args)
        )
        ignored = {"resume", "save_dir"}
        mismatches = {
            key: (saved_args[key], current[key])
            for key in saved_args.keys() & current.keys()
            if key not in ignored and saved_args[key] != current[key]
        }
        if mismatches:
            details = ", ".join(
                f"{key}: saved={old!r}, current={new!r}"
                for key, (old, new) in sorted(mismatches.items())
            )
            raise ValueError(f"resume arguments differ from saved run: {details}")

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    _restore_loader_generator(
        train_loader, checkpoint.get("train_loader_generator_state")
    )
    restore_rng_state(checkpoint.get("rng_state"))
    return {
        "full_state": True,
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "best_metrics": dict(checkpoint.get("best_metrics") or {}),
        "saved_args": saved_args,
        "provenance": dict(checkpoint.get("provenance") or {}),
        "resume_path": str(resume_path),
    }
