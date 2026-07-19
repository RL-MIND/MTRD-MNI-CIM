"""Reproducibility, dataset, checkpoint, and result I/O helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


CIFAR_LAYOUT = {
    "cifar10": "cifar-10-batches-py",
    "cifar100": "cifar-100-python",
}
CIFAR_NORMALIZATION_PROFILES = ("dataset-native", "released-legacy")
CODE_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_identity() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "aihwkit": package_version("aihwkit"),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "command": [sys.executable, *sys.argv],
    }


def git_identity(start: Path) -> dict[str, object]:
    from utils.source_tree import source_tree_sha256

    identity: dict[str, object] = {
        "first_party_source_sha256": source_tree_sha256(CODE_ROOT),
    }
    try:
        proc = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return {**identity, "commit": None, "available": False}
    if proc.returncode != 0:
        return {**identity, "commit": None, "available": False}
    status = subprocess.run(
        ["git", "-C", str(start), "status", "--porcelain"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        **identity,
        "commit": proc.stdout.strip(),
        "available": True,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_csv(path: Path, row: dict[str, object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        torch.use_deterministic_algorithms(True)


def worker_seed(worker_id: int) -> None:
    del worker_id
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def cifar_normalization(
    dataset: str,
    profile: str = "dataset-native",
) -> dict[str, object]:
    """Return the explicit normalization contract for a CIFAR run.

    The released scripts reused CIFAR-10 statistics for CIFAR-100. New runs
    default to each dataset's own statistics, while the legacy behavior remains
    available for auditing historical checkpoints.
    """
    if dataset not in CIFAR_LAYOUT:
        raise ValueError(f"unsupported dataset: {dataset}")
    if profile not in CIFAR_NORMALIZATION_PROFILES:
        raise ValueError(
            f"unsupported normalization profile {profile!r}; "
            f"expected one of {CIFAR_NORMALIZATION_PROFILES}"
        )
    from utils.data import CIFAR_STATS

    statistics = CIFAR_STATS["cifar10" if profile == "released-legacy" else dataset]
    return {
        "profile": profile,
        "mean": tuple(float(value) for value in statistics["mean"]),
        "std": tuple(float(value) for value in statistics["std"]),
    }


def build_cifar_loaders(
    dataset: str,
    data_root: Path,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    download: bool,
    normalization_profile: str = "dataset-native",
):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    normalization = cifar_normalization(dataset, normalization_profile)
    mean = normalization["mean"]
    std = normalization["std"]
    train_transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset_class = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    train_set = dataset_class(data_root, train=True, transform=train_transform, download=download)
    test_set = dataset_class(data_root, train=False, transform=test_transform, download=download)
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_seed,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    return train_loader, test_loader, generator


def cifar_identity(
    dataset: str,
    data_root: Path,
    *,
    include_hash: bool = True,
    normalization_profile: str = "dataset-native",
) -> dict[str, object]:
    from torchvision import datasets
    from torchvision.datasets.utils import check_integrity

    normalization = cifar_normalization(dataset, normalization_profile)
    dataset_class = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    folder = data_root / CIFAR_LAYOUT[dataset]
    expected = [
        *dataset_class.train_list,
        *dataset_class.test_list,
        (dataset_class.meta["filename"], dataset_class.meta["md5"]),
    ]
    integrity = [
        {
            "path": str((folder / filename).resolve()),
            "expected_md5": expected_md5,
            "size_bytes": (
                (folder / filename).stat().st_size
                if (folder / filename).is_file() else None
            ),
            "ok": check_integrity(str(folder / filename), expected_md5),
        }
        for filename, expected_md5 in expected
    ]
    present = all(item["ok"] for item in integrity)
    result: dict[str, object] = {
        "dataset": dataset,
        "root": str(data_root.resolve()),
        "folder": str(folder.resolve()),
        "present": present,
        "integrity": integrity,
        "normalization": {
            "profile": normalization["profile"],
            "mean": list(normalization["mean"]),
            "std": list(normalization["std"]),
        },
    }
    if present:
        files = [item for item in folder.rglob("*") if item.is_file()]
        result["files"] = len(files)
        result["bytes"] = sum(item.stat().st_size for item in files)
        if include_hash:
            result["sha256_tree"] = sha256_tree(folder)
    return result


def checkpoint_path(root: Path, dataset: str, device_type: str, role: str, noise: float = 0.0) -> Path:
    directory = root / dataset / device_type
    if role in {"clean", "nominal"}:
        return directory / "teacher_clean.pt"
    if role == "teacher":
        return directory / f"teacher_{device_type}_{noise:g}.pt"
    if role == "mtrd":
        return directory / f"student_mtrd_{device_type}.pt"
    raise ValueError(f"unknown checkpoint role: {role}")


def checkpoint_manifest(
    checkpoint: Path,
    *,
    role: str,
    dataset_identity: dict[str, object],
    protocol: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "mtrd.classification.checkpoint.v1",
        "created_utc": utc_now(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_format": {
            "serialization": "torch.save(raw_state_dict)",
            "contains_optimizer_state": False,
            "portable_inference_command": "bash code/run classification test-checkpoint",
            "strict_loader": "classification.model.load_checkpoint_strict",
        },
        "role": role,
        "architecture": "legacy-compatible-vgg16-bn-13conv-3fc",
        "dataset": dataset_identity,
        "protocol": protocol,
        "metrics": metrics,
        "environment": environment_identity(),
        "source": git_identity(CODE_ROOT),
    }


def checkpoint_normalization_identity(
    checkpoint: Path,
    *,
    dataset: str,
    requested_profile: str,
) -> dict[str, object]:
    """Validate preprocessing provenance when a training manifest is available."""
    requested = cifar_normalization(dataset, requested_profile)
    manifest_path = checkpoint.with_suffix(".manifest.json")
    result: dict[str, object] = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_present": manifest_path.is_file(),
        "requested_dataset": dataset,
        "requested_normalization": {
            "profile": requested["profile"],
            "mean": list(requested["mean"]),
            "std": list(requested["std"]),
        },
        "verified": False,
    }
    if not manifest_path.is_file():
        result["note"] = (
            "No adjacent training manifest is available; preprocessing provenance "
            "cannot be verified automatically."
        )
        return result

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint training manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint training manifest must be an object: {manifest_path}")
    if payload.get("schema") != "mtrd.classification.checkpoint.v1":
        raise ValueError(f"unsupported checkpoint training manifest schema: {manifest_path}")
    declared_hash = payload.get("checkpoint_sha256")
    measured_hash = sha256_file(checkpoint)
    if declared_hash != measured_hash:
        raise ValueError(
            f"checkpoint training manifest SHA-256 mismatch for {checkpoint}: "
            f"declared {declared_hash}, found {measured_hash}"
        )
    dataset_identity = payload.get("dataset")
    if not isinstance(dataset_identity, dict):
        raise ValueError(f"checkpoint training manifest has no dataset identity: {manifest_path}")
    recorded_dataset = dataset_identity.get("dataset")
    normalization = dataset_identity.get("normalization")
    if recorded_dataset != dataset:
        raise ValueError(
            f"checkpoint dataset mismatch: trained for {recorded_dataset!r}, "
            f"requested {dataset!r}"
        )
    if not isinstance(normalization, dict):
        raise ValueError(
            f"checkpoint training manifest has no normalization identity: {manifest_path}"
        )
    recorded_profile = normalization.get("profile")
    if recorded_profile != requested_profile:
        raise ValueError(
            f"checkpoint normalization mismatch: trained with {recorded_profile!r}, "
            f"requested {requested_profile!r}"
        )
    expected_mean = list(requested["mean"])
    expected_std = list(requested["std"])
    if normalization.get("mean") != expected_mean or normalization.get("std") != expected_std:
        raise ValueError(
            f"checkpoint normalization statistics do not match profile "
            f"{requested_profile!r}: {manifest_path}"
        )
    result.update({
        "manifest_sha256": sha256_file(manifest_path),
        "recorded_dataset": recorded_dataset,
        "recorded_normalization": normalization,
        "verified": True,
    })
    return result


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng_state(state: dict[str, object], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader_generator"])


def locate_neurosim(requested: str | None) -> Path | None:
    candidates = []
    if requested:
        candidates.append(Path(requested))
    if os.environ.get("NEUROSIM_HOME"):
        candidates.append(Path(os.environ["NEUROSIM_HOME"]))
    candidates.extend(
        [
            Path("/opt/neurosim"),
            Path(__file__).resolve().parents[2] / "NeuroSim",
        ]
    )
    return next((path for path in candidates if path.is_dir()), None)


def neurosim_functional_preflight(requested: str | None = None) -> dict[str, object]:
    from simulators.neurosim import neurosim_functional_status

    home = locate_neurosim(requested)
    if home is None:
        return neurosim_functional_status(Path("/__missing_neurosim__"), "vgg16")
    result = neurosim_functional_status(home, "vgg16")
    executable = home / "NeuroSIM" / "main"
    result["ppa_executable_ready"] = executable.is_file() and os.access(executable, os.X_OK)
    return result
