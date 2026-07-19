#!/usr/bin/env python3
"""Train and evaluate the PyTorch VGG16 classification workflow.

Every run records the complete implemented protocol and produces fresh metrics.
The module does not contain reference-figure data or plotting code.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "mtrd.classification.pytorch.v1"
TRAINING_STATE_FORMAT = "mtrd.classification.pytorch-training-state.v2"
FILE_PATH = Path(__file__).resolve()
CODE_ROOT = FILE_PATH.parents[1]
CAPSULE_ROOT = CODE_ROOT.parent
IS_CODE_OCEAN = CODE_ROOT == Path("/code") or (
    Path("/code").is_dir()
    and Path("/data").is_dir()
    and CODE_ROOT.parent == Path("/")
)
DEFAULT_MOUNT_DATA_ROOT = Path(
    os.environ.get(
        "MTRD_DATA_ROOT",
        str(Path("/data") if IS_CODE_OCEAN else CAPSULE_ROOT / "data"),
    )
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MTRD_CIFAR_ROOT",
        str(DEFAULT_MOUNT_DATA_ROOT / "datasets" / "cifar"),
    )
)
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get(
        "MTRD_RESULTS_ROOT",
        str(Path("/results") if IS_CODE_OCEAN else CAPSULE_ROOT / "results"),
    )
)
DEFAULT_WORK_DIR = Path(
    os.environ.get(
        "MTRD_CLASSIFICATION_PYTORCH_WORK_DIR",
        str(DEFAULT_RESULTS_ROOT / "classification" / "pytorch"),
    )
)
NOISE_LEVELS = {
    "rram": (0.1, 0.2, 0.3, 0.4, 0.5),
    "pcm": (0.02, 0.04, 0.06, 0.08, 0.10),
}
STUDENT_NOISE = {"rram": 0.3, "pcm": 0.06}
VARIATION_BASELINES = {"rram": (0.3, 0.5), "pcm": (0.06, 0.10)}
METHOD_ORDER = (
    "mtrd",
    "variation_aware_primary",
    "variation_aware_secondary",
    "five_network_ensemble",
    "nominal",
)
METHOD_LABELS = {
    "mtrd": "MTRD",
    "variation_aware_primary": "Variation-aware (primary)",
    "variation_aware_secondary": "Variation-aware (secondary)",
    "five_network_ensemble": "Five-network ensemble",
    "nominal": "Nominal",
}
BALANCING_POLICIES = (
    "positive_delta_softmax",
    "negative_delta_softmax",
)


@dataclass(frozen=True)
class TeacherSpec:
    label: str
    noise_type: str
    noise_level: float
    checkpoint: Path


@dataclass(frozen=True)
class TrainingJob:
    name: str
    command: tuple[str, ...]
    expected_checkpoint: Path
    log_path: Path
    done_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def noise_token(value: float) -> str:
    return f"{float(value):g}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def package_versions(required_imports: Iterable[str] = ()) -> dict[str, str | None]:
    required = set(required_imports)
    versions: dict[str, str | None] = {}
    for name in ("torch", "torchvision", "numpy"):
        if importlib.util.find_spec(name) is None:
            versions[name] = None
            continue
        try:
            versions[name] = importlib.metadata.version(name)
            if name in required:
                __import__(name)
        except Exception as exc:  # preflight must report a broken import, not crash
            versions[name] = f"IMPORT_ERROR: {type(exc).__name__}: {exc}"
    return versions


def git_provenance() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=CODE_ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    _install_code_import_path()
    from utils.source_tree import source_tree_sha256

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "root": run("rev-parse", "--show-toplevel"),
        "first_party_source_sha256": source_tree_sha256(CODE_ROOT),
    }


def protocol_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "claim": "fresh_execution_not_numerically_verified",
        "numerical_reproduction_verified": False,
        "author_raw_reference": None,
        "dataset": "cifar10",
        "model": "vgg16",
        "seeds": {
            "base_seed": args.seed,
            "paired_evaluation_noise": True,
            "deterministic_algorithms": True,
        },
        "noise_models": {
            "rram": {
                "equation": "W_noisy = W_nominal * exp(N(0, sigma^2))",
                "levels": list(NOISE_LEVELS["rram"]),
                "student_training_level": STUDENT_NOISE["rram"],
            },
            "pcm": {
                "equation": "W_noisy = W_nominal + N(0, (eta * max(W))^2) per layer",
                "wmax_definition": "signed maximum weight value over each logical layer",
                "levels": list(NOISE_LEVELS["pcm"]),
                "student_training_level": STUDENT_NOISE["pcm"],
            },
        },
        "teacher_training": {
            "clean_teacher": True,
            "one_variation_aware_teacher_per_noise_level": True,
            "epochs": getattr(args, "teacher_epochs", None),
            "optimizer": "SGD",
            "initial_learning_rate": getattr(args, "teacher_lr", None),
            "checkpoint_selection": (
                "highest CIFAR-10 test accuracy observed during training; this follows "
                "the supplied legacy implementation and uses the test set for selection"
            ),
        },
        "student_training": {
            "epochs": getattr(args, "student_epochs", None),
            "optimizer": "SGD",
            "initial_learning_rate": getattr(args, "student_lr", None),
            "initialization": "variation-aware teacher at sigma=0.3 or eta=0.06",
            "teacher_distillation_forward": "nominal student (noise disabled)",
            "task_loss_forward": "noise-injected student at the fixed initialization level",
            "teacher_forward": "each teacher evaluated at its own assigned noise level",
            "eq4_distillation_loss": (
                "beta-weighted sum of individual teacher-to-student KL losses"
            ),
            "kd_weight": getattr(args, "kd_weight", None),
            "temperature": getattr(args, "temperature", None),
            "gradient_clip_norm": getattr(args, "grad_clip", None),
            "checkpoint_selection": "highest mean accuracy across the five controlled noise levels",
            "eq6_initial_feedback": (
                "measure p_0 before epoch 1; the update after epoch 1 compares p_1 with p_0"
            ),
        },
        "adaptive_balancing": {
            "selected_policy": getattr(args, "balancing_policy", None),
            "performance_unit": getattr(args, "balancing_unit", None),
            "positive_delta_policy": "softmax(p_t - p_t_minus_1)",
            "negative_delta_policy": "softmax(-(p_t - p_t_minus_1))",
            "ambiguity": (
                "The displayed Equation (6) is not a normalized softmax, and the prose says "
                "a stagnant or degrading teacher condition should receive more weight. "
                "The manuscript also does not state whether p is an accuracy fraction or "
                "accuracy percentage points; softmax is scale-sensitive. Both policy and "
                "unit are therefore selected explicitly and recorded."
            ),
            "temperature": getattr(args, "balancing_temperature", None),
            "first_epoch_weights": "uniform",
        },
        "evaluation": {
            "controlled": "all five methods at every fixed paper noise level",
            "randomized": getattr(args, "randomized_distribution", None),
            "device_realization_policy": getattr(args, "realization_policy", None),
            "device_realization_ambiguity": (
                "The manuscript does not state whether one sampled device realization is "
                "held fixed for the complete test set or resampled for every batch. Formal "
                "evaluation is fail-closed until --realization-policy is selected explicitly."
            ),
            "controlled_trials": getattr(args, "controlled_trials", None),
            "randomized_trials": getattr(args, "randomized_trials", None),
            "accuracy_definition": "100 * n_correct / n_samples",
            "post_hoc_metric_offset": 0.0,
        },
        "known_manuscript_ambiguities": [
            "The implemented panel order is controlled RRAM, randomized RRAM, controlled PCM, and randomized PCM. The manuscript caption uses a conflicting order and requires correction before publication.",
            "The Methods text names sigmoid plus binary cross entropy for CIFAR classification; the supplied source and this runner use VGG16 logits plus multiclass cross entropy.",
            "The earlier training implementation differs from printed Eq. (4) in KL reduction, scaling, and teacher aggregation. This runner implements the documented interpretation; author confirmation is required before numerical release.",
            "PCM Eq. (2) uses the signed maximum weight value max(W) over each logical layer; max(abs(W)) is used only for symmetric weight quantization.",
        ],
    }


def checkpoint_dir(args: argparse.Namespace) -> Path:
    configured = getattr(args, "checkpoint_root", None)
    if configured:
        return Path(configured) / "cifar10" / "vgg16"
    return Path(args.work_dir) / "checkpoints" / "cifar10" / "vgg16"


def clean_checkpoint(args: argparse.Namespace) -> Path:
    return checkpoint_dir(args) / "ckpt_clean.pth"


def teacher_checkpoint(args: argparse.Namespace, noise_type: str, level: float) -> Path:
    return checkpoint_dir(args) / f"ckpt_{noise_type}_{float(level)}.pth"


def mtrd_checkpoint(args: argparse.Namespace, noise_type: str) -> Path:
    level = noise_token(STUDENT_NOISE[noise_type])
    policy = getattr(args, "balancing_policy", "positive_delta_softmax")
    return checkpoint_dir(args) / f"mtrd_{noise_type}_{policy}_S{level}.pth"


def expected_checkpoints(args: argparse.Namespace, noise_types: Iterable[str]) -> list[Path]:
    paths = [clean_checkpoint(args)]
    for noise_type in noise_types:
        paths.extend(
            teacher_checkpoint(args, noise_type, level)
            for level in NOISE_LEVELS[noise_type]
        )
        paths.append(mtrd_checkpoint(args, noise_type))
    return paths


def cifar_required_files(data_root: Path) -> list[Path]:
    folder = data_root / "cifar-10-batches-py"
    return [
        folder / "batches.meta",
        *(folder / f"data_batch_{index}" for index in range(1, 6)),
        folder / "test_batch",
    ]


def cifar_integrity_checks(data_root: Path) -> list[dict[str, Any]]:
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity

    folder = data_root / CIFAR10.base_folder
    expected = [
        *CIFAR10.train_list,
        *CIFAR10.test_list,
        (CIFAR10.meta["filename"], CIFAR10.meta["md5"]),
    ]
    return [
        {
            "kind": "dataset",
            "path": str(folder / filename),
            "expected_md5": expected_md5,
            "size_bytes": (
                (folder / filename).stat().st_size
                if (folder / filename).is_file() else None
            ),
            "ok": check_integrity(str(folder / filename), expected_md5),
        }
        for filename, expected_md5 in expected
    ]


def dataset_fingerprint(data_root: Path) -> dict[str, Any] | None:
    integrity = cifar_integrity_checks(data_root)
    if not all(item["ok"] for item in integrity):
        return None
    paths = cifar_required_files(data_root)
    entries = [
        {
            "path": str(path.relative_to(data_root)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    aggregate = hashlib.sha256()
    for entry in entries:
        aggregate.update(
            f"{entry['path']}\0{entry['size']}\0{entry['sha256']}\n".encode("ascii")
        )
    return {"sha256": aggregate.hexdigest(), "files": entries}


def checkpoint_preflight(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "checkpoint",
        "path": str(path),
        "ok": False,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }
    if not path.is_file() or path.stat().st_size <= 0:
        result["error"] = "missing or empty"
        return result
    try:
        _install_code_import_path()
        from models import get_model
        from utils.checkpoints import (
            load_state_dict,
            remap_legacy_vgg_state_dict,
            state_dict_finiteness,
        )

        state = load_state_dict(path, map_location="cpu")
        nonfinite, total = state_dict_finiteness(state)
        if nonfinite:
            raise ValueError(f"{nonfinite}/{total} floating values are non-finite")
        model = get_model("vgg16", num_classes=10)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as original:
            mapped = remap_legacy_vgg_state_dict(state)
            if tuple(mapped) == tuple(state):
                raise original
            model.load_state_dict(mapped, strict=True)
        result.update({"ok": True, "sha256": sha256_file(path)})
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def classification_checkpoint_roles(args: argparse.Namespace) -> dict[str, Path]:
    roles = {"classification.cifar10.vgg16.nominal": clean_checkpoint(args)}
    for noise_type in args.noise_types:
        for level in NOISE_LEVELS[noise_type]:
            roles[
                f"classification.cifar10.vgg16.{noise_type}.teacher.{noise_token(level)}"
            ] = teacher_checkpoint(args, noise_type, level)
        roles[
            f"classification.cifar10.vgg16.{noise_type}.mtrd.{args.balancing_policy}"
        ] = mtrd_checkpoint(args, noise_type)
    return roles


def run_preflight(args: argparse.Namespace, *, emit: bool = True) -> tuple[dict[str, Any], bool]:
    data_root = Path(args.data_root).resolve()
    source_paths = [
        CODE_ROOT / "models" / "vgg.py",
        CODE_ROOT / "models" / "noisy_layers.py",
        CODE_ROOT / "scripts" / "train_teacher.py",
        CODE_ROOT / "utils" / "data.py",
        CODE_ROOT / "utils" / "checkpoints.py",
    ]
    required_packages = ["torch", "torchvision", "numpy"]
    dependency_versions = package_versions(required_packages)
    checks: list[dict[str, Any]] = []
    for path in source_paths:
        checks.append({"kind": "source", "path": str(path), "ok": path.is_file()})
    for package in required_packages:
        value = dependency_versions[package]
        checks.append({
            "kind": "python_package", "name": package,
            "value": value, "ok": value is not None and not value.startswith("IMPORT_ERROR"),
        })
    checks.extend(cifar_integrity_checks(data_root))
    role_manifest_identity = None
    if args.stage == "evaluate":
        for path in expected_checkpoints(args, args.noise_types):
            checks.append(checkpoint_preflight(path))
        try:
            from utils.checkpoint_roles import validate_checkpoint_roles

            if not args.checkpoint_role_manifest:
                raise ValueError("--checkpoint-role-manifest is required for evaluation")
            role_manifest_identity = validate_checkpoint_roles(
                args.checkpoint_role_manifest, classification_checkpoint_roles(args),
            )
            checks.append({
                "kind": "checkpoint_role_manifest",
                "path": str(Path(args.checkpoint_role_manifest).resolve()),
                "ok": True,
            })
        except Exception as error:
            checks.append({
                "kind": "checkpoint_role_manifest",
                "path": str(args.checkpoint_role_manifest or "missing"),
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            })

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "stage": args.stage,
        "ready": all(check["ok"] for check in checks),
        "checks": checks,
        "dataset_fingerprint": (
            dataset_fingerprint(data_root)
        ),
        "checkpoint_role_manifest": role_manifest_identity,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": dependency_versions,
        },
        "git": git_provenance(),
        "protocol": protocol_manifest(args),
    }
    if emit:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    if args.preflight_json:
        atomic_json(Path(args.preflight_json), report)
    return report, bool(report["ready"])


def _common_train_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--task", "classification",
        "--model", "vgg16",
        "--dataset", "cifar10",
        "--data_root", str(Path(args.data_root).resolve()),
        "--save_dir", str(
            Path(args.checkpoint_root)
            if getattr(args, "checkpoint_root", None)
            else Path(args.work_dir) / "checkpoints"
        ),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--momentum", str(args.momentum),
        "--weight_decay", str(args.weight_decay),
    ]


def build_training_jobs(args: argparse.Namespace) -> list[TrainingJob]:
    python = str(Path(args.python).resolve()) if os.sep in args.python else args.python
    logs = Path(args.work_dir) / "training" / "logs"
    done = Path(args.work_dir) / "training" / "done"
    jobs: list[TrainingJob] = []

    clean_cmd = (
        python, str(CODE_ROOT / "scripts" / "train_clean_teacher.py"),
        *_common_train_arguments(args),
        "--epochs", str(args.teacher_epochs), "--lr", str(args.teacher_lr),
    )
    jobs.append(TrainingJob(
        "clean_teacher", clean_cmd, clean_checkpoint(args),
        logs / "clean_teacher.log", done / "clean_teacher.json",
    ))

    for noise_type in args.noise_types:
        for level in NOISE_LEVELS[noise_type]:
            token = noise_token(level)
            command = (
                python, str(CODE_ROOT / "scripts" / "train_teacher.py"),
                *_common_train_arguments(args),
                "--noise_type", noise_type, "--noise_std", token,
                "--epochs", str(args.teacher_epochs), "--lr", str(args.teacher_lr),
            )
            name = f"teacher_{noise_type}_{token}"
            jobs.append(TrainingJob(
                name, command, teacher_checkpoint(args, noise_type, level),
                logs / f"{name}.log", done / f"{name}.json",
            ))

        command = (
            python, str(FILE_PATH), "_train-mtrd",
            "--noise-type", noise_type,
            "--data-root", str(Path(args.data_root).resolve()),
            "--work-dir", str(Path(args.work_dir)),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--device", str(args.device),
            "--student-epochs", str(args.student_epochs),
            "--student-lr", str(args.student_lr),
            "--momentum", str(args.momentum),
            "--weight-decay", str(args.weight_decay),
            "--temperature", str(args.temperature),
            "--kd-weight", str(args.kd_weight),
            "--grad-clip", str(args.grad_clip),
            "--balancing-policy", args.balancing_policy,
            "--balancing-temperature", str(args.balancing_temperature),
            "--balancing-unit", args.balancing_unit,
        )
        if args.checkpoint_root:
            command = (*command, "--checkpoint-root", str(Path(args.checkpoint_root)))
        name = f"mtrd_{noise_type}"
        jobs.append(TrainingJob(
            name, command, mtrd_checkpoint(args, noise_type),
            logs / f"{name}.log", done / f"{name}.json",
        ))
    return jobs


def maybe_resume_command(job: TrainingJob) -> tuple[str, ...]:
    if job.name.startswith("mtrd_"):
        return job.command
    sidecar = (
        job.expected_checkpoint.with_name("ckpt_none_0.0.train-state.pt")
        if job.name == "clean_teacher"
        else job.expected_checkpoint.with_suffix(".train-state.pt")
    )
    if sidecar.is_file() and "--resume" not in job.command:
        return (*job.command, "--resume", str(sidecar))
    return job.command


def training_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "protocol": protocol_manifest(args),
        "jobs": [
            {
                "name": job.name,
                "command": list(job.command),
                "expected_checkpoint": str(job.expected_checkpoint),
                "log": str(job.log_path),
                "done": str(job.done_path),
            }
            for job in build_training_jobs(args)
        ],
    }


def training_job_identity(job: TrainingJob, preflight: dict[str, Any]) -> str:
    source_paths = (
        FILE_PATH,
        CODE_ROOT / "models" / "vgg.py",
        CODE_ROOT / "models" / "noisy_layers.py",
        CODE_ROOT / "scripts" / "train_teacher.py",
        CODE_ROOT / "scripts" / "train_clean_teacher.py",
        CODE_ROOT / "utils" / "data.py",
        CODE_ROOT / "utils" / "checkpoints.py",
    )
    payload = {
        "base_command": list(job.command),
        "dataset_sha256": (preflight.get("dataset_fingerprint") or {}).get("sha256"),
        "environment": preflight.get("environment"),
        "git": preflight.get("git"),
        "protocol": preflight.get("protocol"),
        "sources": {
            str(path.relative_to(CODE_ROOT)): sha256_file(path)
            for path in source_paths
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_completed_job(
    job: TrainingJob, preflight: dict[str, Any], expected_identity: str,
) -> None:
    try:
        record = json.loads(job.done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid completed sentinel for {job.name}: {error}") from error
    measured_identity = record.get("job_identity_sha256")
    measured_checkpoint = sha256_file(job.expected_checkpoint)
    if measured_identity != expected_identity:
        raise RuntimeError(
            f"completed job identity changed for {job.name}; use --force to retrain "
            "or select a new --work-dir"
        )
    if record.get("checkpoint_sha256") != measured_checkpoint:
        raise RuntimeError(
            f"completed checkpoint hash changed for {job.name}: {job.expected_checkpoint}"
        )
    if record.get("base_command") != list(job.command):
        raise RuntimeError(f"completed base command changed for {job.name}")


def cmd_train(args: argparse.Namespace) -> int:
    if args.balancing_unit == "unconfirmed":
        raise ValueError(
            "Eq. (6) does not define the scale of p; select --balancing-unit "
            "fraction or percentage_points explicitly before training"
        )
    report, ready = run_preflight(args, emit=False)
    if not ready:
        failed = [check for check in report["checks"] if not check["ok"]]
        raise RuntimeError(
            "training preflight failed:\n" + "\n".join(
                f"- {item.get('path') or item.get('name')}: {item.get('value', 'missing')}"
                for item in failed
            )
        )
    plan = training_plan(args)
    plan["preflight"] = report
    manifest_path = Path(args.work_dir) / "training" / "manifest.json"
    plan["status"] = "running"
    plan["started_utc"] = utc_now()
    atomic_json(manifest_path, plan)

    for index, job in enumerate(build_training_jobs(args)):
        job_identity = training_job_identity(job, report)
        if (
            not args.force
            and job.done_path.is_file()
            and job.expected_checkpoint.is_file()
        ):
            validate_completed_job(job, report, job_identity)
            print(f"[skip] {job.name}: completed sentinel and checkpoint exist")
            continue
        command = job.command if args.force else maybe_resume_command(job)
        if args.force and job.name.startswith("mtrd_"):
            command = (*command, "--restart")
        print(f"[run] {job.name}")
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(args.seed)
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with job.log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=CODE_ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
            )
        record = {
            "schema_version": SCHEMA_VERSION,
            "job": job.name,
            "base_command": list(job.command),
            "command": list(command),
            "job_identity_sha256": job_identity,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
            "checkpoint": str(job.expected_checkpoint),
            "checkpoint_exists": job.expected_checkpoint.is_file(),
            "checkpoint_sha256": (
                sha256_file(job.expected_checkpoint)
                if job.expected_checkpoint.is_file() else None
            ),
            "log": str(job.log_path),
            "completed_utc": utc_now(),
        }
        if result.returncode != 0 or not job.expected_checkpoint.is_file():
            plan["status"] = "failed"
            plan["failed_job"] = record
            atomic_json(manifest_path, plan)
            raise RuntimeError(
                f"training job {job.name} failed; inspect {job.log_path}"
            )
        atomic_json(job.done_path, record)
        plan["jobs"][index]["result"] = record
        atomic_json(manifest_path, plan)

    plan["status"] = "complete"
    plan["completed_utc"] = utc_now()
    plan["checkpoint_hashes"] = {
        str(path): sha256_file(path)
        for path in expected_checkpoints(args, args.noise_types)
    }
    _install_code_import_path()
    from utils.checkpoint_roles import write_checkpoint_roles

    role_manifest_path = (
        Path(args.work_dir) / "training" / "checkpoint-roles.generated.json"
    )
    role_manifest = write_checkpoint_roles(
        role_manifest_path,
        classification_checkpoint_roles(args),
        group_id=(
            f"classification-cifar10-vgg16-seed-{args.seed}-{args.balancing_policy}"
        ),
        role_assignments_author_verified=False,
    )
    plan["checkpoint_role_manifest"] = role_manifest
    atomic_json(manifest_path, plan)
    print(f"Training manifest: {manifest_path}")
    print(f"Checkpoint role manifest: {role_manifest_path}")
    return 0


def _install_code_import_path() -> None:
    value = str(CODE_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def _load_model(path: Path, device: Any, num_classes: int = 10) -> Any:
    _install_code_import_path()
    from models import get_model
    from utils.checkpoints import load_state_dict, remap_legacy_vgg_state_dict

    model = get_model("vgg16", num_classes=num_classes).to(device)
    state = load_state_dict(path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as original:
        mapped = remap_legacy_vgg_state_dict(state)
        if tuple(mapped) == tuple(state):
            raise original
        model.load_state_dict(mapped, strict=True)
    return model


def _set_seed(seed: int) -> None:
    _install_code_import_path()
    from utils.helpers import set_seed

    set_seed(seed)


def _device(device_arg: str) -> Any:
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device_arg}, but CUDA is unavailable")
    return device


def _teacher_specs(args: argparse.Namespace, noise_type: str) -> list[TeacherSpec]:
    specs = [TeacherSpec("clean", "none", 0.0, clean_checkpoint(args))]
    specs.extend(
        TeacherSpec(noise_token(level), noise_type, level,
                    teacher_checkpoint(args, noise_type, level))
        for level in NOISE_LEVELS[noise_type]
    )
    return specs


def _capture_rng_state(torch: Any, np: Any) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], torch: Any, np: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _softmax_values(values: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("balancing temperature must be finite and positive")
    scaled = [float(value) / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def update_balancing_weights(
    current: Sequence[float], previous: Sequence[float] | None,
    policy: str, temperature: float,
) -> list[float]:
    if previous is None:
        return [1.0 / len(current)] * len(current)
    if len(current) != len(previous) or not current:
        raise ValueError("current and previous performance vectors must be non-empty and equal length")
    deltas = [float(new) - float(old) for new, old in zip(current, previous)]
    if policy == "positive_delta_softmax":
        scores = deltas
    elif policy == "negative_delta_softmax":
        scores = [-value for value in deltas]
    else:
        raise ValueError(f"unknown balancing policy: {policy}")
    return _softmax_values(scores, temperature)


def _accuracy_model(
    model: Any, loader: Any, device: Any, noise_type: str, noise_level: float,
    *, max_samples: int = 0,
) -> tuple[int, int]:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if max_samples and total >= max_samples:
                break
            if max_samples:
                remaining = max_samples - total
                inputs, targets = inputs[:remaining], targets[:remaining]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs, lamda=noise_level, noise_type=noise_type)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)
    if total == 0:
        raise RuntimeError("evaluation loader produced zero samples")
    return int(correct), int(total)


def _train_mtrd_device(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch.nn.functional as functional

    _install_code_import_path()
    from models import get_model
    from utils.data import get_classification_loaders

    if args.balancing_policy not in BALANCING_POLICIES:
        raise ValueError(f"unsupported balancing policy: {args.balancing_policy}")
    if args.balancing_unit == "unconfirmed":
        raise ValueError(
            "select --balancing-unit fraction or percentage_points explicitly"
        )
    _set_seed(args.seed)
    device = _device(args.device)
    train_loader, test_loader, num_classes = get_classification_loaders(
        "cifar10", args.data_root, args.batch_size, args.num_workers, seed=args.seed,
    )
    specs = _teacher_specs(args, args.noise_type)
    missing = [str(spec.checkpoint) for spec in specs if not spec.checkpoint.is_file()]
    if missing:
        raise FileNotFoundError("missing MTRD teachers:\n" + "\n".join(missing))

    teachers = []
    for spec in specs:
        teacher = _load_model(spec.checkpoint, device, num_classes)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teachers.append(teacher)
    student = get_model("vgg16", num_classes=num_classes).to(device)
    initial_checkpoint = teacher_checkpoint(
        args, args.noise_type, STUDENT_NOISE[args.noise_type]
    )
    initial_state = _load_model(initial_checkpoint, device, num_classes).state_dict()
    student.load_state_dict(initial_state, strict=True)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.student_lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.student_epochs, 1)
    )
    output = mtrd_checkpoint(args, args.noise_type)
    output.parent.mkdir(parents=True, exist_ok=True)
    last_output = output.with_name(output.stem + ".last.pth")
    state_path = output.with_suffix(".train-state.pt")
    history_path = output.with_suffix(".history.csv")
    teacher_hashes = {str(spec.checkpoint): sha256_file(spec.checkpoint) for spec in specs}
    data_identity = dataset_fingerprint(Path(args.data_root).resolve())
    if data_identity is None:
        raise RuntimeError("CIFAR-10 integrity changed after training preflight")
    config = {
        "noise_type": args.noise_type,
        "teacher_levels": [spec.noise_level for spec in specs],
        "student_noise": STUDENT_NOISE[args.noise_type],
        "epochs": args.student_epochs,
        "lr": args.student_lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "kd_weight": args.kd_weight,
        "balancing_policy": args.balancing_policy,
        "balancing_temperature": args.balancing_temperature,
        "balancing_unit": args.balancing_unit,
        "gradient_clip_norm": args.grad_clip,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "requested_device": args.device,
        "resolved_device": str(device),
        "dataset_sha256": data_identity["sha256"],
        "teacher_hashes": teacher_hashes,
    }
    start_epoch = 0
    weights = [1.0 / len(specs)] * len(specs)
    previous_performance: list[float] | None = None
    best_mean = float("-inf")
    history: list[dict[str, Any]] = []

    if state_path.is_file() and not args.restart:
        try:
            saved = torch.load(state_path, map_location=device, weights_only=False)
        except TypeError:
            saved = torch.load(state_path, map_location=device)
        if saved.get("format") != TRAINING_STATE_FORMAT:
            raise ValueError(f"unsupported training state: {state_path}")
        if saved.get("config") != config:
            raise ValueError("existing MTRD training state uses a different configuration")
        student.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch = int(saved["epoch"]) + 1
        weights = [float(value) for value in saved["weights"]]
        previous_performance = saved.get("previous_performance")
        best_mean = float(saved["best_mean"])
        history = list(saved.get("history", []))
        _restore_rng_state(saved["rng_state"], torch, np)
        generator = getattr(train_loader, "generator", None)
        if generator is not None and saved.get("loader_generator_state") is not None:
            generator.set_state(saved["loader_generator_state"].cpu())
        print(f"Resumed {state_path} at epoch {start_epoch + 1}")
    else:
        initial_performance = []
        for spec_index, spec in enumerate(specs):
            _set_seed(args.seed + spec_index)
            n_correct, n_total = _accuracy_model(
                student, test_loader, device, spec.noise_type, spec.noise_level
            )
            initial_performance.append(100.0 * n_correct / n_total)
        previous_performance = (
            [value / 100.0 for value in initial_performance]
            if args.balancing_unit == "fraction" else list(initial_performance)
        )
        initial_row: dict[str, Any] = {
            "epoch": 0,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": None,
            "train_noisy_accuracy_percent": None,
            "robust_mean_accuracy_percent": statistics.fmean(initial_performance[1:]),
        }
        for index, spec in enumerate(specs):
            initial_row[f"accuracy_{spec.label}"] = initial_performance[index]
            initial_row[f"balancing_performance_{spec.label}"] = previous_performance[index]
            initial_row[f"weight_used_{spec.label}"] = weights[index]
            initial_row[f"weight_next_{spec.label}"] = weights[index]
        history.append(initial_row)
        _set_seed(args.seed)
        print(
            f"[{args.noise_type}] initialization robust_mean="
            f"{initial_row['robust_mean_accuracy_percent']:.4f}"
        )

    if start_epoch >= args.student_epochs:
        if not output.is_file():
            raise RuntimeError("training state is complete but the best checkpoint is missing")
        print(f"MTRD training already complete: {output}")
        return 0

    for epoch in range(start_epoch, args.student_epochs):
        student.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            nominal_logits = student(inputs, lamda=0.0, noise_type="none")
            noisy_logits = student(
                inputs, lamda=STUDENT_NOISE[args.noise_type],
                noise_type=args.noise_type,
            )
            with torch.no_grad():
                teacher_probabilities = []
                for teacher, spec in zip(teachers, specs):
                    logits = teacher(
                        inputs, lamda=spec.noise_level, noise_type=spec.noise_type
                    )
                    teacher_probabilities.append(
                        torch.softmax(logits / args.temperature, dim=1)
                    )
            student_log_probabilities = torch.log_softmax(
                nominal_logits / args.temperature, dim=1,
            )
            individual_kd = [
                functional.kl_div(
                    student_log_probabilities,
                    probability,
                    reduction="batchmean",
                )
                for probability in teacher_probabilities
            ]
            kd_loss = sum(
                weight * teacher_kd
                for weight, teacher_kd in zip(weights, individual_kd)
            ) * (args.temperature ** 2)
            task_loss = functional.cross_entropy(noisy_logits, targets)
            loss = args.kd_weight * kd_loss + (1.0 - args.kd_weight) * task_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch + 1}, sample {total}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item()) * targets.size(0)
            correct += noisy_logits.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)

        performance = []
        for spec_index, spec in enumerate(specs):
            _set_seed(args.seed + (epoch + 1) * 10_000 + spec_index)
            n_correct, n_total = _accuracy_model(
                student, test_loader, device, spec.noise_type, spec.noise_level
            )
            performance.append(100.0 * n_correct / n_total)
        performance_for_balancing = (
            [value / 100.0 for value in performance]
            if args.balancing_unit == "fraction" else list(performance)
        )
        next_weights = update_balancing_weights(
            performance_for_balancing, previous_performance, args.balancing_policy,
            args.balancing_temperature,
        )
        robust_mean = statistics.fmean(performance[1:])
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": epoch_loss / max(total, 1),
            "train_noisy_accuracy_percent": 100.0 * correct / max(total, 1),
            "robust_mean_accuracy_percent": robust_mean,
        }
        for index, spec in enumerate(specs):
            row[f"accuracy_{spec.label}"] = performance[index]
            row[f"balancing_performance_{spec.label}"] = performance_for_balancing[index]
            row[f"weight_used_{spec.label}"] = weights[index]
            row[f"weight_next_{spec.label}"] = next_weights[index]
        history.append(row)

        payload = {
            "format": SCHEMA_VERSION + ".checkpoint",
            "model_state": student.state_dict(),
            "epoch": epoch + 1,
            "config": config,
            "performance": performance,
            "performance_for_balancing": performance_for_balancing,
            "balancing_unit": args.balancing_unit,
            "weights_used": weights,
            "weights_next": next_weights,
            "teacher_specs": [
                {**asdict(spec), "checkpoint": str(spec.checkpoint)} for spec in specs
            ],
        }
        torch.save(payload, last_output)
        if robust_mean > best_mean:
            best_mean = robust_mean
            torch.save(payload, output)

        previous_performance = performance_for_balancing
        weights = next_weights
        scheduler.step()
        state = {
            "format": TRAINING_STATE_FORMAT,
            "config": config,
            "epoch": epoch,
            "model_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "weights": weights,
            "previous_performance": previous_performance,
            "best_mean": best_mean,
            "history": history,
            "rng_state": _capture_rng_state(torch, np),
            "loader_generator_state": (
                train_loader.generator.get_state()
                if getattr(train_loader, "generator", None) is not None else None
            ),
        }
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, state_path)
        fields = list(history[0])
        atomic_csv(history_path, history, fields)
        print(
            f"[{args.noise_type}] epoch {epoch + 1}/{args.student_epochs} "
            f"loss={row['train_loss']:.5f} robust_mean={robust_mean:.4f} "
            f"weights_next={[round(value, 4) for value in weights]}"
        )
    print(f"MTRD best checkpoint: {output}")
    return 0


def trial_seed(
    base_seed: int, noise_type: str, protocol: str, level_index: int, trial: int,
) -> int:
    device_offset = {"rram": 1_000_000, "pcm": 2_000_000}[noise_type]
    protocol_offset = {
        "software_baseline": 10_000,
        "controlled": 20_000,
        "randomized": 30_000,
    }[protocol]
    return int(base_seed + device_offset + protocol_offset + level_index * 1000 + trial)


def randomized_level(
    base_seed: int, noise_type: str, trial: int, distribution: str,
) -> float:
    rng = random.Random(trial_seed(base_seed, noise_type, "randomized", 0, trial))
    levels = NOISE_LEVELS[noise_type]
    if distribution == "discrete_uniform_protocol_grid":
        return float(rng.choice(levels))
    if distribution == "continuous_uniform_protocol_range":
        return float(rng.uniform(min(levels), max(levels)))
    raise ValueError(f"unknown randomized distribution: {distribution}")


def _method_models(args: argparse.Namespace, noise_type: str, device: Any) -> dict[str, list[Any]]:
    primary, secondary = VARIATION_BASELINES[noise_type]
    paths = {
        "mtrd": [mtrd_checkpoint(args, noise_type)],
        "variation_aware_primary": [teacher_checkpoint(args, noise_type, primary)],
        "variation_aware_secondary": [teacher_checkpoint(args, noise_type, secondary)],
        "five_network_ensemble": [
            teacher_checkpoint(args, noise_type, level)
            for level in NOISE_LEVELS[noise_type]
        ],
        "nominal": [clean_checkpoint(args)],
    }
    return {
        method: [_load_model(path, device) for path in method_paths]
        for method, method_paths in paths.items()
    }


def _method_checkpoint_paths(args: argparse.Namespace, noise_type: str) -> dict[str, list[Path]]:
    primary, secondary = VARIATION_BASELINES[noise_type]
    return {
        "mtrd": [mtrd_checkpoint(args, noise_type)],
        "variation_aware_primary": [teacher_checkpoint(args, noise_type, primary)],
        "variation_aware_secondary": [teacher_checkpoint(args, noise_type, secondary)],
        "five_network_ensemble": [
            teacher_checkpoint(args, noise_type, level)
            for level in NOISE_LEVELS[noise_type]
        ],
        "nominal": [clean_checkpoint(args)],
    }


def _accuracy_method(
    models: Sequence[Any], loader: Any, device: Any, noise_type: str,
    noise_level: float, max_samples: int, realization_policy: str,
) -> tuple[int, int]:
    import torch

    for model in models:
        model.eval()
    correct = 0
    total = 0
    if realization_policy == "fixed_test_set_realization":
        noise_context = _temporary_fixed_weight_noise(models, noise_type, noise_level)
        forward_noise_type, forward_noise_level = "none", 0.0
    elif realization_policy == "per_batch_resample":
        noise_context = contextlib.nullcontext()
        forward_noise_type, forward_noise_level = noise_type, noise_level
    elif realization_policy == "no_noise":
        noise_context = contextlib.nullcontext()
        forward_noise_type, forward_noise_level = "none", 0.0
    else:
        raise ValueError(f"unsupported realization policy: {realization_policy}")
    with noise_context:
        with torch.no_grad():
            for inputs, targets in loader:
                if max_samples and total >= max_samples:
                    break
                if max_samples:
                    remaining = max_samples - total
                    inputs, targets = inputs[:remaining], targets[:remaining]
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = None
                for model in models:
                    current = model(
                        inputs, lamda=forward_noise_level,
                        noise_type=forward_noise_type,
                    )
                    logits = current if logits is None else logits + current
                predictions = logits.argmax(dim=1)
                correct += predictions.eq(targets).sum().item()
                total += targets.size(0)
    if total == 0:
        raise RuntimeError("evaluation loader produced zero samples")
    return int(correct), int(total)


@contextlib.contextmanager
def _temporary_fixed_weight_noise(
    models: Sequence[Any], noise_type: str, noise_level: float,
) -> Iterable[None]:
    """Sample one device realization per model and hold it for the full loader."""
    import torch

    if noise_type not in ("rram", "pcm"):
        raise ValueError(f"fixed realization requires rram or pcm, got {noise_type}")
    backups: list[tuple[Any, Any]] = []
    with torch.no_grad():
        for model in models:
            for module in model.modules():
                if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    continue
                backups.append((module.weight, module.weight.detach().clone()))
                if noise_type == "rram":
                    module.weight.mul_(
                        torch.exp(torch.randn_like(module.weight) * noise_level)
                    )
                else:
                    _install_code_import_path()
                    from models.noisy_layers import pcm_layerwise_wmax

                    standard_deviation = (
                        noise_level * pcm_layerwise_wmax(module.weight).item()
                    )
                    if standard_deviation > 0:
                        module.weight.add_(
                            torch.randn_like(module.weight) * standard_deviation
                        )
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, original in backups:
                parameter.copy_(original)


RAW_FIELDS = (
    "schema_version", "run_id", "created_utc", "dataset", "model",
    "noise_type", "protocol", "method", "method_label", "training_noise",
    "eval_noise", "trial", "seed", "n_samples", "n_correct",
    "accuracy_percent", "checkpoint_paths_json", "checkpoint_sha256_json",
    "realization_policy", "elapsed_seconds",
)


def _training_noise(noise_type: str, method: str) -> str:
    if method == "mtrd":
        return noise_token(STUDENT_NOISE[noise_type])
    if method == "variation_aware_primary":
        return noise_token(VARIATION_BASELINES[noise_type][0])
    if method == "variation_aware_secondary":
        return noise_token(VARIATION_BASELINES[noise_type][1])
    if method == "five_network_ensemble":
        return json.dumps(list(NOISE_LEVELS[noise_type]), separators=(",", ":"))
    return "0"


def _raw_row(
    *, run_id: str, noise_type: str, protocol: str, method: str,
    eval_noise: float, trial: int, seed: int, n_samples: int, n_correct: int,
    paths: Sequence[Path], checkpoint_hashes: Sequence[str],
    realization_policy: str, elapsed_seconds: float,
) -> dict[str, Any]:
    if len(paths) != len(checkpoint_hashes):
        raise ValueError("checkpoint paths and hashes must have equal length")
    accuracy = 100.0 * n_correct / n_samples
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": utc_now(),
        "dataset": "cifar10",
        "model": "vgg16",
        "noise_type": noise_type,
        "protocol": protocol,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "training_noise": _training_noise(noise_type, method),
        "eval_noise": f"{eval_noise:.12g}",
        "trial": trial,
        "seed": seed,
        "n_samples": n_samples,
        "n_correct": n_correct,
        "accuracy_percent": f"{accuracy:.12g}",
        "checkpoint_paths_json": json.dumps([str(path) for path in paths], separators=(",", ":")),
        "checkpoint_sha256_json": json.dumps(list(checkpoint_hashes), separators=(",", ":")),
        "realization_policy": realization_policy,
        "elapsed_seconds": f"{elapsed_seconds:.6f}",
    }


def validate_raw_rows(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("raw result set is empty")
    for index, row in enumerate(rows):
        samples = int(row["n_samples"])
        correct = int(row["n_correct"])
        accuracy = float(row["accuracy_percent"])
        if samples <= 0 or not 0 <= correct <= samples:
            raise ValueError(f"invalid counts at raw row {index}")
        direct = 100.0 * correct / samples
        if not math.isclose(accuracy, direct, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"raw row {index} contains an adjusted metric: {accuracy} != {direct}"
            )


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        eval_noise = row["eval_noise"] if row["protocol"] != "randomized" else "random"
        key = (row["noise_type"], row["protocol"], eval_noise, row["method"])
        groups.setdefault(key, []).append(float(row["accuracy_percent"]))
    summary = []
    for key in sorted(
        groups,
        key=lambda item: (
            ("rram", "pcm").index(item[0]),
            ("software_baseline", "controlled", "randomized").index(item[1]),
            float(item[2]) if item[2] != "random" else math.inf,
            METHOD_ORDER.index(item[3]),
        ),
    ):
        values = groups[key]
        summary.append({
            "noise_type": key[0],
            "protocol": key[1],
            "eval_noise": key[2],
            "method": key[3],
            "method_label": METHOD_LABELS[key[3]],
            "trials": len(values),
            "mean_accuracy_percent": f"{statistics.fmean(values):.12g}",
            "std_accuracy_percent": f"{statistics.stdev(values) if len(values) > 1 else 0.0:.12g}",
            "min_accuracy_percent": f"{min(values):.12g}",
            "max_accuracy_percent": f"{max(values):.12g}",
        })
    return summary


SUMMARY_FIELDS = (
    "noise_type", "protocol", "eval_noise", "method", "method_label",
    "trials", "mean_accuracy_percent", "std_accuracy_percent",
    "min_accuracy_percent", "max_accuracy_percent",
)


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.realization_policy == "unconfirmed":
        raise ValueError(
            "the paper does not specify device-realization lifetime; choose "
            "--realization-policy fixed_test_set_realization or per_batch_resample "
            "explicitly so the decision is recorded"
        )
    args.stage = "evaluate"
    report, ready = run_preflight(args, emit=False)
    if not ready:
        failed = [item for item in report["checks"] if not item["ok"]]
        raise RuntimeError(
            "evaluation preflight failed:\n" + "\n".join(
                f"- {item.get('path') or item.get('name')}" for item in failed
            )
        )
    output_dir = Path(args.work_dir) / "evaluation"
    raw_path = output_dir / "raw.csv"
    if raw_path.exists() and not args.overwrite:
        raise FileExistsError(f"{raw_path} exists; pass --overwrite to replace it")

    _install_code_import_path()
    from utils.data import get_classification_loaders

    _set_seed(args.seed)
    device = _device(args.device)
    _, test_loader, _ = get_classification_loaders(
        "cifar10", args.data_root, args.batch_size, args.num_workers, seed=args.seed,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    started_utc = utc_now()
    for noise_type in args.noise_types:
        print(f"Loading {noise_type.upper()} methods on {device}")
        models = _method_models(args, noise_type, device)
        paths = _method_checkpoint_paths(args, noise_type)
        checkpoint_hashes = {
            method: [sha256_file(path) for path in method_paths]
            for method, method_paths in paths.items()
        }

        baseline_seed = trial_seed(args.seed, noise_type, "software_baseline", 0, 0)
        _set_seed(baseline_seed)
        started = time.time()
        correct, total = _accuracy_method(
            models["nominal"], test_loader, device, "none", 0.0,
            args.max_samples, "no_noise",
        )
        rows.append(_raw_row(
            run_id=run_id, noise_type=noise_type, protocol="software_baseline",
            method="nominal", eval_noise=0.0, trial=0, seed=baseline_seed,
            n_samples=total, n_correct=correct, paths=paths["nominal"],
            checkpoint_hashes=checkpoint_hashes["nominal"],
            realization_policy="no_noise",
            elapsed_seconds=time.time() - started,
        ))

        for level_index, level in enumerate(NOISE_LEVELS[noise_type]):
            for trial in range(args.controlled_trials):
                seed = trial_seed(args.seed, noise_type, "controlled", level_index, trial)
                for method in METHOD_ORDER:
                    _set_seed(seed)
                    started = time.time()
                    correct, total = _accuracy_method(
                        models[method], test_loader, device, noise_type, level,
                        args.max_samples, args.realization_policy,
                    )
                    rows.append(_raw_row(
                        run_id=run_id, noise_type=noise_type, protocol="controlled",
                        method=method, eval_noise=level, trial=trial, seed=seed,
                        n_samples=total, n_correct=correct, paths=paths[method],
                        checkpoint_hashes=checkpoint_hashes[method],
                        realization_policy=args.realization_policy,
                        elapsed_seconds=time.time() - started,
                    ))
                print(
                    f"[{noise_type}] controlled level={level:g} "
                    f"trial={trial + 1}/{args.controlled_trials}"
                )

        for trial in range(args.randomized_trials):
            level = randomized_level(
                args.seed, noise_type, trial, args.randomized_distribution
            )
            seed = trial_seed(args.seed, noise_type, "randomized", 1, trial)
            for method in METHOD_ORDER:
                _set_seed(seed)
                started = time.time()
                correct, total = _accuracy_method(
                    models[method], test_loader, device, noise_type, level,
                    args.max_samples, args.realization_policy,
                )
                rows.append(_raw_row(
                    run_id=run_id, noise_type=noise_type, protocol="randomized",
                    method=method, eval_noise=level, trial=trial, seed=seed,
                    n_samples=total, n_correct=correct, paths=paths[method],
                    checkpoint_hashes=checkpoint_hashes[method],
                    realization_policy=args.realization_policy,
                    elapsed_seconds=time.time() - started,
                ))
            print(
                f"[{noise_type}] randomized level={level:g} "
                f"trial={trial + 1}/{args.randomized_trials}"
            )
        del models

    validate_raw_rows(rows)
    summary = summarize_rows(rows)
    atomic_csv(raw_path, rows, RAW_FIELDS)
    atomic_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_id": run_id,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "command": sys.argv,
        "protocol": protocol_manifest(args),
        "preflight": report,
        "raw_results": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
        "summary": {
            "path": str(output_dir / "summary.csv"),
            "sha256": sha256_file(output_dir / "summary.csv"),
        },
        "max_samples": args.max_samples or None,
        "device": str(device),
        "realization_policy": args.realization_policy,
        "numerical_reproduction_verified": False,
        "author_raw_reference": None,
        "verification_blockers": [
            "author raw per-trial values and an approved tolerance are unavailable",
            "the Eq. (6) performance unit/sign and device realization lifetime require author confirmation",
        ],
    }
    training_manifest = Path(args.work_dir) / "training" / "manifest.json"
    manifest["training_manifest"] = (
        {
            "path": str(training_manifest),
            "sha256": sha256_file(training_manifest),
            "content": json.loads(training_manifest.read_text(encoding="utf-8")),
        }
        if training_manifest.is_file() else None
    )
    atomic_json(output_dir / "manifest.json", manifest)
    print(f"Raw measurements: {raw_path}")
    print(f"Summary: {output_dir / 'summary.csv'}")
    return 0


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "checkpoint base containing cifar10/vgg16; omit while training to use "
            "<work-dir>/checkpoints, or point evaluation at a read-only Data Asset"
        ),
    )


def add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="auto")


def add_noise_types(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--noise-types", nargs="+", choices=("rram", "pcm"),
        default=["rram", "pcm"],
    )


def add_training_options(parser: argparse.ArgumentParser) -> None:
    add_paths(parser)
    add_runtime(parser)
    add_noise_types(parser)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--teacher-epochs", type=int, default=200)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--student-epochs", type=int, default=300)
    parser.add_argument("--student-lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--kd-weight", type=float, default=0.7)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--balancing-policy", choices=BALANCING_POLICIES,
        default="positive_delta_softmax",
    )
    parser.add_argument("--balancing-temperature", type=float, default=1.0)
    parser.add_argument(
        "--balancing-unit",
        choices=("unconfirmed", "fraction", "percentage_points"),
        default="unconfirmed",
        help="must be selected explicitly because Eq. (6) does not define the scale of p",
    )


def add_evaluation_options(parser: argparse.ArgumentParser) -> None:
    add_paths(parser)
    add_runtime(parser)
    add_noise_types(parser)
    parser.add_argument(
        "--balancing-policy", choices=BALANCING_POLICIES,
        default="positive_delta_softmax",
        help="select the trained MTRD checkpoint matching this policy",
    )
    parser.add_argument(
        "--checkpoint-role-manifest", type=Path, required=True,
        help="manifest binding every evaluated checkpoint role to its path, size, and SHA-256",
    )
    parser.add_argument("--controlled-trials", type=int, default=20)
    parser.add_argument("--randomized-trials", type=int, default=20)
    parser.add_argument(
        "--randomized-distribution",
        choices=(
            "discrete_uniform_protocol_grid",
            "continuous_uniform_protocol_range",
        ),
        default="discrete_uniform_protocol_grid",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="0 evaluates the complete CIFAR-10 test set; positive values produce partial evaluations",
    )
    parser.add_argument(
        "--realization-policy",
        choices=(
            "unconfirmed", "fixed_test_set_realization", "per_batch_resample",
        ),
        default="unconfirmed",
        help="must be selected explicitly for a formal run; the paper leaves this lifetime ambiguous",
    )
    parser.add_argument("--overwrite", action="store_true")


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "teacher_epochs", "student_epochs", "controlled_trials", "randomized_trials"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if hasattr(args, "num_workers") and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if hasattr(args, "max_samples") and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if hasattr(args, "kd_weight") and not 0.0 <= args.kd_weight <= 1.0:
        raise ValueError("--kd-weight must be in [0, 1]")
    for name in (
        "teacher_lr", "student_lr", "temperature", "balancing_temperature",
        "grad_clip",
    ):
        if hasattr(args, name):
            value = float(getattr(args, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")


def _build_internal_mtrd_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    add_paths(parser)
    add_runtime(parser)
    parser.add_argument("--noise-type", choices=("rram", "pcm"), required=True)
    parser.add_argument("--student-epochs", type=int, default=300)
    parser.add_argument("--student-lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--kd-weight", type=float, default=0.7)
    parser.add_argument("--balancing-policy", choices=BALANCING_POLICIES, default="positive_delta_softmax")
    parser.add_argument("--balancing-temperature", type=float, default=1.0)
    parser.add_argument(
        "--balancing-unit",
        choices=("unconfirmed", "fraction", "percentage_points"),
        default="unconfirmed",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--restart", action="store_true")
    parser.set_defaults(func=_train_mtrd_device)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Complete CIFAR-10 VGG16 PyTorch training and evaluation chain."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check code, packages, data, and checkpoints")
    add_paths(preflight)
    add_noise_types(preflight)
    preflight.add_argument("--stage", choices=("train", "evaluate"), default="train")
    preflight.add_argument("--preflight-json", type=Path)
    preflight.add_argument("--seed", type=int, default=2025)
    preflight.add_argument(
        "--balancing-policy", choices=BALANCING_POLICIES,
        default="positive_delta_softmax",
    )
    preflight.add_argument("--checkpoint-role-manifest", type=Path)
    preflight.set_defaults(func=lambda args: 0 if run_preflight(args)[1] else 2)

    train = subparsers.add_parser("train", help="train clean, variation-aware, and MTRD checkpoints")
    add_training_options(train)
    train.add_argument("--force", action="store_true")
    train.add_argument("--preflight-json", type=Path)
    train.set_defaults(stage="train", func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="run controlled and randomized five-method tests")
    add_evaluation_options(evaluate)
    evaluate.add_argument("--preflight-json", type=Path)
    evaluate.set_defaults(stage="evaluate", func=cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_train-mtrd":
        parser = _build_internal_mtrd_parser()
        args = parser.parse_args(arguments[1:])
        validate_args(args)
        return int(args.func(args))
    parser = build_parser()
    args = parser.parse_args(arguments)
    validate_args(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
