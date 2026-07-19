"""Training and simulator-backed evaluation for image restoration tasks.

Training implements the manuscript's weight-variation equations and the
cross-epoch teacher balancer. Evaluation is deliberately fail-closed: the
public package records every selected mapping and quantization decision.
The NeuroSim path is a pinned-source-gated PyTorch Eq. (1) extension rather
than the upstream native CIM kernel, and AIHWKit 1.1 cannot convert UNet's
``ConvTranspose2d`` layers.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from models import get_model
from models.noisy_layers import NoisyConv2d, NoisyLinear, pcm_layerwise_wmax
from utils.checkpoints import (
    extract_state_dict,
    load_checkpoint,
    remap_legacy_dncnn_state_dict,
    remap_legacy_unet_state_dict,
    state_dict_finiteness,
)
from utils.source_tree import source_tree_sha256


SCHEMA = "mtrd.image-workflows.dncnn-unet.v2"
# This namespace is intentionally retained only to replay historical fixed
# trials. Changing it would change programmed conductance draws, including the
# audited sigma=0.5 UNet exemplar.
LEGACY_EVALUATION_SEED_NAMESPACE = "fig5-evaluation"
TASKS = ("denoising", "segmentation")
DEVICE_MODELS = ("rram", "pcm")
RRAM_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)
PCM_LEVELS = (0.02, 0.04, 0.06, 0.08, 0.10)
STUDENT_LEVELS = {"rram": 0.3, "pcm": 0.06}
AIHWKIT_PIN = "1.1.0"
CODE_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MASK_SUFFIXES = {".gif", ".png", ".jpg", ".jpeg"}
CARVANA_RESIZE_BACKENDS = {"pil-bilinear", "released-opencv"}
EQ6_CHOICES = {
    "equation_literal_current_minus_previous": 1.0,
    "narrative_underperformance_previous_minus_current": -1.0,
    "released_rank_underperformance": 0.0,
}
DNCNN_MTRD_PROFILES = {
    "released_bh4_source_semantics",
    "paper_interpreted_residual_mse_v1",
}
DNCNN_PAPER_EQ6_CHOICES = {
    "equation_literal_current_minus_previous",
    "narrative_underperformance_previous_minus_current",
}
DNCNN_SET12_PREPROCESSING = {
    "reader": "cv2.imread(path)",
    "color_order": "BGR",
    "channel": "index 0 (B)",
    "dtype": "float32",
    "normalization": "divide by 255.0",
    "source": "DnCNN-PyTorch-master_NC/dataset.py and released test scripts",
}
DENOISING_H5_MODES = {
    "regenerated-from-raw",
    "source-provided",
}
DENOISING_H5_LAYOUT = "released-flat-sequential-float32-v1"
DNCNN_TRAINING_RESIZE = {
    "backend": "cv2.resize",
    "interpolation": "cv2.INTER_CUBIC",
    "scales": [1.0, 0.9, 0.8, 0.7],
    "dsize_expression": "(int(source_height * scale), int(source_width * scale))",
    "dsize_argument_order": "released_literal_height_then_width",
    "source": "DnCNN-PyTorch-master_NC/dataset.py:42",
}
PCM_SCOPES = {"fixed_trial"}
RRAM_SCOPES = {"fixed_trial"}
SEGMENTATION_METRIC_CONTRACTS: dict[str, dict[str, object]] = {
    "per_image_mean": {
        "metric_aggregation": "per_image_mean",
        "observation_unit": "image",
        "observation_metric": "Dice_fraction",
        "summary_reduction": "arithmetic_mean_over_per_image_dice",
        "prediction_rule": "sigmoid(logits) > 0.5",
        "empty_prediction_and_target": "Dice is one through symmetric epsilon smoothing",
    },
    "released_batch_global": {
        "metric_aggregation": "released_batch_global",
        "observation_unit": "batch",
        "observation_metric": "Dice_batch_global_fraction",
        "summary_reduction": "arithmetic_mean_over_per_batch_global_dice",
        "prediction_rule": "sigmoid(logits) > 0.5",
        "observation_formula": (
            "2*sum(prediction*target)/(sum(prediction)+sum(target)+1e-8)"
        ),
        "empty_prediction_and_target": (
            "Dice is zero because the released batch-global formula smooths only "
            "the denominator"
        ),
    },
}
SEGMENTATION_MTRD_BN_UPDATE_POLICIES: dict[str, dict[str, object]] = {
    "legacy_dual_branch": {
        "policy": "legacy_dual_branch",
        "nominal_kd_branch_batch_norm_buffers": "updated",
        "noisy_task_branch_batch_norm_buffers": "updated",
        "nominal_kd_implementation": "regular training-mode module forward",
        "compatibility": "preserves the previous unified implementation",
        "warning": (
            "UNet MTRD uses the legacy dual-branch BatchNorm update policy: both the "
            "nominal KD and noisy task forwards update persistent BatchNorm buffers."
        ),
    },
    "noisy_task_only": {
        "policy": "noisy_task_only",
        "nominal_kd_branch_batch_norm_buffers": "not_updated",
        "noisy_task_branch_batch_norm_buffers": "updated",
        "nominal_kd_implementation": (
            "functional call with cloned persistent buffers and original parameters"
        ),
        "compatibility": "alternative diagnostic policy; not author verified",
        "warning": (
            "UNet MTRD updates persistent BatchNorm buffers only through the noisy task "
            "forward; the nominal KD forward retains parameter gradients using cloned buffers."
        ),
    },
}
UNET_OPTIMIZER_PROFILES: dict[str, dict[str, object]] = {
    "manuscript-stated-sgd": {
        "profile": "manuscript-stated-sgd",
        "roles": {"teacher": "SGD", "student": "SGD"},
        "optimizer": {
            "name": "SGD",
            "class": "torch.optim.SGD",
            "learning_rate": 1e-4,
            "kwargs": {
                "momentum": 0.0,
                "weight_decay": 0.0,
                "dampening": 0.0,
                "nesterov": False,
            },
        },
        "provenance": (
            "The manuscript states SGD with learning rate 1e-4 for UNet, but does not "
            "fully specify all optimizer hyperparameters."
        ),
        "author_verified": False,
        "warning": (
            "UNet uses the manuscript-stated SGD optimizer profile. Momentum and the "
            "remaining training protocol details are not author verified."
        ),
    },
    "released-source-adam": {
        "profile": "released-source-adam",
        "roles": {"teacher": "Adam", "student": "Adam"},
        "optimizer": {
            "name": "Adam",
            "class": "torch.optim.Adam",
            "learning_rate": 1e-4,
            "kwargs": {
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
        },
        "provenance": (
            "The released UNet training scripts instantiate torch.optim.Adam with "
            "learning rate 1e-4 and otherwise use PyTorch defaults."
        ),
        "author_verified": False,
        "warning": (
            "UNet uses the released-source Adam optimizer profile. This is an "
            "optimizer-only reconstruction: the released scripts do not define a "
            "complete unified MTRD UNet training specification."
        ),
    },
}
UNET_LEGACY_OPTIMIZER_KEYS = {
    "teacher_optimizer",
    "student_optimizer",
    "learning_rate",
    "momentum",
    "weight_decay",
}
PROTOCOL_WARNING = (
    "The manuscript does not fully specify the Eq. (6) sign, evaluation trial count, "
    "random seeds, or PCM realization scope. Author confirmation is required before "
    "claiming exact numerical reproduction. The released single-output UNet uses "
    "sigmoid/BCE, which differs from the manuscript's stated pixel-wise "
    "softmax/cross-entropy objective and also requires author confirmation."
)


class PreflightError(RuntimeError):
    """Raised after a complete fail-closed preflight report is assembled."""


@dataclass(frozen=True)
class TeacherSpec:
    label: str
    level: float
    clean: bool = False


@dataclass(frozen=True)
class DnCNNMTRDContract:
    """Immutable, task-local DnCNN MTRD training contract."""

    name: str
    device_model: str
    mtrd_teacher_specs: tuple[TeacherSpec, ...]
    preparation_teacher_levels: tuple[float, ...]
    student_initial_level: float
    include_clean_teacher: bool
    teacher_epochs: int
    student_epochs: int
    optimizer: str
    optimizer_kwargs: Mapping[str, object]
    initial_learning_rate: float
    teacher_lr_milestones: tuple[int, ...]
    student_lr_milestones: tuple[int, ...]
    eq4_alpha: float
    distillation_temperature: float
    eq6_direction: str
    eq6_temperature: float | None
    loss_name: str
    loss_reduction: str
    kd_is_degenerate: bool
    author_confirmation_required: bool
    source_files: tuple[str, ...]
    feedback_noise_policy: str
    student_training_forward: str
    source_invocation: str | None
    source_default_ambiguity: str | None


@dataclass(frozen=True)
class SegmentationMTRDContract:
    """Immutable, material UNet MTRD training semantics for one device model."""

    name: str
    device_model: str
    mtrd_teacher_specs: tuple[TeacherSpec, ...]
    student_initial_level: float
    include_clean_teacher: bool
    teacher_epochs: int
    student_epochs: int
    checkpoint_selection: str
    training_batch_size: int
    image_height: int
    image_width: int
    resize_backend: str
    feedback_split: str
    feedback_sample_count: int
    eq4_alpha: float
    distillation_temperature: float
    eq6_direction: str
    eq6_temperature: float
    teacher_optimizer: Mapping[str, object]
    student_optimizer: Mapping[str, object]
    metric_contract: Mapping[str, object]
    batch_norm_update_policy: Mapping[str, object]
    model_output: str
    teacher_loss: str
    distillation_loss: str
    noisy_task_loss: str
    total_loss: str
    nominal_student_forward: str
    noisy_student_forward: str
    variation_teacher_forward: str
    training_noise_realization: str
    author_confirmation_required: bool


@dataclass(frozen=True)
class CarvanaRecord:
    sample_id: str
    image: Path
    mask: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "\x1f".join([str(int(base)), *(str(part) for part in parts)])
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return value % (2**31 - 1)


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_execution_device(value: str | torch.device) -> torch.device:
    """Resolve an explicit evaluation device without silently falling back."""
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid torch execution device: {value!r}") from error
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError("image-workflow evaluation supports only cpu or cuda devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA evaluation device {device} is unavailable; "
            f"visible devices: {torch.cuda.device_count()}"
        )
    return device


@contextlib.contextmanager
def seeded_forward(seed: int, device: torch.device) -> Iterator[None]:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _expanded(value: object) -> object:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expanded(item) for item in value]
    if isinstance(value, dict):
        return {key: _expanded(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a JSON object")
    config = _expanded(config)
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("configuration schema_version must be 1")
    return config, config_path


def write_json(path: str | Path, value: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    return target


def append_csv(path: str | Path, row: Mapping[str, object], fields: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.is_file() and target.stat().st_size > 0
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_identity() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    identity: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "numpy": package_version("numpy"),
        "pillow": package_version("Pillow"),
        "opencv_python_headless": package_version("opencv-python-headless"),
        "h5py": package_version("h5py"),
        "aihwkit": package_version("aihwkit"),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
    }
    if cuda_available:
        identity.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_device_capability": list(
                torch.cuda.get_device_capability(torch.cuda.current_device())
            ),
        })
    return identity


def git_identity(repo: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "root": str(repo.resolve()),
        "first_party_source_sha256": source_tree_sha256(CODE_ROOT),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        result.update({"commit": commit, "dirty": bool(dirty)})
    except (OSError, subprocess.CalledProcessError):
        result.update({"commit": None, "dirty": None})
    return result


def levels(config: Mapping[str, Any], device_model: str) -> tuple[float, ...]:
    key = f"{device_model}_levels"
    fallback = RRAM_LEVELS if device_model == "rram" else PCM_LEVELS
    values = tuple(float(item) for item in config["protocol"].get(key, fallback))
    if values != fallback:
        raise ValueError(f"{key} must equal the manuscript grid {fallback}, found {values}")
    return values


def student_level(config: Mapping[str, Any], device_model: str) -> float:
    key = f"{device_model}_student_level"
    value = float(config["protocol"].get(key, STUDENT_LEVELS[device_model]))
    if not math.isclose(value, STUDENT_LEVELS[device_model], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{key} must be {STUDENT_LEVELS[device_model]:g}")
    return value


def _validated_dncnn_common_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config["protocol"].get("denoising")
    if not isinstance(settings, Mapping):
        raise ValueError("protocol.denoising must be an object")
    if (
        int(settings.get("teacher_epochs", -1)) != 50
        or int(settings.get("student_epochs", -1)) != 80
        or settings.get("teacher_optimizer") != "adam"
        or settings.get("student_optimizer") != "adam"
        or not math.isclose(float(settings.get("initial_learning_rate", -1)), 1e-3)
    ):
        raise ValueError(
            "DnCNN protocol must use teacher=50/student=80, Adam, and lr=1e-3"
        )
    if settings.get("checkpoint_selection") != "final_epoch":
        raise ValueError("DnCNN checkpoint_selection must be final_epoch")
    if settings.get("feedback_split") not in {"set12_test", "berkeley_training"}:
        raise ValueError(
            "DnCNN feedback_split must explicitly be set12_test or berkeley_training"
        )
    if int(settings.get("feedback_sample_count", -1)) < 0:
        raise ValueError(
            "DnCNN feedback_sample_count must be >=0; zero means full split"
        )
    return settings


def _require_exact_float(value: object, expected: float, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be {expected:g}") from error
    if not math.isclose(resolved, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} must be {expected:g}")
    return resolved


def resolve_dncnn_mtrd_contract(
    config: Mapping[str, Any], device_model: str,
) -> DnCNNMTRDContract:
    """Resolve one explicit DnCNN MTRD profile without reading UNet settings."""
    if device_model not in DEVICE_MODELS:
        raise ValueError(f"unsupported device model for DnCNN: {device_model}")
    settings = _validated_dncnn_common_settings(config)
    profile = settings.get("mtrd_profile")
    if profile not in DNCNN_MTRD_PROFILES:
        raise ValueError(
            "protocol.denoising.mtrd_profile must be one of "
            f"{sorted(DNCNN_MTRD_PROFILES)}"
        )
    parameters = settings.get("mtrd_profile_parameters")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise ValueError("protocol.denoising.mtrd_profile_parameters must be an object")

    grid = levels(config, device_model)
    initial = student_level(config, device_model)
    optimizer_kwargs: Mapping[str, object] = {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
    }
    if profile == "released_bh4_source_semantics":
        if parameters:
            raise ValueError(
                "released_bh4_source_semantics fixes its source constants; "
                "mtrd_profile_parameters must be empty"
            )
        if tuple(int(item) for item in settings.get("teacher_lr_milestones", [])) != (30,):
            raise ValueError("BH4 DnCNN teacher_lr_milestones must be [30]")
        if tuple(int(item) for item in settings.get("student_lr_milestones", [])) != (30,):
            raise ValueError("BH4 DnCNN student_lr_milestones must be [30]")
        pool = tuple(
            TeacherSpec(f"{value:g}", value, False)
            for value in grid
            if not math.isclose(value, initial, rel_tol=0.0, abs_tol=1e-12)
        )
        if len(pool) != 4:
            raise AssertionError("the fixed manuscript grid must yield four BH4 teachers")
        source_files = (
            "denosing/train_kd_NC_bh.py",
            "denosing/train_kd_pcm_NC_bh.py",
            "denosing/train_nonid.py",
            "denosing/train_nonid_pcm.py",
            "denosing/train.py",
            "denosing/models.py",
            "denosing/dataset.py",
            "denosing/utils.py",
            "denosing/README.md",
            "denosing/test_quant.py",
        )
        return DnCNNMTRDContract(
            name=profile,
            device_model=device_model,
            mtrd_teacher_specs=pool,
            preparation_teacher_levels=grid,
            student_initial_level=initial,
            include_clean_teacher=False,
            teacher_epochs=50,
            student_epochs=80,
            optimizer="Adam",
            optimizer_kwargs=optimizer_kwargs,
            initial_learning_rate=1e-3,
            teacher_lr_milestones=(30,),
            student_lr_milestones=(30,),
            eq4_alpha=0.7,
            distillation_temperature=5.0,
            eq6_direction="released_bh4_rank_current_minus_previous_zero_baseline",
            eq6_temperature=None,
            loss_name="released_single_channel_kl_plus_sum_mse",
            loss_reduction="kl_mean_plus_mse_sum_divided_by_2_batch",
            kd_is_degenerate=True,
            author_confirmation_required=True,
            source_files=source_files,
            feedback_noise_policy=(
                "fresh_deterministic_per_epoch_and_teacher_level_for_input_and_device_noise"
            ),
            student_training_forward=(
                "intermediate_variation_level_from_documented_readme_invocation"
            ),
            source_invocation=(
                "CUDA_VISIBLE_DEVICES=0 python train_kd_NC_bh.py "
                "--num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --w_noiseL 0.3"
                if device_model == "rram" else
                "CUDA_VISIBLE_DEVICES=2 python train_kd_pcm_NC_bh.py "
                "--num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --w_noiseL 0.06"
            ),
            source_default_ambiguity=(
                "The script parser defaults --w_noiseL to 0.0, but the released "
                "README invocation specifies the intermediate 0.3/0.06 level; "
                "this profile follows the documented invocation."
            ),
        )

    required = {
        "eq4_alpha",
        "distillation_temperature",
        "eq6_direction",
        "eq6_temperature",
        "include_clean_teacher",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(
            "paper_interpreted_residual_mse_v1 requires explicit "
            f"mtrd_profile_parameters: {missing}"
        )
    unknown = sorted(set(parameters) - required)
    if unknown:
        raise ValueError(
            "paper_interpreted_residual_mse_v1 does not accept unknown "
            f"mtrd_profile_parameters: {unknown}"
        )
    alpha = _require_exact_float(parameters["eq4_alpha"], 0.7, "DnCNN Eq.(4) alpha")
    temperature = _require_exact_float(
        parameters["distillation_temperature"], 5.0,
        "DnCNN distillation temperature",
    )
    direction = parameters["eq6_direction"]
    if direction not in DNCNN_PAPER_EQ6_CHOICES:
        raise ValueError(
            "paper_interpreted_residual_mse_v1 Eq.(6) direction must be one of "
            f"{sorted(DNCNN_PAPER_EQ6_CHOICES)}"
        )
    eq6_temperature = float(parameters["eq6_temperature"])
    if not math.isfinite(eq6_temperature) or eq6_temperature <= 0:
        raise ValueError("DnCNN Eq.(6) temperature must be positive")
    if parameters["include_clean_teacher"] is not True:
        raise ValueError(
            "paper_interpreted_residual_mse_v1 requires include_clean_teacher=true"
        )
    if tuple(int(item) for item in settings.get("teacher_lr_milestones", [])) != (30,):
        raise ValueError("interpreted DnCNN teacher_lr_milestones must be [30]")
    if tuple(int(item) for item in settings.get("student_lr_milestones", [])) != (30, 50):
        raise ValueError(
            "interpreted DnCNN student_lr_milestones must be [30, 50]"
        )
    return DnCNNMTRDContract(
        name=profile,
        device_model=device_model,
        mtrd_teacher_specs=(
            TeacherSpec("clean", 0.0, True),
            *(TeacherSpec(f"{value:g}", value, False) for value in grid),
        ),
        preparation_teacher_levels=grid,
        student_initial_level=initial,
        include_clean_teacher=True,
        teacher_epochs=50,
        student_epochs=80,
        optimizer="Adam",
        optimizer_kwargs=optimizer_kwargs,
        initial_learning_rate=1e-3,
        teacher_lr_milestones=(30,),
        student_lr_milestones=(30, 50),
        eq4_alpha=alpha,
        distillation_temperature=temperature,
        eq6_direction=str(direction),
        eq6_temperature=eq6_temperature,
        loss_name="beta_weighted_per_teacher_residual_mse_plus_noisy_task_mse",
        loss_reduction="mean",
        kd_is_degenerate=False,
        author_confirmation_required=True,
        source_files=(),
        feedback_noise_policy="fixed_set12_input_noise",
        student_training_forward=(
            "nominal_residual_distillation_plus_initial_variation_task_path"
        ),
        source_invocation=None,
        source_default_ambiguity=None,
    )


def dncnn_mtrd_contract_payload(contract: DnCNNMTRDContract) -> dict[str, object]:
    """Return a JSON-safe resolved DnCNN contract for manifests and resumes."""
    return {
        "name": contract.name,
        "device_model": contract.device_model,
        "mtrd_teacher_pool": [asdict(spec) for spec in contract.mtrd_teacher_specs],
        "preparation_teacher_levels": list(contract.preparation_teacher_levels),
        "student_initial_level": contract.student_initial_level,
        "include_clean_teacher": contract.include_clean_teacher,
        "teacher_epochs": contract.teacher_epochs,
        "student_epochs": contract.student_epochs,
        "optimizer": contract.optimizer,
        "optimizer_kwargs": dict(contract.optimizer_kwargs),
        "initial_learning_rate": contract.initial_learning_rate,
        "teacher_lr_milestones": list(contract.teacher_lr_milestones),
        "student_lr_milestones": list(contract.student_lr_milestones),
        "eq4_alpha": contract.eq4_alpha,
        "distillation_temperature": contract.distillation_temperature,
        "eq6_direction": contract.eq6_direction,
        "eq6_temperature": contract.eq6_temperature,
        "loss_name": contract.loss_name,
        "loss_reduction": contract.loss_reduction,
        "kd_is_degenerate": contract.kd_is_degenerate,
        "author_confirmation_required": contract.author_confirmation_required,
        "source_files": list(contract.source_files),
        "feedback_noise_policy": contract.feedback_noise_policy,
        "student_training_forward": contract.student_training_forward,
        "source_invocation": contract.source_invocation,
        "source_default_ambiguity": contract.source_default_ambiguity,
    }


def teacher_specs(config: Mapping[str, Any], device_model: str) -> tuple[TeacherSpec, ...]:
    specs: list[TeacherSpec] = []
    if bool(config["protocol"].get("include_clean_teacher", False)):
        specs.append(TeacherSpec("clean", 0.0, True))
    specs.extend(TeacherSpec(f"{value:g}", value, False) for value in levels(config, device_model))
    return tuple(specs)


def mtrd_teacher_specs(
    config: Mapping[str, Any], task: str, device_model: str,
) -> tuple[TeacherSpec, ...]:
    """Return the teacher pool consumed by the selected task's MTRD student."""
    if task == "denoising":
        return resolve_dncnn_mtrd_contract(config, device_model).mtrd_teacher_specs
    return teacher_specs(config, device_model)


def mtrd_prerequisite_specs(
    config: Mapping[str, Any], task: str, device_model: str,
) -> tuple[TeacherSpec, ...]:
    """Return all checkpoints that must exist before a task-local MTRD stage."""
    if task != "denoising":
        return teacher_specs(config, device_model)
    contract = resolve_dncnn_mtrd_contract(config, device_model)
    specs: list[TeacherSpec] = []
    if contract.include_clean_teacher:
        specs.append(TeacherSpec("clean", 0.0, True))
    specs.extend(
        TeacherSpec(f"{value:g}", value, False)
        for value in contract.preparation_teacher_levels
    )
    return tuple(specs)


def noise_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def checkpoint_path(
    config: Mapping[str, Any], task: str, role: str,
    device_model: str | None = None, level: float | None = None,
) -> Path:
    root = Path(config["checkpoint_root"])
    if role == "clean":
        return root / task / "clean" / "teacher.pth"
    if device_model not in DEVICE_MODELS:
        raise ValueError("device_model is required for non-clean checkpoints")
    if role == "teacher":
        if level is None:
            raise ValueError("teacher level is required")
        return root / task / device_model / f"teacher_{noise_token(level)}.pth"
    if role == "mtrd":
        return root / task / device_model / "mtrd.pth"
    raise ValueError(f"unknown checkpoint role: {role}")


def evaluation_checkpoint(config: Mapping[str, Any], task: str, device_model: str, role: str) -> Path:
    mapping = config.get("evaluation_checkpoints", {}).get(task, {})
    key = "nominal" if role == "nominal" else f"{device_model}_mtrd"
    value = mapping.get(key)
    if value:
        return Path(value)
    return checkpoint_path(
        config, task, "clean" if role == "nominal" else "mtrd",
        None if role == "nominal" else device_model,
    )


def build_model(task: str) -> nn.Module:
    if task == "denoising":
        return get_model("dncnn", channels=1, num_of_layers=17)
    if task == "segmentation":
        return get_model("unet", in_channels=3, out_channels=1)
    raise ValueError(f"unknown task: {task}")


def load_model_strict(path: str | Path, task: str, device: torch.device | str = "cpu") -> nn.Module:
    model = build_model(task)
    payload = load_checkpoint(path, map_location="cpu")
    state = extract_state_dict(payload)
    nonfinite, total = state_dict_finiteness(state)
    if nonfinite:
        raise ValueError(f"checkpoint has {nonfinite}/{total} non-finite floating values: {path}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as direct_error:
        mapped = (
            remap_legacy_dncnn_state_dict(state, num_layers=17)
            if task == "denoising"
            else remap_legacy_unet_state_dict(state)
        )
        if tuple(mapped) == tuple(state):
            raise direct_error
        model.load_state_dict(mapped, strict=True)
    return model.to(device)


def _checkpoint_epoch(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        resolved = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if resolved <= 0 or not math.isfinite(numeric) or numeric != float(resolved):
        raise ValueError(f"{label} must be a positive integer")
    return resolved


def _checkpoint_noise_level(value: object, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be a finite number")
    return resolved


def _validate_checkpoint_outer_identity(
    payload: Mapping[str, object], *, task: str, checkpoint_role: str,
    device_model: str, noise_level: float, epoch: int, label: str,
) -> dict[str, object]:
    """Validate fields that distinguish same-architecture checkpoint roles."""
    required = ("task", "role", "device_model", "noise_level", "epoch")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(
            f"{label} checkpoint has incomplete outer identity; missing "
            + ", ".join(missing)
        )
    if payload["task"] != task:
        raise ValueError(
            f"{label} checkpoint task mismatch: expected {task!r}, found {payload['task']!r}"
        )
    if payload["role"] != checkpoint_role:
        raise ValueError(
            f"{label} checkpoint role mismatch: expected {checkpoint_role!r}, "
            f"found {payload['role']!r}"
        )
    if payload["device_model"] != device_model:
        raise ValueError(
            f"{label} checkpoint device-model mismatch: expected {device_model!r}, "
            f"found {payload['device_model']!r}"
        )
    embedded_noise_level = _checkpoint_noise_level(
        payload["noise_level"], f"{label} checkpoint noise_level",
    )
    if not math.isclose(
        embedded_noise_level, noise_level, rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError(
            f"{label} checkpoint noise-level mismatch: expected {noise_level:g}, "
            f"found {embedded_noise_level:g}"
        )
    embedded_epoch = _checkpoint_epoch(payload["epoch"], f"{label} checkpoint epoch")
    if embedded_epoch != epoch:
        raise ValueError(
            f"{label} checkpoint final-epoch mismatch: expected {epoch}, "
            f"found {embedded_epoch}"
        )
    return {
        "embedded_task": payload["task"],
        "embedded_checkpoint_role": payload["role"],
        "embedded_device_model": payload["device_model"],
        "embedded_noise_level": embedded_noise_level,
        "embedded_epoch": embedded_epoch,
    }


def dncnn_checkpoint_profile_identity(
    path: str | Path, expected: DnCNNMTRDContract, *, role: str,
    expected_level: float | None = None,
) -> dict[str, object]:
    """Validate DnCNN profile and final-checkpoint role provenance.

    Raw legacy state dictionaries can be evaluated with an explicit unverified
    label. An enveloped checkpoint that claims a resolved profile must also
    identify its task, role, device, variation level, and final training epoch.
    """
    if role not in {"nominal", "teacher", "mtrd"}:
        raise ValueError(f"unsupported DnCNN checkpoint role: {role}")
    if role == "nominal":
        checkpoint_role, checkpoint_device = "clean_teacher", "none"
        checkpoint_level, checkpoint_epoch = 0.0, expected.teacher_epochs
    elif role == "teacher":
        if expected_level is None:
            raise ValueError("expected_level is required for a DnCNN variation teacher")
        checkpoint_role, checkpoint_device = "variation_teacher", expected.device_model
        checkpoint_level, checkpoint_epoch = float(expected_level), expected.teacher_epochs
        if not any(
            math.isclose(checkpoint_level, value, rel_tol=0.0, abs_tol=1e-12)
            for value in expected.preparation_teacher_levels
        ):
            raise ValueError(
                "DnCNN variation-teacher level is not in the selected preparation pool: "
                f"{checkpoint_level:g}"
            )
    else:
        checkpoint_role, checkpoint_device = "mtrd_student", expected.device_model
        checkpoint_level, checkpoint_epoch = expected.student_initial_level, expected.student_epochs
    expected_payload = dncnn_mtrd_contract_payload(expected)
    expected_hash = canonical_json_hash(expected_payload)
    identity: dict[str, object] = {
        "role": role,
        "expected_task": "denoising",
        "expected_checkpoint_role": checkpoint_role,
        "expected_device_model": checkpoint_device,
        "expected_noise_level": checkpoint_level,
        "expected_epoch": checkpoint_epoch,
        "expected_profile": expected.name,
        "expected_contract_sha256": expected_hash,
    }
    payload = load_checkpoint(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        identity.update({
            "verification": "unverified_legacy_checkpoint",
            "reason": "checkpoint is a raw state dictionary without DnCNN profile metadata",
        })
        return identity
    embedded = payload.get("dncnn_mtrd_contract")
    if not isinstance(embedded, Mapping):
        identity.update({
            "verification": "unverified_legacy_checkpoint",
            "reason": "checkpoint does not embed a resolved DnCNN MTRD contract",
            "embedded_schema": payload.get("schema"),
        })
        return identity
    identity.update(_validate_checkpoint_outer_identity(
        payload,
        task="denoising",
        checkpoint_role=checkpoint_role,
        device_model=checkpoint_device,
        noise_level=checkpoint_level,
        epoch=checkpoint_epoch,
        label="DnCNN",
    ))
    actual = dict(embedded)
    identity.update({
        "embedded_profile": actual.get("name"),
        "embedded_contract_sha256": canonical_json_hash(actual),
    })
    if actual.get("name") != expected.name:
        raise ValueError(
            "DnCNN checkpoint profile mismatch: "
            f"expected {expected.name!r}, found {actual.get('name')!r}"
        )
    if role in {"teacher", "mtrd"}:
        if canonical_json_hash(actual) != expected_hash:
            raise ValueError(
                "DnCNN checkpoint contract differs from the selected profile "
                "configuration"
            )
        identity["verification"] = "exact_resolved_contract"
        return identity

    # A clean DnCNN checkpoint is intentionally shared by RRAM and PCM. Its
    # profile must agree on all training semantics that apply to clean training,
    # while device-specific teacher pools and initial levels are irrelevant.
    clean_keys = (
        "name",
        "teacher_epochs",
        "optimizer",
        "optimizer_kwargs",
        "initial_learning_rate",
        "teacher_lr_milestones",
        "loss_name",
        "loss_reduction",
        "kd_is_degenerate",
        "feedback_noise_policy",
        "student_training_forward",
    )
    mismatched = [
        key for key in clean_keys
        if actual.get(key) != expected_payload.get(key)
    ]
    if mismatched:
        raise ValueError(
            "DnCNN nominal checkpoint clean-training contract differs for keys: "
            + ", ".join(mismatched)
        )
    identity["verification"] = "clean_training_contract"
    return identity


def segmentation_checkpoint_training_identity(
    path: str | Path, config: Mapping[str, Any], *, device_model: str,
    role: str, expected_level: float | None = None,
) -> dict[str, object]:
    """Validate UNet checkpoint provenance without inferring omitted semantics.

    Teacher checkpoints carry role-local optimizer and metric provenance. MTRD
    students additionally require the complete resolved objective, teacher-pool,
    feedback, geometry, and BatchNorm contract. Older MTRD envelopes remain
    evaluable only with an explicit unverified-legacy label.
    """
    if device_model not in DEVICE_MODELS:
        raise ValueError(f"unsupported UNet checkpoint device model: {device_model}")
    contract = resolve_segmentation_mtrd_contract(config, device_model)
    expected_roles = {
        "nominal": ("clean_teacher", "none", False, 0.0, contract.teacher_epochs),
        "teacher": ("variation_teacher", device_model, False, expected_level, contract.teacher_epochs),
        "mtrd": (
            "mtrd_student",
            device_model,
            True,
            contract.student_initial_level,
            contract.student_epochs,
        ),
    }
    if role not in expected_roles:
        raise ValueError(f"unsupported UNet checkpoint role: {role}")
    expected_role, expected_device, student, level, expected_epoch = expected_roles[role]
    if level is None:
        raise ValueError("expected_level is required for a UNet variation teacher")
    expected_level_float = _checkpoint_noise_level(level, "UNet expected noise level")
    if role == "teacher" and not any(
        not spec.clean
        and math.isclose(spec.level, expected_level_float, rel_tol=0.0, abs_tol=1e-12)
        for spec in contract.mtrd_teacher_specs
    ):
        raise ValueError(
            "UNet variation-teacher level is not in the selected teacher pool: "
            f"{expected_level_float:g}"
        )
    expected_optimizer = semantic_contract_value(
        unet_optimizer_contract(config, student=student)
    )
    expected_metric = semantic_contract_value(resolve_segmentation_metric_contract(config))
    expected_bn_policy = (
        semantic_contract_value(resolve_segmentation_mtrd_bn_update_policy(config))
        if student else None
    )
    expected_mtrd_payload = segmentation_mtrd_contract_payload(contract)
    identity: dict[str, object] = {
        "role": role,
        "contract_hash_scope": "semantic_training_fields",
        "expected_task": "segmentation",
        "expected_checkpoint_role": expected_role,
        "expected_device_model": expected_device,
        "expected_noise_level": expected_level_float,
        "expected_epoch": expected_epoch,
        "expected_optimizer_contract_sha256": canonical_json_hash(expected_optimizer),
        "expected_metric_contract_sha256": canonical_json_hash(expected_metric),
        "expected_batch_norm_policy_sha256": (
            canonical_json_hash(expected_bn_policy)
            if expected_bn_policy is not None else None
        ),
        "expected_mtrd_contract_sha256": (
            canonical_json_hash(semantic_contract_value(expected_mtrd_payload))
            if student else None
        ),
    }
    payload = load_checkpoint(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        identity.update({
            "verification": "unverified_legacy_checkpoint",
            "reason": "checkpoint is not an enveloped mapping with UNet training contracts",
        })
        return identity

    base_contract_keys = {
        "optimizer_contract",
        "segmentation_metric_contract",
        "segmentation_mtrd_bn_update_policy",
    }
    mtrd_contract_key = "segmentation_mtrd_contract"
    present_base = base_contract_keys.intersection(payload)
    has_mtrd_contract = mtrd_contract_key in payload
    if not present_base and not has_mtrd_contract:
        identity.update({
            "verification": "unverified_legacy_checkpoint",
            "reason": "checkpoint does not embed public UNet training contracts",
            "embedded_schema": payload.get("schema"),
        })
        return identity
    if student and not has_mtrd_contract:
        identity.update({
            "verification": "unverified_legacy_checkpoint",
            "reason": (
                "checkpoint predates the complete resolved UNet MTRD contract; "
                "its objective and teacher-feedback semantics cannot be verified"
            ),
            "embedded_schema": payload.get("schema"),
        })
        return identity

    required = {"optimizer_contract", "segmentation_metric_contract"}
    if student:
        required.update({"segmentation_mtrd_bn_update_policy", mtrd_contract_key})
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "UNet checkpoint has an incomplete embedded training contract; missing "
            + ", ".join(missing)
        )
    identity.update(_validate_checkpoint_outer_identity(
        payload,
        task="segmentation",
        checkpoint_role=expected_role,
        device_model=expected_device,
        noise_level=expected_level_float,
        epoch=expected_epoch,
        label="UNet",
    ))

    embedded_optimizer = payload["optimizer_contract"]
    embedded_metric = payload["segmentation_metric_contract"]
    if not isinstance(embedded_optimizer, Mapping) or not isinstance(embedded_metric, Mapping):
        raise ValueError("UNet checkpoint embeds a malformed optimizer or metric contract")
    semantic_optimizer = semantic_contract_value(embedded_optimizer)
    semantic_metric = semantic_contract_value(embedded_metric)
    mismatches: list[str] = []
    if canonical_json_hash(semantic_optimizer) != identity["expected_optimizer_contract_sha256"]:
        mismatches.append("optimizer_contract")
    if canonical_json_hash(semantic_metric) != identity["expected_metric_contract_sha256"]:
        mismatches.append("segmentation_metric_contract")

    embedded_bn_policy: object | None = None
    if student:
        candidate = payload["segmentation_mtrd_bn_update_policy"]
        if not isinstance(candidate, Mapping):
            raise ValueError("UNet checkpoint embeds a malformed BatchNorm policy")
        embedded_bn_policy = semantic_contract_value(candidate)
        if canonical_json_hash(embedded_bn_policy) != identity["expected_batch_norm_policy_sha256"]:
            mismatches.append("segmentation_mtrd_bn_update_policy")
        embedded_mtrd_contract = payload[mtrd_contract_key]
        if not isinstance(embedded_mtrd_contract, Mapping):
            raise ValueError("UNet checkpoint embeds a malformed resolved MTRD contract")
        semantic_mtrd_contract = semantic_contract_value(embedded_mtrd_contract)
        if canonical_json_hash(semantic_mtrd_contract) != identity["expected_mtrd_contract_sha256"]:
            mismatches.append("segmentation_mtrd_contract")
    else:
        semantic_mtrd_contract = None
    if mismatches:
        raise ValueError(
            "UNet checkpoint training contract mismatch for: " + ", ".join(mismatches)
        )

    identity.update({
        "verification": "exact_resolved_contract",
        "embedded_schema": payload.get("schema"),
        "embedded_optimizer_contract_sha256": canonical_json_hash(semantic_optimizer),
        "embedded_metric_contract_sha256": canonical_json_hash(semantic_metric),
        "embedded_batch_norm_policy_sha256": (
            canonical_json_hash(embedded_bn_policy)
            if embedded_bn_policy is not None else None
        ),
        "embedded_mtrd_contract_sha256": (
            canonical_json_hash(semantic_mtrd_contract)
            if semantic_mtrd_contract is not None else None
        ),
    })
    return identity


def _canonical_carvana_id(name: str) -> str:
    stem = Path(name).stem
    return stem[:-5] if stem.endswith("_mask") else stem


def build_carvana_index(config: Mapping[str, Any]) -> dict[str, CarvanaRecord]:
    data = config["data"]
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for directory_value in data.get("carvana_image_dirs", []):
        directory = Path(directory_value)
        if not directory.is_dir():
            raise FileNotFoundError(f"Carvana image directory is missing: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                sample_id = _canonical_carvana_id(path.name)
                if sample_id in images:
                    raise ValueError(f"duplicate Carvana image id {sample_id}: {images[sample_id]} and {path}")
                images[sample_id] = path
    for directory_value in data.get("carvana_mask_dirs", []):
        directory = Path(directory_value)
        if not directory.is_dir():
            raise FileNotFoundError(f"Carvana mask directory is missing: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in MASK_SUFFIXES:
                sample_id = _canonical_carvana_id(path.name)
                if sample_id in masks:
                    raise ValueError(f"duplicate Carvana mask id {sample_id}: {masks[sample_id]} and {path}")
                masks[sample_id] = path
    missing_masks = sorted(set(images) - set(masks))
    orphan_masks = sorted(set(masks) - set(images))
    if missing_masks or orphan_masks:
        raise ValueError(
            f"Carvana image/mask mismatch: missing_masks={missing_masks[:5]}, "
            f"orphan_masks={orphan_masks[:5]}"
        )
    return {
        sample_id: CarvanaRecord(sample_id, images[sample_id], masks[sample_id])
        for sample_id in sorted(images)
    }


_CARVANA_IDENTITY_CACHE: dict[tuple[tuple[object, ...], ...], dict[str, object]] = {}


def carvana_asset_identity(
    index: Mapping[str, CarvanaRecord], *, include_hash: bool = True,
) -> dict[str, object]:
    signature = tuple(
        (
            sample_id,
            str(record.image.resolve()),
            record.image.stat().st_size,
            record.image.stat().st_mtime_ns,
            str(record.mask.resolve()),
            record.mask.stat().st_size,
            record.mask.stat().st_mtime_ns,
        )
        for sample_id, record in sorted(index.items())
    )
    if include_hash and signature in _CARVANA_IDENTITY_CACHE:
        return dict(_CARVANA_IDENTITY_CACHE[signature])
    inventory: dict[str, object] = {
        "sample_count": len(index),
        "paired_names_sha256": canonical_json_hash([
            [sample_id, record.image.name, record.mask.name]
            for sample_id, record in sorted(index.items())
        ]),
        "paired_sizes_sha256": canonical_json_hash([
            [sample_id, record.image.stat().st_size, record.mask.stat().st_size]
            for sample_id, record in sorted(index.items())
        ]),
    }
    if include_hash:
        content = {
            sample_id: {
                "image_sha256": sha256_file(record.image),
                "mask_sha256": sha256_file(record.mask),
            }
            for sample_id, record in sorted(index.items())
        }
        inventory["paired_content_manifest_sha256"] = canonical_json_hash(content)
        _CARVANA_IDENTITY_CACHE[signature] = dict(inventory)
    return inventory


def load_carvana_split(config: Mapping[str, Any], index: Mapping[str, CarvanaRecord]) -> dict[str, Any]:
    path = Path(config["data"]["carvana_split_manifest"])
    if not path.is_file():
        raise FileNotFoundError(
            "exact Carvana split manifest is missing; no directory split fallback is allowed: "
            f"{path}"
        )
    with path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    if int(split.get("schema_version", -1)) != 1:
        raise ValueError("Carvana split schema_version must be 1")
    author_verified = split.get("author_verified")
    if not isinstance(author_verified, bool):
        raise ValueError("Carvana split must declare author_verified as true or false")
    allow_derived = config["data"].get("allow_derived_carvana_split") is True
    if not author_verified and not allow_derived:
        raise ValueError(
            "Carvana split must set author_verified=true after author confirmation; "
            "set data.allow_derived_carvana_split=true only for a labeled, non-author-verified "
            "reconstruction run"
        )
    split_contract = str(
        split.get("split_contract", "manuscript-4700-318-70")
    )
    contracts = {
        "manuscript-4700-318-70": {
            "train_ids": 4700,
            "test_ids": 318,
            "excluded_ids": 70,
        },
        "released-directory-4588-500": {
            "train_ids": 4588,
            "test_ids": 500,
            "excluded_ids": 0,
        },
    }
    if split_contract not in contracts:
        raise ValueError(
            f"unsupported Carvana split_contract: {split_contract!r}"
        )
    expected = contracts[split_contract]
    normalized: dict[str, list[str]] = {}
    for key, count in expected.items():
        values = [_canonical_carvana_id(str(item)) for item in split.get(key, [])]
        if len(values) != count:
            raise ValueError(f"Carvana {key} must contain exactly {count} ids, found {len(values)}")
        if len(set(values)) != len(values):
            raise ValueError(f"Carvana {key} contains duplicate ids")
        normalized[key] = values
    sets = {key: set(value) for key, value in normalized.items()}
    if sets["train_ids"] & sets["test_ids"]:
        raise ValueError("Carvana train/test overlap")
    if sets["train_ids"] & sets["excluded_ids"] or sets["test_ids"] & sets["excluded_ids"]:
        raise ValueError("Carvana excluded ids overlap train or test")
    declared = set().union(*sets.values())
    actual = set(index)
    if declared != actual:
        raise ValueError(
            f"Carvana split does not partition the 5088-file asset: "
            f"missing={sorted(actual-declared)[:5]}, unknown={sorted(declared-actual)[:5]}"
        )
    if split_contract == "manuscript-4700-318-70":
        test_cars = {
            sample_id.rsplit("_", 1)[0]
            for sample_id in normalized["test_ids"]
        }
        all_cars = {sample_id.rsplit("_", 1)[0] for sample_id in actual}
        if len(test_cars) != 318 or test_cars != all_cars:
            raise ValueError(
                "Carvana manuscript test split must contain exactly one image "
                "for each of 318 cars"
            )
    result = dict(split)
    result.update(normalized)
    result["reference_split_author_verified"] = author_verified
    result["derived_split_allowed"] = allow_derived
    result["split_contract"] = split_contract
    result["path"] = str(path.resolve())
    result["sha256"] = sha256_file(path)
    return result


def _denoising_png_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    unsupported = sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.suffix.lower() != ".png"
    )
    if unsupported:
        raise ValueError(
            "DnCNN preprocessing consumes PNG files only; convert or remove unsupported "
            f"images below {directory}: {[path.name for path in unsupported[:5]]}"
        )
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


_DENOISING_ASSET_IDENTITY_CACHE: dict[
    tuple[tuple[str, int, int], ...], dict[str, object]
] = {}


def denoising_asset_identity(config: Mapping[str, Any], include_hash: bool = True) -> dict[str, object]:
    berkeley_root = Path(config["data"]["berkeley_root"])
    train_files = _denoising_png_files(berkeley_root / "train")
    set12_files = _denoising_png_files(Path(config["data"]["set12_dir"]))
    signature = (
        (str((berkeley_root / "train").resolve()), -1, -1),
        (str(Path(config["data"]["set12_dir"]).resolve()), -1, -1),
        *(
            (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
            for path in [*train_files, *set12_files]
        ),
    )
    if include_hash and signature in _DENOISING_ASSET_IDENTITY_CACHE:
        return dict(_DENOISING_ASSET_IDENTITY_CACHE[signature])
    identity: dict[str, object] = {
        "berkeley_root": str(berkeley_root.resolve()),
        "training_count": len(train_files),
        "set12_root": str(Path(config["data"]["set12_dir"]).resolve()),
        "set12_count": len(set12_files),
        "training_names_sha256": canonical_json_hash([path.name for path in train_files]),
        "set12_names_sha256": canonical_json_hash([path.name for path in set12_files]),
        "set12_preprocessing": copy.deepcopy(DNCNN_SET12_PREPROCESSING),
    }
    if include_hash:
        train_hashes = {path.name: sha256_file(path) for path in train_files}
        test_hashes = {path.name: sha256_file(path) for path in set12_files}
        overlap = sorted(set(train_hashes.values()) & set(test_hashes.values()))
        identity.update({
            "training_content_manifest_sha256": canonical_json_hash(train_hashes),
            "set12_content_manifest_sha256": canonical_json_hash(test_hashes),
            "train_test_duplicate_content_hashes": overlap,
        })
        _DENOISING_ASSET_IDENTITY_CACHE[signature] = dict(identity)
    return identity


def _check(report: dict[str, Any], name: str, function) -> Any:
    try:
        value = function()
        report["checks"].append({"name": name, "status": "pass"})
        return value
    except Exception as error:
        report["checks"].append({
            "name": name,
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        report["errors"].append(f"{name}: {error}")
        return None


def resolve_backend(config: Mapping[str, Any], device_model: str, override: str | None) -> str:
    backend = override if override and override != "auto" else config["simulation"].get(
        f"{device_model}_backend"
    )
    if not backend:
        raise ValueError(f"{device_model} backend must be explicit")
    expected = "neurosim" if device_model == "rram" else "aihwkit-additive"
    if backend != expected:
        raise ValueError(
            f"{device_model} evaluation requires backend={expected}; "
            "software-only and incomplete operator paths are not public evaluation backends"
        )
    return str(backend)


def resolve_realization_scope(
    config: Mapping[str, Any], device_model: str, override: str | None,
) -> str:
    scope = override or config["simulation"].get(f"{device_model}_realization_scope")
    if device_model == "pcm" and scope == "per_mac":
        raise RuntimeError(
            "AIHWKit 1.1.0 per-MAC forward-noise RNG cannot be seeded or "
            "replayed. Formal evaluation requires pcm realization_scope="
            "fixed_trial. The per-MAC path is diagnostic only."
        )
    allowed = RRAM_SCOPES if device_model == "rram" else PCM_SCOPES
    if scope not in allowed:
        raise ValueError(
            f"{device_model} realization_scope must be one of {sorted(allowed)}; "
            "the manuscript does not define it, so no default is allowed"
        )
    return str(scope)


def configured_neurosim_root(config: Mapping[str, Any]) -> Path:
    """Resolve the pinned NeuroSim tree consistently inside and outside Docker."""
    environment_value = os.environ.get("MTRD_NEUROSIM_ROOT") or os.environ.get(
        "NEUROSIM_HOME"
    )
    if environment_value:
        return Path(environment_value).expanduser().resolve()
    configured = config["simulation"].get("neurosim_root")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    container_root = Path("/opt/neurosim")
    return container_root if container_root.is_dir() else CODE_ROOT.parent / "NeuroSim"


def configured_neurosim_profile(config: Mapping[str, Any]) -> str:
    """Validate the explicit functional mapping profile."""
    from simulators.neurosim_functional import PROFILES

    profile = str(config["simulation"].get("neurosim_functional_profile", ""))
    if profile not in PROFILES:
        raise ValueError(
            "simulation.neurosim_functional_profile must be one of "
            f"{sorted(PROFILES)}"
        )
    return profile


def configured_neurosim_ptq(config: Mapping[str, Any]) -> dict[str, object]:
    """Validate the optional released eager-mode static PTQ protocol."""
    raw = config["simulation"].get(
        "neurosim_post_training_quantization", {"mode": "none"}
    )
    if not isinstance(raw, Mapping):
        raise TypeError(
            "simulation.neurosim_post_training_quantization must be an object"
        )
    mode = str(raw.get("mode", "none"))
    if mode == "none":
        return {"mode": "none"}
    if mode != "released-eager-static":
        raise ValueError(
            "simulation.neurosim_post_training_quantization.mode must be "
            "none or released-eager-static"
        )
    if configured_neurosim_profile(config) != "released-legacy":
        raise ValueError(
            "released eager static PTQ must be paired with "
            "neurosim_functional_profile=released-legacy"
        )
    activation_bits = int(raw.get("activation_bits", -1))
    weight_bits = int(raw.get("weight_bits", -1))
    if not 2 <= activation_bits <= 8 or not 2 <= weight_bits <= 8:
        raise ValueError("released static PTQ bit widths must be in [2, 8]")
    calibration_data = str(raw.get("calibration_data", ""))
    if calibration_data != "evaluation":
        raise ValueError(
            "released static PTQ requires calibration_data=evaluation to match "
            "the released image-task scripts"
        )
    engine = str(raw.get("engine", ""))
    if engine not in torch.backends.quantized.supported_engines:
        raise RuntimeError(
            f"static PTQ engine {engine!r} is unavailable; supported engines: "
            f"{torch.backends.quantized.supported_engines}"
        )
    return {
        "mode": mode,
        "activation_bits": activation_bits,
        "weight_bits": weight_bits,
        "calibration_data": calibration_data,
        "engine": engine,
    }


def configured_carvana_resize_backend(config: Mapping[str, Any]) -> str:
    """Return the explicit Carvana resize implementation."""
    settings = config["protocol"]["segmentation"]
    backend = str(settings.get("resize_backend", ""))
    if backend not in CARVANA_RESIZE_BACKENDS:
        raise ValueError(
            "protocol.segmentation.resize_backend must be one of "
            f"{sorted(CARVANA_RESIZE_BACKENDS)}"
        )
    return backend


def resolve_unet_optimizer_profile(config: Mapping[str, Any]) -> dict[str, object]:
    """Resolve the immutable optimizer contract for both UNet training roles."""
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be an object before resolving the UNet optimizer")
    settings = protocol.get("segmentation")
    if not isinstance(settings, Mapping):
        raise TypeError("protocol.segmentation must be an object")
    profile = settings.get("optimizer_profile")
    if not isinstance(profile, str) or profile not in UNET_OPTIMIZER_PROFILES:
        raise ValueError(
            "protocol.segmentation.optimizer_profile must be one of "
            f"{sorted(UNET_OPTIMIZER_PROFILES)}"
        )
    legacy = sorted(UNET_LEGACY_OPTIMIZER_KEYS.intersection(settings))
    if legacy:
        raise ValueError(
            "UNet optimizer settings are profile-defined; remove legacy keys "
            f"{legacy} and use protocol.segmentation.optimizer_profile"
        )
    return copy.deepcopy(UNET_OPTIMIZER_PROFILES[str(profile)])


def unet_optimizer_contract(
    config: Mapping[str, Any], *, student: bool | None = None,
) -> dict[str, object]:
    """Return the resolved profile with an optional teacher or student role label."""
    contract = resolve_unet_optimizer_profile(config)
    if student is not None:
        role = "student" if student else "teacher"
        roles = contract["roles"]
        if not isinstance(roles, Mapping):
            raise RuntimeError("UNet optimizer profile has an invalid roles contract")
        contract["optimization_role"] = role
        contract["role_optimizer"] = str(roles[role])
    return contract


def _segmentation_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be an object before resolving segmentation settings")
    settings = protocol.get("segmentation")
    if not isinstance(settings, Mapping):
        raise TypeError("protocol.segmentation must be an object")
    return settings


def resolve_segmentation_metric_contract(config: Mapping[str, Any]) -> dict[str, object]:
    """Resolve the metric observation and reduction semantics for Carvana."""
    settings = _segmentation_settings(config)
    configured_explicitly = "metric_aggregation" in settings
    aggregation = settings.get("metric_aggregation", "per_image_mean")
    if not isinstance(aggregation, str) or aggregation not in SEGMENTATION_METRIC_CONTRACTS:
        raise ValueError(
            "protocol.segmentation.metric_aggregation must be one of "
            f"{sorted(SEGMENTATION_METRIC_CONTRACTS)}"
        )
    contract = copy.deepcopy(SEGMENTATION_METRIC_CONTRACTS[aggregation])
    contract["configured_explicitly"] = configured_explicitly
    return contract


def resolve_segmentation_mtrd_bn_update_policy(
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Resolve persistent BatchNorm buffer behavior for the UNet MTRD student."""
    settings = _segmentation_settings(config)
    configured_explicitly = "mtrd_bn_update_policy" in settings
    policy = settings.get("mtrd_bn_update_policy", "legacy_dual_branch")
    if (
        not isinstance(policy, str)
        or policy not in SEGMENTATION_MTRD_BN_UPDATE_POLICIES
    ):
        raise ValueError(
            "protocol.segmentation.mtrd_bn_update_policy must be one of "
            f"{sorted(SEGMENTATION_MTRD_BN_UPDATE_POLICIES)}"
        )
    contract = copy.deepcopy(SEGMENTATION_MTRD_BN_UPDATE_POLICIES[policy])
    contract["configured_explicitly"] = configured_explicitly
    contract["default_policy"] = "legacy_dual_branch"
    contract["author_verified"] = False
    return contract


_NON_SEMANTIC_CONTRACT_KEYS = frozenset({
    "author_confirmation_required",
    "author_verified",
    "compatibility",
    "configured_explicitly",
    "provenance",
    "warning",
})


def semantic_contract_value(value: object) -> object:
    """Remove descriptive provenance fields before comparing training behavior."""
    if isinstance(value, Mapping):
        return {
            str(key): semantic_contract_value(item)
            for key, item in value.items()
            if str(key) not in _NON_SEMANTIC_CONTRACT_KEYS
        }
    if isinstance(value, tuple):
        return [semantic_contract_value(item) for item in value]
    if isinstance(value, list):
        return [semantic_contract_value(item) for item in value]
    return value


def _required_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if resolved <= 0 or float(value) != float(resolved):
        raise ValueError(f"{label} must be a positive integer")
    return resolved


def _required_positive_float(value: object, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive finite number") from error
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return resolved


def resolve_segmentation_mtrd_contract(
    config: Mapping[str, Any], device_model: str,
) -> SegmentationMTRDContract:
    """Resolve every material public UNet MTRD training decision.

    The contract intentionally excludes paths, environment identity, and output
    locations. It does include the objective, teacher pool, feedback policy,
    geometry, and optimizer/metric/BatchNorm semantics that affect a trained
    student or make two reported student checkpoints incomparable.
    """
    if device_model not in DEVICE_MODELS:
        raise ValueError(f"unsupported UNet device model: {device_model}")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be an object before resolving the UNet MTRD contract")
    settings = _segmentation_settings(config)
    teacher_epochs = _required_positive_int(
        settings.get("teacher_epochs"), "UNet teacher_epochs",
    )
    student_epochs = _required_positive_int(
        settings.get("student_epochs"), "UNet student_epochs",
    )
    if (teacher_epochs, student_epochs) != (20, 30):
        raise ValueError("UNet protocol must use teacher=20/student=30")
    checkpoint_selection = settings.get("checkpoint_selection")
    if checkpoint_selection != "final_epoch":
        raise ValueError("UNet checkpoint_selection must be final_epoch")
    batch_size = _required_positive_int(settings.get("batch_size"), "UNet batch_size")
    image_height = _required_positive_int(settings.get("image_height"), "UNet image_height")
    image_width = _required_positive_int(settings.get("image_width"), "UNet image_width")
    resize_backend = configured_carvana_resize_backend(config)
    feedback_split = settings.get("feedback_split")
    if feedback_split not in {"carvana_test", "carvana_training"}:
        raise ValueError(
            "UNet feedback_split must explicitly be carvana_test or carvana_training"
        )
    feedback_sample_count = settings.get("feedback_sample_count")
    if isinstance(feedback_sample_count, bool):
        raise ValueError("UNet feedback_sample_count must be >=0; zero means full split")
    try:
        feedback_sample_count = int(feedback_sample_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "UNet feedback_sample_count must be >=0; zero means full split"
        ) from error
    if feedback_sample_count < 0:
        raise ValueError("UNet feedback_sample_count must be >=0; zero means full split")

    try:
        alpha = float(protocol.get("eq4_alpha"))
    except (TypeError, ValueError) as error:
        raise ValueError("UNet Eq.(4) alpha must be an explicit finite value in [0, 1]") from error
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("UNet Eq.(4) alpha must be an explicit finite value in [0, 1]")
    temperature = _required_positive_float(
        protocol.get("distillation_temperature"), "UNet distillation temperature",
    )
    direction = protocol.get("eq6_direction")
    if direction not in EQ6_CHOICES:
        raise ValueError(f"eq6_direction must be explicit: {sorted(EQ6_CHOICES)}")
    eq6_temperature = _required_positive_float(
        protocol.get("eq6_temperature"), "UNet Eq.(6) temperature",
    )
    include_clean = protocol.get("include_clean_teacher")
    if direction == "released_rank_underperformance":
        if include_clean is not False:
            raise ValueError(
                "released rank training requires exactly five robust teachers and "
                "include_clean_teacher=false"
            )
    elif include_clean is not True:
        raise ValueError("the configured softmax chain requires include_clean_teacher=true")

    teacher_pool = teacher_specs(config, device_model)
    robust_pool = [spec for spec in teacher_pool if not spec.clean]
    if not robust_pool:
        raise ValueError("UNet MTRD requires at least one variation teacher")
    if direction == "released_rank_underperformance" and len(robust_pool) != 5:
        raise ValueError("released rank training requires exactly five robust teachers")

    return SegmentationMTRDContract(
        name="unified_binary_bce_mtrd_v1",
        device_model=device_model,
        mtrd_teacher_specs=teacher_pool,
        student_initial_level=student_level(config, device_model),
        include_clean_teacher=bool(include_clean),
        teacher_epochs=teacher_epochs,
        student_epochs=student_epochs,
        checkpoint_selection=str(checkpoint_selection),
        training_batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
        resize_backend=resize_backend,
        feedback_split=str(feedback_split),
        feedback_sample_count=feedback_sample_count,
        eq4_alpha=alpha,
        distillation_temperature=temperature,
        eq6_direction=str(direction),
        eq6_temperature=eq6_temperature,
        teacher_optimizer=semantic_contract_value(
            unet_optimizer_contract(config, student=False)
        ),
        student_optimizer=semantic_contract_value(
            unet_optimizer_contract(config, student=True)
        ),
        metric_contract=semantic_contract_value(
            resolve_segmentation_metric_contract(config)
        ),
        batch_norm_update_policy=semantic_contract_value(
            resolve_segmentation_mtrd_bn_update_policy(config)
        ),
        model_output="single_foreground_logit_with_sigmoid_probability",
        teacher_loss="binary_cross_entropy_with_logits_mean",
        distillation_loss=(
            "beta_weighted_binary_cross_entropy_with_logits("
            "nominal_student_logits/temperature, sigmoid(teacher_logits/temperature))"
            "*temperature^2"
        ),
        noisy_task_loss="binary_cross_entropy_with_logits_mean",
        total_loss="eq4_alpha*distillation_loss+(1-eq4_alpha)*noisy_task_loss",
        nominal_student_forward="student(inputs, 0.0, 'none')",
        noisy_student_forward="equation_forward(student, inputs, device_model, student_initial_level)",
        variation_teacher_forward="equation_forward(teacher, inputs, device_model, teacher_level)",
        training_noise_realization="fresh_independent_tensor_per_compute_layer_and_forward",
        author_confirmation_required=True,
    )


def segmentation_mtrd_contract_payload(
    contract: SegmentationMTRDContract,
) -> dict[str, object]:
    """Return a JSON-safe, semantic UNet MTRD contract for checkpoint metadata."""
    return {
        "contract_schema_version": 1,
        "name": contract.name,
        "device_model": contract.device_model,
        "mtrd_teacher_pool": [asdict(spec) for spec in contract.mtrd_teacher_specs],
        "student_initial_level": contract.student_initial_level,
        "include_clean_teacher": contract.include_clean_teacher,
        "teacher_epochs": contract.teacher_epochs,
        "student_epochs": contract.student_epochs,
        "checkpoint_selection": contract.checkpoint_selection,
        "training_batch_size": contract.training_batch_size,
        "image_height": contract.image_height,
        "image_width": contract.image_width,
        "resize_backend": contract.resize_backend,
        "feedback_split": contract.feedback_split,
        "feedback_sample_count": contract.feedback_sample_count,
        "eq4_alpha": contract.eq4_alpha,
        "distillation_temperature": contract.distillation_temperature,
        "eq6_direction": contract.eq6_direction,
        "eq6_temperature": contract.eq6_temperature,
        "teacher_optimizer": dict(contract.teacher_optimizer),
        "student_optimizer": dict(contract.student_optimizer),
        "metric_contract": dict(contract.metric_contract),
        "batch_norm_update_policy": dict(contract.batch_norm_update_policy),
        "model_output": contract.model_output,
        "teacher_loss": contract.teacher_loss,
        "distillation_loss": contract.distillation_loss,
        "noisy_task_loss": contract.noisy_task_loss,
        "total_loss": contract.total_loss,
        "nominal_student_forward": contract.nominal_student_forward,
        "noisy_student_forward": contract.noisy_student_forward,
        "variation_teacher_forward": contract.variation_teacher_forward,
        "training_noise_realization": contract.training_noise_realization,
        "author_confirmation_required": contract.author_confirmation_required,
    }


def aihwkit_runtime_preflight(
    config: Mapping[str, Any], realization_scope: str,
) -> dict[str, object]:
    from simulators.aihwkit import (
        build_additive_config,
        convert_model,
    )

    shared_scope = "per-mac" if realization_scope == "per_mac" else "fixed-trial"
    weight_bits = int(config["simulation"]["weight_bits"])
    dac_bits = int(config["simulation"]["dac_bits"])
    adc_bits = int(config["simulation"]["adc_bits"])
    seed = stable_seed(int(config["seed"]), "aihwkit-runtime-preflight", shared_scope)
    rpu_config = build_additive_config(
        0.06, input_bits=dac_bits, output_bits=adc_bits, seed=seed,
        realization_scope=shared_scope,
        per_mac_signed_max_ratio=1.0 if shared_scope == "per-mac" else None,
    )
    probe_model = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, bias=False)).eval()
    with torch.no_grad():
        probe_model[0].weight.fill_(0.5)
    analog = convert_model(
        probe_model, 0.06, input_bits=dac_bits, output_bits=adc_bits, seed=seed,
        realization_scope=shared_scope,
    ).eval()
    with torch.no_grad():
        output = analog(torch.ones(1, 1, 2, 2))
        repeated = analog(torch.ones(1, 1, 2, 2))
    if output.shape != (1, 1, 2, 2) or not torch.isfinite(output).all():
        raise RuntimeError("AIHWKit converted Conv2d probe returned an invalid tensor")
    if realization_scope == "fixed_trial" and not torch.equal(output, repeated):
        raise RuntimeError("AIHWKit fixed-trial probe changed across repeated forwards")
    return {
        "aihwkit_version": package_version("aihwkit"),
        "realization_scope": realization_scope,
        "probe": "converted Conv2d(1,1,1) forward",
        "output_shape": list(output.shape),
        "output_finite": True,
        "fixed_trial_repeat_equal": (
            bool(torch.equal(output, repeated))
            if realization_scope == "fixed_trial" else "not-applicable"
        ),
        "mapping_max_input_size": int(rpu_config.mapping.max_input_size),
        "mapping_max_output_size": int(rpu_config.mapping.max_output_size),
        "weight_bits": weight_bits,
        "dac_bits": dac_bits,
        "adc_bits": adc_bits,
        "pcm_wmax_definition": "signed maximum weight value over each logical layer",
        "signed_wmax_ratios": getattr(analog, "_mtrd_pcm_signed_max_ratios", {}),
        "inp_res": float(rpu_config.forward.inp_res),
        "out_res": float(rpu_config.forward.out_res),
    }


def run_preflight(
    config: Mapping[str, Any], *, scope: str, tasks: Sequence[str],
    device_models: Sequence[str], backend_override: str | None = None,
    realization_scope_override: str | None = None, stage: str = "all",
    write_report: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": f"{SCHEMA}.preflight",
        "created_utc": utc_now(),
        "scope": scope,
        "stage": stage,
        "tasks": list(tasks),
        "device_models": list(device_models),
        "config_sha256": canonical_json_hash(config),
        "checks": [],
        "warnings": [],
        "errors": [],
        "environment": environment_identity(),
    }
    protocol = config.get("protocol", {})
    simulation = config.get("simulation", {})
    report["warnings"].append(PROTOCOL_WARNING)

    def protocol_check() -> dict[str, object] | None:
        resolved_segmentation = None
        for device_model in device_models:
            levels(config, device_model)
            student_level(config, device_model)
        if "denoising" in tasks:
            for device_model in device_models:
                resolve_dncnn_mtrd_contract(config, device_model)
        if "segmentation" in tasks:
            segmentation_optimizer = resolve_unet_optimizer_profile(config)
            segmentation_metric = resolve_segmentation_metric_contract(config)
            segmentation_bn_update_policy = resolve_segmentation_mtrd_bn_update_policy(
                config
            )
            segmentation_mtrd_contracts = {
                device_model: segmentation_mtrd_contract_payload(
                    resolve_segmentation_mtrd_contract(config, device_model)
                )
                for device_model in device_models
            }
            resolved_segmentation = {
                "segmentation_optimizer": segmentation_optimizer,
                "segmentation_metric_contract": segmentation_metric,
                "segmentation_mtrd_bn_update_policy": segmentation_bn_update_policy,
                "segmentation_mtrd_contracts": segmentation_mtrd_contracts,
            }
        return resolved_segmentation

    resolved_segmentation = _check(report, "image-task protocol constants", protocol_check)
    if resolved_segmentation is not None:
        report["resolved_protocol"] = resolved_segmentation
        optimizer = resolved_segmentation["segmentation_optimizer"]
        bn_update_policy = resolved_segmentation["segmentation_mtrd_bn_update_policy"]
        if not isinstance(optimizer, Mapping) or not isinstance(bn_update_policy, Mapping):
            raise RuntimeError("segmentation protocol resolution returned an invalid contract")
        report["warnings"].append(str(optimizer["warning"]))
        report["warnings"].append(str(bn_update_policy["warning"]))

    denoise_identity = None
    if "denoising" in tasks:
        def validate_denoising_assets() -> dict[str, Any]:
            identity = denoising_asset_identity(config, include_hash=True)
            if identity["training_count"] != 400:
                raise ValueError(
                    "Berkeley training set must contain 400 images, "
                    f"found {identity['training_count']}"
                )
            if identity["set12_count"] != 12:
                raise ValueError(
                    f"Set12 must contain 12 images, found {identity['set12_count']}"
                )
            if identity.get("train_test_duplicate_content_hashes"):
                raise ValueError("Set12 content overlaps Berkeley training content")
            return identity

        denoise_identity = _check(
            report, "Berkeley400 and Set12 assets",
            validate_denoising_assets,
        )
        if scope in {"train", "all"}:
            h5_identity = _check(
                report, "denoising HDF5 provenance and content",
                lambda: denoising_h5_identity(config),
            )
            if h5_identity is not None:
                report["denoising_h5"] = h5_identity
                configured_hashes = h5_identity.get("configured_artifact_sha256")
                if (
                    h5_identity.get("mode") == "source-provided"
                    and isinstance(configured_hashes, Mapping)
                    and configured_hashes.get("train_h5") is None
                ):
                    report["warnings"].append(
                        "source-provided DnCNN HDF5 artifacts have no configured "
                        "expected SHA-256 values; computed identities are recorded, but "
                        "the upload is not bound to a predeclared source artifact"
                    )

    carvana_index = carvana_split = None
    if "segmentation" in tasks:
        def validate_carvana_index() -> dict[str, CarvanaRecord]:
            index = build_carvana_index(config)
            if len(index) != 5088:
                raise ValueError(
                    f"Carvana asset must contain 5088 paired samples, found {len(index)}"
                )
            return index

        carvana_index = _check(
            report, "Carvana image/mask index", validate_carvana_index,
        )
        if carvana_index is not None:
            carvana_split = _check(
                report, "Carvana split contract",
                lambda: load_carvana_split(config, carvana_index),
            )
            if carvana_split is not None:
                if not carvana_split["reference_split_author_verified"]:
                    report["warnings"].append(
                        "a non-author-verified Carvana split is enabled; outputs are "
                        "reproducible under its recorded contract but are not the "
                        "author-verified split"
                    )
                asset_identity = _check(
                    report, "Carvana paired file content manifest",
                    lambda: carvana_asset_identity(carvana_index, include_hash=True),
                )
                if asset_identity is not None:
                    report["carvana_asset"] = asset_identity

    if scope in {"train", "all"} and stage == "mtrd":
        report["mtrd_training_prerequisite_contracts"] = {}
        for task in tasks:
            for device_model in device_models:
                for spec in mtrd_prerequisite_specs(config, task, device_model):
                    path = checkpoint_path(
                        config,
                        task,
                        "clean" if spec.clean else "teacher",
                        None if spec.clean else device_model,
                        None if spec.clean else spec.level,
                    )
                    label = (
                        f"{task} clean teacher checkpoint"
                        if spec.clean
                        else f"{task} {device_model} teacher {spec.level:g} checkpoint"
                    )
                    size = _check(report, label, lambda p=path: p.stat().st_size)
                    if size is None:
                        continue
                    model = _check(
                        report,
                        f"{label} strict model-state load",
                        lambda p=path, t=task: load_model_strict(p, t, "cpu"),
                    )
                    if model is None:
                        continue
                    del model
                    checkpoint_role = "nominal" if spec.clean else "teacher"
                    key = f"{task}.{device_model}.{spec.label}"
                    if task == "denoising":
                        identity = _check(
                            report,
                            f"{label} DnCNN training profile provenance",
                            lambda p=path, dm=device_model, r=checkpoint_role: (
                                dncnn_checkpoint_profile_identity(
                                    p,
                                    resolve_dncnn_mtrd_contract(config, dm),
                                    role=r,
                                    expected_level=spec.level,
                                )
                            ),
                        )
                    else:
                        identity = _check(
                            report,
                            f"{label} UNet training contract provenance",
                            lambda p=path, dm=device_model, r=checkpoint_role: (
                                segmentation_checkpoint_training_identity(
                                    p, config, device_model=dm, role=r,
                                    expected_level=spec.level,
                                )
                            ),
                        )
                    if identity is not None:
                        report["mtrd_training_prerequisite_contracts"][key] = identity
                        if identity.get("verification") == "unverified_legacy_checkpoint":
                            message = (
                                f"{label} is an unverified legacy checkpoint and cannot be "
                                "used to start a reproducible MTRD training run"
                            )
                            report["checks"].append({
                                "name": f"{label} verified MTRD training provenance",
                                "status": "fail",
                                "error": message,
                            })
                            report["errors"].append(message)
                        else:
                            report["checks"].append({
                                "name": f"{label} verified MTRD training provenance",
                                "status": "pass",
                            })

    if scope in {"eval", "all"}:
        def validate_simulation_constants() -> dict[str, Any]:
            trial_count = int(simulation.get("trial_count", 0))
            if trial_count <= 0:
                raise ValueError("simulation.trial_count must be a positive explicit integer")
            precision = tuple(
                int(simulation.get(key, -1))
                for key in ("weight_bits", "dac_bits", "adc_bits")
            )
            if precision != (8, 8, 8):
                raise ValueError("this evaluation profile requires 8-bit weight, DAC, and ADC precision")
            return {"trial_count": trial_count, "precision_bits": list(precision)}

        _check(report, "simulation constants", validate_simulation_constants)
        if simulation.get("trial_count_author_verified") is not True:
            report["warnings"].append(
                "trial count is not author-verified; results are reproducible but not an exact reference claim"
            )
        for device_model in device_models:
            backend = _check(
                report, f"{device_model} backend identity",
                lambda dm=device_model: resolve_backend(config, dm, backend_override),
            )
            _check(
                report, f"{device_model} realization scope",
                lambda dm=device_model: resolve_realization_scope(
                    config, dm, realization_scope_override,
                ),
            )
            resolved_scope = None
            with contextlib.suppress(Exception):
                resolved_scope = resolve_realization_scope(
                    config, device_model, realization_scope_override,
                )
            if backend == "neurosim":
                from simulators.neurosim import require_functional_adapter

                neurosim_root = configured_neurosim_root(config)
                _check(
                    report,
                    "NeuroSim functional mapping profile",
                    lambda: configured_neurosim_profile(config),
                )
                ptq = _check(
                    report,
                    "NeuroSim post-training quantization protocol",
                    lambda: configured_neurosim_ptq(config),
                )
                if ptq and ptq.get("mode") == "released-eager-static":
                    report["warnings"].append(
                        "released eager static PTQ calibrates on the evaluation set; "
                        "the released scripts use 6-bit observers while the manuscript "
                        "labels the image-task evaluation as 8-bit"
                    )
                    if "segmentation" in tasks:
                        def validate_released_carvana_resize() -> str:
                            backend = configured_carvana_resize_backend(config)
                            if backend != "released-opencv":
                                raise ValueError(
                                    "released UNet static PTQ requires "
                                    "resize_backend=released-opencv"
                                )
                            return backend

                        _check(
                            report,
                            "released UNet OpenCV resize protocol",
                            validate_released_carvana_resize,
                        )
                for task in tasks:
                    model_name = "dncnn" if task == "denoising" else "unet"
                    _check(
                        report,
                        f"NeuroSim {model_name} functional adapter",
                        lambda name=model_name, root=neurosim_root: (
                            require_functional_adapter(root, name)
                        ),
                    )
            if backend == "aihwkit-additive":
                actual = package_version("aihwkit")
                required = str(simulation.get("aihwkit_required_version", AIHWKIT_PIN))
                def validate_aihwkit_version() -> str:
                    if actual != required:
                        raise RuntimeError(
                            "AIHWKit version mismatch for the configured PCM backend: "
                            f"expected {required}, found {actual}"
                        )
                    return str(actual)

                _check(report, "AIHWKit exact version", validate_aihwkit_version)
                if resolved_scope in PCM_SCOPES:
                    runtime = _check(
                        report,
                        f"AIHWKit {resolved_scope} import, conversion, and forward probe",
                        lambda rs=resolved_scope: aihwkit_runtime_preflight(config, rs),
                    )
                    if runtime is not None:
                        report[f"aihwkit_runtime_{resolved_scope}"] = runtime
                if "segmentation" in tasks:
                    def reject_unet_operator_coverage() -> None:
                        raise RuntimeError(
                            "AIHWKit 1.1 cannot convert ConvTranspose2d, so a complete "
                            "analog operator mapping cannot be built"
                        )

                    _check(
                        report,
                        "UNet AIHWKit weighted-operator coverage",
                        reject_unet_operator_coverage,
                    )
        if "denoising" in tasks:
            report["dncnn_checkpoint_profiles"] = {}
        if "segmentation" in tasks:
            report["segmentation_checkpoint_training_contracts"] = {}
        for task in tasks:
            for device_model in device_models:
                for role in ("nominal", "mtrd"):
                    path = evaluation_checkpoint(config, task, device_model, role)
                    label = f"{task} {device_model} {role} checkpoint"
                    if device_model == "pcm" and role == "mtrd" and not path.is_file():
                        report["errors"].append(
                            f"PCM MTRD checkpoint missing for {task}; RRAM MTRD substitution is forbidden: {path}"
                        )
                        report["checks"].append({"name": label, "status": "fail", "error": "missing"})
                        continue
                    model = _check(
                        report, label,
                        lambda p=path, t=task: load_model_strict(p, t, "cpu"),
                    )
                    if model is not None:
                        del model
                        if task == "denoising":
                            contract = resolve_dncnn_mtrd_contract(config, device_model)
                            profile = _check(
                                report,
                                f"{label} DnCNN MTRD profile provenance",
                                lambda p=path, c=contract, r=role: (
                                    dncnn_checkpoint_profile_identity(p, c, role=r)
                                ),
                            )
                            if profile is not None:
                                key = f"{task}.{device_model}.{role}"
                                report["dncnn_checkpoint_profiles"][key] = profile
                                if profile.get("verification") == "unverified_legacy_checkpoint":
                                    report["warnings"].append(
                                        f"{label} has no embedded DnCNN profile contract; "
                                        "its training semantics cannot be verified"
                                    )
                        else:
                            identity = _check(
                                report,
                                f"{label} UNet training contract provenance",
                                lambda p=path, dm=device_model, r=role: (
                                    segmentation_checkpoint_training_identity(
                                        p, config, device_model=dm, role=r,
                                    )
                                ),
                            )
                            if identity is not None:
                                key = f"{task}.{device_model}.{role}"
                                report["segmentation_checkpoint_training_contracts"][key] = identity
                                if identity.get("verification") == "unverified_legacy_checkpoint":
                                    report["warnings"].append(
                                        f"{label} has no embedded UNet training contract; its "
                                        "training semantics cannot be verified"
                                    )
        try:
            from utils.checkpoint_roles import validate_checkpoint_roles

            role_manifest_path = config.get("checkpoint_role_manifest")
            if not isinstance(role_manifest_path, str) or not role_manifest_path:
                raise ValueError("checkpoint_role_manifest must be configured for evaluation")
            expected_roles = {
                f"image.{task}.{device_model}.{role}": evaluation_checkpoint(
                    config, task, device_model, role,
                )
                for task in tasks
                for device_model in device_models
                for role in ("nominal", "mtrd")
            }
            report["checkpoint_role_manifest"] = validate_checkpoint_roles(
                role_manifest_path, expected_roles,
            )
            report["checks"].append({
                "name": "checkpoint role manifest",
                "status": "pass",
            })
        except Exception as error:
            report["checks"].append({
                "name": "checkpoint role manifest",
                "status": "fail",
                "error_type": type(error).__name__,
                "error": str(error),
            })
            report["errors"].append(f"checkpoint role manifest: {error}")
    report["status"] = "fail" if report["errors"] else "pass"
    if denoise_identity is not None:
        report["denoising_data"] = denoise_identity
    if carvana_split is not None:
        report["carvana_split"] = {
            "path": carvana_split["path"],
            "sha256": carvana_split["sha256"],
            "train_count": len(carvana_split["train_ids"]),
            "test_count": len(carvana_split["test_ids"]),
            "excluded_count": len(carvana_split["excluded_ids"]),
            "split_contract": carvana_split["split_contract"],
            "author_verified": carvana_split["reference_split_author_verified"],
        }
    if write_report:
        output = Path(config.get("output_root", "/results/image_workflows"))
        try:
            write_json(output / f"preflight_{scope}.json", report)
        except OSError as error:
            report["warnings"].append(f"could not write preflight report: {error}")
    return report


class H5DenoisingDataset(Dataset):
    """Lazy HDF5 dataset that is safe with multi-worker DataLoaders."""

    def __init__(self, path: str | Path):
        self.path = str(Path(path).resolve())
        self._file = None
        import h5py

        with h5py.File(self.path, "r") as handle:
            self.batch_mode = "data" in handle
            self.keys = [str(index) for index in range(len(handle["data"]))] if self.batch_mode else sorted(
                handle.keys(), key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item)
            )

    def __len__(self) -> int:
        return len(self.keys)

    def _handle(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.path, "r")
        return self._file

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        handle = self._handle()
        value = handle["data"][index] if self.batch_mode else handle[self.keys[index]][()]
        return torch.from_numpy(np.asarray(value, dtype=np.float32)), self.keys[index]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_file", None)
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()


class Set12Dataset(Dataset):
    """Set12 inputs with the released OpenCV BGR-channel preprocessing."""

    def __init__(self, directory: str | Path):
        self.paths = _denoising_png_files(Path(directory))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not decode Set12 image: {path}")
        channel = image[:, :, 0].copy().astype(np.float32) / 255.0
        return torch.from_numpy(channel).unsqueeze(0), path.stem


class CarvanaSplitDataset(Dataset):
    """Stateless deterministic transforms for the author-supplied split."""

    def __init__(
        self, records: Mapping[str, CarvanaRecord], sample_ids: Sequence[str],
        *, height: int, width: int, train: bool, seed: int,
        resize_backend: str,
    ):
        self.records = records
        self.sample_ids = list(sample_ids)
        self.height = int(height)
        self.width = int(width)
        self.train = bool(train)
        self.seed = int(seed)
        if resize_backend not in CARVANA_RESIZE_BACKENDS:
            raise ValueError(f"unsupported Carvana resize backend: {resize_backend}")
        self.resize_backend = resize_backend
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample_id = self.sample_ids[index]
        record = self.records[sample_id]
        image = Image.open(record.image).convert("RGB")
        mask = Image.open(record.mask).convert("L")
        if self.resize_backend == "released-opencv":
            import cv2

            image = Image.fromarray(
                cv2.resize(
                    np.asarray(image),
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            mask = Image.fromarray(
                cv2.resize(
                    np.asarray(mask),
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
        else:
            image = image.resize(
                (self.width, self.height), resample=Image.Resampling.BILINEAR
            )
            mask = mask.resize(
                (self.width, self.height), resample=Image.Resampling.NEAREST
            )
        if self.train:
            rng = random.Random(stable_seed(self.seed, "carvana-augment", self.epoch, sample_id))
            angle = rng.uniform(-35.0, 35.0)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < 0.1:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        return (
            torch.from_numpy(image_array.copy()),
            torch.from_numpy(mask_array.copy()),
            sample_id,
        )


def _worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)


def epoch_loader(
    dataset: Dataset, *, batch_size: int, workers: int, seed: int,
    epoch: int, shuffle: bool,
) -> DataLoader:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(stable_seed(seed, "loader", epoch, shuffle))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=False,
    )


def denoising_preprocessing_spec(config: Mapping[str, Any]) -> dict[str, object]:
    source = denoising_asset_identity(config, include_hash=True)
    if source["training_count"] != 400 or source["set12_count"] != 12:
        raise ValueError(
            "Berkeley/Set12 preprocessing requires exactly 400 training and 12 test images"
        )
    if source.get("train_test_duplicate_content_hashes"):
        raise ValueError("Set12 content overlaps Berkeley training content")
    settings = config["protocol"]["denoising"]
    return {
        "source": source,
        "patch_size": int(settings.get("patch_size", 40)),
        "patch_stride": int(settings.get("patch_stride", 10)),
        "augmentation_repeats": 1,
        "seed": int(config["seed"]),
        "preprocessor": "utils.data.prepare_denoising_data",
        "set12_preprocessing": copy.deepcopy(DNCNN_SET12_PREPROCESSING),
        "training_resize": copy.deepcopy(DNCNN_TRAINING_RESIZE),
    }


_DENOISING_H5_IDENTITY_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


def _sha256_config_value(value: object, label: str) -> str:
    """Validate a configured immutable artifact digest."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    if value.lower() != value or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def resolve_denoising_h5_input(config: Mapping[str, Any]) -> dict[str, object]:
    """Resolve the explicit DnCNN HDF5 provenance mode and immutable inputs.

    ``denoising_h5_dir`` remains a compatibility-only spelling for existing
    regenerated runs. New public configurations must select one named mode in
    ``data.denoising_h5`` so a source-provided artifact cannot be mistaken for
    locally regenerated data.
    """
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("data must be an object")
    declared = data.get("denoising_h5")
    legacy_directory = data.get("denoising_h5_dir")
    if declared is None:
        if not isinstance(legacy_directory, str) or not legacy_directory:
            raise ValueError(
                "data.denoising_h5 must select a mode, or the legacy "
                "data.denoising_h5_dir must name a regenerated output directory"
            )
        directory = Path(legacy_directory)
        return {
            "schema": f"{SCHEMA}.denoising-h5-input",
            "mode": "regenerated-from-raw",
            "configuration_origin": "legacy-denoising_h5_dir",
            "directory": str(directory.resolve()),
            "train_h5": str((directory / "train.h5").resolve()),
            "val_h5": str((directory / "val.h5").resolve()),
            "preprocessing_manifest": str(
                (directory / "preprocessing_manifest.json").resolve()
            ),
            "configured_artifact_sha256": {
                "train_h5": None,
                "val_h5": None,
            },
        }
    if not isinstance(declared, Mapping):
        raise ValueError("data.denoising_h5 must be an object")
    if legacy_directory is not None:
        raise ValueError(
            "configure exactly one DnCNN HDF5 input form: data.denoising_h5, "
            "not the legacy data.denoising_h5_dir"
        )
    mode = declared.get("mode")
    if mode not in DENOISING_H5_MODES:
        raise ValueError(
            "data.denoising_h5.mode must be one of "
            f"{sorted(DENOISING_H5_MODES)}"
        )
    if mode == "regenerated-from-raw":
        allowed = {"mode", "directory"}
        unknown = sorted(set(declared) - allowed)
        if unknown:
            raise ValueError(
                "regenerated-from-raw DnCNN HDF5 settings do not accept: "
                f"{unknown}"
            )
        directory_value = declared.get("directory")
        if not isinstance(directory_value, str) or not directory_value:
            raise ValueError(
                "regenerated-from-raw DnCNN HDF5 settings require a non-empty directory"
            )
        directory = Path(directory_value)
        return {
            "schema": f"{SCHEMA}.denoising-h5-input",
            "mode": mode,
            "configuration_origin": "explicit",
            "directory": str(directory.resolve()),
            "train_h5": str((directory / "train.h5").resolve()),
            "val_h5": str((directory / "val.h5").resolve()),
            "preprocessing_manifest": str(
                (directory / "preprocessing_manifest.json").resolve()
            ),
            "configured_artifact_sha256": {
                "train_h5": None,
                "val_h5": None,
            },
        }

    allowed = {
        "mode",
        "train_h5",
        "val_h5",
        "train_h5_sha256",
        "val_h5_sha256",
    }
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(
            "source-provided DnCNN HDF5 settings do not accept: "
            f"{unknown}"
        )
    train_value = declared.get("train_h5")
    val_value = declared.get("val_h5")
    if not isinstance(train_value, str) or not train_value:
        raise ValueError("source-provided DnCNN HDF5 settings require train_h5")
    if not isinstance(val_value, str) or not val_value:
        raise ValueError("source-provided DnCNN HDF5 settings require val_h5")
    if Path(train_value).resolve() == Path(val_value).resolve():
        raise ValueError("source-provided train_h5 and val_h5 must be different files")
    train_digest = declared.get("train_h5_sha256")
    val_digest = declared.get("val_h5_sha256")
    if (train_digest is None) != (val_digest is None):
        raise ValueError(
            "source-provided DnCNN HDF5 requires both train_h5_sha256 and "
            "val_h5_sha256 when either digest is configured"
        )
    if train_digest is not None:
        train_digest = _sha256_config_value(train_digest, "train_h5_sha256")
        val_digest = _sha256_config_value(val_digest, "val_h5_sha256")
    return {
        "schema": f"{SCHEMA}.denoising-h5-input",
        "mode": mode,
        "configuration_origin": "explicit",
        "directory": None,
        "train_h5": str(Path(train_value).resolve()),
        "val_h5": str(Path(val_value).resolve()),
        "preprocessing_manifest": None,
        "configured_artifact_sha256": {
            "train_h5": train_digest,
            "val_h5": val_digest,
        },
    }


def _expected_dncnn_train_patch_count(config: Mapping[str, Any]) -> int:
    """Calculate the source literal patch count without regenerating HDF5 data."""
    settings = config["protocol"]["denoising"]
    patch_size = int(settings.get("patch_size", 40))
    stride = int(settings.get("patch_stride", 10))
    if patch_size <= 0 or stride <= 0:
        raise ValueError("DnCNN patch_size and patch_stride must be positive")
    import cv2

    files = _denoising_png_files(Path(config["data"]["berkeley_root"]) / "train")
    total = 0
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not decode DnCNN training image: {path}")
        height, width = image.shape[:2]
        for scale in DNCNN_TRAINING_RESIZE["scales"]:
            # The source passes (height, width) to cv2.resize as (width, height).
            resized_height = int(width * float(scale))
            resized_width = int(height * float(scale))
            rows = max(0, (resized_height - patch_size) // stride + 1)
            columns = max(0, (resized_width - patch_size) // stride + 1)
            total += rows * columns
    return total


def _expected_set12_h5_values(config: Mapping[str, Any]) -> list[np.ndarray]:
    """Read the exact source-style Set12 tensors used to bind ``val.h5``."""
    import cv2

    values: list[np.ndarray] = []
    for path in _denoising_png_files(Path(config["data"]["set12_dir"])):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not decode Set12 image: {path}")
        values.append(
            np.expand_dims(image[:, :, 0].copy(), 0).astype(np.float32) / 255.0
        )
    return values


def _validate_flat_dncnn_h5(
    path: Path, *, label: str, expected_count: int,
    expected_shape: tuple[int, ...] | None,
    expected_values: Sequence[np.ndarray] | None = None,
) -> dict[str, object]:
    """Validate the flat, sequential HDF5 layout emitted by the released script."""
    import h5py

    if not path.is_file():
        raise FileNotFoundError(f"DnCNN {label} HDF5 artifact is missing: {path}")
    if expected_values is not None and len(expected_values) != expected_count:
        raise AssertionError("Set12 HDF5 validator received an invalid expected-value count")
    with h5py.File(path, "r") as handle:
        if len(handle) != expected_count:
            raise ValueError(
                f"DnCNN {label} HDF5 must contain exactly {expected_count} datasets, "
                f"found {len(handle)}"
            )
        observed_shapes: set[tuple[int, ...]] = set()
        observed_dtypes: set[str] = set()
        for index in range(expected_count):
            key = str(index)
            if key not in handle:
                raise ValueError(
                    f"DnCNN {label} HDF5 must use sequential decimal dataset keys; "
                    f"missing {key!r}"
                )
            dataset = handle[key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"DnCNN {label} HDF5 key {key!r} is not a dataset")
            shape = tuple(int(value) for value in dataset.shape)
            dtype = np.dtype(dataset.dtype)
            observed_shapes.add(shape)
            observed_dtypes.add(str(dtype))
            required_shape = (
                tuple(int(value) for value in expected_values[index].shape)
                if expected_values is not None else expected_shape
            )
            if shape != required_shape:
                raise ValueError(
                    f"DnCNN {label} HDF5 dataset {key!r} has shape {shape}; "
                    f"expected {required_shape}"
                )
            if dtype != np.dtype(np.float32):
                raise ValueError(
                    f"DnCNN {label} HDF5 dataset {key!r} has dtype {dtype}; "
                    "expected float32"
                )
            if expected_values is not None:
                observed = np.asarray(dataset)
                expected = expected_values[index]
                if not np.isfinite(observed).all() or observed.min() < 0.0 or observed.max() > 1.0:
                    raise ValueError(
                        f"DnCNN {label} HDF5 dataset {key!r} contains values outside [0, 1]"
                    )
                if not np.array_equal(observed, expected):
                    raise ValueError(
                        f"DnCNN {label} HDF5 dataset {key!r} does not match the "
                        "configured Set12 asset under the released BGR-channel preprocessing"
                    )
    return {
        "layout": DENOISING_H5_LAYOUT,
        "dataset_count": expected_count,
        "dataset_key_contract": "sequential_decimal_0_to_count_minus_one",
        "dataset_shapes": [list(shape) for shape in sorted(observed_shapes)],
        "dataset_dtypes": sorted(observed_dtypes),
        "set12_content_relation": (
            "exact_released_bgr_channel_match" if expected_values is not None else None
        ),
    }


def _denoising_h5_structure_identity(
    config: Mapping[str, Any], input_spec: Mapping[str, object],
) -> dict[str, object]:
    """Validate HDF5 structure and bind it to the configured Berkeley/Set12 asset."""
    source = denoising_preprocessing_spec(config)["source"]
    train_count = _expected_dncnn_train_patch_count(config)
    set12_values = _expected_set12_h5_values(config)
    train = _validate_flat_dncnn_h5(
        Path(str(input_spec["train_h5"])),
        label="training",
        expected_count=train_count,
        expected_shape=(
            1,
            int(config["protocol"]["denoising"].get("patch_size", 40)),
            int(config["protocol"]["denoising"].get("patch_size", 40)),
        ),
    )
    validation = _validate_flat_dncnn_h5(
        Path(str(input_spec["val_h5"])),
        label="validation",
        expected_count=len(set12_values),
        expected_shape=None,
        expected_values=set12_values,
    )
    return {
        "training": train,
        "validation": validation,
        "raw_asset_relation": {
            "berkeley_training_content_manifest_sha256": source[
                "training_content_manifest_sha256"
            ],
            "set12_content_manifest_sha256": source["set12_content_manifest_sha256"],
            "expected_training_patch_count": train_count,
            "set12_h5_relation": "exact_released_bgr_channel_match",
        },
    }


def denoising_h5_identity(config: Mapping[str, Any]) -> dict[str, object]:
    """Return a verified identity for either allowed DnCNN HDF5 input mode."""
    input_spec = resolve_denoising_h5_input(config)
    mode = str(input_spec["mode"])
    train_h5 = Path(str(input_spec["train_h5"]))
    val_h5 = Path(str(input_spec["val_h5"]))
    manifest_value = input_spec["preprocessing_manifest"]
    manifest_path = Path(str(manifest_value)) if manifest_value is not None else None
    paths = [train_h5, val_h5]
    if manifest_path is not None:
        paths.append(manifest_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"DnCNN HDF5 provenance artifact is missing: {path}")
    spec = denoising_preprocessing_spec(config)
    spec_hash = canonical_json_hash(spec)
    signature = (
        canonical_json_hash(input_spec),
        spec_hash,
        *(item for path in paths for item in (
            str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns,
        )),
    )
    if signature in _DENOISING_H5_IDENTITY_CACHE:
        return dict(_DENOISING_H5_IDENTITY_CACHE[signature])

    train_hash = sha256_file(train_h5)
    val_hash = sha256_file(val_h5)
    configured_hashes = input_spec["configured_artifact_sha256"]
    if not isinstance(configured_hashes, Mapping):
        raise AssertionError("resolved DnCNN HDF5 configuration is malformed")
    if mode == "source-provided":
        expected_train_hash = configured_hashes.get("train_h5")
        expected_val_hash = configured_hashes.get("val_h5")
        if expected_train_hash is not None and (
            train_hash != expected_train_hash or val_hash != expected_val_hash
        ):
            raise ValueError(
                "source-provided DnCNN HDF5 digest does not match the configured "
                "train_h5_sha256/val_h5_sha256 values"
            )
        preprocessing_manifest_identity: dict[str, object] | None = None
    else:
        if manifest_path is None:
            raise AssertionError("regenerated DnCNN HDF5 mode must have a manifest path")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != f"{SCHEMA}.denoising-preprocessing":
            raise ValueError("unrecognized denoising preprocessing manifest schema")
        manifest_mode = manifest.get("h5_mode")
        if manifest_mode is None and input_spec["configuration_origin"] != "legacy-denoising_h5_dir":
            raise ValueError(
                "regenerated DnCNN HDF5 manifest must declare h5_mode="
                "regenerated-from-raw; regenerate with train --overwrite-h5"
            )
        if manifest_mode not in {None, "regenerated-from-raw"}:
            raise ValueError("DnCNN preprocessing manifest has an incompatible h5_mode")
        if manifest.get("preprocessing_spec_sha256") != spec_hash:
            raise ValueError(
                "denoising HDF5 source or preprocessing settings changed; "
                "regenerate with train --overwrite-h5"
            )
        if manifest.get("train_h5_sha256") != train_hash or manifest.get("val_h5_sha256") != val_hash:
            raise ValueError("denoising HDF5 content does not match its preprocessing manifest")
        preprocessing_manifest_identity = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "h5_mode": manifest_mode or "legacy-unspecified",
            "preprocessing_spec_sha256": spec_hash,
        }

    structure = _denoising_h5_structure_identity(config, input_spec)
    identity: dict[str, object] = {
        "mode": mode,
        "input_write_policy": (
            "source-provided-artifacts-never-rewritten"
            if mode == "source-provided" else "regenerated-output-may-be-replaced-only-with-overwrite-h5"
        ),
        "configuration_origin": input_spec["configuration_origin"],
        "input_contract": input_spec,
        "train_h5": str(train_h5.resolve()),
        "train_h5_bytes": train_h5.stat().st_size,
        "train_h5_sha256": train_hash,
        "val_h5": str(val_h5.resolve()),
        "val_h5_bytes": val_h5.stat().st_size,
        "val_h5_sha256": val_hash,
        "configured_artifact_sha256": dict(configured_hashes),
        "h5_structure": structure,
        "preprocessing_manifest": preprocessing_manifest_identity,
        "raw_preprocessing_spec_sha256": spec_hash,
        "source_provided_training_content_note": (
            "Training patch tensors are structurally validated and file-hashed, but are "
            "not regenerated byte-for-byte because the supplied source HDF5 may have been "
            "created with a different OpenCV version."
            if mode == "source-provided" else None
        ),
    }
    _DENOISING_H5_IDENTITY_CACHE[signature] = dict(identity)
    return identity


def prepare_denoising_h5(config: Mapping[str, Any], overwrite: bool = False) -> dict[str, object]:
    """Create regenerated HDF5 data or validate immutable source-provided inputs."""
    input_spec = resolve_denoising_h5_input(config)
    mode = str(input_spec["mode"])
    if mode == "source-provided":
        if overwrite:
            raise ValueError(
                "--overwrite-h5 is forbidden for source-provided DnCNN HDF5 inputs; "
                "the public workflow never rewrites uploaded source artifacts"
            )
        return {"status": "source-provided-validated", **denoising_h5_identity(config)}

    data = config["data"]
    source = Path(data["berkeley_root"])
    set12 = Path(data["set12_dir"])
    expected_set12 = source / "Set12"
    if set12.resolve() != expected_set12.resolve():
        raise ValueError(
            "HDF5 preprocessing requires set12_dir to equal berkeley_root/Set12 so the "
            "validated test asset is used"
        )
    output = Path(str(input_spec["directory"]))
    train_h5 = Path(str(input_spec["train_h5"]))
    val_h5 = Path(str(input_spec["val_h5"]))
    manifest_path = Path(str(input_spec["preprocessing_manifest"]))
    existing = [path.exists() for path in (train_h5, val_h5, manifest_path)]
    if any(existing) and not overwrite:
        if all(existing):
            result = denoising_h5_identity(config)
            return {"status": "existing", **result}
        raise FileExistsError("partial denoising HDF5 output exists; use train --overwrite-h5")
    spec = denoising_preprocessing_spec(config)
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in (train_h5, val_h5, manifest_path):
            path.unlink(missing_ok=True)
    from utils.data import prepare_denoising_data

    prepare_denoising_data(
        data_path=str(source),
        patch_size=int(spec["patch_size"]),
        stride=int(spec["patch_stride"]),
        aug_times=1,
        h5_dir=str(output),
        seed=int(config["seed"]),
    )
    manifest = {
        "schema": f"{SCHEMA}.denoising-preprocessing",
        "created_utc": utc_now(),
        "status": "created",
        "h5_mode": "regenerated-from-raw",
        "train_h5": str(train_h5.resolve()),
        "val_h5": str(val_h5.resolve()),
        "train_h5_sha256": sha256_file(train_h5),
        "val_h5_sha256": sha256_file(val_h5),
        "preprocessing_spec": spec,
        "preprocessing_spec_sha256": canonical_json_hash(spec),
    }
    write_json(manifest_path, manifest)
    return {"status": "created", **denoising_h5_identity(config)}


def gaussian_noise_batch(
    images: torch.Tensor, sample_ids: Sequence[str], *, std: float,
    base_seed: int, epoch: int, namespace: str,
) -> tuple[torch.Tensor, list[int]]:
    noises = []
    seeds = []
    for sample_id, image in zip(sample_ids, images):
        seed = stable_seed(base_seed, namespace, epoch, sample_id)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noises.append(torch.randn(image.shape, generator=generator, dtype=image.dtype) * std)
        seeds.append(seed)
    return torch.stack(noises, dim=0), seeds


try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - older torch fallback
    from torch.nn.utils.stateless import functional_call as _functional_call


def _compute_weight_names(model: nn.Module) -> tuple[str, ...]:
    modules = dict(model.named_modules())
    names: list[str] = []
    for name, _parameter in model.named_parameters():
        owner_name, _, local_name = name.rpartition(".")
        owner = modules.get(owner_name)
        if local_name == "weight" and isinstance(owner, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            names.append(name)
    return tuple(names)


def equation_forward(
    model: nn.Module, inputs: torch.Tensor, device_model: str, level: float,
    seed: int,
) -> torch.Tensor:
    """Differentiable Eq.(1)/(2) forward with all convolutional weights."""
    if level < 0 or not math.isfinite(level):
        raise ValueError("noise level must be finite and non-negative")
    if level == 0:
        return model(inputs, 0.0, "none")
    if device_model not in DEVICE_MODELS:
        raise ValueError(device_model)
    replacements: dict[str, torch.Tensor] = {}
    parameter_map = dict(model.named_parameters())
    with seeded_forward(seed, inputs.device):
        for name in _compute_weight_names(model):
            weight = parameter_map[name]
            if device_model == "rram":
                replacements[name] = weight * torch.exp(torch.randn_like(weight) * level)
            else:
                maximum = pcm_layerwise_wmax(weight)
                replacements[name] = weight + torch.randn_like(weight) * (level * maximum)
    return _functional_call(model, replacements, (inputs, 0.0, "none"), strict=False)


def segmentation_nominal_kd_forward(
    model: nn.Module, inputs: torch.Tensor, *,
    bn_update_policy: Mapping[str, object],
) -> torch.Tensor:
    """Run the nominal KD branch under the selected persistent-buffer policy."""
    policy = bn_update_policy.get("policy")
    if policy == "legacy_dual_branch":
        return model(inputs, 0.0, "none")
    if policy != "noisy_task_only":
        raise ValueError("UNet MTRD BatchNorm policy is unsupported")

    # Original parameters retain autograd links; cloned buffers prevent this branch
    # from mutating BatchNorm running statistics before the noisy task forward.
    state: dict[str, torch.Tensor] = dict(model.named_parameters())
    state.update({
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    })
    return _functional_call(model, state, (inputs, 0.0, "none"), strict=True)


class EpochDeltaBalancer:
    """Cross-epoch Eq.(6) softmax with an explicit sign interpretation."""

    def __init__(self, count: int, direction: str, temperature: float):
        if count <= 0:
            raise ValueError("teacher count must be positive")
        if direction not in EQ6_CHOICES:
            raise ValueError(f"unknown Eq.(6) direction: {direction}")
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("Eq.(6) temperature must be positive")
        self.count = count
        self.direction = direction
        self.temperature = float(temperature)
        self.previous: torch.Tensor | None = None
        self.beta = torch.full((count,), 1.0 / count, dtype=torch.float64)

    @property
    def equation(self) -> str:
        if self.direction == "released_rank_underperformance":
            return "beta=normalize(rank_weights(delta_ascending,[2.5,2,1.5,1,0.5]))"
        sign = "current-previous" if EQ6_CHOICES[self.direction] > 0 else "previous-current"
        return f"beta=softmax(({sign})/{self.temperature:g})"

    def update(self, performance: Sequence[float]) -> torch.Tensor:
        current = torch.as_tensor(performance, dtype=torch.float64)
        if current.shape != (self.count,) or not torch.isfinite(current).all():
            raise ValueError("Eq.(6) performance vector is invalid")
        if self.previous is not None:
            delta = current - self.previous
            if self.direction == "released_rank_underperformance":
                if self.count != 5:
                    raise ValueError(
                        "the released rank policy requires exactly five robust teachers"
                    )
                order = sorted(range(self.count), key=lambda index: (delta[index].item(), index))
                weights = torch.empty(self.count, dtype=torch.float64)
                for rank, index in enumerate(order):
                    weights[index] = 2.5 - 0.5 * rank
                self.beta = weights / weights.sum()
            else:
                sign = EQ6_CHOICES[self.direction]
                self.beta = torch.softmax(sign * delta / self.temperature, dim=0)
        self.previous = current.clone()
        return self.beta.clone()

    def state_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "direction": self.direction,
            "temperature": self.temperature,
            "previous": None if self.previous is None else self.previous.tolist(),
            "beta": self.beta.tolist(),
            "equation": self.equation,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["count"]) != self.count or state["direction"] != self.direction:
            raise ValueError("Eq.(6) balancer resume mismatch")
        if not math.isclose(float(state["temperature"]), self.temperature):
            raise ValueError("Eq.(6) temperature resume mismatch")
        previous = state.get("previous")
        self.previous = None if previous is None else torch.as_tensor(previous, dtype=torch.float64)
        self.beta = torch.as_tensor(state["beta"], dtype=torch.float64)


class ReleasedBH4RankBalancer:
    """Reproduce the four-teacher rank update in the released BH trainer."""

    raw_rank_weights = (2.5, 2.0, 1.5, 1.0)

    def __init__(self) -> None:
        self.count = 4
        self.direction = "released_bh4_rank_current_minus_previous_zero_baseline"
        self.previous = torch.zeros(self.count, dtype=torch.float64)
        self.beta = torch.full((self.count,), 1.0 / self.count, dtype=torch.float64)

    @property
    def equation(self) -> str:
        return (
            "beta=normalize(rank_weights(current-previous ascending,"
            "[2.5,2,1.5,1]); previous=zeros before epoch 1)"
        )

    def update(self, performance: Sequence[float]) -> torch.Tensor:
        current = torch.as_tensor(performance, dtype=torch.float64)
        if current.shape != (self.count,) or not torch.isfinite(current).all():
            raise ValueError("BH4 performance vector is invalid")
        delta = current - self.previous
        # Python's stable sort preserves the released teacher order for ties.
        order = sorted(range(self.count), key=lambda index: delta[index].item())
        raw = torch.empty(self.count, dtype=torch.float64)
        for rank, index in enumerate(order):
            raw[index] = self.raw_rank_weights[rank]
        self.beta = raw / raw.sum()
        self.previous = current.clone()
        return self.beta.clone()

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": "released_bh4_rank",
            "count": self.count,
            "direction": self.direction,
            "previous": self.previous.tolist(),
            "beta": self.beta.tolist(),
            "raw_rank_weights": list(self.raw_rank_weights),
            "equation": self.equation,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("kind") != "released_bh4_rank" or int(state.get("count", -1)) != self.count:
            raise ValueError("BH4 balancer resume mismatch")
        if tuple(float(item) for item in state.get("raw_rank_weights", [])) != self.raw_rank_weights:
            raise ValueError("BH4 rank-weight resume mismatch")
        previous = torch.as_tensor(state["previous"], dtype=torch.float64)
        beta = torch.as_tensor(state["beta"], dtype=torch.float64)
        if previous.shape != (self.count,) or beta.shape != (self.count,):
            raise ValueError("BH4 balancer state has an invalid shape")
        self.previous = previous
        self.beta = beta


def build_dncnn_balancer(
    contract: DnCNNMTRDContract,
) -> EpochDeltaBalancer | ReleasedBH4RankBalancer:
    if contract.name == "released_bh4_source_semantics":
        return ReleasedBH4RankBalancer()
    if contract.eq6_temperature is None:
        raise AssertionError("paper-interpreted DnCNN requires an Eq.(6) temperature")
    return EpochDeltaBalancer(
        len(contract.mtrd_teacher_specs), contract.eq6_direction, contract.eq6_temperature,
    )


def paper_mtrd_loss(
    task: str, nominal_student: torch.Tensor, noisy_student: torch.Tensor,
    teacher_outputs: Sequence[torch.Tensor], target: torch.Tensor,
    beta: torch.Tensor, *, alpha: float, temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply Eq.(4) as a beta-weighted sum of per-teacher losses."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    if len(teacher_outputs) == 0 or beta.numel() != len(teacher_outputs):
        raise ValueError("one beta value is required per teacher")
    weights = beta.to(nominal_student.device, nominal_student.dtype)
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    if task == "denoising":
        individual_kd = torch.stack([
            F.mse_loss(nominal_student, teacher.detach())
            for teacher in teacher_outputs
        ])
        kd = torch.sum(weights * individual_kd)
        task_loss = F.mse_loss(noisy_student, target)
        interpretation = nominal_student.new_tensor(0.0)
    elif task == "segmentation":
        if temperature <= 0:
            raise ValueError("distillation temperature must be positive")
        student_logits = nominal_student / temperature
        individual_kd = torch.stack([
            F.binary_cross_entropy_with_logits(
                student_logits, torch.sigmoid(teacher.detach() / temperature),
            )
            for teacher in teacher_outputs
        ])
        kd = torch.sum(weights * individual_kd) * temperature**2
        task_loss = F.binary_cross_entropy_with_logits(noisy_student, target)
        interpretation = nominal_student.new_tensor(1.0)
    else:
        raise ValueError(task)
    total = alpha * kd + (1.0 - alpha) * task_loss
    return total, {
        "kd": kd.detach(),
        "task": task_loss.detach(),
        "task_specific_kd_code": interpretation,
    }


def released_bh4_source_loss(
    student_output: torch.Tensor, target: torch.Tensor,
    teacher_outputs: Sequence[torch.Tensor], beta: torch.Tensor, *,
    alpha: float = 0.7, temperature: float = 5.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the released BH4 one-channel KL-plus-sum-MSE training expression.

    This intentionally preserves the source behavior: KL is evaluated after a
    channel-wise softmax on a one-channel residual map and is therefore zero.
    The returned metadata makes that degeneracy explicit instead of presenting
    it as a functional distillation signal.
    """
    if student_output.ndim < 2 or student_output.shape[1] != 1:
        raise ValueError("BH4 source semantics require a one-channel DnCNN residual map")
    if len(teacher_outputs) != 4 or beta.numel() != 4:
        raise ValueError("BH4 source semantics require exactly four teacher outputs")
    if not math.isclose(float(alpha), 0.7, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("BH4 source semantics fix alpha=0.7")
    if not math.isclose(float(temperature), 5.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("BH4 source semantics fix temperature=5")
    weights = beta.to(student_output.device, student_output.dtype)
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    mixed_teacher = sum(
        weight * teacher.detach()
        for weight, teacher in zip(weights, teacher_outputs)
    )
    kd = F.kl_div(
        F.log_softmax(student_output / temperature, dim=1),
        F.softmax(mixed_teacher / temperature, dim=1),
        reduction="mean",
    )
    task_sum = F.mse_loss(student_output, target, reduction="sum")
    batch_size = int(student_output.shape[0])
    total = (
        kd * (temperature**2 * 2.0 * alpha) + task_sum * (1.0 - alpha)
    ) / (batch_size * 2.0)
    return total, {
        "kd": kd.detach(),
        "task": (task_sum / (batch_size * 2.0)).detach(),
        "task_sum": task_sum.detach(),
        "mixed_teacher": mixed_teacher.detach(),
        "kd_is_degenerate": True,
        "loss_reduction": "kl_mean_plus_mse_sum_divided_by_2_batch",
    }


def quantize_weight_tensor(weight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("weight quantization requires at least two bits")
    maximum = weight.detach().abs().amax()
    if not torch.isfinite(maximum) or maximum.item() == 0.0:
        return weight.detach().clone()
    qmax = 2 ** (bits - 1) - 1
    scale = maximum / qmax
    return torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale


def quantize_compute_weights_(model: nn.Module, bits: int) -> None:
    names = set(_compute_weight_names(model))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.copy_(quantize_weight_tensor(parameter, bits))


def _get_parent_module(model: nn.Module, name: str) -> nn.Module:
    parent = model
    for part in name.split(".")[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent


def convert_noisy_to_standard(model: nn.Module) -> nn.Module:
    for name, module in list(model.named_modules()):
        if isinstance(module, NoisyConv2d):
            setattr(_get_parent_module(model, name), name.split(".")[-1], module.conv)
        elif isinstance(module, NoisyLinear):
            setattr(_get_parent_module(model, name), name.split(".")[-1], module.linear)
    return model


def patch_aihwkit_extra_args(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if not module.__class__.__module__.startswith("aihwkit.nn"):
            continue
        original = module.forward

        def forward(self, inputs, *unused_args, _forward=original, **unused_kwargs):
            return _forward(inputs)

        module.forward = MethodType(forward, module)
    return model


def build_aihwkit_paper_model(
    base: nn.Module, *, eta: float, weight_bits: int, dac_bits: int,
    adc_bits: int, seed: int,
    realization_scope: str, execution_device: torch.device | str = "cpu",
) -> nn.Module:
    """Convert to AIHWKit using the classification chain's shared config."""
    if realization_scope == "per_mac":
        from simulators.aihwkit import require_replayable_realization_scope

        require_replayable_realization_scope("per-mac")
    if realization_scope not in PCM_SCOPES:
        raise ValueError(realization_scope)
    from simulators.aihwkit import convert_model

    standard = convert_noisy_to_standard(copy.deepcopy(base).cpu())
    unsupported = [
        name for name, module in standard.named_modules()
        if isinstance(module, nn.ConvTranspose2d)
    ]
    if unsupported:
        raise RuntimeError(
            "AIHWKit 1.1 cannot convert ConvTranspose2d; refusing a mixed analog/digital "
            f"UNet evaluation. Unsupported operators: {unsupported}"
        )
    quantize_compute_weights_(standard, weight_bits)
    shared_scope = "per-mac" if realization_scope == "per_mac" else "fixed-trial"
    analog = convert_model(
        standard,
        eta,
        input_bits=dac_bits,
        output_bits=adc_bits,
        seed=seed,
        realization_scope=shared_scope,
    )
    patch_aihwkit_extra_args(analog)
    return analog.to(execution_device).eval()


def psnr_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction.float(), target.float()).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def dice_values(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prediction = (torch.sigmoid(logits) > 0.5).to(targets.dtype)
    intersection = (prediction * targets).sum(dim=(1, 2, 3))
    total = (prediction + targets).sum(dim=(1, 2, 3))
    return (2.0 * intersection + 1e-8) / (total + 1e-8)


def segmentation_metric_observations(
    logits: torch.Tensor, targets: torch.Tensor, sample_ids: Sequence[str], *,
    metric_contract: Mapping[str, object], batch_index: int,
) -> list[tuple[str, float, str]]:
    """Return one Carvana metric observation per configured reporting unit."""
    aggregation = metric_contract.get("metric_aggregation")
    if aggregation == "per_image_mean":
        metrics = dice_values(logits, targets).detach().cpu().tolist()
        return [
            (str(sample_id), float(metric), "Dice_fraction")
            for sample_id, metric in zip(sample_ids, metrics)
        ]
    if aggregation == "released_batch_global":
        if not sample_ids:
            raise ValueError("released batch-global Dice requires at least one sample")
        predictions = (torch.sigmoid(logits) > 0.5).to(targets.dtype)
        intersection = (predictions * targets).sum()
        total = (predictions + targets).sum()
        metric = float((2.0 * intersection / (total + 1e-8)).item())
        return [
            (
                f"batch-{batch_index:04d}:{sample_ids[0]}..{sample_ids[-1]}",
                metric,
                "Dice_batch_global_fraction",
            )
        ]
    raise ValueError("segmentation metric contract has an unsupported aggregation")


def build_task_datasets(
    config: Mapping[str, Any], task: str, *, include_training: bool = True,
) -> tuple[Dataset | None, Dataset, dict[str, object]]:
    if task == "denoising":
        h5_input = resolve_denoising_h5_input(config)
        train = (
            H5DenoisingDataset(Path(str(h5_input["train_h5"])))
            if include_training else None
        )
        validation = Set12Dataset(config["data"]["set12_dir"])
        identity = denoising_asset_identity(config, include_hash=True)
        if include_training:
            identity["preprocessed_h5"] = denoising_h5_identity(config)
        else:
            identity["preprocessed_h5_input"] = h5_input
        return train, validation, identity
    index = build_carvana_index(config)
    split = load_carvana_split(config, index)
    settings = config["protocol"]["segmentation"]
    resize_backend = configured_carvana_resize_backend(config)
    train = (
        CarvanaSplitDataset(
            index, split["train_ids"],
            height=int(settings["image_height"]), width=int(settings["image_width"]),
            train=True, seed=int(config["seed"]), resize_backend=resize_backend,
        )
        if include_training else None
    )
    validation = CarvanaSplitDataset(
        index, split["test_ids"],
        height=int(settings["image_height"]), width=int(settings["image_width"]),
        train=False, seed=int(config["seed"]), resize_backend=resize_backend,
    )
    identity = {
        "name": "Carvana",
        "asset_count": len(index),
        "split_path": split["path"],
        "split_sha256": split["sha256"],
        "train_count": len(split["train_ids"]),
        "test_count": len(split["test_ids"]),
        "excluded_count": len(split["excluded_ids"]),
        "split_contract": split["split_contract"],
        "reference_split_author_verified": split["reference_split_author_verified"],
        "resize_backend": resize_backend,
        "resize_interpolation": (
            {"image": "cv2.INTER_LINEAR", "mask": "cv2.INTER_NEAREST"}
            if resize_backend == "released-opencv"
            else {"image": "PIL.BILINEAR", "mask": "PIL.NEAREST"}
        ),
        "split_derivation": split.get("derivation"),
        "asset": carvana_asset_identity(index, include_hash=True),
    }
    return train, validation, identity


def build_feedback_dataset(
    config: Mapping[str, Any], task: str, train: Dataset, test: Dataset,
) -> tuple[Dataset, dict[str, object]]:
    settings = config["protocol"][task]
    split_name = str(settings["feedback_split"])
    requested = int(settings.get("feedback_sample_count", 0))
    if task == "denoising":
        if split_name == "set12_test":
            selected: Dataset = test
            sample_ids = [path.stem for path in test.paths]
            reuses_test = True
        elif split_name == "berkeley_training":
            count = len(train) if requested == 0 else min(requested, len(train))
            selected = Subset(train, list(range(count)))
            sample_ids = [train.keys[index] for index in range(count)]
            reuses_test = False
        else:
            raise ValueError(split_name)
    else:
        if split_name == "carvana_test":
            selected = test
            sample_ids = list(test.sample_ids)
            reuses_test = True
        elif split_name == "carvana_training":
            index = build_carvana_index(config)
            split = load_carvana_split(config, index)
            count = len(split["train_ids"]) if requested == 0 else min(
                requested, len(split["train_ids"])
            )
            sample_ids = list(split["train_ids"][:count])
            selected = CarvanaSplitDataset(
                index, sample_ids,
                height=int(settings["image_height"]), width=int(settings["image_width"]),
                train=False, seed=int(config["seed"]),
                resize_backend=configured_carvana_resize_backend(config),
            )
            reuses_test = False
        else:
            raise ValueError(split_name)
    return selected, {
        "split": split_name,
        "sample_count": len(sample_ids),
        "sample_ids_sha256": canonical_json_hash(sample_ids),
        "reuses_final_test_set": reuses_test,
        "warning": (
            "feedback and final reporting reuse the same test set"
            if reuses_test else None
        ),
    }


def optimizer_for(
    config: Mapping[str, Any], task: str, model: nn.Module, *, student: bool,
    device_model: str | None = None,
) -> torch.optim.Optimizer:
    if task == "denoising":
        if device_model is None:
            raise ValueError("device_model is required for a DnCNN optimizer")
        contract = resolve_dncnn_mtrd_contract(config, device_model)
        kwargs = dict(contract.optimizer_kwargs)
        kwargs["betas"] = tuple(float(item) for item in kwargs["betas"])
        return torch.optim.Adam(
            model.parameters(), lr=contract.initial_learning_rate, **kwargs,
        )
    contract = unet_optimizer_contract(config, student=student)
    optimizer = contract["optimizer"]
    if not isinstance(optimizer, Mapping):
        raise RuntimeError("UNet optimizer profile has an invalid optimizer contract")
    name = str(optimizer["name"])
    learning_rate = float(optimizer["learning_rate"])
    raw_kwargs = optimizer["kwargs"]
    if not isinstance(raw_kwargs, Mapping):
        raise RuntimeError("UNet optimizer profile has invalid optimizer kwargs")
    kwargs = dict(raw_kwargs)
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, **kwargs)
    if name == "Adam":
        betas = kwargs.get("betas")
        if not isinstance(betas, list) or len(betas) != 2:
            raise RuntimeError("UNet Adam profile must define exactly two beta values")
        kwargs["betas"] = tuple(float(value) for value in betas)
        return torch.optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    raise RuntimeError(f"unsupported UNet optimizer profile constructor: {name}")


def learning_rate(
    config: Mapping[str, Any], task: str, epoch: int, *, student: bool,
    device_model: str | None = None,
) -> float:
    if task == "segmentation":
        optimizer = resolve_unet_optimizer_profile(config)["optimizer"]
        if not isinstance(optimizer, Mapping):
            raise RuntimeError("UNet optimizer profile has an invalid optimizer contract")
        return float(optimizer["learning_rate"])
    if device_model is None:
        raise ValueError("device_model is required for a DnCNN learning-rate schedule")
    contract = resolve_dncnn_mtrd_contract(config, device_model)
    milestones = (
        contract.student_lr_milestones if student else contract.teacher_lr_milestones
    )
    reductions = sum(epoch >= milestone for milestone in milestones)
    return contract.initial_learning_rate / (10.0 ** reductions)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


@torch.no_grad()
def feedback_metric(
    model: nn.Module, validation: Dataset, *, task: str,
    device_model: str, level: float, config: Mapping[str, Any],
    device: torch.device, seed_namespace: str, feedback_epoch: int = 0,
    dncnn_contract: DnCNNMTRDContract | None = None,
) -> float:
    model.eval()
    base_seed = int(config["seed"])
    if task == "denoising":
        std = float(config["protocol"]["denoising"]["image_noise_std_255"]) / 255.0
        source_semantics = (
            dncnn_contract is not None
            and dncnn_contract.name == "released_bh4_source_semantics"
        )
        values: list[float] = []
        for index in range(len(validation)):
            clean, sample_id = validation[index]
            clean = clean.unsqueeze(0)
            input_noise, _ = gaussian_noise_batch(
                clean, [sample_id], std=std, base_seed=base_seed,
                epoch=feedback_epoch if source_semantics else 0,
                namespace=(
                    f"{seed_namespace}-source-feedback-input-{level:g}"
                    if source_semantics else "set12-fixed-input-noise"
                ),
            )
            noisy = (clean + input_noise).to(device)
            clean_device = clean.to(device)
            forward_seed = (
                stable_seed(
                    base_seed,
                    seed_namespace,
                    task,
                    device_model,
                    f"{level:g}",
                    feedback_epoch,
                    sample_id,
                )
                if source_semantics else stable_seed(
                    base_seed, seed_namespace, task, device_model, f"{level:g}", sample_id,
                )
            )
            prediction = noisy - equation_forward(
                model, noisy, device_model, level, forward_seed,
            )
            prediction = prediction.clamp(0.0, 1.0)
            values.append(psnr_value(prediction, clean_device))
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError("DnCNN feedback PSNR is empty or non-finite")
        return float(sum(values) / len(values))

    settings = config["protocol"]["segmentation"]
    metric_contract = resolve_segmentation_metric_contract(config)
    loader = epoch_loader(
        validation,
        batch_size=int(settings["batch_size"]),
        workers=int(settings["num_workers"]),
        seed=base_seed,
        epoch=0,
        shuffle=False,
    )
    values: list[float] = []
    for batch_index, (images, masks, sample_ids) in enumerate(loader):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)
        forward_seed = stable_seed(
            base_seed, seed_namespace, task, device_model, f"{level:g}", batch_index,
        )
        logits = equation_forward(model, images, device_model, level, forward_seed)
        observations = segmentation_metric_observations(
            logits,
            masks,
            sample_ids,
            metric_contract=metric_contract,
            batch_index=batch_index,
        )
        values.extend(value for _sample_id, value, _metric in observations)
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Carvana feedback Dice is empty or non-finite")
    return float(sum(values) / len(values))


def checkpoint_payload(
    model: nn.Module, *, task: str, role: str, device_model: str,
    level: float, epoch: int, metric: float, config: Mapping[str, Any],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": f"{SCHEMA}.checkpoint",
        "state_dict": model.state_dict(),
        "task": task,
        "role": role,
        "device_model": device_model,
        "noise_level": level,
        "epoch": epoch,
        "validation_metric": metric,
        "config_sha256": canonical_json_hash(config),
        "environment": environment_identity(),
        "first_party_source_sha256": source_tree_sha256(CODE_ROOT),
    }
    if extra:
        payload.update(extra)
    if task == "segmentation":
        payload["optimizer_contract"] = unet_optimizer_contract(
            config, student=role == "mtrd_student",
        )
        payload["segmentation_metric_contract"] = resolve_segmentation_metric_contract(
            config
        )
        if role == "mtrd_student":
            payload["segmentation_mtrd_bn_update_policy"] = (
                resolve_segmentation_mtrd_bn_update_policy(config)
            )
            payload["segmentation_mtrd_contract"] = segmentation_mtrd_contract_payload(
                resolve_segmentation_mtrd_contract(config, device_model)
            )
    return payload


def training_manifest(
    path: Path, *, config: Mapping[str, Any], task: str, role: str,
    device_model: str, level: float, metric: float, dataset: Mapping[str, object],
    log_path: Path, extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema": f"{SCHEMA}.training-manifest",
        "created_utc": utc_now(),
        "task": task,
        "role": role,
        "device_model": device_model,
        "noise_level": level,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_feedback_metric": metric,
        "dataset": dict(dataset),
        "epoch_log": str(log_path.resolve()),
        "epoch_log_sha256": sha256_file(log_path),
        "config_sha256": canonical_json_hash(config),
        "environment": environment_identity(),
        "source": git_identity(CODE_ROOT),
        "no_manual_metric_offset": True,
        "equation_implementation": True,
        "author_training_details_confirmed": False,
        "protocol_warning": PROTOCOL_WARNING,
    }
    if extra:
        manifest.update(extra)
    if task == "segmentation":
        manifest["optimizer_contract"] = unet_optimizer_contract(
            config, student=role == "mtrd_student",
        )
        manifest["segmentation_metric_contract"] = resolve_segmentation_metric_contract(
            config
        )
        if role == "mtrd_student":
            manifest["segmentation_mtrd_bn_update_policy"] = (
                resolve_segmentation_mtrd_bn_update_policy(config)
            )
            manifest["segmentation_mtrd_contract"] = segmentation_mtrd_contract_payload(
                resolve_segmentation_mtrd_contract(config, device_model)
            )
    return manifest


TEACHER_LOG_FIELDS = (
    "epoch", "learning_rate", "train_loss", "validation_metric", "best_metric",
    "last_metric",
)
STUDENT_LOG_FIELDS = (
    "epoch", "learning_rate", "train_loss", "train_kd_loss", "train_task_loss",
    "robust_mean_metric", "best_robust_mean_metric", "beta_used_json",
    "beta_next_json", "performance_json",
)


def train_one_teacher(
    config: Mapping[str, Any], *, task: str, device_model: str,
    spec: TeacherSpec, device: torch.device, resume: bool, overwrite: bool,
) -> Path:
    train_dataset, final_test, dataset_identity = build_task_datasets(config, task)
    feedback, feedback_identity = build_feedback_dataset(
        config, task, train_dataset, final_test,
    )
    output = checkpoint_path(
        config, task, "clean" if spec.clean else "teacher",
        None if spec.clean else device_model,
        None if spec.clean else spec.level,
    )
    state_path = output.with_suffix(".train-state.pt")
    log_path = output.with_suffix(".epochs.csv")
    if output.exists() and not resume and not overwrite:
        raise FileExistsError(f"checkpoint exists; use --resume or --overwrite: {output}")
    if overwrite and not resume and log_path.exists():
        log_path.unlink()

    seed_device = "none" if spec.clean else device_model
    run_seed = stable_seed(int(config["seed"]), "teacher", task, seed_device, spec.label)
    set_deterministic(run_seed)
    dncnn_contract = (
        resolve_dncnn_mtrd_contract(config, device_model)
        if task == "denoising" else None
    )
    dncnn_contract_payload = (
        dncnn_mtrd_contract_payload(dncnn_contract)
        if dncnn_contract is not None else None
    )
    model = build_model(task).to(device)
    optimizer = optimizer_for(
        config, task, model, student=False, device_model=device_model,
    )
    start_epoch = 0
    best = float("-inf")
    final_metric = float("-inf")
    resume_identity = {
        "config_sha256": canonical_json_hash(config),
        "environment_identity_sha256": canonical_json_hash(environment_identity()),
        "first_party_source_sha256": source_tree_sha256(CODE_ROOT),
        "dataset_identity_sha256": canonical_json_hash(dataset_identity),
        "feedback_identity_sha256": canonical_json_hash(feedback_identity),
        "task": task,
        "device_model": seed_device,
        "teacher_label": spec.label,
        "resolved_device": str(device),
        "dncnn_mtrd_contract_sha256": (
            canonical_json_hash(dncnn_contract_payload)
            if dncnn_contract_payload is not None else None
        ),
    }
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state is missing: {state_path}")
        state = load_checkpoint(state_path, map_location=device)
        if state.get("resume_identity") != resume_identity:
            raise ValueError(
                "resume identity mismatch: configuration, data, feedback split, "
                "teacher role, environment, source, or device changed"
            )
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best_metric"])
        final_metric = float(state.get("last_metric", best))

    settings = config["protocol"][task]
    epochs = (
        dncnn_contract.teacher_epochs
        if dncnn_contract is not None else int(settings["teacher_epochs"])
    )
    batch_size = int(settings["batch_size"])
    workers = int(settings["num_workers"])
    for epoch in range(start_epoch, epochs):
        lr = learning_rate(
            config, task, epoch, student=False, device_model=device_model,
        )
        set_optimizer_lr(optimizer, lr)
        loader = epoch_loader(
            train_dataset, batch_size=batch_size, workers=workers,
            seed=run_seed, epoch=epoch, shuffle=True,
        )
        model.train()
        loss_total = 0.0
        item_total = 0
        for batch_index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            if task == "denoising":
                clean, sample_ids = batch
                input_noise, _ = gaussian_noise_batch(
                    clean, sample_ids,
                    std=float(settings["image_noise_std_255"]) / 255.0,
                    base_seed=run_seed, epoch=epoch, namespace="teacher-input-noise",
                )
                noisy = (clean + input_noise).to(device)
                target = input_noise.to(device)
                if spec.clean:
                    prediction = model(noisy, 0.0, "none")
                else:
                    prediction = equation_forward(
                        model, noisy, device_model, spec.level,
                        stable_seed(run_seed, epoch, batch_index, "teacher-device-noise"),
                    )
                count = int(clean.size(0))
                if dncnn_contract is not None and dncnn_contract.name == "released_bh4_source_semantics":
                    loss = F.mse_loss(prediction, target, reduction="sum") / (count * 2.0)
                else:
                    loss = F.mse_loss(prediction, target)
            else:
                images, masks, _sample_ids = batch
                images = images.to(device)
                targets = masks.unsqueeze(1).to(device)
                if spec.clean:
                    prediction = model(images, 0.0, "none")
                else:
                    prediction = equation_forward(
                        model, images, device_model, spec.level,
                        stable_seed(run_seed, epoch, batch_index, "teacher-device-noise"),
                    )
                loss = F.binary_cross_entropy_with_logits(prediction, targets)
                count = int(images.size(0))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite teacher loss at epoch={epoch} batch={batch_index}")
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * count
            item_total += count
        eval_level = 0.0 if spec.clean else spec.level
        metric = feedback_metric(
            model, feedback, task=task, device_model=device_model,
            level=eval_level, config=config, device=device,
            seed_namespace=f"teacher-validation-{spec.label}",
            feedback_epoch=epoch + 1,
            dncnn_contract=dncnn_contract,
        )
        final_metric = metric
        if metric > best:
            best = metric
        atomic_torch_save(
            output,
            checkpoint_payload(
                model, task=task,
                role="clean_teacher" if spec.clean else "variation_teacher",
                device_model="none" if spec.clean else device_model,
                level=eval_level, epoch=epoch + 1, metric=metric, config=config,
                extra=(
                    {"dncnn_mtrd_contract": dncnn_contract_payload}
                    if dncnn_contract_payload is not None else None
                ),
            ),
        )
        append_csv(
            log_path,
            {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "train_loss": loss_total / max(1, item_total),
                "validation_metric": metric,
                "best_metric": best,
                "last_metric": final_metric,
            },
            TEACHER_LOG_FIELDS,
        )
        atomic_torch_save(
            state_path,
            {
                "schema": f"{SCHEMA}.training-state",
                "config_sha256": canonical_json_hash(config),
                "resume_identity": resume_identity,
                "epoch": epoch,
                "best_metric": best,
                "last_metric": final_metric,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dncnn_mtrd_contract": dncnn_contract_payload,
            },
        )
        print(
            f"{task} teacher={spec.label} epoch={epoch+1}/{epochs} "
            f"loss={loss_total/max(1,item_total):.6f} metric={metric:.6f} best={best:.6f}"
        )

    manifest = training_manifest(
        output, config=config, task=task,
        role="clean_teacher" if spec.clean else "variation_teacher",
        device_model="none" if spec.clean else device_model,
        level=0.0 if spec.clean else spec.level,
        metric=final_metric, dataset=dataset_identity, log_path=log_path,
        extra={
            "training_noise_realization": "fresh independent tensor per compute layer and forward",
            "optimizer": (
                {
                    "name": dncnn_contract.optimizer,
                    **dict(dncnn_contract.optimizer_kwargs),
                }
                if dncnn_contract is not None else "SGD"
            ),
            "epochs": epochs,
            "checkpoint_selection": "final_epoch",
            "best_observed_feedback_metric": best,
            "feedback_dataset": feedback_identity,
            "resume_identity": resume_identity,
            "dncnn_mtrd_contract": dncnn_contract_payload,
            "clean_checkpoint_is_mtrd_teacher": (
                None if dncnn_contract is None else dncnn_contract.include_clean_teacher
            ),
            "equation_implementation": (
                False
                if dncnn_contract is not None and dncnn_contract.kd_is_degenerate
                else True
            ),
        },
    )
    write_json(output.with_suffix(".manifest.json"), manifest)
    return output


def load_teacher_pool(
    config: Mapping[str, Any], *, task: str, device_model: str,
    device: torch.device,
) -> tuple[tuple[TeacherSpec, ...], list[nn.Module], list[Path]]:
    specs = mtrd_teacher_specs(config, task, device_model)
    teachers: list[nn.Module] = []
    paths: list[Path] = []
    for spec in specs:
        path = checkpoint_path(
            config, task, "clean" if spec.clean else "teacher",
            None if spec.clean else device_model,
            None if spec.clean else spec.level,
        )
        teacher = load_model_strict(path, task, device)
        teacher.eval()
        teacher.requires_grad_(False)
        teachers.append(teacher)
        paths.append(path)
    return specs, teachers, paths


def train_mtrd_student(
    config: Mapping[str, Any], *, task: str, device_model: str,
    device: torch.device, resume: bool, overwrite: bool,
) -> Path:
    train_dataset, final_test, dataset_identity = build_task_datasets(config, task)
    feedback, feedback_identity = build_feedback_dataset(
        config, task, train_dataset, final_test,
    )
    dncnn_contract = (
        resolve_dncnn_mtrd_contract(config, device_model)
        if task == "denoising" else None
    )
    dncnn_contract_payload = (
        dncnn_mtrd_contract_payload(dncnn_contract)
        if dncnn_contract is not None else None
    )
    source_bh4 = (
        dncnn_contract is not None
        and dncnn_contract.name == "released_bh4_source_semantics"
    )
    segmentation_metric_contract = (
        resolve_segmentation_metric_contract(config)
        if task == "segmentation" else None
    )
    segmentation_bn_update_policy = (
        resolve_segmentation_mtrd_bn_update_policy(config)
        if task == "segmentation" else None
    )
    segmentation_mtrd_contract = (
        resolve_segmentation_mtrd_contract(config, device_model)
        if task == "segmentation" else None
    )
    segmentation_mtrd_payload = (
        segmentation_mtrd_contract_payload(segmentation_mtrd_contract)
        if segmentation_mtrd_contract is not None else None
    )
    specs, teachers, teacher_paths = load_teacher_pool(
        config, task=task, device_model=device_model, device=device,
    )
    initial_level = (
        dncnn_contract.student_initial_level
        if dncnn_contract is not None else segmentation_mtrd_contract.student_initial_level
    )
    initial_path = checkpoint_path(config, task, "teacher", device_model, initial_level)
    output = checkpoint_path(config, task, "mtrd", device_model)
    state_path = output.with_suffix(".train-state.pt")
    log_path = output.with_suffix(".epochs.csv")
    if output.exists() and not resume and not overwrite:
        raise FileExistsError(f"checkpoint exists; use --resume or --overwrite: {output}")
    if overwrite and not resume and log_path.exists():
        log_path.unlink()

    run_seed = stable_seed(int(config["seed"]), "mtrd", task, device_model)
    set_deterministic(run_seed)
    student = load_model_strict(initial_path, task, device)
    optimizer = optimizer_for(
        config, task, student, student=True, device_model=device_model,
    )
    protocol = config["protocol"]
    balancer = (
        build_dncnn_balancer(dncnn_contract)
        if dncnn_contract is not None else EpochDeltaBalancer(
            len(specs),
            segmentation_mtrd_contract.eq6_direction,
            segmentation_mtrd_contract.eq6_temperature,
        )
    )
    start_epoch = 0
    best = float("-inf")
    final_metric = float("-inf")
    history: list[dict[str, object]] = []
    resume_identity = {
        "config_sha256": canonical_json_hash(config),
        "environment_identity_sha256": canonical_json_hash(environment_identity()),
        "first_party_source_sha256": source_tree_sha256(CODE_ROOT),
        "dataset_identity_sha256": canonical_json_hash(dataset_identity),
        "feedback_identity_sha256": canonical_json_hash(feedback_identity),
        "teacher_sha256": [sha256_file(path) for path in teacher_paths],
        "initial_checkpoint_sha256": sha256_file(initial_path),
        "resolved_device": str(device),
        "dncnn_mtrd_contract_sha256": (
            canonical_json_hash(dncnn_contract_payload)
            if dncnn_contract_payload is not None else None
        ),
        "segmentation_metric_contract_sha256": (
            canonical_json_hash(segmentation_metric_contract)
            if segmentation_metric_contract is not None else None
        ),
        "segmentation_mtrd_bn_update_policy_sha256": (
            canonical_json_hash(semantic_contract_value(segmentation_bn_update_policy))
            if segmentation_bn_update_policy is not None else None
        ),
        "segmentation_mtrd_contract_sha256": (
            canonical_json_hash(segmentation_mtrd_payload)
            if segmentation_mtrd_payload is not None else None
        ),
    }
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state is missing: {state_path}")
        state = load_checkpoint(state_path, map_location=device)
        if state.get("resume_identity") != resume_identity:
            raise ValueError(
                "resume identity mismatch: configuration, data, feedback split, "
                "teacher pool, initialization checkpoint, environment, source, "
                "or device changed"
            )
        student.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        balancer.load_state_dict(state["balancer_state"])
        history = list(state["performance_history"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best_metric"])
        final_metric = float(state.get("last_metric", best))
    elif source_bh4:
        history.append({
            "epoch": 0,
            "performance": [0.0] * len(specs),
            "beta": balancer.beta.tolist(),
            "baseline": "released_bh4_zero_psnr_before_epoch_1",
        })
    else:
        initial_performance = [
            feedback_metric(
                student, feedback, task=task, device_model=device_model,
                level=spec.level, config=config, device=device,
                seed_namespace=f"mtrd-feedback-{spec.label}",
                dncnn_contract=dncnn_contract,
            )
            for spec in specs
        ]
        balancer.update(initial_performance)
        history.append({
            "epoch": 0,
            "performance": initial_performance,
            "beta": balancer.beta.tolist(),
        })

    settings = protocol[task]
    epochs = (
        dncnn_contract.student_epochs
        if dncnn_contract is not None else segmentation_mtrd_contract.student_epochs
    )
    batch_size = (
        int(settings["batch_size"])
        if dncnn_contract is not None else segmentation_mtrd_contract.training_batch_size
    )
    workers = int(settings["num_workers"])
    alpha = (
        dncnn_contract.eq4_alpha
        if dncnn_contract is not None else segmentation_mtrd_contract.eq4_alpha
    )
    temperature = (
        dncnn_contract.distillation_temperature
        if dncnn_contract is not None else segmentation_mtrd_contract.distillation_temperature
    )
    for epoch in range(start_epoch, epochs):
        lr = learning_rate(
            config, task, epoch, student=True, device_model=device_model,
        )
        set_optimizer_lr(optimizer, lr)
        loader = epoch_loader(
            train_dataset, batch_size=batch_size, workers=workers,
            seed=run_seed, epoch=epoch, shuffle=True,
        )
        student.train()
        beta_used = balancer.beta.clone()
        loss_total = kd_total = task_total = 0.0
        item_total = 0
        for batch_index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            if task == "denoising":
                clean, sample_ids = batch
                input_noise, _ = gaussian_noise_batch(
                    clean, sample_ids,
                    std=float(settings["image_noise_std_255"]) / 255.0,
                    base_seed=run_seed, epoch=epoch, namespace="mtrd-input-noise",
                )
                inputs = (clean + input_noise).to(device)
                targets = input_noise.to(device)
                count = int(clean.size(0))
            else:
                images, masks, _sample_ids = batch
                inputs = images.to(device)
                targets = masks.unsqueeze(1).to(device)
                count = int(images.size(0))

            if source_bh4:
                # The parser defaults to zero, but the released README invokes
                # BH4 with --w_noiseL equal to the intermediate .3/.06 level.
                noisy_output = equation_forward(
                    student, inputs, device_model, initial_level,
                    stable_seed(run_seed, epoch, batch_index, "student-source-bh4-forward"),
                )
                nominal_output = None
            else:
                if task == "segmentation":
                    if segmentation_bn_update_policy is None:
                        raise AssertionError("UNet MTRD BatchNorm policy was not resolved")
                    nominal_output = segmentation_nominal_kd_forward(
                        student,
                        inputs,
                        bn_update_policy=segmentation_bn_update_policy,
                    )
                else:
                    nominal_output = student(inputs, 0.0, "none")
                noisy_output = equation_forward(
                    student, inputs, device_model, initial_level,
                    stable_seed(run_seed, epoch, batch_index, "student-task-forward"),
                )
            teacher_outputs = []
            with torch.no_grad():
                for teacher_index, (teacher, spec) in enumerate(zip(teachers, specs)):
                    if spec.clean:
                        teacher_output = teacher(inputs, 0.0, "none")
                    else:
                        teacher_output = equation_forward(
                            teacher, inputs, device_model, spec.level,
                            stable_seed(
                                run_seed, epoch, batch_index, teacher_index,
                                "teacher-kd-forward",
                            ),
                        )
                    teacher_outputs.append(teacher_output)
            if source_bh4:
                loss, pieces = released_bh4_source_loss(
                    noisy_output, targets, teacher_outputs, beta_used,
                    alpha=alpha, temperature=temperature,
                )
            else:
                if nominal_output is None:
                    raise AssertionError("MTRD loss requires a nominal student output")
                loss, pieces = paper_mtrd_loss(
                    task, nominal_output, noisy_output, teacher_outputs, targets,
                    beta_used, alpha=alpha, temperature=temperature,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MTRD loss at epoch={epoch} batch={batch_index}")
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * count
            kd_total += pieces["kd"].item() * count
            task_total += pieces["task"].item() * count
            item_total += count

        performance = [
            feedback_metric(
                student, feedback, task=task, device_model=device_model,
                level=spec.level, config=config, device=device,
                seed_namespace=f"mtrd-feedback-{spec.label}",
                feedback_epoch=epoch + 1,
                dncnn_contract=dncnn_contract,
            )
            for spec in specs
        ]
        robust_values = [value for value, spec in zip(performance, specs) if not spec.clean]
        robust_mean = float(sum(robust_values) / len(robust_values))
        final_metric = robust_mean
        beta_next = balancer.update(performance)
        history.append({
            "epoch": epoch + 1,
            "performance": performance,
            "beta_used": beta_used.tolist(),
            "beta_next": beta_next.tolist(),
        })
        if robust_mean > best:
            best = robust_mean
        atomic_torch_save(
            output,
            checkpoint_payload(
                student, task=task, role="mtrd_student",
                device_model=device_model, level=initial_level,
                epoch=epoch + 1, metric=robust_mean, config=config,
                extra=(
                    {"dncnn_mtrd_contract": dncnn_contract_payload}
                    if dncnn_contract_payload is not None else None
                ),
            ),
        )
        append_csv(
            log_path,
            {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "train_loss": loss_total / max(1, item_total),
                "train_kd_loss": kd_total / max(1, item_total),
                "train_task_loss": task_total / max(1, item_total),
                "robust_mean_metric": robust_mean,
                "best_robust_mean_metric": best,
                "beta_used_json": json.dumps(beta_used.tolist()),
                "beta_next_json": json.dumps(beta_next.tolist()),
                "performance_json": json.dumps(performance),
            },
            STUDENT_LOG_FIELDS,
        )
        atomic_torch_save(
            state_path,
            {
                "schema": f"{SCHEMA}.training-state",
                "config_sha256": canonical_json_hash(config),
                "resume_identity": resume_identity,
                "epoch": epoch,
                "best_metric": best,
                "last_metric": final_metric,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "balancer_state": balancer.state_dict(),
                "performance_history": history,
                "dncnn_mtrd_contract": dncnn_contract_payload,
                "segmentation_metric_contract": segmentation_metric_contract,
                "segmentation_mtrd_bn_update_policy": segmentation_bn_update_policy,
                "segmentation_mtrd_contract": segmentation_mtrd_payload,
            },
        )
        print(
            f"{task} MTRD {device_model} epoch={epoch+1}/{epochs} "
            f"loss={loss_total/max(1,item_total):.6f} robust_mean={robust_mean:.6f}"
        )

    if source_bh4:
        eq4_manifest: dict[str, object] = {
            "total": (
                "(KLDiv(log_softmax(student_intermediate_variation/5,dim=1),"
                "softmax(weighted_teacher/5,dim=1))*5^2*2*0.7 + "
                "0.3*MSE_sum(student_intermediate_variation,input_noise))/(2*batch)"
            ),
            "alpha": alpha,
            "denoising_kd": "weighted teacher residual-map mixture followed by one-channel KLDiv",
            "temperature": temperature,
            "loss_reduction": "kl_mean_plus_mse_sum_divided_by_2_batch",
            "kd_is_degenerate": True,
            "protocol_status": (
                "Released BH4 source semantics are recorded because the one-channel "
                "KL term is degenerate; this is not an author-verified objective."
            ),
        }
    else:
        eq4_manifest = {
            "total": "alpha*sum_i(beta_i*KD(nominal_student,teacher_i))+(1-alpha)*task(noisy_student,target)",
            "alpha": alpha,
            "denoising_kd": "beta-weighted sum of per-teacher residual-map MSE losses",
            "segmentation_kd": "beta-weighted sum of per-teacher soft BCE losses",
            "temperature": temperature,
            "protocol_status": (
                "The manuscript does not state the loss reduction and temperature-scaling "
                "details required for an independent numerical acceptance run."
            ),
        }

    manifest = training_manifest(
        output, config=config, task=task, role="mtrd_student",
        device_model=device_model, level=initial_level, metric=final_metric,
        dataset=dataset_identity, log_path=log_path,
        extra={
            "teacher_pool": [
                {
                    "label": spec.label,
                    "level": spec.level,
                    "clean": spec.clean,
                    "checkpoint": str(path.resolve()),
                    "checkpoint_sha256": sha256_file(path),
                }
                for spec, path in zip(specs, teacher_paths)
            ],
            "student_initial_checkpoint": str(initial_path.resolve()),
            "student_initial_checkpoint_sha256": sha256_file(initial_path),
            "eq4": eq4_manifest,
            "eq6": {
                "direction": balancer.direction,
                "equation": balancer.equation,
                "ambiguity": (
                    "printed equation uses current-previous; prose says degraded/minimal improvement "
                    "should receive more weight"
                ),
                "performance_unit": "PSNR dB" if task == "denoising" else "Dice fraction",
                "initial_previous": (
                    [0.0] * len(specs) if source_bh4 else "measured_initial_feedback"
                ),
                "raw_rank_weights": (
                    list(ReleasedBH4RankBalancer.raw_rank_weights)
                    if source_bh4 else None
                ),
            },
            "training_noise_realization": "fresh independent tensor per compute layer and forward",
            "pcm_wmax_definition": (
                "signed maximum weight value max(W) over each logical layer"
            ),
            "performance_history": history,
            "checkpoint_selection": "final_epoch",
            "best_observed_feedback_metric": best,
            "feedback_dataset": feedback_identity,
            "resume_identity": resume_identity,
            "dncnn_mtrd_contract": dncnn_contract_payload,
            "segmentation_mtrd_contract": segmentation_mtrd_payload,
            "student_training_forward": (
                dncnn_contract.student_training_forward
                if dncnn_contract is not None else None
            ),
            "equation_implementation": not source_bh4,
        },
    )
    write_json(output.with_suffix(".manifest.json"), manifest)
    return output


def build_training_plan(
    config: Mapping[str, Any], tasks: Sequence[str], device_models: Sequence[str], stage: str,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    if "denoising" in tasks and stage in {"all", "clean", "teachers", "mtrd"}:
        h5_input = resolve_denoising_h5_input(config)
        plan.append({
            "action": (
                "prepare_denoising_h5"
                if h5_input["mode"] == "regenerated-from-raw"
                else "validate_source_provided_denoising_h5"
            ),
            "mode": h5_input["mode"],
            "input": h5_input,
            "output": (
                h5_input["directory"]
                if h5_input["mode"] == "regenerated-from-raw"
                else None
            ),
        })
    if stage in {"all", "clean"}:
        for task in tasks:
            dncnn_contract = (
                resolve_dncnn_mtrd_contract(config, device_models[0])
                if task == "denoising" else None
            )
            entry: dict[str, object] = {
                "action": (
                    "train_clean_nominal_reference"
                    if dncnn_contract is not None
                    and not dncnn_contract.include_clean_teacher
                    else "train_clean_teacher"
                ),
                "task": task,
                "epochs": (
                    dncnn_contract.teacher_epochs
                    if dncnn_contract is not None
                    else int(config["protocol"][task]["teacher_epochs"])
                ),
                "output": str(checkpoint_path(config, task, "clean")),
                "dncnn_mtrd_contract": (
                    dncnn_mtrd_contract_payload(dncnn_contract)
                    if dncnn_contract is not None else None
                ),
            }
            if task == "segmentation":
                entry["optimizer_contract"] = unet_optimizer_contract(config, student=False)
                entry["segmentation_metric_contract"] = (
                    resolve_segmentation_metric_contract(config)
                )
            plan.append(entry)
    if stage in {"all", "teachers"}:
        for task in tasks:
            for device_model in device_models:
                dncnn_contract = (
                    resolve_dncnn_mtrd_contract(config, device_model)
                    if task == "denoising" else None
                )
                for value in levels(config, device_model):
                    entry = {
                        "action": "train_variation_teacher",
                        "task": task,
                        "device_model": device_model,
                        "level": value,
                        "epochs": (
                            dncnn_contract.teacher_epochs
                            if dncnn_contract is not None
                            else int(config["protocol"][task]["teacher_epochs"])
                        ),
                        "output": str(checkpoint_path(config, task, "teacher", device_model, value)),
                        "dncnn_mtrd_contract": (
                            dncnn_mtrd_contract_payload(dncnn_contract)
                            if dncnn_contract is not None else None
                        ),
                    }
                    if task == "segmentation":
                        entry["optimizer_contract"] = unet_optimizer_contract(
                            config, student=False,
                        )
                        entry["segmentation_metric_contract"] = (
                            resolve_segmentation_metric_contract(config)
                        )
                    plan.append(entry)
    if stage in {"all", "mtrd"}:
        for task in tasks:
            for device_model in device_models:
                dncnn_contract = (
                    resolve_dncnn_mtrd_contract(config, device_model)
                    if task == "denoising" else None
                )
                segmentation_contract = (
                    resolve_segmentation_mtrd_contract(config, device_model)
                    if task == "segmentation" else None
                )
                entry: dict[str, object] = {
                    "action": "train_mtrd_student",
                    "task": task,
                    "device_model": device_model,
                    "level": (
                        dncnn_contract.student_initial_level
                        if dncnn_contract is not None else segmentation_contract.student_initial_level
                    ),
                    "epochs": (
                        dncnn_contract.student_epochs
                        if dncnn_contract is not None
                        else segmentation_contract.student_epochs
                    ),
                    "output": str(checkpoint_path(config, task, "mtrd", device_model)),
                    "dncnn_mtrd_contract": (
                        dncnn_mtrd_contract_payload(dncnn_contract)
                        if dncnn_contract is not None else None
                    ),
                }
                if task == "segmentation":
                    entry["optimizer_contract"] = unet_optimizer_contract(
                        config, student=True,
                    )
                    entry["segmentation_metric_contract"] = (
                        resolve_segmentation_metric_contract(config)
                    )
                    entry["segmentation_mtrd_bn_update_policy"] = (
                        resolve_segmentation_mtrd_bn_update_policy(config)
                    )
                    entry["segmentation_mtrd_contract"] = segmentation_mtrd_contract_payload(
                        segmentation_contract
                    )
                plan.append(entry)
    return plan


def run_training(
    config: Mapping[str, Any], *, tasks: Sequence[str], device_models: Sequence[str],
    stage: str, torch_device: str, resume: bool, overwrite: bool,
    overwrite_h5: bool, dry_run: bool,
) -> list[Path]:
    plan = build_training_plan(config, tasks, device_models, stage)
    output_root = Path(config["output_root"])
    write_json(output_root / "training_plan.json", {
        "schema": f"{SCHEMA}.training-plan",
        "created_utc": utc_now(),
        "config_sha256": canonical_json_hash(config),
        "plan": plan,
    })
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return []
    if "denoising" in tasks:
        prepare_denoising_h5(config, overwrite=overwrite_h5)
    report = run_preflight(
        config, scope="train", tasks=tasks, device_models=device_models,
        stage=stage, write_report=True,
    )
    if report["status"] != "pass":
        raise PreflightError(
            "training preflight failed:\n" + "\n".join(report["errors"])
        )
    device = torch.device(torch_device)
    outputs: list[Path] = []
    if stage in {"all", "clean"}:
        for task in tasks:
            outputs.append(train_one_teacher(
                config, task=task, device_model=device_models[0],
                spec=TeacherSpec("clean", 0.0, True), device=device,
                resume=resume, overwrite=overwrite,
            ))
    if stage in {"all", "teachers"}:
        for task in tasks:
            for device_model in device_models:
                for value in levels(config, device_model):
                    outputs.append(train_one_teacher(
                        config, task=task, device_model=device_model,
                        spec=TeacherSpec(f"{value:g}", value, False),
                        device=device, resume=resume, overwrite=overwrite,
                    ))
    if stage in {"all", "mtrd"}:
        for task in tasks:
            for device_model in device_models:
                outputs.append(train_mtrd_student(
                    config, task=task, device_model=device_model, device=device,
                    resume=resume, overwrite=overwrite,
                ))
    checkpoint_role_manifest = None
    role_paths: dict[str, Path] = {}
    for available_task in tasks:
        for available_device in device_models:
            nominal = evaluation_checkpoint(
                config, available_task, available_device, "nominal"
            )
            mtrd = evaluation_checkpoint(
                config, available_task, available_device, "mtrd"
            )
            if (
                nominal.is_file() and nominal.stat().st_size > 0
                and mtrd.is_file() and mtrd.stat().st_size > 0
            ):
                role_paths[
                    f"image.{available_task}.{available_device}.nominal"
                ] = nominal
                role_paths[
                    f"image.{available_task}.{available_device}.mtrd"
                ] = mtrd
    if role_paths:
        from utils.checkpoint_roles import write_checkpoint_roles

        role_path = output_root / "checkpoint-roles.generated.json"
        checkpoint_role_manifest = write_checkpoint_roles(
            role_path,
            role_paths,
            group_id=f"image-workflows-config-{canonical_json_hash(config)[:16]}",
            role_assignments_author_verified=False,
        )
    training_run_manifest: dict[str, object] = {
        "schema": f"{SCHEMA}.training-run",
        "created_utc": utc_now(),
        "config_sha256": canonical_json_hash(config),
        "outputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in outputs
        ],
        "environment": environment_identity(),
        "source": git_identity(CODE_ROOT),
        "checkpoint_role_manifest": checkpoint_role_manifest,
        "dncnn_mtrd_contracts": (
            {
                device_model: dncnn_mtrd_contract_payload(
                    resolve_dncnn_mtrd_contract(config, device_model)
                )
                for device_model in device_models
            }
            if "denoising" in tasks else {}
        ),
    }
    if "denoising" in tasks:
        training_run_manifest["denoising_h5"] = denoising_h5_identity(config)
    if "segmentation" in tasks:
        training_run_manifest["segmentation_optimizer_contract"] = unet_optimizer_contract(
            config,
        )
        training_run_manifest["segmentation_metric_contract"] = (
            resolve_segmentation_metric_contract(config)
        )
        training_run_manifest["segmentation_mtrd_bn_update_policy"] = (
            resolve_segmentation_mtrd_bn_update_policy(config)
        )
        training_run_manifest["segmentation_mtrd_contracts"] = {
            device_model: segmentation_mtrd_contract_payload(
                resolve_segmentation_mtrd_contract(config, device_model)
            )
            for device_model in device_models
        }
    write_json(output_root / "training_run_manifest.json", training_run_manifest)
    return outputs


RAW_FIELDS = (
    "task", "dataset", "metric", "method", "device_model", "backend",
    "realization_scope",
    "noise_symbol", "noise_level", "trial", "trial_seed",
    "sample_id", "input_noise_seed", "checkpoint_path", "checkpoint_sha256",
    "weight_bits", "dac_bits", "adc_bits", "metric_value",
)
TRIAL_FIELDS = (
    "task", "dataset", "metric", "method", "device_model", "backend",
    "realization_scope", "noise_level", "trial",
    "trial_seed", "observation_count", "metric_mean",
    "metric_std_across_observations",
)
SUMMARY_FIELDS = (
    "task", "dataset", "metric", "method", "device_model", "backend",
    "realization_scope", "noise_level", "trial_count",
    "metric_mean_across_trials", "metric_std_across_trials", "metric_min_trial",
    "metric_max_trial",
)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot aggregate an empty metric list")
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(values) > 1 else 0.0


def backend_identity(
    config: Mapping[str, Any], *, task: str, device_model: str, backend: str,
    realization_scope: str, runtime_validations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    simulation = config["simulation"]
    if device_model == "rram" and backend == "neurosim":
        import simulators.neurosim_functional as neurosim_functional
        from simulators.neurosim import require_functional_adapter
        from simulators.neurosim_functional import PAPER_RRAM_EQUATION, PROFILES

        profile_id = configured_neurosim_profile(config)
        profile = PROFILES[profile_id]
        ptq = configured_neurosim_ptq(config)
        ptq_enabled = ptq["mode"] == "released-eager-static"
        ptq_suffix = (
            f"_released-ptq{ptq['weight_bits']}"
            if ptq_enabled else ""
        )
        status = require_functional_adapter(
            configured_neurosim_root(config), "dncnn" if task == "denoising" else "unet",
        )
        neurosim_root = configured_neurosim_root(config)
        source_tree_matches = bool(
            status["revision_matches"] and status["source_tree_matches"]
        )
        return {
            "name": (
                f"neurosim_source_gated_pytorch_eq1_"
                f"{profile_id}{ptq_suffix}_{realization_scope}"
            ),
            "simulator_family": "DNN+NeuroSim-2DInferenceV1.5-dev",
            "neurosim_source_gate_verified": source_tree_matches,
            "realization_scope": realization_scope,
            "scope_author_verified": simulation.get(
                "rram_realization_scope_author_verified"
            ) is True,
            "scope_warning": "the manuscript does not state the realization lifetime",
            "neurosim_revision": status["revision"],
            "neurosim_source_tree_sha256": status["source_tree_sha256"],
            "functional_adapter": status["public_functional_adapter"],
            "functional_adapter_source_sha256": sha256_file(
                Path(neurosim_functional.__file__)
            ),
            "functional_execution_engine": (
                "source-gated PyTorch Eq. (1) extension followed by released eager "
                "static post-training quantization"
                if ptq_enabled
                else (
                    "source-gated PyTorch functional extension using capsule-defined "
                    "signed symmetric fake quantization"
                    if profile.quantize_weights or profile.quantize_activations
                    else "source-gated PyTorch Eq. (1) extension using the released "
                    "Conv2d/Linear perturbation boundary without quantization"
                )
            ),
            "neurosim_source_gate_only": True,
            "upstream_native_cim_array_kernel_used": False,
            "neurosim_quantize_py_sha256": sha256_file(neurosim_root / "quantize.py"),
            "neurosim_macro_py_sha256": sha256_file(
                neurosim_root
                / "pytorch-quantization"
                / "pytorch_quantization"
                / "cim"
                / "modules"
                / "macro.py"
            ),
            "functional_profile": asdict(profile),
            "post_training_quantization": ptq,
            "quantization_policy_author_verified": profile.profile_author_verified,
            "rram_equation": PAPER_RRAM_EQUATION,
            "weight_bits": (
                int(ptq["weight_bits"])
                if ptq_enabled
                else int(simulation["weight_bits"])
                if profile.quantize_weights
                else None
            ),
            "dac_bits": (
                int(ptq["activation_bits"])
                if ptq_enabled
                else int(simulation["dac_bits"])
                if profile.quantize_activations
                else None
            ),
            "adc_bits": (
                int(ptq["activation_bits"])
                if ptq_enabled
                else int(simulation["adc_bits"])
                if profile.quantize_activations
                else None
            ),
            "programming_noise": True,
            "runtime_validations": list(runtime_validations),
            "operator_coverage": (
                "Conv2d, ConvTranspose2d, and Linear are functionally mapped; "
                "normalization, activation, pooling, and resize remain digital"
                if profile.complete_weighted_operator_coverage
                else (
                    "all DnCNN Conv2d operators are functionally mapped; normalization "
                    "and activation remain digital"
                    if task == "denoising"
                    else "released code maps Conv2d and Linear only; ConvTranspose2d "
                    "remains digital"
                )
            ),
            "activation_quantization": (
                "released unsigned per-tensor MinMaxObserver eager static PTQ"
                if ptq_enabled
                else "signed dynamic per-tensor fake quantization at every mapped "
                "DAC input and ADC output"
                if profile.quantize_activations
                else None
            ),
        }
    if device_model != "pcm" or backend != "aihwkit-additive":
        raise RuntimeError(f"unsupported functional backend: {device_model}/{backend}")
    if task == "segmentation":
        raise RuntimeError(
            "UNet contains ConvTranspose2d operators that AIHWKit 1.1 cannot convert"
        )
    return {
        "name": f"aihwkit_additive_constant_{realization_scope}",
        "realization_scope": realization_scope,
        "scope_author_verified": simulation.get("pcm_realization_scope_author_verified") is True,
        "scope_warning": "the manuscript does not state per-MAC versus fixed-trial scope",
        "aihwkit_version": package_version("aihwkit"),
        "required_aihwkit_version": simulation.get("aihwkit_required_version", AIHWKIT_PIN),
        "weight_mapping": (
            "AIHWKit internally maps layerwise abs-max to 1.0; per-layer w_noise "
            "uses max(W)/max(abs(W)) compensation"
            if realization_scope == "per_mac"
            else "AIHWKit internally maps layerwise abs-max to 1.0; the custom "
            "programming model samples using max(W) in mapped coordinates"
        ),
        "mapping_max_input_size": 0,
        "mapping_max_output_size": 0,
        "weight_noise_type": (
            "WeightNoiseType.ADDITIVE_CONSTANT"
            if realization_scope == "per_mac" else "WeightNoiseType.NONE"
        ),
        "w_noise": (
            "eta*max(W)/max(abs(W)) per logical layer"
            if realization_scope == "per_mac" else 0.0
        ),
        "pcm_wmax_definition": "signed maximum weight value over each logical layer",
        "weight_bits": int(simulation["weight_bits"]),
        "dac_bits": int(simulation["dac_bits"]),
        "adc_bits": int(simulation["adc_bits"]),
        "programming_noise": realization_scope == "fixed_trial",
        "read_drift_model": False,
        "runtime_validations": list(runtime_validations),
        "operator_coverage": (
            "all DnCNN Conv2d operators use AIHWKit tiles; BatchNorm remains digital"
        ),
    }


def run_noise_scale_validation(
    config: Mapping[str, Any], *, backend: str, realization_scope: str,
) -> list[dict[str, object]]:
    if backend == "neurosim":
        return []
    if backend != "aihwkit-additive":
        raise ValueError("noise-scale validation is only available for AIHWKit")
    simulation = config["simulation"]
    samples = int(simulation.get("runtime_validation_samples", 4096))
    tolerance = float(simulation.get("runtime_validation_relative_std_tolerance", 0.10))
    results: list[dict[str, object]] = []
    from simulators.aihwkit import validate_noise_scale

    shared_scope = "per-mac" if realization_scope == "per_mac" else "fixed-trial"
    for eta in levels(config, "pcm"):
        result = validate_noise_scale(
            eta,
            samples=samples,
            seed=stable_seed(int(config["seed"]), "aihwkit-runtime-validation", eta),
            realization_scope=shared_scope,
        )
        result = dict(result)
        result["configured_tolerance"] = tolerance
        result["passed"] = (
            bool(result.get("passed"))
            and float(result["relative_std_error"]) <= tolerance
        )
        results.append(result)
    failed = [result for result in results if not result.get("passed")]
    if failed:
        raise RuntimeError(
            "AIHWKit noise-scale validation failed; refusing to emit metrics:\n"
            + json.dumps(failed, indent=2, sort_keys=True)
        )
    return results


def static_ptq_calibration_batches(
    task: str,
    validation: Dataset,
    *,
    config: Mapping[str, Any],
    trial_seed: int,
) -> Iterable[torch.Tensor]:
    """Yield the evaluation-set calibration inputs used by released scripts."""
    if task == "denoising":
        if not isinstance(validation, Set12Dataset):
            raise TypeError("DnCNN PTQ calibration requires Set12Dataset")
        std = float(config["protocol"]["denoising"]["image_noise_std_255"]) / 255.0
        for sample_index in range(len(validation)):
            clean, sample_id = validation[sample_index]
            input_noise, _ = gaussian_noise_batch(
                clean.unsqueeze(0),
                [sample_id],
                std=std,
                base_seed=trial_seed,
                epoch=0,
                namespace="set12-static-ptq-calibration",
            )
            yield clean.unsqueeze(0) + input_noise
        return
    if task != "segmentation":
        raise ValueError(f"unsupported static PTQ task: {task}")
    settings = config["protocol"]["segmentation"]
    loader = epoch_loader(
        validation,
        batch_size=int(settings["batch_size"]),
        workers=int(settings["num_workers"]),
        seed=int(config["seed"]),
        epoch=0,
        shuffle=False,
    )
    for images, _masks, _sample_ids in loader:
        yield images


def apply_configured_neurosim_ptq(
    model: nn.Module,
    task: str,
    validation: Dataset,
    *,
    config: Mapping[str, Any],
    trial_seed: int,
) -> tuple[nn.Module, dict[str, object] | None]:
    settings = configured_neurosim_ptq(config)
    if settings["mode"] == "none":
        return model, None
    if (
        task == "segmentation"
        and configured_carvana_resize_backend(config) != "released-opencv"
    ):
        raise ValueError(
            "released UNet static PTQ requires resize_backend=released-opencv"
        )
    from simulators.static_ptq import static_quantize_calibrated

    converted, manifest = static_quantize_calibrated(
        model,
        static_ptq_calibration_batches(
            task,
            validation,
            config=config,
            trial_seed=trial_seed,
        ),
        activation_bits=int(settings["activation_bits"]),
        weight_bits=int(settings["weight_bits"]),
        engine=str(settings["engine"]),
    )
    return converted, manifest


@torch.no_grad()
def evaluate_denoising_model(
    model: nn.Module, validation: Set12Dataset, *, backend: str,
    backend_info: Mapping[str, object], device_model: str, level: float,
    trial: int, trial_seed: int, config: Mapping[str, Any],
    method: str, checkpoint: Path, checkpoint_hash: str,
    execution_device: torch.device | None = None,
) -> list[dict[str, object]]:
    execution_device = torch.device("cpu") if execution_device is None else execution_device
    rows: list[dict[str, object]] = []
    settings = config["protocol"]["denoising"]
    std = float(settings["image_noise_std_255"]) / 255.0
    input_noise_scope = str(
        settings.get("evaluation_input_noise", "fixed_dataset")
    )
    if input_noise_scope == "fixed_dataset":
        input_noise_base_seed = int(config["seed"])
        input_noise_namespace = "set12-fixed-input-noise"
    elif input_noise_scope == "released_per_trial":
        input_noise_base_seed = trial_seed
        input_noise_namespace = "set12-released-evaluation-input-noise"
    else:
        raise ValueError(
            "protocol.denoising.evaluation_input_noise must be fixed_dataset "
            "or released_per_trial"
        )
    scope = str(backend_info["realization_scope"])
    for sample_index in range(len(validation)):
        clean, sample_id = validation[sample_index]
        input_noise, input_seeds = gaussian_noise_batch(
            clean.unsqueeze(0), [sample_id], std=std,
            base_seed=input_noise_base_seed, epoch=0,
            namespace=input_noise_namespace,
        )
        noisy = (clean.unsqueeze(0) + input_noise).to(execution_device)
        if backend not in {"neurosim", "aihwkit-additive"}:
            raise RuntimeError(f"unsupported DnCNN functional backend: {backend}")
        residual = model(noisy, 0.0, "none")
        prediction = (noisy - residual).clamp(0.0, 1.0)
        metric = psnr_value(prediction, clean.unsqueeze(0).to(prediction.device))
        rows.append({
            "task": "denoising",
            "dataset": "Set12",
            "metric": "PSNR_dB",
            "method": method,
            "device_model": device_model,
            "backend": backend_info["name"],
            "realization_scope": scope,
            "noise_symbol": "sigma" if device_model == "rram" else "eta",
            "noise_level": level,
            "trial": trial,
            "trial_seed": trial_seed,
            "sample_id": sample_id,
            "input_noise_seed": input_seeds[0],
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "weight_bits": backend_info.get("weight_bits"),
            "dac_bits": backend_info.get("dac_bits"),
            "adc_bits": backend_info.get("adc_bits"),
            "metric_value": metric,
        })
    return rows


@torch.no_grad()
def evaluate_segmentation_model(
    model: nn.Module, validation: CarvanaSplitDataset, *, backend: str,
    backend_info: Mapping[str, object], device_model: str, level: float,
    trial: int, trial_seed: int, config: Mapping[str, Any],
    method: str, checkpoint: Path, checkpoint_hash: str,
    execution_device: torch.device | None = None,
) -> list[dict[str, object]]:
    execution_device = torch.device("cpu") if execution_device is None else execution_device
    settings = config["protocol"]["segmentation"]
    loader = epoch_loader(
        validation, batch_size=int(settings["batch_size"]),
        workers=int(settings["num_workers"]), seed=int(config["seed"]),
        epoch=0, shuffle=False,
    )
    rows: list[dict[str, object]] = []
    scope = str(backend_info["realization_scope"])
    metric_contract = resolve_segmentation_metric_contract(config)
    for batch_index, (images, masks, sample_ids) in enumerate(loader):
        if backend not in {"neurosim", "aihwkit-additive"}:
            raise RuntimeError(f"unsupported UNet functional backend: {backend}")
        logits = model(images.to(execution_device), 0.0, "none")
        targets = masks.unsqueeze(1).to(execution_device)
        metric_records = segmentation_metric_observations(
            logits,
            targets,
            sample_ids,
            metric_contract=metric_contract,
            batch_index=batch_index,
        )
        for sample_id, metric, metric_name in metric_records:
            rows.append({
                "task": "segmentation",
                "dataset": "Carvana",
                "metric": metric_name,
                "method": method,
                "device_model": device_model,
                "backend": backend_info["name"],
                "realization_scope": scope,
                "noise_symbol": "sigma" if device_model == "rram" else "eta",
                "noise_level": level,
                "trial": trial,
                "trial_seed": trial_seed,
                "sample_id": sample_id,
                "input_noise_seed": "",
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "weight_bits": backend_info.get("weight_bits"),
                "dac_bits": backend_info.get("dac_bits"),
                "adc_bits": backend_info.get("adc_bits"),
                "metric_value": metric,
            })
    return rows


def aggregate_rows(
    raw_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trial_groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    trial_metadata: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in raw_rows:
        key = (
            row["task"], row["method"], row["device_model"], row["backend"],
            row["realization_scope"], float(row["noise_level"]), int(row["trial"]),
        )
        trial_groups[key].append(float(row["metric_value"]))
        trial_metadata[key] = row
    trial_rows: list[dict[str, object]] = []
    for key in sorted(trial_groups, key=str):
        values = trial_groups[key]
        mean, std = _mean_std(values)
        row = trial_metadata[key]
        trial_rows.append({
            "task": row["task"],
            "dataset": row["dataset"],
            "metric": row["metric"],
            "method": row["method"],
            "device_model": row["device_model"],
            "backend": row["backend"],
            "realization_scope": row["realization_scope"],
            "noise_level": row["noise_level"],
            "trial": row["trial"],
            "trial_seed": row["trial_seed"],
            "observation_count": len(values),
            "metric_mean": mean,
            "metric_std_across_observations": std,
        })
    summary_groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    summary_metadata: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in trial_rows:
        key = (
            row["task"], row["method"], row["device_model"], row["backend"],
            row["realization_scope"], float(row["noise_level"]),
        )
        summary_groups[key].append(float(row["metric_mean"]))
        summary_metadata[key] = row
    summary_rows: list[dict[str, object]] = []
    for key in sorted(summary_groups, key=str):
        values = summary_groups[key]
        mean, std = _mean_std(values)
        row = summary_metadata[key]
        summary_rows.append({
            "task": row["task"],
            "dataset": row["dataset"],
            "metric": row["metric"],
            "method": row["method"],
            "device_model": row["device_model"],
            "backend": row["backend"],
            "realization_scope": row["realization_scope"],
            "noise_level": row["noise_level"],
            "trial_count": len(values),
            "metric_mean_across_trials": mean,
            "metric_std_across_trials": std,
            "metric_min_trial": min(values),
            "metric_max_trial": max(values),
        })
    return trial_rows, summary_rows


def resolve_evaluation_trial_selection(
    config: Mapping[str, Any], *, trials: int | None, trial_index: int | None,
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, object]]:
    """Resolve either a complete trial range or one explicitly replayed trial.

    A selected trial is retained as a reproducible demonstration path. It is
    intentionally marked as a single-trial result instead of being treated as
    a substitute for the configured multi-trial summary.
    """
    if trials is not None and trial_index is not None:
        raise ValueError("--trials and --trial-index cannot be used together")
    resolved = copy.deepcopy(config)
    simulation = resolved.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be an object")
    configured_count = int(simulation.get("trial_count", 0))
    if configured_count <= 0:
        raise ValueError("simulation.trial_count must be a positive explicit integer")
    if trials is not None:
        if trials <= 0:
            raise ValueError("--trials must be positive")
        simulation["trial_count"] = int(trials)
        simulation["trial_count_author_verified"] = False
        return (
            resolved,
            tuple(range(int(trials))),
            {
                "policy": "contiguous-prefix",
                "configured_trial_count": int(trials),
                "executed_trial_indices": list(range(int(trials))),
                "single_trial_exemplar": False,
            },
        )
    if trial_index is not None:
        if trial_index < 0:
            raise ValueError("--trial-index must be zero or greater")
        simulation["trial_count_author_verified"] = False
        return (
            resolved,
            (int(trial_index),),
            {
                "policy": "explicit-single-trial",
                "configured_trial_count": configured_count,
                "executed_trial_indices": [int(trial_index)],
                "single_trial_exemplar": True,
                "reporting_constraint": (
                    "This is a reproducible selected-trial demonstration, not a "
                    "multi-trial aggregate or an exact-reference reproduction claim."
                ),
            },
        )
    return (
        resolved,
        tuple(range(configured_count)),
        {
            "policy": "configured-contiguous-range",
            "configured_trial_count": configured_count,
            "executed_trial_indices": list(range(configured_count)),
            "single_trial_exemplar": False,
        },
    )


def run_evaluation(
    config: Mapping[str, Any], *, tasks: Sequence[str], device_models: Sequence[str],
    backend_override: str | None, realization_scope_override: str | None,
    trials: int | None, trial_index: int | None, overwrite: bool,
    torch_device: str = "cpu",
) -> tuple[Path, Path, Path]:
    config, trial_indices, trial_selection = resolve_evaluation_trial_selection(
        config, trials=trials, trial_index=trial_index,
    )
    execution_device = resolve_execution_device(torch_device)
    if execution_device.type == "cuda" and any(
        device_model == "rram"
        and configured_neurosim_ptq(config)["mode"] == "released-eager-static"
        for device_model in device_models
    ):
        raise RuntimeError(
            "released eager static PTQ produces CPU-only quantized operators; "
            "use evaluate --torch-device cpu or select a non-PTQ functional profile"
        )
    output = Path(config["output_root"]) / "evaluation"
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"evaluation output already exists: {output}; use evaluate --overwrite to replace it"
        )
    report = run_preflight(
        config, scope="eval", tasks=tasks, device_models=device_models,
        backend_override=backend_override,
        realization_scope_override=realization_scope_override,
        write_report=True,
    )
    if report["status"] != "pass":
        raise PreflightError(
            "evaluation preflight failed:\n" + "\n".join(report["errors"])
        )
    if output.exists():
        shutil.rmtree(output)
    trial_count = len(trial_indices)
    raw_rows: list[dict[str, object]] = []
    backend_manifests: dict[str, object] = {}
    checkpoint_manifests: dict[str, object] = {}
    dataset_manifests: dict[str, object] = {}
    programming_manifests: dict[str, object] = {}

    task_datasets: dict[str, Dataset] = {}
    for task in tasks:
        _train, validation, identity = build_task_datasets(
            config, task, include_training=False,
        )
        task_datasets[task] = validation
        dataset_manifests[task] = identity

    for device_model in device_models:
        backend = resolve_backend(config, device_model, backend_override)
        scope = resolve_realization_scope(config, device_model, realization_scope_override)
        if backend == "neurosim":
            from simulators.neurosim import require_functional_adapter

            root = configured_neurosim_root(config)
            for selected_task in tasks:
                model_name = "dncnn" if selected_task == "denoising" else "unet"
                require_functional_adapter(root, model_name)
        runtime_validations = run_noise_scale_validation(
            config, backend=backend, realization_scope=scope,
        )
        for task in tasks:
            info = backend_identity(
                config, task=task, device_model=device_model, backend=backend,
                realization_scope=scope, runtime_validations=runtime_validations,
            )
            backend_manifests[f"{task}.{device_model}"] = info
            validation = task_datasets[task]
            for method in ("nominal", "mtrd"):
                path = evaluation_checkpoint(config, task, device_model, method)
                checkpoint_hash = sha256_file(path)
                checkpoint_manifests[f"{task}.{device_model}.{method}"] = {
                    "path": str(path.resolve()),
                    "sha256": checkpoint_hash,
                    "dncnn_mtrd_profile": (
                        report.get("dncnn_checkpoint_profiles", {}).get(
                            f"{task}.{device_model}.{method}"
                        )
                        if task == "denoising" else None
                    ),
                    "segmentation_training_contract": (
                        report.get("segmentation_checkpoint_training_contracts", {}).get(
                            f"{task}.{device_model}.{method}"
                        )
                        if task == "segmentation" else None
                    ),
                }
                base = load_model_strict(path, task, "cpu")
                for value in levels(config, device_model):
                    for trial in trial_indices:
                        trial_seed = stable_seed(
                            int(config["seed"]), LEGACY_EVALUATION_SEED_NAMESPACE, task,
                            device_model, f"{value:g}", trial,
                        )
                        set_deterministic(trial_seed)
                        if backend == "neurosim":
                            from simulators.neurosim_functional import (
                                program_rram_fixed_trial,
                            )

                            model, programming = program_rram_fixed_trial(
                                base,
                                model_name="dncnn" if task == "denoising" else "unet",
                                sigma=value,
                                seed=trial_seed,
                                profile=configured_neurosim_profile(config),
                                weight_bits=int(config["simulation"]["weight_bits"]),
                                dac_bits=int(config["simulation"]["dac_bits"]),
                                adc_bits=int(config["simulation"]["adc_bits"]),
                                preserve_standard_modules=(
                                    configured_neurosim_ptq(config)["mode"]
                                    == "released-eager-static"
                                ),
                            )
                            model, ptq_manifest = apply_configured_neurosim_ptq(
                                model,
                                task,
                                validation,
                                config=config,
                                trial_seed=trial_seed,
                            )
                            programming = dict(programming)
                            programming["post_training_quantization"] = ptq_manifest
                            programming_manifests[
                                f"{task}.{device_model}.{method}.{value:g}.{trial}"
                            ] = programming
                            model = model.to(execution_device)
                        else:
                            model = build_aihwkit_paper_model(
                                base, eta=value,
                                weight_bits=int(config["simulation"]["weight_bits"]),
                                dac_bits=int(config["simulation"]["dac_bits"]),
                                adc_bits=int(config["simulation"]["adc_bits"]),
                                seed=trial_seed, realization_scope=scope,
                                execution_device=execution_device,
                            )
                        model.eval()
                        if task == "denoising":
                            rows = evaluate_denoising_model(
                                model, validation, backend=backend, backend_info=info,
                                device_model=device_model, level=value, trial=trial,
                                trial_seed=trial_seed, config=config, method=method,
                                checkpoint=path, checkpoint_hash=checkpoint_hash,
                                execution_device=execution_device,
                            )
                        else:
                            rows = evaluate_segmentation_model(
                                model, validation, backend=backend, backend_info=info,
                                device_model=device_model, level=value, trial=trial,
                                trial_seed=trial_seed, config=config, method=method,
                                checkpoint=path, checkpoint_hash=checkpoint_hash,
                                execution_device=execution_device,
                            )
                        raw_rows.extend(rows)
                        mean, _ = _mean_std([float(row["metric_value"]) for row in rows])
                        print(
                            f"{task} {device_model} {method} level={value:g} "
                            f"trial={trial} seed={trial_seed} mean={mean:.6f}"
                        )
                        del model
                        if execution_device.type == "cuda":
                            torch.cuda.empty_cache()
                del base

    trial_rows, summary_rows = aggregate_rows(raw_rows)
    artifact_prefix = tasks[0] if len(tasks) == 1 else "image_workflows"
    raw_path = write_csv(
        output / f"{artifact_prefix}_raw_observations.csv", raw_rows, RAW_FIELDS
    )
    trial_path = write_csv(
        output / f"{artifact_prefix}_per_trial.csv", trial_rows, TRIAL_FIELDS
    )
    summary_path = write_csv(
        output / f"{artifact_prefix}_summary.csv", summary_rows, SUMMARY_FIELDS
    )
    manifest = {
        "schema": f"{SCHEMA}.evaluation-manifest",
        "created_utc": utc_now(),
        "task_device_groups": {
            f"{task}.{device_model}": f"{task}-{device_model}"
            for task in tasks
            for device_model in device_models
        },
        "tasks": list(tasks),
        "device_models": list(device_models),
        "noise_grids": {device_model: list(levels(config, device_model)) for device_model in device_models},
        "trials": trial_count,
        "trial_selection": trial_selection,
        "execution": {
            "requested_torch_device": str(torch_device),
            "resolved_torch_device": str(execution_device),
            "cuda_available": torch.cuda.is_available(),
        },
        "base_seed": int(config["seed"]),
        "seed_derivation": (
            "sha256(base_seed, task, device, level, trial); the result seeds "
            "simulator construction and fixed-trial programming"
        ),
        "input_noise": {
            "denoising": {
                "distribution": "Set12 AWGN with configured sigma/255",
                "realization": config["protocol"]["denoising"].get(
                    "evaluation_input_noise", "fixed_dataset"
                ),
                "paired_between_methods": True,
            }
            if "denoising" in tasks
            else None,
            "segmentation": {"distribution": "none"}
            if "segmentation" in tasks
            else None,
        },
        "segmentation_metric_aggregation": (
            config["protocol"]["segmentation"].get(
                "metric_aggregation", "per_image_mean"
            )
            if "segmentation" in tasks
            else None
        ),
        "segmentation_metric_contract": (
            resolve_segmentation_metric_contract(config)
            if "segmentation" in tasks
            else None
        ),
        "carvana_resize_backend": (
            configured_carvana_resize_backend(config)
            if "segmentation" in tasks
            else None
        ),
        "backends": backend_manifests,
        "functional_programming": programming_manifests,
        "checkpoints": checkpoint_manifests,
        "dncnn_mtrd_contracts": (
            {
                device_model: dncnn_mtrd_contract_payload(
                    resolve_dncnn_mtrd_contract(config, device_model)
                )
                for device_model in device_models
            }
            if "denoising" in tasks else {}
        ),
        "segmentation_checkpoint_training_contracts": (
            report.get("segmentation_checkpoint_training_contracts", {})
            if "segmentation" in tasks else {}
        ),
        "segmentation_mtrd_contracts": (
            {
                device_model: segmentation_mtrd_contract_payload(
                    resolve_segmentation_mtrd_contract(config, device_model)
                )
                for device_model in device_models
            }
            if "segmentation" in tasks else {}
        ),
        "checkpoint_role_manifest": report.get("checkpoint_role_manifest"),
        "datasets": dataset_manifests,
        "config": config,
        "config_sha256": canonical_json_hash(config),
        "outputs": {
            "raw_observations": {
                "path": str(raw_path.resolve()),
                "sha256": sha256_file(raw_path),
            },
            "per_trial": {"path": str(trial_path.resolve()), "sha256": sha256_file(trial_path)},
            "summary": {"path": str(summary_path.resolve()), "sha256": sha256_file(summary_path)},
        },
        "environment": environment_identity(),
        "source": git_identity(CODE_ROOT),
        "source_file_sha256": sha256_file(__file__),
        "output_policy": "clean directory created for this evaluation run",
        "no_manual_metric_offset": True,
        "numerical_reproduction_verified": False,
        "author_raw_reference": None,
        "numerical_reproduction_blocker": (
            "author raw per-trial reference data, exact trial count/seeds, and confirmed "
            "device realization scope are unavailable; remaining manuscript ambiguities require "
            "author confirmation"
        ),
    }
    write_json(output / "evaluation_manifest.json", manifest)
    return raw_path, trial_path, summary_path


def _selection(value: str, allowed: Sequence[str]) -> tuple[str, ...]:
    return tuple(allowed) if value == "all" else (value,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Denoising and segmentation training with simulator-backed evaluation"
    )
    parser.add_argument("--config", help="JSON protocol configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--scope", choices=["train", "eval", "all"], default="all")
    preflight.add_argument("--stage", choices=["clean", "teachers", "mtrd", "all"], default="all")
    preflight.add_argument("--task", choices=[*TASKS, "all"], default="all")
    preflight.add_argument("--device-model", choices=[*DEVICE_MODELS, "all"], default="all")
    preflight.add_argument(
        "--backend",
        choices=["auto", "neurosim", "aihwkit-additive"],
        default="auto",
    )
    preflight.add_argument(
        "--realization-scope", choices=["fixed_trial"]
    )

    train = subparsers.add_parser("train")
    train.add_argument("--stage", choices=["clean", "teachers", "mtrd", "all"], default="all")
    train.add_argument("--task", choices=[*TASKS, "all"], default="all")
    train.add_argument("--device-model", choices=[*DEVICE_MODELS, "all"], default="all")
    train.add_argument("--torch-device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--overwrite", action="store_true")
    train.add_argument(
        "--overwrite-h5", action="store_true",
        help="regenerate denoising HDF5 after validating the 400/12 source asset",
    )
    train.add_argument("--dry-run", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--task", choices=[*TASKS, "all"], default="all")
    evaluate.add_argument("--device-model", choices=[*DEVICE_MODELS, "all"], default="all")
    evaluate.add_argument(
        "--backend",
        choices=["auto", "neurosim", "aihwkit-additive"],
        default="auto",
    )
    evaluate.add_argument(
        "--realization-scope", choices=["fixed_trial"]
    )
    evaluate.add_argument(
        "--torch-device",
        default="cpu",
        help="explicit execution device for functional inference; default preserves CPU behavior",
    )
    evaluate.add_argument("--trials", type=int)
    evaluate.add_argument(
        "--trial-index", type=int,
        help=(
            "replay exactly one zero-based fixed trial; the manifest marks it as "
            "a single-trial exemplar rather than a multi-trial aggregate"
        ),
    )
    evaluate.add_argument(
        "--overwrite", action="store_true",
        help="replace the complete evaluation directory after preflight passes",
    )

    return parser


def build_task_parser(task: str) -> argparse.ArgumentParser:
    """Build a task-local parser for the public denoising and segmentation entry points."""
    if task not in TASKS:
        raise ValueError(f"unsupported image workflow task: {task}")
    parser = build_parser()
    task_name = "DnCNN denoising" if task == "denoising" else "UNet segmentation"
    parser.description = f"{task_name} training with simulator-backed evaluation"
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if not isinstance(choices, Mapping):
            continue
        for subparser in choices.values():
            for subaction in subparser._actions:
                if subaction.dest == "task":
                    subaction.choices = (task,)
                    subaction.default = task
                    subaction.help = argparse.SUPPRESS
                elif subaction.dest == "overwrite_h5" and task != "denoising":
                    subaction.help = argparse.SUPPRESS
    return parser


def main(
    argv: Sequence[str] | None = None, *, fixed_task: str | None = None,
) -> int:
    parser = build_task_parser(fixed_task) if fixed_task is not None else build_parser()
    args = parser.parse_args(argv)
    if not args.config:
        raise SystemExit("--config is required for preflight, train, and evaluate")
    config, config_path = load_config(args.config)
    print(f"config={config_path} sha256={canonical_json_hash(config)}")
    tasks = (fixed_task,) if fixed_task is not None else _selection(args.task, TASKS)
    device_models = _selection(args.device_model, DEVICE_MODELS)
    if args.command == "preflight":
        report = run_preflight(
            config, scope=args.scope, tasks=tasks, device_models=device_models,
            backend_override=args.backend, realization_scope_override=args.realization_scope,
            stage=args.stage, write_report=True,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 2
    elif args.command == "train":
        run_training(
            config, tasks=tasks, device_models=device_models, stage=args.stage,
            torch_device=args.torch_device, resume=args.resume,
            overwrite=args.overwrite, overwrite_h5=args.overwrite_h5,
            dry_run=args.dry_run,
        )
        return 0
    elif args.command == "evaluate":
        paths = run_evaluation(
            config, tasks=tasks, device_models=device_models,
            backend_override=args.backend,
            realization_scope_override=args.realization_scope,
            trials=args.trials, trial_index=args.trial_index, overwrite=args.overwrite,
            torch_device=args.torch_device,
        )
        print("\n".join(str(path) for path in paths))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
