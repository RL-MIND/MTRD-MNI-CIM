#!/usr/bin/env python3
"""CLI for VGG16 classification training and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .aihwkit_backend import runtime_probe
from .evaluation import evaluate
from .inference import test_checkpoint as run_checkpoint_test
from .model import PaperVGG16, load_checkpoint_strict

from .protocol import PCM_GRID, RRAM_GRID, teacher_specs
from .repro import (
    CIFAR_NORMALIZATION_PROFILES,
    checkpoint_path,
    checkpoint_normalization_identity,
    cifar_identity,
    environment_identity,
    neurosim_functional_preflight,
    resolve_device,
    sha256_file,
    write_json,
)
from .training import train_student, train_teachers


CODE_ROOT = Path(__file__).resolve().parents[1]
CAPSULE_ROOT = CODE_ROOT.parent
IS_CODE_OCEAN = CODE_ROOT == Path("/code") or (
    Path("/code").is_dir() and Path("/data").is_dir() and CODE_ROOT.parent == Path("/")
)
_mount_data_root = Path(
    os.environ.get(
        "MTRD_DATA_ROOT",
        str(Path("/data") if IS_CODE_OCEAN else CAPSULE_ROOT / "data"),
    )
)
DEFAULT_DATA_ROOT = os.environ.get(
    "MTRD_CIFAR_ROOT", str(_mount_data_root / "datasets" / "cifar")
)
_results_base = Path(
    os.environ.get(
        "MTRD_RESULTS_ROOT",
        str(Path("/results") if IS_CODE_OCEAN else CAPSULE_ROOT / "results"),
    )
)
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "MTRD_CLASSIFICATION_ROOT",
    str(_results_base / "classification" / "cim"),
)
DEFAULT_CHECKPOINT_ROOT = str(Path(DEFAULT_OUTPUT_ROOT) / "checkpoints")


def _dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--normalization-profile",
        choices=CIFAR_NORMALIZATION_PROFILES,
        default="dataset-native",
        help=(
            "use dataset-native statistics by default; released-legacy reproduces "
            "the historical CIFAR-10 statistics on CIFAR-100"
        ),
    )


def _training_args(parser: argparse.ArgumentParser, *, student: bool) -> None:
    _dataset_args(parser)
    parser.add_argument("--device-type", choices=("rram", "pcm"), required=True)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=300 if student else 200)
    parser.add_argument("--lr", type=float, default=0.001 if student else 0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-nonpaper-training",
        action="store_true",
        help="allow epochs/lr other than the manuscript values; manifest remains explicit",
    )
    if student:
        parser.add_argument("--alpha", type=float, default=0.7)
        parser.add_argument("--temperature", type=float, default=5.0)
    else:
        parser.set_defaults(no_clean_teacher=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classification pipeline. Simulator identities are fail-closed: "
            "the source-gated PyTorch RRAM extension is never labeled as the native "
            "NeuroSim CIM kernel."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="audit data, checkpoints, and simulators")
    preflight.add_argument("--stage", choices=("train", "evaluate"), default="train")
    preflight.add_argument("--dataset", choices=("cifar10", "cifar100", "all"), default="cifar10")
    preflight.add_argument("--device-type", choices=("rram", "pcm", "all"), default="all")
    preflight.add_argument("--data-root", default="/data/CIFAR/cifar-10/cifar-10-batches-py")
    preflight.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    preflight.add_argument("--checkpoint-role-manifest")
    preflight.add_argument("--neurosim-home")
    preflight.add_argument("--hash-data", action=argparse.BooleanOptionalAction, default=True)
    preflight.add_argument(
        "--normalization-profile",
        choices=CIFAR_NORMALIZATION_PROFILES,
        default="dataset-native",
    )
    preflight.add_argument(
        "--output", default=str(Path(DEFAULT_OUTPUT_ROOT) / "preflight.json")
    )
    preflight.add_argument("--strict", action="store_true")

    teachers = subparsers.add_parser(
        "train-teachers", help="train clean and all five variation-aware VGG16 teachers"
    )
    _training_args(teachers, student=False)

    student = subparsers.add_parser(
        "train-student", help="train the Eq.(4)/(6) MTRD VGG16 student"
    )
    _training_args(student, student=True)

    checkpoint_test = subparsers.add_parser(
        "test-checkpoint",
        help="strict-load one portable VGG16 weight file and test it on official CIFAR",
    )
    _dataset_args(checkpoint_test)
    checkpoint_test.add_argument(
        "--checkpoint",
        required=True,
        help="trusted PyTorch checkpoint saved by this workflow or a strict-compatible export",
    )
    checkpoint_test.add_argument(
        "--device",
        default="auto",
        help="PyTorch device for the clean digital test, for example cuda or cpu",
    )
    checkpoint_test.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_OUTPUT_ROOT) / "checkpoint-test"),
        help="empty result directory for prediction CSV and provenance manifest",
    )
    checkpoint_test.add_argument(
        "--require-checkpoint-manifest",
        action="store_true",
        help="require an adjacent, hash-valid training manifest with matching preprocessing",
    )
    checkpoint_test.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a non-empty output directory that does not contain the input weight",
    )

    evaluation = subparsers.add_parser(
        "evaluate", help="evaluate nominal and MTRD checkpoints on the configured variation grid"
    )
    _dataset_args(evaluation)
    evaluation.add_argument("--device-type", choices=("rram", "pcm"), required=True)
    evaluation.add_argument(
        "--backend",
        choices=("auto", "neurosim", "aihwkit-additive"),
        default="auto",
    )
    evaluation.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    evaluation.add_argument("--checkpoint-role-manifest", required=True)
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--nominal-checkpoint")
    evaluation.add_argument("--mtrd-checkpoint")
    evaluation.add_argument(
        "--checkpoint-role", choices=("nominal", "mtrd", "both"), default="both"
    )
    evaluation.add_argument(
        "--method-realization-policy",
        choices=("paired", "independent"),
        required=True,
        help="whether nominal and MTRD use the same or independent device realization seeds",
    )
    evaluation.add_argument("--trials", type=int, default=20)
    evaluation.add_argument(
        "--realization-scope",
        choices=("fixed-trial",),
        required=True,
        help="program one seed-replayable device realization for the complete test set",
    )
    evaluation.add_argument("--quantization-bits", type=int, default=8)
    evaluation.add_argument("--allow-nonpaper-quantization", action="store_true")
    evaluation.add_argument("--neurosim-home")
    evaluation.add_argument(
        "--neurosim-functional-profile",
        choices=("paper-8bit", "released-legacy"),
        default="paper-8bit",
        help="explicit RRAM functional mapping and quantization profile",
    )
    evaluation.add_argument("--output-dir", default=str(Path(DEFAULT_OUTPUT_ROOT) / "evaluation"))
    evaluation.add_argument("--overwrite", action="store_true")

    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    datasets = ("cifar10", "cifar100") if args.dataset == "all" else (args.dataset,)
    devices = ("rram", "pcm") if args.device_type == "all" else (args.device_type,)
    data = {
        dataset: cifar_identity(
            dataset,
            Path(args.data_root),
            include_hash=args.hash_data,
            normalization_profile=args.normalization_profile,
        )
        for dataset in datasets
    }
    checkpoints: dict[str, object] = {}
    checkpoint_ready = True
    role_manifest_identity = None
    role_paths: dict[str, Path] = {}
    for dataset in datasets:
        for device_type in devices:
            key = f"{dataset}/{device_type}"
            specs = teacher_specs(device_type, include_clean=True)
            if args.stage == "evaluate":
                expected = [
                    checkpoint_path(
                        Path(args.checkpoint_root), dataset, device_type, role,
                    )
                    for role in ("nominal", "mtrd")
                ]
            else:
                expected = [
                    checkpoint_path(
                        Path(args.checkpoint_root),
                        dataset,
                        device_type,
                        "clean" if spec.clean else "teacher",
                        spec.noise,
                    )
                    for spec in specs
                ]
                expected.append(
                    checkpoint_path(Path(args.checkpoint_root), dataset, device_type, "mtrd")
                )
            entries = []
            for index, path in enumerate(expected):
                present = path.is_file()
                valid = present and path.stat().st_size > 0
                error = None
                normalization_provenance = None
                if args.stage == "evaluate":
                    if valid:
                        try:
                            model = PaperVGG16(10 if dataset == "cifar10" else 100)
                            load_checkpoint_strict(model, str(path), "cpu")
                            normalization_provenance = checkpoint_normalization_identity(
                                path,
                                dataset=dataset,
                                requested_profile=args.normalization_profile,
                            )
                        except Exception as caught:
                            valid = False
                            error = f"{type(caught).__name__}: {caught}"
                    checkpoint_ready &= valid
                    role = ("nominal", "mtrd")[index]
                    role_paths[
                        f"classification.{dataset}.{device_type}.{role}"
                    ] = path
                entries.append(
                    {
                        "path": str(path.resolve()),
                        "present": present,
                        "size_bytes": path.stat().st_size if present else None,
                        "strict_load_valid": valid if args.stage == "evaluate" else None,
                        "error": error,
                        "sha256": sha256_file(path) if valid else None,
                        "normalization_provenance": normalization_provenance,
                    }
                )
            checkpoints[key] = entries

    if args.stage == "evaluate":
        try:
            from utils.checkpoint_roles import validate_checkpoint_roles

            if not args.checkpoint_role_manifest:
                raise ValueError("--checkpoint-role-manifest is required for evaluation")
            role_manifest_identity = validate_checkpoint_roles(
                args.checkpoint_role_manifest, role_paths,
            )
        except Exception as error:
            checkpoint_ready = False
            role_manifest_identity = {
                "valid": False,
                "error": f"{type(error).__name__}: {error}",
            }

    neurosim = neurosim_functional_preflight(args.neurosim_home)
    aihwkit = {
        **runtime_probe(),
        "noise_scale_validation_required_before_evaluation": True,
        "diagnostic_realization_scopes": ["fixed-trial", "per-mac"],
        "formal_evaluation_scopes": ["fixed-trial"],
        "per_mac_forward_rng_replayable": False,
        "scope_must_be_reported": True,
        "author_confirmation_required": True,
    }
    data_ready = all(bool(item["present"]) for item in data.values())
    requested_backend_ready = True
    if args.stage == "evaluate":
        if "rram" in devices:
            requested_backend_ready &= bool(neurosim["functional_ready"])
        if "pcm" in devices:
            requested_backend_ready &= bool(aihwkit["runtime_ready"])
    execution_ready = data_ready and checkpoint_ready and requested_backend_ready
    report = {
        "schema": "mtrd.classification.preflight.v1",
        "stage": args.stage,
        "ready_for_protocol_execution": execution_ready,
        "ready_for_reference_evaluation": False,
        "ready_for_exact_numerical_claim": False,
        "training_code_present": True,
        "testing_code_present": True,
        "checkpoints_required_for_stage": args.stage == "evaluate",
        "simulator_required_for_stage": args.stage == "evaluate",
        "variation_grids": {"rram_sigma": list(RRAM_GRID), "pcm_eta": list(PCM_GRID)},
        "data": data,
        "checkpoints": checkpoints,
        "checkpoint_role_manifest": role_manifest_identity,
        "neurosim": neurosim,
        "aihwkit": aihwkit,
        "environment": environment_identity(),
        "blocking_note": (
            "Training readiness does not require an inference simulator. Exact numerical claims "
            "still require author-confirmed protocol decisions and raw reference trials."
            if args.stage == "train"
            else (
                "The functional adapter and C++ PPA interface are separate. The author must "
                "confirm the RRAM mapping, quantization, realization lifetime, and provide raw "
                "reference values plus an "
                "acceptance tolerance. AIHWKit 1.1.0 per-MAC forward noise is not seed-replayable, "
                "so this capsule permits only fixed-trial formal evaluation."
            )
        ),
    }
    write_json(Path(args.output), report)
    return report, execution_ready


def _validate_training_protocol(args: argparse.Namespace, *, student: bool) -> None:
    expected_epochs = 300 if student else 200
    expected_lr = 0.001 if student else 0.01
    if not args.allow_nonpaper_training and (
        args.epochs != expected_epochs or abs(args.lr - expected_lr) > 1e-15
    ):
        raise ValueError(
            f"the configured training schedule requires epochs={expected_epochs}, lr={expected_lr}; "
            "use --allow-nonpaper-training only for an explicitly recorded alternate run"
        )
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")


def _write_evaluation_checkpoint_roles(args: argparse.Namespace) -> Path:
    from utils.checkpoint_roles import write_checkpoint_roles

    directory = Path(args.checkpoint_root) / args.dataset / args.device_type
    roles = {
        f"classification.{args.dataset}.{args.device_type}.nominal": checkpoint_path(
            Path(args.checkpoint_root), args.dataset, args.device_type, "nominal"
        ),
        f"classification.{args.dataset}.{args.device_type}.mtrd": checkpoint_path(
            Path(args.checkpoint_root), args.dataset, args.device_type, "mtrd"
        ),
    }
    path = directory / "checkpoint-roles.generated.json"
    write_checkpoint_roles(
        path,
        roles,
        group_id=f"classification-{args.dataset}-{args.device_type}-seed-{args.seed}",
        role_assignments_author_verified=False,
    )
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        report, ready = _preflight(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if ready or not args.strict else 2
    if args.command == "train-teachers":
        _validate_training_protocol(args, student=False)
        args.torch_device = resolve_device(args.device)
        paths = train_teachers(args)
        print("\n".join(str(path) for path in paths))
        return 0
    if args.command == "train-student":
        _validate_training_protocol(args, student=True)
        args.torch_device = resolve_device(args.device)
        path = train_student(args)
        print(path)
        print(_write_evaluation_checkpoint_roles(args))
        return 0
    if args.command == "test-checkpoint":
        predictions, manifest = run_checkpoint_test(args)
        print(predictions)
        print(manifest)
        return 0
    if args.command == "evaluate":
        if args.trials <= 0:
            raise ValueError("--trials must be positive")
        if args.quantization_bits != 8 and not args.allow_nonpaper_quantization:
            raise ValueError("the manuscript reports an 8-bit simulator experiment")
        args.backend = (
            "neurosim" if args.device_type == "rram" else "aihwkit-additive"
        ) if args.backend == "auto" else args.backend
        raw, summary = evaluate(args)
        print(raw)
        print(summary)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
