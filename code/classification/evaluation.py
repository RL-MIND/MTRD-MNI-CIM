"""CIM classification evaluation with explicit simulator identities."""

from __future__ import annotations

import json
import shutil
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from .aihwkit_backend import (
    PINNED_AIHWKIT_VERSION,
    validate_noise_scale,
    convert_model,
    installed_version,
    require_replayable_realization_scope,
)
from .model import PaperVGG16, load_checkpoint_strict, quantized_model
from .protocol import PCM_GRID, RRAM_GRID
from .repro import (
    CODE_ROOT,
    build_cifar_loaders,
    checkpoint_path,
    checkpoint_normalization_identity,
    cifar_identity,
    environment_identity,
    git_identity,
    neurosim_functional_preflight,
    seed_everything,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)


RAW_FIELDS = (
    "dataset",
    "architecture",
    "checkpoint_role",
    "checkpoint_sha256",
    "device_type",
    "backend",
    "realization_scope",
    "noise_symbol",
    "noise_level",
    "trial",
    "seed",
    "quantization_bits",
    "correct",
    "total",
    "accuracy_fraction",
    "accuracy_percent",
)
SUMMARY_FIELDS = (
    "dataset",
    "architecture",
    "checkpoint_role",
    "device_type",
    "backend",
    "realization_scope",
    "noise_symbol",
    "noise_level",
    "trials",
    "accuracy_mean_fraction",
    "accuracy_std_fraction",
    "accuracy_min_fraction",
    "accuracy_max_fraction",
)


def _num_classes(dataset: str) -> int:
    return 10 if dataset == "cifar10" else 100


def _grid(device_type: str) -> tuple[float, ...]:
    if device_type == "rram":
        return RRAM_GRID
    if device_type == "pcm":
        return PCM_GRID
    raise ValueError(device_type)


def _roles(requested: str) -> tuple[str, ...]:
    return ("nominal", "mtrd") if requested == "both" else (requested,)


@torch.no_grad()
def _analog_trial(model, loader) -> tuple[int, int]:
    model.eval()
    correct = total = 0
    for inputs, targets in loader:
        logits = model(inputs.cpu())
        correct += logits.argmax(dim=1).cpu().eq(targets).sum().item()
        total += targets.numel()
    return correct, total


def _backend_identity(args, runtime_checks: list[dict[str, object]]) -> dict[str, object]:
    if args.backend == "neurosim":
        import simulators.neurosim_functional as neurosim_functional
        from simulators.neurosim import require_functional_adapter
        from simulators.neurosim_functional import PAPER_RRAM_EQUATION, PROFILES

        discovered = neurosim_functional_preflight(args.neurosim_home)
        status = require_functional_adapter(discovered["root"], "vgg16")
        profile = PROFILES[args.neurosim_functional_profile]
        root = Path(status["root"])
        source_tree_matches = bool(
            status["revision_matches"] and status["source_tree_matches"]
        )
        return {
            "name": (
                f"neurosim-source-gated-pytorch-eq1-"
                f"{args.neurosim_functional_profile}-"
                f"{args.realization_scope}"
            ),
            "simulator_family": "DNN+NeuroSim-2DInferenceV1.5-dev",
            "neurosim_source_gate_verified": source_tree_matches,
            "effective_equation": PAPER_RRAM_EQUATION,
            "realization_scope": args.realization_scope,
            "realization_scope_author_confirmed": False,
            "author_confirmation_required": True,
            "neurosim_revision": status["revision"],
            "neurosim_source_tree_sha256": status["source_tree_sha256"],
            "functional_adapter": status["public_functional_adapter"],
            "functional_adapter_source_sha256": sha256_file(
                Path(neurosim_functional.__file__)
            ),
            "functional_execution_engine": (
                "source-gated PyTorch functional extension using capsule-defined "
                "signed symmetric fake quantization"
                if profile.quantize_weights or profile.quantize_activations
                else "source-gated PyTorch Eq. (1) extension using the released "
                "Conv2d/Linear perturbation boundary without quantization"
            ),
            "neurosim_source_gate_only": True,
            "upstream_native_cim_array_kernel_used": False,
            "neurosim_quantize_py_sha256": sha256_file(root / "quantize.py"),
            "neurosim_macro_py_sha256": sha256_file(
                root
                / "pytorch-quantization"
                / "pytorch_quantization"
                / "cim"
                / "modules"
                / "macro.py"
            ),
            "functional_profile": asdict(profile),
            "quantization_policy_author_verified": profile.profile_author_verified,
            "weight_bits": args.quantization_bits if profile.quantize_weights else None,
            "dac_bits": args.quantization_bits if profile.quantize_activations else None,
            "adc_bits": args.quantization_bits if profile.quantize_activations else None,
            "programming_noise": True,
            "runtime_noise_scale_checks": [],
            "operator_coverage": (
                "all VGG16 Conv2d and Linear operators are functionally mapped; "
                "normalization, activation, pooling, and bias remain digital"
            ),
            "scope_ambiguity": (
                "the manuscript does not identify the realization lifetime, trial count, "
                "or detailed 8-bit calibration and mapping policy"
            ),
        }
    return {
        "name": (
            "aihwkit-additive-constant-per-mac"
            if args.realization_scope == "per-mac"
            else "aihwkit-equation-programmed-fixed-trial"
        ),
        "aihwkit_version": installed_version(),
        "required_aihwkit_version": PINNED_AIHWKIT_VERSION,
        "weight_mapping": (
            "AIHWKit internally maps layerwise abs-max to 1.0; per-layer w_noise "
            "is multiplied by max(W)/max(abs(W)) so model-space sigma is eta*max(W)"
            if args.realization_scope == "per-mac"
            else "AIHWKit internally maps layerwise abs-max to 1.0; the custom "
            "programming model samples using max(W) in mapped coordinates"
        ),
        "mapping": {
            "weight_scaling_omega": 1.0,
            "weight_scaling_columnwise": False,
            "digital_bias": True,
            "max_input_size": 0,
            "max_output_size": 0,
            "scope": "one_unsplit_logical_layer",
        },
        "realization_scope": args.realization_scope,
        "realization_scope_author_confirmed": False,
        "author_confirmation_required": True,
        "weight_noise_type": (
            "WeightNoiseType.ADDITIVE_CONSTANT"
            if args.realization_scope == "per-mac"
            else "custom BaseNoiseModel programming noise; forward WeightNoiseType.NONE"
        ),
        "w_noise": (
            "eta*max(W)/max(abs(W)) per logical layer"
            if args.realization_scope == "per-mac" else 0.0
        ),
        "effective_equation": "W_g=W+N(0,(eta*max(W))^2)",
        "pcm_wmax_definition": "signed maximum weight value over each logical layer",
        "dac_adc_bits": args.quantization_bits,
        "io_parameters": {
            "is_perfect": False,
            "bound_management": "ITERATIVE",
            "noise_management": "ABS_MAX",
            "inp_bound": 1.0,
            "out_bound": 12.0,
            "inp_res": 1.0 / (2 ** args.quantization_bits - 2),
            "out_res": 1.0 / (2 ** args.quantization_bits - 2),
            "inp_sto_round": False,
            "out_sto_round": False,
            "inp_noise": 0.0,
            "out_noise": 0.0,
            "out_scale": 1.0,
        },
        "weight_bits": args.quantization_bits,
        "programming_noise": args.realization_scope == "fixed-trial",
        "drift": False,
        "runtime_noise_scale_checks": runtime_checks,
        "scope_ambiguity": (
            "the manuscript permits device- or MAC-output-level injection and does not identify "
            "which realization scope generated the reported experiment"
        ),
    }


def evaluate(args) -> tuple[Path, Path]:
    if args.backend == "neurosim" and args.device_type != "rram":
        raise ValueError("neurosim is the RRAM simulator and cannot be selected for PCM")
    if args.backend == "aihwkit-additive" and args.device_type != "pcm":
        raise ValueError("aihwkit-additive is the PCM simulator and cannot be selected for RRAM")
    if args.backend == "neurosim":
        from simulators.neurosim import require_functional_adapter

        status = neurosim_functional_preflight(args.neurosim_home)
        require_functional_adapter(status["root"], "vgg16")
    if args.backend == "aihwkit-additive":
        require_replayable_realization_scope(args.realization_scope)
        actual = installed_version()
        if actual != PINNED_AIHWKIT_VERSION:
            raise RuntimeError(
                f"AIHWKit version mismatch: expected {PINNED_AIHWKIT_VERSION}, found {actual}"
            )

    _, test_loader, _ = build_cifar_loaders(
        args.dataset,
        Path(args.data_root),
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        download=args.download,
        normalization_profile=args.normalization_profile,
    )
    checkpoint_root = Path(args.checkpoint_root)
    roles = _roles(args.checkpoint_role)
    if args.checkpoint and (
        len(roles) != 1 or args.nominal_checkpoint or args.mtrd_checkpoint
    ):
        raise ValueError("--checkpoint requires --checkpoint-role nominal or mtrd")
    explicit_paths = {
        "nominal": args.nominal_checkpoint,
        "mtrd": args.mtrd_checkpoint,
    }
    paths = {
        role: (
            Path(args.checkpoint)
            if args.checkpoint
            else (
                Path(explicit_paths[role])
                if explicit_paths[role]
                else checkpoint_path(checkpoint_root, args.dataset, args.device_type, role)
            )
        )
        for role in roles
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("evaluation checkpoints are missing:\n" + "\n".join(missing))
    from utils.checkpoint_roles import validate_checkpoint_roles

    checkpoint_role_identity = validate_checkpoint_roles(
        args.checkpoint_role_manifest,
        {
            f"classification.{args.dataset}.{args.device_type}.{role}": path
            for role, path in paths.items()
        },
    )
    checkpoint_normalization = {
        role: checkpoint_normalization_identity(
            path,
            dataset=args.dataset,
            requested_profile=args.normalization_profile,
        )
        for role, path in paths.items()
    }

    runtime_checks: list[dict[str, object]] = []
    if args.backend == "aihwkit-additive":
        for noise_index, noise in enumerate(_grid(args.device_type)):
            result = validate_noise_scale(
                noise,
                samples=1024,
                seed=args.seed + 50000000 + noise_index,
                realization_scope=args.realization_scope,
            )
            runtime_checks.append(result)
            if not result["passed"]:
                raise RuntimeError(
                    "AIHWKit additive-noise validation failed; refusing to produce "
                    "CIM metrics:\n" + json.dumps(result, indent=2, sort_keys=True)
                )

    backend = _backend_identity(args, runtime_checks)
    output_dir = Path(args.output_dir) / args.dataset / args.device_type / str(backend["name"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"evaluation output exists; pass --overwrite to replace it: {output_dir}"
            )
        shutil.rmtree(output_dir)
    rows: list[dict[str, object]] = []
    programming_manifests: dict[str, object] = {}
    for role_index, (role, path) in enumerate(paths.items()):
        base = PaperVGG16(_num_classes(args.dataset))
        load_checkpoint_strict(base, str(path), "cpu")
        if args.backend == "aihwkit-additive":
            base = quantized_model(base, args.quantization_bits)
        checkpoint_hash = sha256_file(path)
        for noise_index, noise in enumerate(_grid(args.device_type)):
            for trial in range(args.trials):
                role_offset = role_index * 1000000 if args.method_realization_policy == "independent" else 0
                trial_seed = args.seed + role_offset + noise_index * 10000 + trial
                seed_everything(trial_seed)
                if args.backend == "neurosim":
                    from simulators.neurosim_functional import program_rram_fixed_trial

                    analog, programming = program_rram_fixed_trial(
                        base,
                        model_name="vgg16",
                        sigma=noise,
                        seed=trial_seed,
                        profile=args.neurosim_functional_profile,
                        weight_bits=args.quantization_bits,
                        dac_bits=args.quantization_bits,
                        adc_bits=args.quantization_bits,
                    )
                    programming_manifests[f"{role}.{noise:g}.{trial}"] = programming
                else:
                    analog = convert_model(
                        base,
                        noise,
                        input_bits=args.quantization_bits,
                        output_bits=args.quantization_bits,
                        seed=trial_seed,
                        realization_scope=args.realization_scope,
                    )
                correct, total = _analog_trial(analog, test_loader)
                del analog
                if total == 0:
                    raise RuntimeError("empty CIFAR test set")
                accuracy = correct / total
                row = {
                    "dataset": args.dataset,
                    "architecture": "VGG16",
                    "checkpoint_role": role,
                    "checkpoint_sha256": checkpoint_hash,
                    "device_type": args.device_type,
                    "backend": backend["name"],
                    "realization_scope": args.realization_scope,
                    "noise_symbol": "sigma" if args.device_type == "rram" else "eta",
                    "noise_level": noise,
                    "trial": trial,
                    "seed": trial_seed,
                    "quantization_bits": backend["weight_bits"],
                    "correct": correct,
                    "total": total,
                    "accuracy_fraction": accuracy,
                    "accuracy_percent": 100.0 * accuracy,
                }
                rows.append(row)
                print(
                    f"{args.dataset} {role} {args.device_type} {noise:g} "
                    f"trial={trial} seed={trial_seed} accuracy={accuracy:.6f}"
                )

    summary_rows: list[dict[str, object]] = []
    for role in roles:
        for noise in _grid(args.device_type):
            values = [
                float(row["accuracy_fraction"])
                for row in rows
                if row["checkpoint_role"] == role and float(row["noise_level"]) == noise
            ]
            summary_rows.append(
                {
                    "dataset": args.dataset,
                    "architecture": "VGG16",
                    "checkpoint_role": role,
                    "device_type": args.device_type,
                    "backend": backend["name"],
                    "realization_scope": args.realization_scope,
                    "noise_symbol": "sigma" if args.device_type == "rram" else "eta",
                    "noise_level": noise,
                    "trials": len(values),
                    "accuracy_mean_fraction": statistics.fmean(values),
                    "accuracy_std_fraction": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "accuracy_min_fraction": min(values),
                    "accuracy_max_fraction": max(values),
                }
            )

    raw_path = output_dir / "classification_raw_trials.csv"
    summary_path = output_dir / "classification_summary.csv"
    write_csv(raw_path, rows, RAW_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    manifest = {
        "schema": "mtrd.classification.evaluation.v1",
        "created_utc": utc_now(),
        "task": "classification",
        "dataset": cifar_identity(
            args.dataset,
            Path(args.data_root),
            include_hash=True,
            normalization_profile=args.normalization_profile,
        ),
        "architecture": "VGG16",
        "device_type": args.device_type,
        "noise_grid": list(_grid(args.device_type)),
        "trials": args.trials,
        "base_seed": args.seed,
        "quantization_bits_requested": args.quantization_bits,
        "effective_weight_bits": backend["weight_bits"],
        "effective_dac_bits": backend.get("dac_bits"),
        "effective_adc_bits": backend.get("adc_bits"),
        "realization_scope": args.realization_scope,
        "realization_scope_author_confirmed": False,
        "method_realization_policy": args.method_realization_policy,
        "method_realization_policy_author_confirmed": False,
        "author_confirmation_required": True,
        "backend": backend,
        "functional_programming": programming_manifests,
        "evaluation_split": "official CIFAR test split",
        "held_out_from_training_feedback_and_checkpoint_selection": False,
        "data_leakage_warning": (
            "the supplied legacy training protocol reuses this split for Eq.(6) feedback and "
            "checkpoint selection, so this is a training-set diagnostic rather than an independent holdout"
        ),
        "numerical_reproduction_verified": False,
        "author_raw_reference": None,
        "verification_blocker": (
            "author raw trial values and an acceptance tolerance are not included in the public assets"
        ),
        "checkpoints": {
            role: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "normalization_provenance": checkpoint_normalization[role],
            }
            for role, path in paths.items()
        },
        "checkpoint_role_manifest": checkpoint_role_identity,
        "raw_csv": str(raw_path.resolve()),
        "raw_csv_sha256": sha256_file(raw_path),
        "summary_csv": str(summary_path.resolve()),
        "summary_csv_sha256": sha256_file(summary_path),
        "environment": environment_identity(),
        "source": git_identity(CODE_ROOT),
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    return raw_path, summary_path
