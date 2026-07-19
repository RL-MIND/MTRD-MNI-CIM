"""Paper-specific teacher and MTRD student training loops."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import PaperVGG16, load_checkpoint_strict, noisy_forward
from .protocol import EpochDeltaBalancer, STUDENT_NOISE, TeacherSpec, paper_mtrd_loss, teacher_specs
from .repro import (
    append_csv,
    build_cifar_loaders,
    capture_rng_state,
    checkpoint_manifest,
    checkpoint_path,
    cifar_identity,
    restore_rng_state,
    seed_everything,
    sha256_file,
    write_json,
)


TRAIN_LOG_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy_fraction",
    "checkpoint_selection_test_accuracy_fraction",
    "best_checkpoint_selection_test_accuracy_fraction",
)
STUDENT_LOG_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_kd_loss",
    "train_task_loss",
    "train_noisy_accuracy_fraction",
    "robust_mean_accuracy_fraction",
    "best_robust_mean_accuracy_fraction",
    "beta_json",
    "performance_json",
)


def _num_classes(dataset: str) -> int:
    return 10 if dataset == "cifar10" else 100


def _identity_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_resume_identity(state: dict[str, object], expected: str) -> None:
    measured = state.get("resume_identity_sha256")
    if measured != expected:
        raise RuntimeError(
            "resume state was created with different data, teachers, or training protocol: "
            f"expected identity {expected}, found {measured!r}"
        )


@torch.no_grad()
def equation_accuracy(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    device_type: str,
    noise: float,
    seed: int,
) -> tuple[float, int, int]:
    seed_everything(seed)
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = noisy_forward(model, inputs, device_type, noise)
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        total += targets.numel()
    if total == 0:
        raise RuntimeError("empty evaluation dataset")
    return correct / total, correct, total


def _teacher_protocol(args, spec: TeacherSpec) -> dict[str, object]:
    return {
        "task": "classification",
        "stage": "clean_teacher" if spec.clean else "variation_aware_teacher",
        "training_backend": "PyTorch equation injection",
        "analog_tile_hardware_aware_training": False,
        "aihwkit_usage": "inference only; not used by this training command",
        "device_type": args.device_type,
        "noise_equation": (
            "W_g=W_nominal*exp(N(0,sigma^2))"
            if args.device_type == "rram"
            else "W_g=W_nominal+N(0,(eta*max(W_nominal))^2)"
        ),
        "noise": spec.noise,
        "noise_realization": "fresh independent tensor per Conv2d/Linear layer and forward pass",
        "epochs": args.epochs,
        "optimizer": "SGD",
        "initial_lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss",
        "manuscript_loss_warning": (
            "the manuscript says sigmoid/BCE, but released single-label source uses CrossEntropyLoss"
        ),
        "checkpoint_selection_split": "official CIFAR test split",
        "independent_held_out_test": False,
        "data_leakage_warning": (
            "the supplied legacy method uses the official CIFAR test split every epoch for "
            "checkpoint selection; final paper-grid evaluation reuses that split"
        ),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "normalization_profile": args.normalization_profile,
        "resolved_device": str(args.torch_device),
        "seed": args.seed,
    }


def _load_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, float, dict[str, object]]:
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch before weights_only
        state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    restore_rng_state(state["rng_state"], generator)
    return int(state["epoch"]) + 1, float(state["best_score"]), state


def _save_training_state(
    path: Path,
    *,
    epoch: int,
    best_score: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    generator: torch.Generator,
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "schema": "mtrd.classification.training-state.v1",
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(generator),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one_teacher(args, spec: TeacherSpec, dataset_info: dict[str, object]) -> Path:
    output = checkpoint_path(
        Path(args.checkpoint_root), args.dataset, args.device_type,
        "clean" if spec.clean else "teacher", spec.noise,
    )
    state_path = output.with_suffix(".train-state.pt")
    log_path = output.with_suffix(".epochs.csv")
    if output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"checkpoint exists; use --resume or --overwrite: {output}")
    if args.overwrite and not args.resume and log_path.exists():
        log_path.unlink()

    run_seed = args.seed + int(round(spec.noise * 10000)) + (0 if spec.clean else 100000)
    seed_everything(run_seed)
    train_loader, test_loader, generator = build_cifar_loaders(
        args.dataset,
        Path(args.data_root),
        batch_size=args.batch_size,
        workers=args.workers,
        seed=run_seed,
        download=args.download,
        normalization_profile=args.normalization_profile,
    )
    model = PaperVGG16(_num_classes(args.dataset)).to(args.torch_device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 0
    best = -1.0
    resume_identity = _identity_sha256({
        "dataset": dataset_info,
        "protocol": _teacher_protocol(args, spec),
        "architecture": "PaperVGG16",
    })
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state is missing: {state_path}")
        start_epoch, best, state = _load_training_state(
            state_path, model, optimizer, scheduler, generator, args.torch_device
        )
        _require_resume_identity(state, resume_identity)

    for epoch in range(start_epoch, args.epochs):
        seed_everything(run_seed + epoch * 1000003)
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(args.torch_device, non_blocking=True)
            targets = targets.to(args.torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = noisy_forward(model, inputs, args.device_type, spec.noise)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.numel()
            correct += logits.argmax(dim=1).eq(targets).sum().item()
            total += targets.numel()

        feedback_test_accuracy, _, _ = equation_accuracy(
            model,
            test_loader,
            args.torch_device,
            args.device_type,
            spec.noise,
            run_seed + 900000000 + epoch,
        )
        if feedback_test_accuracy > best:
            best = feedback_test_accuracy
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)
        append_csv(
            log_path,
            {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": loss_sum / total,
                "train_accuracy_fraction": correct / total,
                "checkpoint_selection_test_accuracy_fraction": feedback_test_accuracy,
                "best_checkpoint_selection_test_accuracy_fraction": best,
            },
            TRAIN_LOG_FIELDS,
        )
        scheduler.step()
        _save_training_state(
            state_path,
            epoch=epoch,
            best_score=best,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            extra={"resume_identity_sha256": resume_identity},
        )
        print(
            f"teacher={spec.label} epoch={epoch + 1}/{args.epochs} "
            f"train={correct / total:.6f} feedback_test={feedback_test_accuracy:.6f} "
            f"best={best:.6f}"
        )

    manifest = checkpoint_manifest(
        output,
        role="clean_teacher" if spec.clean else "variation_aware_teacher",
        dataset_identity=dataset_info,
        protocol=_teacher_protocol(args, spec),
        metrics={"best_checkpoint_selection_test_accuracy_fraction": best},
    )
    write_json(output.with_suffix(".manifest.json"), manifest)
    return output


def train_teachers(args) -> list[Path]:
    dataset_info = cifar_identity(
        args.dataset,
        Path(args.data_root),
        include_hash=True,
        normalization_profile=args.normalization_profile,
    )
    if not dataset_info["present"] and not args.download:
        raise FileNotFoundError(f"CIFAR asset is missing below {args.data_root}")
    specs = teacher_specs(args.device_type, include_clean=not args.no_clean_teacher)
    outputs = [train_one_teacher(args, spec, dataset_info) for spec in specs]
    index = {
        "schema": "mtrd.classification.teacher-pool.v1",
        "dataset": args.dataset,
        "device_type": args.device_type,
        "normalization": dataset_info["normalization"],
        "teachers": [
            {
                "label": spec.label,
                "noise": spec.noise,
                "clean": spec.clean,
                "checkpoint": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for spec, path in zip(specs, outputs)
        ],
    }
    pool_path = Path(args.checkpoint_root) / args.dataset / args.device_type / "teacher_pool.json"
    write_json(pool_path, index)
    return outputs


def _student_protocol(args, specs: tuple[TeacherSpec, ...], initial_checkpoint: Path) -> dict[str, object]:
    return {
        "task": "classification",
        "stage": "mtrd_student",
        "training_backend": "PyTorch equation injection",
        "analog_tile_hardware_aware_training": False,
        "aihwkit_usage": "inference only; not used by this training command",
        "device_type": args.device_type,
        "teacher_noises": [spec.noise for spec in specs],
        "includes_clean_teacher": any(spec.clean for spec in specs),
        "initial_checkpoint": str(initial_checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "student_training_noise": STUDENT_NOISE[args.device_type],
        "epochs": args.epochs,
        "optimizer": "SGD",
        "initial_lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "eq4": (
            "alpha*sum_i(beta_i*KL(teacher_i, nominal_student)) + "
            "(1-alpha)*CrossEntropy(noisy_student, label)"
        ),
        "manuscript_loss_warning": (
            "the Methods text says sigmoid/BCE for single-label CIFAR classification; "
            "the released source and this runner use multiclass CrossEntropyLoss"
        ),
        "alpha": args.alpha,
        "temperature": args.temperature,
        "eq6": EpochDeltaBalancer.equation,
        "eq6_performance_unit": "accuracy fraction in [0,1]",
        "eq6_warning": (
            "literal printed positive delta is implemented; manuscript prose describes the opposite preference"
        ),
        "legacy_implementation_warning": (
            "the supplied legacy KD code uses different KL reduction/scaling and teacher-logit "
            "aggregation; this runner implements the printed Eq.(4), and the author must identify "
            "which implementation produced the reported checkpoints"
        ),
        "pcm_wmax_definition": (
            "signed maximum weight value max(W_nominal) over each logical layer"
        ),
        "noise_realization": "fresh independent tensor per Conv2d/Linear layer and forward pass",
        "eq6_feedback_split": "official CIFAR test split",
        "checkpoint_selection_split": "official CIFAR test split",
        "independent_held_out_test": False,
        "data_leakage_warning": (
            "the supplied legacy method uses the official CIFAR test split for Eq.(6), "
            "checkpoint selection, and final paper-grid evaluation"
        ),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "normalization_profile": args.normalization_profile,
        "resolved_device": str(args.torch_device),
        "seed": args.seed,
    }


def train_student(args) -> Path:
    dataset_info = cifar_identity(
        args.dataset,
        Path(args.data_root),
        include_hash=True,
        normalization_profile=args.normalization_profile,
    )
    if not dataset_info["present"] and not args.download:
        raise FileNotFoundError(f"CIFAR asset is missing below {args.data_root}")
    specs = teacher_specs(args.device_type, include_clean=True)
    teacher_paths = [
        checkpoint_path(
            Path(args.checkpoint_root), args.dataset, args.device_type,
            "clean" if spec.clean else "teacher", spec.noise,
        )
        for spec in specs
    ]
    missing = [str(path) for path in teacher_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("teacher pool is incomplete:\n" + "\n".join(missing))
    initial_noise = STUDENT_NOISE[args.device_type]
    initial_checkpoint = checkpoint_path(
        Path(args.checkpoint_root), args.dataset, args.device_type, "teacher", initial_noise
    )
    output = checkpoint_path(Path(args.checkpoint_root), args.dataset, args.device_type, "mtrd")
    state_path = output.with_suffix(".train-state.pt")
    log_path = output.with_suffix(".epochs.csv")
    if output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"checkpoint exists; use --resume or --overwrite: {output}")
    if args.overwrite and not args.resume and log_path.exists():
        log_path.unlink()

    seed_everything(args.seed)
    train_loader, test_loader, generator = build_cifar_loaders(
        args.dataset,
        Path(args.data_root),
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        download=args.download,
        normalization_profile=args.normalization_profile,
    )
    student = PaperVGG16(_num_classes(args.dataset)).to(args.torch_device)
    load_checkpoint_strict(student, str(initial_checkpoint), args.torch_device)
    teachers = []
    for path in teacher_paths:
        teacher = PaperVGG16(_num_classes(args.dataset)).to(args.torch_device)
        load_checkpoint_strict(teacher, str(path), args.torch_device)
        teacher.eval()
        teacher.requires_grad_(False)
        teachers.append(teacher)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    balancer = EpochDeltaBalancer(len(specs))
    initial_performance = [
        equation_accuracy(
            student,
            test_loader,
            args.torch_device,
            args.device_type,
            spec.noise,
            args.seed + 700000000 + index,
        )[0]
        for index, spec in enumerate(specs)
    ]
    balancer.update(initial_performance)
    history: list[dict[str, object]] = [
        {"epoch": 0, "performance": initial_performance, "beta": balancer.beta.tolist()}
    ]
    start_epoch = 0
    best = -1.0
    teacher_hashes = [sha256_file(path) for path in teacher_paths]
    resume_identity = _identity_sha256({
        "dataset": dataset_info,
        "protocol": _student_protocol(args, specs, initial_checkpoint),
        "architecture": "PaperVGG16",
        "teacher_sha256": teacher_hashes,
    })
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state is missing: {state_path}")
        start_epoch, best, state = _load_training_state(
            state_path, student, optimizer, scheduler, generator, args.torch_device
        )
        balancer.load_state_dict(state["balancer_state"])
        history = state["performance_history"]
        _require_resume_identity(state, resume_identity)
        if state.get("teacher_sha256") != teacher_hashes:
            raise RuntimeError("resume state teacher hashes differ from the current teacher pool")

    for epoch in range(start_epoch, args.epochs):
        seed_everything(args.seed + epoch * 1000003)
        student.train()
        loss_sum = kd_sum = task_sum = 0.0
        correct = total = 0
        beta = balancer.beta.to(args.torch_device, dtype=torch.float32)
        for inputs, targets in train_loader:
            inputs = inputs.to(args.torch_device, non_blocking=True)
            targets = targets.to(args.torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            nominal_logits = student(inputs)
            noisy_logits = noisy_forward(student, inputs, args.device_type, initial_noise)
            with torch.no_grad():
                teacher_logits = [
                    noisy_forward(teacher, inputs, args.device_type, spec.noise)
                    for teacher, spec in zip(teachers, specs)
                ]
            loss, pieces = paper_mtrd_loss(
                nominal_logits,
                noisy_logits,
                teacher_logits,
                targets,
                beta,
                alpha=args.alpha,
                temperature=args.temperature,
            )
            loss.backward()
            optimizer.step()
            count = targets.numel()
            loss_sum += loss.item() * count
            kd_sum += pieces["kd"].item() * count
            task_sum += pieces["task"].item() * count
            correct += noisy_logits.argmax(dim=1).eq(targets).sum().item()
            total += count

        performance = [
            equation_accuracy(
                student,
                test_loader,
                args.torch_device,
                args.device_type,
                spec.noise,
                args.seed + 800000000 + epoch * 100 + index,
            )[0]
            for index, spec in enumerate(specs)
        ]
        next_beta = balancer.update(performance)
        robust_values = [value for value, spec in zip(performance, specs) if not spec.clean]
        robust_mean = sum(robust_values) / len(robust_values)
        if robust_mean > best:
            best = robust_mean
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), output)
        history.append(
            {"epoch": epoch + 1, "performance": performance, "beta": next_beta.tolist()}
        )
        append_csv(
            log_path,
            {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": loss_sum / total,
                "train_kd_loss": kd_sum / total,
                "train_task_loss": task_sum / total,
                "train_noisy_accuracy_fraction": correct / total,
                "robust_mean_accuracy_fraction": robust_mean,
                "best_robust_mean_accuracy_fraction": best,
                "beta_json": json.dumps(next_beta.tolist(), separators=(",", ":")),
                "performance_json": json.dumps(performance, separators=(",", ":")),
            },
            STUDENT_LOG_FIELDS,
        )
        scheduler.step()
        _save_training_state(
            state_path,
            epoch=epoch,
            best_score=best,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            extra={
                "balancer_state": balancer.state_dict(),
                "performance_history": history,
                "teacher_sha256": teacher_hashes,
                "resume_identity_sha256": resume_identity,
            },
        )
        print(
            f"student epoch={epoch + 1}/{args.epochs} noisy_train={correct / total:.6f} "
            f"robust_mean={robust_mean:.6f} best={best:.6f} beta={next_beta.tolist()}"
        )

    manifest = checkpoint_manifest(
        output,
        role="mtrd_student",
        dataset_identity=dataset_info,
        protocol=_student_protocol(args, specs, initial_checkpoint),
        metrics={
            "best_mean_accuracy_over_paper_noise_grid_fraction": best,
            "performance_history": history,
        },
    )
    manifest["teachers"] = [
        {
            "label": spec.label,
            "noise": spec.noise,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": sha256_file(path),
        }
        for spec, path in zip(specs, teacher_paths)
    ]
    write_json(output.with_suffix(".manifest.json"), manifest)
    return output
