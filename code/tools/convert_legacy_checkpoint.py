#!/usr/bin/env python3
"""Convert an original-repository checkpoint to the unified strict format."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import get_model
from utils.checkpoints import (
    load_state_dict,
    remap_legacy_state_dict,
    state_dict_finiteness,
)
from utils.paths import results_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_target(
    task: str, model_name: str, num_classes: int, num_layers: int
) -> torch.nn.Module:
    if task == "classification":
        return get_model(model_name, num_classes=num_classes)
    if task == "denoising":
        return get_model("dncnn", channels=1, num_of_layers=num_layers)
    if task == "segmentation":
        return get_model("unet", in_channels=3, out_channels=1)
    raise ValueError(task)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a trusted legacy MTRD checkpoint and strictly verify every key."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("classification", "denoising", "segmentation"))
    parser.add_argument("--model", default="vgg16", help="Classification model; legacy conversion supports vgg16.")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=17, help="DnCNN depth; use 17 for DnCNN-S or 20 for DnCNN-B.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--state-dict-only", action="store_true", help="Write a bare state_dict instead of a provenance envelope.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = args.output
    if output is None:
        output = results_root() / "converted_checkpoints" / f"{source.stem}.unified.pth"
    output = output.resolve()
    if output == source:
        raise ValueError("refusing to overwrite the legacy source checkpoint")
    report_path = output.with_suffix(output.suffix + ".json")
    existing = [path for path in (output, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing conversion output: "
            + ", ".join(str(path) for path in existing)
        )

    original = load_state_dict(source, map_location="cpu")
    nonfinite, tensor_values = state_dict_finiteness(original)
    if nonfinite:
        raise ValueError(f"source checkpoint has {nonfinite}/{tensor_values} non-finite values")
    converted = remap_legacy_state_dict(
        original, args.task, args.model, num_layers=args.num_layers
    )
    if len(converted) != len(original):
        raise RuntimeError("conversion changed the number of state-dict entries")

    model = create_target(args.task, args.model, args.num_classes, args.num_layers)
    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict validation failed: {incompatible}")

    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_name = {
        "classification": args.model,
        "denoising": "dncnn",
        "segmentation": "unet",
    }[args.task]
    provenance = {
        "schema_version": 1,
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "model_name": resolved_model_name,
        "num_classes": args.num_classes if args.task == "classification" else None,
        "num_layers": args.num_layers if args.task == "denoising" else None,
        "source_path": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "source_state_entries": len(original),
        "tensor_values": tensor_values,
        "conversion": "strict-key-remap-no-tensor-modification",
    }
    payload = converted if args.state_dict_only else {**provenance, "state_dict": converted}
    torch.save(payload, output)

    reloaded = load_state_dict(output, map_location="cpu")
    create_target(args.task, args.model, args.num_classes, args.num_layers).load_state_dict(
        reloaded, strict=True
    )
    report = {**provenance, "output_path": str(output), "output_sha256": sha256(output)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Converted checkpoint: {output}")
    print(f"Conversion report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
