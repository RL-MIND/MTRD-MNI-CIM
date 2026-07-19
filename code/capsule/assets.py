"""Immutable external-data asset validation commands for public capsule users."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from image_workflows.workflow import (
    build_carvana_index,
    carvana_asset_identity,
    denoising_asset_identity,
    load_carvana_split,
)
from classification.repro import cifar_identity, write_json


def _failure(report: dict[str, Any], name: str, error: Exception) -> None:
    report["checks"].append(
        {
            "name": name,
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    report["errors"].append(f"{name}: {error}")


def _success(report: dict[str, Any], name: str) -> None:
    report["checks"].append({"name": name, "status": "pass"})


def _base_report(kind: str, *, hash_content: bool) -> dict[str, Any]:
    return {
        "schema": "mtrd.codeocean.data-asset-preflight.v1",
        "asset_kind": kind,
        "content_hash_requested": hash_content,
        "checks": [],
        "warnings": [],
        "errors": [],
    }


def validate_cifar(
    dataset: str,
    data_root: str | Path,
    *,
    hash_content: bool,
) -> dict[str, Any]:
    """Validate torchvision's extracted CIFAR layout and official checksums."""
    report = _base_report(dataset, hash_content=hash_content)
    report["data_root"] = str(Path(data_root).expanduser().resolve())
    try:
        identity = cifar_identity(
            dataset,
            Path(data_root),
            include_hash=hash_content,
            normalization_profile="dataset-native",
        )
        report["identity"] = identity
        if not identity["present"]:
            raise FileNotFoundError(
                "the extracted torchvision CIFAR files are missing or fail their official MD5 checks"
            )
        _success(report, "CIFAR extracted layout and official integrity")
    except Exception as error:
        _failure(report, "CIFAR extracted layout and official integrity", error)
    report["ready_for_training_or_clean_test"] = not report["errors"]
    report["status"] = "pass" if report["ready_for_training_or_clean_test"] else "fail"
    return report


def validate_denoising(
    berkeley_root: str | Path,
    set12_dir: str | Path,
    *,
    hash_content: bool,
) -> dict[str, Any]:
    """Validate the Berkeley400 training and Set12 evaluation asset contract."""
    report = _base_report("berkeley400-set12", hash_content=hash_content)
    report["berkeley_root"] = str(Path(berkeley_root).expanduser().resolve())
    report["set12_dir"] = str(Path(set12_dir).expanduser().resolve())
    try:
        identity = denoising_asset_identity(
            {
                "data": {
                    "berkeley_root": str(berkeley_root),
                    "set12_dir": str(set12_dir),
                }
            },
            include_hash=hash_content,
        )
        report["identity"] = identity
        if identity["training_count"] != 400:
            raise ValueError(
                "Berkeley400 training asset must contain exactly 400 PNG images; "
                f"found {identity['training_count']}"
            )
        if identity["set12_count"] != 12:
            raise ValueError(
                f"Set12 asset must contain exactly 12 PNG images; found {identity['set12_count']}"
            )
        if hash_content and identity.get("train_test_duplicate_content_hashes"):
            raise ValueError("Set12 content overlaps Berkeley400 training content")
        if not hash_content:
            report["warnings"].append(
                "content hashing was disabled, so train/test duplicate image content was not checked"
            )
        _success(report, "Berkeley400/Set12 layout, count, and content separation")
    except Exception as error:
        _failure(report, "Berkeley400/Set12 layout, count, and content separation", error)
    report["ready_for_training_or_clean_test"] = not report["errors"]
    report["status"] = "pass" if report["ready_for_training_or_clean_test"] else "fail"
    return report


def validate_carvana(
    image_dirs: Sequence[str | Path],
    mask_dirs: Sequence[str | Path],
    *,
    split_manifest: str | Path | None,
    allow_derived_split: bool,
    require_paper_split: bool,
    hash_content: bool,
) -> dict[str, Any]:
    """Validate paired Carvana files and optionally the explicit split manifest."""
    report = _base_report("carvana", hash_content=hash_content)
    report["image_dirs"] = [str(Path(path).expanduser().resolve()) for path in image_dirs]
    report["mask_dirs"] = [str(Path(path).expanduser().resolve()) for path in mask_dirs]
    config: dict[str, Any] = {
        "data": {
            "carvana_image_dirs": [str(path) for path in image_dirs],
            "carvana_mask_dirs": [str(path) for path in mask_dirs],
            "allow_derived_carvana_split": allow_derived_split,
        }
    }
    index = None
    try:
        index = build_carvana_index(config)
        report["pair_count"] = len(index)
        if len(index) != 5088:
            raise ValueError(
                f"Carvana asset must contain exactly 5,088 paired samples; found {len(index)}"
            )
        report["identity"] = carvana_asset_identity(index, include_hash=hash_content)
        if not hash_content:
            report["warnings"].append(
                "content hashing was disabled, so the paired content manifest was not recorded"
            )
        _success(report, "Carvana image/mask pairing and sample count")
    except Exception as error:
        _failure(report, "Carvana image/mask pairing and sample count", error)

    split_verified = False
    if split_manifest is not None and index is not None:
        config["data"]["carvana_split_manifest"] = str(split_manifest)
        try:
            split = load_carvana_split(config, index)
            split_verified = bool(split["reference_split_author_verified"])
            report["split"] = {
                "path": split["path"],
                "sha256": split["sha256"],
                "split_contract": split["split_contract"],
                "author_verified": split_verified,
                "train_count": len(split["train_ids"]),
                "test_count": len(split["test_ids"]),
                "excluded_count": len(split["excluded_ids"]),
            }
            _success(report, "Carvana explicit split manifest")
        except Exception as error:
            _failure(report, "Carvana explicit split manifest", error)
    elif split_manifest is None:
        report["warnings"].append(
            "no Carvana split manifest was supplied; the raw data asset can be uploaded, "
            "but a paper-value run remains blocked"
        )
    else:
        report["warnings"].append(
            "Carvana split validation was skipped because the paired raw asset did not pass"
        )

    if require_paper_split and not split_verified:
        error = ValueError(
            "an author-verified Carvana split manifest is required for this preflight"
        )
        _failure(report, "author-verified Carvana split requirement", error)
    report["author_verified_paper_split"] = split_verified
    report["ready_for_raw_asset_upload"] = not any(
        check["status"] == "fail"
        for check in report["checks"]
        if check["name"] == "Carvana image/mask pairing and sample count"
    )
    report["ready_for_author_verified_paper_run"] = (
        report["ready_for_raw_asset_upload"] and split_verified
    )
    report["status"] = "pass" if not report["errors"] else "fail"
    return report


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        help="optional JSON report path; use a writable /results path in Code Ocean",
    )
    parser.add_argument(
        "--hash-content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record content identities; enabled by default for publication preflight",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate immutable external data assets before a capsule run"
    )
    subparsers = parser.add_subparsers(dest="asset_kind", required=True)

    cifar = subparsers.add_parser("cifar", help="validate extracted CIFAR-10 or CIFAR-100")
    cifar.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    cifar.add_argument("--data-root", required=True)
    _common_arguments(cifar)

    denoising = subparsers.add_parser(
        "denoising", help="validate Berkeley400 training images and Set12"
    )
    denoising.add_argument("--berkeley-root", required=True)
    denoising.add_argument("--set12-dir", required=True)
    _common_arguments(denoising)

    carvana = subparsers.add_parser("carvana", help="validate paired Carvana images and masks")
    carvana.add_argument("--image-dir", action="append", required=True)
    carvana.add_argument("--mask-dir", action="append", required=True)
    carvana.add_argument("--split-manifest")
    carvana.add_argument(
        "--allow-derived-split",
        action="store_true",
        help="permit a manifest explicitly labeled author_verified=false",
    )
    carvana.add_argument(
        "--require-paper-split",
        action="store_true",
        help="fail unless the supplied split manifest is author verified",
    )
    _common_arguments(carvana)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.asset_kind == "cifar":
        report = validate_cifar(args.dataset, args.data_root, hash_content=args.hash_content)
    elif args.asset_kind == "denoising":
        report = validate_denoising(
            args.berkeley_root, args.set12_dir, hash_content=args.hash_content
        )
    elif args.asset_kind == "carvana":
        report = validate_carvana(
            args.image_dir,
            args.mask_dir,
            split_manifest=args.split_manifest,
            allow_derived_split=args.allow_derived_split,
            require_paper_split=args.require_paper_split,
            hash_content=args.hash_content,
        )
    else:  # pragma: no cover - argparse keeps this unreachable.
        raise AssertionError(args.asset_kind)
    if args.output:
        write_json(Path(args.output).expanduser(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
