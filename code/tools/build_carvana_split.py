"""Build a deterministic 4700/318/70 Carvana split manifest.

Generated manifests are deliberately marked ``author_verified=false``. They
make a reconstruction run executable, but they do not replace the reference
unpublished sample-ID assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_workflows.workflow import build_carvana_index, carvana_asset_identity


EXPECTED_CARS = 318
EXPECTED_VIEWS_PER_CAR = 16
EXPECTED_IMAGES = EXPECTED_CARS * EXPECTED_VIEWS_PER_CAR
TEST_IMAGES = 318
EXCLUDED_IMAGES = 70
TRAIN_IMAGES = 4700


def _rank(seed: int, namespace: str, sample_id: str) -> bytes:
    payload = f"{int(seed)}\0{namespace}\0{sample_id}".encode("ascii")
    return hashlib.sha256(payload).digest()


def build_split(
    *,
    image_dirs: Sequence[str | Path],
    mask_dirs: Sequence[str | Path],
    seed: int,
    test_policy: str,
    test_view: int,
) -> dict[str, object]:
    """Return a complete deterministic split for one verified Carvana asset."""
    config = {
        "data": {
            "carvana_image_dirs": [str(Path(path)) for path in image_dirs],
            "carvana_mask_dirs": [str(Path(path)) for path in mask_dirs],
        }
    }
    index = build_carvana_index(config)
    if len(index) != EXPECTED_IMAGES:
        raise ValueError(
            f"Carvana must contain {EXPECTED_IMAGES} paired images, found {len(index)}"
        )
    cars: dict[str, list[str]] = defaultdict(list)
    for sample_id in index:
        car_id, separator, raw_view = sample_id.rpartition("_")
        if not separator or not raw_view.isdigit():
            raise ValueError(f"invalid Carvana sample ID: {sample_id}")
        cars[car_id].append(sample_id)
    if len(cars) != EXPECTED_CARS:
        raise ValueError(f"Carvana must contain {EXPECTED_CARS} cars, found {len(cars)}")
    invalid = {
        car_id: len(sample_ids)
        for car_id, sample_ids in cars.items()
        if len(sample_ids) != EXPECTED_VIEWS_PER_CAR
    }
    if invalid:
        raise ValueError(f"every Carvana car must contain 16 views: {invalid}")

    if test_policy not in {"fixed-view", "seeded-random"}:
        raise ValueError("test_policy must be fixed-view or seeded-random")
    if not 1 <= int(test_view) <= EXPECTED_VIEWS_PER_CAR:
        raise ValueError("test_view must be in [1, 16]")
    test_ids: list[str] = []
    for car_id, sample_ids in sorted(cars.items()):
        ordered = sorted(sample_ids)
        if test_policy == "fixed-view":
            suffix = f"_{int(test_view):02d}"
            selected = [sample_id for sample_id in ordered if sample_id.endswith(suffix)]
            if len(selected) != 1:
                raise ValueError(f"car {car_id} has no unique view {suffix}")
            test_ids.append(selected[0])
        else:
            test_ids.append(min(ordered, key=lambda value: _rank(seed, "test", value)))

    test_set = set(test_ids)
    remaining = sorted(set(index) - test_set)
    excluded_ids = sorted(
        sorted(remaining, key=lambda value: _rank(seed, "excluded", value))[
            :EXCLUDED_IMAGES
        ]
    )
    excluded_set = set(excluded_ids)
    train_ids = sorted(set(remaining) - excluded_set)
    if (len(train_ids), len(test_ids), len(excluded_ids)) != (
        TRAIN_IMAGES, TEST_IMAGES, EXCLUDED_IMAGES,
    ):
        raise AssertionError("derived Carvana split has an invalid partition size")

    return {
        "schema_version": 1,
        "dataset": "Carvana Image Masking Challenge",
        "author_verified": False,
        "split_contract": "manuscript-4700-318-70",
        "train_ids": train_ids,
        "test_ids": sorted(test_ids),
        "excluded_ids": excluded_ids,
        "derivation": {
            "tool": "code/tools/build_carvana_split.py",
            "policy": test_policy,
            "test_view": int(test_view) if test_policy == "fixed-view" else None,
            "seed": int(seed),
            "excluded_selection": "lowest sha256(seed, namespace, sample_id)",
            "paper_claim": False,
        },
        "asset_inventory": carvana_asset_identity(index, include_hash=False),
    }


def write_split(path: str | Path, payload: object, *, overwrite: bool) -> Path:
    """Write one manifest atomically without silently replacing an asset."""
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"split manifest already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", action="append", required=True)
    parser.add_argument("--mask-dir", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--test-policy", choices=["fixed-view", "seeded-random"],
        default="fixed-view",
    )
    parser.add_argument("--test-view", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_split(
        image_dirs=args.image_dir,
        mask_dirs=args.mask_dir,
        seed=args.seed,
        test_policy=args.test_policy,
        test_view=args.test_view,
    )
    target = write_split(args.output, payload, overwrite=args.overwrite)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{target}\nsha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
