"""Build the 4,588/500 Carvana directory split used by released scripts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_workflows.workflow import build_carvana_index, carvana_asset_identity
from tools.build_carvana_split import write_split


TRAIN_IMAGES = 4588
TEST_IMAGES = 500
TOTAL_IMAGES = TRAIN_IMAGES + TEST_IMAGES


def _index(image_dir: str | Path, mask_dir: str | Path):
    return build_carvana_index({
        "data": {
            "carvana_image_dirs": [str(Path(image_dir))],
            "carvana_mask_dirs": [str(Path(mask_dir))],
        }
    })


def build_released_split(
    *,
    train_image_dir: str | Path,
    train_mask_dir: str | Path,
    test_image_dir: str | Path,
    test_mask_dir: str | Path,
) -> dict[str, object]:
    """Return a manifest for the immutable released directory assignment."""
    train = _index(train_image_dir, train_mask_dir)
    test = _index(test_image_dir, test_mask_dir)
    if len(train) != TRAIN_IMAGES:
        raise ValueError(
            f"released Carvana train directory must contain {TRAIN_IMAGES} pairs, "
            f"found {len(train)}"
        )
    if len(test) != TEST_IMAGES:
        raise ValueError(
            f"released Carvana test directory must contain {TEST_IMAGES} pairs, "
            f"found {len(test)}"
        )
    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(
            f"released Carvana train/test directories overlap: {sorted(overlap)[:5]}"
        )
    combined = {**train, **test}
    if len(combined) != TOTAL_IMAGES:
        raise AssertionError("released Carvana directory split is incomplete")
    return {
        "schema_version": 1,
        "dataset": "Carvana Image Masking Challenge",
        "author_verified": False,
        "split_contract": "released-directory-4588-500",
        "train_ids": sorted(train),
        "test_ids": sorted(test),
        "excluded_ids": [],
        "derivation": {
            "tool": "code/tools/build_released_carvana_split.py",
            "policy": "preserve released train_hq/test_hq directory assignment",
            "paper_claim": False,
        },
        "asset_inventory": carvana_asset_identity(combined, include_hash=False),
        "train_inventory": carvana_asset_identity(train, include_hash=False),
        "test_inventory": carvana_asset_identity(test, include_hash=False),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--train-mask-dir", required=True)
    parser.add_argument("--test-image-dir", required=True)
    parser.add_argument("--test-mask-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_released_split(
        train_image_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir,
        test_image_dir=args.test_image_dir,
        test_mask_dir=args.test_mask_dir,
    )
    target = write_split(args.output, payload, overwrite=args.overwrite)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{target}\nsha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
