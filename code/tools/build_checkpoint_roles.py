"""Bind experiment roles to immutable checkpoint paths and SHA-256 values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.checkpoint_roles import write_checkpoint_roles


def _role(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("roles must use NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--role", action="append", type=_role, required=True)
    parser.add_argument("--author-verified", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roles: dict[str, Path] = {}
    for name, path in args.role:
        if name in roles:
            raise ValueError(f"duplicate checkpoint role: {name}")
        roles[name] = path
    result = write_checkpoint_roles(
        args.output,
        roles,
        group_id=args.group_id,
        role_assignments_author_verified=args.author_verified,
    )
    print(result["manifest_path"])
    print(f"sha256={result['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
