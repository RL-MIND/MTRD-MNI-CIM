"""Compute a deterministic identity for the public first-party code tree."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def source_tree_sha256(root: str | Path) -> str:
    """Hash relative paths and contents while excluding runtime cache files."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"source tree is missing: {base}")
    digest = hashlib.sha256()
    files = [
        path
        for path in base.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(base).parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    ]
    for path in sorted(files, key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()
