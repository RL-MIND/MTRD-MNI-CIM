"""Resolve the public Code Ocean data and result roots."""

from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
CAPSULE_ROOT = CODE_ROOT.parent


def is_code_ocean() -> bool:
    """Return whether this source tree is mounted as a Code Ocean capsule."""
    return CODE_ROOT == Path("/code") or (
        Path("/code").is_dir()
        and Path("/data").is_dir()
        and CODE_ROOT.parent == Path("/")
    )


def _requested_path(
    explicit: str | os.PathLike[str] | None,
    environment_name: str,
) -> Path | None:
    value = explicit if explicit is not None else os.environ.get(environment_name)
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser()


def data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the immutable input-data root unless an explicit path is given."""
    requested = _requested_path(explicit, "MTRD_DATA_ROOT")
    if requested is not None:
        return requested
    return Path("/data") if is_code_ocean() else CAPSULE_ROOT / "data"


def results_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the writable result root unless an explicit path is given."""
    requested = _requested_path(explicit, "MTRD_RESULTS_ROOT")
    if requested is not None:
        return requested
    return Path("/results") if is_code_ocean() else CAPSULE_ROOT / "results"
