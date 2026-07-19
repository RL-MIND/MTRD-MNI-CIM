"""Validate explicit checkpoint-to-experiment role assignments."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_checkpoint_roles(
    manifest_path: str | Path,
    roles: Mapping[str, str | Path],
    *,
    group_id: str,
    role_assignments_author_verified: bool = False,
) -> dict[str, Any]:
    """Write an atomic, content-bound role manifest for existing checkpoints."""
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("checkpoint role manifest group_id must be a non-empty string")
    if not roles:
        raise ValueError("at least one checkpoint role is required")
    entries: dict[str, dict[str, Any]] = {}
    for role, value in sorted(roles.items()):
        if not isinstance(role, str) or not role.strip():
            raise ValueError("checkpoint role names must be non-empty strings")
        path = Path(value).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"checkpoint role {role} is missing or empty: {path}")
        entries[role] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {
        "schema_version": 1,
        "group_id": group_id,
        "asset_root": "/",
        "role_assignments_author_verified": bool(role_assignments_author_verified),
        "roles": entries,
    }
    target = Path(manifest_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return validate_checkpoint_roles(target, roles)


def validate_checkpoint_roles(
    manifest_path: str | Path,
    expected_roles: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Require exact path, size, and digest bindings for every expected role."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint role manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid checkpoint role manifest JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("checkpoint role manifest schema_version must be 1")
    if not isinstance(payload.get("group_id"), str) or not payload["group_id"].strip():
        raise ValueError("checkpoint role manifest group_id must be a non-empty string")
    if not isinstance(payload.get("role_assignments_author_verified"), bool):
        raise ValueError(
            "checkpoint role manifest must set role_assignments_author_verified "
            "to true or false explicitly"
        )
    asset_root_value = payload.get("asset_root")
    if not isinstance(asset_root_value, str) or not asset_root_value:
        raise ValueError("checkpoint role manifest asset_root must be an absolute path")
    asset_root = Path(asset_root_value).expanduser()
    if not asset_root.is_absolute():
        raise ValueError("checkpoint role manifest asset_root must be an absolute path")
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("checkpoint role manifest roles must be an object")

    validated: dict[str, Any] = {}
    for role, expected_value in expected_roles.items():
        entry = roles.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"checkpoint role is missing or not a single object: {role}")
        entry_path_value = entry.get("path")
        if not isinstance(entry_path_value, str) or not entry_path_value:
            raise ValueError(f"checkpoint role {role} has no path")
        entry_path = Path(entry_path_value).expanduser()
        resolved = (
            entry_path.resolve()
            if entry_path.is_absolute()
            else (asset_root / entry_path).resolve()
        )
        expected = Path(expected_value).expanduser().resolve()
        if resolved != expected:
            raise ValueError(
                f"checkpoint role {role} resolves to {resolved}, expected {expected}"
            )
        if not resolved.is_file():
            raise FileNotFoundError(f"checkpoint role {role} is missing: {resolved}")
        size = resolved.stat().st_size
        declared_size = entry.get("size_bytes")
        if not isinstance(declared_size, int) or declared_size <= 0:
            raise ValueError(f"checkpoint role {role} size_bytes must be positive")
        if size != declared_size:
            raise ValueError(
                f"checkpoint role {role} size mismatch: declared {declared_size}, found {size}"
            )
        declared_hash = entry.get("sha256")
        if not isinstance(declared_hash, str) or not SHA256_PATTERN.fullmatch(declared_hash):
            raise ValueError(f"checkpoint role {role} must declare a lowercase SHA-256")
        measured_hash = sha256_file(resolved)
        if measured_hash != declared_hash:
            raise ValueError(
                f"checkpoint role {role} SHA-256 mismatch: declared {declared_hash}, "
                f"found {measured_hash}"
            )
        validated[role] = {
            "path": str(resolved),
            "size_bytes": size,
            "sha256": measured_hash,
        }

    return {
        "schema_version": 1,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "group_id": payload["group_id"],
        "role_assignments_author_verified": payload[
            "role_assignments_author_verified"
        ],
        "asset_root": str(asset_root.resolve()),
        "validated_roles": validated,
    }
