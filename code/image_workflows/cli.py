"""Shared helpers for task-specific image workflow entry points."""

from __future__ import annotations

import sys
from typing import Sequence

from .workflow import TASKS, main as workflow_main


def run_task(task: str, argv: Sequence[str] | None = None) -> int:
    """Run one task and prevent accidental execution of the other task."""
    if task not in TASKS:
        raise ValueError(f"unsupported image workflow task: {task}")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--task" in arguments:
        raise SystemExit(
            f"{task} entry point fixes --task to {task}; remove the explicit --task option"
        )
    if task != "denoising" and "--overwrite-h5" in arguments:
        raise SystemExit(
            "--overwrite-h5 is only available from the denosing entry point"
        )
    return workflow_main(arguments, fixed_task=task)
