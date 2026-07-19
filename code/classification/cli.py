"""Dispatch the public classification workflows without figure-based names."""

from __future__ import annotations

import sys
from typing import Sequence


def _usage() -> str:
    return """Usage: bash code/run classification <pytorch|cim> [arguments]

Classification workflows:
  pytorch    Train or evaluate the PyTorch VGG16 workflow.
  cim        Train or evaluate the CIM VGG16 workflow for CIFAR-10/CIFAR-100.

Run `bash code/run classification <workflow> --help` for workflow-specific options.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested classification workflow."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    workflow, workflow_args = arguments[0], arguments[1:]
    if workflow == "pytorch":
        from .pytorch import main as workflow_main
    elif workflow == "cim":
        from .run import main as workflow_main
    else:
        raise ValueError(f"unknown classification workflow: {workflow}\n\n{_usage()}")
    result = workflow_main(workflow_args)
    return int(result or 0)
