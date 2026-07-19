"""Run only the DnCNN denoising workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_workflows.cli import run_task


def main(argv: Sequence[str] | None = None) -> int:
    """Run denoising preflight, training, or evaluation."""
    return run_task("denoising", argv)


if __name__ == "__main__":
    raise SystemExit(main())
