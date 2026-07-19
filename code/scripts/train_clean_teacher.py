#!/usr/bin/env python3
"""Train the clean teacher through the shared classification runner."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
for path in (SCRIPT_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_teacher as teacher_entry


def main() -> int:
    args = teacher_entry.parse_args()
    args.noise_type = "none"
    args.noise_std = 0.0
    print("Clean teacher: noise_type=none, noise_std=0.0")
    teacher_entry.train_classification(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
