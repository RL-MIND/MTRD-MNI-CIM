#!/usr/bin/env python3
"""Run the shared denoising and segmentation workflow implementation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_workflows.workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
