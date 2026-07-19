"""Command-line interface for the public simulator APIs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .neurosim import (
    NeuroSimLayerFiles,
    NeuroSimPPARequest,
    build_neurosim,
    neurosim_capabilities,
    require_functional_adapter,
    run_neurosim_ppa,
    run_neurosim_smoke,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CAPSULE_ROOT = CODE_ROOT.parent


def _default_neurosim_root() -> Path:
    configured = os.environ.get("MTRD_NEUROSIM_ROOT") or os.environ.get("NEUROSIM_HOME")
    if configured:
        return Path(configured)
    if Path("/opt/neurosim").is_dir():
        return Path("/opt/neurosim")
    return CAPSULE_ROOT / "NeuroSim"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the supported NeuroSim and AIHWKit interfaces.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="report simulator versions and capabilities")
    status.add_argument("--neurosim-root", type=Path, default=_default_neurosim_root())

    build = subparsers.add_parser("neurosim-build", help="compile the pinned NeuroSim C++ PPA binary")
    build.add_argument("--neurosim-root", type=Path, default=_default_neurosim_root())
    build.add_argument("--jobs", type=int)

    ppa = subparsers.add_parser("neurosim-ppa", help="run validated C++ PPA inputs")
    ppa.add_argument("--neurosim-root", type=Path, default=_default_neurosim_root())
    ppa.add_argument("--network-csv", type=Path, required=True)
    ppa.add_argument(
        "--layer",
        action="append",
        nargs=2,
        metavar=("WEIGHT_CSV", "INPUT_CSV"),
        required=True,
        help="repeat once per network.csv row, in layer order",
    )
    ppa.add_argument("--synapse-bits", type=int, default=8)
    ppa.add_argument("--input-bits", type=int, default=8)
    ppa.add_argument("--subarray-rows", type=int, default=128)
    ppa.add_argument("--parallel-rows", type=int, default=128)
    ppa.add_argument("--timeout-seconds", type=int, default=300)
    ppa.add_argument("--output-dir", type=Path, required=True)

    smoke = subparsers.add_parser(
        "neurosim-smoke",
        help="run a deterministic 64x128 fixture through the C++ PPA binary",
    )
    smoke.add_argument("--neurosim-root", type=Path, default=_default_neurosim_root())
    smoke.add_argument("--timeout-seconds", type=int, default=60)
    smoke.add_argument("--output-dir", type=Path, required=True)

    functional = subparsers.add_parser(
        "neurosim-functional",
        help="check the fail-closed status of the functional metric adapter",
    )
    functional.add_argument("--neurosim-root", type=Path, default=_default_neurosim_root())
    functional.add_argument("--model", choices=("vgg16", "unet", "dncnn"), required=True)

    probe = subparsers.add_parser("aihwkit-probe", help="validate the pinned AIHWKit runtime")
    probe.set_defaults()

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code = 0
    if args.command == "status":
        from .aihwkit import runtime_probe

        payload = {
            "neurosim": neurosim_capabilities(args.neurosim_root),
            "aihwkit": runtime_probe(),
        }
    elif args.command == "neurosim-build":
        binary = build_neurosim(args.neurosim_root, jobs=args.jobs)
        payload = {"binary": str(binary), "capabilities": neurosim_capabilities(args.neurosim_root)}
    elif args.command == "neurosim-ppa":
        request = NeuroSimPPARequest(
            network_csv=args.network_csv,
            layers=tuple(
                NeuroSimLayerFiles(Path(weight), Path(inputs))
                for weight, inputs in args.layer
            ),
            neurosim_root=args.neurosim_root,
            synapse_bits=args.synapse_bits,
            input_bits=args.input_bits,
            subarray_rows=args.subarray_rows,
            parallel_rows=args.parallel_rows,
            timeout_seconds=args.timeout_seconds,
        )
        payload = run_neurosim_ppa(request, args.output_dir)
    elif args.command == "neurosim-smoke":
        payload = run_neurosim_smoke(
            args.neurosim_root,
            args.output_dir,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "neurosim-functional":
        payload = require_functional_adapter(args.neurosim_root, args.model)
    elif args.command == "aihwkit-probe":
        from .aihwkit import runtime_probe

        payload = runtime_probe()
        if not payload.get("runtime_ready", False):
            exit_code = 2
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
