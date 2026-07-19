"""Public Code Ocean entry point for training, evaluation, and simulation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
CAPSULE_ROOT = CODE_ROOT.parent


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _neurosim_root() -> Path:
    configured = os.environ.get("MTRD_NEUROSIM_ROOT") or os.environ.get("NEUROSIM_HOME")
    if configured:
        return Path(configured)
    return Path("/opt/neurosim") if Path("/opt/neurosim").is_dir() else CAPSULE_ROOT / "NeuroSim"


def _runtime_root(environment_variable: str, code_ocean_path: str, local_name: str) -> Path:
    configured = os.environ.get(environment_variable)
    if configured:
        return Path(configured).expanduser().resolve()
    if CODE_ROOT == Path("/code"):
        return Path(code_ocean_path)
    return CAPSULE_ROOT / local_name


def status() -> int:
    """Report installed components without running an experiment."""
    from simulators.neurosim import neurosim_capabilities

    payload = {
        "schema": "mtrd.codeocean.status.v1",
        "code_root": str(CODE_ROOT),
        "data_root": str(_runtime_root("MTRD_DATA_ROOT", "/data", "data")),
        "results_root": str(_runtime_root("MTRD_RESULTS_ROOT", "/results", "results")),
        "python": sys.version.split()[0],
        "packages": {
            name: _version(name)
            for name in ("torch", "torchvision", "numpy", "aihwkit", "h5py")
        },
        "neurosim": neurosim_capabilities(_neurosim_root()),
        "workflows": ["assets", "classification", "denosing", "segmentation", "simulate"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _usage() -> str:
    return """Usage: bash code/run <workflow> [arguments]

Workflows:
  status                Report installed packages and simulator capabilities.
  assets                Validate external CIFAR, Carvana, or Set12 data assets.
  classification        Train or evaluate VGG16 classification workflows.
  denosing              Train or evaluate DnCNN on Set12.
  segmentation          Train or evaluate UNet on Carvana.
  simulate              Build/run NeuroSim PPA or validate the AIHWKit backend.

Run `bash code/run <workflow> --help` for workflow-specific options.
"""


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    workflow, workflow_args = arguments[0], arguments[1:]
    if workflow == "status":
        if workflow_args:
            raise ValueError("status does not accept additional arguments")
        return status()
    if workflow == "assets":
        from capsule.assets import main as workflow_main
    elif workflow == "classification":
        from classification.cli import main as workflow_main
    elif workflow == "denosing":
        from denosing.run import main as workflow_main
    elif workflow == "segmentation":
        from segmentation.run import main as workflow_main
    elif workflow == "simulate":
        from simulators.cli import main as workflow_main
    else:
        raise ValueError(f"Unknown workflow: {workflow}\n\n{_usage()}")
    result = workflow_main(workflow_args)
    return int(result or 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
