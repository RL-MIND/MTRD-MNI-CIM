"""Validated interfaces tied to the pinned DNN+NeuroSim source tree.

The C++ interface reports circuit PPA only. The separately versioned Python
functional interface supplies fixed-trial Eq. (1) programming for supported
paper models and never presents its Accuracy, Dice, or PSNR output as a C++
circuit measurement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


EXPECTED_NEUROSIM_COMMIT = "cddb7d346a9f1fc5a39b6c3abcb378c4b2dfc555"
EXPECTED_NEUROSIM_SOURCE_TREE_SHA256 = (
    "8d2a0a4db81838bf2e9738c3140871cc09966431d3ac8693acdad70c6e281d61"
)
PAPER_RRAM_EQUATION_ID = "paper-eq1-lognormal-multiplicative"
PAPER_RRAM_EQUATION = "W_g = W_nominal * exp(theta), theta ~ N(0, sigma^2)"
PAPER_RRAM_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
FIXED_SUBARRAY_COLUMNS = 128
NETWORK_COLUMN_COUNT = 8
_SOURCE_HASH_EXCLUDED_PARTS = {".git", "__pycache__"}
_SOURCE_HASH_EXCLUDED_SUFFIXES = {".md", ".o", ".pyc"}
_SOURCE_HASH_EXCLUDED_FILES = {"NeuroSIM/.depend", "NeuroSIM/main"}


class FunctionalAdapterUnavailable(RuntimeError):
    """Raised when a caller requests an unsupported functional adapter."""


@dataclass(frozen=True)
class NeuroSimLayerFiles:
    """Weight and activation CSV files for one NeuroSim network layer."""

    weight_csv: Path
    input_csv: Path


@dataclass(frozen=True)
class NeuroSimPPARequest:
    """Complete request for one C++ NeuroSim PPA execution."""

    network_csv: Path
    layers: tuple[NeuroSimLayerFiles, ...]
    neurosim_root: Path
    synapse_bits: int = 8
    input_bits: int = 8
    subarray_rows: int = 128
    parallel_rows: int = 128
    timeout_seconds: int = 300
    expected_commit: str = EXPECTED_NEUROSIM_COMMIT
    expected_source_tree_sha256: str = EXPECTED_NEUROSIM_SOURCE_TREE_SHA256


_METRIC_PATTERNS = {
    "area_um2": r"ChipArea\s*:\s*([0-9.eE+-]+)um\^2",
    "array_area_um2": r"Chip total CIM array\s*:\s*([0-9.eE+-]+)um\^2",
    "clock_period_ns": r"Chip clock period is:\s*([0-9.eE+-]+)ns",
    "latency_ns": (
        r"Chip (?:pipeline-system-clock-cycle|layer-by-layer readLatency) "
        r"\(per image\) is:\s*([0-9.eE+-]+)ns"
    ),
    "dynamic_energy_pj": (
        r"Chip (?:pipeline-system readDynamicEnergy|total readDynamicEnergy)"
        r"(?: \(per image\))? is:\s*([0-9.eE+-]+)pJ"
    ),
    "leakage_energy_pj": (
        r"Chip (?:pipeline-system leakage Energy|total leakage Energy)"
        r"(?: \(per image\))? is:\s*([0-9.eE+-]+)pJ"
    ),
    "energy_efficiency_tops_per_w": r"Energy Efficiency TOPS/W .*:\s*([0-9.eE+-]+)",
    "throughput_tops": r"Throughput TOPS .*:\s*([0-9.eE+-]+)",
    "throughput_fps": r"Throughput FPS .*:\s*([0-9.eE+-]+)",
    "compute_efficiency_tops_per_mm2": (
        r"Compute efficiency TOPS/mm\^2 .*:\s*([0-9.eE+-]+)"
    ),
    "simulation_runtime_seconds": r"Total Run-time of NeuroSim:\s*([0-9.eE+-]+)\s*seconds",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neurosim_source_tree_sha256(root: str | Path) -> str:
    """Hash executable NeuroSim sources while excluding generated files and docs."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"NeuroSim source tree is missing: {base}")
    digest = hashlib.sha256()
    files: list[Path] = []
    for path in base.rglob("*"):
        relative = path.relative_to(base)
        if path.is_symlink():
            raise RuntimeError(f"NeuroSim source tree contains a symlink: {relative}")
        if not path.is_file():
            continue
        if _SOURCE_HASH_EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if path.suffix in _SOURCE_HASH_EXCLUDED_SUFFIXES:
            continue
        if relative.as_posix() in _SOURCE_HASH_EXCLUDED_FILES:
            continue
        files.append(path)
    for path in sorted(files, key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_source_hash(root: Path, expected: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(
            "expected NeuroSim source-tree SHA-256 must be 64 lowercase hex digits"
        )
    actual = neurosim_source_tree_sha256(root)
    if actual != expected:
        raise RuntimeError(
            "NeuroSim source-tree hash mismatch: "
            f"expected {expected}, found {actual}"
        )
    return actual


def _revision(root: Path) -> str:
    marker = root / "NEUROSIM_COMMIT"
    if marker.is_file():
        return marker.read_text(encoding="ascii").strip()
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Cannot determine the NeuroSim source revision. Add NEUROSIM_COMMIT "
            f"or provide a Git checkout: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle) if any(value.strip() for value in row)]


def _numeric_matrix_shape(
    path: Path,
    *,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[int, int]:
    rows = _csv_rows(path)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"CSV matrix is ragged: {path}")
    for row_index, row in enumerate(rows):
        for column_index, raw in enumerate(row):
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(
                    f"CSV contains a non-numeric value at row {row_index}, "
                    f"column {column_index}: {path}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"CSV contains a non-finite value at row {row_index}, "
                    f"column {column_index}: {path}"
                )
            if integer and not value.is_integer():
                raise ValueError(
                    f"CSV contains a non-integer activation at row {row_index}, "
                    f"column {column_index}: {path}"
                )
            if minimum is not None and value < minimum:
                raise ValueError(f"CSV value {value} is below {minimum}: {path}")
            if maximum is not None and value > maximum:
                raise ValueError(f"CSV value {value} is above {maximum}: {path}")
    return len(rows), width


def neurosim_functional_status(root: str | Path, model: str) -> dict[str, object]:
    """Audit whether a NeuroSim tree implements a paper functional adapter.

    The public adapter is enabled only for the pinned, content-verified
    upstream tree. A custom in-tree adapter must include the explicit source
    marker ``MTRD_PAPER_EQ1_LOGNORMAL``. This avoids inferring Eq. (1) support
    merely from unrelated exponential operations in the simulator source.
    """
    normalized = model.lower()
    if normalized not in {"vgg16", "unet", "dncnn"}:
        raise ValueError(f"Unknown paper model: {model}")

    base = Path(root).resolve()
    result: dict[str, object] = {
        "backend": "DNN+NeuroSim-2DInferenceV1.5-dev",
        "root": str(base),
        "model": normalized,
        "required_dataset": {
            "vgg16": "CIFAR10/CIFAR100",
            "unet": "Carvana",
            "dncnn": "Set12",
        }[normalized],
        "required_metric": {
            "vgg16": "top1_accuracy",
            "unet": "dice",
            "dncnn": "psnr_db",
        }[normalized],
        "paper_rram_equation_id": PAPER_RRAM_EQUATION_ID,
        "paper_rram_equation": PAPER_RRAM_EQUATION,
        "paper_rram_grid": list(PAPER_RRAM_GRID),
        "functional_ready": False,
        "reasons": [],
        "warnings": [],
    }
    reasons: list[str] = result["reasons"]  # type: ignore[assignment]
    warnings: list[str] = result["warnings"]  # type: ignore[assignment]
    if not base.is_dir():
        reasons.append("NeuroSim source directory was not found")
        return result

    try:
        revision = _revision(base)
    except RuntimeError as error:
        revision = None
        reasons.append(str(error))
    result["revision"] = revision
    result["expected_revision"] = EXPECTED_NEUROSIM_COMMIT
    result["revision_matches"] = revision == EXPECTED_NEUROSIM_COMMIT
    if revision != EXPECTED_NEUROSIM_COMMIT:
        reasons.append("NeuroSim revision is missing or differs from the pinned commit")

    try:
        source_tree_sha256 = neurosim_source_tree_sha256(base)
        source_tree_error = None
    except (OSError, RuntimeError) as error:
        source_tree_sha256 = None
        source_tree_error = str(error)
    source_tree_matches = source_tree_sha256 == EXPECTED_NEUROSIM_SOURCE_TREE_SHA256
    result.update({
        "source_tree_sha256": source_tree_sha256,
        "source_tree_error": source_tree_error,
        "expected_source_tree_sha256": EXPECTED_NEUROSIM_SOURCE_TREE_SHA256,
        "source_tree_matches": source_tree_matches,
    })

    source_paths = [
        path
        for path in (
            *base.glob("*.py"),
            *base.glob("models/*.py"),
            *base.glob("pytorch-quantization/pytorch_quantization/cim/modules/*.py"),
        )
        if path.is_file()
    ]
    sources = {
        path.relative_to(base).as_posix(): path.read_text(
            encoding="utf-8", errors="replace"
        )
        for path in source_paths
    }
    combined = "\n".join(sources.values()).lower()
    model_patterns = {
        "vgg16": (r"args\.model\s*==\s*['\"]vgg16['\"]", r"def\s+vgg16\b", r"class\s+vgg16\b"),
        "unet": (r"args\.model\s*==\s*['\"]unet['\"]", r"class\s+unet\b"),
        "dncnn": (r"args\.model\s*==\s*['\"]dncnn['\"]", r"class\s+dncnn\b"),
    }
    native_adapter_detected = any(
        re.search(pattern, combined, flags=re.IGNORECASE)
        for pattern in model_patterns[normalized]
    )
    equation_marker_sources = sorted(
        name for name, source in sources.items() if "MTRD_PAPER_EQ1_LOGNORMAL" in source
    )
    native_equation_supported = bool(equation_marker_sources)
    native_gaussian_sampling = "conductance_sampling" in combined and "torch.normal(" in combined

    from .neurosim_functional import adapter_status

    public_adapter = adapter_status(normalized)
    public_adapter_ready = bool(public_adapter["supported"] and source_tree_matches)
    adapter_detected = native_adapter_detected or public_adapter_ready
    equation_supported = native_equation_supported or public_adapter_ready
    if public_adapter_ready:
        equation_marker_sources.append("public:simulators/neurosim_functional.py")

    result.update(
        {
            "native_frontend_vgg8_detected": "vgg8" in combined,
            "native_frontend_resnet18_detected": "resnet18" in combined,
            "native_frontend_cifar100_detected": "cifar100" in combined,
            "native_paper_model_adapter_detected": native_adapter_detected,
            "public_functional_adapter": public_adapter,
            "public_functional_adapter_ready": public_adapter_ready,
            "paper_model_adapter_detected": adapter_detected,
            "paper_equation_marker": "MTRD_PAPER_EQ1_LOGNORMAL",
            "paper_equation_marker_sources": equation_marker_sources,
            "paper_rram_equation_supported": equation_supported,
            "native_device_sampler_detected": native_gaussian_sampling,
            "native_device_sampling_distribution": (
                "gaussian-conductance" if native_gaussian_sampling else None
            ),
            "functional_execution_engine": (
                "upstream-native-cim-array-kernel"
                if native_adapter_detected and native_equation_supported
                else public_adapter.get("functional_execution_engine")
            ),
            "upstream_native_cim_array_kernel_used": bool(
                native_adapter_detected and native_equation_supported
            ),
            "neurosim_source_gate_only": bool(
                public_adapter_ready
                and not (native_adapter_detected and native_equation_supported)
            ),
        }
    )
    if not adapter_detected:
        reasons.append(f"no {normalized} functional adapter was detected")
    if public_adapter["supported"] and not source_tree_matches:
        reasons.append(
            "the built-in adapter requires the canonical pinned NeuroSim source tree"
        )
    if not equation_supported:
        reasons.append(
            "no adapter explicitly implements the paper Eq. (1) lognormal multiplicative model"
        )
    if native_gaussian_sampling:
        warnings.append(
            "the native frontend samples Gaussian conductance states and is not, by itself, "
            "the paper Eq. (1) lognormal weight model"
        )

    native_ready = native_adapter_detected and native_equation_supported
    result["functional_ready"] = bool(
        result["revision_matches"] and (native_ready or public_adapter_ready)
    )
    if result["functional_ready"]:
        result["reasons"] = []
    return result


def neurosim_capabilities(root: str | Path) -> dict[str, object]:
    """Return machine-readable PPA and functional capability identities."""
    root = Path(root).resolve()
    binary = root / "NeuroSIM" / "main"
    revision = None
    revision_error = None
    source_tree_sha256 = None
    source_tree_error = None
    try:
        revision = _revision(root)
    except RuntimeError as error:
        revision_error = str(error)
    try:
        source_tree_sha256 = neurosim_source_tree_sha256(root)
    except (OSError, RuntimeError) as error:
        source_tree_error = str(error)
    revision_matches = revision == EXPECTED_NEUROSIM_COMMIT
    source_tree_matches = source_tree_sha256 == EXPECTED_NEUROSIM_SOURCE_TREE_SHA256
    functional_status = {
        model: neurosim_functional_status(root, model)
        for model in ("vgg16", "unet", "dncnn")
    }
    return {
        "backend": "DNN+NeuroSim-2DInferenceV1.5-dev",
        "root": str(root),
        "revision": revision,
        "revision_error": revision_error,
        "expected_revision": EXPECTED_NEUROSIM_COMMIT,
        "revision_matches": revision_matches,
        "source_tree_sha256": source_tree_sha256,
        "source_tree_error": source_tree_error,
        "expected_source_tree_sha256": EXPECTED_NEUROSIM_SOURCE_TREE_SHA256,
        "source_tree_matches": source_tree_matches,
        "ppa_binary": str(binary),
        "ppa_ready": (
            revision_matches
            and source_tree_matches
            and binary.is_file()
            and os.access(binary, os.X_OK)
        ),
        "fixed_subarray_columns": FIXED_SUBARRAY_COLUMNS,
        "functional_adapters": {
            model: bool(status["functional_ready"])
            for model, status in functional_status.items()
        },
        "functional_status": functional_status,
        "functional_metrics_supported": [
            status["required_metric"]
            for status in functional_status.values()
            if status["functional_ready"]
        ],
        "ppa_metrics_supported": ["area", "latency", "energy", "throughput"],
    }


def require_functional_adapter(root: str | Path, model: str) -> dict[str, object]:
    """Fail closed for functional inference requests without complete support."""
    status = neurosim_functional_status(root, model)
    if not status["functional_ready"]:
        raise FunctionalAdapterUnavailable(
            "The NeuroSim tree cannot produce the requested functional metric:\n"
            + json.dumps(status, indent=2, sort_keys=True)
        )
    return status


def build_neurosim(
    root: str | Path,
    *,
    jobs: int | None = None,
    expected_commit: str = EXPECTED_NEUROSIM_COMMIT,
    expected_source_tree_sha256: str = EXPECTED_NEUROSIM_SOURCE_TREE_SHA256,
) -> Path:
    """Build the C++ PPA executable from the pinned source tree."""
    root = Path(root).resolve()
    revision = _revision(root)
    if expected_commit and revision != expected_commit:
        raise RuntimeError(
            f"NeuroSim revision mismatch: expected {expected_commit}, found {revision}"
        )
    _validated_source_hash(root, expected_source_tree_sha256)
    source = root / "NeuroSIM"
    if not (source / "makefile").is_file():
        raise FileNotFoundError(f"NeuroSim makefile not found: {source / 'makefile'}")
    subprocess.run(["make", "-C", str(source), "clean"], check=True)
    command = ["make", "-C", str(source)]
    if jobs is not None:
        if jobs < 1:
            raise ValueError("jobs must be positive")
        command.append(f"-j{jobs}")
    subprocess.run(command, check=True)
    binary = source / "main"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"NeuroSim build did not create an executable: {binary}")
    return binary


def validate_neurosim_inputs(request: NeuroSimPPARequest) -> dict[str, object]:
    """Validate revision, argument ranges, layer count, and CSV matrix shapes."""
    root = request.neurosim_root.resolve()
    network_csv = request.network_csv.resolve()
    if request.synapse_bits < 1 or request.input_bits < 1:
        raise ValueError("synapse_bits and input_bits must be positive")
    if request.subarray_rows < 1:
        raise ValueError("subarray_rows must be positive")
    if not 1 <= request.parallel_rows <= request.subarray_rows:
        raise ValueError("parallel_rows must be in [1, subarray_rows]")
    if request.timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    revision = _revision(root)
    if request.expected_commit and revision != request.expected_commit:
        raise RuntimeError(
            f"NeuroSim revision mismatch: expected {request.expected_commit}, found {revision}"
        )
    source_tree_sha256 = _validated_source_hash(
        root, request.expected_source_tree_sha256,
    )
    binary = root / "NeuroSIM" / "main"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"NeuroSim executable not found: {binary}. Run the neurosim-build command first."
        )
    if not network_csv.is_file():
        raise FileNotFoundError(network_csv)
    network_rows = _csv_rows(network_csv)
    if not network_rows:
        raise ValueError(f"Network CSV is empty: {network_csv}")
    for index, row in enumerate(network_rows):
        if len(row) != NETWORK_COLUMN_COUNT:
            raise ValueError(
                f"Network row {index} has {len(row)} columns; expected {NETWORK_COLUMN_COUNT}"
            )
        try:
            numeric_values = [float(value) for value in row]
        except ValueError as error:
            raise ValueError(f"Network row {index} contains a non-numeric field") from error
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError(f"Network row {index} contains a non-finite field")
        if any(not value.is_integer() for value in numeric_values):
            raise ValueError(f"Network row {index} must contain integers")
        values = [int(value) for value in numeric_values]
        if any(value <= 0 for value in values[:6]) or values[7] <= 0:
            raise ValueError(f"Network row {index} has invalid dimensions: {values}")
        if values[6] not in {0, 1}:
            raise ValueError(f"Network row {index} max-pool flag must be zero or one")
    if len(request.layers) != len(network_rows):
        raise ValueError(
            f"Network has {len(network_rows)} layers but {len(request.layers)} layer file pairs were supplied"
        )

    layer_files: list[dict[str, object]] = []
    for index, (row, layer) in enumerate(zip(network_rows, request.layers)):
        weight = layer.weight_csv.resolve()
        inputs = layer.input_csv.resolve()
        if not weight.is_file():
            raise FileNotFoundError(weight)
        if not inputs.is_file():
            raise FileNotFoundError(inputs)
        input_height, input_width, input_channels, kernel_height, kernel_width, output_channels, _, stride = [
            int(float(value)) for value in row
        ]
        expected_rows = input_channels * kernel_height * kernel_width
        if input_height < kernel_height or input_width < kernel_width:
            raise ValueError(
                f"Layer {index} kernel exceeds the declared input dimensions"
            )
        height_positions = input_height - kernel_height + 1
        width_positions = input_width - kernel_width + 1
        if height_positions % stride != 0 or width_positions % stride != 0:
            raise ValueError(
                f"Layer {index} requires (input-kernel+1) to be exactly "
                "divisible by stride in both dimensions for the pinned "
                "NeuroSim C++ position calculation"
            )
        expected_positions = (
            (height_positions // stride) * (width_positions // stride)
        )
        weight_shape = _numeric_matrix_shape(weight, minimum=-1.0, maximum=1.0)
        input_shape = _numeric_matrix_shape(
            inputs,
            integer=True,
            minimum=0,
            maximum=2 ** request.input_bits - 1,
        )
        if weight_shape != (expected_rows, output_channels):
            raise ValueError(
                f"Layer {index} weight shape is {weight_shape}; expected "
                f"({expected_rows}, {output_channels})"
            )
        if input_shape != (expected_rows, expected_positions):
            raise ValueError(
                f"Layer {index} input shape is {input_shape}; expected "
                f"({expected_rows}, {expected_positions})"
            )
        layer_files.append(
            {
                "index": index,
                "weight_csv": str(weight),
                "weight_sha256": _sha256(weight),
                "weight_shape": list(weight_shape),
                "input_csv": str(inputs),
                "input_sha256": _sha256(inputs),
                "input_shape": list(input_shape),
            }
        )
    return {
        "revision": revision,
        "source_tree_sha256": source_tree_sha256,
        "expected_source_tree_sha256": request.expected_source_tree_sha256,
        "binary": str(binary),
        "binary_sha256": _sha256(binary),
        "param_cpp_sha256": _sha256(root / "NeuroSIM" / "Param.cpp"),
        "network_csv": str(network_csv),
        "network_sha256": _sha256(network_csv),
        "layer_count": len(network_rows),
        "layers": layer_files,
    }


def parse_neurosim_output(text: str) -> dict[str, float]:
    """Parse stable top-level PPA fields from NeuroSim stdout."""
    metrics: dict[str, float] = {}
    for name, pattern in _METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        metrics[name] = float(match.group(1)) if match else 0.0
    metrics["area_mm2"] = metrics["area_um2"] * 1e-6
    metrics["energy_pj"] = metrics["dynamic_energy_pj"] + metrics["leakage_energy_pj"]
    metrics["edp_pj_ns"] = metrics["energy_pj"] * metrics["latency_ns"]
    return metrics


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _create_fresh_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"NeuroSim output directory already exists and will not be reused: {path}"
        ) from error


def run_neurosim_ppa(
    request: NeuroSimPPARequest,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run one PPA job in a new output directory and publish a success manifest."""
    identity = validate_neurosim_inputs(request)
    output_dir = Path(output_dir).resolve()
    _create_fresh_output_dir(output_dir)
    binary = Path(identity["binary"])
    command = [
        str(binary),
        str(request.network_csv.resolve()),
        str(request.synapse_bits),
        str(request.input_bits),
        str(request.subarray_rows),
        str(request.parallel_rows),
    ]
    for layer in request.layers:
        command.extend([str(layer.weight_csv.resolve()), str(layer.input_csv.resolve())])
    try:
        process = subprocess.run(
            command,
            cwd=request.neurosim_root.resolve(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=request.timeout_seconds,
            check=False,
        )
        raw_output = process.stdout
    except subprocess.TimeoutExpired as error:
        raw_output = error.stdout or ""
        if isinstance(raw_output, bytes):
            raw_output = raw_output.decode("utf-8", errors="replace")
        failed_log = output_dir / "neurosim_failed.log"
        _write_text(failed_log, raw_output)
        raise RuntimeError(
            f"NeuroSim exceeded the {request.timeout_seconds}-second timeout. "
            f"See {failed_log}"
        ) from error
    if process.returncode != 0:
        failed_log = output_dir / "neurosim_failed.log"
        _write_text(failed_log, raw_output)
        raise RuntimeError(
            f"NeuroSim exited with code {process.returncode}. See {failed_log}"
        )
    metrics = parse_neurosim_output(raw_output)
    missing = [
        name for name in ("area_um2", "latency_ns", "dynamic_energy_pj")
        if metrics[name] <= 0.0
    ]
    if missing:
        failed_log = output_dir / "neurosim_failed.log"
        _write_text(failed_log, raw_output)
        raise RuntimeError(
            f"NeuroSim completed without positive required metrics {missing}. "
            f"See {failed_log}"
        )
    raw_log = output_dir / "neurosim_raw.log"
    _write_text(raw_log, raw_output)
    result: dict[str, object] = {
        "schema": "mtrd.neurosim.ppa.v1",
        "backend": "DNN+NeuroSim C++ PPA",
        "functional_inference": False,
        "request": {
            **asdict(request),
            "network_csv": str(request.network_csv.resolve()),
            "neurosim_root": str(request.neurosim_root.resolve()),
            "layers": [
                {
                    "weight_csv": str(layer.weight_csv.resolve()),
                    "input_csv": str(layer.input_csv.resolve()),
                }
                for layer in request.layers
            ],
        },
        "subarray_columns": FIXED_SUBARRAY_COLUMNS,
        "input_identity": identity,
        "command": command,
        "return_code": process.returncode,
        "raw_log": str(raw_log),
        "raw_log_sha256": _sha256(raw_log),
        "metrics": metrics,
    }
    _write_json(output_dir / "neurosim_ppa.json", result)
    return result


def run_neurosim_smoke(
    neurosim_root: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: int = 60,
    expected_source_tree_sha256: str = EXPECTED_NEUROSIM_SOURCE_TREE_SHA256,
) -> dict[str, object]:
    """Run a deterministic one-layer PPA fixture through the real C++ binary.

    The fixture is deliberately large enough to exercise one complete 64x128
    logical matrix. Smaller degenerate matrices are not a reliable NeuroSim
    smoke test. This command validates PPA only and does not perform functional
    neural-network inference.
    """
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    output_dir = Path(output_dir).resolve()
    input_dir = output_dir.with_name(output_dir.name + "-inputs")
    if output_dir.exists():
        raise FileExistsError(
            f"NeuroSim output directory already exists and will not be reused: {output_dir}"
        )
    _create_fresh_output_dir(input_dir)

    network_csv = input_dir / "network.csv"
    weight_csv = input_dir / "weights.csv"
    input_csv = input_dir / "inputs.csv"
    with network_csv.open("w", encoding="ascii", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow((1, 1, 64, 1, 1, 128, 0, 1))
    with weight_csv.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for _ in range(64):
            writer.writerow(("0.5",) * 128)
    with input_csv.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for index in range(64):
            writer.writerow(((index * 37) % 256,))

    request = NeuroSimPPARequest(
        network_csv=network_csv,
        layers=(NeuroSimLayerFiles(weight_csv, input_csv),),
        neurosim_root=Path(neurosim_root),
        synapse_bits=8,
        input_bits=8,
        subarray_rows=64,
        parallel_rows=32,
        timeout_seconds=timeout_seconds,
        expected_source_tree_sha256=expected_source_tree_sha256,
    )
    result = run_neurosim_ppa(request, output_dir)
    result["smoke_test"] = {
        "fixture": "deterministic_64x128_ppa",
        "functional_inference": False,
        "input_dir": str(input_dir),
    }
    _write_json(output_dir / "neurosim_ppa.json", result)
    return result
