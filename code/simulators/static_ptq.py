"""Released PyTorch eager-mode static quantization for CIM evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


def _validate_bits(name: str, value: int) -> int:
    value = int(value)
    if value < 2 or value > 8:
        raise ValueError(f"{name} must be in [2, 8], got {value}")
    return value


def released_static_qconfig(
    *,
    activation_bits: int,
    weight_bits: int,
) -> torch.ao.quantization.QConfig:
    """Build the per-tensor observer configuration from the released scripts."""
    activation_bits = _validate_bits("activation_bits", activation_bits)
    weight_bits = _validate_bits("weight_bits", weight_bits)
    return torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.MinMaxObserver.with_args(
            quant_min=0,
            quant_max=2**activation_bits - 1,
        ),
        weight=torch.ao.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8,
            quant_min=-(2 ** (weight_bits - 1)),
            quant_max=2 ** (weight_bits - 1) - 1,
        ),
    )


def _quantized_module_types(model: nn.Module) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for module in model.modules():
        module_name = type(module).__module__
        if ".quantized" in module_name:
            key = f"{module_name}.{type(module).__name__}"
            counts[key] += 1
    return dict(sorted(counts.items()))


def static_quantize_calibrated(
    model: nn.Module,
    calibration_batches: Iterable[torch.Tensor],
    *,
    activation_bits: int,
    weight_bits: int,
    engine: str = "fbgemm",
) -> tuple[nn.Module, dict[str, Any]]:
    """Calibrate and convert a programmed CPU model using released eager PTQ."""
    activation_bits = _validate_bits("activation_bits", activation_bits)
    weight_bits = _validate_bits("weight_bits", weight_bits)
    supported = tuple(torch.backends.quantized.supported_engines)
    if engine not in supported:
        raise RuntimeError(
            f"quantized engine {engine!r} is unavailable; supported engines: {supported}"
        )
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("static PTQ requires a CPU model")

    torch.backends.quantized.engine = engine
    model = model.eval()
    model.qconfig = released_static_qconfig(
        activation_bits=activation_bits,
        weight_bits=weight_bits,
    )
    prepared = torch.ao.quantization.prepare(model, inplace=False)
    calibration_batch_count = 0
    calibration_sample_count = 0
    with torch.inference_mode():
        for batch in calibration_batches:
            if not isinstance(batch, torch.Tensor) or batch.ndim < 1:
                raise TypeError("calibration batches must be non-scalar tensors")
            batch = batch.cpu()
            prepared(batch, 0.0, "none")
            calibration_batch_count += 1
            calibration_sample_count += int(batch.shape[0])
    if calibration_batch_count == 0:
        raise ValueError("static PTQ calibration requires at least one batch")

    converted = torch.ao.quantization.convert(prepared, inplace=False).eval()
    module_types = _quantized_module_types(converted)
    if not module_types:
        raise RuntimeError("static PTQ did not convert any quantized operators")
    manifest = {
        "schema": "mtrd.released-static-ptq.v1",
        "implementation": "torch.ao.quantization eager static PTQ",
        "torch_version": torch.__version__,
        "quantized_engine": torch.backends.quantized.engine,
        "activation_bits": activation_bits,
        "weight_bits": weight_bits,
        "activation_observer": "MinMaxObserver per tensor unsigned",
        "activation_dtype": "torch.quint8",
        "activation_qscheme": "torch.per_tensor_affine",
        "activation_quant_min": 0,
        "activation_quant_max": 2**activation_bits - 1,
        "weight_observer": "MinMaxObserver per tensor signed",
        "weight_dtype": "torch.qint8",
        "weight_qscheme": "torch.per_tensor_affine",
        "weight_quant_min": -(2 ** (weight_bits - 1)),
        "weight_quant_max": 2 ** (weight_bits - 1) - 1,
        "calibration_batch_count": calibration_batch_count,
        "calibration_sample_count": calibration_sample_count,
        "quantized_module_types": module_types,
    }
    return converted, manifest
