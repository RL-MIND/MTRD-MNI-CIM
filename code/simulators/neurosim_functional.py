"""Functional DNN+NeuroSim adapters for the paper's RRAM equation.

The upstream DNN+NeuroSim release separates functional inference from its C++
PPA estimator. This module supplies the missing model-independent functional
programming stage used by the public CIM workflows. It never reports the
functional result as a C++ PPA measurement.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ADAPTER_ID = "mtrd-neurosim-functional-v1"
MTRD_PAPER_EQ1_LOGNORMAL = True
PAPER_RRAM_EQUATION = "W_g = W_nominal * exp(theta), theta ~ N(0, sigma^2)"
SUPPORTED_MODELS = ("vgg16", "unet", "dncnn")
SUPPORTED_SCOPES = ("fixed_trial",)
MAPPABLE_WEIGHTED_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
UNSUPPORTED_WEIGHTED_TYPES = (
    nn.Conv1d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose3d,
    nn.Bilinear,
)


@dataclass(frozen=True)
class FunctionalProfile:
    """A complete functional mapping policy."""

    profile_id: str
    quantize_weights: bool
    quantize_activations: bool
    include_conv_transpose: bool
    implements_released_legacy_mapping: bool
    complete_weighted_operator_coverage: bool
    profile_author_verified: bool
    quantization_calibration: str | None
    bias_mapping: str
    uses_upstream_cim_array_kernel: bool


PROFILES = {
    "paper-8bit": FunctionalProfile(
        profile_id="paper-8bit",
        quantize_weights=True,
        quantize_activations=True,
        include_conv_transpose=True,
        implements_released_legacy_mapping=False,
        complete_weighted_operator_coverage=True,
        profile_author_verified=False,
        quantization_calibration="dynamic absolute maximum for each mapped tensor",
        bias_mapping="digital after ADC-output fake quantization",
        uses_upstream_cim_array_kernel=False,
    ),
    "released-legacy": FunctionalProfile(
        profile_id="released-legacy",
        quantize_weights=False,
        quantize_activations=False,
        include_conv_transpose=False,
        implements_released_legacy_mapping=True,
        complete_weighted_operator_coverage=False,
        profile_author_verified=False,
        quantization_calibration=None,
        bias_mapping="original digital bias path",
        uses_upstream_cim_array_kernel=False,
    ),
}


def adapter_status(model: str) -> dict[str, object]:
    """Return the built-in adapter contract without inspecting NeuroSim files."""
    normalized = str(model).lower()
    return {
        "adapter_id": ADAPTER_ID,
        "model": normalized,
        "supported": normalized in SUPPORTED_MODELS,
        "supported_models": list(SUPPORTED_MODELS),
        "supported_profiles": sorted(PROFILES),
        "supported_realization_scopes": list(SUPPORTED_SCOPES),
        "paper_equation_marker": "MTRD_PAPER_EQ1_LOGNORMAL",
        "paper_equation": PAPER_RRAM_EQUATION,
        "fixed_trial_programming": True,
        "functional_execution_engine": "source-gated-pytorch-extension",
        "neurosim_source_gate_only": True,
        "upstream_native_cim_array_kernel_used": False,
    }


def _positive_bits(value: int, name: str) -> int:
    bits = int(value)
    if bits < 2 or bits > 16:
        raise ValueError(f"{name} must be in [2, 16], found {bits}")
    return bits


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return result


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def symmetric_fake_quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Apply deterministic signed per-tensor fake quantization."""
    bits = _positive_bits(bits, "activation bits")
    if tensor.numel() == 0:
        return tensor
    maximum = tensor.detach().abs().amax()
    if not torch.isfinite(maximum):
        raise ValueError("activation quantization requires finite values")
    if maximum.item() == 0.0:
        return tensor
    bound = float(2 ** (bits - 1) - 1)
    scale = maximum / bound
    return torch.clamp(torch.round(tensor / scale), -bound, bound) * scale


def _weight_fake_quantize(
    weight: torch.Tensor, bits: int, channel_axis: int,
) -> torch.Tensor:
    """Apply signed per-output-channel weight fake quantization."""
    bits = _positive_bits(bits, "weight bits")
    if weight.numel() == 0 or not torch.isfinite(weight).all():
        raise ValueError("weight quantization requires a non-empty finite tensor")
    axis = int(channel_axis) % weight.ndim
    reduce_dimensions = tuple(index for index in range(weight.ndim) if index != axis)
    maximum = weight.detach().abs().amax(dim=reduce_dimensions, keepdim=True)
    bound = float(2 ** (bits - 1) - 1)
    scale = maximum / bound
    safe_scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    quantized = torch.clamp(torch.round(weight / safe_scale), -bound, bound) * safe_scale
    return torch.where(scale == 0.0, torch.zeros_like(quantized), quantized)


class _FunctionalLayer(nn.Module):
    """Common quantization behavior for functional CIM layers."""

    def __init__(self, *, dac_bits: int | None, adc_bits: int | None):
        super().__init__()
        self.dac_bits = dac_bits
        self.adc_bits = adc_bits

    def _input(self, value: torch.Tensor) -> torch.Tensor:
        return value if self.dac_bits is None else symmetric_fake_quantize(value, self.dac_bits)

    def _output(self, value: torch.Tensor) -> torch.Tensor:
        return value if self.adc_bits is None else symmetric_fake_quantize(value, self.adc_bits)


class FunctionalCIMConv2d(_FunctionalLayer):
    """Conv2d using one fixed programmed RRAM weight realization."""

    def __init__(
        self, source: nn.Conv2d, programmed_weight: torch.Tensor,
        *, dac_bits: int | None, adc_bits: int | None,
    ):
        super().__init__(dac_bits=dac_bits, adc_bits=adc_bits)
        self.register_buffer("weight", programmed_weight.detach().clone())
        self.register_buffer(
            "bias", None if source.bias is None else source.bias.detach().clone(),
        )
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        analog = F.conv2d(
            self._input(value), self.weight, None, self.stride,
            self.padding, self.dilation, self.groups,
        )
        output = self._output(analog)
        return output if self.bias is None else output + self.bias.view(1, -1, 1, 1)


class FunctionalCIMConvTranspose2d(_FunctionalLayer):
    """ConvTranspose2d extension with the same fixed programmed realization."""

    def __init__(
        self, source: nn.ConvTranspose2d, programmed_weight: torch.Tensor,
        *, dac_bits: int | None, adc_bits: int | None,
    ):
        super().__init__(dac_bits=dac_bits, adc_bits=adc_bits)
        self.register_buffer("weight", programmed_weight.detach().clone())
        self.register_buffer(
            "bias", None if source.bias is None else source.bias.detach().clone(),
        )
        self.stride = source.stride
        self.padding = source.padding
        self.output_padding = source.output_padding
        self.groups = source.groups
        self.dilation = source.dilation

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        analog = F.conv_transpose2d(
            self._input(value), self.weight, None, self.stride,
            self.padding, self.output_padding, self.groups, self.dilation,
        )
        output = self._output(analog)
        return output if self.bias is None else output + self.bias.view(1, -1, 1, 1)


class FunctionalCIMLinear(_FunctionalLayer):
    """Linear layer using one fixed programmed RRAM weight realization."""

    def __init__(
        self, source: nn.Linear, programmed_weight: torch.Tensor,
        *, dac_bits: int | None, adc_bits: int | None,
    ):
        super().__init__(dac_bits=dac_bits, adc_bits=adc_bits)
        self.register_buffer("weight", programmed_weight.detach().clone())
        self.register_buffer(
            "bias", None if source.bias is None else source.bias.detach().clone(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        analog = F.linear(self._input(value), self.weight, None)
        output = self._output(analog)
        return output if self.bias is None else output + self.bias


def _program_weight(
    weight: torch.Tensor, *, sigma: float, bits: int, quantize: bool,
    channel_axis: int, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nominal = weight.detach().cpu().clone()
    if nominal.numel() == 0 or not torch.isfinite(nominal).all():
        raise ValueError("RRAM programming requires a non-empty finite weight tensor")
    mapped = (
        _weight_fake_quantize(nominal, bits, channel_axis)
        if quantize else nominal
    )
    theta = torch.randn(
        mapped.shape, dtype=mapped.dtype, device="cpu", generator=generator,
    ) * sigma
    programmed = mapped * torch.exp(theta)
    if not torch.isfinite(programmed).all():
        raise ValueError("RRAM programming produced non-finite weights")
    return nominal, mapped, programmed, theta


def program_rram_fixed_trial(
    model: nn.Module,
    *,
    model_name: str,
    sigma: float,
    seed: int,
    profile: str = "paper-8bit",
    weight_bits: int = 8,
    dac_bits: int = 8,
    adc_bits: int = 8,
    preserve_standard_modules: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Create one immutable Eq. (1) realization and its audit manifest."""
    normalized_model = str(model_name).lower()
    if normalized_model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported NeuroSim functional model: {model_name}")
    if profile not in PROFILES:
        raise ValueError(
            f"unknown NeuroSim functional profile {profile!r}; "
            f"expected one of {sorted(PROFILES)}"
        )
    selected = PROFILES[profile]
    sigma = _finite_nonnegative(sigma, "sigma")
    weight_bits = _positive_bits(weight_bits, "weight bits")
    dac_bits = _positive_bits(dac_bits, "DAC bits")
    adc_bits = _positive_bits(adc_bits, "ADC bits")
    seed = int(seed)
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in [0, 2^63)")

    programmed = copy.deepcopy(model).cpu().eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layer_records: list[dict[str, Any]] = []
    programmed_names: set[str] = set()
    theta_count = 0
    theta_sum = 0.0
    theta_square_sum = 0.0

    def ordered_children(parent: nn.Module, prefix: str) -> list[tuple[str, nn.Module]]:
        children = list(parent.named_children())
        if (
            not prefix
            and normalized_model == "unet"
            and selected.implements_released_legacy_mapping
        ):
            source_order = {
                "ups_conv": 0,
                "downs": 1,
                "bottleneck": 2,
                "final_conv": 3,
            }
            children.sort(key=lambda item: source_order.get(item[0], 4))
        return children

    def replace(parent: nn.Module, prefix: str = "") -> None:
        nonlocal theta_count, theta_sum, theta_square_sum
        for child_name, child in ordered_children(parent, prefix):
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            supported = isinstance(child, (nn.Conv2d, nn.Linear)) or (
                selected.include_conv_transpose and isinstance(child, nn.ConvTranspose2d)
            )
            if not supported:
                replace(child, qualified)
                continue

            if isinstance(child, nn.ConvTranspose2d):
                if child.groups != 1:
                    raise RuntimeError(
                        "grouped ConvTranspose2d is not supported by the functional mapping"
                    )
                channel_axis = 1
                layer_type = "ConvTranspose2d"
            elif isinstance(child, nn.Conv2d):
                channel_axis = 0
                layer_type = "Conv2d"
            else:
                channel_axis = 0
                layer_type = "Linear"
            nominal, mapped, noisy, theta = _program_weight(
                child.weight,
                sigma=sigma,
                bits=weight_bits,
                quantize=selected.quantize_weights,
                channel_axis=channel_axis,
                generator=generator,
            )
            theta64 = theta.double()
            theta_count += theta.numel()
            theta_sum += float(theta64.sum())
            theta_square_sum += float((theta64 * theta64).sum())
            activation_bits = dac_bits if selected.quantize_activations else None
            output_bits = adc_bits if selected.quantize_activations else None
            if preserve_standard_modules:
                if selected.quantize_activations:
                    raise ValueError(
                        "preserve_standard_modules cannot represent functional "
                        "DAC/ADC fake quantization"
                    )
                with torch.no_grad():
                    child.weight.copy_(noisy)
            else:
                if isinstance(child, nn.ConvTranspose2d):
                    replacement: nn.Module = FunctionalCIMConvTranspose2d(
                        child, noisy, dac_bits=activation_bits, adc_bits=output_bits,
                    )
                elif isinstance(child, nn.Conv2d):
                    replacement = FunctionalCIMConv2d(
                        child, noisy, dac_bits=activation_bits, adc_bits=output_bits,
                    )
                else:
                    replacement = FunctionalCIMLinear(
                        child, noisy, dac_bits=activation_bits, adc_bits=output_bits,
                    )
                setattr(parent, child_name, replacement)
            programmed_names.add(qualified)
            layer_records.append({
                "name": qualified,
                "type": layer_type,
                "shape": list(nominal.shape),
                "parameter_count": nominal.numel(),
                "nominal_sha256": _tensor_sha256(nominal),
                "mapped_sha256": _tensor_sha256(mapped),
                "programmed_sha256": _tensor_sha256(noisy),
                "theta_mean": float(theta64.mean()) if theta.numel() else 0.0,
                "theta_std_population": float(theta64.std(unbiased=False)) if theta.numel() else 0.0,
            })

    replace(programmed)
    remaining = [
        {"name": name, "type": type(module).__name__}
        for name, module in programmed.named_modules()
        if isinstance(module, MAPPABLE_WEIGHTED_TYPES + UNSUPPORTED_WEIGHTED_TYPES)
        and name not in programmed_names
    ]
    variance = (
        theta_square_sum / theta_count - (theta_sum / theta_count) ** 2
        if theta_count else 0.0
    )
    manifest = {
        "schema": "mtrd.neurosim.functional-programming.v1",
        "adapter_id": ADAPTER_ID,
        "model": normalized_model,
        "equation_marker": "MTRD_PAPER_EQ1_LOGNORMAL",
        "equation": PAPER_RRAM_EQUATION,
        "profile": asdict(selected),
        "realization_scope": "fixed_trial",
        "sigma": sigma,
        "seed": seed,
        "weight_bits": weight_bits if selected.quantize_weights else None,
        "dac_bits": dac_bits if selected.quantize_activations else None,
        "adc_bits": adc_bits if selected.quantize_activations else None,
        "programmed_layer_count": len(layer_records),
        "programmed_parameter_count": theta_count,
        "preserved_standard_modules_for_ptq": bool(preserve_standard_modules),
        "theta_mean": theta_sum / theta_count if theta_count else 0.0,
        "theta_std_population": math.sqrt(max(variance, 0.0)),
        "layers": layer_records,
        "remaining_weighted_operators": remaining,
        "operator_coverage_complete": bool(layer_records) and not remaining,
    }
    if not layer_records:
        raise RuntimeError("the functional mapping did not find any weighted operators")
    if selected.complete_weighted_operator_coverage and remaining:
        raise RuntimeError(
            "paper-8bit NeuroSim mapping left weighted operators unconverted: "
            f"{remaining}"
        )
    return programmed, manifest
