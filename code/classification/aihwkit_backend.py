"""AIHWKit implementation of the manuscript's additive PCM equation."""

from __future__ import annotations

import copy
import math
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import torch
import torch.nn as nn

from models.noisy_layers import pcm_layerwise_wmax


PINNED_AIHWKIT_VERSION = "1.1.0"
RealizationScope = Literal["fixed-trial", "per-mac"]
FORMAL_EVALUATION_SCOPES = ("fixed-trial",)
PER_MAC_FORWARD_RNG_REPLAYABLE = False


def require_replayable_realization_scope(realization_scope: RealizationScope) -> None:
    """Reject AIHWKit modes that cannot be replayed from the recorded seed."""
    if realization_scope not in {"fixed-trial", "per-mac"}:
        raise ValueError(f"unsupported realization scope: {realization_scope}")
    if realization_scope == "per-mac":
        raise RuntimeError(
            "AIHWKit 1.1.0 per-MAC forward-noise RNG has no public seed or "
            "state-replay API. Formal evaluation requires realization_scope="
            "fixed-trial; per-mac is available only as a non-replayable "
            "diagnostic in runtime_probe() and validate_noise_scale()."
        )


def installed_version() -> str | None:
    try:
        return version("aihwkit")
    except PackageNotFoundError:
        return None


def runtime_probe() -> dict[str, object]:
    """Exercise imports, Conv2d/Linear conversion, programming, and forward."""
    result: dict[str, object] = {
        "installed_version": installed_version(),
        "required_version": PINNED_AIHWKIT_VERSION,
        "version_match": installed_version() == PINNED_AIHWKIT_VERSION,
        "runtime_ready": False,
        "formal_evaluation_scopes": list(FORMAL_EVALUATION_SCOPES),
        "per_mac_forward_rng_replayable": PER_MAC_FORWARD_RNG_REPLAYABLE,
        "scopes": {},
    }
    if installed_version() is None:
        result["error"] = "AIHWKit distribution is not installed"
        return result

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, bias=True)
            self.fc = nn.Linear(32, 3, bias=True)

        def forward(self, inputs):
            return self.fc(self.conv(inputs).flatten(1))

    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2025)
            model = Probe().eval()
            inputs = torch.randn(2, 1, 6, 6)
            scopes: dict[str, object] = {}
            for index, scope in enumerate(("fixed-trial", "per-mac")):
                analog = convert_model(
                    model, 0.02, input_bits=8, output_bits=8, seed=2025 + index,
                    realization_scope=scope,
                )
                first = analog(inputs)
                second = analog(inputs)
                same_seed_replay_equal: bool | str = "not-applicable"
                if scope == "fixed-trial":
                    replay = convert_model(
                        model, 0.02, input_bits=8, output_bits=8,
                        seed=2025 + index,
                        realization_scope=scope,
                    )
                    same_seed_replay_equal = bool(torch.equal(first, replay(inputs)))
                config = build_additive_config(
                    0.02, input_bits=8, output_bits=8, seed=2025 + index,
                    realization_scope=scope,
                    per_mac_signed_max_ratio=1.0 if scope == "per-mac" else None,
                )
                scopes[scope] = {
                    "output_shape": list(first.shape),
                    "finite": bool(torch.isfinite(first).all() and torch.isfinite(second).all()),
                    "repeated_forward_equal": bool(torch.equal(first, second)),
                    "same_seed_replay_equal": same_seed_replay_equal,
                    "seed_replayable_for_formal_evaluation": scope == "fixed-trial",
                    "mapping_max_input_size": int(config.mapping.max_input_size),
                    "mapping_max_output_size": int(config.mapping.max_output_size),
                    "signed_wmax_ratios": getattr(
                        analog, "_mtrd_pcm_signed_max_ratios", {}
                    ),
                }
        result["scopes"] = scopes
        result["runtime_ready"] = bool(
            result["version_match"]
            and all(item["finite"] for item in scopes.values())
            and scopes["fixed-trial"]["repeated_forward_equal"]
            and scopes["fixed-trial"]["same_seed_replay_equal"]
            and not scopes["per-mac"]["repeated_forward_equal"]
            and bool(scopes["per-mac"]["signed_wmax_ratios"])
            and all(
                item["mapping_max_input_size"] == 0
                and item["mapping_max_output_size"] == 0
                for item in scopes.values()
            )
        )
    except Exception as error:  # preflight reports broken compiled/runtime symbols
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _fixed_programming_noise_model(eta: float):
    """Create an AIHWKit noise model that samples Eq. (2) at programming."""
    from aihwkit.inference.noise.base import BaseNoiseModel

    class LayerwiseAdditiveProgrammingNoise(BaseNoiseModel):
        def __init__(self, coefficient: float):
            super().__init__()
            self.eta = float(coefficient)

        @torch.no_grad()
        def apply_programming_noise(self, weights):
            maximum = pcm_layerwise_wmax(weights)
            programmed = weights + torch.randn_like(weights) * (self.eta * maximum)
            return programmed, [None]

        def apply_programming_noise_to_conductance(self, g_target):  # pragma: no cover
            raise RuntimeError("direct conductance programming is not used by this model")

        def apply_drift_noise_to_conductance(  # pragma: no cover
            self, g_prog, drift_noise_param, t_inference
        ):
            del drift_noise_param, t_inference
            return g_prog

    return LayerwiseAdditiveProgrammingNoise(eta)


def build_additive_config(
    eta: float,
    *,
    input_bits: int | None = 8,
    output_bits: int | None = 8,
    seed: int = 2025,
    realization_scope: RealizationScope,
    per_mac_signed_max_ratio: float | None = None,
):
    """Build the AIHWKit configuration for manuscript PCM Eq. (2).

    AIHWKit 1.1 internally maps ``max(abs(W))`` to one. For per-MAC noise, the
    caller must therefore supply ``max(W) / max(abs(W))`` for the specific
    logical layer. The resulting model-space standard deviation is exactly
    ``eta * max(W)``. ``convert_model`` computes this ratio independently for
    every Conv2d and Linear layer through AIHWKit's per-module config callback.
    """
    if not math.isfinite(float(eta)) or eta < 0:
        raise ValueError("eta must be non-negative")
    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.simulator.parameters.enums import (
        BoundManagementType,
        NoiseManagementType,
        WeightNoiseType,
    )

    config = InferenceRPUConfig()
    if hasattr(config.device, "construction_seed"):
        config.device.construction_seed = int(seed)
    if realization_scope not in {"fixed-trial", "per-mac"}:
        raise ValueError(f"unsupported realization scope: {realization_scope}")
    config.mapping.weight_scaling_omega = 1.0
    config.mapping.weight_scaling_columnwise = False
    config.mapping.digital_bias = True
    # Eq. (2) defines Wmax over the complete logical layer. AIHWKit's mapping
    # uses absmax internally, so tile splitting would change the compensation
    # ratio and is disabled.
    config.mapping.max_input_size = 0
    config.mapping.max_output_size = 0
    config.forward.is_perfect = False
    if realization_scope == "per-mac":
        if per_mac_signed_max_ratio is None:
            raise ValueError(
                "per-mac PCM Eq. (2) requires the layer-specific "
                "max(W)/max(abs(W)) ratio"
            )
        ratio = float(per_mac_signed_max_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("per_mac_signed_max_ratio must be finite and in [0,1]")
        config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        config.forward.w_noise = float(eta) * ratio
    else:
        config.forward.w_noise_type = WeightNoiseType.NONE
        config.forward.w_noise = 0.0
        config.noise_model = _fixed_programming_noise_model(eta)
        config.drift_compensation = None
    config.forward.inp_noise = 0.0
    config.forward.out_noise = 0.0
    config.forward.inp_bound = 1.0
    config.forward.out_bound = 12.0
    config.forward.inp_sto_round = False
    config.forward.out_sto_round = False
    config.forward.out_scale = 1.0
    config.forward.bound_management = BoundManagementType.ITERATIVE
    config.forward.noise_management = NoiseManagementType.ABS_MAX
    config.drift_compensation = None
    for bits, attribute, label in (
        (input_bits, "inp_res", "input"),
        (output_bits, "out_res", "output"),
    ):
        if bits is None:
            resolution = -1.0
        else:
            if bits < 2:
                raise ValueError(f"{label} quantization requires at least 2 bits")
            resolution = 1.0 / (2 ** bits - 2)
        setattr(config.forward, attribute, resolution)
    return config


def convert_model(
    model: nn.Module,
    eta: float,
    *,
    input_bits: int = 8,
    output_bits: int = 8,
    seed: int = 2025,
    realization_scope: RealizationScope,
) -> nn.Module:
    unsupported = [
        name or "<root>"
        for name, module in model.named_modules()
        if isinstance(module, nn.ConvTranspose2d)
    ]
    if unsupported:
        raise TypeError(
            "AIHWKit 1.1 does not convert ConvTranspose2d. Refusing a mixed "
            f"digital/analog result for weighted operators: {unsupported}"
        )
    from aihwkit.nn.conversion import convert_to_analog

    config = build_additive_config(
        eta,
        input_bits=input_bits,
        output_bits=output_bits,
        seed=seed,
        realization_scope=realization_scope,
        # The per-layer callback below replaces this neutral base value.
        per_mac_signed_max_ratio=0.0 if realization_scope == "per-mac" else None,
    )
    signed_max_ratios: dict[str, float] = {}

    def layerwise_config(module_name, module, layer_config):
        if realization_scope != "per-mac" or not isinstance(
            module, (nn.Conv2d, nn.Linear)
        ):
            return layer_config
        maximum = pcm_layerwise_wmax(module.weight).item()
        absolute_maximum = module.weight.detach().abs().amax().item()
        ratio = 0.0 if absolute_maximum == 0.0 else maximum / absolute_maximum
        layer_config.forward.w_noise = float(eta) * ratio
        signed_max_ratios[module_name or "<root>"] = ratio
        return layer_config

    # Conversion and programming use PyTorch RNG internally. Isolate both from
    # ambient caller state. This fully controls fixed-trial programming. The
    # AIHWKit 1.1 per-MAC forward RNG is internal and cannot be replayed from
    # this seed, so formal evaluators reject that scope.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        analog = convert_to_analog(
            copy.deepcopy(model).cpu(),
            config,
            specific_rpu_config_fun=layerwise_config,
        )
        analog.eval()
        if realization_scope == "fixed-trial":
            from aihwkit.inference.utils import program_analog_weights

            program_analog_weights(analog, config.noise_model)
    analog._mtrd_pcm_signed_max_ratios = signed_max_ratios
    return analog


def validate_noise_scale(
    eta: float,
    *,
    samples: int = 4096,
    seed: int = 2025,
    realization_scope: RealizationScope,
) -> dict[str, float | int | str | bool]:
    """Verify the implemented noise scale without changing caller RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _validate_noise_scale_seeded(
            eta,
            samples=samples,
            seed=seed,
            realization_scope=realization_scope,
        )


def _validate_noise_scale_seeded(
    eta: float,
    *,
    samples: int,
    seed: int,
    realization_scope: RealizationScope,
) -> dict[str, float | int | str | bool]:
    """Empirically verify AIHWKit's effective additive weight-noise scale."""
    if eta <= 0:
        raise ValueError("eta must be positive")
    if samples < 256:
        raise ValueError("noise-scale validation requires at least 256 samples")
    from aihwkit.nn import AnalogLinear

    # The largest-magnitude weight is negative, so this fixture distinguishes
    # manuscript max(W)=0.5 from symmetric-quantization max(abs(W))=2.0.
    weights = torch.tensor([[-2.0, 0.5, -0.25, 0.125], [0.4, -0.2, 0.1, 0.05]])
    signed_maximum = pcm_layerwise_wmax(weights).item()
    absolute_maximum = weights.abs().amax().item()
    signed_max_ratio = signed_maximum / absolute_maximum

    config = build_additive_config(
        eta,
        input_bits=None,
        output_bits=None,
        seed=seed,
        realization_scope=realization_scope,
        per_mac_signed_max_ratio=(
            signed_max_ratio if realization_scope == "per-mac" else None
        ),
    )
    layer = AnalogLinear(4, 2, bias=False, rpu_config=config)
    layer.set_weights(weights)
    layer.eval()
    probe = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    observations = []
    frozen_forward_equal: bool | str = "not-applicable"
    if realization_scope == "per-mac":
        # Separate forwards are required: a repeated batch cannot establish
        # whether AIHWKit samples one error per MAC or broadcasts one error.
        for _ in range(samples):
            observations.append(layer(probe)[0, 0].detach().float())
    else:
        from aihwkit.inference.utils import program_analog_weights

        for repeat in range(samples):
            program_analog_weights(layer, config.noise_model)
            observations.append(layer(probe)[0, 0].detach().float())
            if repeat == 0:
                first = layer(probe).detach().clone()
                second = layer(probe).detach().clone()
                frozen_forward_equal = bool(torch.equal(first, second))
    outputs = torch.stack(observations)
    measured_mean = outputs.mean().item()
    measured_std = outputs.std(unbiased=True).item()
    expected_std = eta * signed_maximum
    relative_error = abs(measured_std - expected_std) / expected_std
    return {
        "backend": "aihwkit-additive-constant",
        "aihwkit_version": installed_version() or "missing",
        "eta": eta,
        "realization_scope": realization_scope,
        "seed_replayable_for_formal_evaluation": realization_scope == "fixed-trial",
        "samples": samples,
        "seed": seed,
        "measured_mean": measured_mean,
        "measured_std": measured_std,
        "expected_std": expected_std,
        "signed_wmax": signed_maximum,
        "absolute_weight_max": absolute_maximum,
        "signed_max_to_absmax_ratio": signed_max_ratio,
        "relative_std_error": relative_error,
        "frozen_forward_equal": frozen_forward_equal,
        "mapping_max_input_size": int(config.mapping.max_input_size),
        "mapping_max_output_size": int(config.mapping.max_output_size),
        "mapping_scope": "one_unsplit_logical_layer",
        "pcm_wmax_definition": "signed maximum weight value max(W)",
        "aihwkit_internal_mapping": "max(abs(W)) with signed-max compensation",
        "passed": (
            relative_error <= 0.10
            and abs(measured_mean - weights[0, 0].item()) <= 0.03
            and config.mapping.max_input_size == 0
            and config.mapping.max_output_size == 0
            and (realization_scope != "fixed-trial" or frozen_forward_equal is True)
        ),
    }
