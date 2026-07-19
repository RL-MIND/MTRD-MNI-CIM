"""VGG16 and differentiable paper-equation weight perturbations."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.noisy_layers import pcm_layerwise_wmax

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - only for older PyTorch environments
    from torch.nn.utils.stateless import functional_call


CHANNELS = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 256, 128)


class PaperVGG16(nn.Module):
    """VGG16 used by the released legacy checkpoints.

    `lonear*` intentionally preserves the historical misspelling in checkpoint
    keys.  The architecture has 13 convolutional layers, batch normalization,
    five pooling stages, and three fully connected layers.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        c = CHANNELS
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        for index, (in_channels, out_channels) in enumerate(
            zip((3,) + c[:12], c[:13]), start=1
        ):
            setattr(self, f"conv{index}", nn.Conv2d(in_channels, out_channels, 3, padding=1))
            setattr(self, f"bn{index}", nn.BatchNorm2d(out_channels))
        self.avgpool = nn.AvgPool2d(2)
        self.lonear1 = nn.Linear(c[12], c[13])
        self.lbn1 = nn.BatchNorm1d(c[13])
        self.lonear2 = nn.Linear(c[13], c[14])
        self.lbn2 = nn.BatchNorm1d(c[14])
        self.lonear3 = nn.Linear(c[14], num_classes)

    def _block(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for index in range(start, end + 1):
            x = getattr(self, f"conv{index}")(x)
            x = getattr(self, f"bn{index}")(x)
            x = self.relu(x)
        return self.maxpool2d(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._block(x, 1, 2)
        x = self._block(x, 3, 4)
        x = self._block(x, 5, 7)
        x = self._block(x, 8, 10)
        for index in range(11, 14):
            x = getattr(self, f"conv{index}")(x)
            x = getattr(self, f"bn{index}")(x)
            x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.lbn1(self.lonear1(x)))
        x = self.relu(self.lbn2(self.lonear2(x)))
        return self.lonear3(x)


def _perturbed_parameters(
    model: nn.Module,
    device_type: str,
    noise: float,
) -> dict[str, torch.Tensor]:
    if device_type not in {"rram", "pcm"}:
        raise ValueError(f"unsupported device type: {device_type}")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and non-negative")
    if noise == 0:
        return {}

    modules = dict(model.named_modules())
    replacements: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        owner_name, _, local_name = name.rpartition(".")
        owner = modules[owner_name]
        if local_name != "weight" or not isinstance(owner, (nn.Conv2d, nn.Linear)):
            continue
        if device_type == "rram":
            replacements[name] = parameter * torch.exp(torch.randn_like(parameter) * noise)
        else:
            maximum = pcm_layerwise_wmax(parameter)
            replacements[name] = parameter + torch.randn_like(parameter) * (noise * maximum)
    return replacements


def noisy_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    device_type: str,
    noise: float,
) -> torch.Tensor:
    """Forward with a fresh independent noise realization for every layer."""
    replacements = _perturbed_parameters(model, device_type, noise)
    if not replacements:
        return model(inputs)
    return functional_call(model, replacements, (inputs,), strict=False)


def frozen_perturbed_model(
    model: nn.Module,
    device_type: str,
    noise: float,
) -> nn.Module:
    """Draw one device realization and keep it fixed for a complete trial."""
    result = copy.deepcopy(model)
    replacements = _perturbed_parameters(result, device_type, noise)
    parameters = dict(result.named_parameters())
    with torch.no_grad():
        for name, value in replacements.items():
            parameters[name].copy_(value)
    return result


def quantize_weight_tensor(weight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("weight quantization requires at least 2 bits")
    maximum = weight.detach().abs().amax()
    if not torch.isfinite(maximum) or maximum == 0:
        return weight.detach().clone()
    qmax = 2 ** (bits - 1) - 1
    scale = maximum / qmax
    return torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale


def quantized_model(model: nn.Module, bits: int) -> nn.Module:
    result = copy.deepcopy(model)
    with torch.no_grad():
        for module in result.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(quantize_weight_tensor(module.weight, bits))
    return result


def extract_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping) and candidate:
                payload = candidate
                break
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("checkpoint does not contain a non-empty state dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
        raise TypeError("checkpoint state dict must map string keys to tensors")
    state = dict(payload)
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def canonicalize_checkpoint_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the public modular VGG key layout to ``PaperVGG16`` keys.

    The historical repository uses ``conv1``/``lonear1`` names, while the
    cleaned training package wraps the same tensors below
    ``conv_layers``/``fc1.linear``. The mapping is one-to-one and never drops
    tensors; the final model load remains strict.
    """
    paper_markers = (
        re.compile(r"^conv\d+\."),
        re.compile(r"^bn\d+\."),
        re.compile(r"^lonear\d+\."),
        re.compile(r"^lbn\d+\."),
    )
    modular_prefixes = (
        "conv_layers.", "bn_layers.", "fc1.", "fc2.", "fc3.",
        "fbn1.", "fbn2.",
    )
    has_paper_keys = any(
        any(pattern.match(key) for pattern in paper_markers) for key in state
    )
    has_modular_keys = any(key.startswith(modular_prefixes) for key in state)
    if has_paper_keys and has_modular_keys:
        raise ValueError("checkpoint mixes historical and modular VGG key layouts")
    if not has_modular_keys:
        return dict(state)

    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = re.fullmatch(r"conv_layers\.(\d+)\.conv\.(weight|bias)", key)
        if match:
            target = f"conv{int(match.group(1)) + 1}.{match.group(2)}"
        else:
            match = re.fullmatch(r"bn_layers\.(\d+)\.(.+)", key)
            if match:
                target = f"bn{int(match.group(1)) + 1}.{match.group(2)}"
            elif key.startswith("fc1.linear."):
                target = key.replace("fc1.linear.", "lonear1.", 1)
            elif key.startswith("fc2.linear."):
                target = key.replace("fc2.linear.", "lonear2.", 1)
            elif key.startswith("fc3.linear."):
                target = key.replace("fc3.linear.", "lonear3.", 1)
            elif key.startswith("fbn1."):
                target = key.replace("fbn1.", "lbn1.", 1)
            elif key.startswith("fbn2."):
                target = key.replace("fbn2.", "lbn2.", 1)
            else:
                target = key
        if target in converted:
            raise ValueError(f"checkpoint key mapping collision: {key} -> {target}")
        converted[target] = value
    if len(converted) != len(state):
        raise RuntimeError("checkpoint key mapping changed the state entry count")
    return converted


def load_checkpoint_strict(model: nn.Module, path: str, map_location: object = "cpu") -> None:
    payload = torch.load(path, map_location=map_location)
    state = canonicalize_checkpoint_state_dict(extract_state_dict(payload))
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor is non-finite: {name}")
    model.load_state_dict(state, strict=True)
