"""Checkpoint compatibility helpers shared by training and evaluation tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


STATE_DICT_KEYS = ("state_dict", "model_state", "model_state_dict", "model", "net")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load trusted project checkpoints across supported PyTorch releases."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return a state dict from either a raw mapping or a common envelope."""
    value = checkpoint
    if isinstance(value, Mapping):
        for key in STATE_DICT_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, Mapping) and candidate:
                value = candidate
                break
    if not isinstance(value, Mapping) or not value:
        raise TypeError("checkpoint does not contain a non-empty state dictionary")

    state = dict(value)
    if not any(torch.is_tensor(item) for item in state.values()):
        raise TypeError("checkpoint mapping contains no tensors")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): item for key, item in state.items()}
    return state


def load_state_dict(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return extract_state_dict(load_checkpoint(path, map_location=map_location))


def state_dict_finiteness(state: Mapping[str, Any]) -> tuple[int, int]:
    nonfinite = 0
    total = 0
    for value in state.values():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        total += value.numel()
        nonfinite += (~torch.isfinite(value)).sum().item()
    return int(nonfinite), int(total)


def remap_legacy_vgg_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the original repository's VGG keys to the unified noisy VGG."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = re.match(r"^conv(\d+)\.(weight|bias)$", key)
        if match:
            key = f"conv_layers.{int(match.group(1)) - 1}.conv.{match.group(2)}"
        else:
            match = re.match(r"^bn(\d+)\.(.+)$", key)
            if match:
                key = f"bn_layers.{int(match.group(1)) - 1}.{match.group(2)}"
            elif key.startswith("lonear1."):
                key = key.replace("lonear1.", "fc1.linear.", 1)
            elif key.startswith("lonear2."):
                key = key.replace("lonear2.", "fc2.linear.", 1)
            elif key.startswith("lonear3."):
                key = key.replace("lonear3.", "fc3.linear.", 1)
            elif key.startswith("lbn1."):
                key = key.replace("lbn1.", "fbn1.", 1)
            elif key.startswith("lbn2."):
                key = key.replace("lbn2.", "fbn2.", 1)
        remapped[key] = value
    return remapped


def remap_legacy_dncnn_state_dict(
    state: Mapping[str, torch.Tensor], num_layers: int = 17,
) -> dict[str, torch.Tensor]:
    """Map explicit or sequential legacy DnCNN keys to the unified model."""
    if num_layers < 3:
        raise ValueError(f"DnCNN requires at least 3 layers, got {num_layers}")
    remapped: dict[str, torch.Tensor] = {}
    last_sequential_index = 3 * num_layers - 4
    for key, value in state.items():
        sequential = re.match(r"^dncnn\.(\d+)\.(.+)$", key)
        if sequential:
            index = int(sequential.group(1))
            suffix = sequential.group(2)
            if index == 0:
                key = f"first_conv.conv.{suffix}"
            elif index == last_sequential_index:
                key = f"last_conv.conv.{suffix}"
            elif index >= 2 and (index - 2) % 3 == 0:
                key = f"mid_convs.{(index - 2) // 3}.conv.{suffix}"
            elif index >= 3 and (index - 3) % 3 == 0:
                key = f"mid_bns.{(index - 3) // 3}.{suffix}"
            else:
                raise ValueError(f"unsupported state-bearing DnCNN sequential key: {key}")
            remapped[key] = value
            continue

        match = re.match(r"^conv(\d+)\.(.+)$", key)
        if match:
            index = int(match.group(1))
            suffix = match.group(2)
            if not 1 <= index <= num_layers:
                raise ValueError(
                    f"legacy DnCNN conv index {index} is outside 1..{num_layers}: {key}"
                )
            if index == 1:
                key = f"first_conv.conv.{suffix}"
            elif index == num_layers:
                key = f"last_conv.conv.{suffix}"
            else:
                key = f"mid_convs.{index - 2}.conv.{suffix}"
        else:
            match = re.match(r"^bn(\d+)\.(.+)$", key)
            if match:
                index = int(match.group(1))
                if not 2 <= index < num_layers:
                    raise ValueError(
                        f"legacy DnCNN batch-norm index {index} is outside 2..{num_layers - 1}: {key}"
                    )
                key = f"mid_bns.{index - 2}.{match.group(2)}"
        remapped[key] = value
    return remapped


def _remap_legacy_unet_block(prefix: str, suffix: str) -> str:
    if suffix.startswith(("conv1.conv.", "conv2.conv.")):
        return f"{prefix}.{suffix}"
    replacements = (
        ("conv.0.", "conv1.conv."),
        ("conv.1.", "bn1."),
        ("conv.3.", "conv2.conv."),
        ("conv.4.", "bn2."),
        ("conv1.", "conv1.conv."),
        ("conv2.", "conv2.conv."),
    )
    for old, new in replacements:
        if suffix.startswith(old):
            return f"{prefix}.{new}{suffix[len(old):]}"
    return f"{prefix}.{suffix}"


def remap_legacy_unet_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map sequential or explicit legacy UNet blocks to the unified UNet."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        parts = key.split(".")
        if key.startswith("downs.") and len(parts) >= 3:
            key = _remap_legacy_unet_block(
                f"downs.{parts[1]}", ".".join(parts[2:])
            )
        elif key.startswith("ups.") and len(parts) >= 3:
            index = int(parts[1])
            suffix = ".".join(parts[2:])
            if index % 2 == 0:
                key = f"ups_transpose.{index // 2}.{suffix}"
            else:
                key = _remap_legacy_unet_block(f"ups_conv.{index // 2}", suffix)
        elif key.startswith("bottleneck."):
            key = _remap_legacy_unet_block(
                "bottleneck", key.removeprefix("bottleneck.")
            )
        elif key.startswith("final_conv.") and not key.startswith("final_conv.conv."):
            key = f"final_conv.conv.{key.removeprefix('final_conv.')}"
        remapped[key] = value
    return remapped


def remap_legacy_state_dict(
    state: Mapping[str, torch.Tensor], task: str, model_name: str | None = None,
    num_layers: int = 17,
) -> dict[str, torch.Tensor]:
    """Convert a known legacy key layout without dropping or inventing tensors."""
    if task == "classification":
        if model_name != "vgg16":
            raise ValueError("legacy classification conversion currently supports --model vgg16")
        return remap_legacy_vgg_state_dict(state)
    if task == "denoising":
        return remap_legacy_dncnn_state_dict(state, num_layers=num_layers)
    if task == "segmentation":
        return remap_legacy_unet_state_dict(state)
    raise ValueError(f"unsupported task: {task}")
