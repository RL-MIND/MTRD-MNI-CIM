"""PyTorch layers implementing the manuscript's weight-noise equations."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _finite_nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"noise scale must be finite, got {value}")
    if value < 0:
        raise ValueError(f"noise scale must be non-negative, got {value}")
    return value


def pcm_layerwise_wmax(weight: torch.Tensor) -> torch.Tensor:
    """Return manuscript Eq. (2) ``Wmax`` for one logical weight layer.

    The manuscript defines ``Wmax`` as the maximum weight value, matching the
    released implementation's ``torch.max(weight)``. This is intentionally not
    the absolute maximum used by symmetric weight quantization.
    """
    detached = weight.detach()
    if detached.numel() == 0 or not torch.isfinite(detached).all():
        raise ValueError("PCM Eq. (2) requires a non-empty finite weight tensor")
    maximum = detached.max()
    if maximum.item() < 0.0:
        raise ValueError(
            "PCM Eq. (2) produced a negative Wmax; a Gaussian standard "
            "deviation cannot be negative"
        )
    return maximum


class NoisyConv2d(nn.Module):
    """Conv2d with a fresh weight-noise realization on every noisy forward."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, lamda=0.0, noise_type="rram"):
        if noise_type not in ("none", "rram", "pcm"):
            raise ValueError(
                f"unknown noise_type {noise_type!r}; expected none, rram, or pcm"
            )
        if noise_type == "none":
            return self.conv(x)
        lamda = _finite_nonnegative_float(lamda)
        if lamda == 0:
            return self.conv(x)

        weight = self.conv.weight
        if noise_type == "rram":
            noise = torch.randn_like(weight) * lamda
            noisy_weight = torch.exp(noise) * weight
        else:
            standard_deviation = lamda * pcm_layerwise_wmax(weight).item()
            if standard_deviation == 0.0:
                return self.conv(x)
            noise = torch.randn_like(weight) * standard_deviation
            noisy_weight = weight + noise
        return F.conv2d(
            x,
            noisy_weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


class NoisyLinear(nn.Module):
    """Linear layer with a fresh weight-noise realization per noisy forward."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, lamda=0.0, noise_type="rram"):
        if noise_type not in ("none", "rram", "pcm"):
            raise ValueError(
                f"unknown noise_type {noise_type!r}; expected none, rram, or pcm"
            )
        if noise_type == "none":
            return self.linear(x)
        lamda = _finite_nonnegative_float(lamda)
        if lamda == 0:
            return self.linear(x)

        weight = self.linear.weight
        if noise_type == "rram":
            noise = torch.randn_like(weight) * lamda
            noisy_weight = torch.exp(noise) * weight
        else:
            standard_deviation = lamda * pcm_layerwise_wmax(weight).item()
            if standard_deviation == 0.0:
                return self.linear(x)
            noise = torch.randn_like(weight) * standard_deviation
            noisy_weight = weight + noise
        return F.linear(x, noisy_weight, self.linear.bias)
