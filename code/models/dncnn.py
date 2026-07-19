"""DnCNN residual denoiser with optional weight perturbations."""

import math

import torch
import torch.nn as nn

from .noisy_layers import NoisyConv2d


class DnCNN(nn.Module):
    """DnCNN-S network used for the Berkeley400 and Set12 experiments.

    The network predicts additive image noise. The caller obtains the denoised
    image by subtracting this output from the noisy input.
    """

    def __init__(self, channels=1, num_of_layers=17, features=64):
        super().__init__()
        if num_of_layers < 3:
            raise ValueError(f"DnCNN requires at least three layers, got {num_of_layers}")
        self.num_of_layers = num_of_layers

        self.first_conv = NoisyConv2d(
            channels, features, kernel_size=3, padding=1, bias=False
        )
        self.mid_convs = nn.ModuleList()
        self.mid_bns = nn.ModuleList()
        for _ in range(num_of_layers - 2):
            self.mid_convs.append(
                NoisyConv2d(
                    features,
                    features,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            self.mid_bns.append(nn.BatchNorm2d(features))
        self.last_conv = NoisyConv2d(
            features, channels, kernel_size=3, padding=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self._initialize_weights()

    def forward(self, x, lamda=0.0, noise_type="rram"):
        x = self.quant(x)
        output = self.relu(self.first_conv(x, lamda, noise_type))
        for conv, batch_norm in zip(self.mid_convs, self.mid_bns):
            output = self.relu(batch_norm(conv(output, lamda, noise_type)))
        output = self.last_conv(output, lamda, noise_type)
        return self.dequant(output)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, NoisyConv2d):
                nn.init.kaiming_normal_(
                    module.conv.weight, a=0, mode="fan_in"
                )
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(
                    mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)
                ).clamp_(-0.025, 0.025)
                nn.init.constant_(module.bias, 0.0)
