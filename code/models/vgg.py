"""VGG-16 classifier used by the CIFAR experiments.

The module names and initialization intentionally match the released training
code so that existing project checkpoints load without key remapping.
"""

import math

import torch
import torch.nn as nn

from .noisy_layers import NoisyConv2d, NoisyLinear


VGG16_CONFIG = (
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
)


class VGG(nn.Module):
    """Project VGG-16 with optional RRAM or PCM weight perturbations.

    This is the architecture used by the project, not torchvision's VGG-16.
    Its convolutional stack is followed by adaptive 1x1 pooling and the
    512-256-128-class classifier used by the released checkpoints.
    """

    def __init__(self, cfg_name="vgg16", num_classes=10, input_size=32):
        super().__init__()
        if cfg_name != "vgg16":
            raise ValueError("the public package only includes the VGG-16 configuration")
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        self.cfg_name = cfg_name

        conv_layers = []
        bn_layers = []
        pool_after = []
        in_channels = 3
        conv_index = 0
        for value in VGG16_CONFIG:
            if value == "M":
                if conv_index > 0:
                    pool_after.append(conv_index - 1)
                continue
            conv_layers.append(
                NoisyConv2d(in_channels, value, kernel_size=3, padding=1)
            )
            bn_layers.append(nn.BatchNorm2d(value))
            in_channels = value
            conv_index += 1

        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.pool_after = set(pool_after)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = NoisyLinear(in_channels, 256)
        self.fbn1 = nn.BatchNorm1d(256)
        self.fc2 = NoisyLinear(256, 128)
        self.fbn2 = nn.BatchNorm1d(128)
        self.fc3 = NoisyLinear(128, num_classes)

        self._initialize_weights()

    def forward(self, x, lamda=0.0, noise_type="rram"):
        for index, (conv, batch_norm) in enumerate(
            zip(self.conv_layers, self.bn_layers)
        ):
            x = conv(x, lamda, noise_type)
            x = batch_norm(x)
            x = self.relu(x)
            if index in self.pool_after:
                x = self.pool(x)

        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu(self.fbn1(self.fc1(x, lamda, noise_type)))
        x = self.relu(self.fbn2(self.fc2(x, lamda, noise_type)))
        return self.fc3(x, lamda, noise_type)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, NoisyConv2d):
                conv = module.conv
                fan_out = (
                    conv.kernel_size[0]
                    * conv.kernel_size[1]
                    * conv.out_channels
                )
                conv.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if conv.bias is not None:
                    conv.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(0.5)
                module.bias.data.zero_()
            elif isinstance(module, NoisyLinear):
                module.linear.weight.data.normal_(0, 0.01)
                module.linear.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def VGG16(num_classes=10, input_size=32):
    """Construct the checkpoint-compatible project VGG-16."""
    return VGG("vgg16", num_classes, input_size)
