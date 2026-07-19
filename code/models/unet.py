"""UNet used for binary Carvana segmentation."""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .noisy_layers import NoisyConv2d


class DoubleConv(nn.Module):
    """Two convolution, batch-normalization, and ReLU stages."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = NoisyConv2d(
            in_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = NoisyConv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, lamda=0.0, noise_type="rram"):
        x = self.relu(self.bn1(self.conv1(x, lamda, noise_type)))
        return self.relu(self.bn2(self.conv2(x, lamda, noise_type)))


class UNet(nn.Module):
    """Checkpoint-compatible UNet with perturbable convolution weights.

    The transposed convolutions remain ordinary PyTorch layers, matching the
    released architecture. Simulator adapters must account for that boundary
    explicitly instead of silently treating a partially converted model as a
    complete analog implementation.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=(64, 128, 256, 512),
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups_transpose = nn.ModuleList()
        self.ups_conv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature

        for feature in reversed(features):
            self.ups_transpose.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups_conv.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = NoisyConv2d(features[0], out_channels, kernel_size=1)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, lamda=0.0, noise_type="rram"):
        x = self.quant(x)
        skip_connections = []
        for down in self.downs:
            x = down(x, lamda, noise_type)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, lamda, noise_type)
        skip_connections = skip_connections[::-1]
        for index, upsample in enumerate(self.ups_transpose):
            x = upsample(x)
            skip = skip_connections[index]
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups_conv[index](x, lamda, noise_type)
        x = self.final_conv(x, lamda, noise_type)
        return self.dequant(x)
