"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
================================================================
Implementation based on the original paper by Ronneberger et al. (2015).

The architecture consists of:
  - Encoder (contracting path): captures context via Conv + MaxPool
  - Decoder (expanding path): enables precise localization via UpConv + skip connections
  - Skip connections: concatenate encoder features with decoder features

Reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    MICCAI 2015. arXiv:1505.04597
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """Encoder block: ConvBlock followed by 2x2 MaxPool downsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)       # save for skip connection
        downsampled = self.pool(features)
        return features, downsampled


class Decoder(nn.Module):
    """Decoder block: Upsample + 1x1 Conv to reduce channels, concatenate skip features, then ConvBlock.
    
    Uses nn.Upsample + nn.Conv2d instead of nn.ConvTranspose2d for better
    compatibility with GETA's dependency graph analysis.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv = ConvBlock(out_channels * 2, out_channels)  # *2 because of concatenation

    def forward(self, x, skip_features):
        x = self.up(x)
        x = self.reduce(x)
        # Concatenate along channel dimension (skip connection)
        x = torch.cat([skip_features, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for binary segmentation.

    Args:
        in_channels: number of input channels (3 for RGB images)
        out_channels: number of output channels (1 for binary segmentation)
        features: list of feature sizes for each encoder level
                  default [32, 64, 128, 256] gives a model with ~2M parameters
    """

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # Encoder path
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for f in features:
            self.encoders.append(Encoder(prev_channels, f))
            prev_channels = f

        # Bottleneck (deepest layer)
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder path (reverse order)
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for f in reversed_features:
            self.decoders.append(Decoder(prev_channels, f))
            prev_channels = f

        # Final 1x1 convolution to map to output classes
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: collect skip connections
        skip_connections = []
        for encoder in self.encoders:
            features, x = encoder(x)
            skip_connections.append(features)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: use skip connections in reverse order
        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        # Final output
        return self.final_conv(x)


def unet_tiny(in_channels=3, out_channels=1):
    """Tiny U-Net with ~0.5M parameters — base channels 16."""
    return UNet(in_channels, out_channels, features=[16, 32, 64, 128])


def unet_small(in_channels=3, out_channels=1):
    """Small U-Net with ~7.24M parameters — base channels 32."""
    return UNet(in_channels, out_channels, features=[32, 64, 128, 256])


def unet_standard(in_channels=3, out_channels=1):
    """Standard U-Net with ~28.95M parameters — base channels 64."""
    return UNet(in_channels, out_channels, features=[64, 128, 256, 512])


def unet_large(in_channels=3, out_channels=1):
    """Large U-Net with ~65M parameters — base channels 96."""
    return UNet(in_channels, out_channels, features=[96, 192, 384, 768])


if __name__ == "__main__":
    for name, fn in [
        ("unet_tiny", unet_tiny),
        ("unet_small", unet_small),
        ("unet_standard", unet_standard),
        ("unet_large", unet_large),
    ]:
        model = fn()
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name:20s}: output {out.shape}, params {params:.2f} M")
