"""
Plain Encoder-Decoder: Convolutional Segmentation Network (no skip connections)
================================================================================
Architecture-equivalent to U-Net but with skip connections removed.
Used as an ablation to isolate the contribution of skip connections.

Same depth, same channel widths as U-Net — only difference is skip connections
are absent. This makes it a direct test of whether U-Net's skip connections
(not the encoder-decoder structure itself) drive performance.

Reference architecture comparison:
    U-Net small   : ~2M params, features=[32, 64, 128, 256]
    EncDec small  : ~1.5M params, features=[32, 64, 128, 256] (fewer due to no concat)
    U-Net standard: ~7.8M params, features=[64, 128, 256, 512]
    EncDec standard: ~6M params, features=[64, 128, 256, 512]
"""

import torch
import torch.nn as nn

from unet import ConvBlock


class EncDecEncoderBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPool. Does NOT return skip features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


class EncDecDecoderBlock(nn.Module):
    """Decoder block: Upsample + 1x1 Conv + ConvBlock. No skip connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv = ConvBlock(out_channels, out_channels)  # no *2 — no concat

    def forward(self, x):
        x = self.up(x)
        x = self.reduce(x)
        return self.conv(x)


class EncDec(nn.Module):
    """
    Plain Encoder-Decoder for binary segmentation (no skip connections).

    Args:
        in_channels: number of input channels (3 for RGB)
        out_channels: number of output channels (1 for binary segmentation)
        features: list of feature sizes per encoder level
    """

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # Encoder path (no skip connections returned)
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for f in features:
            self.encoders.append(EncDecEncoderBlock(prev_channels, f))
            prev_channels = f

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder path
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for f in reversed_features:
            self.decoders.append(EncDecDecoderBlock(prev_channels, f))
            prev_channels = f

        # Final 1x1 conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)

        x = self.bottleneck(x)

        for decoder in self.decoders:
            x = decoder(x)

        return self.final_conv(x)


def encdec_small(in_channels=3, out_channels=1):
    """Small plain encoder-decoder, same channel widths as unet_small (~1.5M params)."""
    return EncDec(in_channels, out_channels, features=[32, 64, 128, 256])


def encdec_standard(in_channels=3, out_channels=1):
    """Standard plain encoder-decoder, same channel widths as unet_standard (~6M params)."""
    return EncDec(in_channels, out_channels, features=[64, 128, 256, 512])


if __name__ == "__main__":
    for name, model in [("encdec_small", encdec_small()), ("encdec_standard", encdec_standard())]:
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name}: input {x.shape} -> output {out.shape}, params {params:.2f} M")
