from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = F.leaky_relu(self.norm1(self.conv1(x)), negative_slope=0.01, inplace=True)
        x = self.norm2(self.conv2(x))
        return F.leaky_relu(x + identity, negative_slope=0.01, inplace=True)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.block = ResBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TopologyCenterlineUNet(nn.Module):
    """
    Structure-aware 3D U-Net:
    - vessel segmentation head
    - centerline head
    - branchpoint head
    - radius head
    The centerline stream is fused back into the segmentation path so topology
    is used as a conditioning signal rather than only as an auxiliary output.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 24):
        super().__init__()
        c = base_channels

        self.stem = ResBlock3D(in_channels, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)
        self.bridge = DownBlock(c * 8, c * 10)

        self.up3 = UpBlock(c * 10, c * 8, c * 8)
        self.up2 = UpBlock(c * 8, c * 4, c * 4)
        self.up1 = UpBlock(c * 4, c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c)

        self.centerline_head = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c, 1, kernel_size=1),
        )
        self.branchpoint_head = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c, 1, kernel_size=1),
        )
        self.radius_head = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c, 1, kernel_size=1),
        )
        self.centerline_fuser = nn.Sequential(
            nn.Conv3d(c + 2, c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(0.01, inplace=True),
            ResBlock3D(c, c),
        )
        self.seg_head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bridge(s3)

        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)

        centerline_logits = self.centerline_head(x)
        branchpoint_logits = self.branchpoint_head(x)
        radius_logits = self.radius_head(x)

        structure_features = torch.cat(
            [
                x,
                torch.sigmoid(centerline_logits),
                torch.sigmoid(branchpoint_logits),
            ],
            dim=1,
        )
        fused = self.centerline_fuser(structure_features)

        return {
            "seg_logits": self.seg_head(fused),
            "centerline_logits": centerline_logits,
            "branchpoint_logits": branchpoint_logits,
            "radius_logits": radius_logits,
        }
