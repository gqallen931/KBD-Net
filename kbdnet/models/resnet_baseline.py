"""Standard ResNet-18 baseline for fair comparison.

Kept purposefully simple so it can be used as a drop-in reference alongside
KBD-Net. Uses the same stem / stage / head layout as KBD-Net so that any
performance difference is attributable to the KBD-Conv mechanism alone.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


__all__ = ["ResNet18Baseline", "resnet18"]


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = _relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = _relu(out + identity)
        return out


class ResNet18Baseline(nn.Module):
    """ResNet-18 baseline.

    Args:
        num_classes:       Number of output classes.
        stem_channels:     Stem conv out channels (64).
        stage_channels:    Per-stage channel tuple (64,128,256,512).
        stage_blocks:      Blocks per stage (2,2,2,2).
        strides:           Stride of the first block in each stage.
        imagenet_input:    True -> ImageNet stem (stride-2 conv + maxpool).
                           False -> CIFAR stem (stride-1 conv).
        dropout:           Dropout rate before FC (0.0 disables).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        stem_channels: int = 64,
        stage_channels: Tuple[int, ...] = (64, 128, 256, 512),
        stage_blocks: Tuple[int, ...] = (2, 2, 2, 2),
        strides: Tuple[int, ...] = (1, 2, 2, 2),
        imagenet_input: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(stage_channels) == len(stage_blocks) == len(strides) == 4

        self.imagenet_input = imagenet_input
        if imagenet_input:
            self.stem = nn.Sequential(
                nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, stem_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )

        in_ch = stem_channels
        stages = []
        for out_ch, n_blk, s in zip(stage_channels, stage_blocks, strides):
            blocks = []
            for b in range(n_blk):
                stride_b = s if b == 0 else 1
                in_b = in_ch if b == 0 else out_ch
                blocks.append(_BasicBlock(in_b, out_ch, stride_b))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(stage_channels[-1], num_classes)

        # Weight init.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def resnet18(**kwargs) -> ResNet18Baseline:
    return ResNet18Baseline(**kwargs)
