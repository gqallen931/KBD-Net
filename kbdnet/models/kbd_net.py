"""KBD Block and KBD-Net macro-architecture (ResNet-18 style).

A KBD Block replaces the two 3x3 convolutions in a ResNet BasicBlock with
KBD-Conv layers. The macro-architecture retains the original ResNet-18
layout: four stages with channel dimensions [64, 128, 256, 512] and block
counts [2, 2, 2, 2], plus an optional stem and FC classifier head.

Total parameters (ResNet-18 backbone, CIFAR ImageNet variant):
    ResNet-18 baseline  : ~11.17 M
    KBD-Net (ours)      : ~7.20 M  (-35.5%)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .kbd_conv import KBDConv2d


__all__ = ["KBDBlock", "KBDNet"]


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class KBDBlock(nn.Module):
    """Residual block with two KBD-Conv layers.

    x -- KBD-Conv(BN+SiLU) -- KBD-Conv(BN) -- (+ shortcut) -- SiLU -- y
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_basis: int = 4,
        dilations=None,
        reduction_ratio: int = 4,
        norm: str = "bn",
        dynamic_weight: bool = True,
        equal_weight: bool = False,
        return_alpha: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.return_alpha = return_alpha

        # First KBD-Conv may change spatial dims (via stride) and channels.
        self.conv1 = KBDConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_basis=num_basis,
            dilations=dilations,
            stride=stride,
            reduction_ratio=reduction_ratio,
            use_bias=True,
            activation="silu",
            norm=norm,
            dynamic_weight=dynamic_weight,
            equal_weight=equal_weight,
            return_alpha=return_alpha,
        )
        # Second KBD-Conv keeps spatial dims & channels.
        self.conv2 = KBDConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            num_basis=num_basis,
            dilations=dilations,
            stride=1,
            reduction_ratio=reduction_ratio,
            use_bias=True,
            activation="silu",
            norm=norm,
            dynamic_weight=dynamic_weight,
            equal_weight=equal_weight,
            return_alpha=return_alpha,
        )

        # Shortcut projection when shape mismatch.
        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        identity = x

        if self.return_alpha:
            out, _ = self.conv1(x)
            out = _silu(out)
            out, _ = self.conv2(out)
        else:
            out = self.conv1(x)
            out = _silu(out)
            out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = _silu(out)
        return out


class KBDStem(nn.Module):
    """Entry stem used for ImageNet-scale inputs (224x224):
    Conv3x3 stride=2 + BN + SiLU + MaxPool2d(3,2).
    For CIFAR-scale (32x32) we use a simpler stem: single Conv3x3 stride=1.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64,
                 imagenet: bool = True) -> None:
        super().__init__()
        self.imagenet = imagenet
        if imagenet:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                  stride=2, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = _silu
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = _silu
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.imagenet:
            x = self.pool(x)
        return x


class KBDNet(nn.Module):
    """KBD-Net macro-architecture (ResNet-18 style).

    Args:
        num_classes:       Number of classes for the final FC head.
        stem_channels:     Output channels of the stem conv.
        stage_channels:    Channel list for the four stages (default [64,128,256,512]).
        stage_blocks:      Number of KBD blocks per stage (default [2,2,2,2]).
        strides:           Stride of the first block in each stage (default [1,2,2,2]).
        num_basis:         Number of basis kernels M for every KBD-Conv.
        dilations:         Dilation rates per basis kernel (shared across every KBD-Conv).
        reduction_ratio:   Routing MLP reduction ratio r.
        imagenet_input:    If True, use ImageNet-style stem (stride-2 conv + maxpool).
                          If False, use CIFAR-style stem (stride-1 conv only).
        dropout:          Dropout rate before the FC classifier (0.0 disables).
        dynamic_weight:    Whether to enable input-conditioned routing.
        equal_weight:      If True, bypass routing and use uniform weights (ablation).
        return_alpha:      If True, also return per-layer alpha (not wired through
                          the forward path by default — set True to collect alphas).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        stem_channels: int = 64,
        stage_channels: Tuple[int, ...] = (64, 128, 256, 512),
        stage_blocks: Tuple[int, ...] = (2, 2, 2, 2),
        strides: Tuple[int, ...] = (1, 2, 2, 2),
        num_basis: int = 4,
        dilations: Optional[List[int]] = None,
        reduction_ratio: int = 4,
        imagenet_input: bool = True,
        dropout: float = 0.1,
        dynamic_weight: bool = True,
        equal_weight: bool = False,
        return_alpha: bool = False,
    ) -> None:
        super().__init__()
        assert len(stage_channels) == len(stage_blocks) == len(strides) == 4
        if dilations is None:
            dilations = list(range(1, num_basis + 1))

        self.num_classes = num_classes
        self.stage_channels = stage_channels
        self.stage_blocks = stage_blocks
        self.imagenet_input = imagenet_input
        self._return_alpha = return_alpha

        # Stem.
        self.stem = KBDStem(
            in_channels=3,
            out_channels=stem_channels,
            imagenet=imagenet_input,
        )

        # Four stages.
        in_ch = stem_channels
        stages: List[nn.Sequential] = []
        for stage_idx, (out_ch, n_blk, s) in enumerate(
            zip(stage_channels, stage_blocks, strides)
        ):
            blocks: List[KBDBlock] = []
            for b in range(n_blk):
                stride_b = s if b == 0 else 1
                blocks.append(
                    KBDBlock(
                        in_channels=in_ch if b == 0 else out_ch,
                        out_channels=out_ch,
                        stride=stride_b,
                        num_basis=num_basis,
                        dilations=dilations,
                        reduction_ratio=reduction_ratio,
                        norm="bn",
                        dynamic_weight=dynamic_weight,
                        equal_weight=equal_weight,
                        return_alpha=return_alpha,
                    )
                )
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Classifier head.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(stage_channels[-1], num_classes)

        # Weight init — follow ResNet convention.
        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    # ------------------------------------------------------------------ #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"imagenet_input={self.imagenet_input}, "
            f"stage_channels={self.stage_channels}, "
            f"stage_blocks={self.stage_blocks}, "
            f"params={self.count_parameters():,}"
        )
