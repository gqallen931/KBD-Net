"""Optimizer, scheduler and loss helpers used by the training engine.

Follows the KBD-Net experimental guide v5.0 exactly:
    - Optimizer : AdamW (weight_decay=0.05)
    - LR schedule : Cosine annealing with 5-epoch linear warmup
    - Loss      : Cross-entropy with label smoothing 0.1
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.05,
) -> torch.optim.Optimizer:
    """Build an optimizer. Only AdamW is officially used in KBD-Net experiments."""
    name = name.lower()
    params = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    steps_per_epoch: int = 0,
) -> "CosineAnnealingWarmup":
    """Cosine annealing schedule with linear warmup."""
    return CosineAnnealingWarmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
    )


class CosineAnnealingWarmup(torch.optim.lr_scheduler.LRScheduler):
    """Cosine annealing schedule with a linear warmup phase.

    lr increases linearly from ``base_lr / warmup_epochs`` to ``base_lr``
    over ``warmup_epochs`` epochs, then decays as cosine over the remaining
    epochs down to 1e-6 of base lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int = 0,
        eta_min_ratio: float = 1e-3,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = max(steps_per_epoch, 0)
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch=-1)

    def _current_lr(self, step: int) -> float:
        # step is an epoch index here (0-based).
        warmup = self.warmup_epochs
        total = self.total_epochs
        eta_min = self.base_lrs[0] * self.eta_min_ratio

        if step < warmup:
            return self.base_lrs[0] * (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return eta_min + 0.5 * (self.base_lrs[0] - eta_min) * (
            1 + math.cos(math.pi * progress)
        )

    def get_lr(self) -> list:
        lr = self._current_lr(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]


def build_criterion(label_smoothing: float = 0.1) -> nn.Module:
    """Label-smoothed cross-entropy loss."""
    return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
