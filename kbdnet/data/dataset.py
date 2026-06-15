"""Dataset builders for CIFAR-10, CIFAR-100 and ImageNet-style classification.

All builders return standard torchvision dataloaders with the augmentation
recipe described in the KBD-Net paper / experimental guide v5.0:

    CIFAR:   RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
    ImageNet: RandomResizedCrop(224) + RandomHorizontalFlip + Normalize

Optional Mixup / CutMix are implemented as callable transforms that can be
wrapped around a batch tensor.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


__all__ = ["build_cifar", "build_imagenet"]


# ---------------------------------------------------------------------- #
# CIFAR                                                                 #
# ---------------------------------------------------------------------- #
_CIFAR_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
    },
}


def build_cifar(
    name: str = "cifar100",
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for CIFAR-10 or CIFAR-100."""
    name = name.lower()
    assert name in _CIFAR_STATS, f"Unknown CIFAR dataset: {name}"
    mean = _CIFAR_STATS[name]["mean"]
    std = _CIFAR_STATS[name]["std"]

    os.makedirs(root, exist_ok=True)

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    train_ds = cls(root=root, train=True,  download=download, transform=train_tf)
    val_ds   = cls(root=root, train=False, download=download, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------- #
# ImageNet                                                              #
# ---------------------------------------------------------------------- #
_IMAGENET_STATS = {
    "mean": (0.485, 0.456, 0.406),
    "std":  (0.229, 0.224, 0.225),
}


def build_imagenet(
    root: str = "./data/imagenet",
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 224,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for ImageNet (ILSVRC 2012 style).

    Expected directory layout:
        root/train/<class_name>/*.jpg
        root/val/<class_name>/*.jpg
    """
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet directories not found at {root}. Expected train/ and val/."
        )

    mean, std = _IMAGENET_STATS["mean"], _IMAGENET_STATS["std"]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------- #
# Mixup / CutMix helpers                                                #
# ---------------------------------------------------------------------- #
def mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Sample a Mixup coefficient from Beta(alpha, alpha)."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample(()))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def cutmix(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix to a batch."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample(()))
    batch_size, _, H, W = x.shape
    index = torch.randperm(batch_size, device=x.device)
    # Random box.
    rw = int(W * (lam ** 0.5))
    rh = int(H * (lam ** 0.5))
    rx = torch.randint(0, W - rw + 1, (1,), device=x.device).item()
    ry = torch.randint(0, H - rh + 1, (1,), device=x.device).item()
    x[:, :, ry:ry + rh, rx:rx + rw] = x[index, :, ry:ry + rh, rx:rx + rw]
    lam = 1.0 - (rh * rw) / (H * W)  # adjust lambda to match area ratio
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(
    criterion, pred: torch.Tensor, y_a: torch.Tensor,
    y_b: torch.Tensor, lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
