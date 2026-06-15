"""Training and evaluation engine for KBD-Net classification.

Supports:
    - single-stage standard AdamW + cosine warmup/annealing
    - label smoothing cross-entropy
    - optional Mixup / CutMix
    - automatic mixed precision (AMP)
    - best-checkpoint tracking + periodic snapshots
    - logging to console + CSV
    - seed management for reproducibility
"""

from __future__ import annotations

import csv
import os
import random
import time
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..data.dataset import mixup, cutmix, mixup_criterion
from .optim import build_optimizer, build_scheduler, build_criterion


__all__ = ["set_seed", "train_one_epoch", "evaluate", "fit"]


# ---------------------------------------------------------------------- #
# Utilities                                                             #
# ---------------------------------------------------------------------- #
def set_seed(seed: int, use_cuda: bool = True) -> None:
    """Seed every source of randomness (Python, NumPy, Torch, CUDA cudnn)."""
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic but slower — we go with cudnn benchmark for speed
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------- #
# Core loops                                                            #
# ---------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    log_interval: int = 50,
) -> Dict[str, float]:
    """Train model for ONE epoch. Returns dict with loss and top-1."""
    model.train()
    scaler = GradScaler(enabled=use_amp)
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    steps_per_epoch = len(loader)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            if mixup_alpha > 0:
                x, y_a, y_b, lam = mixup(x, y, mixup_alpha)
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            elif cutmix_alpha > 0:
                x, y_a, y_b, lam = cutmix(x, y, cutmix_alpha)
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(x)
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update lr per step (cosine + warmup are per-step in this schedule).
        # Our scheduler is epoch-based; step-based schedule not required here.

        total_loss += loss.item() * bs
        total_samples += bs

        # Top-1 / Top-5 without mixup/cutmix (still using logits).
        with torch.no_grad():
            _, pred = logits.topk(5, dim=1, largest=True, sorted=True)
            if mixup_alpha > 0 or cutmix_alpha > 0:
                # Use y_a as reference for reporting — fair enough for logging.
                targets = y_a if mixup_alpha > 0 else y_a
            else:
                targets = y
            correct_top1 = (pred[:, 0] == targets).sum().item()
            correct_top5 = (pred == targets.unsqueeze(1)).any(dim=1).sum().item()
        total_top1 += correct_top1
        total_top5 += correct_top5

        if (step + 1) % log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch={epoch:>3} step={step+1:>4}/{steps_per_epoch} "
                f"loss={loss.item():.4f} top1={correct_top1/bs*100:>5.2f}% "
                f"lr={lr_now:.2e}"
            )

    if scheduler is not None:
        try:
            scheduler.step()
        except Exception:
            pass

    return {
        "loss": total_loss / max(total_samples, 1),
        "top1": 100.0 * total_top1 / max(total_samples, 1),
        "top5": 100.0 * total_top5 / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device("cuda"),
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluate model on loader. Returns loss/top1/top5 dict."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)
        with autocast(enabled=use_amp):
            logits = model(x)
            if criterion is not None:
                loss = criterion(logits, y)
            else:
                loss = torch.tensor(0.0, device=x.device)

        total_loss += loss.item() * bs
        total_samples += bs

        _, pred = logits.topk(5, dim=1, largest=True, sorted=True)
        correct_top1 = (pred[:, 0] == y).sum().item()
        correct_top5 = (pred == y.unsqueeze(1)).any(dim=1).sum().item()
        total_top1 += correct_top1
        total_top5 += correct_top5

    return {
        "loss": total_loss / max(total_samples, 1),
        "top1": 100.0 * total_top1 / max(total_samples, 1),
        "top5": 100.0 * total_top5 / max(total_samples, 1),
    }


# ---------------------------------------------------------------------- #
# High-level trainer                                                    #
# ---------------------------------------------------------------------- #
def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    *,
    epochs: int = 300,
    optimizer_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
    use_amp: bool = True,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    log_interval: int = 50,
    save_interval: int = 25,
    resume: Optional[str] = None,
) -> Dict[str, float]:
    """Full training loop. Returns final best top-1 and test metrics."""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed, use_cuda=device.type == "cuda")

    model = model.to(device)
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = build_scheduler(
        optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs,
        steps_per_epoch=len(train_loader),
    )
    criterion = build_criterion(label_smoothing).to(device)

    start_epoch = 1
    best_top1 = -1.0
    best_state_path = os.path.join(output_dir, "best.pth")

    # CSV log.
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_fp = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow([
        "epoch", "train_loss", "train_top1", "train_top5",
        "val_loss", "val_top1", "val_top5", "lr", "seconds",
    ])
    csv_fp.flush()

    # Resume?
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        best_top1 = float(ckpt.get("best_top1", -1.0))
        print(f"[fit] Resumed from {resume}, epoch={start_epoch}, best_top1={best_top1:.2f}")

    print(f"[fit] start  epochs={epochs}  lr={lr}  seed={seed}  AMP={use_amp}")
    print(f"[fit] device={device}  params={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device=device, epoch=epoch, use_amp=use_amp,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
            log_interval=log_interval,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device=device, use_amp=use_amp
        )
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[epoch {epoch:>3}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:>5.2f}% "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:>5.2f}% "
            f"top5={val_metrics['top5']:>5.2f}% lr={lr_now:.2e} time={elapsed:.0f}s"
        )

        csv_writer.writerow([
            epoch,
            f"{train_metrics['loss']:.6f}",
            f"{train_metrics['top1']:.4f}",
            f"{train_metrics['top5']:.4f}",
            f"{val_metrics['loss']:.6f}",
            f"{val_metrics['top1']:.4f}",
            f"{val_metrics['top5']:.4f}",
            f"{lr_now:.2e}",
            f"{elapsed:.2f}",
        ])
        csv_fp.flush()

        # Best checkpoint.
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_top1": best_top1,
                    "metrics": val_metrics,
                },
                best_state_path,
            )
            print(f"  -> NEW BEST val_top1={best_top1:.2f}% saved to {best_state_path}")

        # Periodic snapshot.
        if save_interval > 0 and epoch % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_top1": best_top1,
                },
                os.path.join(output_dir, f"epoch_{epoch:03d}.pth"),
            )

    csv_fp.close()
    print(f"[fit] training complete. best val_top1={best_top1:.2f}%  log={csv_path}")

    best_metrics = {
        "best_top1": best_top1,
        "final_val_top1": val_metrics["top1"],
        "final_val_top5": val_metrics["top5"],
        "final_val_loss": val_metrics["loss"],
    }
    return best_metrics
