"""Utility helpers: config parsing, metrics, profiling, checkpointing."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file. Falls back gracefully if PyYAML missing."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            import yaml  # type: ignore
            return yaml.safe_load(f)
        except ImportError:
            # Minimal fallback — only for very simple key: value files.
            data: Dict[str, Any] = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    data[k.strip()] = v.strip().strip('"').strip("'")
            return data


def get_device(cuda: bool = True) -> torch.device:
    if cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_flops(
    model: nn.Module, input_size: int | tuple = 224, device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    """Return params (M) and FLOPs (G) using thop if available.

    Falls back to a parameter-count-only dict if thop is not installed.
    """
    params_m = count_parameters(model) / 1e6
    if isinstance(input_size, int):
        inp = (1, 3, input_size, input_size)
    else:
        inp = (1, 3, *input_size)
    x = torch.randn(*inp, device=device)
    try:
        from thop import profile as thop_profile  # type: ignore
        macs, _ = thop_profile(model, (x,), verbose=False)
        flops_g = 2 * macs / 1e9
        return {"params_m": params_m, "flops_g": flops_g}
    except Exception:
        return {"params_m": params_m, "flops_g": None}


def load_checkpoint(model: nn.Module, path: str, device: torch.device = torch.device("cpu"),
                    strict: bool = True) -> Dict[str, Any]:
    """Load a checkpoint. Returns the raw checkpoint dict."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Strip "module." prefix if from DataParallel.
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=strict)
    return ckpt if isinstance(ckpt, dict) else {"epoch": None, "best_top1": None}


class AverageMeter:
    """Simple running average."""
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}={self.avg:.4f}"
