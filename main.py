"""KBD-Net main entry.

Usage examples (from the project root):

    python main.py --config configs/cifar100/kbd_net_full.yaml --seed 42
    python main.py --config configs/cifar100/resnet18_baseline.yaml --seed 42
    python main.py --config configs/imagenet100/kbd_net_full.yaml \
                    --data-root ./data/imagenet100
    python main.py --config configs/cifar100/kbd_ablation_M1.yaml --seed 3407
    python main.py --config configs/cifar100/kbd_ablation_dilation_uniform.yaml --seed 2026
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import torch

# Ensure the project root is on sys.path so `kbdnet.*` is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kbdnet.engine.train import fit
from kbdnet.utils.misc import load_yaml, get_device, count_parameters, profile_flops


# ---------------------------------------------------------------------- #
# Config merging                                                         #
# ---------------------------------------------------------------------- #
def merge_args_into_config(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config values with CLI arguments when provided."""
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = int(args.seed)
    if args.data_root is not None:
        cfg.setdefault("dataset", {})["root"] = args.data_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.resume is not None:
        cfg["resume"] = args.resume
    if args.no_cuda:
        cfg["cuda"] = False
    return cfg


def build_model(cfg: Dict[str, Any]):
    """Build the network described by cfg['model']."""
    mcfg = cfg["model"]
    name = (mcfg.get("name", "kbd_net")).lower()
    num_classes = int(mcfg.get("num_classes", 100))
    imagenet_input = bool(cfg.get("dataset", {}).get("name", "cifar100").startswith("imagenet"))

    if name == "resnet18" or name == "resnet_baseline":
        from kbdnet.models.resnet_baseline import ResNet18Baseline
        return ResNet18Baseline(
            num_classes=num_classes,
            imagenet_input=imagenet_input,
            dropout=float(mcfg.get("dropout", 0.1)),
        )

    # Default: KBD-Net.
    from kbdnet.models.kbd_net import KBDNet
    kcfg = mcfg.get("kbd_conv", {})
    num_basis = int(kcfg.get("num_basis", 4))
    dilations = kcfg.get("dilations")
    if dilations is None:
        dilations = list(range(1, num_basis + 1))
    dilations = [int(d) for d in dilations]

    dynamic_weight = True
    equal_weight = False
    routing = (kcfg.get("routing", {}) or {})
    mode = routing.get("mode", "dynamic")
    if mode in ("equal", "uniform"):
        equal_weight = True
        dynamic_weight = False
    elif mode in ("static", "learnable"):
        dynamic_weight = False
        equal_weight = False
    elif mode in ("channel", "per_channel"):
        # Per-channel routing not implemented here; fall back to dynamic routing.
        dynamic_weight = True

    return KBDNet(
        num_classes=num_classes,
        num_basis=num_basis,
        dilations=dilations,
        reduction_ratio=int(routing.get("reduction_ratio", 4)),
        imagenet_input=imagenet_input,
        dropout=float(mcfg.get("dropout", 0.1)),
        dynamic_weight=dynamic_weight,
        equal_weight=equal_weight,
    )


def build_dataloaders(cfg: Dict[str, Any]):
    """Return (train_loader, val_loader) per the dataset config."""
    dcfg = cfg["dataset"]
    name = dcfg.get("name", "cifar100").lower()
    root = dcfg.get("root", "./data")
    tcfg = cfg["train"]
    batch_size = int(tcfg.get("batch_size", 128))
    num_workers = int(tcfg.get("num_workers", 4))

    if name.startswith("cifar"):
        from kbdnet.data.dataset import build_cifar
        return build_cifar(
            name=name,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=True,
        )
    if name.startswith("imagenet"):
        from kbdnet.data.dataset import build_imagenet
        return build_imagenet(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    raise ValueError(f"Unsupported dataset: {name}")


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KBD-Net training entry")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--seed",     type=int, default=None,  help="Override random seed")
    p.add_argument("--data-root", type=str, default=None,  help="Override dataset root")
    p.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    p.add_argument("--epochs",    type=int, default=None,  help="Override epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--resume",    type=str, default=None, help="Resume from .pth")
    p.add_argument("--no-cuda",   action="store_true", help="Disable CUDA")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = merge_args_into_config(cfg, args)

    output_dir = cfg.get("output_dir", "./checkpoints/run")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device(cuda=cfg.get("cuda", True) and not args.no_cuda)
    torch.backends.cudnn.benchmark = True if device.type == "cuda" else False

    model = build_model(cfg).to(device)
    print(f"[build] model={cfg['model']['name']} classes={cfg['model'].get('num_classes',100)} "
          f"params={count_parameters(model):,}  device={device}")

    try:
        profiling = profile_flops(model, input_size=32 if cfg["dataset"]["name"].startswith("cifar") else 224,
                                  device=device)
        print(f"[build] profiling: params_m={profiling.get('params_m')} flops_g={profiling.get('flops_g')}")
    except Exception as e:
        print(f"[build] profiling skipped: {e}")

    train_loader, val_loader = build_dataloaders(cfg)

    tcfg = cfg["train"]
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        epochs=int(tcfg.get("epochs", 300)),
        optimizer_name=tcfg.get("optimizer", "adamw"),
        lr=float(tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.05)),
        warmup_epochs=int(tcfg.get("warmup_epochs", 5)),
        label_smoothing=float(tcfg.get("label_smoothing", 0.1)),
        seed=int(tcfg.get("seed", 42)),
        device=device,
        use_amp=bool(tcfg.get("amp", True)),
        mixup_alpha=float(tcfg.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(tcfg.get("cutmix_alpha", 0.0)),
        log_interval=int(tcfg.get("log_interval", 50)),
        save_interval=int(tcfg.get("save_interval", 25)),
        resume=cfg.get("resume"),
    )


if __name__ == "__main__":
    main()
