"""Inference / prediction script.

Loads a trained KBD-Net checkpoint and runs prediction on:
    - a single image file, or
    - a directory of images, or
    - a standard torchvision dataset (CIFAR / ImageNet val).

Usage:

    python inference.py --checkpoint checkpoints/kbd_net/best.pth \
                        --image ./samples/cat.jpg

    python inference.py --checkpoint checkpoints/kbd_net/best.pth \
                        --dataset cifar100 --data-root ./data --split val

    python inference.py --checkpoint checkpoints/kbd_net/best.pth \
                        --model kbd_net --num-classes 100 \
                        --dilations 1 2 3 4 --num-basis 4 \
                        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kbdnet.utils.misc import load_yaml, load_checkpoint, get_device, count_parameters


CIFAR_CLASSES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "cifar100": [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
        "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
        "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
        "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
        "worm",
    ],
}


def build_model_from_args(args):
    name = (args.model or "kbd_net").lower()
    if name == "resnet18":
        from kbdnet.models.resnet_baseline import ResNet18Baseline
        return ResNet18Baseline(
            num_classes=args.num_classes,
            imagenet_input=bool(args.dataset is None or not args.dataset.startswith("cifar")),
            dropout=0.1,
        )
    from kbdnet.models.kbd_net import KBDNet
    dilations = list(args.dilations) if args.dilations else list(range(1, (args.num_basis or 4) + 1))
    return KBDNet(
        num_classes=args.num_classes,
        num_basis=args.num_basis or 4,
        dilations=dilations,
        reduction_ratio=args.reduction_ratio or 4,
        imagenet_input=bool(args.dataset is None or not args.dataset.startswith("cifar")),
        dropout=0.1,
        dynamic_weight=not args.equal_weight,
        equal_weight=bool(args.equal_weight),
    )


def image_transform_for(dataset: str | None):
    if dataset is not None and dataset.startswith("cifar"):
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    # ImageNet style.
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


def predict_image(model, path: str, transform, device, topk: int = 5,
                  class_names: List[str] | None = None):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        top_p, top_i = probs.topk(topk, dim=1)
    top_p = top_p.squeeze(0).cpu().numpy()
    top_i = top_i.squeeze(0).cpu().numpy()
    print(f"\n[image] {path}")
    for p, idx in zip(top_p, top_i):
        name = class_names[idx] if class_names else f"class_{idx}"
        print(f"  {idx:>4}  {p*100:>6.2f}%  {name}")


def evaluate_dataset(model, dataset: str, data_root: str, transform, device, bs: int = 128):
    if dataset == "cifar10":
        ds = datasets.CIFAR10(root=data_root, train=False, download=True,
                              transform=transform)
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=False, download=True,
                               transform=transform)
    elif dataset.startswith("imagenet"):
        ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                         num_workers=4, pin_memory=True)
    model.eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            _, pred = logits.topk(5, dim=1, largest=True, sorted=True)
            top1 += (pred[:, 0] == y).sum().item()
            top5 += (pred == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
    print(f"[eval] top1={100*top1/total:.2f}%  top5={100*top5/total:.2f}%  n={total}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KBD-Net inference")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--config",    default=None, help="Optional YAML config")
    p.add_argument("--image",     default=None, help="Path to a single image to predict")
    p.add_argument("--dataset",   default=None, choices=["cifar10", "cifar100", "imagenet100", "imagenet"])
    p.add_argument("--data-root", default="./data", help="Dataset root dir")
    p.add_argument("--model",     default="kbd_net", choices=["kbd_net", "resnet18"])
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--num-basis",   type=int, default=4)
    p.add_argument("--dilations",   type=int, nargs="*", default=[1,2,3,4])
    p.add_argument("--reduction-ratio", type=int, default=4)
    p.add_argument("--equal-weight", action="store_true")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--topk",       type=int, default=5)
    p.add_argument("--batch-size",  type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Optional YAML overrides CLI.
    if args.config and os.path.isfile(args.config):
        cfg = load_yaml(args.config)
        args.model = cfg.get("model", {}).get("name", args.model)
        args.num_classes = int(cfg.get("model", {}).get("num_classes", args.num_classes))
        kcfg = cfg.get("model", {}).get("kbd_conv", {})
        args.num_basis = int(kcfg.get("num_basis", args.num_basis))
        args.dilations = [int(d) for d in kcfg.get("dilations", args.dilations)]

    model = build_model_from_args(args).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device=device, strict=True)
    print(f"[load] checkpoint={args.checkpoint}  params={count_parameters(model):,} "
          f"epoch={ckpt.get('epoch','?')} best_top1={ckpt.get('best_top1','?')}")

    class_names = None
    if args.dataset in CIFAR_CLASSES:
        class_names = CIFAR_CLASSES[args.dataset]

    transform = image_transform_for(args.dataset)

    if args.image:
        predict_image(model, args.image, transform, device,
                      topk=args.topk, class_names=class_names)
    elif args.dataset:
        evaluate_dataset(model, args.dataset, args.data_root, transform, device,
                         bs=args.batch_size)
    else:
        print("[inference] nothing to do — provide --image or --dataset")


if __name__ == "__main__":
    main()
