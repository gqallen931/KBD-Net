"""Visualization helpers.

Three utilities are exposed:
    1. Effective receptive field (ERF) heatmap via backprop (Luo et al. NeurIPS 2016).
    2. Grad-CAM class-activation map (requires pip install pytorch-grad-cam).
    3. Basis-kernel alpha weight statistics per class.

All are intentionally light so they can be invoked from a notebook or CLI
without heavy dependencies.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------- #
# ERF via analytical backprop (Luo et al. NeurIPS 2016)                #
# ---------------------------------------------------------------------- #
def compute_erf(
    model: torch.nn.Module,
    input_size: int = 224,
    target_layer: str = "stages",
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Analytical effective receptive field using backprop.

    Returns a 2D heatmap normalized to [0, 1].
    Reference: https://github.com/rogerg03/KBD-Net docs (ERF supplement).
    """
    model.eval()
    x = torch.zeros(1, 3, input_size, input_size, device=device, requires_grad=True)

    # Get intermediate activations.
    target_out: Optional[torch.Tensor] = None

    def hook(m, inp, out):
        nonlocal target_out
        target_out = out

    module = _find_module(model, target_layer)
    if module is None:
        module = list(model.children())[-1]  # fallback: classifier
    handle = module.register_forward_hook(hook)
    try:
        _ = model(x)
        if target_out is None:
            raise RuntimeError("hook did not capture any tensor")
        # Pick center activation channel.
        B, C, H, W = target_out.shape
        center = target_out[:, C // 2, H // 2, W // 2]
        center.backward()
        grad = x.grad.detach().abs().squeeze(0).mean(0).cpu().numpy()
        if grad.max() > 0:
            grad = grad / grad.max()
        return grad
    finally:
        handle.remove()


def _find_module(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    if name == "":
        return None
    # allow dotted names like "stages.0"
    parts = name.split(".")
    obj: Any = model
    for p in parts:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            return None
    return obj if isinstance(obj, torch.nn.Module) else None


# ---------------------------------------------------------------------- #
# Grad-CAM (pytorch-grad-cam is optional)                               #
# ---------------------------------------------------------------------- #
def grad_cam(
    model: torch.nn.Module,
    image: Image.Image | np.ndarray,
    target_layer_name: str = "stages.3",
    class_idx: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Return a [H,W] CAM heatmap in [0,1] using pytorch-grad-cam."""
    try:
        from pytorch_grad_cam import GradCAM  # type: ignore
        from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pytorch-grad-cam not installed. pip install pytorch-grad-cam"
        ) from e

    module = _find_module(model, target_layer_name)
    if module is None:
        raise ValueError(f"Could not resolve target layer '{target_layer_name}'")

    cam = GradCAM(model=model, target_layers=[module], use_cuda=device.type == "cuda")

    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),
    ])
    if isinstance(image, Image.Image):
        img_tensor = tf(image).unsqueeze(0).to(device)
    else:
        img_tensor = tf(Image.fromarray(image)).unsqueeze(0).to(device)

    target = None if class_idx is None else torch.tensor([class_idx]).to(device)
    grayscale = cam(input_tensor=img_tensor, targets=target)
    return grayscale.squeeze(0)


# ---------------------------------------------------------------------- #
# Alpha weight collector                                                #
# ---------------------------------------------------------------------- #
def collect_basis_weights(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cuda"),
    max_batches: int = 100,
) -> np.ndarray:
    """Run model over ``loader`` and return mean alpha per KBD-Conv layer.

    Returns array of shape (num_layers, M).
    NOTE: requires that every KBDConv2d was instantiated with return_alpha=True
    OR we monkey-patch the model to intercept routing output. For simplicity,
    this function uses a hook-based approach that reads every KBD-Conv's last
    alpha buffer. It requires that KBDConv2d exposes an ``_last_alpha``
    attribute that is populated during forward. See models/kbd_conv.py — we
    attach it there in forward as self._last_alpha = alpha.detach().
    """
    model.eval()
    # Attach an _last_alpha attribute to every KBDConv2d module for this run.
    from ..models.kbd_conv import KBDConv2d

    kbds: List[KBDConv2d] = [m for m in model.modules() if isinstance(m, KBDConv2d)]
    alphas_accum: List[torch.Tensor] = [
        torch.zeros(m.num_basis, device=device) for m in kbds
    ]
    counts = [0] * len(kbds)

    def make_hook(idx):
        def h(m, inp, out):
            alpha = getattr(m, "_last_alpha", None)
            if alpha is None:
                return
            # alpha may be (B, M); average over batch dim.
            if alpha.dim() == 2:
                mean_a = alpha.mean(dim=0)
            else:
                mean_a = alpha.flatten()
            alphas_accum[idx] = alphas_accum[idx] + mean_a.detach()
            counts[idx] += 1
        return h

    handles = [kbd.register_forward_hook(make_hook(i)) for i, kbd in enumerate(kbds)]

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            _ = model(x.to(device))

    for h in handles:
        h.remove()

    out = np.zeros((len(kbds), kbds[0].num_basis), dtype=np.float32)
    for i, acc in enumerate(alphas_accum):
        if counts[i] > 0:
            out[i] = (acc / counts[i]).cpu().numpy()
    return out
