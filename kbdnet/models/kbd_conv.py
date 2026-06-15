"""KBD-Conv: Kernel Basis Decomposition Convolution.

Core building block of KBD-Net. Replaces a standard 3x3 convolution with
an input-adaptive multi-scale operation defined as:

    K_eff(X) = sum_i alpha_i(X) . B_i           (parameter-level assembly)
    Y        = sum_i alpha_i(X) . (B_i *_{d_i} X)   (feature-level implementation)

where:
    - B_i are M depthwise basis kernels with distinct dilation rates d_i
    - alpha(X) = Softmax( W2 . SiLU(W1 . GAP(X)) ) are input-conditioned
      mixing weights produced by a lightweight kernel-routing MLP
    - *_{d_i} denotes a depthwise dilated convolution

M=1, d_1=1 reduces to depthwise separable convolution. Standard convolution
is recovered as a special case (conceptually) when all basis kernels share the
same dilation rate and no routing is used.

Reference:
    Kernel Basis Decomposition Network (KBD-Net) — 2026.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["KBDConv2d"]


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


class _BasisConv2d(nn.Module):
    """A single depthwise dilated 3x3 basis kernel.

    Applies a 3x3 depthwise convolution with a fixed dilation ``d``.
    Padding is set to ``d`` so that spatial dimensions are preserved when
    stride == 1.
    """

    def __init__(
        self,
        in_channels: int,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.stride = int(stride)
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _KernelRouter(nn.Module):
    """Lightweight kernel-routing MLP.

    GAP(x) -> FC(C_in, C_in/r) -> SiLU -> FC(C_in/r, M) -> Softmax
    Output: alpha in R^{M}, sum alpha_i = 1, per-sample in batch dim.
    """

    def __init__(
        self,
        in_channels: int,
        num_basis: int = 4,
        reduction_ratio: int = 4,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 8)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, num_basis)
        self.activation = _silu if activation.lower() == "silu" else nn.ReLU()
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> GAP -> (B, C)
        z = x.mean(dim=(2, 3))
        z = self.fc1(z)
        z = _silu(z)
        z = self.fc2(z)
        alpha = F.softmax(z, dim=1)  # (B, M)
        return alpha


class KBDConv2d(nn.Module):
    """Kernel Basis Decomposition Convolution (KBD-Conv).

    Replaces a standard 3x3 convolution with a multi-scale input-adaptive
    operator. See file-level docstring for the mathematical definition.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        num_basis:    Number of basis kernels M (default 4).
        dilations:    List of dilation rates for each basis kernel.
                      If None, defaults to [1, 2, 3, 4] for M=4.
        stride:       Stride of the overall KBD-Conv (applied to every
                      basis kernel; spatial output size determined by stride).
        reduction_ratio:  Channel reduction ratio for the routing MLP.
        use_bias:     Whether to attach a bias to the pointwise projection.
        activation:   Activation after pointwise projection ("silu" or "relu").
        norm:         Normalization layer type ("bn" or "none").
        dynamic_weight: If False, use learnable but static basis weights
                      (overrides routing).
        equal_weight: If True, disable routing and use uniform weights.
        return_alpha: If True, also return the mixing weights alpha.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_basis: int = 4,
        dilations: Optional[List[int]] = None,
        stride: int = 1,
        reduction_ratio: int = 4,
        use_bias: bool = True,
        activation: str = "silu",
        norm: str = "bn",
        dynamic_weight: bool = True,
        equal_weight: bool = False,
        return_alpha: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_basis = int(num_basis)
        self.stride = int(stride)
        self.return_alpha = bool(return_alpha)
        self.dynamic_weight = bool(dynamic_weight) and not bool(equal_weight)
        self.equal_weight = bool(equal_weight)

        if dilations is None:
            dilations = list(range(1, self.num_basis + 1))
        else:
            dilations = list(dilations)
        assert len(dilations) == self.num_basis, (
            f"dilations length {len(dilations)} != num_basis {self.num_basis}"
        )
        self.dilations = dilations

        # Basis kernels: M x depthwise 3x3 conv with distinct dilations.
        self.bases = nn.ModuleList(
            [
                _BasisConv2d(in_channels, dilation=d, stride=stride, bias=False)
                for d in self.dilations
            ]
        )

        # Channel-mixing projection (shared 1x1 pointwise conv).
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias
        )

        # Routing (only when dynamic_weight).
        if self.dynamic_weight and not self.equal_weight:
            self.router = _KernelRouter(
                in_channels=in_channels,
                num_basis=self.num_basis,
                reduction_ratio=reduction_ratio,
                activation=activation,
            )
            self.static_alpha = None
        else:
            self.router = None
            if self.equal_weight:
                self.register_buffer(
                    "static_alpha",
                    torch.ones(self.num_basis, dtype=torch.float32) / self.num_basis,
                )
            else:
                # Learnable but static weights (sample-shared).
                self.static_alpha = nn.Parameter(
                    torch.zeros(self.num_basis, dtype=torch.float32)
                )

        # Post-projection normalization + activation (applied in KBDBlock too,
        # so here we keep it optional; use_norm switches it on).
        self.norm_type = norm.lower()
        if self.norm_type == "bn":
            self.norm: Optional[nn.Module] = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

        act = activation.lower()
        self.activation_fn = _silu if act == "silu" else nn.ReLU()

    # ------------------------------------------------------------------ #
    # Forward                                                           #
    # ------------------------------------------------------------------ #
    def _get_alpha(self, x: torch.Tensor) -> torch.Tensor:
        """Return mixing weights alpha with shape (B, M)."""
        B = x.size(0)
        if self.router is not None and self.dynamic_weight:
            return self.router(x)  # (B, M)
        if self.equal_weight:
            return self.static_alpha.unsqueeze(0).expand(B, -1)  # (B, M)
        # Learnable static: softmax over unconstrained parameters.
        return F.softmax(self.static_alpha, dim=0).unsqueeze(0).expand(B, -1)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Step 1: kernel routing -> mixing weights alpha (B, M).
        alpha = self._get_alpha(x)
        B, M = alpha.shape

        # Step 2: M dilated depthwise basis convolutions.
        # Each F_i: (B, C, H', W') where H', W' depend on stride/dilation.
        F_list = [self.bases[i](x) for i in range(M)]

        # Step 3: weighted feature aggregation (kernel assembly in concept).
        # alpha[:, i]: (B,) -> reshape to (B, 1, 1, 1) for broadcasting.
        alpha_exp = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1, 1)
        # Actually we have F_i as (B,C,H',W'); we need alpha per-basis only.
        # Shape alpha (B, M); want alpha_i (B,1,1,1) to broadcast over C,H',W'.
        agg = torch.zeros_like(F_list[0])
        for i in range(M):
            a_i = alpha[:, i].view(B, 1, 1, 1)
            agg = agg + a_i * F_list[i]

        # Step 4: shared pointwise projection (channel mixing).
        out = self.pointwise(agg)
        if self.norm is not None:
            out = self.norm(out)

        if self.return_alpha:
            return out, alpha
        return out

    # ------------------------------------------------------------------ #
    # Introspection                                                     #
    # ------------------------------------------------------------------ #
    @property
    def basis_kernels(self) -> List[torch.Tensor]:
        """Return list of basis weight tensors, each (C, 1, 3, 3)."""
        return [b.conv.weight.data.clone() for b in self.bases]

    def effective_kernel(self) -> torch.Tensor:
        """Return a synthetic (C_out, C_in, 3, 3) kernel by averaging basis
        kernels with equal weights and zero dilation — only meaningful as a
        conceptual reference, NOT how KBD-Conv actually operates.
        """
        # Not used in forward; exposed for inspection only.
        with torch.no_grad():
            base = torch.zeros(
                self.in_channels, 1, 3, 3, device=next(self.parameters()).device
            )
            for b in self.bases:
                w = b.conv.weight.data
                # w is (C_in, 1, 3, 3). Center it into a standard 3x3 kernel.
                base = base + w
            base = base / max(self.num_basis, 1)
            # Convolve base with pointwise: (C_out, C_in, 1, 1) * base -> ...
            # Conceptually only; real forward applies dilated depthwise first.
            return base

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"M={self.num_basis}, dilations={self.dilations}, stride={self.stride}, "
            f"dynamic={self.dynamic_weight}, equal={self.equal_weight}"
        )
