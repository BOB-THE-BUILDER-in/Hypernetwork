"""Per-layer hypernetwork.

Takes a camera direction (3D unit vector) and outputs the full weight vector
of an image-SIREN. One dedicated MLP head per SIREN parameter group (weight
matrix or bias vector), with small-init final layers.

Why per-layer heads instead of one big MLP?
    SIREN uses sin(w0 * Wx). Small errors in W get amplified by w0=30,
    causing huge output errors. A monolithic head can't control per-layer
    precision. Per-layer heads with small-init final layers keep each
    parameter group's prediction tight. (Validated April 2026 — monolithic
    head destroyed torus topology even at loss ~1e-7.)
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from src.siren import SIREN


class HyperNetHead(nn.Module):
    """Small MLP: input_dim -> target_numel."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int,
                 final_scale: float = 1e-2):
        super().__init__()
        nets = []
        for i in range(layers):
            d_in = in_dim if i == 0 else hidden
            d_out = hidden if i < layers - 1 else out_dim
            nets.append(nn.Linear(d_in, d_out))
            if i < layers - 1:
                nets.append(nn.ReLU())
        self.net = nn.Sequential(*nets)

        # small-init on final layer — critical for SIREN targets
        with torch.no_grad():
            self.net[-1].weight.mul_(final_scale)
            self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HyperNet(nn.Module):
    """Camera direction (3D) -> image-SIREN weights.

    Builds one HyperNetHead per parameter in the target SIREN.
    """

    def __init__(
        self,
        target_siren: SIREN,
        in_dim: int = 3,
        head_hidden: int = 256,
        head_layers: int = 3,
        final_init_scale: float = 1e-2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.param_names: list[str] = []
        self.param_shapes: list[torch.Size] = []
        self.param_numels: list[int] = []

        heads = OrderedDict()
        for name, param in target_siren.named_parameters():
            safe_name = name.replace(".", "_")
            self.param_names.append(name)
            self.param_shapes.append(param.shape)
            self.param_numels.append(param.numel())
            heads[safe_name] = HyperNetHead(
                in_dim, param.numel(), head_hidden, head_layers, final_init_scale
            )
        self.heads = nn.ModuleDict(heads)
        self.total_target_params = sum(self.param_numels)

    def forward(self, cam_dir: torch.Tensor) -> torch.Tensor:
        """cam_dir: (3,) -> flat weight vector (total_target_params,)."""
        parts = []
        for name, head in self.heads.items():
            parts.append(head(cam_dir))
        return torch.cat(parts, dim=-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
