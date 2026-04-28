"""SIREN with proper Sitzmann init. Used for both image and SDF fitting.

flatten_weights / unflatten_weights use canonical parameter order so
anchor weights and hypernet outputs are binary-compatible.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, is_first: bool = False, w0: float = 30.0):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self.in_f = in_f
        self._init()

    def _init(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_f
            else:
                bound = math.sqrt(6.0 / self.in_f) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        w0_first: float = 30.0,
        w0_hidden: float = 30.0,
    ):
        super().__init__()
        self.cfg = dict(
            in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, w0_first=w0_first, w0_hidden=w0_hidden,
        )

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(SineLayer(in_dim, hidden_dim, is_first=True, w0=w0_first))
            else:
                layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, w0=w0_hidden))
        self.net = nn.Sequential(*layers)

        # final linear (no sin)
        self.final = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / w0_hidden
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def flatten_weights(model: SIREN) -> torch.Tensor:
    """Flatten all parameters into a single 1-D tensor (canonical order)."""
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def unflatten_weights(model: SIREN, flat: torch.Tensor) -> None:
    """Load a flat weight vector back into the model."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].reshape(p.shape))
        offset += n
    assert offset == flat.numel(), f"size mismatch: {offset} vs {flat.numel()}"
