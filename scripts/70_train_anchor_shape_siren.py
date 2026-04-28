"""Train anchor shape-SIREN on unit sphere SDF.

All 10 per-object shape-SIRENs warm-start from this.
Uses L1 loss, no eikonal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN


def sphere_sdf(pts: torch.Tensor, r: float = 0.6) -> torch.Tensor:
    return pts.norm(dim=-1, keepdim=True) - r


def main():
    dev = CFG.device
    c = CFG.shape_siren
    c.anchor_path.parent.mkdir(parents=True, exist_ok=True)

    siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                  c.w0_first, c.w0_hidden).to(dev)
    opt = torch.optim.Adam(siren.parameters(), lr=c.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=c.steps_anchor, eta_min=1e-6
    )

    print(f"[anchor-sdf] SIREN: {siren.num_params():,} params  "
          f"(hidden={c.hidden_dim}, layers={c.num_layers})")
    print(f"[anchor-sdf] training on unit sphere for {c.steps_anchor} steps  |  L1 loss")

    for step in range(c.steps_anchor):
        pts = torch.empty(c.batch_points, 3, device=dev).uniform_(-1.0, 1.0)
        pred = siren(pts)
        target = sphere_sdf(pts).clamp(-c.truncation, c.truncation)
        pred_clamped = pred.clamp(-c.truncation, c.truncation)

        loss = F.l1_loss(pred_clamped, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

        if step % 500 == 0:
            print(f"  step {step:5d}  l1 {loss.item():.6f}")

    torch.save({"state_dict": siren.state_dict(), "cfg": siren.cfg}, c.anchor_path)
    print(f"[anchor-sdf] saved -> {c.anchor_path}")


if __name__ == "__main__":
    main()