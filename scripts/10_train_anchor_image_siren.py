"""Train shared anchor image-SIREN on a uniform grey image.

All 240 per-view image SIRENs warm-start from this. A neutral grey gives an
unbiased initialization — every SIREN starts equidistant from any real image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN


def main():
    dev = CFG.device
    c = CFG.img_siren
    c.anchor_path.parent.mkdir(parents=True, exist_ok=True)

    siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers, c.w0_first, c.w0_hidden).to(dev)
    opt = torch.optim.Adam(siren.parameters(), lr=c.lr)

    print(f"[anchor-img] SIREN: {siren.num_params():,} params")
    print(f"[anchor-img] training on grey for {c.steps_anchor} steps")

    # grey target: RGB = 0.5
    grey = torch.full((3,), 0.5, device=dev)

    for step in range(c.steps_anchor):
        # random pixel coords in [-1, 1]
        coords = torch.empty(c.batch_pixels, 2, device=dev).uniform_(-1.0, 1.0)
        pred = siren(coords)                    # (B, 3)
        target = grey.unsqueeze(0).expand_as(pred)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.6f}")

    torch.save({"state_dict": siren.state_dict(), "cfg": siren.cfg}, c.anchor_path)
    print(f"[anchor-img] saved -> {c.anchor_path}")


if __name__ == "__main__":
    main()
