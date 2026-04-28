"""Fine-tune one shape-SIREN per object, warm-started from the sphere anchor.

Matches working siren_sdf.py exactly:
    - Full-batch training (all points every step)
    - L1 loss, no eikonal
    - Cosine LR schedule
    - Saves best checkpoint

Output: data/shape_sirens/obj_XX.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN


def main():
    dev = CFG.device
    c = CFG.shape_siren
    d = CFG.data

    # load anchor
    assert c.anchor_path.exists(), (
        f"anchor not found at {c.anchor_path} — run 70_train_anchor_shape_siren.py first"
    )
    ckpt = torch.load(c.anchor_path, map_location=dev, weights_only=True)
    anchor_sd = ckpt["state_dict"]
    print(f"[shape-siren] loaded anchor from {c.anchor_path}")

    c.out_dir.mkdir(parents=True, exist_ok=True)

    lr_max = 1e-4
    lr_min = 1e-6

    for obj_i in range(d.num_objects):
        out_path = c.out_dir / f"obj_{obj_i:02d}.pt"
        if out_path.exists():
            print(f"[shape-siren] skip obj_{obj_i:02d}")
            continue

        # load SDF samples
        sdf_path = d.sdf_dir / f"obj_{obj_i:02d}.npz"
        assert sdf_path.exists(), f"missing {sdf_path} — run 60_sample_sdf.py first"
        data = np.load(sdf_path)
        all_pts = torch.from_numpy(data["points"]).to(dev)
        all_sdf = torch.from_numpy(data["sdf"]).to(dev).unsqueeze(-1)
        N = all_pts.shape[0]

        # clamp targets
        all_sdf_clamped = all_sdf.clamp(-c.truncation, c.truncation)

        # fresh SIREN from anchor
        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(dev)
        siren.load_state_dict(anchor_sd)
        opt = torch.optim.Adam(siren.parameters(), lr=lr_max)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=c.steps_warm, eta_min=lr_min
        )

        print(f"[shape-siren] obj_{obj_i:02d}  fine-tuning for {c.steps_warm} steps  ({N:,} samples)")
        print(f"  SIREN: {siren.num_params():,} params  |  FULL-BATCH  |  L1 loss")

        best_loss = float("inf")
        best_sd = None

        for step in range(c.steps_warm):
            # full-batch: all points every step
            pred = siren(all_pts)
            pred_clamped = pred.clamp(-c.truncation, c.truncation)

            loss = F.l1_loss(pred_clamped, all_sdf_clamped)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

            # track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_sd = {k: v.clone() for k, v in siren.state_dict().items()}

            if step % 100 == 0 or step == c.steps_warm - 1:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  step {step:5d}  l1 {loss.item():.6f}  best {best_loss:.6f}  lr {cur_lr:.2e}")

        # save best weights
        torch.save(best_sd, out_path)
        print(f"[shape-siren] saved -> {out_path}  (best l1: {best_loss:.6f})")

    print(f"\n[shape-siren] done — 10 shape-SIRENs saved to {c.out_dir}")


if __name__ == "__main__":
    main()