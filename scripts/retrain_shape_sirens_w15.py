"""
Clean-slate retrain of shape-SIRENs at omega_0 = 15.

Preserves the existing ω₀=30 checkpoints untouched. New outputs go to
  checkpoints/anchor_shape_siren_w15.pt
  data/shape_sirens_w15/obj_NN.pt

After this runs, use train_mapper_w15.py to retrain the mapper on the new
shape-SIRENs (keeping the existing hypernets, which don't depend on ω₀).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/hypernet")
from configs.config import CFG
from src.siren import SIREN


# ----- overrides in-memory only --------------------------------------------
NEW_OMEGA = 15.0
NEW_ANCHOR_PATH = Path("/workspace/hypernet/checkpoints/anchor_shape_siren_w15.pt")
NEW_OUT_DIR    = Path("/workspace/hypernet/data/shape_sirens_w15")


def sphere_sdf(pts, r=0.6):
    return pts.norm(dim=-1, keepdim=True) - r


def train_anchor():
    dev = CFG.device
    c = CFG.shape_siren

    if NEW_ANCHOR_PATH.exists():
        print(f"[anchor-sdf@w{int(NEW_OMEGA)}] already exists at {NEW_ANCHOR_PATH}, skipping")
        return

    NEW_ANCHOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                  NEW_OMEGA, NEW_OMEGA).to(dev)
    opt = torch.optim.Adam(siren.parameters(), lr=c.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=c.steps_anchor, eta_min=1e-6
    )

    print(f"[anchor-sdf@w{int(NEW_OMEGA)}] SIREN: {siren.num_params():,} params  "
          f"(hidden={c.hidden_dim}, layers={c.num_layers}, omega_0={NEW_OMEGA})")
    print(f"[anchor-sdf@w{int(NEW_OMEGA)}] training on unit sphere for {c.steps_anchor} steps  |  L1 loss")

    for step in range(c.steps_anchor):
        pts = torch.empty(c.batch_points, 3, device=dev).uniform_(-1.0, 1.0)
        pred = siren(pts)
        target = sphere_sdf(pts).clamp(-c.truncation, c.truncation)
        pred_clamped = pred.clamp(-c.truncation, c.truncation)
        loss = F.l1_loss(pred_clamped, target)

        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); scheduler.step()

        if step % 500 == 0:
            print(f"  step {step:5d}  l1 {loss.item():.6f}")

    torch.save({"state_dict": siren.state_dict(), "cfg": siren.cfg}, NEW_ANCHOR_PATH)
    print(f"[anchor-sdf@w{int(NEW_OMEGA)}] saved -> {NEW_ANCHOR_PATH}")


def train_per_object():
    dev = CFG.device
    c = CFG.shape_siren
    d = CFG.data

    assert NEW_ANCHOR_PATH.exists(), "run train_anchor() first"
    ckpt = torch.load(NEW_ANCHOR_PATH, map_location=dev, weights_only=True)
    anchor_sd = ckpt["state_dict"]
    print(f"[shape-siren@w{int(NEW_OMEGA)}] loaded anchor from {NEW_ANCHOR_PATH}")

    NEW_OUT_DIR.mkdir(parents=True, exist_ok=True)

    lr_max, lr_min = 1e-4, 1e-6

    for obj_i in range(d.num_objects):
        out_path = NEW_OUT_DIR / f"obj_{obj_i:02d}.pt"
        if out_path.exists():
            print(f"[shape-siren@w{int(NEW_OMEGA)}] skip obj_{obj_i:02d}")
            continue

        sdf_path = d.sdf_dir / f"obj_{obj_i:02d}.npz"
        assert sdf_path.exists(), f"missing {sdf_path}"
        data = np.load(sdf_path)
        all_pts = torch.from_numpy(data["points"]).to(dev)
        all_sdf = torch.from_numpy(data["sdf"]).to(dev).unsqueeze(-1)
        N = all_pts.shape[0]
        all_sdf_clamped = all_sdf.clamp(-c.truncation, c.truncation)

        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      NEW_OMEGA, NEW_OMEGA).to(dev)
        siren.load_state_dict(anchor_sd)
        opt = torch.optim.Adam(siren.parameters(), lr=lr_max)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=c.steps_warm, eta_min=lr_min
        )

        print(f"[shape-siren@w{int(NEW_OMEGA)}] obj_{obj_i:02d}  "
              f"fine-tuning for {c.steps_warm} steps ({N:,} pts)")

        best_loss = float("inf"); best_sd = None
        for step in range(c.steps_warm):
            pred = siren(all_pts)
            pred_clamped = pred.clamp(-c.truncation, c.truncation)
            loss = F.l1_loss(pred_clamped, all_sdf_clamped)

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_sd = {k: v.clone() for k, v in siren.state_dict().items()}

            if step % 100 == 0 or step == c.steps_warm - 1:
                print(f"  step {step:5d}  l1 {loss.item():.6f}  best {best_loss:.6f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

        torch.save(best_sd, out_path)
        print(f"[shape-siren@w{int(NEW_OMEGA)}] saved -> {out_path}  (best l1 {best_loss:.6f})")

    print(f"\n[shape-siren@w{int(NEW_OMEGA)}] done — 10 SIRENs in {NEW_OUT_DIR}")


if __name__ == "__main__":
    train_anchor()
    train_per_object()
