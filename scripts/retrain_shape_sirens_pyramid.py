"""
Pyramid SIREN: per-layer w0, otherwise identical to the existing SIREN.

Parameter names and tensor shapes match src/siren.py exactly, so:
  - flatten / unflatten from src/siren.py work unchanged
  - saved state_dicts can be loaded into either class (the only difference is
    the w0 attribute on each SineLayer, which lives on the module not in the
    state_dict)

Usage:
    schedule = [15.0, 15.0, 20.0, 30.0, 30.0]
    siren = SIRENPyramid(3, 1, 256, schedule)

This file also contains training functions for the pyramid anchor and
per-object shape-SIRENs. Run directly:
    python retrain_shape_sirens_pyramid.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/hypernet")
from configs.config import CFG


# ----------------------------------------------------------------------------
# Pyramid SIREN
# ----------------------------------------------------------------------------

class SineLayer(nn.Module):
    """Identical to the one in src/siren.py -- we re-declare so this file is
    self-contained and the w0 can be set explicitly per layer."""
    def __init__(self, in_f: int, out_f: int, is_first: bool, w0: float):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self.in_f = in_f
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_f
            else:
                bound = math.sqrt(6.0 / in_f) / w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class SIRENPyramid(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int,
                 w0_schedule: list[float]):
        super().__init__()
        assert len(w0_schedule) >= 2, "need at least 2 layers"
        self.cfg = dict(
            in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim,
            num_layers=len(w0_schedule),
            w0_schedule=list(map(float, w0_schedule)),
        )
        layers = []
        for i, w0 in enumerate(w0_schedule):
            if i == 0:
                layers.append(SineLayer(in_dim, hidden_dim, is_first=True, w0=w0))
            else:
                layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, w0=w0))
        self.net = nn.Sequential(*layers)

        # Final linear init uses the *last* layer's w0 for the bound scale.
        # This matches the intuition: the final output integrates from the
        # last sinusoidal layer, which runs at the highest frequency.
        self.final = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / w0_schedule[-1]
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.zero_()

    def forward(self, x):
        return self.final(self.net(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# Training: anchor + per-object, same recipe as 80_train_shape_sirens.py
# ----------------------------------------------------------------------------

W0_SCHEDULE = [15.0, 15.0, 20.0, 30.0, 30.0]
TAG = "pyr_" + "_".join(str(int(w)) for w in W0_SCHEDULE)   # "pyr_15_15_20_30_30"

ANCHOR_PATH = Path("/workspace/hypernet/checkpoints") / f"anchor_shape_siren_{TAG}.pt"
OUT_DIR = Path("/workspace/hypernet/data") / f"shape_sirens_{TAG}"
MESH_DIR = OUT_DIR / "meshes"


def sphere_sdf(pts, r=0.6):
    return pts.norm(dim=-1, keepdim=True) - r


@torch.no_grad()
def dump_mesh(siren, tag, dev, res: int = 256, bound: float = 1.0):
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    lin = torch.linspace(-bound, bound, res, device=dev)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=dev)
    chunk = 65536
    for i in range(0, pts.shape[0], chunk):
        s = siren(pts[i:i + chunk])
        if s.dim() > 1:
            s = s.squeeze(-1)
        out[i:i + chunk] = s
    vol = out.reshape(res, res, res).cpu().numpy()

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print(f"  [mesh] scikit-image missing, skip {tag}")
        return
    if not (vol.min() <= 0.0 <= vol.max()):
        print(f"  [mesh] {tag}: SDF range {vol.min():.4f}..{vol.max():.4f}, no zero crossing")
        return
    spacing = (2 * bound / (res - 1),) * 3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    v = v - bound
    p = MESH_DIR / f"{tag}.obj"
    with open(p, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for tri in f:
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"  [mesh] {tag}: {v.shape[0]} v / {f.shape[0]} f -> {p}")


def train_anchor():
    dev = CFG.device
    c = CFG.shape_siren

    if ANCHOR_PATH.exists():
        print(f"[anchor@{TAG}] already exists at {ANCHOR_PATH}, skipping")
        return

    ANCHOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    siren = SIRENPyramid(c.in_dim, c.out_dim, c.hidden_dim, W0_SCHEDULE).to(dev)
    opt = torch.optim.Adam(siren.parameters(), lr=c.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c.steps_anchor, eta_min=1e-6)

    print(f"[anchor@{TAG}] SIREN: {siren.num_params():,} params, w0 schedule {W0_SCHEDULE}")
    print(f"[anchor@{TAG}] training on unit sphere for {c.steps_anchor} steps, L1")

    for step in range(c.steps_anchor):
        pts = torch.empty(c.batch_points, 3, device=dev).uniform_(-1.0, 1.0)
        pred = siren(pts)
        tgt = sphere_sdf(pts).clamp(-c.truncation, c.truncation)
        loss = F.l1_loss(pred.clamp(-c.truncation, c.truncation), tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            print(f"  step {step:5d}  l1 {loss.item():.6f}")

    torch.save({"state_dict": siren.state_dict(), "cfg": siren.cfg}, ANCHOR_PATH)
    print(f"[anchor@{TAG}] saved -> {ANCHOR_PATH}")
    dump_mesh(siren, "anchor_sphere", dev)


def train_per_object():
    dev = CFG.device
    c = CFG.shape_siren
    d = CFG.data

    assert ANCHOR_PATH.exists(), "run train_anchor() first"
    ck = torch.load(ANCHOR_PATH, map_location=dev, weights_only=True)
    anchor_sd = ck["state_dict"]
    print(f"[shape@{TAG}] loaded anchor from {ANCHOR_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_max, lr_min = 1e-4, 1e-6

    for obj_i in range(d.num_objects):
        out_path = OUT_DIR / f"obj_{obj_i:02d}.pt"
        if out_path.exists():
            print(f"[shape@{TAG}] skip obj_{obj_i:02d}")
            continue

        sdf_path = d.sdf_dir / f"obj_{obj_i:02d}.npz"
        assert sdf_path.exists(), f"missing {sdf_path}"
        data = np.load(sdf_path)
        all_pts = torch.from_numpy(data["points"]).to(dev)
        all_sdf = torch.from_numpy(data["sdf"]).to(dev).unsqueeze(-1)
        N = all_pts.shape[0]
        all_sdf_cl = all_sdf.clamp(-c.truncation, c.truncation)

        siren = SIRENPyramid(c.in_dim, c.out_dim, c.hidden_dim, W0_SCHEDULE).to(dev)
        siren.load_state_dict(anchor_sd)
        opt = torch.optim.Adam(siren.parameters(), lr=lr_max)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c.steps_warm, eta_min=lr_min)

        print(f"[shape@{TAG}] obj_{obj_i:02d}  fine-tuning {c.steps_warm} steps ({N:,} pts)")

        best_loss, best_sd = float("inf"), None
        for step in range(c.steps_warm):
            pred = siren(all_pts)
            loss = F.l1_loss(pred.clamp(-c.truncation, c.truncation), all_sdf_cl)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_sd = {k: v.clone() for k, v in siren.state_dict().items()}
            if step % 100 == 0 or step == c.steps_warm - 1:
                print(f"  step {step:5d}  l1 {loss.item():.6f}  best {best_loss:.6f}  lr {sched.get_last_lr()[0]:.2e}")

        torch.save(best_sd, out_path)
        print(f"[shape@{TAG}] saved -> {out_path}  (best l1 {best_loss:.6f})")
        siren.load_state_dict(best_sd)
        dump_mesh(siren, f"obj_{obj_i:02d}", dev)

    print(f"\n[shape@{TAG}] done -> {OUT_DIR}")


if __name__ == "__main__":
    train_anchor()
    train_per_object()
