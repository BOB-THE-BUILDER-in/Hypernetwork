"""
Dump meshes from already-trained ω₀=15 shape-SIRENs.

Reads:
  /workspace/hypernet/checkpoints/anchor_shape_siren_w15.pt
  /workspace/hypernet/data/shape_sirens_w15/obj_NN.pt

Writes:
  /workspace/hypernet/data/shape_sirens_w15/meshes/anchor_sphere.obj
  /workspace/hypernet/data/shape_sirens_w15/meshes/obj_NN.obj
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/hypernet")
from src.siren import SIREN


OMEGA = 15.0
CKPT_DIR = Path("/workspace/hypernet/data/shape_sirens_w15")
MESH_DIR = CKPT_DIR / "meshes"
ANCHOR = Path("/workspace/hypernet/checkpoints/anchor_shape_siren_w15.pt")

# Architecture matches CFG.shape_siren
IN_DIM, OUT_DIM = 3, 1
HIDDEN, N_LAYERS = 256, 5


def build_siren(sd, device):
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    net = SIREN(IN_DIM, OUT_DIM, HIDDEN, N_LAYERS, OMEGA, OMEGA).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


@torch.no_grad()
def query_and_mesh(net, device, out_path: Path, res: int = 256, bound: float = 1.0):
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    chunk = 65536
    for i in range(0, pts.shape[0], chunk):
        s = net(pts[i:i + chunk])
        if s.dim() > 1:
            s = s.squeeze(-1)
        out[i:i + chunk] = s
    vol = out.reshape(res, res, res).cpu().numpy()

    from skimage.measure import marching_cubes
    if not (vol.min() <= 0.0 <= vol.max()):
        print(f"  {out_path.name}: SDF range {vol.min():.4f}..{vol.max():.4f} does NOT cross zero — EMPTY")
        return

    spacing = (2 * bound / (res - 1),) * 3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    v = v - bound

    with open(out_path, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for tri in f:
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"  {out_path.name}: {v.shape[0]} verts / {f.shape[0]} faces  sdf {vol.min():.4f}..{vol.max():.4f}")


def main():
    device = torch.device("cuda")
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    # Anchor
    if ANCHOR.exists():
        print("[mesh] anchor")
        sd = torch.load(ANCHOR, map_location=device, weights_only=True)
        net = build_siren(sd, device)
        query_and_mesh(net, device, MESH_DIR / "anchor_sphere.obj")

    # Per-object
    for pt in sorted(CKPT_DIR.glob("obj_*.pt")):
        print(f"[mesh] {pt.stem}")
        sd = torch.load(pt, map_location=device, weights_only=True)
        net = build_siren(sd, device)
        query_and_mesh(net, device, MESH_DIR / f"{pt.stem}.obj")

    print(f"\n[done] meshes in {MESH_DIR}")


if __name__ == "__main__":
    main()
