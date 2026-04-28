"""
Mesh predictions from the N=100 mapper.

Loads predicted_shape_weights.pt (saved by eval_n100_mapper.py) and writes
out meshes for selected shapes so we can compare to the N=10 mapper
predictions visually.

Outputs to /workspace/hypernet/scripts/t1_n100/meshes/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/hypernet")
from configs.config import CFG
from src.siren import SIREN

T1_N100 = Path("/workspace/hypernet/scripts/t1_n100")
MESH_DIR = T1_N100 / "meshes"

# Shapes to mesh: the two we have baselines for, plus best/worst-MSE shapes
# for full coverage of the spectrum.
SHAPES_TO_MESH = [
    3,   # weave (we have w30 N=10 baseline)
    6,   # humanoid (we have w30 N=10 baseline)
    22,  # min MSE in training set (0.139)
    36,  # max MSE in training set (0.377)
    49,  # other low-MSE
    70,  # other high-MSE
]


def unflatten_into_siren(flat_weights, keys, shapes, hidden_dim, num_layers, w0_first, w0_hidden, device):
    siren = SIREN(3, 1, hidden_dim, num_layers, w0_first, w0_hidden).to(device)
    sd = {}
    off = 0
    for n, s in zip(keys, shapes):
        size = 1
        for d in s:
            size *= d
        sd[n] = flat_weights[off:off + size].view(*s).to(device)
        off += size
    siren.load_state_dict(sd)
    siren.eval()
    return siren


@torch.no_grad()
def dump_mesh(siren, device, out_path, res=256, bound=1.0):
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
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
        print(f"  scikit-image missing, skip {out_path.name}")
        return
    if not (vol.min() <= 0.0 <= vol.max()):
        print(f"  {out_path.name}: SDF range {vol.min():.4f}..{vol.max():.4f}, no zero crossing")
        return
    spacing = (2 * bound / (res - 1),) * 3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    v = v - bound
    with open(out_path, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for tri in f:
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"  {out_path.name}: {v.shape[0]} verts / {f.shape[0]} faces")


def main():
    device = torch.device("cuda")
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    # Read mapper.pt for keys and shapes (we need to unflatten correctly)
    ckpt = torch.load(T1_N100 / "mapper.pt", map_location="cpu", weights_only=False)
    keys = ckpt["shp_names"]
    shp_shapes = ckpt["shp_shapes"]

    # SIREN architecture from CFG
    c = CFG.shape_siren
    hidden_dim = c.hidden_dim
    num_layers = c.num_layers
    w0f, w0h = c.w0_first, c.w0_hidden
    print(f"[arch] hidden={hidden_dim} layers={num_layers} w0_first={w0f} w0_hidden={w0h}")

    # Predicted absolute weights (100, D)
    pred_abs = torch.load(T1_N100 / "predicted_shape_weights.pt",
                          map_location="cpu", weights_only=True)
    print(f"[load] predicted_shape_weights {tuple(pred_abs.shape)}")

    print(f"\n[mesh] writing predictions for shapes {SHAPES_TO_MESH}")
    for idx in SHAPES_TO_MESH:
        if idx >= pred_abs.shape[0]:
            print(f"  skip obj_{idx:02d}: out of range")
            continue
        flat = pred_abs[idx]
        siren = unflatten_into_siren(flat, keys, shp_shapes,
                                     hidden_dim, num_layers, w0f, w0h, device)
        dump_mesh(siren, device, MESH_DIR / f"obj_{idx:02d}_pred_n100.obj")

    print(f"\n[done] meshes -> {MESH_DIR}")
    print("\nFor comparison, equivalent N=10 prediction meshes should be in:")
    print("  /workspace/hypernet/scripts/t1/meshes/  (if eval was run)")
    print("Watertight GT meshes:")
    print("  /workspace/hypernet/data/watertight/obj_03.obj  etc.")


if __name__ == "__main__":
    main()
