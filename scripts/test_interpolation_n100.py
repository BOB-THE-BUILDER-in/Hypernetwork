"""
Interpolation test: hypernet-space interpolation through the N=100 mapper.

For each pair of training hypernets, walks t in {0, 0.25, 0.5, 0.75, 1.0} in
residual hypernet space, feeds each interpolated hypernet through the mapper,
meshes the result. The decisive observation is whether intermediate t values
produce valid (clean, connected) shapes.

At N=10 we saw t=0.4 fall off the manifold for obj_03 <-> obj_06 (1758 verts,
disconnected components, no recognizable shape). Now with N=100 the mapper
should have learned a more continuous function; mid-points should be clean.

Outputs to /workspace/hypernet/scripts/t1_n100/interpolation/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/hypernet")
sys.path.insert(0, "/workspace/hypernet/scripts")

from configs.config import CFG
from src.siren import SIREN
from hypernet_to_shape_mapper import (
    HypernetToShapeMapper,
    ResidualPairedWeightsDataset,
)

T1_N100 = Path("/workspace/hypernet/scripts/t1_n100")
OUT_DIR = T1_N100 / "interpolation"

# Pairs to interpolate between. Pick from the LOW-MSE pool so endpoints are
# clean; that way any failure at midpoints is the manifold's fault, not a
# bad endpoint.
# From the per-shape MSE eval, the lowest-MSE shapes were:
#   22 (0.139), 49 (0.143), 85 (0.143), 18 (0.162), 45 (0.164), 81 (0.161)
# Also include the obj_03 <-> obj_06 pair we tested at N=10 for direct
# comparison of the same trajectory.
PAIRS = [
    (3, 6),     # same as N=10 test (the one that fell off manifold at t=0.4)
    (22, 49),   # both very-low-MSE, should be smoothest
    (18, 81),   # both clean, different categories
    (22, 70),   # low-MSE -> high-MSE; should we see degradation as we approach 70?
]
TS = [0.0, 0.25, 0.5, 0.75, 1.0]


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

    from skimage.measure import marching_cubes
    if not (vol.min() <= 0.0 <= vol.max()):
        print(f"  {out_path.name}: SDF doesn't cross zero, no mesh")
        return None
    spacing = (2 * bound / (res - 1),) * 3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    v = v - bound
    with open(out_path, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for tri in f:
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    return v.shape[0], f.shape[0]


def main():
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and dataset
    ckpt = torch.load(T1_N100 / "mapper.pt", map_location=device, weights_only=False)
    args = ckpt["args"]
    ds = ResidualPairedWeightsDataset(
        args["manifest"], args["anchor_hyp"], args["anchor_shp"], device=device,
    )

    model = HypernetToShapeMapper(
        shape_dim=ds.shp_norm.shape[1],
        cond_dim=ds.hyp_norm.shape[1],
        chunk_shape=args["chunk_shape"], chunk_cond=args["chunk_cond"],
        d_model=args["d_model"], n_layers=args["n_layers"], n_heads=args["n_heads"],
        ff_mult=args["ff_mult"], cond_enc_layers=args["cond_enc_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Architecture for unflattening predicted shape weights
    keys = ckpt["shp_names"]
    shp_shapes = ckpt["shp_shapes"]
    c = CFG.shape_siren

    # We need standardized hyp_residuals as model inputs.
    # ds.hyp_norm = (hyp_residual - hyp_mean) / hyp_std
    # ds.shp_norm reconstruct happens via ds.reconstruct_shape_weights.
    hyp_norm_cpu = ds.hyp_norm   # (100, cond_dim) on CPU

    print(f"\n[interp] running interpolation for {len(PAIRS)} pairs at t={TS}")
    print(f"[interp] outputs -> {OUT_DIR}\n")

    for (a, b) in PAIRS:
        print(f"=== pair {a:02d} <-> {b:02d} ===")
        h_a = hyp_norm_cpu[a].to(device)
        h_b = hyp_norm_cpu[b].to(device)
        for t in TS:
            h_t = (1 - t) * h_a + t * h_b
            with torch.no_grad():
                pred_norm = model(h_t.unsqueeze(0))                       # (1, shape_dim)
                pred_abs  = ds.reconstruct_shape_weights(pred_norm).cpu()  # absolute weights
            siren = unflatten_into_siren(pred_abs[0], keys, shp_shapes,
                                         c.hidden_dim, c.num_layers,
                                         c.w0_first, c.w0_hidden, device)
            tag = f"interp_{a:02d}_to_{b:02d}_t{int(t*100):03d}"
            res = dump_mesh(siren, device, OUT_DIR / f"{tag}.obj")
            if res is None:
                print(f"  t={t:.2f}: no mesh (off manifold)")
            else:
                v, f = res
                print(f"  t={t:.2f}: {v:>7d} verts / {f:>7d} faces -> {tag}.obj")
        print()

    print(f"[done] meshes in {OUT_DIR}")


if __name__ == "__main__":
    main()
