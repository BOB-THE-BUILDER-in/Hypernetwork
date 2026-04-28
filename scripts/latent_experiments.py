"""
Latent-space experiments on the 10 trained hypernets.

Experiment 1: PCA on hypernet residuals
    - Confirms whether the 10 hypernets lie on a low-dim manifold.
    - Reports explained variance per component.
    - Saves 2D projection with labels for visual inspection.

Experiment 2: Linear interpolation between two shapes in hypernet-space
    - Picks two training hypernets (default obj_03 and obj_06).
    - Walks linearly: h(t) = (1-t)*h_a + t*h_b for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0].
    - Feeds each interpolated hypernet to the mapper.
    - Reconstructs shape-SIREN, runs marching cubes, saves mesh.
    - A smooth morph from shape A to shape B is the core evidence that
      weight space is functioning as a latent space.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --- utilities (same as other scripts) --------------------------------------

def _unwrap(o):
    if isinstance(o, dict) and "state_dict" in o and isinstance(o["state_dict"], dict):
        return o["state_dict"]
    return o


def flat(path, keys):
    sd = _unwrap(torch.load(path, map_location="cpu"))
    return torch.cat([sd[k].detach().float().flatten() for k in keys])


def unflatten(flat_vec, names, shapes):
    sd = {}
    offset = 0
    for n, s in zip(names, shapes):
        size = 1
        for d in s:
            size *= d
        sd[n] = flat_vec[offset:offset + size].view(*s).clone()
        offset += size
    return sd


def build_siren(sd, device, hidden=256, n_layers=5, omega=30.0):
    from src.siren import SIREN
    sd = _unwrap(sd)
    for kw in [
        dict(in_dim=3, out_dim=1, hidden=hidden, n_layers=n_layers, omega=omega),
        dict(in_features=3, out_features=1, hidden_features=hidden,
             hidden_layers=n_layers, first_omega_0=omega, hidden_omega_0=omega),
    ]:
        try:
            net = SIREN(**kw); break
        except TypeError:
            net = None
    if net is None:
        net = SIREN(3, 1, hidden, n_layers)
    net.load_state_dict(sd)
    net.to(device).eval()
    return net


@torch.no_grad()
def query_sdf_grid(net, device, res=256, bound=1.0, chunk=65536):
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    for i in range(0, pts.shape[0], chunk):
        s = net(pts[i:i + chunk])
        if s.dim() > 1:
            s = s.squeeze(-1)
        out[i:i + chunk] = s
    return out.reshape(res, res, res).cpu().numpy()


def sdf_to_mesh(volume, bound=1.0, level=0.0):
    from skimage.measure import marching_cubes
    if not (volume.min() <= level <= volume.max()):
        return None, None
    res = volume.shape[0]
    spacing = (2 * bound / (res - 1),) * 3
    v, f, _, _ = marching_cubes(volume, level=level, spacing=spacing)
    return v - bound, f


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


# --- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/workspace/hypernet/scripts/manifest.pt")
    ap.add_argument("--mapper_ckpt", default="/workspace/hypernet/scripts/t1/mapper.pt")
    ap.add_argument("--out_dir", default="/workspace/hypernet/data/latent_experiments")
    ap.add_argument("--obj_a", type=int, default=3, help="first endpoint for interpolation")
    ap.add_argument("--obj_b", type=int, default=6, help="second endpoint for interpolation")
    ap.add_argument("--steps", type=int, default=6, help="interpolation steps including endpoints")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=5)
    ap.add_argument("--omega", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/hypernet")
    sys.path.insert(0, "/workspace/hypernet/scripts")

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.mapper_ckpt, map_location=device, weights_only=False)
    manifest = torch.load(args.manifest, map_location="cpu", weights_only=False)

    hyp_keys = ckpt["hyp_names"]
    shp_names = ckpt["shp_names"]
    shp_shapes = ckpt["shp_shapes"]

    # Load all 10 hypernet residuals (absolute - anchor)
    anchor_hyp_cpu = ckpt["anchor_hyp"].cpu()
    hyps = torch.stack([flat(p, hyp_keys) for p in manifest["hypernet_paths"]])  # (10, D)
    hyps_res = hyps - anchor_hyp_cpu[None]                                        # (10, D)

    # ========================================================================
    # EXPERIMENT 1: PCA on hypernet residuals
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: PCA on hypernet residuals")
    print("=" * 60)

    # Use SVD on residuals. Center again for safety (they should be near 0-mean
    # since anchor ≈ empirical mean, but do it to be correct).
    X = hyps_res - hyps_res.mean(0, keepdim=True)  # (10, D)
    # For 10 points in 17.9M dim, do SVD on X @ X.T (10x10) then derive components
    gram = X @ X.T                                  # (10, 10)
    eigvals, eigvecs = torch.linalg.eigh(gram)      # ascending
    eigvals = eigvals.flip(0)                       # descending
    eigvecs = eigvecs.flip(1)                       # match

    # Singular values and explained variance ratio
    sv = eigvals.clamp_min(0).sqrt()
    total_var = eigvals.clamp_min(0).sum()
    explained = eigvals.clamp_min(0) / total_var

    print(f"{'comp':>5s}  {'singular':>12s}  {'expl_var':>10s}  {'cumul':>8s}")
    cum = 0.0
    for i, (s, e) in enumerate(zip(sv, explained)):
        cum += e.item()
        print(f"  {i:2d}   {s.item():12.4f}  {e.item()*100:9.3f}%  {cum*100:7.2f}%")

    # 2D projection: scores on first 2 components
    # Scores = U * S where X = U S V^T, so U = eigvecs of X X^T / sqrt(eigvals)
    scores_2d = eigvecs[:, :2] * sv[:2][None, :]    # (10, 2)
    print("\n2D PCA projection (obj_NN -> [PC1, PC2]):")
    for i, s in enumerate(scores_2d):
        print(f"  obj_{i:02d}:  [{s[0].item():+8.4f},  {s[1].item():+8.4f}]")

    # Save PCA stats
    np.savez(out_dir / "pca_stats.npz",
             singular_values=sv.numpy(),
             explained_variance_ratio=explained.numpy(),
             scores_2d=scores_2d.numpy())
    print(f"\n[saved] PCA stats -> {out_dir/'pca_stats.npz'}")

    # ========================================================================
    # EXPERIMENT 2: Linear interpolation between obj_A and obj_B
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 2: Interpolation obj_{args.obj_a:02d} <-> obj_{args.obj_b:02d}")
    print("=" * 60)

    from hypernet_to_shape_mapper import HypernetToShapeMapper
    a = ckpt["args"]
    model = HypernetToShapeMapper(
        shape_dim=ckpt["shp_mean"].shape[1],
        cond_dim=ckpt["hyp_mean"].shape[1],
        chunk_shape=a["chunk_shape"], chunk_cond=a["chunk_cond"],
        d_model=a["d_model"], n_layers=a["n_layers"], n_heads=a["n_heads"],
        ff_mult=a["ff_mult"], cond_enc_layers=a["cond_enc_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # Endpoints in raw hypernet space
    h_a = hyps[args.obj_a].to(device)   # (D,)
    h_b = hyps[args.obj_b].to(device)   # (D,)

    # Everything the mapper needs
    anchor_hyp = ckpt["anchor_hyp"].to(device)
    hyp_mean = ckpt["hyp_mean"].to(device)
    hyp_std  = ckpt["hyp_std"].to(device)
    anchor_shp = ckpt["anchor_shp"].to(device).unsqueeze(0)
    shp_mean = ckpt["shp_mean"].to(device)
    shp_std  = ckpt["shp_std"].to(device)

    ts = torch.linspace(0.0, 1.0, args.steps, device=device)
    interp_dir = out_dir / f"interp_{args.obj_a:02d}_to_{args.obj_b:02d}"
    interp_dir.mkdir(parents=True, exist_ok=True)

    for step_i, t in enumerate(ts):
        # Linear interpolation in raw hypernet space
        h_t = (1.0 - t) * h_a + t * h_b
        # Apply the exact preprocessing the mapper expects
        z = (h_t - anchor_hyp - hyp_mean.squeeze(0)) / hyp_std.squeeze(0)
        z = z.unsqueeze(0)

        with torch.no_grad():
            pred_norm = model(z)

        # Also report similarity to both endpoints in prediction space to see
        # whether the prediction actually moves as we change t
        pred_abs = (pred_norm * shp_std + shp_mean + anchor_shp).squeeze(0).cpu()

        # Marching cubes
        sd_pred = unflatten(pred_abs, shp_names, shp_shapes)
        net_pred = build_siren(sd_pred, device,
                               hidden=args.hidden, n_layers=args.n_layers, omega=args.omega)
        vol = query_sdf_grid(net_pred, device, res=args.res, bound=args.bound)
        v, f = sdf_to_mesh(vol, bound=args.bound)

        tag = f"t{t.item():.2f}".replace(".", "_")
        mesh_path = interp_dir / f"interp_{tag}.obj"
        if v is not None:
            save_obj(mesh_path, v, f)
            msg = f"  t={t.item():.2f}  verts={v.shape[0]}  faces={f.shape[0]}  sdf {vol.min():.3f}..{vol.max():.3f}"
        else:
            msg = f"  t={t.item():.2f}  EMPTY  sdf {vol.min():.3f}..{vol.max():.3f}"
        print(msg)

    # Also copy/save the endpoint GTs for side-by-side comparison
    for idx in [args.obj_a, args.obj_b]:
        shp_path = Path(manifest["shape_paths"][idx])
        gt_flat = flat(shp_path, shp_names)
        sd_gt = unflatten(gt_flat, shp_names, shp_shapes)
        net_gt = build_siren(sd_gt, device,
                             hidden=args.hidden, n_layers=args.n_layers, omega=args.omega)
        vol = query_sdf_grid(net_gt, device, res=args.res, bound=args.bound)
        v, f = sdf_to_mesh(vol, bound=args.bound)
        if v is not None:
            save_obj(interp_dir / f"gt_obj_{idx:02d}.obj", v, f)

    print(f"\n[done] interpolation meshes -> {interp_dir}")
    print(f"[done] open interp_t0_00.obj ... interp_t1_00.obj in order to see the morph")


if __name__ == "__main__":
    main()
