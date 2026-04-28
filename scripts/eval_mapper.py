"""
Eval the deterministic hypernet-to-shape mapper (or the rectified flow).

Loads predicted_shape_weights.pt from the output dir, unflattens into
shape-SIREN state_dicts using the metadata stored in the checkpoint, queries
each network on a 3D grid, runs marching cubes, and writes OBJ files for
predicted and ground-truth shapes side by side.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _unwrap_sd(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def unflatten(flat, names, shapes):
    sd = {}
    offset = 0
    for n, s in zip(names, shapes):
        size = 1
        for d in s:
            size *= d
        sd[n] = flat[offset:offset + size].view(*s).clone()
        offset += size
    assert offset == flat.numel()
    return sd


def load_siren_class(module_path, class_name):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def build_siren_from_state(SIRENCls, sd, device, hidden=256, n_layers=5, omega=30.0):
    sd = _unwrap_sd(sd)
    ctor_attempts = [
        dict(in_dim=3, out_dim=1, hidden=hidden, n_layers=n_layers, omega=omega),
        dict(in_features=3, out_features=1, hidden_features=hidden,
             hidden_layers=n_layers, first_omega_0=omega, hidden_omega_0=omega),
        dict(in_features=3, out_features=1, hidden_features=hidden,
             num_hidden_layers=n_layers, omega=omega),
    ]
    net = None
    last_err = None
    for kw in ctor_attempts:
        try:
            net = SIRENCls(**kw)
            break
        except TypeError as e:
            last_err = e
            continue
    if net is None:
        try:
            net = SIRENCls(3, 1, hidden, n_layers)
        except TypeError:
            raise RuntimeError(f"Could not construct SIREN with any known signature. Last error: {last_err}")
    net.load_state_dict(sd)
    net.to(device).eval()
    return net


@torch.no_grad()
def query_sdf_grid(net, device, res=128, bound=1.0, chunk=65536):
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
    vmin, vmax = volume.min(), volume.max()
    if not (vmin <= level <= vmax):
        return None, None
    res = volume.shape[0]
    spacing = (2 * bound / (res - 1),) * 3
    verts, faces, _, _ = marching_cubes(volume, level=level, spacing=spacing)
    verts = verts - bound
    return verts, faces


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="dir with mapper.pt + predicted_shape_weights.pt")
    ap.add_argument("--manifest", default="/workspace/hypernet/scripts/manifest.pt")
    ap.add_argument("--ckpt_name", default="mapper.pt", help="mapper.pt or rectified_flow.pt")
    ap.add_argument("--siren_module", default="src.siren")
    ap.add_argument("--siren_class", default="SIREN")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=5)
    ap.add_argument("--omega", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    project_root = Path("/workspace/hypernet")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    ckpt_path = out_dir / args.ckpt_name
    if not ckpt_path.exists():
        # fallback
        for alt in ["mapper.pt", "rectified_flow.pt"]:
            if (out_dir / alt).exists():
                ckpt_path = out_dir / alt
                break
    print(f"[eval] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pred_flat = torch.load(out_dir / "predicted_shape_weights.pt", map_location="cpu")

    shp_names = ckpt["shp_names"]
    shp_shapes = ckpt["shp_shapes"]

    manifest = torch.load(args.manifest, map_location="cpu", weights_only=False)
    gt_shape_paths = manifest["shape_paths"]

    SIRENCls = load_siren_class(args.siren_module, args.siren_class)

    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] {pred_flat.shape[0]} predictions, res={args.res}")
    for i in range(pred_flat.shape[0]):
        tag = Path(gt_shape_paths[i]).stem

        sd_pred = unflatten(pred_flat[i], shp_names, shp_shapes)
        net_pred = build_siren_from_state(SIRENCls, sd_pred, device,
                                          hidden=args.hidden, n_layers=args.n_layers,
                                          omega=args.omega)
        vol_pred = query_sdf_grid(net_pred, device, res=args.res, bound=args.bound)
        v, f = sdf_to_mesh(vol_pred, bound=args.bound)
        if v is not None:
            save_obj(mesh_dir / f"{tag}_pred.obj", v, f)
            pred_msg = f"pred: {v.shape[0]} v / {f.shape[0]} f  sdf {vol_pred.min():.3f}..{vol_pred.max():.3f}"
        else:
            pred_msg = f"pred: EMPTY  sdf {vol_pred.min():.3f}..{vol_pred.max():.3f}"

        sd_gt = torch.load(gt_shape_paths[i], map_location="cpu")
        net_gt = build_siren_from_state(SIRENCls, sd_gt, device,
                                        hidden=args.hidden, n_layers=args.n_layers,
                                        omega=args.omega)
        vol_gt = query_sdf_grid(net_gt, device, res=args.res, bound=args.bound)
        v, f = sdf_to_mesh(vol_gt, bound=args.bound)
        if v is not None:
            save_obj(mesh_dir / f"{tag}_gt.obj", v, f)
            gt_msg = f"gt: {v.shape[0]} v / {f.shape[0]} f"
        else:
            gt_msg = "gt: EMPTY"

        print(f"  [{i:2d}] {tag}  {pred_msg}  |  {gt_msg}")

    print(f"\n[done] meshes -> {mesh_dir}")


if __name__ == "__main__":
    main()
