"""
End-to-end sphere test for the hypernet->shape mapper.

Stages:
    1. Generate a watertight icosphere mesh -> data/watertight/obj_10.obj
    2. Bump CFG.data.num_objects to 11 (in-memory only, not touching config.py)
    3. Run 03_render_views.py as a module -> 24 views of obj_10
    4. Run 20_train_image_sirens.py as a module -> 24 image SIRENs for obj_10
    5. Run 50_train_hypernets.py as a module -> hypernet for obj_10
    6. Load trained mapper (t1/mapper.pt), flatten obj_10's hypernet, predict
       shape-SIREN weights, save as data/shape_sirens/obj_10_PREDICTED.pt
    7. Run marching cubes, save mesh.

Assumes mapper checkpoint at /workspace/hypernet/scripts/t1/mapper.pt
(from the deterministic mapper training). Override with --mapper_ckpt.
"""
from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh

# Make project importable
PROJECT_ROOT = Path("/workspace/hypernet")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CFG


# ----------------------------------------------------------------------------
# Stage 1: generate the sphere as a watertight .obj
# ----------------------------------------------------------------------------

def make_sphere(out_path: Path, subdivisions: int = 4, radius: float = 0.5):
    """Icospheres are watertight by construction. radius=0.5 to fit in [-1,1]."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    assert mesh.is_watertight, "icosphere is always watertight"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    print(f"[sphere] wrote {out_path}  verts={len(mesh.vertices)}  faces={len(mesh.faces)}  watertight=True")


# ----------------------------------------------------------------------------
# Stage 6-7: mapper inference and mesh reconstruction
# ----------------------------------------------------------------------------

def _unwrap_sd(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def flatten_state_dict(sd, keys=None):
    sd = _unwrap_sd(sd)
    if keys is None:
        keys = list(sd.keys())
    tensors = [sd[k].detach().float().flatten() for k in keys]
    return torch.cat(tensors), keys, [tuple(sd[k].shape) for k in keys]


def unflatten(flat, names, shapes):
    sd = {}
    offset = 0
    for n, s in zip(names, shapes):
        size = 1
        for d in s:
            size *= d
        sd[n] = flat[offset:offset + size].view(*s).clone()
        offset += size
    return sd


def load_mapper(ckpt_path, device):
    """Dynamically import the mapper class from the training script."""
    # hypernet_to_shape_mapper.py lives in scripts/
    spec_path = Path("/workspace/hypernet/scripts")
    if str(spec_path) not in sys.path:
        sys.path.insert(0, str(spec_path))
    from hypernet_to_shape_mapper import HypernetToShapeMapper

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    # Infer dims from saved stats
    shape_dim = ckpt["shp_mean"].shape[1]
    cond_dim = ckpt["hyp_mean"].shape[1]

    model = HypernetToShapeMapper(
        shape_dim=shape_dim,
        cond_dim=cond_dim,
        chunk_shape=a["chunk_shape"],
        chunk_cond=a["chunk_cond"],
        d_model=a["d_model"],
        n_layers=a["n_layers"],
        n_heads=a["n_heads"],
        ff_mult=a["ff_mult"],
        cond_enc_layers=a["cond_enc_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def build_siren(sd, device, hidden=256, n_layers=5, omega=30.0):
    """Import the project's SIREN and construct from state dict."""
    from src.siren import SIREN
    sd = _unwrap_sd(sd)
    # Try the ctor signatures the project might use
    for kw in [
        dict(in_dim=3, out_dim=1, hidden=hidden, n_layers=n_layers, omega=omega),
        dict(in_features=3, out_features=1, hidden_features=hidden,
             hidden_layers=n_layers, first_omega_0=omega, hidden_omega_0=omega),
        dict(in_features=3, out_features=1, hidden_features=hidden,
             num_hidden_layers=n_layers, omega=omega),
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
    verts, faces, _, _ = marching_cubes(volume, level=level, spacing=spacing)
    return verts - bound, faces


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ----------------------------------------------------------------------------
# Utility: run one of the existing pipeline scripts with num_objects patched
# ----------------------------------------------------------------------------

def run_stage(script_path: Path, num_objects: int):
    print(f"\n============================================================")
    print(f"[stage] running {script_path.name} with num_objects={num_objects}")
    print(f"============================================================")
    # Patch CFG before executing the script so it sees the new count
    CFG.data.num_objects = num_objects
    # Also patch sys.argv so argparse-less scripts don't choke
    old_argv = sys.argv
    sys.argv = [str(script_path)]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_index", type=int, default=10,
                    help="slot index to assign the new object")
    ap.add_argument("--radius", type=float, default=0.5,
                    help="sphere radius in [0,1] scene bounds")
    ap.add_argument("--subdiv", type=int, default=4,
                    help="icosphere subdivision level (4 = 2562 verts)")
    ap.add_argument("--mapper_ckpt", type=str,
                    default="/workspace/hypernet/scripts/t1/mapper.pt")
    ap.add_argument("--scripts_dir", type=str, default="/workspace/hypernet/scripts")
    ap.add_argument("--out_dir", type=str,
                    default="/workspace/hypernet/data/sphere_test")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=5)
    ap.add_argument("--omega", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    idx = args.obj_index
    tag = f"obj_{idx:02d}"
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(args.scripts_dir)

    # -------------------- stage 1: watertight sphere --------------------
    wt_path = CFG.data.watertight_dir / f"{tag}.obj"
    if wt_path.exists():
        print(f"[sphere] watertight {tag} already exists, skipping mesh gen")
    else:
        make_sphere(wt_path, subdivisions=args.subdiv, radius=args.radius)

    # -------------------- stage 2: render 24 views ----------------------
    # 03_render_views.py iterates watertight/*.obj and writes per-stem
    # directories in views/. Since our obj_10.obj exists, it'll process it.
    # The script doesn't actually use num_objects for rendering (it iterates
    # everything in watertight_dir), but we bump it anyway for consistency.
    run_stage(scripts_dir / "03_render_views.py", num_objects=idx + 1)

    # -------------------- stage 3: train 24 image SIRENs ----------------
    run_stage(scripts_dir / "20_train_image_sirens.py", num_objects=idx + 1)

    # -------------------- stage 4: train hypernet -----------------------
    run_stage(scripts_dir / "50_train_hypernets.py", num_objects=idx + 1)

    # -------------------- stage 5: mapper inference ---------------------
    hyp_path = CFG.hypernet.out_dir / f"{tag}.pt"
    assert hyp_path.exists(), f"hypernet not found at {hyp_path}"
    print(f"\n[mapper] loading hypernet from {hyp_path}")

    print(f"[mapper] loading trained mapper from {args.mapper_ckpt}")
    model, ckpt = load_mapper(args.mapper_ckpt, device)

    # Flatten the new hypernet using the saved key order
    hyp_sd = torch.load(hyp_path, map_location="cpu")
    hyp_flat, _, _ = flatten_state_dict(hyp_sd, ckpt["hyp_names"])
    assert hyp_flat.numel() == ckpt["hyp_mean"].shape[1], \
        f"hypernet dim mismatch: got {hyp_flat.numel()}, expected {ckpt['hyp_mean'].shape[1]}"

    # Apply the same residual + standardization the mapper was trained with
    hyp_flat = hyp_flat.to(device).unsqueeze(0)          # (1, cond_dim)
    anchor_hyp = ckpt["anchor_hyp"].to(device).unsqueeze(0)
    hyp_mean = ckpt["hyp_mean"].to(device)
    hyp_std = ckpt["hyp_std"].to(device)
    hyp_res = hyp_flat - anchor_hyp
    hyp_norm = (hyp_res - hyp_mean) / hyp_std

    with torch.no_grad():
        pred_norm = model(hyp_norm)                      # (1, shape_dim)

    # Reconstruct absolute shape-SIREN weights
    shp_mean = ckpt["shp_mean"].to(device)
    shp_std = ckpt["shp_std"].to(device)
    anchor_shp = ckpt["anchor_shp"].to(device).unsqueeze(0)
    pred_abs = pred_norm * shp_std + shp_mean + anchor_shp
    pred_abs = pred_abs.squeeze(0).cpu()

    # Also produce the "anchor-only" baseline (i.e. what you'd get if the
    # mapper predicted zero residual) for comparison
    anchor_abs = ckpt["anchor_shp"]

    sd_pred = unflatten(pred_abs, ckpt["shp_names"], ckpt["shp_shapes"])
    sd_anchor = unflatten(anchor_abs, ckpt["shp_names"], ckpt["shp_shapes"])

    torch.save(sd_pred, out_dir / f"{tag}_predicted_shape_siren.pt")
    print(f"[save] predicted shape-SIREN -> {out_dir/f'{tag}_predicted_shape_siren.pt'}")

    # -------------------- stage 6: marching cubes ----------------------
    print("\n[eval] running marching cubes on predicted shape-SIREN")
    net_pred = build_siren(sd_pred, device,
                           hidden=args.hidden, n_layers=args.n_layers, omega=args.omega)
    vol_pred = query_sdf_grid(net_pred, device, res=args.res, bound=args.bound)
    v, f = sdf_to_mesh(vol_pred, bound=args.bound)
    if v is not None:
        save_obj(out_dir / f"{tag}_pred.obj", v, f)
        print(f"  pred: {v.shape[0]} v / {f.shape[0]} f  sdf {vol_pred.min():.3f}..{vol_pred.max():.3f}")
    else:
        print(f"  pred: EMPTY  sdf {vol_pred.min():.3f}..{vol_pred.max():.3f}")

    print("[eval] running marching cubes on anchor baseline (no prediction)")
    net_anchor = build_siren(sd_anchor, device,
                             hidden=args.hidden, n_layers=args.n_layers, omega=args.omega)
    vol_anchor = query_sdf_grid(net_anchor, device, res=args.res, bound=args.bound)
    v, f = sdf_to_mesh(vol_anchor, bound=args.bound)
    if v is not None:
        save_obj(out_dir / f"{tag}_anchor.obj", v, f)
        print(f"  anchor: {v.shape[0]} v / {f.shape[0]} f  sdf {vol_anchor.min():.3f}..{vol_anchor.max():.3f}")
    else:
        print(f"  anchor: EMPTY  sdf {vol_anchor.min():.3f}..{vol_anchor.max():.3f}")

    # Also copy the watertight reference for side-by-side
    import shutil
    shutil.copy(wt_path, out_dir / f"{tag}_gt.obj")
    print(f"[save] gt mesh copy -> {out_dir/f'{tag}_gt.obj'}")

    print(f"\n[done] all outputs in {out_dir}")
    print("  compare: {tag}_gt.obj (input)")
    print("           {tag}_pred.obj (mapper prediction)")
    print("           {tag}_anchor.obj (anchor baseline)")


if __name__ == "__main__":
    main()
