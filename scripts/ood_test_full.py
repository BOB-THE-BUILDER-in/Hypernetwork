"""
OOD generalization test: predict a never-seen shape from its image-SIRENs.

Pipeline:
    1. Download a new Objaverse shape (slot obj_100, untouched by training)
    2. Watertight it
    3. Render 24 views
    4. Sample SDF (so we can also overfit a shape-SIREN as a "ceiling" reference)
    5. Train 24 image-SIRENs (warm-start from anchor)
    6. Train 1 hypernet for it (warm-start from anchor)
    7. Feed the hypernet through the trained N=100 mapper
    8. Mesh the prediction
    9. Mesh GT (overfit shape-SIREN) for upper-bound reference
   10. Mesh the watertight ground-truth mesh as the geometric reference

Output meshes go to /workspace/hypernet/data/ood_test/meshes/:
    obj_100_gt.obj          - direct from watertight (true GT)
    obj_100_overfit.obj     - SIREN that overfit obj_100 (upper bound)
    obj_100_predicted.obj   - the actual generalization result we care about

Restart-safe: every stage is skip-if-exists.
"""
from __future__ import annotations

import argparse
import random
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path("/workspace/hypernet")
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SCRIPTS))

from configs.config import CFG
from src.siren import SIREN, flatten_weights
from src.hypernet import HyperNet
from src.render import render_views, scan_rig_directions
from src.watertight import watertight
from hypernet_to_shape_mapper import (
    HypernetToShapeMapper,
    ResidualPairedWeightsDataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJ_IDX = 100  # held-out slot
OBJ_TAG = f"obj_{OBJ_IDX:03d}"  # 3-digit so it doesn't collide with existing 2-digit naming
DOWNLOAD_TIMEOUT_SEC = 60

OOD_DIR = PROJECT / "data" / "ood_test"
MESH_DIR = OOD_DIR / "meshes"
OOD_GLB = OOD_DIR / f"{OBJ_TAG}.glb"
OOD_WT  = OOD_DIR / f"{OBJ_TAG}.obj"
OOD_VIEWS = OOD_DIR / OBJ_TAG / "views"
OOD_SDF = OOD_DIR / f"{OBJ_TAG}.npz"
OOD_IMG_SIRENS = OOD_DIR / OBJ_TAG / "image_sirens"
OOD_HYPERNET = OOD_DIR / f"{OBJ_TAG}_hypernet.pt"
OOD_OVERFIT_SHP = OOD_DIR / f"{OBJ_TAG}_shape_siren_overfit.pt"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage 1: download
# ---------------------------------------------------------------------------

class _Timeout:
    def __init__(self, sec): self.sec = sec
    def __enter__(self):
        def handler(s, f): raise TimeoutError("timeout")
        self._prev = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.sec)
    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._prev)


def fetch_one(uid):
    import objaverse
    try:
        with _Timeout(DOWNLOAD_TIMEOUT_SEC):
            r = objaverse.load_objects(uids=[uid])
        s = r.get(uid)
        if not s:
            return None
        p = Path(s)
        return p if p.exists() else None
    except Exception as e:
        return None


def download():
    if OOD_GLB.exists() and OOD_GLB.stat().st_size > 10_000:
        log(f"download: {OOD_GLB.name} already exists ({OOD_GLB.stat().st_size/1e6:.1f}MB), skipping")
        return

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    import objaverse
    all_uids = objaverse.load_uids()
    rng = random.Random(CFG.seed + 2000)  # different seed from training downloads
    candidates = rng.sample(all_uids, 200)

    log(f"download: trying up to 200 candidates for {OBJ_TAG}")
    for i, uid in enumerate(candidates):
        log(f"  candidate {i+1}: {uid[:12]}")
        src = fetch_one(uid)
        if src is None:
            continue
        try:
            shutil.copyfile(src, OOD_GLB)
        except Exception as e:
            log(f"    copy failed: {e}")
            continue
        if OOD_GLB.stat().st_size < 10_000:
            log(f"    too small, retry")
            continue
        log(f"  ok -> {OOD_GLB.name} ({OOD_GLB.stat().st_size/1e6:.1f}MB)")
        return
    raise RuntimeError("download: exhausted candidates without success")


# ---------------------------------------------------------------------------
# Stage 2: watertight
# ---------------------------------------------------------------------------

def watertight_stage():
    if OOD_WT.exists():
        log(f"watertight: {OOD_WT.name} exists, skip")
        return
    log(f"watertight: {OOD_GLB.name} -> {OOD_WT.name}")
    watertight(
        OOD_GLB, OOD_WT,
        resolution=384,
        scatter_count=3_000_000,
        particle_radius=2.0,
        smooth_sigma=0.8,
    )


# ---------------------------------------------------------------------------
# Stage 3: render views
# ---------------------------------------------------------------------------

def render_stage():
    if OOD_VIEWS.exists() and len(list(OOD_VIEWS.glob("view_*.png"))) == CFG.data.num_views:
        log(f"render: views exist, skip")
        return
    log(f"render: {OOD_WT.name} -> {OOD_VIEWS}")
    OOD_VIEWS.mkdir(parents=True, exist_ok=True)
    render_views(OOD_WT, OOD_VIEWS, num_views=CFG.data.num_views, img_res=CFG.data.img_res)


# ---------------------------------------------------------------------------
# Stage 4: SDF sampling (for the overfit reference SIREN)
# ---------------------------------------------------------------------------

def sdf_stage():
    if OOD_SDF.exists():
        log(f"sdf: {OOD_SDF.name} exists, skip")
        return

    log(f"sdf: sampling from {OOD_WT.name}")
    import trimesh
    mesh = trimesh.load(OOD_WT, force="mesh")

    # normalize to [-1,1] (matches 60_sample_sdf.py)
    verts = mesh.vertices.copy()
    center = (verts.max(0) + verts.min(0)) / 2
    verts -= center
    scale = np.abs(verts).max()
    verts /= scale * 1.1
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

    n_surf, n_space = 100_000, 50_000
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surf)
    surface_pts = surface_pts.astype(np.float32)
    n_s = surface_pts + np.random.randn(*surface_pts.shape).astype(np.float32) * 0.005
    n_m = surface_pts + np.random.randn(*surface_pts.shape).astype(np.float32) * 0.02
    n_l = surface_pts + np.random.randn(*surface_pts.shape).astype(np.float32) * 0.1
    uni = np.random.uniform(-1, 1, (n_space, 3)).astype(np.float32)
    all_pts = np.concatenate([surface_pts, n_s, n_m, n_l, uni], axis=0)

    from pysdf import SDF
    sdf_fn = SDF(mesh.vertices, mesh.faces)
    chunk = 200_000
    sdf = np.zeros(len(all_pts), dtype=np.float32)
    for i in range(0, len(all_pts), chunk):
        sdf[i:i+chunk] = -sdf_fn(all_pts[i:i+chunk]).astype(np.float32)

    sdf = np.clip(sdf, -CFG.data.sdf_truncation, CFG.data.sdf_truncation)
    np.savez_compressed(OOD_SDF, points=all_pts, sdf=sdf)
    log(f"sdf: saved {OOD_SDF.name}  pts={all_pts.shape}  sdf range=[{sdf.min():.3f},{sdf.max():.3f}]")


# ---------------------------------------------------------------------------
# Stage 5: 24 image-SIRENs
# ---------------------------------------------------------------------------

def image_siren_stage():
    OOD_IMG_SIRENS.mkdir(parents=True, exist_ok=True)
    c = CFG.img_siren
    dev = CFG.device

    anchor_path = c.anchor_path
    assert anchor_path.exists(), f"missing anchor {anchor_path}"
    anchor_sd = torch.load(anchor_path, map_location=dev, weights_only=True)["state_dict"]

    from PIL import Image

    for j in range(CFG.data.num_views):
        out = OOD_IMG_SIRENS / f"view_{j:02d}.pt"
        if out.exists():
            continue
        img_path = OOD_VIEWS / f"view_{j:02d}.png"
        if not img_path.exists():
            log(f"  view {j:02d}: missing image, skip")
            continue

        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        H, W, _ = img.shape
        ys = np.linspace(-1, 1, H, dtype=np.float32)
        xs = np.linspace(-1, 1, W, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        coords = torch.from_numpy(np.stack([gx, gy], -1).reshape(-1, 2)).to(dev)
        pixels = torch.from_numpy(img.reshape(-1, 3)).to(dev)

        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(dev)
        siren.load_state_dict(anchor_sd)
        opt = torch.optim.Adam(siren.parameters(), lr=c.lr)
        steps = c.steps_warm

        for step in range(steps):
            idx = torch.randint(0, coords.shape[0], (c.batch_pixels,), device=dev)
            pred = siren(coords[idx])
            loss = F.mse_loss(pred, pixels[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        torch.save(siren.state_dict(), out)
        log(f"  view {j:02d}: l2 {loss.item():.5f}  -> {out.name}")


# ---------------------------------------------------------------------------
# Stage 6: hypernet for the new shape
# ---------------------------------------------------------------------------

def hypernet_stage():
    if OOD_HYPERNET.exists():
        log(f"hypernet: {OOD_HYPERNET.name} exists, skip")
        return

    dev = CFG.device
    c = CFG.img_siren
    hc = CFG.hypernet

    # Load anchor hypernet
    anchor_path = PROJECT / "data" / "checkpoints" / "anchor_hypernet.pt"
    anchor_ckpt = torch.load(anchor_path, map_location=dev, weights_only=True)

    # Build reference SIREN to define HyperNet's target architecture
    ref = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers, c.w0_first, c.w0_hidden)
    hypernet = HyperNet(
        target_siren=ref, in_dim=3,
        head_hidden=hc.head_hidden, head_layers=hc.head_layers,
        final_init_scale=hc.final_init_scale,
    ).to(dev)
    hypernet.load_state_dict(anchor_ckpt["state_dict"])

    # Load 24 image-SIRENs as targets
    cam_dirs = torch.from_numpy(scan_rig_directions()).to(dev)
    weight_list = []
    for j in range(CFG.data.num_views):
        sp = OOD_IMG_SIRENS / f"view_{j:02d}.pt"
        s = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers, c.w0_first, c.w0_hidden).to(dev)
        s.load_state_dict(torch.load(sp, map_location=dev, weights_only=True))
        weight_list.append(flatten_weights(s))
    target_weights = torch.stack(weight_list)

    log(f"hypernet: training on {target_weights.shape[0]} target SIRENs, params={hypernet.num_params():,}")
    opt = torch.optim.Adam(hypernet.parameters(), lr=hc.lr_hypernet)

    for step in range(hc.steps):
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for j in range(cam_dirs.shape[0]):
            pred = hypernet(cam_dirs[j])
            loss = F.mse_loss(pred, target_weights[j])
            loss.backward()
            total_loss += loss.item()
        opt.step()
        if step % 200 == 0 or step == hc.steps - 1:
            log(f"  step {step:5d}  avg {total_loss / cam_dirs.shape[0]:.6e}")

    torch.save(hypernet.state_dict(), OOD_HYPERNET)
    log(f"hypernet: saved -> {OOD_HYPERNET.name}")


# ---------------------------------------------------------------------------
# Stage 7: ALSO overfit a shape-SIREN to obj_100 directly, for upper-bound
# reference. This shows what the SIREN representation could do if the mapper
# were perfect — useful for diagnosing whether holes come from the mapper or
# from the representation itself.
# ---------------------------------------------------------------------------

def overfit_shape_siren_stage():
    if OOD_OVERFIT_SHP.exists():
        log(f"overfit-siren: exists, skip")
        return

    dev = CFG.device
    c = CFG.shape_siren

    anchor_ckpt = torch.load(c.anchor_path, map_location=dev, weights_only=True)
    anchor_sd = anchor_ckpt["state_dict"]

    data = np.load(OOD_SDF)
    pts = torch.from_numpy(data["points"]).to(dev)
    sdf = torch.from_numpy(data["sdf"]).to(dev).unsqueeze(-1)
    sdf_cl = sdf.clamp(-c.truncation, c.truncation)

    siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                  c.w0_first, c.w0_hidden).to(dev)
    siren.load_state_dict(anchor_sd)
    opt = torch.optim.Adam(siren.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c.steps_warm, eta_min=1e-6)

    log(f"overfit-siren: full-batch L1 over {pts.shape[0]} pts for {c.steps_warm} steps")
    best_loss, best_sd = float("inf"), None
    for step in range(c.steps_warm):
        pred = siren(pts).clamp(-c.truncation, c.truncation)
        loss = F.l1_loss(pred, sdf_cl)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_sd = {k: v.clone() for k, v in siren.state_dict().items()}
        if step % 200 == 0 or step == c.steps_warm - 1:
            log(f"  step {step:5d}  l1 {loss.item():.5f}  best {best_loss:.5f}")

    torch.save(best_sd, OOD_OVERFIT_SHP)
    log(f"overfit-siren: saved (best l1 {best_loss:.5f})")


# ---------------------------------------------------------------------------
# Stage 8: predict from mapper, mesh everything
# ---------------------------------------------------------------------------

@torch.no_grad()
def mesh_siren(siren, device, out_path, res=256, bound=1.0):
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    chunk = 65536
    for i in range(0, pts.shape[0], chunk):
        s = siren(pts[i:i+chunk])
        if s.dim() > 1: s = s.squeeze(-1)
        out[i:i+chunk] = s
    vol = out.reshape(res, res, res).cpu().numpy()
    from skimage.measure import marching_cubes
    if not (vol.min() <= 0.0 <= vol.max()):
        log(f"  mesh: SDF range {vol.min():.3f}..{vol.max():.3f}, no zero crossing")
        return None
    spacing = (2*bound/(res-1),)*3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    v = v - bound
    with open(out_path, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for tri in f:
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    return v.shape[0], f.shape[0]


def predict_and_mesh_stage():
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    dev = CFG.device

    # 1) Mesh GT (the watertight mesh, normalized)
    gt_mesh_out = MESH_DIR / f"{OBJ_TAG}_GT_watertight.obj"
    if not gt_mesh_out.exists():
        import trimesh
        m = trimesh.load(OOD_WT, force="mesh")
        verts = m.vertices.copy()
        center = (verts.max(0)+verts.min(0))/2
        verts -= center
        verts /= np.abs(verts).max() * 1.1
        with open(gt_mesh_out, "w") as fh:
            for v in verts:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for f in m.faces:
                fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
        log(f"GT watertight (normalized) -> {gt_mesh_out.name}  ({len(verts)} v)")

    # 2) Mesh the overfit shape-SIREN (the upper bound)
    overfit_out = MESH_DIR / f"{OBJ_TAG}_overfit_siren.obj"
    if not overfit_out.exists() and OOD_OVERFIT_SHP.exists():
        c = CFG.shape_siren
        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(dev)
        siren.load_state_dict(torch.load(OOD_OVERFIT_SHP, map_location=dev, weights_only=True))
        siren.eval()
        res = mesh_siren(siren, dev, overfit_out)
        if res:
            log(f"overfit shape-SIREN -> {overfit_out.name}  ({res[0]} v)")

    # 3) Mesh the MAPPER PREDICTION (the actual generalization test)
    pred_out = MESH_DIR / f"{OBJ_TAG}_predicted_n100.obj"
    if pred_out.exists():
        log(f"prediction mesh exists at {pred_out.name}, skip")
        return

    # Load N=100 mapper
    t1 = SCRIPTS / "t1_n100"
    ckpt = torch.load(t1 / "mapper.pt", map_location=dev, weights_only=False)
    args = ckpt["args"]

    ds = ResidualPairedWeightsDataset(
        args["manifest"], args["anchor_hyp"], args["anchor_shp"], device=dev,
    )
    model = HypernetToShapeMapper(
        shape_dim=ds.shp_norm.shape[1], cond_dim=ds.hyp_norm.shape[1],
        chunk_shape=args["chunk_shape"], chunk_cond=args["chunk_cond"],
        d_model=args["d_model"], n_layers=args["n_layers"], n_heads=args["n_heads"],
        ff_mult=args["ff_mult"], cond_enc_layers=args["cond_enc_layers"],
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load OOD hypernet, flatten using same key order
    ood_hyp_sd = torch.load(OOD_HYPERNET, map_location=dev, weights_only=True)
    flat_parts = []
    for k in ds.hyp_names:
        flat_parts.append(ood_hyp_sd[k].detach().float().flatten())
    ood_hyp_flat = torch.cat(flat_parts).to(dev)
    log(f"OOD hypernet flattened: shape={tuple(ood_hyp_flat.shape)}")

    # Standardize residual the same way training data was
    ood_residual = ood_hyp_flat - ds.anchor_hyp.to(dev)
    ood_norm = (ood_residual - ds.hyp_mean.to(dev).squeeze()) / ds.hyp_std.to(dev).squeeze()

    # Sanity check: print norm of the standardized residual vs training set
    train_norms = (ds.hyp_norm ** 2).sum(dim=1).sqrt()
    ood_norm_mag = (ood_norm ** 2).sum().sqrt()
    log(f"OOD hyp standardized norm: {ood_norm_mag:.2f}")
    log(f"training hyp norms: min={train_norms.min():.2f} mean={train_norms.mean():.2f} max={train_norms.max():.2f}")
    if ood_norm_mag > train_norms.max() * 1.5:
        log("  WARNING: OOD hypernet is far outside training distribution")

    with torch.no_grad():
        pred_norm = model(ood_norm.unsqueeze(0))
        pred_abs = ds.reconstruct_shape_weights(pred_norm).cpu()

    # Unflatten into a SIREN
    c = CFG.shape_siren
    siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                  c.w0_first, c.w0_hidden).to(dev)
    sd = {}
    off = 0
    for n, s in zip(ds.shp_names, ds.shp_shapes):
        size = 1
        for d in s: size *= d
        sd[n] = pred_abs[0, off:off+size].view(*s).to(dev)
        off += size
    siren.load_state_dict(sd)
    siren.eval()

    res = mesh_siren(siren, dev, pred_out)
    if res:
        log(f"PREDICTED -> {pred_out.name}  ({res[0]} v / {res[1]} f)")
    else:
        log(f"PREDICTED: no zero crossing -- mapper output is off-manifold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_overfit", action="store_true",
                    help="skip the upper-bound overfit shape-SIREN")
    args = ap.parse_args()

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== OOD test pipeline for {OBJ_TAG} ===")

    download()
    watertight_stage()
    render_stage()
    sdf_stage()
    image_siren_stage()
    hypernet_stage()
    if not args.skip_overfit:
        overfit_shape_siren_stage()
    predict_and_mesh_stage()

    log(f"=== done. meshes in {MESH_DIR} ===")
    log("compare:")
    log(f"  GT:        {OBJ_TAG}_GT_watertight.obj")
    log(f"  upper bnd: {OBJ_TAG}_overfit_siren.obj")
    log(f"  predicted: {OBJ_TAG}_predicted_n100.obj  <-- the test")


if __name__ == "__main__":
    main()
