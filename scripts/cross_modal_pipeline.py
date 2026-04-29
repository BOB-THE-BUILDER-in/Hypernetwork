"""
Cross-modal shared latent manifold for SIREN weight spaces.

Architecture:
    EncoderImg(24 image-SIRENs) -> z_img (128)
    EncoderShp(1 shape-SIREN)   -> z_shp (128)
    DecoderImg(z, cam_dir)      -> image-SIREN weights
    DecoderShp(z)               -> shape-SIREN weights

Training losses:
    L_img_recon    : DecoderImg(z_img, cam_j) matches image_siren_ij
    L_shp_recon    : DecoderShp(z_shp) matches shape_siren_i
    L_align (NCE)  : z_img_i should be close to z_shp_i, far from z_shp_{j!=i}

For paired (image_set_i, shape_i), encoders should produce the SAME z.
This forces a single shared manifold where image and shape information coexist.

OOD inference: new image-SIRENs -> EncoderImg -> z -> DecoderShp -> shape-SIREN -> mesh.
No mapper. No test-time latent optimization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path("/workspace/hypernet")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from configs.config import CFG
from src.siren import SIREN, flatten_weights
from src.render import scan_rig_directions

OUT_DIR = PROJECT / "data" / "cross_modal_l2_n100"
MESH_DIR = OUT_DIR / "meshes"


# ============================================================================
# Encoders & decoders
# ============================================================================

class EncoderImg(nn.Module):
    """Average 24 image-SIRENs first, then MLP -> z."""
    def __init__(self, siren_dim, z_dim=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(siren_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, sirens_set):
        # sirens_set: (B, 24, siren_dim)
        avg = sirens_set.mean(dim=1)  # (B, siren_dim)
        return self.net(avg)


class EncoderShp(nn.Module):
    """Shape-SIREN weights -> z."""
    def __init__(self, shape_siren_dim, z_dim=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shape_siren_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, weights):
        # weights: (B, shape_siren_dim)
        return self.net(weights)


class DecoderImg(nn.Module):
    """(z, cam_dir) -> image-SIREN weights."""
    def __init__(self, z_dim, cam_dim, out_dim, hidden=512, num_layers=4):
        super().__init__()
        in_dim = z_dim + cam_dim
        layers = []
        for i in range(num_layers - 1):
            d_in = in_dim if i == 0 else hidden
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.GELU())
        self.body = nn.Sequential(*layers)
        self.final = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            self.final.weight.mul_(1e-2)
            self.final.bias.zero_()

    def forward(self, z, cam):
        x = torch.cat([z, cam], dim=-1)
        return self.final(self.body(x))


class DecoderShp(nn.Module):
    """z -> shape-SIREN weights."""
    def __init__(self, z_dim, out_dim, hidden=512, num_layers=4):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.final = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            self.final.weight.mul_(1e-2)
            self.final.bias.zero_()

    def forward(self, z):
        return self.final(self.body(z))


# ============================================================================
# Helpers
# ============================================================================

def _unwrap(o):
    if isinstance(o, dict) and "state_dict" in o and isinstance(o["state_dict"], dict):
        return o["state_dict"]
    return o


def flatten_sd(sd, keys=None):
    sd = _unwrap(sd)
    if keys is None:
        keys = list(sd.keys())
    flat = torch.cat([sd[k].detach().float().flatten() for k in keys])
    return flat, keys


def load_image_sirens(image_siren_dir, device):
    """Load all 100x24 image-SIRENs as (100, 24, siren_dim)."""
    image_siren_dir = Path(image_siren_dir)
    obj_dirs = sorted([d for d in image_siren_dir.iterdir()
                       if d.is_dir() and d.name.startswith("obj_")])
    assert len(obj_dirs) == 100, f"expected 100, got {len(obj_dirs)}"

    keys = None
    flats = []
    for i, obj_dir in enumerate(obj_dirs):
        obj_flats = []
        for j in range(CFG.data.num_views):
            sd = torch.load(obj_dir / f"view_{j:02d}.pt",
                            map_location="cpu", weights_only=True)
            flat, keys = flatten_sd(sd, keys)
            obj_flats.append(flat)
        flats.append(torch.stack(obj_flats))
        if (i + 1) % 25 == 0:
            print(f"  image-SIRENs: {i + 1}/100", flush=True)
    return torch.stack(flats).to(device), keys


def load_shape_sirens(shape_siren_dir, device):
    """Load all 100 shape-SIRENs as (100, shape_siren_dim)."""
    shape_siren_dir = Path(shape_siren_dir)
    paths = sorted(shape_siren_dir.glob("obj_*.pt"))
    assert len(paths) == 100, f"expected 100, got {len(paths)}"

    keys = None
    flats = []
    for i, p in enumerate(paths):
        sd = torch.load(p, map_location="cpu", weights_only=True)
        flat, keys = flatten_sd(sd, keys)
        flats.append(flat)
        if (i + 1) % 25 == 0:
            print(f"  shape-SIRENs: {i + 1}/100", flush=True)
    return torch.stack(flats).to(device), keys


# ============================================================================
# Training
# ============================================================================

def info_nce_loss(z_img, z_shp, temperature=0.1):
    """Symmetric InfoNCE: pull paired (z_img_i, z_shp_i) together,
    push unpaired apart. Standard CLIP-style loss."""
    # Normalize for cosine similarity
    z_img_n = F.normalize(z_img, dim=-1)
    z_shp_n = F.normalize(z_shp, dim=-1)

    # Similarity matrix (B, B)
    logits = z_img_n @ z_shp_n.T / temperature

    # Labels: diagonal indices (positive pairs)
    B = z_img.shape[0]
    labels = torch.arange(B, device=z_img.device)

    # Symmetric: image-to-shape and shape-to-image
    loss_i2s = F.cross_entropy(logits, labels)
    loss_s2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2s + loss_s2i)


def train(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("[load] image-SIRENs...")
    W_img, img_keys = load_image_sirens(PROJECT / "data" / "image_sirens", device)
    print(f"[load] image-SIRENs shape: {tuple(W_img.shape)}")
    img_siren_dim = W_img.shape[-1]

    print("[load] shape-SIRENs...")
    W_shp, shp_keys = load_shape_sirens(PROJECT / "data" / "shape_sirens", device)
    print(f"[load] shape-SIRENs shape: {tuple(W_shp.shape)}")
    shp_siren_dim = W_shp.shape[-1]

    # Anchor (use mean of training data) for residual parameterization
    img_anchor = W_img.mean(dim=(0, 1))  # (img_siren_dim,)
    shp_anchor = W_shp.mean(dim=0)        # (shp_siren_dim,)

    W_img_res = W_img - img_anchor[None, None, :]
    W_shp_res = W_shp - shp_anchor[None, :]

    # Per-dim standardization
    img_std = W_img_res.reshape(-1, img_siren_dim).std(0).clamp_min(1e-6)
    shp_std = W_shp_res.std(0).clamp_min(1e-6)

    W_img_norm = W_img_res / img_std[None, None, :]
    W_shp_norm = W_shp_res / shp_std[None, :]
    print(f"[load] normalized variance: img={W_img_norm.var().item():.3f}  "
          f"shp={W_shp_norm.var().item():.3f}")

    # Camera directions (fixed, known)
    cam_dirs = torch.from_numpy(scan_rig_directions()).float().to(device)  # (24, 3)

    # Build all 4 networks
    enc_img = EncoderImg(img_siren_dim, z_dim=args.z_dim, hidden=args.enc_hidden).to(device)
    enc_shp = EncoderShp(shp_siren_dim, z_dim=args.z_dim, hidden=args.enc_hidden).to(device)
    dec_img = DecoderImg(args.z_dim, cam_dim=3, out_dim=img_siren_dim,
                         hidden=args.dec_hidden, num_layers=args.dec_layers).to(device)
    dec_shp = DecoderShp(args.z_dim, out_dim=shp_siren_dim,
                         hidden=args.dec_hidden, num_layers=args.dec_layers).to(device)

    n_params = sum(sum(p.numel() for p in m.parameters())
                   for m in [enc_img, enc_shp, dec_img, dec_shp])
    print(f"[arch] total params: {n_params:,} (~{n_params*4/1e6:.1f} MB)")
    print(f"  enc_img: {sum(p.numel() for p in enc_img.parameters()):,}")
    print(f"  enc_shp: {sum(p.numel() for p in enc_shp.parameters()):,}")
    print(f"  dec_img: {sum(p.numel() for p in dec_img.parameters()):,}")
    print(f"  dec_shp: {sum(p.numel() for p in dec_shp.parameters()):,}")

    opt = torch.optim.AdamW(
        list(enc_img.parameters()) + list(enc_shp.parameters())
        + list(dec_img.parameters()) + list(dec_shp.parameters()),
        lr=args.lr,
        weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    print(f"[train] {args.steps} steps, batch={args.batch_size}, "
          f"loss weights: img={args.w_img:.2f} shp={args.w_shp:.2f} align={args.w_align:.2f}")

    import random as _rnd
    rng = _rnd.Random(0)

    for step in range(1, args.steps + 1):
        for m in [enc_img, enc_shp, dec_img, dec_shp]:
            m.train()

        idx = rng.sample(range(100), args.batch_size)
        bs = len(idx)

        x_img = W_img_norm[idx]                      # (bs, 24, img_dim)
        x_shp = W_shp_norm[idx]                      # (bs, shp_dim)

        z_img = enc_img(x_img)                       # (bs, z_dim)
        z_shp = enc_shp(x_shp)                       # (bs, z_dim)

        # Image reconstruction: dec_img(z_img, cam) should match each of 24 views
        z_img_b = z_img.unsqueeze(1).expand(bs, 24, -1).reshape(bs * 24, -1)
        cam_b = cam_dirs.unsqueeze(0).expand(bs, 24, -1).reshape(bs * 24, -1)
        target_img = x_img.reshape(bs * 24, img_siren_dim)
        pred_img = dec_img(z_img_b, cam_b)
        L_img = F.mse_loss(pred_img, target_img)

        # Shape reconstruction: dec_shp(z_shp) should match shape SIREN
        pred_shp = dec_shp(z_shp)
        L_shp = F.mse_loss(pred_shp, x_shp)

        # Alignment: z_img_i should match z_shp_i (contrastive)
        L_align = F.mse_loss(z_img, z_shp)  # L2 alignment, preserves magnitude

        loss = args.w_img * L_img + args.w_shp * L_shp + args.w_align * L_align

        opt.zero_grad(set_to_none=True)
        loss.backward()
        for m in [enc_img, enc_shp, dec_img, dec_shp]:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            # Diagnostic: cosine similarity of paired latents
            with torch.no_grad():
                z_img_n = F.normalize(z_img, dim=-1)
                z_shp_n = F.normalize(z_shp, dim=-1)
                cos_sim = (z_img_n * z_shp_n).sum(-1).mean().item()
            print(f"  step {step:5d}  L_img {L_img.item():.4e}  "
                  f"L_shp {L_shp.item():.4e}  L_align {L_align.item():.4e}  "
                  f"cos_sim {cos_sim:.3f}  lr {sched.get_last_lr()[0]:.2e}")

    # Save everything
    torch.save({
        "enc_img": enc_img.state_dict(),
        "enc_shp": enc_shp.state_dict(),
        "dec_img": dec_img.state_dict(),
        "dec_shp": dec_shp.state_dict(),
        "args": vars(args),
        "img_keys": img_keys,
        "shp_keys": shp_keys,
        "img_anchor": img_anchor.cpu(),
        "shp_anchor": shp_anchor.cpu(),
        "img_std": img_std.cpu(),
        "shp_std": shp_std.cpu(),
        "img_siren_dim": img_siren_dim,
        "shp_siren_dim": shp_siren_dim,
        "cam_dirs": cam_dirs.cpu(),
    }, OUT_DIR / "cross_modal.pt")
    print(f"[save] -> {OUT_DIR/'cross_modal.pt'}")

    # Compute per-object latents (from both modalities) for analysis
    enc_img.eval(); enc_shp.eval()
    with torch.no_grad():
        Z_img_all = enc_img(W_img_norm).cpu()
        Z_shp_all = enc_shp(W_shp_norm).cpu()

    # Diagnostic: how well-aligned are train latents?
    with torch.no_grad():
        zi = F.normalize(Z_img_all, dim=-1)
        zs = F.normalize(Z_shp_all, dim=-1)
        # Diagonal cos sim
        diag = (zi * zs).sum(-1)
        # Off-diagonal cos sim
        off = zi @ zs.T
        off_diag = off - torch.diag(diag)
        print(f"\n[align] paired cos sim:    mean {diag.mean():.3f}  min {diag.min():.3f}")
        print(f"[align] unpaired cos sim:  mean {(off_diag.sum() / (100*99)):.3f}")

    torch.save(Z_img_all, OUT_DIR / "z_img_all.pt")
    torch.save(Z_shp_all, OUT_DIR / "z_shp_all.pt")

    return enc_img, enc_shp, dec_img, dec_shp


# ============================================================================
# OOD inference
# ============================================================================

def ood_test(enc_img, dec_shp, ckpt, args, device):
    print("\n[ood] testing on obj_101 via cross-modal pipeline")
    ood_dir = PROJECT / "data" / "ood_test" / "obj_101" / "image_sirens"
    if not ood_dir.exists():
        print(f"[ood] {ood_dir} missing")
        return

    img_keys = ckpt["img_keys"]
    img_anchor = ckpt["img_anchor"].to(device)
    img_std = ckpt["img_std"].to(device)

    flats = []
    for j in range(CFG.data.num_views):
        sd = torch.load(ood_dir / f"view_{j:02d}.pt",
                        map_location="cpu", weights_only=True)
        flat, _ = flatten_sd(sd, img_keys)
        flats.append(flat)
    sirens = torch.stack(flats).to(device)  # (24, img_dim)
    sirens_norm = (sirens - img_anchor[None, :]) / img_std[None, :]
    sirens_norm = sirens_norm.unsqueeze(0)  # (1, 24, img_dim)

    enc_img.eval()
    dec_shp.eval()

    with torch.no_grad():
        z = enc_img(sirens_norm)             # (1, z_dim)
        print(f"[ood] image-encoded z norm: {z.norm().item():.3f}")
        recon_norm = dec_shp(z).cpu()        # (1, shp_dim)

    # Denormalize back to absolute SIREN weights
    shp_anchor = ckpt["shp_anchor"].cpu()
    shp_std = ckpt["shp_std"].cpu()
    recon_abs = recon_norm * shp_std[None, :] + shp_anchor[None, :]

    # Build shape-SIREN from predicted weights and mesh
    sc = CFG.shape_siren
    siren = SIREN(sc.in_dim, sc.out_dim, sc.hidden_dim, sc.num_layers,
                  sc.w0_first, sc.w0_hidden).to(device)
    sd = {}
    off = 0
    for k, s in zip(ckpt["shp_keys"], [siren.state_dict()[k].shape for k in ckpt["shp_keys"]]):
        n = 1
        for d in s:
            n *= d
        sd[k] = recon_abs[0, off:off+n].view(*s).to(device)
        off += n
    siren.load_state_dict(sd)
    siren.eval()

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MESH_DIR / "ood_obj_101_cross_modal.obj"

    res, bound = 256, 1.0
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    with torch.no_grad():
        for i in range(0, pts.shape[0], 65536):
            s = siren(pts[i:i+65536])
            if s.dim() > 1:
                s = s.squeeze(-1)
            out[i:i+65536] = s
    vol = out.reshape(res, res, res).cpu().numpy()
    from skimage.measure import marching_cubes
    if not (vol.min() <= 0.0 <= vol.max()):
        print("[ood] no zero crossing")
        return
    spacing = (2*bound/(res-1),)*3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    v = v - bound
    with open(out_path, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for tri in f:
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"[ood] PREDICTED: {v.shape[0]} v / {f.shape[0]} f -> {out_path.name}")


def reconstruct_training(enc_img, dec_shp, ckpt, device):
    """Sanity check: encode some training images, decode to shape, mesh.
    Should produce coherent meshes for training objects."""
    print("\n[recon] sanity check on training objects via image->shape pipeline")

    img_keys = ckpt["img_keys"]
    img_anchor = ckpt["img_anchor"].to(device)
    img_std = ckpt["img_std"].to(device)
    shp_anchor = ckpt["shp_anchor"].cpu()
    shp_std = ckpt["shp_std"].cpu()

    enc_img.eval(); dec_shp.eval()

    sc = CFG.shape_siren

    for obj_idx in [0, 10, 42, 67, 88]:
        # Load image SIRENs for this object
        d = PROJECT / "data" / "image_sirens" / f"obj_{obj_idx:02d}"
        flats = []
        for j in range(CFG.data.num_views):
            sd = torch.load(d / f"view_{j:02d}.pt",
                            map_location="cpu", weights_only=True)
            flat, _ = flatten_sd(sd, img_keys)
            flats.append(flat)
        sirens = torch.stack(flats).to(device)
        sirens_norm = ((sirens - img_anchor[None, :]) / img_std[None, :]).unsqueeze(0)

        with torch.no_grad():
            z = enc_img(sirens_norm)
            recon_norm = dec_shp(z).cpu()
        recon_abs = recon_norm * shp_std[None, :] + shp_anchor[None, :]

        siren = SIREN(sc.in_dim, sc.out_dim, sc.hidden_dim, sc.num_layers,
                      sc.w0_first, sc.w0_hidden).to(device)
        sd = {}
        off = 0
        for k in ckpt["shp_keys"]:
            shape = siren.state_dict()[k].shape
            n = 1
            for dd in shape:
                n *= dd
            sd[k] = recon_abs[0, off:off+n].view(*shape).to(device)
            off += n
        siren.load_state_dict(sd)
        siren.eval()

        res, bound = 192, 1.0
        lin = torch.linspace(-bound, bound, res, device=device)
        xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
        out = torch.empty(pts.shape[0], device=device)
        with torch.no_grad():
            for i in range(0, pts.shape[0], 65536):
                s = siren(pts[i:i+65536])
                if s.dim() > 1:
                    s = s.squeeze(-1)
                out[i:i+65536] = s
        vol = out.reshape(res, res, res).cpu().numpy()
        from skimage.measure import marching_cubes
        if not (vol.min() <= 0.0 <= vol.max()):
            print(f"  obj_{obj_idx:02d}: no zero crossing")
            continue
        spacing = (2*bound/(res-1),)*3
        v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
        v = v - bound
        out_path = MESH_DIR / f"train_recon_obj_{obj_idx:02d}.obj"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            for vv in v:
                fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
            for tri in f:
                fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        print(f"  obj_{obj_idx:02d}: {v.shape[0]} v / {f.shape[0]} f")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--enc_hidden", type=int, default=512)
    p.add_argument("--dec_hidden", type=int, default=512)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--w_img", type=float, default=1.0)
    p.add_argument("--w_shp", type=float, default=1.0)
    p.add_argument("--w_align", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    enc_img, enc_shp, dec_img, dec_shp = train(args, device)

    # Reload ckpt to get the full saved metadata
    ckpt = torch.load(OUT_DIR / "cross_modal.pt", map_location=device, weights_only=False)

    reconstruct_training(enc_img, dec_shp, ckpt, device)
    ood_test(enc_img, dec_shp, ckpt, args, device)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
