"""
Formulation B: Shared SIREN auto-decoder for image side.

Architecture:
    SharedDecoder(z_obj, cam_dir, xy) -> RGB
    where z_obj is per-object 128-dim latent (learned),
          cam_dir is 3-dim camera direction (given per view),
          xy is 2-dim pixel coordinate.

Training:
    For each (obj_i, view_j), use the existing image-SIREN S_ij as the target.
    Sample pixel coords, query S_ij to get target RGB, train Decoder to match.

Per-object latents and decoder weights trained jointly via backprop.

Stages:
    1. Train shared decoder + 100 latents (~30-60 min on RTX 3060+)
    2. Train tiny mapper: image_latent (128) -> shape_latent (128)
    3. OOD test: optimize fresh latent for obj_100, push through pipeline -> mesh

This replaces the entire hypernet + hypernet-AE path with a single end-to-end
trained shared decoder. Memory cost is trivial (~few hundred MB GPU).
"""
from __future__ import annotations

import argparse
import math
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
from src.siren import SIREN
from src.render import scan_rig_directions

OUT_DIR = PROJECT / "data" / "auto_decoder_n100"
MESH_DIR = OUT_DIR / "meshes"


# ============================================================================
# Shared SIREN auto-decoder
# ============================================================================

class SineLayer(nn.Module):
    """Single SIREN layer: y = sin(w0 * (Wx + b))."""
    def __init__(self, in_dim, out_dim, w0=30.0, is_first=False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_dim, out_dim)
        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_dim, 1 / in_dim)
            else:
                bound = math.sqrt(6 / in_dim) / w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class SharedDecoder(nn.Module):
    """SIREN: (z_obj, cam_dir, xy) -> RGB.

    Concatenates inputs and runs through a stack of SineLayers.
    """
    def __init__(self, latent_dim=128, cam_dim=3, xy_dim=2,
                 hidden=256, num_layers=5, w0_first=30.0, w0_hidden=30.0,
                 out_dim=3):
        super().__init__()
        in_dim = latent_dim + cam_dim + xy_dim
        layers = [SineLayer(in_dim, hidden, w0=w0_first, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden, hidden, w0=w0_hidden, is_first=False))
        self.body = nn.Sequential(*layers)
        # Final linear (no sine), small init
        self.final = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6 / hidden) / w0_hidden
            self.final.weight.uniform_(-bound, bound)

    def forward(self, z, cam, xy):
        # Broadcast: z (B, latent), cam (B, 3), xy (B, 2)
        x = torch.cat([z, cam, xy], dim=-1)
        h = self.body(x)
        return self.final(h)


# ============================================================================
# Tiny mapper: image_latent -> shape_latent (same as before)
# ============================================================================

class LatentMapper(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================================
# Helpers
# ============================================================================

def load_image_sirens(image_siren_dir, device):
    """Load all 100x24 image-SIRENs to GPU. Each is small (~66K params)."""
    c = CFG.img_siren
    image_siren_dir = Path(image_siren_dir)
    obj_dirs = sorted([d for d in image_siren_dir.iterdir()
                       if d.is_dir() and d.name.startswith("obj_")])
    assert len(obj_dirs) == 100, f"expected 100 obj dirs, got {len(obj_dirs)}"

    # Load each image-SIREN's state dict; we'll build/use them on demand
    # to save GPU memory. Storing all 2400 SIREN modules at once would be
    # ~2400 * 66K * 4 = 0.6 GB which is fine, so let's just keep them resident.
    sirens = [[None] * CFG.data.num_views for _ in range(100)]
    for i, obj_dir in enumerate(obj_dirs):
        for j in range(CFG.data.num_views):
            path = obj_dir / f"view_{j:02d}.pt"
            sd = torch.load(path, map_location=device, weights_only=True)
            siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                          c.w0_first, c.w0_hidden).to(device)
            siren.load_state_dict(sd)
            for p in siren.parameters():
                p.requires_grad_(False)
            siren.eval()
            sirens[i][j] = siren
        if (i + 1) % 25 == 0:
            print(f"  loaded {i + 1}/100", flush=True)
    return sirens


def sample_pixels(n, device):
    """Random uniform pixel coords in [-1, 1]^2."""
    return torch.rand(n, 2, device=device) * 2 - 1


@torch.no_grad()
def query_image_siren(siren, xy):
    """Get target RGB from an image-SIREN at given pixel coords."""
    return siren(xy)


# ============================================================================
# Stage 1: train shared decoder + 100 latents jointly
# ============================================================================

def train_auto_decoder(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[ad] loading 100x24 image-SIRENs to GPU as targets...")
    sirens = load_image_sirens(PROJECT / "data" / "image_sirens", device)

    cam_dirs = torch.from_numpy(scan_rig_directions()).float().to(device)  # (24, 3)
    print(f"[ad] cam_dirs: {tuple(cam_dirs.shape)}")

    # Build shared decoder
    decoder = SharedDecoder(
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        num_layers=args.num_layers,
        w0_first=30.0, w0_hidden=30.0,
    ).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"[ad] decoder params: {n_dec:,} ({n_dec*4/1e6:.2f} MB)")

    # Per-object latents (learnable)
    latents = nn.Parameter(
        torch.randn(100, args.latent_dim, device=device) * 0.01
    )
    print(f"[ad] latents: shape {tuple(latents.shape)}")

    # Joint optimizer over decoder + latents
    opt = torch.optim.AdamW(
        [{"params": decoder.parameters(), "lr": args.lr_decoder},
         {"params": [latents], "lr": args.lr_latent}],
        weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=1e-6,
    )

    print(f"[ad] training: {args.steps} steps, "
          f"per-step batch = {args.objs_per_step} objs * {args.views_per_obj} views * "
          f"{args.pixels_per_view} pixels = {args.objs_per_step*args.views_per_obj*args.pixels_per_view:,} samples")

    import random as _rnd
    rng = _rnd.Random(0)

    for step in range(1, args.steps + 1):
        decoder.train()

        # Sample objects, views, and pixels
        obj_idx = rng.sample(range(100), args.objs_per_step)
        view_idx = [rng.sample(range(CFG.data.num_views), args.views_per_obj)
                    for _ in range(args.objs_per_step)]

        # Build big batch
        xs, zs, cams, targets = [], [], [], []
        for oi, ois in enumerate(obj_idx):
            for vi in view_idx[oi]:
                # Sample pixels
                xy = sample_pixels(args.pixels_per_view, device)
                # Get target RGB from image-SIREN
                with torch.no_grad():
                    tgt = sirens[ois][vi](xy)
                # Build inputs for decoder
                z = latents[ois].unsqueeze(0).expand(args.pixels_per_view, -1)
                cam = cam_dirs[vi].unsqueeze(0).expand(args.pixels_per_view, -1)
                xs.append(xy); zs.append(z); cams.append(cam); targets.append(tgt)

        xy_b = torch.cat(xs, dim=0)
        z_b = torch.cat(zs, dim=0)
        cam_b = torch.cat(cams, dim=0)
        tgt_b = torch.cat(targets, dim=0)

        pred = decoder(z_b, cam_b, xy_b)
        loss = F.mse_loss(pred, tgt_b)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_([latents], 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  "
                  f"lr_dec {sched.get_last_lr()[0]:.2e}  lr_lat {sched.get_last_lr()[1]:.2e}")

    # Save
    torch.save({
        "decoder": decoder.state_dict(),
        "latents": latents.detach().cpu(),
        "args": vars(args),
        "cam_dirs": cam_dirs.cpu(),
    }, OUT_DIR / "auto_decoder.pt")
    print(f"[ad] saved -> {OUT_DIR/'auto_decoder.pt'}")

    # Also save latents as a clean tensor for downstream
    Z_image = latents.detach().cpu()
    torch.save(Z_image, OUT_DIR / "image_latents.pt")
    print(f"[ad] image latents shape={tuple(Z_image.shape)} "
          f"var per-dim min={Z_image.var(0).min():.3e} max={Z_image.var(0).max():.3e}")

    return decoder, Z_image, sirens, cam_dirs


# ============================================================================
# Stage 2: train mapper image_latent -> shape_latent
# ============================================================================

def train_latent_mapper(args, Z_image, device):
    print("\n[mapper] training image_latent -> shape_latent")
    Z_shape = torch.load(
        PROJECT / "data" / "ae_n100_mlp" / "latents.pt",
        map_location="cpu", weights_only=True,
    )
    print(f"[mapper] image latents: {tuple(Z_image.shape)}  "
          f"shape latents: {tuple(Z_shape.shape)}")
    assert Z_image.shape[0] == Z_shape.shape[0]

    Zi = Z_image.to(device)
    Zs = Z_shape.to(device)

    mapper = LatentMapper(Z_image.shape[1], Z_shape.shape[1]).to(device)
    print(f"[mapper] params: {sum(p.numel() for p in mapper.parameters()):,}")
    opt = torch.optim.AdamW(mapper.parameters(), lr=args.mapper_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.mapper_steps, eta_min=1e-6,
    )

    print(f"[mapper] full-batch on N=100, {args.mapper_steps} steps, lr={args.mapper_lr}")
    for step in range(1, args.mapper_steps + 1):
        mapper.train()
        pred = mapper(Zi)
        loss = F.mse_loss(pred, Zs)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 1 or step % max(1, args.mapper_steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")

    mapper.eval()
    with torch.no_grad():
        per = np.array([((mapper(Zi[i:i+1]) - Zs[i:i+1])**2).mean().item() for i in range(100)])
    print(f"\n[mapper] per-shape MSE: mean={per.mean():.4e} max={per.max():.4e}")

    perm = torch.randperm(100)
    with torch.no_grad():
        scram = np.array([((mapper(Zi[perm[i]:perm[i]+1]) - Zs[i:i+1])**2).mean().item() for i in range(100)])
        zero = np.array([((mapper(torch.zeros_like(Zi[i:i+1])) - Zs[i:i+1])**2).mean().item() for i in range(100)])
    print(f"[mapper] scrambled mean: {scram.mean():.4e}  ratio: {scram.mean()/per.mean():.2f}x")
    print(f"[mapper] zero-cond mean: {zero.mean():.4e}  ratio: {zero.mean()/per.mean():.2f}x")

    torch.save({
        "mapper": mapper.state_dict(),
        "in_dim": Z_image.shape[1], "out_dim": Z_shape.shape[1],
    }, OUT_DIR / "latent_mapper.pt")
    return mapper


# ============================================================================
# Stage 3: OOD test on obj_100
# ============================================================================

def optimize_ood_latent(decoder, ood_sirens, cam_dirs, args, device):
    """Hold decoder fixed, optimize a fresh 128-dim latent to fit obj_100's
    24 image-SIRENs. This is the equivalent of DeepSDF's test-time inference."""
    print("\n[ood] optimizing fresh latent for obj_100's 24 image-SIRENs...")

    z = nn.Parameter(torch.randn(1, args.latent_dim, device=device) * 0.01)
    opt = torch.optim.AdamW([z], lr=args.ood_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.ood_steps, eta_min=1e-6)
    decoder.eval()

    import random as _rnd
    rng = _rnd.Random(42)

    for step in range(1, args.ood_steps + 1):
        # Sample views and pixels for one optimization step
        view_idx = rng.sample(range(CFG.data.num_views), args.views_per_obj)
        xs, zs, cams, targets = [], [], [], []
        for vi in view_idx:
            xy = sample_pixels(args.pixels_per_view, device)
            with torch.no_grad():
                tgt = ood_sirens[vi](xy)
            zb = z.expand(args.pixels_per_view, -1)
            cam = cam_dirs[vi].unsqueeze(0).expand(args.pixels_per_view, -1)
            xs.append(xy); zs.append(zb); cams.append(cam); targets.append(tgt)

        xy_b = torch.cat(xs, dim=0)
        z_b = torch.cat(zs, dim=0)
        cam_b = torch.cat(cams, dim=0)
        tgt_b = torch.cat(targets, dim=0)

        pred = decoder(z_b, cam_b, xy_b)
        loss = F.mse_loss(pred, tgt_b)
        opt.zero_grad(set_to_none=True); loss.backward()
        opt.step(); sched.step()

        if step == 1 or step % max(1, args.ood_steps // 50) == 0:
            print(f"  ood step {step:5d}  mse {loss.item():.4e}  "
                  f"z_norm {z.norm().item():.3f}")

    return z.detach()


def ood_test(decoder, mapper, args, device):
    print("\n[ood] testing on obj_100 via auto-decoder pipeline")

    ood_dir = PROJECT / "data" / "ood_test" / "obj_100" / "image_sirens"
    if not ood_dir.exists():
        print(f"[ood] {ood_dir} missing, skip")
        return

    c = CFG.img_siren
    # Load obj_100's 24 image-SIRENs
    ood_sirens = []
    for j in range(CFG.data.num_views):
        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(device)
        siren.load_state_dict(torch.load(ood_dir / f"view_{j:02d}.pt",
                                         map_location=device, weights_only=True))
        for p in siren.parameters():
            p.requires_grad_(False)
        siren.eval()
        ood_sirens.append(siren)
    print(f"[ood] loaded 24 obj_100 image-SIRENs")

    cam_dirs = torch.from_numpy(scan_rig_directions()).float().to(device)
    z_image = optimize_ood_latent(decoder, ood_sirens, cam_dirs, args, device)
    print(f"[ood] optimized image latent norm: {z_image.norm().item():.3f}")

    # Push through mapper -> shape latent
    mapper.eval()
    with torch.no_grad():
        z_shape = mapper(z_image)
    print(f"[ood] shape latent norm: {z_shape.norm().item():.3f}")

    # Push through shape decoder -> SIREN -> mesh
    sys.path.insert(0, str(PROJECT / "scripts"))
    from autoencoder_pipeline_n100_mlp import MLPAE

    shape_ckpt = torch.load(
        PROJECT / "data" / "ae_n100_mlp" / "autoencoder.pt",
        map_location=device, weights_only=False,
    )
    shape_ae = MLPAE(shape_ckpt["D"],
                     latent_dim=shape_ckpt["args"]["latent_dim"],
                     hidden=shape_ckpt["args"]["hidden"]).to(device)
    shape_ae.load_state_dict(shape_ckpt["model"])
    shape_ae.eval()

    with torch.no_grad():
        recon_norm = shape_ae.decode(z_shape).cpu()
    recon_abs = (recon_norm * shape_ckpt["res_std"].cpu()
                 + shape_ckpt["res_mean"].cpu()
                 + shape_ckpt["anchor_flat"].cpu()[None])

    sc = CFG.shape_siren
    siren = SIREN(sc.in_dim, sc.out_dim, sc.hidden_dim, sc.num_layers,
                  sc.w0_first, sc.w0_hidden).to(device)
    sd = {}
    off = 0
    for k, s in zip(shape_ckpt["shp_keys"], shape_ckpt["shp_shapes"]):
        n = 1
        for d in s:
            n *= d
        sd[k] = recon_abs[0, off:off+n].view(*s).to(device)
        off += n
    siren.load_state_dict(sd)
    siren.eval()

    # Marching cubes
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MESH_DIR / "ood_obj_100_via_auto_decoder.obj"

    res, bound = 256, 1.0
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    chunk = 65536
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            s = siren(pts[i:i+chunk])
            if s.dim() > 1:
                s = s.squeeze(-1)
            out[i:i+chunk] = s
    vol = out.reshape(res, res, res).cpu().numpy()
    from skimage.measure import marching_cubes
    if not (vol.min() <= 0.0 <= vol.max()):
        print(f"[ood] PREDICTED: no zero crossing")
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


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    # Decoder + latents
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--objs_per_step", type=int, default=4,
                   help="number of objects sampled per training step")
    p.add_argument("--views_per_obj", type=int, default=6,
                   help="number of views sampled per object per step")
    p.add_argument("--pixels_per_view", type=int, default=512,
                   help="pixels sampled per (object, view) per step")
    p.add_argument("--lr_decoder", type=float, default=1e-4)
    p.add_argument("--lr_latent", type=float, default=1e-3)

    # Mapper
    p.add_argument("--mapper_lr", type=float, default=1e-3)
    p.add_argument("--mapper_steps", type=int, default=8000)

    # OOD inference
    p.add_argument("--ood_lr", type=float, default=1e-2)
    p.add_argument("--ood_steps", type=int, default=2000)

    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    decoder, Z_image, _, _ = train_auto_decoder(args, device)
    mapper = train_latent_mapper(args, Z_image, device)
    ood_test(decoder, mapper, args, device)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
