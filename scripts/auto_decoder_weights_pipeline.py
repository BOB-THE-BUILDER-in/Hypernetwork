"""
Formulation B (SIREN-weight version): shared decoder outputs image-SIREN weights.

Architecture:
    SharedDecoder(z_obj, cam_dir) -> image-SIREN weights (~66K numbers)

Training data: 2400 existing image-SIRENs at data/image_sirens/obj_NN/view_JJ.pt
Each one is a known (object, view) -> SIREN weights tuple.

Training:
    For each (i, j) pair:
        z       = latents[i]          # 128-dim, learnable
        cam     = cam_dirs[j]         # 3-dim, fixed
        target  = image_siren_ij_weights  (frozen, on disk)
        pred    = SharedDecoder(z, cam)
        loss    = MSE(pred, target)

Decoder weights and 100 latents trained jointly.

Stages:
    1. Train shared decoder + 100 latents (~1-2 hours on RTX 3060+)
    2. Train tiny mapper: image_latent (128) -> shape_latent (128)
    3. OOD test: optimize fresh latent for obj_100, push through pipeline -> mesh
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

OUT_DIR = PROJECT / "data" / "auto_decoder_weights_n100"
MESH_DIR = OUT_DIR / "meshes"


# ============================================================================
# Shared decoder: (z_obj + cam_dir) -> SIREN weights
# ============================================================================

class SharedDecoder(nn.Module):
    """MLP that maps (z_obj concat cam_dir) -> flat image-SIREN weights."""
    def __init__(self, latent_dim, cam_dim, out_dim, hidden=512, num_layers=4):
        super().__init__()
        in_dim = latent_dim + cam_dim
        layers = []
        for i in range(num_layers - 1):
            d_in = in_dim if i == 0 else hidden
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.GELU())
        # final projection to SIREN weight dim — small init for stability
        # (predicted SIREN weights need to be near anchor for sin(w0*Wx) to work)
        self.body = nn.Sequential(*layers)
        self.final = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            self.final.weight.mul_(1e-2)
            self.final.bias.zero_()

    def forward(self, z, cam):
        # z: (B, latent_dim), cam: (B, cam_dim)
        x = torch.cat([z, cam], dim=-1)
        return self.final(self.body(x))


# ============================================================================
# Tiny mapper: image_latent -> shape_latent
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


def load_all_image_sirens(image_siren_dir, device):
    """Load all 2400 image-SIRENs as a single tensor (100, 24, siren_dim) on device.
    Returns also the key order so we can unflatten later.
    """
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
            print(f"  loaded {i + 1}/100", flush=True)
    W = torch.stack(flats).to(device)  # (100, 24, siren_dim)
    return W, keys


# ============================================================================
# Stage 1: train shared decoder + 100 latents
# ============================================================================

def train_auto_decoder(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[ad] loading 100x24 image-SIREN weights to GPU...")
    W, keys = load_all_image_sirens(PROJECT / "data" / "image_sirens", device)
    siren_dim = W.shape[-1]
    print(f"[ad] W shape: {tuple(W.shape)}  siren_dim={siren_dim:,}")

    # Anchor (image-SIREN anchor) for residual parameterization
    c = CFG.img_siren
    anchor_sd = torch.load(c.anchor_path, map_location=device, weights_only=True)
    anchor_flat, _ = flatten_sd(anchor_sd, keys)
    anchor_flat = anchor_flat.to(device)

    # Train decoder to predict residuals (target - anchor), keeps prediction near 0
    W_res = W - anchor_flat[None, None, :]
    # Standardize for stable training
    flat_view = W_res.reshape(-1, siren_dim)
    res_mean = flat_view.mean(0, keepdim=True)
    res_std = flat_view.std(0, keepdim=True).clamp_min(1e-6)
    W_norm = (W_res - res_mean.unsqueeze(0)) / res_std.unsqueeze(0)
    print(f"[ad] W_norm var: {W_norm.var().item():.4f}")

    cam_dirs = torch.from_numpy(scan_rig_directions()).float().to(device)  # (24, 3)

    # Build shared decoder
    decoder = SharedDecoder(
        latent_dim=args.latent_dim, cam_dim=3, out_dim=siren_dim,
        hidden=args.hidden, num_layers=args.num_layers,
    ).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"[ad] decoder params: {n_dec:,} (~{n_dec*4/1e6:.1f} MB fp32)")

    # Per-object latents
    latents = nn.Parameter(torch.randn(100, args.latent_dim, device=device) * 0.01)

    opt = torch.optim.AdamW(
        [{"params": decoder.parameters(), "lr": args.lr_decoder},
         {"params": [latents], "lr": args.lr_latent}],
        weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    bs = args.batch_objs
    print(f"[ad] training: {args.steps} steps, batch={bs} objs * 24 views = "
          f"{bs * 24} samples per step, lr_dec={args.lr_decoder} lr_lat={args.lr_latent}")

    import random as _rnd
    rng = _rnd.Random(0)

    for step in range(1, args.steps + 1):
        decoder.train()
        idx = rng.sample(range(100), bs)

        # Build batch: (bs * 24, ...)
        z = latents[idx].unsqueeze(1).expand(bs, 24, -1).reshape(bs * 24, -1)
        cam = cam_dirs.unsqueeze(0).expand(bs, 24, -1).reshape(bs * 24, -1)
        target = W_norm[idx].reshape(bs * 24, siren_dim)

        pred = decoder(z, cam)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_([latents], 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  "
                  f"lr_dec {sched.get_last_lr()[0]:.2e}")

    # Save
    torch.save({
        "decoder": decoder.state_dict(),
        "latents": latents.detach().cpu(),
        "args": vars(args),
        "siren_keys": keys,
        "anchor_flat": anchor_flat.cpu(),
        "res_mean": res_mean.cpu(),
        "res_std": res_std.cpu(),
        "siren_dim": siren_dim,
        "cam_dirs": cam_dirs.cpu(),
    }, OUT_DIR / "auto_decoder.pt")
    print(f"[ad] saved -> {OUT_DIR/'auto_decoder.pt'}")

    Z_image = latents.detach().cpu()
    torch.save(Z_image, OUT_DIR / "image_latents.pt")
    print(f"[ad] image latents shape={tuple(Z_image.shape)}  "
          f"var per-dim min={Z_image.var(0).min():.3e} max={Z_image.var(0).max():.3e}")

    return decoder, Z_image, anchor_flat, res_mean, res_std, keys, cam_dirs, siren_dim


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
# Stage 3: OOD test
# ============================================================================

def optimize_ood_latent(decoder, ood_targets_norm, cam_dirs, args, device):
    """Hold decoder fixed, optimize a fresh 128-dim latent to fit obj_100's
    24 standardized image-SIREN weight residuals."""
    print("\n[ood] optimizing fresh latent for obj_100...")

    z = nn.Parameter(torch.randn(1, args.latent_dim, device=device) * 0.01)
    opt = torch.optim.AdamW([z], lr=args.ood_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ood_steps, eta_min=1e-6)
    decoder.eval()

    for step in range(1, args.ood_steps + 1):
        z_b = z.expand(24, -1)             # (24, latent_dim)
        cam_b = cam_dirs                    # (24, 3)
        pred = decoder(z_b, cam_b)         # (24, siren_dim)
        loss = F.mse_loss(pred, ood_targets_norm)
        opt.zero_grad(set_to_none=True); loss.backward()
        opt.step(); sched.step()
        if step == 1 or step % max(1, args.ood_steps // 50) == 0:
            print(f"  ood step {step:5d}  mse {loss.item():.4e}  z_norm {z.norm().item():.3f}")
    return z.detach()


def ood_test(decoder, mapper, anchor_flat, res_mean, res_std,
             siren_keys, cam_dirs, args, device):
    print("\n[ood] testing on obj_100 via shared decoder pipeline")

    ood_dir = PROJECT / "data" / "ood_test" / "obj_100" / "image_sirens"
    if not ood_dir.exists():
        print(f"[ood] {ood_dir} missing, skip")
        return

    # Load obj_100's 24 image-SIRENs as standardized residual targets
    flats = []
    for j in range(CFG.data.num_views):
        sd = torch.load(ood_dir / f"view_{j:02d}.pt",
                        map_location="cpu", weights_only=True)
        flat, _ = flatten_sd(sd, siren_keys)
        flats.append(flat)
    sirens = torch.stack(flats).to(device)  # (24, siren_dim)

    res = sirens - anchor_flat[None, :].to(device)
    norm = (res - res_mean.to(device)) / res_std.to(device)  # (24, siren_dim)
    print(f"[ood] obj_100 standardized targets: {tuple(norm.shape)}")

    # Optimize fresh latent
    z_image = optimize_ood_latent(decoder, norm, cam_dirs.to(device), args, device)
    print(f"[ood] optimized image latent norm: {z_image.norm().item():.3f}")

    # Push through mapper -> shape latent
    mapper.eval()
    with torch.no_grad():
        z_shape = mapper(z_image)
    print(f"[ood] shape latent norm: {z_shape.norm().item():.3f}")

    # Push through shape decoder -> SIREN -> mesh
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

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MESH_DIR / "ood_obj_100_via_auto_decoder.obj"

    res_mc, bound = 256, 1.0
    lin = torch.linspace(-bound, bound, res_mc, device=device)
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
    vol = out.reshape(res_mc, res_mc, res_mc).cpu().numpy()
    from skimage.measure import marching_cubes
    if not (vol.min() <= 0.0 <= vol.max()):
        print(f"[ood] PREDICTED: no zero crossing")
        return
    spacing = (2*bound/(res_mc-1),)*3
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
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch_objs", type=int, default=8)
    p.add_argument("--lr_decoder", type=float, default=1e-4)
    p.add_argument("--lr_latent", type=float, default=1e-3)
    p.add_argument("--mapper_lr", type=float, default=1e-3)
    p.add_argument("--mapper_steps", type=int, default=8000)
    p.add_argument("--ood_lr", type=float, default=1e-2)
    p.add_argument("--ood_steps", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    decoder, Z_image, anchor_flat, res_mean, res_std, keys, cam_dirs, siren_dim = \
        train_auto_decoder(args, device)
    mapper = train_latent_mapper(args, Z_image, device)
    ood_test(decoder, mapper, anchor_flat, res_mean, res_std, keys, cam_dirs, args, device)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
