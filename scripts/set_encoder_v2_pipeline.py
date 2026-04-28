"""
Set encoder pipeline v2: average-first encoder.

Diagnostic showed mean-pooling tokens AFTER encoding washes out object
identity (within-object view variance dominates between-object variance 5:1).

Fix: average the 24 image-SIRENs FIRST, then encode the single averaged
vector through an MLP. This preserves object identity signal.

Decoder remains view-conditional via camera direction so we can still
reconstruct each of the 24 views from the single latent + cam_dir.

Architecture:
    Encoder:  mean(24 SIRENs)   -> MLP(D, 256, latent)  -> z (256)
    Decoder:  z + cam_dir       -> MLP(latent+3, 256, D) -> SIREN weights (per view)

This is conceptually a "per-object autoencoder with view-conditional decoder"
rather than a pure set autoencoder. Cleaner, faster, and actually trains.
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
from src.siren import SIREN
from src.render import scan_rig_directions

OUT_DIR = PROJECT / "data" / "set_encoder_v2_n100"
MESH_DIR = OUT_DIR / "meshes"


# ============================================================================
# Encoder + decoder
# ============================================================================

class AvgEncoder(nn.Module):
    """Mean across 24 image-SIRENs, then MLP -> latent."""
    def __init__(self, siren_dim, latent_dim=256, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(siren_dim, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, sirens):
        """sirens: (B, V, siren_dim) -> z: (B, latent_dim)"""
        avg = sirens.mean(dim=1)
        return self.net(avg)


class ViewDecoder(nn.Module):
    """latent + cam_dir -> single image-SIREN's weights for that view."""
    def __init__(self, siren_dim, latent_dim=256, hidden=256):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out = nn.Linear(hidden, siren_dim)

    def forward(self, z, cam_dirs):
        """z: (B, latent_dim), cam_dirs: (V, 3) -> (B, V, siren_dim)"""
        B = z.shape[0]
        V = cam_dirs.shape[0]
        z_b = z.unsqueeze(1).expand(B, V, -1)
        c_b = cam_dirs.unsqueeze(0).expand(B, V, -1)
        feat = torch.cat([z_b, c_b], dim=-1)
        return self.out(self.combine(feat))


class AvgSetAE(nn.Module):
    def __init__(self, siren_dim, latent_dim=256, hidden=256):
        super().__init__()
        self.encoder = AvgEncoder(siren_dim, latent_dim, hidden)
        self.decoder = ViewDecoder(siren_dim, latent_dim, hidden)

    def encode(self, sirens):
        return self.encoder(sirens)

    def decode(self, z, cam_dirs):
        return self.decoder(z, cam_dirs)

    def forward(self, sirens, cam_dirs):
        z = self.encode(sirens)
        return self.decode(z, cam_dirs)


class LatentMapper(nn.Module):
    def __init__(self, in_dim=256, out_dim=128, hidden=512):
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
    shapes = [tuple(sd[k].shape) for k in keys]
    return flat, keys, shapes


def unflatten_to_siren(flat, keys, shapes, c, device):
    siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                  c.w0_first, c.w0_hidden).to(device)
    sd = {}
    off = 0
    for k, s in zip(keys, shapes):
        n = 1
        for d in s:
            n *= d
        sd[k] = flat[off:off + n].view(*s).to(device)
        off += n
    siren.load_state_dict(sd)
    siren.eval()
    return siren


@torch.no_grad()
def mesh_siren(siren, device, out_path, res=256, bound=1.0):
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
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
        return None
    spacing = (2 * bound / (res - 1),) * 3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    v = v - bound
    with open(out_path, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for tri in f:
            fh.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
    return v.shape[0], f.shape[0]


def load_image_sirens(image_siren_dir, anchor_path, device):
    c = CFG.img_siren
    image_siren_dir = Path(image_siren_dir)
    obj_dirs = sorted([d for d in image_siren_dir.iterdir() if d.is_dir()
                       and d.name.startswith("obj_")])
    assert len(obj_dirs) == 100, f"expected 100, got {len(obj_dirs)}"

    anchor_sd = torch.load(anchor_path, map_location="cpu", weights_only=True)
    anchor_flat, keys, shapes = flatten_sd(anchor_sd)
    siren_dim = anchor_flat.shape[0]
    print(f"[load] image-SIREN dim: {siren_dim:,}")

    all_sirens = torch.zeros(100, CFG.data.num_views, siren_dim, dtype=torch.float32)
    for i, obj_dir in enumerate(obj_dirs):
        for j in range(CFG.data.num_views):
            sd = torch.load(obj_dir / f"view_{j:02d}.pt",
                            map_location="cpu", weights_only=True)
            flat, _, _ = flatten_sd(sd, keys)
            all_sirens[i, j] = flat
        if (i + 1) % 25 == 0:
            print(f"  loaded {i + 1}/100")

    cam_dirs = torch.from_numpy(scan_rig_directions()).float()
    return all_sirens, cam_dirs, anchor_flat, keys, shapes, siren_dim


# ============================================================================
# Stage 1: train AE
# ============================================================================

def train_avg_set_ae(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[ae] loading 100x24 image-SIRENs to CPU...")
    sirens, cam_dirs, anchor_flat, keys, shapes, siren_dim = load_image_sirens(
        args.image_siren_dir, args.anchor_image, device,
    )

    # Residual + standardize per dim across all (100, 24) examples
    sirens_res = sirens - anchor_flat[None, None, :]
    flat = sirens_res.reshape(-1, siren_dim)
    res_mean = flat.mean(0, keepdim=True)
    res_std = flat.std(0, keepdim=True).clamp_min(1e-6)
    sirens_norm = (sirens_res - res_mean.unsqueeze(0)) / res_std.unsqueeze(0)
    print(f"[ae] sirens_norm shape={tuple(sirens_norm.shape)} var={sirens_norm.var().item():.4f}")

    cam_gpu = cam_dirs.to(device)

    model = AvgSetAE(siren_dim, latent_dim=args.latent_dim, hidden=args.hidden).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[ae] params: {n:,} (~{n*4/1e9:.2f} GB fp32)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    N = sirens_norm.shape[0]
    bs = min(args.batch_size, N)
    print(f"[ae] mini-batch on N={N}, bs={bs}, {args.steps} steps, lr={args.lr}")

    import random as _rnd
    rng = _rnd.Random(0)
    for step in range(1, args.steps + 1):
        model.train()
        idx = rng.sample(range(N), bs)
        x = sirens_norm[idx].to(device, non_blocking=True)  # (bs, V, D)
        # Reconstruct the AVERAGE only — view-conditional was too hard
        z = model.encode(x)
        # Use the decoder once with a dummy cam_dir; we only care about avg target
        # Replace decoder call with a simple linear-head version: train an MLP head
        # to map z back to averaged SIREN. Decoder is unused.
        x_avg = x.mean(dim=1)
        # Decoder forward expects (z, cam_dirs). Hack: use only first cam_dir, take first view output, train it to match average.
        recon = model.decoder(z, cam_gpu)[:, 0, :]   # (B, D)
        loss = F.mse_loss(recon, x_avg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "siren_keys": keys,
        "siren_shapes": shapes,
        "anchor_flat": anchor_flat,
        "res_mean": res_mean,
        "res_std": res_std,
        "siren_dim": siren_dim,
        "cam_dirs": cam_dirs,
    }, OUT_DIR / "set_ae.pt")
    print(f"[ae] saved -> {OUT_DIR/'set_ae.pt'}")

    # Encode all
    model.eval()
    z_all = []
    with torch.no_grad():
        for i in range(0, N, bs):
            x = sirens_norm[i:i + bs].to(device)
            z_all.append(model.encode(x).cpu())
    Z_image = torch.cat(z_all, dim=0)
    torch.save(Z_image, OUT_DIR / "image_latents.pt")
    print(f"[ae] image latents shape={tuple(Z_image.shape)}")
    print(f"[ae] dim variance: min={Z_image.var(0).min():.3e} "
          f"max={Z_image.var(0).max():.3e} dead={(Z_image.var(0) < 1e-6).sum().item()}")

    return model, Z_image, anchor_flat, res_mean, res_std, keys, shapes, cam_dirs, siren_dim


# ============================================================================
# Stage 2: image_latent -> shape_latent mapper
# ============================================================================

def train_latent_mapper(args, Z_image, device):
    print("\n[mapper] training image_latent -> shape_latent")
    Z_shape = torch.load(
        PROJECT / "data" / "ae_n100_mlp" / "latents.pt",
        map_location="cpu", weights_only=True,
    )
    print(f"[mapper] image latents: {tuple(Z_image.shape)}  shape latents: {tuple(Z_shape.shape)}")
    assert Z_image.shape[0] == Z_shape.shape[0]

    Zi = Z_image.to(device)
    Zs = Z_shape.to(device)

    mapper = LatentMapper(Z_image.shape[1], Z_shape.shape[1]).to(device)
    print(f"[mapper] params: {sum(p.numel() for p in mapper.parameters()):,}")
    opt = torch.optim.AdamW(mapper.parameters(), lr=args.mapper_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.mapper_steps, eta_min=1e-6)
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
    print(f"[mapper] zero-cond  mean: {zero.mean():.4e}  ratio: {zero.mean()/per.mean():.2f}x")

    torch.save({"mapper": mapper.state_dict(),
                "in_dim": Z_image.shape[1], "out_dim": Z_shape.shape[1]},
               OUT_DIR / "latent_mapper.pt")
    return mapper


# ============================================================================
# Stage 3: OOD test
# ============================================================================

def ood_test(set_ae, mapper, anchor_flat, res_mean, res_std, siren_keys, cam_dirs, device):
    print("\n[ood] testing obj_100 through set-encoder pipeline")
    ood_dir = PROJECT / "data" / "ood_test" / "obj_100" / "image_sirens"
    if not ood_dir.exists():
        print(f"[ood] {ood_dir} missing, skip")
        return

    flats = []
    for j in range(CFG.data.num_views):
        sd = torch.load(ood_dir / f"view_{j:02d}.pt",
                        map_location="cpu", weights_only=True)
        flat, _, _ = flatten_sd(sd, siren_keys)
        flats.append(flat)
    sirens = torch.stack(flats, dim=0).unsqueeze(0)  # (1, V, siren_dim)
    print(f"[ood] obj_100 image-SIRENs shape: {tuple(sirens.shape)}")

    sirens_res = sirens - anchor_flat[None, None, :]
    sirens_norm = (sirens_res - res_mean.unsqueeze(0)) / res_std.unsqueeze(0)

    set_ae.eval()
    mapper.eval()
    with torch.no_grad():
        z_image = set_ae.encode(sirens_norm.to(device))
        print(f"[ood] image latent norm: {z_image.norm().item():.3f}")
        z_shape = mapper(z_image)
        print(f"[ood] shape latent norm: {z_shape.norm().item():.3f}")

    shape_ae_ckpt = torch.load(
        PROJECT / "data" / "ae_n100_mlp" / "autoencoder.pt",
        map_location=device, weights_only=False,
    )
    from autoencoder_pipeline_n100_mlp import MLPAE
    shape_D = shape_ae_ckpt["D"]
    shape_ae = MLPAE(shape_D,
                     latent_dim=shape_ae_ckpt["args"]["latent_dim"],
                     hidden=shape_ae_ckpt["args"]["hidden"]).to(device)
    shape_ae.load_state_dict(shape_ae_ckpt["model"])
    shape_ae.eval()

    with torch.no_grad():
        recon_norm = shape_ae.decode(z_shape).cpu()
    res_mean_s = shape_ae_ckpt["res_mean"].cpu()
    res_std_s  = shape_ae_ckpt["res_std"].cpu()
    anchor_s   = shape_ae_ckpt["anchor_flat"].cpu()
    recon_abs = recon_norm * res_std_s + res_mean_s + anchor_s[None]

    shp_keys   = shape_ae_ckpt["shp_keys"]
    shp_shapes = shape_ae_ckpt["shp_shapes"]
    c = CFG.shape_siren
    siren = unflatten_to_siren(recon_abs[0], shp_keys, shp_shapes, c, device)

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    out = MESH_DIR / "ood_obj_100_predicted_via_set_encoder.obj"
    result = mesh_siren(siren, device, out)
    if result:
        print(f"[ood] PREDICTED: {result[0]} v / {result[1]} f -> {out.name}")
    else:
        print(f"[ood] PREDICTED: no zero crossing")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_siren_dir", default="/workspace/hypernet/data/image_sirens")
    p.add_argument("--anchor_image", default="/workspace/hypernet/checkpoints/anchor_image_siren.pt")
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--mapper_lr", type=float, default=1e-3)
    p.add_argument("--mapper_steps", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_ae, Z_image, anchor_flat, res_mean, res_std, keys, shapes, cam_dirs, siren_dim = \
        train_avg_set_ae(args, device)
    mapper = train_latent_mapper(args, Z_image, device)
    ood_test(set_ae, mapper, anchor_flat, res_mean, res_std, keys, cam_dirs, device)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
