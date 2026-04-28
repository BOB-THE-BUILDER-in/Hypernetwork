"""
Hypernet AE v2: simple MLP architecture (matching shape AE that worked).

Architecture:
    Encoder: Linear(17.9M -> 4096) -> GELU -> Linear(4096 -> 1024)
    Decoder: Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 17.9M)

The huge layers are 17.9M × 4096 = 73M params each. Total ~155M params, fits.

Same training recipe as shape AE: lr=1e-4, weight MSE, AdamW, grad clip 1.0.
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

OUT_DIR = PROJECT / "data" / "ae_image_n100_v2"


class HypernetAE(nn.Module):
    def __init__(self, hyp_dim, latent_dim=1024, hidden=4096):
        super().__init__()
        self.hyp_dim = hyp_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(hyp_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hyp_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


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


def train(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hyp_dir = Path(args.hyp_dir)
    paths = sorted(hyp_dir.glob("obj_*.pt"))
    assert len(paths) == 100, f"expected 100, got {len(paths)}"

    flats, keys, shapes = [], None, None
    print(f"[ae-img] loading 100 hypernets...")
    for i, p in enumerate(paths):
        sd = torch.load(p, map_location="cpu", weights_only=True)
        f, k, s = flatten_sd(sd, keys)
        if keys is None:
            keys, shapes = k, s
        flats.append(f)
        if (i + 1) % 25 == 0:
            print(f"  loaded {i+1}/100")

    H = torch.stack(flats)
    D = H.shape[1]
    print(f"[ae-img] H: {H.shape}, {H.element_size() * H.numel() / 1e9:.2f} GB on CPU")

    anchor = torch.load(args.anchor_hyp, map_location="cpu", weights_only=True)
    anchor_flat, _, _ = flatten_sd(anchor, keys)

    H_res = H - anchor_flat[None]
    res_mean = H_res.mean(0, keepdim=True)
    res_std = H_res.std(0, keepdim=True).clamp_min(1e-6)
    H_norm = (H_res - res_mean) / res_std
    print(f"[ae-img] standardized: var={H_norm.var().item():.4f}")

    ae = HypernetAE(D, latent_dim=args.latent_dim, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"[ae-img] AE params: {n_params:,}  latent={args.latent_dim}  hidden={args.hidden}")

    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    bs = args.batch_size
    accum = args.grad_accum
    N = H_norm.shape[0]
    print(f"[ae-img] training: bs={bs} grad_accum={accum} effective_bs={bs*accum}  "
          f"{args.steps} steps, lr={args.lr}")

    import random as _random
    rng = _random.Random(0)

    for step in range(1, args.steps + 1):
        ae.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accum):
            idx = rng.sample(range(N), bs)
            h_b = H_norm[idx].to(device, non_blocking=True)
            recon = ae(h_b)
            loss = F.mse_loss(recon, h_b) / accum
            loss.backward()
            total_loss += loss.item() * accum
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {total_loss / accum:.4e}  lr {sched.get_last_lr()[0]:.2e}")

    torch.save({
        "model": ae.state_dict(),
        "args": vars(args),
        "hyp_keys": keys,
        "hyp_shapes": shapes,
        "anchor_flat": anchor_flat,
        "res_mean": res_mean,
        "res_std": res_std,
        "D": D,
    }, OUT_DIR / "autoencoder.pt")
    print(f"[ae-img] saved -> {OUT_DIR/'autoencoder.pt'}")

    ae.eval()
    Z = []
    print("\n[ae-img] encoding 100 hypernets")
    with torch.no_grad():
        for i in range(0, N, bs):
            h_b = H_norm[i:i+bs].to(device, non_blocking=True)
            Z.append(ae.encode(h_b).cpu())
    Z = torch.cat(Z, dim=0)
    torch.save(Z, OUT_DIR / "z_image_latents.pt")
    print(f"[ae-img] z_image latents shape={tuple(Z.shape)}  -> {OUT_DIR/'z_image_latents.pt'}")
    print(f"[ae-img] latent var per dim: min={Z.var(0).min().item():.3e}  "
          f"max={Z.var(0).max().item():.3e}  "
          f"dead-dims (<1e-6): {(Z.var(0) < 1e-6).sum().item()}")

    print("\n[ae-img] per-shape recon MSE (standardized)")
    per_mse = []
    with torch.no_grad():
        for i in range(N):
            h_i = H_norm[i:i+1].to(device, non_blocking=True)
            r_i = ae(h_i)
            per_mse.append(((r_i - h_i)**2).mean().item())
    per_mse = np.array(per_mse)
    print(f"  mean={per_mse.mean():.4e}  median={np.median(per_mse):.4e}  "
          f"max={per_mse.max():.4e}  min={per_mse.min():.4e}")

    print(f"\n[done] -> {OUT_DIR}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_dir", default="/workspace/hypernet/data/hypernets")
    p.add_argument("--anchor_hyp", default="/workspace/hypernet/data/checkpoints/anchor_hypernet.pt")
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    train(args, device)


if __name__ == "__main__":
    main()
