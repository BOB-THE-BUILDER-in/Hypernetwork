"""
Hypernet weight autoencoder.

Stage 1 standalone: train AE on 100 hypernets, verify reconstruction.

Architecture:
    Encoder: chunked-transformer (worked for shape AE's mapper input)
        17.9M -> chunk into ~2200 tokens -> transformer -> mean pool -> latent_1024
    Decoder: shared per-chunk MLP with chunk-id embedding (avoids OOM)
        latent_1024 + chunk_id (256 emb) -> shared MLP -> chunk values
        applied 2200 times to reconstruct full 17.9M hypernet

Loss: weight MSE on standardized residuals, same as shape AE.
LR: 1e-4 (same as shape AE that worked).

NOT BUILDING THE TRANSLATOR YET. This script trains AE_image, encodes 100
hypernets to z_image latents, and does sanity reconstruction.

Sanity test we DON'T do here: meshing through the reconstructed hypernet would
require loading and pushing through the trained shape mapper, which we don't
do. Instead we verify reconstruction by per-shape MSE and a "round-trip
identity test" — encode then decode and check the output matches the input.
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

OUT_DIR = PROJECT / "data" / "ae_image_n100"


# ============================================================================
# Encoder: chunked transformer
# ============================================================================

class ChunkedProjector(nn.Module):
    def __init__(self, total_dim, chunk_size, d_model):
        super().__init__()
        self.total_dim = total_dim
        self.chunk_size = chunk_size
        self.num_chunks = math.ceil(total_dim / chunk_size)
        self.padded_dim = self.num_chunks * chunk_size
        self.pad = self.padded_dim - total_dim
        self.proj_in = nn.Linear(chunk_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_chunks, d_model) * 0.02)

    def tokenize(self, x):
        B = x.shape[0]
        if self.pad:
            x = F.pad(x, (0, self.pad))
        chunks = x.view(B, self.num_chunks, self.chunk_size)
        return self.proj_in(chunks) + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class HypernetEncoder(nn.Module):
    """17.9M-dim hypernet -> latent_dim."""
    def __init__(self, hyp_dim, latent_dim, chunk_size=8192, d_model=384, n_layers=4, n_heads=6):
        super().__init__()
        self.chunker = ChunkedProjector(hyp_dim, chunk_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, w):
        x = self.chunker.tokenize(w)
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        return self.to_latent(x.mean(dim=1))


# ============================================================================
# Decoder: shared per-chunk MLP with chunk-id embedding
# ============================================================================

class HypernetDecoder(nn.Module):
    """latent -> 17.9M-dim hypernet via shared per-chunk MLP.
    
    For each output chunk i:
        chunk_i = MLP(concat(latent, chunk_id_emb[i]))
    
    This makes the param count constant in num_chunks (only the embedding table
    grows with num_chunks, which is fine).
    """
    def __init__(self, hyp_dim, latent_dim, chunk_size=8192, hidden=2048, chunk_emb_dim=256):
        super().__init__()
        self.hyp_dim = hyp_dim
        self.chunk_size = chunk_size
        self.num_chunks = math.ceil(hyp_dim / chunk_size)
        self.padded_dim = self.num_chunks * chunk_size
        self.pad = self.padded_dim - hyp_dim

        # Per-chunk learned positional embedding (this is the "address" of each chunk)
        self.chunk_emb = nn.Parameter(torch.randn(self.num_chunks, chunk_emb_dim) * 0.02)

        # Shared MLP that decodes (latent, chunk_emb) -> chunk_size values
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + chunk_emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, chunk_size),
        )

    def forward(self, z):
        # z: (B, latent_dim)
        B = z.shape[0]
        K = self.num_chunks
        # Broadcast: each batch element queries each chunk
        z_exp = z.unsqueeze(1).expand(B, K, -1)                       # (B, K, latent_dim)
        emb_exp = self.chunk_emb.unsqueeze(0).expand(B, -1, -1)       # (B, K, chunk_emb_dim)
        inp = torch.cat([z_exp, emb_exp], dim=-1)                     # (B, K, latent + emb)
        chunks = self.mlp(inp)                                        # (B, K, chunk_size)
        x = chunks.reshape(B, self.padded_dim)
        if self.pad:
            x = x[:, : self.hyp_dim]
        return x


class HypernetAE(nn.Module):
    def __init__(self, hyp_dim, latent_dim=1024, chunk_size=8192,
                 enc_d_model=384, enc_n_layers=4, enc_n_heads=6,
                 dec_hidden=2048, dec_chunk_emb=256):
        super().__init__()
        self.encoder = HypernetEncoder(hyp_dim, latent_dim,
                                       chunk_size=chunk_size,
                                       d_model=enc_d_model,
                                       n_layers=enc_n_layers,
                                       n_heads=enc_n_heads)
        self.decoder = HypernetDecoder(hyp_dim, latent_dim,
                                       chunk_size=chunk_size,
                                       hidden=dec_hidden,
                                       chunk_emb_dim=dec_chunk_emb)

    def encode(self, w):
        return self.encoder(w)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, w):
        return self.decode(self.encode(w))


# ============================================================================
# Data utilities (same as shape AE)
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


# ============================================================================
# Training
# ============================================================================

def train(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load 100 hypernets
    hyp_dir = Path(args.hyp_dir)
    paths = sorted(hyp_dir.glob("obj_*.pt"))
    assert len(paths) == 100, f"expected 100 hypernets, got {len(paths)}"

    flats, keys, shapes = [], None, None
    print(f"[ae-img] loading 100 hypernets...")
    for i, p in enumerate(paths):
        sd = torch.load(p, map_location="cpu", weights_only=True)
        f, k, s = flatten_sd(sd, keys)
        if keys is None:
            keys, shapes = k, s
        flats.append(f)
        if (i+1) % 20 == 0:
            print(f"  loaded {i+1}/100")

    H = torch.stack(flats)  # CPU, (100, 17.9M)
    D = H.shape[1]
    print(f"[ae-img] loaded {H.shape[0]} hypernets, dim={D}, total {H.element_size() * H.numel() / 1e9:.2f} GB on CPU")

    # Anchor for residual
    anchor = torch.load(args.anchor_hyp, map_location="cpu", weights_only=True)
    anchor_flat, _, _ = flatten_sd(anchor, keys)

    H_res = H - anchor_flat[None]
    res_mean = H_res.mean(0, keepdim=True)
    res_std = H_res.std(0, keepdim=True).clamp_min(1e-6)
    H_norm = (H_res - res_mean) / res_std
    print(f"[ae-img] standardized residuals  var={H_norm.var().item():.4f}")

    # Move to GPU just-in-time per batch (not all at once, doesn't fit on 16GB)
    # Plan: full-batch forward pass at 100 shapes, but each forward pass needs
    # only ~7 GB activation, fits if we don't also have optimizer state on GPU.
    # Actually, standard approach: keep H_norm on CPU, mini-batch to GPU.

    # Build model
    ae = HypernetAE(D, latent_dim=args.latent_dim,
                    chunk_size=args.chunk_size,
                    enc_d_model=args.enc_d_model,
                    enc_n_layers=args.enc_n_layers,
                    dec_hidden=args.dec_hidden).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"[ae-img] AE: {n_params:,} params  latent={args.latent_dim}  chunks={ae.encoder.chunker.num_chunks}")

    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    # Mini-batch training (full batch is too big on GPU)
    bs = args.batch_size
    N = H_norm.shape[0]
    print(f"[ae-img] training: bs={bs}, {args.steps} steps, weight MSE")
    import random as _random
    rng = _random.Random(0)

    accum = getattr(args, 'grad_accum', 4)
    print(f"[ae-img] grad_accum={accum}  effective_bs={bs * accum}")
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
        loss = type('L', (), {'item': lambda self: total_loss / accum})()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")

    # Save AE
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

    # Encode all 100 hypernets to latents
    ae.eval()
    Z = []
    print("\n[ae-img] encoding all 100 hypernets")
    with torch.no_grad():
        for i in range(0, N, bs):
            h_b = H_norm[i:i+bs].to(device, non_blocking=True)
            z_b = ae.encode(h_b).cpu()
            Z.append(z_b)
    Z = torch.cat(Z, dim=0)
    torch.save(Z, OUT_DIR / "z_image_latents.pt")
    print(f"[ae-img] z_image latents shape={tuple(Z.shape)}  -> {OUT_DIR/'z_image_latents.pt'}")
    print(f"[ae-img] latent var per dim: min={Z.var(0).min().item():.3e}  max={Z.var(0).max().item():.3e}  "
          f"dead-dims (var<1e-6): {(Z.var(0) < 1e-6).sum().item()}")

    # Round-trip reconstruction MSE per shape
    print("\n[ae-img] per-shape reconstruction MSE (on standardized residuals)")
    per_mse = []
    with torch.no_grad():
        for i in range(N):
            h_i = H_norm[i:i+1].to(device, non_blocking=True)
            r_i = ae(h_i)
            per_mse.append(((r_i - h_i)**2).mean().item())
    per_mse = np.array(per_mse)
    print(f"  mean={per_mse.mean():.4e}  median={np.median(per_mse):.4e}  "
          f"max={per_mse.max():.4e}  min={per_mse.min():.4e}")
    
    print(f"\n[done] {OUT_DIR}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_dir", default="/workspace/hypernet/data/hypernets")
    p.add_argument("--anchor_hyp", default="/workspace/hypernet/data/checkpoints/anchor_hypernet.pt")
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--chunk_size", type=int, default=8192)
    p.add_argument("--enc_d_model", type=int, default=384)
    p.add_argument("--enc_n_layers", type=int, default=4)
    p.add_argument("--dec_hidden", type=int, default=2048)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--steps",     type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation steps")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    train(args, device)


if __name__ == "__main__":
    main()
