"""
Weight-space autoencoder on shape-SIREN weights.

Pipeline:
    shape_weights (264K)  ->  Encoder  ->  z (128)  ->  Decoder  ->  recon_weights (264K)

Loss: SDF MSE. For each shape, load both the original and reconstructed SIREN,
query them on the same SDF sample points, minimize MSE between the outputs.
This directly optimizes mesh fidelity rather than weight precision.

Why this should fix holes:
    Weight MSE treats every parameter equally, but SIREN outputs are insensitive
    to most parameters and hypersensitive to a few. SDF-aligned loss only cares
    about output, so the autoencoder learns to preserve the weight combinations
    that matter for the shape and discard the rest.

After training, the mapper just has to predict the 128-dim z from the hypernet,
not the full 264K weights. Much smaller prediction problem.
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

sys.path.insert(0, "/workspace/hypernet")
from configs.config import CFG
from src.siren import SIREN


# ============================================================================
# Chunked projector (same as mapper)
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
        self.proj_out = nn.Linear(d_model, chunk_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_chunks, d_model) * 0.02)

    def tokenize(self, x):
        B = x.shape[0]
        if self.pad:
            x = F.pad(x, (0, self.pad))
        chunks = x.view(B, self.num_chunks, self.chunk_size)
        return self.proj_in(chunks) + self.pos_embed

    def detokenize(self, tokens):
        B = tokens.shape[0]
        chunks = self.proj_out(tokens)
        x = chunks.view(B, self.padded_dim)
        if self.pad:
            x = x[:, : self.total_dim]
        return x


# ============================================================================
# Encoder: weights -> z
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, shape_dim, chunk_size=1024, d_model=256, n_layers=4, n_heads=4,
                 latent_dim=128, ff_mult=4):
        super().__init__()
        self.chunker = ChunkedProjector(shape_dim, chunk_size, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "norm2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * ff_mult),
                    nn.GELU(),
                    nn.Linear(d_model * ff_mult, d_model),
                ),
            }))
        self.final_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, w):
        x = self.chunker.tokenize(w)              # (B, N, d)
        for blk in self.blocks:
            h = blk["norm1"](x)
            x = x + blk["attn"](h, h, h, need_weights=False)[0]
            x = x + blk["ffn"](blk["norm2"](x))
        x = self.final_norm(x)
        pooled = x.mean(dim=1)                     # (B, d)
        return self.to_latent(pooled)              # (B, latent_dim)


# ============================================================================
# Decoder: z -> weights (chunked, similar to mapper)
# ============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        s_a, sc_a, g_a, s_f, sc_f, g_f = self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s_a, sc_a)
        x = x + g_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), s_f, sc_f)
        x = x + g_f.unsqueeze(1) * self.ffn(h)
        return x


class Decoder(nn.Module):
    def __init__(self, shape_dim, chunk_size=1024, d_model=256, n_layers=6, n_heads=4,
                 latent_dim=128, ff_mult=4):
        super().__init__()
        self.chunker = ChunkedProjector(shape_dim, chunk_size, d_model)
        # Learnable query tokens, one per output chunk
        self.queries = nn.Parameter(torch.randn(1, self.chunker.num_chunks, d_model) * 0.02)
        # Project latent into the modulation vector that drives AdaLN
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        # Zero-init output projection so initial recon = 0 (will be residualized onto anchor)
        nn.init.zeros_(self.chunker.proj_out.weight)
        nn.init.zeros_(self.chunker.proj_out.bias)

    def forward(self, z):
        B = z.shape[0]
        c = self.latent_proj(z)                                    # (B, d_model)
        x = self.queries.expand(B, -1, -1) + self.chunker.pos_embed
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.chunker.detokenize(x)                          # (B, shape_dim)


# ============================================================================
# Dataset: flatten all 10 shape-SIRENs, also load SDF samples for SDF loss
# ============================================================================

def _unwrap(o):
    if isinstance(o, dict) and "state_dict" in o and isinstance(o["state_dict"], dict):
        return o["state_dict"]
    return o


def flatten_sd(sd, keys=None):
    sd = _unwrap(sd)
    if keys is None:
        keys = list(sd.keys())
    return (
        torch.cat([sd[k].detach().float().flatten() for k in keys]),
        keys,
        [tuple(sd[k].shape) for k in keys],
    )


def unflatten_batch(flat, names, shapes):
    """flat: (B, D) -> list[dict]."""
    out = []
    for b in range(flat.shape[0]):
        sd = {}
        off = 0
        for n, s in zip(names, shapes):
            size = 1
            for d in s:
                size *= d
            sd[n] = flat[b, off:off + size].view(*s)
            off += size
        out.append(sd)
    return out


# ============================================================================
# Functional SIREN forward with external weights (so we can run the decoder
# output through SIREN without materializing 10 separate nn.Modules each step)
# ============================================================================

def siren_forward_functional(pts, sd, keys, shapes, hidden_dim, num_layers, w0_first, w0_hidden):
    """Run a SIREN forward pass using weights from a dict. Differentiable
    w.r.t. sd, which is what lets us backprop SDF loss through the decoder.

    Assumes SIREN layout from src/siren.py:
        net.0.linear.weight, net.0.linear.bias, ..., net.{L-1}.linear.weight, net.{L-1}.linear.bias,
        final.weight, final.bias
    """
    x = pts
    for i in range(num_layers):
        w = sd[f"net.{i}.linear.weight"]
        b = sd[f"net.{i}.linear.bias"]
        w0 = w0_first if i == 0 else w0_hidden
        x = torch.sin(w0 * F.linear(x, w, b))
    x = F.linear(x, sd["final.weight"], sd["final.bias"])
    return x


# ============================================================================
# Training
# ============================================================================

def train(args):
    device = torch.device(args.device)
    c = CFG.shape_siren
    d = CFG.data

    # 1) Load all 10 shape-SIRENs, flatten into a (N, D) tensor
    shape_dir = Path(args.shape_dir)
    shp_paths = sorted(shape_dir.glob("obj_*.pt"))
    assert len(shp_paths) == 10, f"expected 10 shape-SIRENs, got {len(shp_paths)}"

    flats, keys, shapes = [], None, None
    for p in shp_paths:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        f, k, s = flatten_sd(sd, keys)
        if keys is None:
            keys, shapes = k, s
        flats.append(f)
    W = torch.stack(flats).to(device)                                # (10, D)
    shape_dim = W.shape[1]
    print(f"[data] loaded {W.shape[0]} shape-SIRENs, dim={shape_dim:,}")

    # Anchor for residual parameterization
    anchor_ckpt = torch.load(args.anchor, map_location="cpu", weights_only=True)
    anchor_flat, _, _ = flatten_sd(anchor_ckpt, keys)
    anchor_flat = anchor_flat.to(device)

    # 2) Load SDF samples per object
    sdf_pts_list, sdf_vals_list = [], []
    for i, p in enumerate(shp_paths):
        obj_idx = int(p.stem.split("_")[-1])
        sdf_path = d.sdf_dir / f"obj_{obj_idx:02d}.npz"
        assert sdf_path.exists(), f"missing {sdf_path}"
        data = np.load(sdf_path)
        sdf_pts_list.append(torch.from_numpy(data["points"]).to(device))
        sdf_vals_list.append(torch.from_numpy(data["sdf"]).to(device))
    print(f"[data] SDF samples loaded for all 10 shapes")

    # 3) Build encoder + decoder
    enc = Encoder(shape_dim=shape_dim, latent_dim=args.latent_dim,
                  d_model=args.d_model, n_layers=4, n_heads=4).to(device)
    dec = Decoder(shape_dim=shape_dim, latent_dim=args.latent_dim,
                  d_model=args.d_model, n_layers=6, n_heads=4).to(device)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"[model] encoder+decoder: {n_params:,} params  latent_dim={args.latent_dim}")

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    # SIREN meta for functional forward
    hidden_dim = c.hidden_dim
    num_layers = c.num_layers
    w0f, w0h = c.w0_first, c.w0_hidden

    # Residual training: encode/decode the anchor-residual, add anchor back
    W_res = W - anchor_flat[None]                                    # (10, D)

    # SDF sampling: at each step, subsample K points per shape to keep loss fast
    K = args.sdf_points
    N_shapes = W.shape[0]

    # Two-stage training:
    #   Stage 1 (steps 1 .. sdf_start): pure weight MSE. Stable, fast, gets the
    #     autoencoder to a good initialization in weight space.
    #   Stage 2 (steps sdf_start .. end): add SDF loss on top of weight MSE.
    #     SDF loss is fragile through SIREN (sin(w0*Wx) has sharp gradients),
    #     so we only enable it once reconstruction is already near the target.
    sdf_start = args.sdf_start if args.sdf_start > 0 else args.steps // 2
    print(f"[train] full-batch over 10 shapes, {args.steps} steps")
    print(f"[train] stage 1 (weight MSE only): steps 1..{sdf_start}")
    print(f"[train] stage 2 (weight + SDF)   : steps {sdf_start+1}..{args.steps}")

    for step in range(1, args.steps + 1):
        enc.train(); dec.train()

        z = enc(W_res)                                               # (10, latent)
        recon_res = dec(z)                                           # (10, D)

        # Weight MSE is always active; normalize by target variance to make the
        # scale comparable across training shapes
        weight_loss = F.mse_loss(recon_res, W_res)

        if step <= sdf_start:
            # Stage 1: weights only
            loss = weight_loss
            sdf_loss_val = 0.0
        else:
            # Stage 2: add SDF loss, scaled so it doesn't dominate initially
            recon = recon_res + anchor_flat[None]
            recon_sds = unflatten_batch(recon, keys, shapes)

            sdf_loss = 0.0
            for i in range(N_shapes):
                idx = torch.randint(0, sdf_pts_list[i].shape[0], (K,), device=device)
                pts_i = sdf_pts_list[i][idx]
                tgt_i = sdf_vals_list[i][idx].clamp(-c.truncation, c.truncation)
                pred_i = siren_forward_functional(
                    pts_i, recon_sds[i], keys, shapes,
                    hidden_dim, num_layers, w0f, w0h
                ).squeeze(-1).clamp(-c.truncation, c.truncation)
                sdf_loss = sdf_loss + F.mse_loss(pred_i, tgt_i)
            sdf_loss = sdf_loss / N_shapes
            sdf_loss_val = sdf_loss.item()

            loss = weight_loss + args.sdf_weight * sdf_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(dec.parameters()), max_norm=1.0
        )
        opt.step(); sched.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            stage = 1 if step <= sdf_start else 2
            print(f"  step {step:5d}  stage{stage}  weight_mse {weight_loss.item():.5e}  "
                  f"sdf_mse {sdf_loss_val:.5e}  lr {sched.get_last_lr()[0]:.2e}")

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "enc": enc.state_dict(),
        "dec": dec.state_dict(),
        "args": vars(args),
        "shp_keys": keys,
        "shp_shapes": shapes,
        "anchor_flat": anchor_flat.cpu(),
    }, out_dir / "autoencoder.pt")
    print(f"[save] -> {out_dir/'autoencoder.pt'}")

    # Evaluate: reconstruct every shape, save mesh
    print("\n[eval] reconstructing meshes")
    enc.eval(); dec.eval()
    with torch.no_grad():
        z = enc(W_res)
        recon_abs = dec(z) + anchor_flat[None]
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_shapes):
        sd = {}
        off = 0
        for n, s in zip(keys, shapes):
            size = 1
            for dim in s:
                size *= dim
            sd[n] = recon_abs[i, off:off + size].view(*s).cpu()
            off += size
        siren = SIREN(c.in_dim, c.out_dim, hidden_dim, num_layers, w0f, w0h).to(device)
        siren.load_state_dict(sd)
        siren.eval()
        dump_single_mesh(siren, device, mesh_dir / f"obj_{int(shp_paths[i].stem.split('_')[-1]):02d}_recon.obj")

    # Also save the latents themselves (these are what the mapper will learn to predict)
    torch.save(z.cpu(), out_dir / "latents.pt")
    print(f"[save] latents -> {out_dir/'latents.pt'}  shape={tuple(z.shape)}")


@torch.no_grad()
def dump_single_mesh(siren, device, out_path, res=256, bound=1.0):
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    chunk = 65536
    for i in range(0, pts.shape[0], chunk):
        s = siren(pts[i:i + chunk])
        if s.dim() > 1:
            s = s.squeeze(-1)
        out[i:i + chunk] = s
    vol = out.reshape(res, res, res).cpu().numpy()
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print(f"  scikit-image missing, skip {out_path.name}")
        return
    if not (vol.min() <= 0.0 <= vol.max()):
        print(f"  {out_path.name}: no zero crossing")
        return
    spacing = (2 * bound / (res - 1),) * 3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    v = v - bound
    with open(out_path, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for t in f:
            fh.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    print(f"  {out_path.name}: {v.shape[0]} v / {f.shape[0]} f")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape_dir", default="/workspace/hypernet/data/shape_sirens")
    p.add_argument("--anchor",    default="/workspace/hypernet/checkpoints/anchor_shape_siren.pt")
    p.add_argument("--out",       default="/workspace/hypernet/data/ae_w30_z128")

    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--d_model",    type=int, default=256)

    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--wd",        type=float, default=0.0)
    p.add_argument("--steps",     type=int, default=6000)
    p.add_argument("--sdf_points", type=int, default=4096,
                   help="SDF samples per shape per step (only stage 2)")
    p.add_argument("--sdf_start", type=int, default=-1,
                   help="step at which SDF loss kicks in (default: halfway)")
    p.add_argument("--sdf_weight", type=float, default=0.1,
                   help="weight of SDF loss when it's active (weight MSE is 1.0)")

    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())