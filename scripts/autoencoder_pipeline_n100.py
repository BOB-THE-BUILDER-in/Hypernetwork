"""
Weight-space autoencoder pipeline for N=100 shapes.

Stages:
  1. Train autoencoder on 100 shape-SIREN weight residuals
     - Weight MSE only (no SDF loss; SDF backprop through sin(w0*Wx) is unstable)
     - 128-dim bottleneck forces real compression at N=100
  2. Encode each of 100 shapes to a latent, save z_table.pt
  3. Mesh reconstruction for sanity check (decode z, plug into SIREN, mesh)
  4. Interpolation test: walk z_03 -> z_06 at t in {0, 0.25, 0.5, 0.75, 1.0}
  5. Train tiny mapper: hypernet -> latent (much smaller than original mapper)
  6. OOD test: feed obj_100 hypernet through latent-mapper -> latent -> decoder -> mesh

This is the test of "does latent compression rescue smoothness and OOD generalization."

If reconstructions look like originals AND interpolations stay on-manifold AND
OOD prediction looks like a real shape, the latent-space approach wins.
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

OUT_DIR = PROJECT / "data" / "ae_n100"
MESH_DIR = OUT_DIR / "meshes"


# ============================================================================
# Architecture
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


class TransformerBlock(nn.Module):
    """Plain transformer block (used by encoder)."""
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


class FiLMBlock(nn.Module):
    """Transformer block with FiLM conditioning on latent z.
    z is projected to (shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff)."""
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
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        # NOTE: don't zero-init -- with random queries, zero-init kills gradients

    def forward(self, x, c):
        # c: (B, d_model) ; x: (B, N, d_model)
        s_a, sc_a, g_a, s_f, sc_f, g_f = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc_a.unsqueeze(1)) + s_a.unsqueeze(1)
        x = x + g_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + sc_f.unsqueeze(1)) + s_f.unsqueeze(1)
        x = x + g_f.unsqueeze(1) * self.ffn(h)
        return x


class Encoder(nn.Module):
    def __init__(self, shape_dim, latent_dim, chunk_size=1024, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.chunker = ChunkedProjector(shape_dim, chunk_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, w):
        x = self.chunker.tokenize(w)
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        return self.to_latent(x.mean(dim=1))


class Decoder(nn.Module):
    """Latent -> weights. FiLM/AdaLN conditioning on z."""
    def __init__(self, shape_dim, latent_dim, chunk_size=1024, d_model=256, n_layers=6, n_heads=4):
        super().__init__()
        self.chunker = ChunkedProjector(shape_dim, chunk_size, d_model)
        self.queries = nn.Parameter(torch.randn(1, self.chunker.num_chunks, d_model) * 0.5)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.blocks = nn.ModuleList([FiLMBlock(d_model, n_heads) for _ in range(n_layers)])
        # Final FiLM-modulated norm
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        # don't zero-init final_adaLN either

    def forward(self, z):
        B = z.shape[0]
        c = self.latent_proj(z)                          # (B, d_model)
        x = self.queries.expand(B, -1, -1)               # (B, num_chunks, d_model)
        for b in self.blocks:
            x = b(x, c)
        # Final FiLM-modulated norm
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.chunker.detokenize(x)


# Tiny mapper: hypernet residual -> latent (much smaller than original 70M-param mapper)
class TinyMapper(nn.Module):
    """Predicts 128-dim latent from 17.9M-dim hypernet residual.

    Architecture: chunked input -> small transformer -> mean-pool -> latent.
    Way smaller than the 70M-param weight-space mapper.
    """
    def __init__(self, cond_dim, latent_dim, chunk_size=8192, d_model=384, n_layers=4, n_heads=6):
        super().__init__()
        self.chunker = ChunkedProjector(cond_dim, chunk_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, hyp_res_norm):
        x = self.chunker.tokenize(hyp_res_norm)
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        return self.to_latent(x.mean(dim=1))


# ============================================================================
# Data utilities
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
        s = siren(pts[i:i+chunk])
        if s.dim() > 1:
            s = s.squeeze(-1)
        out[i:i+chunk] = s
    vol = out.reshape(res, res, res).cpu().numpy()
    from skimage.measure import marching_cubes
    if not (vol.min() <= 0.0 <= vol.max()):
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


# ============================================================================
# Stage 1: Train autoencoder
# ============================================================================

def train_autoencoder(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load 100 shape-SIRENs
    shape_dir = Path(args.shape_dir)
    paths = sorted(shape_dir.glob("obj_*.pt"))
    assert len(paths) == 100, f"expected 100 shape-SIRENs, got {len(paths)}"
    flats, keys, shapes = [], None, None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        f, k, s = flatten_sd(sd, keys)
        if keys is None:
            keys, shapes = k, s
        flats.append(f)
    W = torch.stack(flats).to(device)                          # (100, D)
    D = W.shape[1]
    print(f"[ae] loaded 100 shape-SIRENs, dim={D}")

    # Anchor for residual parameterization
    anchor = torch.load(args.anchor, map_location="cpu", weights_only=True)
    anchor_flat, _, _ = flatten_sd(anchor, keys)
    anchor_flat = anchor_flat.to(device)
    W_res = W - anchor_flat[None]

    # Standardize residuals (helps training)
    res_mean = W_res.mean(dim=0, keepdim=True)
    res_std = W_res.std(dim=0, keepdim=True).clamp_min(1e-6)
    W_norm = (W_res - res_mean) / res_std

    # Build models
    enc = Encoder(D, args.latent_dim).to(device)
    dec = Decoder(D, args.latent_dim).to(device)
    n_params = sum(p.numel() for p in list(enc.parameters()) + list(dec.parameters()))
    print(f"[ae] enc+dec: {n_params:,} params  latent={args.latent_dim}")

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    # Mini-batch loop (full batch of 100 fits easily; using full)
    print(f"[ae] training: full-batch on 100 shapes for {args.steps} steps, weight MSE")
    for step in range(1, args.steps + 1):
        enc.train(); dec.train()
        z = enc(W_norm)
        recon = dec(z)
        loss = F.mse_loss(recon, W_norm)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step(); sched.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")

    # Save AE
    torch.save({
        "enc": enc.state_dict(),
        "dec": dec.state_dict(),
        "args": vars(args),
        "shp_keys": keys,
        "shp_shapes": shapes,
        "anchor_flat": anchor_flat.cpu(),
        "res_mean": res_mean.cpu(),
        "res_std": res_std.cpu(),
    }, OUT_DIR / "autoencoder.pt")
    print(f"[ae] saved -> {OUT_DIR/'autoencoder.pt'}")

    # Encode all 100 shapes -> latents
    enc.eval()
    with torch.no_grad():
        Z = enc(W_norm)
    torch.save(Z.cpu(), OUT_DIR / "latents.pt")
    print(f"[ae] saved latents shape={tuple(Z.shape)} -> {OUT_DIR/'latents.pt'}")

    return enc, dec, W_norm, Z, keys, shapes, anchor_flat, res_mean, res_std


# ============================================================================
# Stage 2: Reconstruction sanity check + interpolation
# ============================================================================

def reconstruct_and_test(dec, Z, keys, shapes, anchor_flat, res_mean, res_std, device):
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    c = CFG.shape_siren

    # Reconstruct a few key shapes
    for idx in [3, 6, 22, 36, 100 if Z.shape[0] > 100 else 99]:
        if idx >= Z.shape[0]:
            continue
        z = Z[idx:idx+1].to(device)
        with torch.no_grad():
            recon_norm = dec(z).cpu()
        recon_abs = recon_norm * res_std.cpu() + res_mean.cpu() + anchor_flat[None].cpu()
        siren = unflatten_to_siren(recon_abs[0], keys, shapes, c, device)
        out = MESH_DIR / f"recon_obj_{idx:02d}.obj"
        result = mesh_siren(siren, device, out)
        if result:
            print(f"  recon obj_{idx:02d}: {result[0]} v / {result[1]} f -> {out.name}")
        else:
            print(f"  recon obj_{idx:02d}: NO MESH (off-manifold)")

    # Interpolation test: obj_03 -> obj_06
    z_a, z_b = Z[3:4].to(device), Z[6:7].to(device)
    print("\n[ae] interpolation test obj_03 -> obj_06")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_t = (1 - t) * z_a + t * z_b
        with torch.no_grad():
            recon_norm = dec(z_t).cpu()
        recon_abs = recon_norm * res_std.cpu() + res_mean.cpu() + anchor_flat[None].cpu()
        siren = unflatten_to_siren(recon_abs[0], keys, shapes, c, device)
        out = MESH_DIR / f"interp_03_to_06_t{int(t*100):03d}.obj"
        result = mesh_siren(siren, device, out)
        if result:
            print(f"  t={t:.2f}: {result[0]:>7d} v / {result[1]:>7d} f")
        else:
            print(f"  t={t:.2f}: NO MESH")


# ============================================================================
# Stage 3: Train tiny mapper hypernet -> latent
# ============================================================================

def train_tiny_mapper(args, Z, device):
    """Train a small mapper that predicts the 128-dim latent from a hypernet.

    Reuses the existing manifest and ResidualPairedWeightsDataset for hypernet
    loading and standardization.
    """
    from hypernet_to_shape_mapper import ResidualPairedWeightsDataset

    print("\n[mapper] training tiny mapper hypernet -> latent_128")
    ds = ResidualPairedWeightsDataset(
        args.manifest, args.anchor_hyp, args.anchor_shp, device=device,
    )
    H = ds.hyp_norm  # CPU, (100, cond_dim)
    Z_cpu = Z.cpu()  # (100, 128)
    cond_dim = H.shape[1]
    latent_dim = Z.shape[1]
    N = H.shape[0]

    mapper = TinyMapper(cond_dim, latent_dim).to(device)
    n_params = sum(p.numel() for p in mapper.parameters())
    print(f"[mapper] params: {n_params:,}")

    opt = torch.optim.AdamW(mapper.parameters(), lr=args.mapper_lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.mapper_steps, eta_min=1e-6)

    bs = min(16, N)
    print(f"[mapper] mini-batch size={bs}, {args.mapper_steps} steps")
    import random as _random
    rng = _random.Random(0)

    for step in range(1, args.mapper_steps + 1):
        mapper.train()
        idx = rng.sample(range(N), bs)
        h_b = H[idx].to(device)
        z_b = Z_cpu[idx].to(device)
        pred = mapper(h_b)
        loss = F.mse_loss(pred, z_b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
        opt.step(); sched.step()

        if step == 1 or step % max(1, args.mapper_steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")

    # Per-shape eval
    mapper.eval()
    with torch.no_grad():
        per_mse = []
        for i in range(N):
            p = mapper(H[i:i+1].to(device))
            per_mse.append(((p - Z_cpu[i:i+1].to(device)) ** 2).mean().item())
    per_mse = np.array(per_mse)
    print(f"\n[mapper] per-shape latent MSE: mean={per_mse.mean():.4e} max={per_mse.max():.4e} min={per_mse.min():.4e}")

    # Ablation
    perm = torch.randperm(N)
    with torch.no_grad():
        scram = []
        zero = []
        for i in range(N):
            p_s = mapper(H[perm[i]:perm[i]+1].to(device))
            scram.append(((p_s - Z_cpu[i:i+1].to(device)) ** 2).mean().item())
            p_z = mapper(torch.zeros_like(H[i:i+1]).to(device))
            zero.append(((p_z - Z_cpu[i:i+1].to(device)) ** 2).mean().item())
    scram = np.array(scram); zero = np.array(zero)
    print(f"[mapper] scrambled mean: {scram.mean():.4e}  ratio: {scram.mean()/per_mse.mean():.2f}x")
    print(f"[mapper] zero-cond mean: {zero.mean():.4e}  ratio: {zero.mean()/per_mse.mean():.2f}x")

    torch.save({
        "mapper": mapper.state_dict(),
        "cond_dim": cond_dim,
        "latent_dim": latent_dim,
        "args": vars(args),
        "ds_hyp_names": ds.hyp_names,
        "ds_anchor_hyp": ds.anchor_hyp.cpu(),
        "ds_hyp_mean": ds.hyp_mean.cpu(),
        "ds_hyp_std": ds.hyp_std.cpu(),
    }, OUT_DIR / "latent_mapper.pt")
    print(f"[mapper] saved -> {OUT_DIR/'latent_mapper.pt'}")
    return mapper, ds


# ============================================================================
# Stage 4: OOD test on obj_100
# ============================================================================

def ood_test(mapper, dec, ds, anchor_flat, res_mean, res_std, keys, shapes, device):
    print("\n[ood] testing on obj_100 hypernet")
    ood_hyp_path = PROJECT / "data" / "ood_test" / "obj_100_hypernet.pt"
    if not ood_hyp_path.exists():
        print(f"[ood] {ood_hyp_path} missing — run ood_test_full.py first")
        return

    sd = torch.load(ood_hyp_path, map_location=device, weights_only=True)
    flat = torch.cat([sd[k].detach().float().flatten() for k in ds.hyp_names]).to(device)
    res = flat - ds.anchor_hyp.to(device)
    norm = (res - ds.hyp_mean.to(device).squeeze()) / ds.hyp_std.to(device).squeeze()

    mapper.eval(); dec.eval()
    with torch.no_grad():
        z_pred = mapper(norm.unsqueeze(0))
        recon_norm = dec(z_pred).cpu()
    recon_abs = recon_norm * res_std.cpu() + res_mean.cpu() + anchor_flat[None].cpu()

    c = CFG.shape_siren
    siren = unflatten_to_siren(recon_abs[0], keys, shapes, c, device)
    out = MESH_DIR / "ood_obj_100_predicted_via_latent.obj"
    result = mesh_siren(siren, device, out)
    if result:
        print(f"[ood] PREDICTED: {result[0]} v / {result[1]} f -> {out.name}")
    else:
        print(f"[ood] PREDICTED: no zero crossing in SDF")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape_dir", default="/workspace/hypernet/data/shape_sirens")
    p.add_argument("--anchor",    default="/workspace/hypernet/checkpoints/anchor_shape_siren.pt")
    p.add_argument("--manifest",  default="/workspace/hypernet/scripts/manifest_n100.pt")
    p.add_argument("--anchor_hyp", default="/workspace/hypernet/data/checkpoints/anchor_hypernet.pt")
    p.add_argument("--anchor_shp", default="/workspace/hypernet/checkpoints/anchor_shape_siren.pt")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--steps",     type=int, default=4000)
    p.add_argument("--mapper_lr", type=float, default=1e-3)
    p.add_argument("--mapper_steps", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_ae", action="store_true",
                   help="skip AE training, load existing autoencoder.pt")
    p.add_argument("--skip_mapper", action="store_true",
                   help="skip mapper training")
    p.add_argument("--skip_ood", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.skip_ae and (OUT_DIR / "autoencoder.pt").exists():
        print("[main] loading existing AE")
        ae_ckpt = torch.load(OUT_DIR / "autoencoder.pt", map_location=device, weights_only=False)
        Z = torch.load(OUT_DIR / "latents.pt", map_location="cpu", weights_only=True)
        keys = ae_ckpt["shp_keys"]
        shapes = ae_ckpt["shp_shapes"]
        anchor_flat = ae_ckpt["anchor_flat"].to(device)
        res_mean = ae_ckpt["res_mean"].to(device)
        res_std  = ae_ckpt["res_std"].to(device)
        # Rebuild encoder + decoder
        D = anchor_flat.shape[0]
        enc = Encoder(D, ae_ckpt["args"]["latent_dim"]).to(device)
        dec = Decoder(D, ae_ckpt["args"]["latent_dim"]).to(device)
        enc.load_state_dict(ae_ckpt["enc"])
        dec.load_state_dict(ae_ckpt["dec"])
        # Need W_norm for some downstream calls (not strictly required if Z saved)
        W_norm = None
    else:
        enc, dec, W_norm, Z, keys, shapes, anchor_flat, res_mean, res_std = train_autoencoder(args, device)

    print("\n[main] reconstruction + interpolation tests")
    reconstruct_and_test(dec, Z.to(device), keys, shapes, anchor_flat, res_mean, res_std, device)

    if not args.skip_mapper:
        mapper, ds = train_tiny_mapper(args, Z.to(device), device)
    else:
        from hypernet_to_shape_mapper import ResidualPairedWeightsDataset
        m_ckpt = torch.load(OUT_DIR / "latent_mapper.pt", map_location=device, weights_only=False)
        mapper = TinyMapper(m_ckpt["cond_dim"], m_ckpt["latent_dim"]).to(device)
        mapper.load_state_dict(m_ckpt["mapper"])
        ds = ResidualPairedWeightsDataset(
            args.manifest, args.anchor_hyp, args.anchor_shp, device=device,
        )

    if not args.skip_ood:
        ood_test(mapper, dec, ds, anchor_flat, res_mean, res_std, keys, shapes, device)

    print(f"\n[done] outputs in {OUT_DIR}")
    print(f"  meshes: {MESH_DIR}")


if __name__ == "__main__":
    main()
