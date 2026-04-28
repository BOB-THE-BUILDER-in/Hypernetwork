"""
Weight-space autoencoder using MLP architecture (chunked-transformer was broken).

Stages identical to autoencoder_pipeline_n100.py but with a brain-dead MLP
encoder/decoder. Diagnostic confirmed MLP reaches MSE 0.005 in 1000 steps where
chunked-transformer plateaued at 0.99.

Architecture:
    Encoder:  D -> 256 -> latent  (GELU between)
    Decoder:  latent -> 256 -> D  (GELU between)

Sized for ~135M params at D=264449, latent=128.

For the tiny mapper hypernet -> latent, we keep the existing chunked design
since that one was working — the original 70M-param mapper learned a useful
function (just not a smooth one), and we only need it to predict 128 dims.
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

OUT_DIR = PROJECT / "data" / "ae_n100_mlp"
MESH_DIR = OUT_DIR / "meshes"


# ============================================================================
# MLP autoencoder
# ============================================================================

class MLPAE(nn.Module):
    """Encoder + decoder, single combined module for clean state-dict load/save."""
    def __init__(self, D, latent_dim=128, hidden=256):
        super().__init__()
        self.D = D
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, D),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


# ============================================================================
# Tiny mapper: hypernet residual -> latent
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


class TinyMapper(nn.Module):
    """hypernet residual (~17.9M dim) -> latent (128 dim).
    This direction works fine with chunked transformer (encoder-only)."""
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
        return self.to_latent(self.final_norm(x).mean(dim=1))


# ============================================================================
# Utilities
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
# Training
# ============================================================================

def train_autoencoder(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shape_dir = Path(args.shape_dir)
    paths = sorted(shape_dir.glob("obj_*.pt"))
    assert len(paths) == 100, f"expected 100, got {len(paths)}"

    flats, keys, shapes = [], None, None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        f, k, s = flatten_sd(sd, keys)
        if keys is None:
            keys, shapes = k, s
        flats.append(f)
    W = torch.stack(flats).to(device)
    D = W.shape[1]
    print(f"[ae] N=100 shape-SIRENs, dim={D}")

    anchor = torch.load(args.anchor, map_location="cpu", weights_only=True)
    anchor_flat, _, _ = flatten_sd(anchor, keys)
    anchor_flat = anchor_flat.to(device)

    W_res = W - anchor_flat[None]
    res_mean = W_res.mean(0, keepdim=True)
    res_std = W_res.std(0, keepdim=True).clamp_min(1e-6)
    W_norm = (W_res - res_mean) / res_std

    model = MLPAE(D, latent_dim=args.latent_dim, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ae] MLPAE: {n_params:,} params  (D={D}, hidden={args.hidden}, latent={args.latent_dim})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    print(f"[ae] full-batch on 100 shapes for {args.steps} steps, weight MSE")
    for step in range(1, args.steps + 1):
        model.train()
        pred = model(W_norm)
        loss = F.mse_loss(pred, W_norm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "shp_keys": keys,
        "shp_shapes": shapes,
        "anchor_flat": anchor_flat.cpu(),
        "res_mean": res_mean.cpu(),
        "res_std": res_std.cpu(),
        "D": D,
    }, OUT_DIR / "autoencoder.pt")
    print(f"[ae] saved -> {OUT_DIR/'autoencoder.pt'}")

    model.eval()
    with torch.no_grad():
        Z = model.encode(W_norm)
    torch.save(Z.cpu(), OUT_DIR / "latents.pt")
    print(f"[ae] latents shape={tuple(Z.shape)}  -> {OUT_DIR/'latents.pt'}")

    # Sanity: latent variance per dim
    latent_var = Z.var(dim=0)
    print(f"[ae] latent dim variance: min={latent_var.min().item():.3e} "
          f"max={latent_var.max().item():.3e} dead-dims (<1e-6)={(latent_var < 1e-6).sum().item()}")

    return model, Z, keys, shapes, anchor_flat, res_mean, res_std


def reconstruct_and_interpolate(model, Z, keys, shapes, anchor_flat, res_mean, res_std, device):
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    c = CFG.shape_siren
    print("\n[recon] meshing reconstructions for selected shapes")
    for idx in [3, 6, 22, 36, 49, 70, 99]:
        if idx >= Z.shape[0]:
            continue
        z = Z[idx:idx+1].to(device)
        with torch.no_grad():
            recon_norm = model.decode(z).cpu()
        recon_abs = recon_norm * res_std.cpu() + res_mean.cpu() + anchor_flat[None].cpu()
        siren = unflatten_to_siren(recon_abs[0], keys, shapes, c, device)
        out = MESH_DIR / f"recon_obj_{idx:02d}.obj"
        result = mesh_siren(siren, device, out)
        if result:
            print(f"  obj_{idx:02d}: {result[0]} v / {result[1]} f -> {out.name}")
        else:
            print(f"  obj_{idx:02d}: NO MESH (off-manifold)")

    print("\n[interp] obj_03 -> obj_06")
    z_a, z_b = Z[3:4].to(device), Z[6:7].to(device)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_t = (1 - t) * z_a + t * z_b
        with torch.no_grad():
            recon_norm = model.decode(z_t).cpu()
        recon_abs = recon_norm * res_std.cpu() + res_mean.cpu() + anchor_flat[None].cpu()
        siren = unflatten_to_siren(recon_abs[0], keys, shapes, c, device)
        out = MESH_DIR / f"interp_03_to_06_t{int(t*100):03d}.obj"
        result = mesh_siren(siren, device, out)
        if result:
            print(f"  t={t:.2f}: {result[0]:>7d} v / {result[1]:>7d} f")
        else:
            print(f"  t={t:.2f}: NO MESH")


def train_tiny_mapper(args, Z, device):
    from hypernet_to_shape_mapper import ResidualPairedWeightsDataset
    print("\n[mapper] training tiny mapper hypernet -> latent")
    ds = ResidualPairedWeightsDataset(args.manifest, args.anchor_hyp, args.anchor_shp, device=device)
    H = ds.hyp_norm
    Z_cpu = Z.cpu()
    cond_dim, latent_dim = H.shape[1], Z.shape[1]
    N = H.shape[0]

    mapper = TinyMapper(cond_dim, latent_dim).to(device)
    n_params = sum(p.numel() for p in mapper.parameters())
    print(f"[mapper] params: {n_params:,}")
    opt = torch.optim.AdamW(mapper.parameters(), lr=args.mapper_lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.mapper_steps, eta_min=1e-6)

    bs = min(16, N)
    print(f"[mapper] bs={bs}, {args.mapper_steps} steps")
    import random as _random
    rng = _random.Random(0)

    for step in range(1, args.mapper_steps + 1):
        mapper.train()
        idx = rng.sample(range(N), bs)
        h_b = H[idx].to(device)
        z_b = Z_cpu[idx].to(device)
        pred = mapper(h_b)
        loss = F.mse_loss(pred, z_b)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 1 or step % max(1, args.mapper_steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")

    mapper.eval()
    per_mse = []
    with torch.no_grad():
        for i in range(N):
            p = mapper(H[i:i+1].to(device))
            per_mse.append(((p - Z_cpu[i:i+1].to(device)) ** 2).mean().item())
    per_mse = np.array(per_mse)
    print(f"\n[mapper] per-shape latent MSE: mean={per_mse.mean():.4e} max={per_mse.max():.4e}")

    perm = torch.randperm(N)
    with torch.no_grad():
        scram, zero = [], []
        for i in range(N):
            ps = mapper(H[perm[i]:perm[i]+1].to(device))
            scram.append(((ps - Z_cpu[i:i+1].to(device))**2).mean().item())
            pz = mapper(torch.zeros_like(H[i:i+1]).to(device))
            zero.append(((pz - Z_cpu[i:i+1].to(device))**2).mean().item())
    scram = np.array(scram); zero = np.array(zero)
    print(f"[mapper] scrambled mean: {scram.mean():.4e}  ratio: {scram.mean()/per_mse.mean():.2f}x")
    print(f"[mapper] zero-cond mean: {zero.mean():.4e}  ratio: {zero.mean()/per_mse.mean():.2f}x")

    torch.save({
        "mapper": mapper.state_dict(),
        "cond_dim": cond_dim, "latent_dim": latent_dim,
        "args": vars(args),
        "ds_hyp_names": ds.hyp_names,
        "ds_anchor_hyp": ds.anchor_hyp.cpu(),
        "ds_hyp_mean": ds.hyp_mean.cpu(),
        "ds_hyp_std": ds.hyp_std.cpu(),
    }, OUT_DIR / "latent_mapper.pt")
    return mapper, ds


def ood_test(mapper, model, ds, anchor_flat, res_mean, res_std, keys, shapes, device):
    print("\n[ood] testing on obj_100 hypernet")
    ood_path = PROJECT / "data" / "ood_test" / "obj_100_hypernet.pt"
    if not ood_path.exists():
        print(f"[ood] missing {ood_path}; skip")
        return
    sd = torch.load(ood_path, map_location=device, weights_only=True)
    flat = torch.cat([sd[k].detach().float().flatten() for k in ds.hyp_names]).to(device)
    res = flat - ds.anchor_hyp.to(device)
    norm = (res - ds.hyp_mean.to(device).squeeze()) / ds.hyp_std.to(device).squeeze()

    mapper.eval(); model.eval()
    with torch.no_grad():
        z_pred = mapper(norm.unsqueeze(0))
        recon_norm = model.decode(z_pred).cpu()
    recon_abs = recon_norm * res_std.cpu() + res_mean.cpu() + anchor_flat[None].cpu()
    c = CFG.shape_siren
    siren = unflatten_to_siren(recon_abs[0], keys, shapes, c, device)
    out = MESH_DIR / "ood_obj_100_predicted_via_latent.obj"
    result = mesh_siren(siren, device, out)
    if result:
        print(f"[ood] PREDICTED: {result[0]} v / {result[1]} f -> {out.name}")
    else:
        print(f"[ood] PREDICTED: no zero crossing")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape_dir", default="/workspace/hypernet/data/shape_sirens")
    p.add_argument("--anchor",    default="/workspace/hypernet/checkpoints/anchor_shape_siren.pt")
    p.add_argument("--manifest",  default="/workspace/hypernet/scripts/manifest_n100.pt")
    p.add_argument("--anchor_hyp", default="/workspace/hypernet/data/checkpoints/anchor_hypernet.pt")
    p.add_argument("--anchor_shp", default="/workspace/hypernet/checkpoints/anchor_shape_siren.pt")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--steps",     type=int, default=4000)
    p.add_argument("--mapper_lr", type=float, default=1e-3)
    p.add_argument("--mapper_steps", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, Z, keys, shapes, anchor_flat, res_mean, res_std = train_autoencoder(args, device)
    reconstruct_and_interpolate(model, Z.to(device), keys, shapes, anchor_flat, res_mean, res_std, device)
    mapper, ds = train_tiny_mapper(args, Z.to(device), device)
    ood_test(mapper, model, ds, anchor_flat, res_mean, res_std, keys, shapes, device)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
