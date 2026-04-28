"""
Hypernet autoencoder pipeline for the hypernet weight space.

Designed to fit in 16 GB GPU memory using mini-batch training:
    - 100 hypernets stay on CPU (~7 GB system RAM)
    - Mini-batches of 8 hypernets cross to GPU per step (~580 MB activations)
    - hidden=64 bottleneck keeps params at ~2.3 GB
    - Total GPU footprint: ~12 GB

Stages:
    1. Train hypernet AE on 100 hypernets (mini-batch, weight MSE)
    2. Encode all 100 hypernets to image latents (256-dim)
    3. Train tiny image-to-shape latent mapper (256 -> 128)
    4. OOD test on obj_100 hypernet through the full new pipeline

If hidden=64 underperforms, increase to 128 (still fits with mini-batch=4).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("[warn] bitsandbytes not installed; falling back to torch.AdamW")

PROJECT = Path("/workspace/hypernet")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from configs.config import CFG
from src.siren import SIREN

OUT_DIR = PROJECT / "data" / "ae_hypernet_n100"
MESH_DIR = OUT_DIR / "meshes"


# ============================================================================
# Hypernet MLP autoencoder
# ============================================================================


class CPUOffloadedAdamW:
    """Manual AdamW where optimizer state (m, v) lives on CPU.

    Per-step flow:
        1. Read gradients from GPU model (already there from .backward())
        2. For each param: copy grad to CPU, update CPU-resident m/v, compute step
        3. Copy updated param back to GPU
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # CPU-resident optimizer state, one per param
        self.m = []
        self.v = []
        for p in self.params:
            self.m.append(torch.zeros_like(p, device="cpu", dtype=torch.float32))
            self.v.append(torch.zeros_like(p, device="cpu", dtype=torch.float32))

        # Group view for compatibility with LR schedulers
        self.param_groups = [{"params": self.params, "lr": lr}]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        bc1 = 1 - self.beta1 ** self.step_count
        bc2 = 1 - self.beta2 ** self.step_count
        lr = self.param_groups[0]["lr"]

        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            # Move grad to CPU for the update math
            g_cpu = p.grad.detach().to("cpu", non_blocking=True).float()

            # AdamW: weight decay applied directly to param
            if self.weight_decay > 0:
                p.data.mul_(1 - lr * self.weight_decay)

            # Update first and second moments on CPU
            m.mul_(self.beta1).add_(g_cpu, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g_cpu, g_cpu, value=1 - self.beta2)

            # Bias-corrected step: -lr * (m_hat) / (sqrt(v_hat) + eps)
            m_hat = m / bc1
            v_hat = v / bc2
            update = m_hat / (v_hat.sqrt() + self.eps)

            # Apply update on GPU
            p.data.add_(update.to(p.device, non_blocking=True), alpha=-lr)


class HypernetAE(nn.Module):
    """Encoder + decoder. Same shape as the working shape AE but bigger D."""
    def __init__(self, D, latent_dim=256, hidden=64):
        super().__init__()
        self.D = D
        self.latent_dim = latent_dim
        # Encoder: D -> hidden -> latent
        self.encoder = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        # Decoder: latent -> hidden -> D
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
# Image-latent -> shape-latent mapper (trivial regression)
# ============================================================================

class LatentMapper(nn.Module):
    """256-dim image latent -> 128-dim shape latent.
    A tiny MLP since both are low-dim."""
    def __init__(self, in_dim=256, out_dim=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_image):
        return self.net(z_image)


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


# ============================================================================
# Stage 1: train hypernet AE
# ============================================================================

def train_hypernet_ae(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load 100 hypernets to CPU
    hyp_dir = Path(args.hypernet_dir)
    paths = sorted(hyp_dir.glob("obj_*.pt"))
    assert len(paths) == 100, f"expected 100 hypernets, got {len(paths)}"

    print(f"[hyp-ae] loading 100 hypernets to CPU...")
    flats, keys, shapes = [], None, None
    for i, p in enumerate(paths):
        sd = torch.load(p, map_location="cpu", weights_only=True)
        f, k, s = flatten_sd(sd, keys)
        if keys is None:
            keys, shapes = k, s
        flats.append(f)
        if (i + 1) % 25 == 0:
            print(f"  loaded {i + 1}/100")
    H = torch.stack(flats)  # CPU, (100, D)
    D = H.shape[1]
    print(f"[hyp-ae] H shape: {H.shape}  on CPU")

    # Anchor for residual parameterization
    anchor = torch.load(args.anchor_hyp, map_location="cpu", weights_only=True)
    anchor_flat, _, _ = flatten_sd(anchor, keys)

    H_res = H - anchor_flat[None]  # (100, D), CPU
    res_mean = H_res.mean(0, keepdim=True)
    res_std = H_res.std(0, keepdim=True).clamp_min(1e-6)
    H_norm = (H_res - res_mean) / res_std  # CPU
    print(f"[hyp-ae] H_norm var: {H_norm.var().item():.4f}")

    # Free the original H tensor to save RAM (we have H_norm + anchor + stats)
    del H, H_res
    import gc; gc.collect()

    # Build model
    model = HypernetAE(D, latent_dim=args.latent_dim, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_gb = n_params * 4 / 1e9
    print(f"[hyp-ae] HypernetAE: {n_params:,} params (~{n_gb:.2f} GB fp32)  "
          f"D={D}  hidden={args.hidden}  latent={args.latent_dim}")

    if args.use_cpu_optim:
        print("[hyp-ae] using CPU-offloaded AdamW (optimizer state on CPU)")
        opt = CPUOffloadedAdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    elif HAS_BNB and not args.no_8bit:
        print("[hyp-ae] using bitsandbytes 8-bit AdamW")
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:
        print("[hyp-ae] using torch AdamW (full fp32 optimizer state)")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Manual cosine schedule (don't use torch scheduler with custom optimizer)
    def get_lr(step):
        import math
        if step <= args.warmup_steps:
            return args.lr * (step / max(args.warmup_steps, 1))
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * progress))

    # Mini-batch training
    N = H_norm.shape[0]
    bs = args.batch_size
    accum = args.accum_steps
    eff_bs = bs * accum
    warmup = args.warmup_steps
    print(f"[hyp-ae] mini-batch on N={N}, bs={bs}, accum={accum}, eff_bs={eff_bs}, "
          f"{args.steps} steps, lr={args.lr}, warmup={warmup}")

    import random as _random
    rng = _random.Random(0)
    for step in range(1, args.steps + 1):
        model.train()
        # Manual cosine + warmup
        cur_lr = get_lr(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for k in range(accum):
            idx = rng.sample(range(N), bs)
            x = H_norm[idx].to(device, non_blocking=True)
            recon = model(x)
            loss = F.mse_loss(recon, x) / accum
            loss.backward()
            total_loss += loss.item() * accum  # un-scale for logging

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {total_loss/accum:.4e}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  gn {gn.item():.2e}")

    # Save AE
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "hyp_keys": keys,
        "hyp_shapes": shapes,
        "anchor_flat": anchor_flat,    # CPU
        "res_mean": res_mean,          # CPU
        "res_std": res_std,            # CPU
        "D": D,
    }, OUT_DIR / "autoencoder.pt")
    print(f"[hyp-ae] saved -> {OUT_DIR/'autoencoder.pt'}")

    # Encode all 100 hypernets, one mini-batch at a time
    model.eval()
    z_chunks = []
    with torch.no_grad():
        for i in range(0, N, bs):
            x = H_norm[i:i + bs].to(device, non_blocking=True)
            z_chunks.append(model.encode(x).cpu())
    Z_image = torch.cat(z_chunks, 0)
    torch.save(Z_image, OUT_DIR / "image_latents.pt")
    print(f"[hyp-ae] image latents shape={tuple(Z_image.shape)}  -> {OUT_DIR/'image_latents.pt'}")

    # Diagnostic: how varied are the latents?
    print(f"[hyp-ae] image latent dim variance: "
          f"min={Z_image.var(0).min().item():.3e}  "
          f"max={Z_image.var(0).max().item():.3e}  "
          f"dead-dims (<1e-6)={(Z_image.var(0) < 1e-6).sum().item()}")

    return model, Z_image, keys, shapes, anchor_flat, res_mean, res_std


# ============================================================================
# Stage 2: train image-latent -> shape-latent mapper
# ============================================================================

def train_latent_mapper(args, Z_image, device):
    """Train tiny mapper image_latent (256) -> shape_latent (128).

    The shape latents come from the previously trained shape autoencoder
    at /workspace/hypernet/data/ae_n100_mlp/latents.pt
    """
    print("\n[latent-mapper] training image_latent -> shape_latent")

    # Load the shape latents (one per training shape, 128-dim)
    shape_latents_path = PROJECT / "data" / "ae_n100_mlp" / "latents.pt"
    Z_shape = torch.load(shape_latents_path, map_location="cpu", weights_only=True)
    print(f"[latent-mapper] image latents: {tuple(Z_image.shape)}  "
          f"shape latents: {tuple(Z_shape.shape)}")

    assert Z_image.shape[0] == Z_shape.shape[0], \
        f"image/shape latent count mismatch: {Z_image.shape[0]} vs {Z_shape.shape[0]}"

    # Move to GPU — these are tiny (100 * 256 + 100 * 128 = ~150 KB)
    Z_image_gpu = Z_image.to(device)
    Z_shape_gpu = Z_shape.to(device)

    in_dim = Z_image.shape[1]
    out_dim = Z_shape.shape[1]
    mapper = LatentMapper(in_dim, out_dim, hidden=512).to(device)
    n_params = sum(p.numel() for p in mapper.parameters())
    print(f"[latent-mapper] params: {n_params:,}")

    opt = torch.optim.AdamW(mapper.parameters(), lr=args.mapper_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.mapper_steps, eta_min=1e-6,
    )

    print(f"[latent-mapper] full-batch on N=100, {args.mapper_steps} steps, lr={args.mapper_lr}")
    for step in range(1, args.mapper_steps + 1):
        mapper.train()
        pred = mapper(Z_image_gpu)
        loss = F.mse_loss(pred, Z_shape_gpu)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
        opt.step(); sched.step()

        if step == 1 or step % max(1, args.mapper_steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")

    # Per-shape eval
    mapper.eval()
    with torch.no_grad():
        per_mse = []
        for i in range(Z_image_gpu.shape[0]):
            p = mapper(Z_image_gpu[i:i + 1])
            per_mse.append(((p - Z_shape_gpu[i:i + 1]) ** 2).mean().item())
    per_mse = np.array(per_mse)
    print(f"\n[latent-mapper] per-shape MSE: mean={per_mse.mean():.4e}  "
          f"max={per_mse.max():.4e}  min={per_mse.min():.4e}")

    # Ablation
    perm = torch.randperm(Z_image_gpu.shape[0])
    with torch.no_grad():
        scram = []
        zero = []
        for i in range(Z_image_gpu.shape[0]):
            ps = mapper(Z_image_gpu[perm[i]:perm[i] + 1])
            scram.append(((ps - Z_shape_gpu[i:i + 1]) ** 2).mean().item())
            pz = mapper(torch.zeros_like(Z_image_gpu[i:i + 1]))
            zero.append(((pz - Z_shape_gpu[i:i + 1]) ** 2).mean().item())
    scram = np.array(scram); zero = np.array(zero)
    print(f"[latent-mapper] scrambled mean: {scram.mean():.4e}  "
          f"ratio: {scram.mean() / per_mse.mean():.2f}x")
    print(f"[latent-mapper] zero-cond mean: {zero.mean():.4e}  "
          f"ratio: {zero.mean() / per_mse.mean():.2f}x")

    torch.save({
        "mapper": mapper.state_dict(),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "args": vars(args),
    }, OUT_DIR / "latent_mapper.pt")
    print(f"[latent-mapper] saved -> {OUT_DIR/'latent_mapper.pt'}")

    return mapper


# ============================================================================
# Stage 3: OOD test on obj_100 through the full new pipeline
# ============================================================================

def ood_test(hyp_ae, latent_mapper, keys_hyp, anchor_hyp_flat, res_mean, res_std, device):
    """Run obj_100's hypernet through:
       hypernet_ae.encode -> image_latent -> latent_mapper -> shape_latent ->
       shape_decoder -> shape weights -> SIREN -> mesh
    """
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[ood] testing obj_100 through hypernet AE + latent mapper + shape decoder")

    # Load obj_100's hypernet
    ood_hyp_path = PROJECT / "data" / "ood_test" / "obj_100_hypernet.pt"
    if not ood_hyp_path.exists():
        print(f"[ood] {ood_hyp_path} missing, skip OOD test")
        return
    sd = torch.load(ood_hyp_path, map_location="cpu", weights_only=True)
    flat = torch.cat([sd[k].detach().float().flatten() for k in keys_hyp])
    print(f"[ood] obj_100 hypernet flattened: shape={tuple(flat.shape)}")

    # Standardize as residual
    res = flat - anchor_hyp_flat
    norm = (res - res_mean.squeeze()) / res_std.squeeze()
    norm = norm.unsqueeze(0).to(device)

    # Forward through new pipeline
    hyp_ae.eval()
    latent_mapper.eval()

    with torch.no_grad():
        z_image = hyp_ae.encode(norm)            # (1, 256)
        print(f"[ood] z_image norm: {z_image.norm().item():.3f}")

        z_shape_pred = latent_mapper(z_image)    # (1, 128)
        print(f"[ood] z_shape predicted norm: {z_shape_pred.norm().item():.3f}")

    # Now run z_shape through the shape decoder (from the previous AE)
    shape_ae_ckpt = torch.load(
        PROJECT / "data" / "ae_n100_mlp" / "autoencoder.pt",
        map_location=device, weights_only=False,
    )
    # Reconstruct the shape AE so we can call decode
    # Note: re-import from the working pipeline since we trust that architecture
    sys.path.insert(0, str(PROJECT / "scripts"))
    from autoencoder_pipeline_n100_mlp import MLPAE
    shape_D = shape_ae_ckpt["D"]
    shape_ae = MLPAE(shape_D,
                     latent_dim=shape_ae_ckpt["args"]["latent_dim"],
                     hidden=shape_ae_ckpt["args"]["hidden"]).to(device)
    shape_ae.load_state_dict(shape_ae_ckpt["model"])
    shape_ae.eval()

    with torch.no_grad():
        recon_norm = shape_ae.decode(z_shape_pred).cpu()

    # Unflatten and mesh
    res_mean_shape = shape_ae_ckpt["res_mean"].cpu()
    res_std_shape  = shape_ae_ckpt["res_std"].cpu()
    anchor_shape   = shape_ae_ckpt["anchor_flat"].cpu()
    recon_abs = recon_norm * res_std_shape + res_mean_shape + anchor_shape[None]

    shp_keys   = shape_ae_ckpt["shp_keys"]
    shp_shapes = shape_ae_ckpt["shp_shapes"]
    c = CFG.shape_siren

    siren = unflatten_to_siren(recon_abs[0], shp_keys, shp_shapes, c, device)
    out = MESH_DIR / "ood_obj_100_predicted_via_two_AEs.obj"
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
    p.add_argument("--hypernet_dir",
                   default="/workspace/hypernet/data/hypernets")
    p.add_argument("--anchor_hyp",
                   default="/workspace/hypernet/data/checkpoints/anchor_hypernet.pt")

    # AE hyperparameters
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256,
                   help="encoder hidden dim. 256 needs 8-bit AdamW + 48GB GPU")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=8)

    # Mapper hyperparameters
    p.add_argument("--mapper_lr", type=float, default=1e-3)
    p.add_argument("--mapper_steps", type=int, default=8000)

    p.add_argument("--accum_steps", type=int, default=4,
                   help="gradient accumulation: effective batch = batch_size * accum_steps")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="standard clipping with bigger hidden (no longer fighting OOM)")
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="linear LR warmup over first N steps")
    p.add_argument("--no_8bit", action="store_true",
                   help="disable bitsandbytes 8-bit Adam, use full fp32 optimizer")
    p.add_argument("--use_cpu_optim", action="store_true", default=True,
                   help="use CPU-offloaded AdamW (default; needed for h>=128)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    hyp_ae, Z_image, keys_hyp, _, anchor_hyp_flat, res_mean, res_std = \
        train_hypernet_ae(args, device)
    latent_mapper = train_latent_mapper(args, Z_image, device)
    ood_test(hyp_ae, latent_mapper, keys_hyp, anchor_hyp_flat, res_mean, res_std, device)

    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
