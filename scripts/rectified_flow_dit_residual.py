"""
Rectified Flow Transformer v3 — RESIDUAL training against shared anchors.

Problem v1/v2 hit: raw hypernet and shape-SIREN weights share ~99% cosine
direction because all were warm-started from a shared anchor. The flow learned
the mean (the anchor) and ignored per-shape structure, plateauing at
standardized MSE ~1.17.

Fix: subtract anchor before training. Target becomes the per-shape offset
(residual), whose pairwise cosines drop from ~0.995 to ~0.1-0.3. Same for
conditioning: cond becomes hypernet - anchor_hypernet.

At sampling: add anchor back to denormalized prediction.

Architecture is identical to v2 (AdaLN-Zero DiT); only the dataset changed.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------
# Chunked projector
# ----------------------------------------------------------------------------

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


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t):
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.mlp(emb)


class CondEncoder(nn.Module):
    def __init__(self, d_model, n_heads=4, n_layers=2, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(d_model),
                "attn":  nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "norm2": nn.LayerNorm(d_model),
                "ffn":   nn.Sequential(
                    nn.Linear(d_model, d_model * ff_mult),
                    nn.GELU(),
                    nn.Linear(d_model * ff_mult, d_model),
                ),
            }))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        x = tokens
        for L in self.layers:
            h = L["norm1"](x)
            x = x + L["attn"](h, h, h, need_weights=False)[0]
            x = x + L["ffn"](L["norm2"](x))
        x = self.final_norm(x)
        return x.mean(dim=1)


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
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        s_a, sc_a, g_a, s_f, sc_f, g_f = self.adaLN_modulation(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s_a, sc_a)
        x = x + g_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), s_f, sc_f)
        x = x + g_f.unsqueeze(1) * self.ffn(h)
        return x


class RectifiedFlowDiT(nn.Module):
    def __init__(self, shape_dim, cond_dim, chunk_shape=1024, chunk_cond=8192,
                 d_model=512, n_layers=8, n_heads=8, ff_mult=4, cond_enc_layers=2):
        super().__init__()
        self.shape_proj = ChunkedProjector(shape_dim, chunk_shape, d_model)
        self.cond_proj = ChunkedProjector(cond_dim, chunk_cond, d_model)
        self.cond_enc = CondEncoder(d_model, n_heads=n_heads, n_layers=cond_enc_layers, ff_mult=ff_mult)
        self.time_embed = TimeEmbedding(d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        print(f"[model] shape tokens={self.shape_proj.num_chunks}  "
              f"cond tokens={self.cond_proj.num_chunks}  d_model={d_model}  "
              f"layers={n_layers}  heads={n_heads}  cond_enc_layers={cond_enc_layers}")
        print(f"[model] trainable params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x_t, t, cond_w):
        x_tok = self.shape_proj.tokenize(x_t)
        c_tok = self.cond_proj.tokenize(cond_w)
        c_vec = self.cond_enc(c_tok)
        t_vec = self.time_embed(t)
        c = c_vec + t_vec
        for blk in self.blocks:
            x_tok = blk(x_tok, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x_tok = modulate(self.final_norm(x_tok), shift, scale)
        return self.shape_proj.detokenize(x_tok)


# ----------------------------------------------------------------------------
# Objective + sampler
# ----------------------------------------------------------------------------

def rectified_flow_loss(model, x1, cond_w):
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    v_target = x1 - x0
    v_pred = model(x_t, t, cond_w)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample(model, cond_w, n_steps=50, shape_dim=None):
    model.eval()
    B = cond_w.shape[0]
    if shape_dim is None:
        shape_dim = model.shape_proj.total_dim
    x = torch.randn(B, shape_dim, device=cond_w.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=cond_w.device)
        v = model(x, t, cond_w)
        x = x + dt * v
    return x


# ----------------------------------------------------------------------------
# Residual dataset
# ----------------------------------------------------------------------------

def _unwrap_sd(obj):
    """Anchor files wrap the state_dict under 'state_dict' key; per-object
    files save the bare state_dict directly. Handle both."""
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def flatten_state_dict(sd, keys=None):
    sd = _unwrap_sd(sd)
    if keys is None:
        keys = list(sd.keys())
    tensors = [sd[k].detach().float().flatten() for k in keys]
    flat = torch.cat(tensors)
    shapes = [tuple(sd[k].shape) for k in keys]
    return flat, keys, shapes


class ResidualPairedWeightsDataset(Dataset):
    """Stores the residuals (weights - anchor) and keeps anchors for reconstruction.

    Standardization is done on the residuals — mean will be ~0 (by construction
    if the saved anchor equals the empirical mean; close otherwise), std brings
    the targets into the unit-variance regime flow expects.
    """

    def __init__(self, manifest_path, anchor_hyp_path, anchor_shp_path, device):
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        hyp_paths = manifest["hypernet_paths"]
        shp_paths = manifest["shape_paths"]
        assert len(hyp_paths) == len(shp_paths)

        hyp_flats, shp_flats = [], []
        self.hyp_names = self.hyp_shapes = None
        self.shp_names = self.shp_shapes = None

        for hp, sp in zip(hyp_paths, shp_paths):
            hyp_sd = torch.load(hp, map_location="cpu")
            shp_sd = torch.load(sp, map_location="cpu")
            hf, hn, hs = flatten_state_dict(hyp_sd, self.hyp_names)
            sf, sn, ss = flatten_state_dict(shp_sd, self.shp_names)
            if self.hyp_names is None:
                self.hyp_names, self.hyp_shapes = hn, hs
                self.shp_names, self.shp_shapes = sn, ss
            hyp_flats.append(hf)
            shp_flats.append(sf)

        hyp = torch.stack(hyp_flats).to(device)
        shp = torch.stack(shp_flats).to(device)

        anchor_hyp_sd = torch.load(anchor_hyp_path, map_location="cpu")
        anchor_shp_sd = torch.load(anchor_shp_path, map_location="cpu")
        anchor_hyp, _, _ = flatten_state_dict(anchor_hyp_sd, self.hyp_names)
        anchor_shp, _, _ = flatten_state_dict(anchor_shp_sd, self.shp_names)
        self.anchor_hyp = anchor_hyp.to(device)
        self.anchor_shp = anchor_shp.to(device)

        # Residuals: everything past here is the per-shape offset from anchor
        hyp_res = hyp - self.anchor_hyp[None]
        shp_res = shp - self.anchor_shp[None]

        # Standardize the residuals. Per-parameter std gives each dim unit
        # variance; we floor to avoid div-by-zero on dead weights.
        self.shp_mean = shp_res.mean(dim=0, keepdim=True)
        self.shp_std  = shp_res.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.shp_norm = (shp_res - self.shp_mean) / self.shp_std

        self.hyp_mean = hyp_res.mean(dim=0, keepdim=True)
        self.hyp_std  = hyp_res.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.hyp_norm = (hyp_res - self.hyp_mean) / self.hyp_std

        # Diagnostic: confirm the residual-cosine separation we expect
        Xn = F.normalize(shp_res, dim=1)
        cos = (Xn @ Xn.T)
        off = cos[~torch.eye(len(cos), dtype=torch.bool, device=cos.device)]
        print(f"[data] N={len(shp)}  shape_dim={shp.shape[1]:,}  cond_dim={hyp.shape[1]:,}")
        print(f"[data] shape-residual pairwise cosine: mean={off.mean():.3f}  max={off.max():.3f}  min={off.min():.3f}")

    def denormalize_shape_residual(self, x_norm):
        return x_norm * self.shp_std + self.shp_mean

    def reconstruct_shape_weights(self, x_norm):
        """x_norm -> absolute weight vector (residual * std + mean + anchor)."""
        return self.denormalize_shape_residual(x_norm) + self.anchor_shp[None]


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

def train_main(args):
    device = torch.device(args.device)
    ds = ResidualPairedWeightsDataset(
        args.manifest, args.anchor_hyp, args.anchor_shp, device=device,
    )

    model = RectifiedFlowDiT(
        shape_dim=ds.shp_norm.shape[1],
        cond_dim=ds.hyp_norm.shape[1],
        chunk_shape=args.chunk_shape,
        chunk_cond=args.chunk_cond,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        cond_enc_layers=args.cond_enc_layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_every = max(1, args.steps // 200)
    hyp_all = ds.hyp_norm
    shp_all = ds.shp_norm

    print(f"[train] full-batch on N={len(ds.shp_norm)} for {args.steps} steps")
    for step in range(1, args.steps + 1):
        model.train()
        loss = rectified_flow_loss(model, shp_all, hyp_all)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            print(f"step {step:6d} / {args.steps}   loss={loss.item():.4e}   lr={sched.get_last_lr()[0]:.2e}")

    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "shp_mean": ds.shp_mean.cpu(),
        "shp_std": ds.shp_std.cpu(),
        "hyp_mean": ds.hyp_mean.cpu(),
        "hyp_std": ds.hyp_std.cpu(),
        "anchor_shp": ds.anchor_shp.cpu(),
        "anchor_hyp": ds.anchor_hyp.cpu(),
        "shp_names": ds.shp_names,
        "shp_shapes": ds.shp_shapes,
        "hyp_names": ds.hyp_names,
        "hyp_shapes": ds.hyp_shapes,
    }
    torch.save(ckpt, out_dir / "rectified_flow.pt")
    print(f"[save] -> {out_dir/'rectified_flow.pt'}")

    print("[eval] sampling training conditions for memorization check")
    with torch.no_grad():
        x_pred_norm = sample(model, hyp_all, n_steps=args.n_sample_steps)
        per_shape_mse = ((x_pred_norm - shp_all) ** 2).mean(dim=1).cpu().numpy()
    print("[eval] per-shape sampled MSE (standardized residual space):")
    for i, m in enumerate(per_shape_mse):
        print(f"       shape {i:2d}: {m:.4e}")
    print(f"[eval] mean={per_shape_mse.mean():.4e}  max={per_shape_mse.max():.4e}")

    # Reconstruct absolute weights (anchor + residual) for downstream eval
    x_abs = ds.reconstruct_shape_weights(x_pred_norm).cpu()
    torch.save(x_abs, out_dir / "predicted_shape_weights.pt")
    print(f"[save] predicted absolute weights -> {out_dir/'predicted_shape_weights.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--anchor_hyp", default="/workspace/hypernet/data/checkpoints/anchor_hypernet.pt")
    p.add_argument("--anchor_shp", default="/workspace/hypernet/checkpoints/anchor_shape_siren.pt")
    p.add_argument("--out", default="./rf_out")

    p.add_argument("--chunk_shape", type=int, default=1024)
    p.add_argument("--chunk_cond",  type=int, default=8192)

    p.add_argument("--d_model",  type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads",  type=int, default=8)
    p.add_argument("--ff_mult",  type=int, default=4)
    p.add_argument("--cond_enc_layers", type=int, default=2)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=10000)

    p.add_argument("--n_sample_steps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train_main(parse_args())
