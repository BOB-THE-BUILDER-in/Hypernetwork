"""
Deterministic hypernet -> shape-SIREN mapping (Route B).

Replaces rectified flow with a direct regression: given hypernet weights,
predict the shape-SIREN residual in a single forward pass. Loss is plain MSE
on standardized residuals.

Why this beats flow for N=10:
    - No per-step velocity error compounding; prediction is one shot
    - Direct supervision on the target, no stochastic (t, x0) sampling
    - Model capacity goes entirely to the mapping, not to modelling a field
      over the whole noise-to-data trajectory

The core paper claim -- "hypernetwork weights encode the shape that their
image-SIRENs describe" -- is tested more cleanly here than with a flow.

Architecture:
    hypernet_w (17.9M)  --chunk(8192)-->  cond tokens (2195, d_model)
                                                  |
    learnable shape query tokens (259, d_model)   |
                               |                  |
                          self-attn   cross-attn on cond
                               |
                         unchunk to shape residual (264K)

No time embedding. AdaLN injects a pooled cond vector globally (same stabilizing
mechanism the DiT used); cross-attention gives the shape queries direct access
to the cond structure. Zero-init final layer so it starts predicting the mean.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Same chunked projector from the flow work
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


# ----------------------------------------------------------------------------
# Transformer block: self-attn (on shape queries) + cross-attn (to cond) + FFN
# AdaLN-Zero driven by a pooled cond vector keeps gates under control
# ----------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MappingBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm2_q  = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2_kv = nn.LayerNorm(d_model)  # cond tokens get a stable norm
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        # 9 modulation vectors: shift/scale/gate for self-attn, cross-attn, FFN
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model, bias=True))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c_pool, c_tokens):
        s_a, sc_a, g_a, s_c, sc_c, g_c, s_f, sc_f, g_f = self.adaLN(c_pool).chunk(9, dim=-1)

        h = modulate(self.norm1(x), s_a, sc_a)
        x = x + g_a.unsqueeze(1) * self.self_attn(h, h, h, need_weights=False)[0]

        q = modulate(self.norm2_q(x), s_c, sc_c)
        kv = self.norm2_kv(c_tokens)
        x = x + g_c.unsqueeze(1) * self.cross_attn(q, kv, kv, need_weights=False)[0]

        h = modulate(self.norm3(x), s_f, sc_f)
        x = x + g_f.unsqueeze(1) * self.ffn(h)
        return x


# ----------------------------------------------------------------------------
# Cond encoder: small self-attn stack, then mean-pool for AdaLN vector
# (the cross-attn branches see the full cond tokens, not the pooled vector)
# ----------------------------------------------------------------------------

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
        return x, x.mean(dim=1)


# ----------------------------------------------------------------------------
# Main model: learnable shape queries + cross-attention stack
# ----------------------------------------------------------------------------

class HypernetToShapeMapper(nn.Module):
    def __init__(
        self,
        shape_dim,
        cond_dim,
        chunk_shape=1024,
        chunk_cond=8192,
        d_model=512,
        n_layers=8,
        n_heads=8,
        ff_mult=4,
        cond_enc_layers=2,
    ):
        super().__init__()
        self.shape_proj = ChunkedProjector(shape_dim, chunk_shape, d_model)
        self.cond_proj = ChunkedProjector(cond_dim, chunk_cond, d_model)
        self.cond_enc = CondEncoder(d_model, n_heads=n_heads, n_layers=cond_enc_layers, ff_mult=ff_mult)

        # Learnable query tokens, one per shape chunk. These replace the flow's
        # noisy x_t input -- the mapper doesn't take a noisy-target input at all.
        self.shape_queries = nn.Parameter(
            torch.randn(1, self.shape_proj.num_chunks, d_model) * 0.02
        )

        self.blocks = nn.ModuleList(
            [MappingBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # Detokenizer is zero-init via the projector's Linear -- force it so the
        # initial prediction is 0 (i.e. the mean residual, which equals the
        # anchor when combined with the reconstruction step)
        nn.init.zeros_(self.shape_proj.proj_out.weight)
        nn.init.zeros_(self.shape_proj.proj_out.bias)

        print(f"[model] shape tokens={self.shape_proj.num_chunks}  "
              f"cond tokens={self.cond_proj.num_chunks}  d_model={d_model}  "
              f"layers={n_layers}  heads={n_heads}  cond_enc_layers={cond_enc_layers}")
        print(f"[model] trainable params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, cond_w):
        B = cond_w.shape[0]

        c_tok_raw = self.cond_proj.tokenize(cond_w)       # (B, Nk, d)
        c_tok, c_pool = self.cond_enc(c_tok_raw)          # (B, Nk, d), (B, d)

        # Broadcast learnable queries across batch, add the chunk position embed
        x = self.shape_queries.expand(B, -1, -1) + self.shape_proj.pos_embed

        for blk in self.blocks:
            x = blk(x, c_pool, c_tok)

        shift, scale = self.final_adaLN(c_pool).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.shape_proj.detokenize(x)             # (B, shape_dim)


# ----------------------------------------------------------------------------
# Dataset (identical to v3)
# ----------------------------------------------------------------------------

def _unwrap_sd(obj):
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


class ResidualPairedWeightsDataset:
    def __init__(self, manifest_path, anchor_hyp_path, anchor_shp_path, device):
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        hyp_paths = manifest["hypernet_paths"]
        shp_paths = manifest["shape_paths"]

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
            hyp_flats.append(hf); shp_flats.append(sf)

        # Keep stacks on CPU; mini-batches moved to GPU per step.
        hyp = torch.stack(hyp_flats)
        shp = torch.stack(shp_flats)

        anchor_hyp, _, _ = flatten_state_dict(torch.load(anchor_hyp_path, map_location="cpu"), self.hyp_names)
        anchor_shp, _, _ = flatten_state_dict(torch.load(anchor_shp_path, map_location="cpu"), self.shp_names)
        # Anchors stay on CPU; reconstruct_shape_weights moves them on demand.
        self.anchor_hyp = anchor_hyp
        self.anchor_shp = anchor_shp

        hyp_res = hyp - self.anchor_hyp[None]
        shp_res = shp - self.anchor_shp[None]

        self.shp_mean = shp_res.mean(dim=0, keepdim=True)
        self.shp_std  = shp_res.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.shp_norm = (shp_res - self.shp_mean) / self.shp_std

        self.hyp_mean = hyp_res.mean(dim=0, keepdim=True)
        self.hyp_std  = hyp_res.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.hyp_norm = (hyp_res - self.hyp_mean) / self.hyp_std

        print(f"[data] N={len(shp)}  shape_dim={shp.shape[1]:,}  cond_dim={hyp.shape[1]:,}")

    def reconstruct_shape_weights(self, x_norm):
        dev = x_norm.device
        return x_norm * self.shp_std.to(dev) + self.shp_mean.to(dev) + self.anchor_shp[None].to(dev)


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

def train_main(args):
    device = torch.device(args.device)
    ds = ResidualPairedWeightsDataset(
        args.manifest, args.anchor_hyp, args.anchor_shp, device=device,
    )

    model = HypernetToShapeMapper(
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
    hyp_all = ds.hyp_norm  # CPU
    shp_all = ds.shp_norm  # CPU
    N = hyp_all.shape[0]
    bs = min(getattr(args, "batch_size", 16), N)
    print(f"[train] mini-batch on N={N}  bs={bs}  for {args.steps} steps "
          f"(~{args.steps * bs / N:.0f} passes/shape, direct MSE)")
    import random as _random
    _rng = _random.Random(0)
    for step in range(1, args.steps + 1):
        model.train()
        idx = _rng.sample(range(N), bs)
        hyp_b = hyp_all[idx].to(device, non_blocking=True)
        shp_b = shp_all[idx].to(device, non_blocking=True)
        pred = model(hyp_b)
        loss = F.mse_loss(pred, shp_b)
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
        "shp_std":  ds.shp_std.cpu(),
        "hyp_mean": ds.hyp_mean.cpu(),
        "hyp_std":  ds.hyp_std.cpu(),
        "anchor_shp": ds.anchor_shp.cpu(),
        "anchor_hyp": ds.anchor_hyp.cpu(),
        "shp_names": ds.shp_names,
        "shp_shapes": ds.shp_shapes,
        "hyp_names": ds.hyp_names,
        "hyp_shapes": ds.hyp_shapes,
    }
    torch.save(ckpt, out_dir / "mapper.pt")
    print(f"[save] -> {out_dir/'mapper.pt'}")

    # Per-shape eval
    print("[eval] per-shape MSE on standardized residuals")
    model.eval()
    hyp_all_gpu = hyp_all.to(device)
    shp_all_gpu = shp_all.to(device)
    with torch.no_grad():
        pred = model(hyp_all_gpu)
        per_shape_mse = ((pred - shp_all_gpu) ** 2).mean(dim=1).cpu().numpy()
    for i, m in enumerate(per_shape_mse):
        print(f"       shape {i:2d}: {m:.4e}")
    print(f"[eval] mean={per_shape_mse.mean():.4e}  max={per_shape_mse.max():.4e}")

    # Ablation: what happens with scrambled cond?
    with torch.no_grad():
        perm = torch.randperm(hyp_all_gpu.shape[0], device=device)
        pred_scrambled = model(hyp_all[perm])
        scram_mse = ((pred_scrambled - shp_all) ** 2).mean(dim=1).cpu().numpy()
        pred_zero = model(torch.zeros_like(hyp_all))
        zero_mse = ((pred_zero - shp_all) ** 2).mean(dim=1).cpu().numpy()
    print(f"[ablation] correct   mean MSE: {per_shape_mse.mean():.4e}")
    print(f"[ablation] scrambled mean MSE: {scram_mse.mean():.4e}")
    print(f"[ablation] zero_cond mean MSE: {zero_mse.mean():.4e}")

    # Reconstruct absolute weights
    with torch.no_grad():
        pred = model(hyp_all)
    x_abs = ds.reconstruct_shape_weights(pred).cpu()
    torch.save(x_abs, out_dir / "predicted_shape_weights.pt")
    print(f"[save] predicted absolute weights -> {out_dir/'predicted_shape_weights.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--anchor_hyp", default="/workspace/hypernet/data/checkpoints/anchor_hypernet.pt")
    p.add_argument("--anchor_shp", default="/workspace/hypernet/checkpoints/anchor_shape_siren.pt")
    p.add_argument("--out", default="./mapper_out")

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
    p.add_argument("--batch_size", type=int, default=16, help="N=100 mini-batch size")
    p.add_argument("--steps", type=int, default=10000)

    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train_main(parse_args())
