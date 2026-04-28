"""
Rectified Flow Transformer — hypernet-conditioned generation of shape-SIREN weights.

Pipeline:
    hypernet weights (~17.9M)  --chunk(8192)-->  cond tokens (~2200, d_model)
                                                              |
                                                              v
    noise (264K)  --chunk(1024)-->  query tokens (258, d_model)  [+ time embed]
                                                              |
                                         cross-attention on cond tokens
                                                              |
                                                              v
                                     velocity tokens --unchunk-->  264K velocity

Training: rectified-flow objective on (noise -> clean shape-SIREN weights),
conditioned on the paired hypernet weights.

Why these choices (decided in the preceding turn):
    * chunk_shape = 1024, chunk_cond = 8192
    * conditioning via shared-projection chunking + cross-attention
    * tiny transformer (d=256, 4 layers, 4 heads) for plumbing on 10 shapes

File is self-contained given:
    * shape-SIREN weights saved as .pt files with known param_names / shapes
    * hypernet weights saved likewise
We load them via torch.load and flatten with the same flatten_weights helper
used elsewhere in the project.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------
# Chunked tokenizer / detokenizer
# ----------------------------------------------------------------------------

class ChunkedProjector(nn.Module):
    """Pad a flat weight vector to a multiple of chunk_size, reshape into
    (num_chunks, chunk_size), and linearly project each chunk to d_model.

    Stores positional embeddings so each chunk index is distinguishable
    (HyperDiffusion does this -- without pos embeds, chunks are interchangeable
    and the transformer cannot recover layer structure).
    """

    def __init__(self, total_dim: int, chunk_size: int, d_model: int):
        super().__init__()
        self.total_dim = total_dim
        self.chunk_size = chunk_size
        self.num_chunks = math.ceil(total_dim / chunk_size)
        self.padded_dim = self.num_chunks * chunk_size
        self.pad = self.padded_dim - total_dim

        self.proj_in = nn.Linear(chunk_size, d_model)
        self.proj_out = nn.Linear(d_model, chunk_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_chunks, d_model) * 0.02)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, total_dim) -> tokens: (B, num_chunks, d_model)."""
        B = x.shape[0]
        if self.pad:
            x = F.pad(x, (0, self.pad))
        chunks = x.view(B, self.num_chunks, self.chunk_size)
        tokens = self.proj_in(chunks) + self.pos_embed
        return tokens

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, num_chunks, d_model) -> x: (B, total_dim)."""
        B = tokens.shape[0]
        chunks = self.proj_out(tokens)          # (B, num_chunks, chunk_size)
        x = chunks.view(B, self.padded_dim)
        if self.pad:
            x = x[:, : self.total_dim]
        return x


# ----------------------------------------------------------------------------
# Time embedding (sinusoidal + MLP, standard diffusion/flow trick)
# ----------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0,1]
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.mlp(emb)  # (B, d_model)


# ----------------------------------------------------------------------------
# Transformer block with self-attn + cross-attn + FFN
# ----------------------------------------------------------------------------

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm2_q = nn.LayerNorm(d_model)
        self.norm2_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]

        q = self.norm2_q(x)
        kv = self.norm2_kv(cond)
        x = x + self.cross_attn(q, kv, kv, need_weights=False)[0]

        x = x + self.ffn(self.norm3(x))
        return x


# ----------------------------------------------------------------------------
# Main model
# ----------------------------------------------------------------------------

class RectifiedFlowTransformer(nn.Module):
    def __init__(
        self,
        shape_dim: int,
        cond_dim: int,
        chunk_shape: int = 1024,
        chunk_cond: int = 8192,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.shape_proj = ChunkedProjector(shape_dim, chunk_shape, d_model)
        self.cond_proj = ChunkedProjector(cond_dim, chunk_cond, d_model)
        self.time_embed = TimeEmbedding(d_model)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        print(
            f"[model] shape tokens={self.shape_proj.num_chunks}  "
            f"cond tokens={self.cond_proj.num_chunks}  d_model={d_model}  "
            f"layers={n_layers}  heads={n_heads}"
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[model] trainable params: {n_params:,}")

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_w: torch.Tensor) -> torch.Tensor:
        """Predict velocity (dx/dt) at noise state x_t, time t, hypernet cond_w.
        x_t:    (B, shape_dim)
        t:      (B,) in [0,1]
        cond_w: (B, cond_dim)
        returns velocity of shape (B, shape_dim).
        """
        x_tok = self.shape_proj.tokenize(x_t)               # (B, Nq, d)
        c_tok = self.cond_proj.tokenize(cond_w)             # (B, Nk, d)

        t_emb = self.time_embed(t).unsqueeze(1)              # (B, 1, d)
        x_tok = x_tok + t_emb                                # broadcast over tokens

        for blk in self.blocks:
            x_tok = blk(x_tok, c_tok)

        x_tok = self.final_norm(x_tok)
        v = self.shape_proj.detokenize(x_tok)                # (B, shape_dim)
        return v


# ----------------------------------------------------------------------------
# Rectified flow objective
# ----------------------------------------------------------------------------

def rectified_flow_loss(
    model: RectifiedFlowTransformer,
    x1: torch.Tensor,
    cond_w: torch.Tensor,
) -> torch.Tensor:
    """Straight-line flow from x0~N(0,I) to data x1.

    x_t = (1-t) x0 + t x1
    target velocity = x1 - x0
    loss = || v_theta(x_t, t, c) - (x1 - x0) ||^2
    """
    B = x1.shape[0]
    device = x1.device

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)
    x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    v_target = x1 - x0

    v_pred = model(x_t, t, cond_w)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample(
    model: RectifiedFlowTransformer,
    cond_w: torch.Tensor,
    n_steps: int = 50,
    shape_dim: int | None = None,
) -> torch.Tensor:
    """Euler sampling from x0~N(0,I) to x1."""
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
# Dataset
# ----------------------------------------------------------------------------

def flatten_state_dict(sd: dict, keys: list[str] | None = None) -> tuple[torch.Tensor, list[str], list[tuple]]:
    """Flatten a state_dict in a deterministic order. Returns (flat, names, shapes)."""
    if keys is None:
        keys = list(sd.keys())
    tensors = [sd[k].detach().float().flatten() for k in keys]
    flat = torch.cat(tensors)
    shapes = [tuple(sd[k].shape) for k in keys]
    return flat, keys, shapes


class PairedWeightsDataset(Dataset):
    """Loads paired (hypernet_weights, shape_siren_weights) from a manifest.

    Manifest file layout: a .pt file containing a dict:
        {
            'hypernet_paths': [Path, Path, ...],   # N entries
            'shape_paths':    [Path, Path, ...],   # N entries, same order
        }
    Each referenced file is a torch-saved state_dict.

    On first access we load all weights into memory (N=10, easily fits).
    """

    def __init__(self, manifest_path: str | Path, device: torch.device):
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        hyp_paths = manifest["hypernet_paths"]
        shp_paths = manifest["shape_paths"]
        assert len(hyp_paths) == len(shp_paths), "mismatched pair counts"

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

        self.hyp = torch.stack(hyp_flats).to(device)   # (N, cond_dim)
        self.shp = torch.stack(shp_flats).to(device)   # (N, shape_dim)

        # Per-parameter standardization on the shape targets. Rectified flow
        # assumes unit-variance endpoints; without it the first-layer SIREN
        # weights dominate the loss and later biases get ignored. We also
        # standardize the cond so cross-attention keys are on a sane scale.
        self.shp_mean = self.shp.mean(dim=0, keepdim=True)
        self.shp_std = self.shp.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.shp_norm = (self.shp - self.shp_mean) / self.shp_std

        self.hyp_mean = self.hyp.mean(dim=0, keepdim=True)
        self.hyp_std = self.hyp.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.hyp_norm = (self.hyp - self.hyp_mean) / self.hyp_std

        print(f"[data] N={len(self.shp)}  shape_dim={self.shp.shape[1]:,}  cond_dim={self.hyp.shape[1]:,}")

    def __len__(self):
        return self.shp.shape[0]

    def __getitem__(self, i):
        return self.hyp_norm[i], self.shp_norm[i]

    def denormalize_shape(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.shp_std + self.shp_mean


# ----------------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------------

def train_main(args):
    device = torch.device(args.device)
    ds = PairedWeightsDataset(args.manifest, device=device)

    # Full-batch training (N=10). Rectified flow on tiny N benefits from this —
    # no gradient noise from mini-batching, and we want to confirm the model
    # can memorize before scaling up.
    model = RectifiedFlowTransformer(
        shape_dim=ds.shp.shape[1],
        cond_dim=ds.hyp.shape[1],
        chunk_shape=args.chunk_shape,
        chunk_cond=args.chunk_cond,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=0.0,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_every = max(1, args.steps // 200)
    hyp_all = ds.hyp_norm
    shp_all = ds.shp_norm

    print(f"[train] full-batch on N={len(ds)} for {args.steps} steps")
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

    # Save
    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "shp_mean": ds.shp_mean.cpu(),
        "shp_std": ds.shp_std.cpu(),
        "hyp_mean": ds.hyp_mean.cpu(),
        "hyp_std": ds.hyp_std.cpu(),
        "shp_names": ds.shp_names,
        "shp_shapes": ds.shp_shapes,
        "hyp_names": ds.hyp_names,
        "hyp_shapes": ds.hyp_shapes,
    }
    torch.save(ckpt, out_dir / "rectified_flow.pt")
    print(f"[save] -> {out_dir/'rectified_flow.pt'}")

    # Sanity: sample each training cond and check MSE against the known target
    print("[eval] sampling training conditions for memorization check")
    with torch.no_grad():
        x_pred_norm = sample(model, hyp_all, n_steps=args.n_sample_steps)
        per_shape_mse = ((x_pred_norm - shp_all) ** 2).mean(dim=1).cpu().numpy()
    print(f"[eval] per-shape sampled MSE (normalized space):")
    for i, m in enumerate(per_shape_mse):
        print(f"       shape {i:2d}: {m:.4e}")
    print(f"[eval] mean={per_shape_mse.mean():.4e}  max={per_shape_mse.max():.4e}")

    # Save denormalized predictions for downstream marching-cubes
    x_pred = ds.denormalize_shape(x_pred_norm).cpu()
    torch.save(x_pred, out_dir / "predicted_shape_weights.pt")
    print(f"[save] predicted weights -> {out_dir/'predicted_shape_weights.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True,
                   help="Path to manifest .pt with hypernet_paths and shape_paths lists")
    p.add_argument("--out", type=str, default="./rf_out")

    # Tokenization
    p.add_argument("--chunk_shape", type=int, default=1024)
    p.add_argument("--chunk_cond",  type=int, default=8192)

    # Model
    p.add_argument("--d_model",  type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads",  type=int, default=4)
    p.add_argument("--ff_mult",  type=int, default=4)

    # Optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20000)

    # Sampling
    p.add_argument("--n_sample_steps", type=int, default=50)

    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train_main(parse_args())
