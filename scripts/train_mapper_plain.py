"""
Mapper variant: no AdaLN, no mean-pool conditioning.

Architecture:
    cond_tok (B, Nk, d)  = ChunkedProjector(hypernet_weights)
    queries  (B, Nq, d)  = learnable positional query tokens
    per layer:
        queries += self_attn(queries)
        queries += cross_attn(queries, cond_tok)     <-- raw, not pooled
        queries += ffn(queries)
    shape_residual = ChunkedProjector_out(queries)

If AdaLN mean-pool was washing out per-chunk signal, this should break past
0.14. If it still plateaus at 0.14, the data ceiling hypothesis is confirmed
and we pivot to Route A.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse dataset from the previous mapper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hypernet_to_shape_mapper import ResidualPairedWeightsDataset


# ----------------------------------------------------------------------------
# Chunked projector (same as before)
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
# Plain transformer block: self + cross + FFN, no AdaLN, standard LayerNorm
# ----------------------------------------------------------------------------

class PlainBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm2_q = nn.LayerNorm(d_model)
        self.norm2_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x, cond_tokens):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]

        q = self.norm2_q(x)
        kv = self.norm2_kv(cond_tokens)
        x = x + self.cross_attn(q, kv, kv, need_weights=False)[0]

        x = x + self.ffn(self.norm3(x))
        return x


class PlainHypernetToShapeMapper(nn.Module):
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
    ):
        super().__init__()
        self.shape_proj = ChunkedProjector(shape_dim, chunk_shape, d_model)
        self.cond_proj = ChunkedProjector(cond_dim, chunk_cond, d_model)

        self.shape_queries = nn.Parameter(
            torch.randn(1, self.shape_proj.num_chunks, d_model) * 0.02
        )

        self.blocks = nn.ModuleList(
            [PlainBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Keep zero-init on the output projection so initial prediction = anchor
        nn.init.zeros_(self.shape_proj.proj_out.weight)
        nn.init.zeros_(self.shape_proj.proj_out.bias)

        print(f"[model] shape tokens={self.shape_proj.num_chunks}  "
              f"cond tokens={self.cond_proj.num_chunks}  d_model={d_model}  "
              f"layers={n_layers}  heads={n_heads}  (plain, no AdaLN)")
        print(f"[model] trainable params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, cond_w):
        B = cond_w.shape[0]
        c_tok = self.cond_proj.tokenize(cond_w)                  # (B, Nk, d)
        x = self.shape_queries.expand(B, -1, -1) + self.shape_proj.pos_embed
        for blk in self.blocks:
            x = blk(x, c_tok)
        x = self.final_norm(x)
        return self.shape_proj.detokenize(x)


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

def train_main(args):
    device = torch.device(args.device)
    ds = ResidualPairedWeightsDataset(
        args.manifest, args.anchor_hyp, args.anchor_shp, device=device,
    )

    model = PlainHypernetToShapeMapper(
        shape_dim=ds.shp_norm.shape[1],
        cond_dim=ds.hyp_norm.shape[1],
        chunk_shape=args.chunk_shape,
        chunk_cond=args.chunk_cond,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = F.l1_loss if args.loss == "l1" else F.mse_loss
    log_every = max(1, args.steps // 200)
    hyp_all = ds.hyp_norm
    shp_all = ds.shp_norm

    print(f"[train] loss={args.loss}  full-batch N={len(ds.shp_norm)}  steps={args.steps}")
    for step in range(1, args.steps + 1):
        model.train()
        pred = model(hyp_all)
        loss = loss_fn(pred, shp_all)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                mse_val = F.mse_loss(pred, shp_all).item()
                l1_val = F.l1_loss(pred, shp_all).item()
            print(f"step {step:6d} / {args.steps}   l1={l1_val:.4e}  mse={mse_val:.4e}  lr={sched.get_last_lr()[0]:.2e}")

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

    model.eval()
    with torch.no_grad():
        pred = model(hyp_all)
        per_shape_mse = ((pred - shp_all) ** 2).mean(dim=1).cpu().numpy()
    for i, m in enumerate(per_shape_mse):
        print(f"       shape {i:2d}: mse={m:.4e}")
    print(f"[eval] mean MSE={per_shape_mse.mean():.4e}")

    with torch.no_grad():
        perm = torch.randperm(hyp_all.shape[0], device=device)
        scram = F.mse_loss(model(hyp_all[perm]), shp_all).item()
        zero = F.mse_loss(model(torch.zeros_like(hyp_all)), shp_all).item()
    print(f"[ablation] correct={per_shape_mse.mean():.4e}  scrambled={scram:.4e}  zero={zero:.4e}")

    with torch.no_grad():
        pred = model(hyp_all)
    torch.save(ds.reconstruct_shape_weights(pred).cpu(), out_dir / "predicted_shape_weights.pt")
    print(f"[save] predicted absolute weights -> {out_dir/'predicted_shape_weights.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--anchor_hyp", default="/workspace/hypernet/data/checkpoints/anchor_hypernet.pt")
    p.add_argument("--anchor_shp", default="/workspace/hypernet/checkpoints/anchor_shape_siren.pt")
    p.add_argument("--out", default="./t_plain")

    p.add_argument("--loss", default="mse", choices=["l1", "mse"])

    p.add_argument("--chunk_shape", type=int, default=1024)
    p.add_argument("--chunk_cond",  type=int, default=8192)

    p.add_argument("--d_model",  type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads",  type=int, default=8)
    p.add_argument("--ff_mult",  type=int, default=4)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=3000)

    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train_main(parse_args())
