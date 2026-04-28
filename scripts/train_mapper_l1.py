"""
Mapper training with L1 loss (instead of MSE).

Motivation: MSE squares errors, so it over-focuses on the biggest errors and
tolerates the many small ones that accumulate into surface holes. L1 penalizes
all errors linearly, pushing the long tail of small-but-nonzero weight errors
down. Expected effect: cleaner SDF, fewer holes, possibly slightly worse MSE
by its own metric.

Uses the HypernetToShapeMapper and dataset from the deterministic mapper file;
we import them rather than copy-paste.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Import the mapper + dataset from the sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hypernet_to_shape_mapper import (
    HypernetToShapeMapper,
    ResidualPairedWeightsDataset,
)


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

    loss_fn = F.l1_loss if args.loss == "l1" else F.mse_loss
    log_every = max(1, args.steps // 200)
    hyp_all = ds.hyp_norm
    shp_all = ds.shp_norm

    print(f"[train] loss={args.loss}  full-batch on N={len(ds.shp_norm)} for {args.steps} steps")
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
            # Report both L1 and MSE at log points for comparison
            with torch.no_grad():
                mse_val = F.mse_loss(pred, shp_all).item()
                l1_val = F.l1_loss(pred, shp_all).item()
            print(f"step {step:6d} / {args.steps}   "
                  f"l1={l1_val:.4e}  mse={mse_val:.4e}  lr={sched.get_last_lr()[0]:.2e}")

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

    print("[eval] per-shape L1 and MSE on standardized residuals")
    model.eval()
    with torch.no_grad():
        pred = model(hyp_all)
        per_shape_l1  = (pred - shp_all).abs().mean(dim=1).cpu().numpy()
        per_shape_mse = ((pred - shp_all) ** 2).mean(dim=1).cpu().numpy()
    for i in range(len(per_shape_l1)):
        print(f"       shape {i:2d}: l1={per_shape_l1[i]:.4e}  mse={per_shape_mse[i]:.4e}")
    print(f"[eval] mean L1={per_shape_l1.mean():.4e}  mean MSE={per_shape_mse.mean():.4e}")

    with torch.no_grad():
        perm = torch.randperm(hyp_all.shape[0], device=device)
        scram = F.mse_loss(model(hyp_all[perm]), shp_all).item()
        zero = F.mse_loss(model(torch.zeros_like(hyp_all)), shp_all).item()
    print(f"[ablation] correct   MSE: {per_shape_mse.mean():.4e}")
    print(f"[ablation] scrambled MSE: {scram:.4e}")
    print(f"[ablation] zero_cond MSE: {zero:.4e}")

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
    p.add_argument("--out", default="./mapper_l1_out")

    p.add_argument("--loss", default="l1", choices=["l1", "mse"])

    p.add_argument("--chunk_shape", type=int, default=1024)
    p.add_argument("--chunk_cond",  type=int, default=8192)

    p.add_argument("--d_model",  type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads",  type=int, default=8)
    p.add_argument("--ff_mult",  type=int, default=4)
    p.add_argument("--cond_enc_layers", type=int, default=2)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=30000)

    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train_main(parse_args())
