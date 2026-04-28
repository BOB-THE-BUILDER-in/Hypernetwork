"""
Eval-only for the N=100 mapper.

Loads:
    /workspace/hypernet/scripts/t1_n100/mapper.pt
    /workspace/hypernet/scripts/manifest_n100.pt
Outputs:
    per-shape MSE on standardized residuals
    ablation MSEs (scrambled cond, zero cond)
    /workspace/hypernet/scripts/t1_n100/predicted_shape_weights.pt (absolute weights)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPTS = Path("/workspace/hypernet/scripts")
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from hypernet_to_shape_mapper import (
    HypernetToShapeMapper,
    ResidualPairedWeightsDataset,
)


def main():
    device = torch.device("cuda")
    out_dir = SCRIPTS / "t1_n100"
    ckpt = torch.load(out_dir / "mapper.pt", map_location=device, weights_only=False)
    args = ckpt["args"]

    print(f"[load] mapper.pt  args.steps={args['steps']}  args.batch_size={args.get('batch_size','?')}")

    # Rebuild dataset (CPU storage)
    ds = ResidualPairedWeightsDataset(
        args["manifest"], args["anchor_hyp"], args["anchor_shp"], device=device,
    )

    # Move data to GPU for eval (fits comfortably for forward pass alone)
    hyp_gpu = ds.hyp_norm.to(device)
    shp_gpu = ds.shp_norm.to(device)

    # Build model and load weights
    model = HypernetToShapeMapper(
        shape_dim=ds.shp_norm.shape[1],
        cond_dim=ds.hyp_norm.shape[1],
        chunk_shape=args["chunk_shape"], chunk_cond=args["chunk_cond"],
        d_model=args["d_model"], n_layers=args["n_layers"], n_heads=args["n_heads"],
        ff_mult=args["ff_mult"], cond_enc_layers=args["cond_enc_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Per-shape MSE - chunked
    print("\n[eval] per-shape MSE on standardized residuals")
    per_shape_mse_list = []
    with torch.no_grad():
        for i in range(hyp_gpu.shape[0]):
            pred_i = model(hyp_gpu[i:i+1])
            per_shape_mse_list.append(((pred_i - shp_gpu[i:i+1]) ** 2).mean().item())
    import numpy as np
    per_shape_mse = np.array(per_shape_mse_list)
    for i, m in enumerate(per_shape_mse):
        print(f"       shape {i:2d}: {m:.4e}")
    print(f"[eval] mean={per_shape_mse.mean():.4e}  "
          f"median={float(np.median(per_shape_mse)):.4e}  "
          f"max={per_shape_mse.max():.4e}  min={per_shape_mse.min():.4e}")

    # Ablations - chunked to fit in GPU
    print("\n[ablation]")
    with torch.no_grad():
        perm = torch.randperm(hyp_gpu.shape[0], device=device)

        scram_mse_list = []
        zero_mse_list = []
        for i in range(hyp_gpu.shape[0]):
            # Scrambled: feed cond from a different shape
            scram_pred = model(hyp_gpu[perm[i]:perm[i]+1])
            scram_mse_list.append(((scram_pred - shp_gpu[i:i+1]) ** 2).mean().item())

            # Zero cond
            zero_pred = model(torch.zeros_like(hyp_gpu[i:i+1]))
            zero_mse_list.append(((zero_pred - shp_gpu[i:i+1]) ** 2).mean().item())

        import numpy as np
        scram_mse = np.array(scram_mse_list)
        zero_mse  = np.array(zero_mse_list)
    print(f"  correct   mean MSE: {per_shape_mse.mean():.4e}")
    print(f"  scrambled mean MSE: {scram_mse.mean():.4e}   "
          f"(should be much higher than correct)")
    print(f"  zero_cond mean MSE: {zero_mse.mean():.4e}   "
          f"(should be much higher than correct)")
    print(f"  signal ratio (scrambled / correct) = {scram_mse.mean() / per_shape_mse.mean():.2f}x")
    print(f"  signal ratio (zero / correct)      = {zero_mse.mean() / per_shape_mse.mean():.2f}x")

    # Reconstruct absolute weights - chunked, push to CPU per shape
    print("\n[reconstruct] saving predicted absolute weights")
    abs_chunks = []
    with torch.no_grad():
        for i in range(hyp_gpu.shape[0]):
            pred_i = model(hyp_gpu[i:i+1])
            abs_i = ds.reconstruct_shape_weights(pred_i).cpu()
            abs_chunks.append(abs_i)
    x_abs = torch.cat(abs_chunks, dim=0)
    torch.save(x_abs, out_dir / "predicted_shape_weights.pt")
    print(f"[save] -> {out_dir/'predicted_shape_weights.pt'}  shape={tuple(x_abs.shape)}")

    print("\n[done]")


if __name__ == "__main__":
    main()