"""
Per-layer error diagnostic for the hypernet->shape mapper.

Test the hierarchical hypothesis: does the mapper predict some SIREN layers
more accurately than others? If yes, the hierarchical/coarse-to-fine idea is
worth pursuing. If error is uniform across layers, reshaping the prediction
won't help.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/hypernet")
sys.path.insert(0, "/workspace/hypernet/scripts")

from hypernet_to_shape_mapper import HypernetToShapeMapper


def _unwrap(o):
    if isinstance(o, dict) and "state_dict" in o and isinstance(o["state_dict"], dict):
        return o["state_dict"]
    return o


def flat(path, keys):
    sd = _unwrap(torch.load(path, map_location="cpu"))
    return torch.cat([sd[k].detach().float().flatten() for k in keys])


def main():
    device = torch.device("cuda")
    mapper_ckpt = "/workspace/hypernet/scripts/t1/mapper.pt"
    manifest = "/workspace/hypernet/scripts/manifest.pt"

    ckpt = torch.load(mapper_ckpt, map_location=device, weights_only=False)
    m = torch.load(manifest, map_location="cpu", weights_only=False)

    # Build mapper
    a = ckpt["args"]
    model = HypernetToShapeMapper(
        shape_dim=ckpt["shp_mean"].shape[1],
        cond_dim=ckpt["hyp_mean"].shape[1],
        chunk_shape=a["chunk_shape"], chunk_cond=a["chunk_cond"],
        d_model=a["d_model"], n_layers=a["n_layers"], n_heads=a["n_heads"],
        ff_mult=a["ff_mult"], cond_enc_layers=a["cond_enc_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load all 10 hypernets and all 10 target shape-SIRENs in raw weight space
    hyp_keys = ckpt["hyp_names"]
    shp_keys = ckpt["shp_names"]
    shp_shapes = ckpt["shp_shapes"]

    # Report: the keys in a standard SIREN state_dict look like
    # net.0.weight, net.0.bias, net.2.weight, net.2.bias, ...
    # where even indices are Linear and odd are Sine. Print them so user sees.
    print(f"[info] Shape-SIREN has {len(shp_keys)} param tensors:")
    for k, s in zip(shp_keys, shp_shapes):
        nel = 1
        for d in s:
            nel *= d
        print(f"  {k:30s}  shape={s}  numel={nel}")

    # Load every hypernet and every target, absolute weights
    hyps_abs = torch.stack([flat(p, hyp_keys) for p in m["hypernet_paths"]]).to(device)
    shps_abs = torch.stack([flat(p, shp_keys) for p in m["shape_paths"]]).to(device)

    # Prep mapper inputs (residual + standardized) and decode back
    anchor_hyp = ckpt["anchor_hyp"].to(device)
    anchor_shp = ckpt["anchor_shp"].to(device)
    hyp_mean = ckpt["hyp_mean"].to(device); hyp_std = ckpt["hyp_std"].to(device)
    shp_mean = ckpt["shp_mean"].to(device); shp_std = ckpt["shp_std"].to(device)

    z = (hyps_abs - anchor_hyp[None] - hyp_mean) / hyp_std

    with torch.no_grad():
        pred_norm = model(z)                                      # (10, shape_dim)
        pred_abs = pred_norm * shp_std + shp_mean + anchor_shp[None]   # absolute weights

    # Compute error PER PARAMETER TENSOR (per SIREN layer's W and b)
    err = (pred_abs - shps_abs) ** 2                              # (10, shape_dim)
    mean_err_per_dim = err.mean(dim=0)                            # (shape_dim,) avg over 10 shapes

    # Also compute the target variance per dim for a normalized comparison
    target_var_per_dim = shps_abs.var(dim=0)                      # (shape_dim,)

    # Break up by SIREN key
    print("\n" + "=" * 80)
    print("Per-layer error (averaged over 10 training shapes)")
    print("=" * 80)
    print(f"{'layer':30s}  {'params':>8s}  {'abs_mse':>12s}  {'tgt_var':>12s}  {'err/var':>8s}")

    offset = 0
    layer_stats = []
    for k, s in zip(shp_keys, shp_shapes):
        nel = 1
        for d in s:
            nel *= d
        layer_mse = mean_err_per_dim[offset:offset + nel].mean().item()
        layer_tgt_var = target_var_per_dim[offset:offset + nel].mean().item()
        ratio = layer_mse / max(layer_tgt_var, 1e-12)
        layer_stats.append((k, nel, layer_mse, layer_tgt_var, ratio))
        print(f"{k:30s}  {nel:>8d}  {layer_mse:>12.3e}  {layer_tgt_var:>12.3e}  {ratio:>8.3f}")
        offset += nel

    # Aggregate into "logical layers" -- most SIREN implementations name them
    # layer_0.linear.weight, etc. Group by any leading stem that repeats.
    print("\n" + "=" * 80)
    print("Aggregated by assumed layer index (weight + bias together)")
    print("=" * 80)
    import re
    by_layer = defaultdict(lambda: [0, 0.0, 0.0])   # [params, sum_err, sum_var]
    for k, nel, mse, tgt_var, _ in layer_stats:
        # try to pull out a layer index -- match first integer in the key
        match = re.search(r"(\d+)", k)
        idx = int(match.group(1)) if match else -1
        by_layer[idx][0] += nel
        by_layer[idx][1] += mse * nel
        by_layer[idx][2] += tgt_var * nel

    print(f"{'layer_idx':>10s}  {'params':>8s}  {'abs_mse':>12s}  {'tgt_var':>12s}  {'err/var':>8s}")
    for idx in sorted(by_layer):
        params, sum_err, sum_var = by_layer[idx]
        avg_mse = sum_err / params
        avg_var = sum_var / params
        ratio = avg_mse / max(avg_var, 1e-12)
        print(f"{idx:>10d}  {params:>8d}  {avg_mse:>12.3e}  {avg_var:>12.3e}  {ratio:>8.3f}")

    # Final verdict
    print("\n" + "=" * 80)
    print("Interpretation")
    print("=" * 80)
    ratios = [r for *_, r in layer_stats]
    max_r = max(ratios)
    min_r = min(ratios)
    print(f"err/var ratio range: min={min_r:.3f}  max={max_r:.3f}  max/min={max_r/max(min_r,1e-9):.2f}x")
    if max_r / max(min_r, 1e-9) < 2.0:
        print("-> err/var is nearly uniform across layers. Hierarchical prediction")
        print("   will not help -- error is evenly distributed.")
    elif max_r / max(min_r, 1e-9) < 5.0:
        print("-> Some layer-level spread. Hierarchical prediction might help a bit")
        print("   but not a clear win.")
    else:
        print("-> STRONG layer-level spread. Some layers predict much worse than")
        print("   others. Hierarchical prediction (extra mapper capacity on the")
        print("   bad layers) has real upside.")


if __name__ == "__main__":
    main()
