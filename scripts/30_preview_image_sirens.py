"""Reconstruct images from trained image-SIRENs and compare with ground truth.

Produces side-by-side comparisons: GT | Reconstructed for a sample of views.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN


def reconstruct(siren: SIREN, res: int, device: str) -> np.ndarray:
    """Query the SIREN on a full grid and return (H, W, 3) uint8 image."""
    ys = np.linspace(-1, 1, res, dtype=np.float32)
    xs = np.linspace(-1, 1, res, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    coords = torch.from_numpy(np.stack([gx, gy], axis=-1).reshape(-1, 2)).to(device)

    with torch.no_grad():
        pred = siren(coords)  # (H*W, 3)
    img = pred.cpu().numpy().reshape(res, res, 3)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def main():
    dev = CFG.device
    c = CFG.img_siren
    d = CFG.data

    out_dir = Path(d.views_dir).parent / "siren_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    # sample: first 3 objects, views 0, 8, 16 (one per ring)
    sample_objects = list(range(min(3, d.num_objects)))
    sample_views = [0, 8, 16]

    for obj_i in sample_objects:
        pairs = []
        for view_j in sample_views:
            # load GT
            gt_path = d.views_dir / f"obj_{obj_i:02d}" / f"view_{view_j:02d}.png"
            gt = np.array(Image.open(gt_path).convert("RGB"))

            # load trained SIREN
            siren_path = c.out_dir / f"obj_{obj_i:02d}" / f"view_{view_j:02d}.pt"
            siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                          c.w0_first, c.w0_hidden).to(dev)
            siren.load_state_dict(torch.load(siren_path, map_location=dev, weights_only=True))
            siren.eval()

            recon = reconstruct(siren, d.img_res, dev)

            # side by side
            pair = np.concatenate([gt, recon], axis=1)
            pairs.append(pair)

        # stack vertically: 3 views per object
        grid = np.concatenate(pairs, axis=0)
        save_path = out_dir / f"obj_{obj_i:02d}_compare.png"
        Image.fromarray(grid).save(save_path)
        print(f"[preview] saved {save_path}")

    print(f"\n[preview] done — check {out_dir}")


if __name__ == "__main__":
    main()
