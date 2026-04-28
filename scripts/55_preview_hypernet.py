"""Inference a hypernetwork: predict 24 SIREN weights, reconstruct images, compare with GT.

Usage:
    python 55_preview_hypernet.py              # defaults to object 0
    python 55_preview_hypernet.py --obj 3      # object 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN, unflatten_weights
from src.hypernet import HyperNet
from src.render import scan_rig_directions


def reconstruct_image(siren: SIREN, res: int, device: str) -> np.ndarray:
    ys = np.linspace(-1, 1, res, dtype=np.float32)
    xs = np.linspace(-1, 1, res, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    coords = torch.from_numpy(np.stack([gx, gy], axis=-1).reshape(-1, 2)).to(device)
    with torch.no_grad():
        pred = siren(coords)
    img = pred.cpu().numpy().reshape(res, res, 3)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=int, default=0)
    args = parser.parse_args()

    dev = CFG.device
    c = CFG.img_siren
    d = CFG.data
    hc = CFG.hypernet
    obj_i = args.obj

    # load hypernetwork
    hypernet_path = hc.out_dir / f"obj_{obj_i:02d}.pt"
    assert hypernet_path.exists(), f"not found: {hypernet_path}"

    ref_siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden)
    hypernet = HyperNet(
        target_siren=ref_siren,
        in_dim=3,
        head_hidden=hc.head_hidden,
        head_layers=hc.head_layers,
        final_init_scale=hc.final_init_scale,
    ).to(dev)
    hypernet.load_state_dict(torch.load(hypernet_path, map_location=dev, weights_only=True))
    hypernet.eval()

    cam_dirs = torch.from_numpy(scan_rig_directions()).to(dev)

    out_dir = Path(d.views_dir).parent / "hypernet_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for view_j in range(d.num_views):
        # predict SIREN weights from camera direction
        with torch.no_grad():
            pred_weights = hypernet(cam_dirs[view_j])

        # load predicted weights into a SIREN
        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(dev)
        unflatten_weights(siren, pred_weights)
        siren.eval()

        # reconstruct image
        recon = reconstruct_image(siren, d.img_res, dev)

        # load GT
        gt_path = d.views_dir / f"obj_{obj_i:02d}" / f"view_{view_j:02d}.png"
        gt = np.array(Image.open(gt_path).convert("RGB"))

        # side by side: GT | Hypernet reconstruction
        pair = np.concatenate([gt, recon], axis=1)
        rows.append(pair)

    # arrange in a grid: 6 rows x 4 columns (24 views)
    grid_rows = []
    for r in range(6):
        grid_rows.append(np.concatenate(rows[r * 4:(r + 1) * 4], axis=1))
    grid = np.concatenate(grid_rows, axis=0)

    save_path = out_dir / f"obj_{obj_i:02d}_hypernet.png"
    Image.fromarray(grid).save(save_path)
    print(f"[preview] saved {save_path}")
    print(f"  grid: 6 rows x 4 cols, each cell is GT|Recon at {d.img_res}x{d.img_res}")


if __name__ == "__main__":
    main()
