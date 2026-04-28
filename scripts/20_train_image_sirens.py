"""Train one image-SIREN per (object, view), warm-started from the shared anchor.

240 SIRENs total (10 objects x 24 views). Each is a fast fine-tune from the
anchor — the anchor did the heavy lifting.

Output: data/image_sirens/obj_XX/view_YY.pt  (state_dict for each SIREN)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN


def load_image_as_target(img_path: Path, device: str) -> torch.Tensor:
    """Load image -> (H*W, 3) float tensor in [0,1], plus (H*W, 2) coords in [-1,1]."""
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    H, W, _ = img.shape

    # pixel coords in [-1, 1]
    ys = np.linspace(-1, 1, H, dtype=np.float32)
    xs = np.linspace(-1, 1, W, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

    pixels = img.reshape(-1, 3)
    return (
        torch.from_numpy(coords).to(device),
        torch.from_numpy(pixels).to(device),
    )


def train_one(
    siren: SIREN,
    coords: torch.Tensor,
    pixels: torch.Tensor,
    steps: int,
    lr: float,
    batch_pixels: int,
    device: str,
) -> float:
    """Fine-tune a single SIREN on one image. Returns final loss."""
    opt = torch.optim.Adam(siren.parameters(), lr=lr)
    N = coords.shape[0]
    final_loss = 0.0

    for step in range(steps):
        idx = torch.randint(0, N, (batch_pixels,), device=device)
        pred = siren(coords[idx])
        loss = F.mse_loss(pred, pixels[idx])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = loss.item()

    return final_loss


def main():
    dev = CFG.device
    c = CFG.img_siren
    d = CFG.data

    # load anchor
    assert c.anchor_path.exists(), (
        f"anchor not found at {c.anchor_path} — run 10_train_anchor_image_siren.py first"
    )
    ckpt = torch.load(c.anchor_path, map_location=dev, weights_only=True)
    anchor_sd = ckpt["state_dict"]
    print(f"[img-siren] loaded anchor from {c.anchor_path}")

    c.out_dir.mkdir(parents=True, exist_ok=True)
    total = d.num_objects * d.num_views
    done = 0

    for obj_i in range(d.num_objects):
        obj_dir = c.out_dir / f"obj_{obj_i:02d}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        views_dir = d.views_dir / f"obj_{obj_i:02d}"

        for view_j in range(d.num_views):
            out_path = obj_dir / f"view_{view_j:02d}.pt"
            if out_path.exists():
                done += 1
                print(f"[img-siren] skip obj_{obj_i:02d}/view_{view_j:02d} ({done}/{total})")
                continue

            img_path = views_dir / f"view_{view_j:02d}.png"
            assert img_path.exists(), f"missing {img_path}"

            # fresh SIREN from anchor
            siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                          c.w0_first, c.w0_hidden).to(dev)
            siren.load_state_dict(anchor_sd)

            coords, pixels = load_image_as_target(img_path, dev)
            final_loss = train_one(
                siren, coords, pixels,
                steps=c.steps_warm, lr=c.lr,
                batch_pixels=c.batch_pixels, device=dev,
            )

            torch.save(siren.state_dict(), out_path)
            done += 1
            print(f"[img-siren] obj_{obj_i:02d}/view_{view_j:02d}  "
                  f"loss={final_loss:.6f}  ({done}/{total})")

    print(f"\n[img-siren] done — {done} SIRENs saved to {c.out_dir}")


if __name__ == "__main__":
    main()
