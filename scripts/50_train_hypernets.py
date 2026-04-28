"""Fine-tune one hypernetwork per object, warm-started from the anchor.

Each hypernetwork learns: camera_direction -> image-SIREN weights
for its object's 24 views.

Output: data/hypernets/obj_XX.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN, flatten_weights
from src.hypernet import HyperNet
from src.render import scan_rig_directions


def load_target_weights(obj_idx: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load 24 camera directions and 24 flattened SIREN weight vectors."""
    c = CFG.img_siren
    d = CFG.data
    cam_dirs = torch.from_numpy(scan_rig_directions()).to(device)

    weight_list = []
    for view_j in range(d.num_views):
        siren_path = c.out_dir / f"obj_{obj_idx:02d}" / f"view_{view_j:02d}.pt"
        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(device)
        siren.load_state_dict(torch.load(siren_path, map_location=device, weights_only=True))
        weight_list.append(flatten_weights(siren))

    return cam_dirs, torch.stack(weight_list)


def main():
    dev = CFG.device
    hc = CFG.hypernet
    d = CFG.data
    c = CFG.img_siren

    anchor_path = hc.out_dir.parent / "checkpoints" / "anchor_hypernet.pt"
    assert anchor_path.exists(), (
        f"anchor not found at {anchor_path} — run 40_train_anchor_hypernet.py first"
    )
    anchor_ckpt = torch.load(anchor_path, map_location=dev, weights_only=True)
    print(f"[hypernet] loaded anchor from {anchor_path}")

    hc.out_dir.mkdir(parents=True, exist_ok=True)
    ref_siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden)

    # fewer steps for fine-tuning (anchor did heavy lifting)
    finetune_steps = hc.steps // 2

    for obj_i in range(d.num_objects):
        out_path = hc.out_dir / f"obj_{obj_i:02d}.pt"
        if out_path.exists():
            print(f"[hypernet] skip obj_{obj_i:02d}")
            continue

        # fresh copy from anchor
        hypernet = HyperNet(
            target_siren=ref_siren,
            in_dim=3,
            head_hidden=hc.head_hidden,
            head_layers=hc.head_layers,
            final_init_scale=hc.final_init_scale,
        ).to(dev)
        hypernet.load_state_dict(anchor_ckpt["state_dict"])

        cam_dirs, target_weights = load_target_weights(obj_i, dev)
        opt = torch.optim.Adam(hypernet.parameters(), lr=hc.lr_hypernet)

        print(f"[hypernet] obj_{obj_i:02d}  fine-tuning for {finetune_steps} steps")
        for step in range(finetune_steps):
            total_loss = 0.0
            opt.zero_grad(set_to_none=True)

            for j in range(cam_dirs.shape[0]):
                pred = hypernet(cam_dirs[j])
                loss = F.mse_loss(pred, target_weights[j])
                loss.backward()
                total_loss += loss.item()

            opt.step()

            avg = total_loss / cam_dirs.shape[0]
            if step % 100 == 0 or step == finetune_steps - 1:
                print(f"  step {step:5d}  avg_loss {avg:.8f}")

        print(f"  final_loss {avg:.8f}")
        torch.save(hypernet.state_dict(), out_path)
        print(f"[hypernet] saved -> {out_path}")

    print(f"\n[hypernet] done — 10 hypernetworks saved to {hc.out_dir}")


if __name__ == "__main__":
    main()