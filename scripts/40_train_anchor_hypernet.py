"""Train anchor hypernetwork on object 0's 24 image-SIRENs.

All 10 per-object hypernetworks warm-start from this checkpoint.

Input:  camera direction (3D unit vector from scan rig)
Output: predicted flat SIREN weight vector
Loss:   MSE between predicted weights and actual trained SIREN weights
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

    # camera directions from scan rig (24, 3)
    cam_dirs = torch.from_numpy(scan_rig_directions()).to(device)

    # load each trained SIREN and flatten its weights
    weight_list = []
    for view_j in range(d.num_views):
        siren_path = c.out_dir / f"obj_{obj_idx:02d}" / f"view_{view_j:02d}.pt"
        assert siren_path.exists(), f"missing {siren_path}"
        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(device)
        siren.load_state_dict(torch.load(siren_path, map_location=device, weights_only=True))
        weight_list.append(flatten_weights(siren))

    weights = torch.stack(weight_list)  # (24, num_siren_params)
    return cam_dirs, weights


def main():
    dev = CFG.device
    hc = CFG.hypernet
    anchor_path = hc.out_dir.parent / "checkpoints" / "anchor_hypernet.pt"
    anchor_path.parent.mkdir(parents=True, exist_ok=True)

    # build a reference SIREN to define the architecture
    c = CFG.img_siren
    ref_siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden)

    # build hypernetwork
    hypernet = HyperNet(
        target_siren=ref_siren,
        in_dim=3,
        head_hidden=hc.head_hidden,
        head_layers=hc.head_layers,
        final_init_scale=hc.final_init_scale,
    ).to(dev)

    print(f"[anchor-hypernet] HyperNet params: {hypernet.num_params():,}")
    print(f"[anchor-hypernet] Target SIREN params: {hypernet.total_target_params:,}")

    # load object 0's 24 SIRENs as targets
    cam_dirs, target_weights = load_target_weights(obj_idx=0, device=dev)
    print(f"[anchor-hypernet] loaded 24 target weight vectors, shape {target_weights.shape}")

    opt = torch.optim.Adam(hypernet.parameters(), lr=hc.lr_hypernet)
    steps = hc.steps

    print(f"[anchor-hypernet] training for {steps} steps on object 0")
    for step in range(steps):
        # each step: predict all 24 views
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for j in range(cam_dirs.shape[0]):
            pred = hypernet(cam_dirs[j])
            loss = F.mse_loss(pred, target_weights[j])
            loss.backward()
            total_loss += loss.item()

        opt.step()

        if step % 500 == 0:
            avg = total_loss / cam_dirs.shape[0]
            print(f"  step {step:5d}  avg_loss {avg:.8f}")

    torch.save({
        "state_dict": hypernet.state_dict(),
        "in_dim": 3,
        "head_hidden": hc.head_hidden,
        "head_layers": hc.head_layers,
        "final_init_scale": hc.final_init_scale,
    }, anchor_path)
    print(f"[anchor-hypernet] saved -> {anchor_path}")


if __name__ == "__main__":
    main()
