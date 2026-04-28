"""Retrain 100 hypernets at head_hidden=64, batched (no per-view loop)."""
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path("/workspace/hypernet")
sys.path.insert(0, str(PROJECT))

# Patch config FIRST (before imports that read CFG)
cfg_path = PROJECT / "configs" / "config.py"
text = cfg_path.read_text()
if "head_hidden: int = 256" in text:
    cfg_path.write_text(text.replace("head_hidden: int = 256", "head_hidden: int = 64"))
    print("[patch] config head_hidden 256 -> 64", flush=True)
else:
    print("[patch] config already at head_hidden=64", flush=True)

from configs.config import CFG
from src.siren import SIREN, flatten_weights
from src.hypernet import HyperNet
from src.render import scan_rig_directions

device = torch.device(CFG.device)
c = CFG.img_siren
hc = CFG.hypernet
out_dir = PROJECT / "data" / "hypernets_small"
out_dir.mkdir(parents=True, exist_ok=True)

cam_dirs = torch.from_numpy(scan_rig_directions()).to(device)  # (24, 3)
print(f"cam_dirs: {cam_dirs.shape}", flush=True)

def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

total_t0 = time.time()
for obj_idx in range(100):
    out_path = out_dir / f"obj_{obj_idx:02d}.pt"
    if out_path.exists():
        continue

    # Load 24 image-SIREN target weights
    target_dir = PROJECT / "data" / "image_sirens" / f"obj_{obj_idx:02d}"
    if not target_dir.exists():
        log(f"obj_{obj_idx:02d}: image-SIRENs missing, skip")
        continue

    target_weights = []
    for j in range(CFG.data.num_views):
        sp = target_dir / f"view_{j:02d}.pt"
        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(device)
        siren.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
        target_weights.append(flatten_weights(siren))
    target_weights = torch.stack(target_weights)  # (24, siren_dim)

    # Build hypernet at the small size
    ref = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers, c.w0_first, c.w0_hidden)
    hypernet = HyperNet(
        target_siren=ref, in_dim=3,
        head_hidden=hc.head_hidden, head_layers=hc.head_layers,
        final_init_scale=hc.final_init_scale,
    ).to(device)

    opt = torch.optim.Adam(hypernet.parameters(), lr=hc.lr_hypernet)
    t0 = time.time()

    # KEY FIX: Pass ALL 24 cam_dirs through hypernet at once.
    # The HyperNetHead is just Linear stacked — handles batched input natively.
    # We bypass HyperNet.forward (which loops) and call heads directly with the batch.
    head_list = list(hypernet.heads.values())

    for step in range(2000):
        opt.zero_grad(set_to_none=True)
        # Each head processes all 24 cam_dirs at once
        parts = [head(cam_dirs) for head in head_list]   # each: (24, head_out)
        preds = torch.cat(parts, dim=-1)                 # (24, siren_dim)
        loss = F.mse_loss(preds, target_weights)
        loss.backward()
        opt.step()

    torch.save(hypernet.state_dict(), out_path)
    elapsed = time.time() - t0
    log(f"obj_{obj_idx:02d}: loss {loss.item():.3e}  ({elapsed:.1f}s)")

log(f"DONE in {(time.time()-total_t0)/60:.1f} min")
