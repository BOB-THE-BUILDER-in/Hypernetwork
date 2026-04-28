"""
Retrain 100 hypernets at a configurable head_hidden, matching the original
protocol from scripts/40_train_anchor_hypernet.py + scripts/50_train_hypernets.py.

Protocol (from original):
  Stage 1: anchor hypernet trained on obj_00's 24 image-SIRENs, 2000 steps
  Stage 2: per-object hypernet, warm-start from anchor, 1000 steps

Defaults to head_hidden=64. Pass --hidden 128 to test the larger size.

Outputs:
  data/checkpoints/anchor_hypernet_h{H}.pt
  data/hypernets_h{H}/obj_NN.pt   (100 files)

Speed optimization: original loops over 24 views with separate backward calls,
~24x kernel launch overhead. We do one batched forward+backward per step.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def patch_config(head_hidden):
    """Update config head_hidden if needed."""
    cfg_path = Path("/workspace/hypernet/configs/config.py")
    text = cfg_path.read_text()
    target_line = f"head_hidden: int = {head_hidden}"
    if target_line in text:
        log(f"config already has head_hidden={head_hidden}")
        return
    for old in ("head_hidden: int = 256", "head_hidden: int = 128",
                "head_hidden: int = 64", "head_hidden: int = 32"):
        if old in text:
            text = text.replace(old, target_line)
            cfg_path.write_text(text)
            log(f"patched config: {old} -> {target_line}")
            return
    raise RuntimeError("could not find head_hidden line in config")


def load_target_weights(obj_idx, device, c):
    """Load 24 image-SIREN flat weight vectors for one object."""
    from src.siren import SIREN, flatten_weights
    target_dir = Path("/workspace/hypernet/data/image_sirens") / f"obj_{obj_idx:02d}"
    weights = []
    for j in range(24):
        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(device)
        siren.load_state_dict(
            torch.load(target_dir / f"view_{j:02d}.pt",
                       map_location=device, weights_only=True)
        )
        weights.append(flatten_weights(siren))
    return torch.stack(weights)  # (24, siren_dim)


def train_anchor(args, device):
    """Stage 1: train anchor on obj_00, 2000 steps (matches original)."""
    from configs.config import CFG
    from src.siren import SIREN
    from src.hypernet import HyperNet
    from src.render import scan_rig_directions

    out_path = Path("/workspace/hypernet/data/checkpoints") / f"anchor_hypernet_h{args.hidden}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        log(f"anchor exists: {out_path}")
        return out_path

    c = CFG.img_siren
    hc = CFG.hypernet
    log(f"[anchor] head_hidden={hc.head_hidden}  steps={hc.steps}")

    cam_dirs = torch.from_numpy(scan_rig_directions()).to(device)
    targets = load_target_weights(0, device, c)  # obj_00 only, original protocol
    log(f"[anchor] target weights shape: {tuple(targets.shape)}")

    ref = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers, c.w0_first, c.w0_hidden)
    hn = HyperNet(target_siren=ref, in_dim=3,
                  head_hidden=hc.head_hidden, head_layers=hc.head_layers,
                  final_init_scale=hc.final_init_scale).to(device)
    log(f"[anchor] hypernet params: {hn.num_params():,}")

    head_list = list(hn.heads.values())
    opt = torch.optim.Adam(hn.parameters(), lr=hc.lr_hypernet)

    t0 = time.time()
    for step in range(hc.steps):
        opt.zero_grad(set_to_none=True)
        # Batched: each head processes all 24 cam_dirs at once
        parts = [head(cam_dirs) for head in head_list]
        preds = torch.cat(parts, dim=-1)  # (24, siren_dim)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == hc.steps - 1:
            log(f"  anchor step {step:5d}  loss {loss.item():.3e}")
    log(f"[anchor] done in {time.time()-t0:.1f}s, final loss {loss.item():.3e}")

    torch.save({"state_dict": hn.state_dict(),
                "in_dim": 3,
                "head_hidden": hc.head_hidden,
                "head_layers": hc.head_layers,
                "final_init_scale": hc.final_init_scale}, out_path)
    log(f"[anchor] saved -> {out_path}")
    return out_path


def train_per_object(args, device, anchor_path):
    """Stage 2: fine-tune each object's hypernet from anchor."""
    from configs.config import CFG
    from src.siren import SIREN
    from src.hypernet import HyperNet
    from src.render import scan_rig_directions

    c = CFG.img_siren
    hc = CFG.hypernet
    out_dir = Path(f"/workspace/hypernet/data/hypernets_h{args.hidden}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[per-obj] output: {out_dir}")

    finetune_steps = hc.steps // 2  # matches original
    log(f"[per-obj] finetune_steps={finetune_steps} per object")

    cam_dirs = torch.from_numpy(scan_rig_directions()).to(device)
    anchor_ckpt = torch.load(anchor_path, map_location=device, weights_only=True)
    anchor_sd = anchor_ckpt["state_dict"] if "state_dict" in anchor_ckpt else anchor_ckpt

    ref = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers, c.w0_first, c.w0_hidden)

    total_t0 = time.time()
    for obj_idx in range(args.start, args.end):
        out_path = out_dir / f"obj_{obj_idx:02d}.pt"
        if out_path.exists():
            continue
        target_dir = Path("/workspace/hypernet/data/image_sirens") / f"obj_{obj_idx:02d}"
        if not target_dir.exists():
            log(f"obj_{obj_idx:02d}: image-SIRENs missing, skip")
            continue

        targets = load_target_weights(obj_idx, device, c)
        hn = HyperNet(target_siren=ref, in_dim=3,
                      head_hidden=hc.head_hidden, head_layers=hc.head_layers,
                      final_init_scale=hc.final_init_scale).to(device)
        hn.load_state_dict(anchor_sd)  # warm-start
        head_list = list(hn.heads.values())
        opt = torch.optim.Adam(hn.parameters(), lr=hc.lr_hypernet)

        t0 = time.time()
        for step in range(finetune_steps):
            opt.zero_grad(set_to_none=True)
            parts = [head(cam_dirs) for head in head_list]
            preds = torch.cat(parts, dim=-1)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            opt.step()
        torch.save(hn.state_dict(), out_path)
        log(f"obj_{obj_idx:02d}: loss {loss.item():.3e}  ({time.time()-t0:.1f}s)")

    log(f"[per-obj] DONE all {args.end-args.start} in {(time.time()-total_t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64,
                    help="head_hidden value (try 64 first, bump to 128 if quality bad)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100)
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/hypernet")
    patch_config(args.hidden)

    # Force fresh import after config patch
    if "configs.config" in sys.modules:
        del sys.modules["configs.config"]
    from configs.config import CFG

    device = torch.device(CFG.device)
    log(f"using device: {device}, head_hidden={CFG.hypernet.head_hidden}")

    anchor_path = train_anchor(args, device)
    train_per_object(args, device, anchor_path)


if __name__ == "__main__":
    main()
