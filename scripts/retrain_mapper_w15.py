"""
Retrain the hypernet→shape mapper on the new ω₀=15 shape-SIRENs.

Hypernets are unchanged (they only know about image-SIRENs, which we didn't
retrain). Only the shape-SIREN side of each pair changes. Writes:
  /workspace/hypernet/scripts/manifest_w15.pt  (new manifest)
  /workspace/hypernet/scripts/t1_w15/mapper.pt (new mapper checkpoint)
  /workspace/hypernet/scripts/t1_w15/predicted_shape_weights.pt
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import torch

SCRIPTS = Path("/workspace/hypernet/scripts")
DATA = Path("/workspace/hypernet/data")


def build_manifest():
    """Create a new manifest that pairs existing hypernets with new shape-SIRENs."""
    out_path = SCRIPTS / "manifest_w15.pt"

    hyp_dir = DATA / "hypernets"
    shp_dir = DATA / "shape_sirens_w15"

    import re
    key_re = re.compile(r"obj_(\d+)")

    hyp_files = {}
    for p in sorted(hyp_dir.glob("obj_*.pt")):
        m = key_re.search(p.name)
        if m:
            hyp_files[m.group(1)] = p

    shp_files = {}
    for p in sorted(shp_dir.glob("obj_*.pt")):
        m = key_re.search(p.name)
        if m:
            shp_files[m.group(1)] = p

    keys = sorted(set(hyp_files) & set(shp_files), key=lambda k: int(k))
    if not keys:
        raise RuntimeError(f"no paired files. hyp_dir={hyp_dir} shp_dir={shp_dir}")

    missing_shp = sorted(set(hyp_files) - set(shp_files))
    missing_hyp = sorted(set(shp_files) - set(hyp_files))
    if missing_shp:
        print(f"[manifest] warning: hypernets without ω₀=15 shape-SIREN: {missing_shp}")
    if missing_hyp:
        print(f"[manifest] warning: ω₀=15 shape-SIRENs without hypernet: {missing_hyp}")

    hyp_paths = [hyp_files[k] for k in keys]
    shp_paths = [shp_files[k] for k in keys]

    print(f"[manifest] paired {len(keys)} objects for ω₀=15 retrain")
    for k, hp, sp in zip(keys, hyp_paths, shp_paths):
        print(f"  key={k}  hyp={hp.name}  shape={sp.name}")

    torch.save({"hypernet_paths": hyp_paths, "shape_paths": shp_paths}, out_path)
    print(f"[manifest] saved -> {out_path}")
    return out_path


def train_mapper(manifest_path: Path):
    """Call the existing mapper training script with the new manifest and anchor."""
    cmd = [
        sys.executable,
        str(SCRIPTS / "hypernet_to_shape_mapper.py"),
        "--manifest", str(manifest_path),
        "--anchor_hyp", "/workspace/hypernet/data/checkpoints/anchor_hypernet.pt",
        "--anchor_shp", "/workspace/hypernet/checkpoints/anchor_shape_siren_w15.pt",
        "--out", str(SCRIPTS / "t1_w15"),
        "--lr", "1e-3",
        "--wd", "0",
        "--grad_clip", "0",
        "--steps", "3000",
    ]
    print(f"\n[mapper@w15] launching training:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    manifest_path = build_manifest()
    train_mapper(manifest_path)
