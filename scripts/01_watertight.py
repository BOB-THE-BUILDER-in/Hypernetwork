"""Watertight all meshes via scatter + VDB-from-particles pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.watertight import watertight


def main():
    CFG.data.watertight_dir.mkdir(parents=True, exist_ok=True)
    meshes = sorted(CFG.data.meshes_dir.glob("*.glb")) + sorted(CFG.data.meshes_dir.glob("*.obj"))
    assert len(meshes) >= CFG.data.num_objects, f"need {CFG.data.num_objects} meshes, found {len(meshes)}"

    for i, src in enumerate(meshes[:CFG.data.num_objects]):
        dst = CFG.data.watertight_dir / f"obj_{i:02d}.obj"
        if dst.exists():
            print(f"[watertight] skip {dst.name}")
            continue
        print(f"[watertight] {src.name} -> {dst.name}")
        watertight(
            src, dst,
            resolution=384,
            scatter_count=3_000_000,
            particle_radius=2.0,
            smooth_sigma=0.8,
        )


if __name__ == "__main__":
    main()