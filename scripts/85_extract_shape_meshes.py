"""Extract meshes from trained shape-SIRENs via marching cubes.

Output: data/shape_sirens/obj_XX.obj
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN


def extract_mesh(siren: SIREN, resolution: int, device: str) -> trimesh.Trimesh:
    """Query SIREN on a dense grid and extract zero-isosurface."""
    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    xs, ys, zs = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    pts_t = torch.from_numpy(pts).to(device)

    # chunk to avoid OOM
    chunk = 100_000
    sdf = np.empty(pts.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            sdf[i:i + chunk] = siren(pts_t[i:i + chunk]).squeeze(-1).cpu().numpy()

    sdf = sdf.reshape(resolution, resolution, resolution)
    verts, faces, normals, _ = measure.marching_cubes(sdf, level=0.0)
    verts = verts / (resolution - 1) * 2.0 - 1.0

    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)


def main():
    dev = CFG.device
    c = CFG.shape_siren
    d = CFG.data
    resolution = 256

    for obj_i in range(d.num_objects):
        siren_path = c.out_dir / f"obj_{obj_i:02d}.pt"
        out_path = c.out_dir / f"obj_{obj_i:02d}.obj"

        if out_path.exists():
            print(f"[extract] skip obj_{obj_i:02d}")
            continue

        siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                      c.w0_first, c.w0_hidden).to(dev)
        siren.load_state_dict(torch.load(siren_path, map_location=dev, weights_only=True))
        siren.eval()

        print(f"[extract] obj_{obj_i:02d}  marching cubes @ {resolution}³...")
        mesh = extract_mesh(siren, resolution, dev)
        mesh.export(out_path)
        print(f"[extract] saved {out_path}  verts={len(mesh.vertices)}  faces={len(mesh.faces)}")

    print(f"\n[extract] done")


if __name__ == "__main__":
    main()
