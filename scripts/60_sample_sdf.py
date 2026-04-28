"""Sample SDF training points from watertight meshes.

Uses pysdf for fast SDF computation (falls back to trimesh if unavailable).
Multi-scale noise sampling around surface for good near-surface coverage.

Output: data/sdf_samples/obj_XX.npz  (points, sdf)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG


def make_sdf_fn(mesh: trimesh.Trimesh):
    """Create SDF function. pysdf is ~100x faster than trimesh.proximity."""
    try:
        from pysdf import SDF
        sdf_fn = SDF(mesh.vertices, mesh.faces)
        # pysdf returns negative outside, positive inside — flip to standard convention
        return lambda pts: -sdf_fn(pts)
    except ImportError:
        print("  WARNING: pysdf not installed, using slow trimesh fallback")
        return lambda pts: -trimesh.proximity.signed_distance(mesh, pts)


def sample_object(mesh_path: Path, n_surface: int = 100_000,
                  n_space: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """Sample points and SDF values from a watertight mesh."""
    mesh = trimesh.load(mesh_path, force="mesh")

    # normalize to [-1, 1]
    verts = mesh.vertices.copy()
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts -= center
    scale = np.abs(verts).max()
    verts /= scale * 1.1
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

    print(f"  mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    # surface points
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
    surface_pts = surface_pts.astype(np.float32)

    # multi-scale noise around surface
    noise_small = surface_pts + np.random.randn(*surface_pts.shape).astype(np.float32) * 0.005
    noise_medium = surface_pts + np.random.randn(*surface_pts.shape).astype(np.float32) * 0.02
    noise_large = surface_pts + np.random.randn(*surface_pts.shape).astype(np.float32) * 0.1

    # uniform volume
    uniform_pts = np.random.uniform(-1, 1, (n_space, 3)).astype(np.float32)

    # combine all
    all_pts = np.concatenate([surface_pts, noise_small, noise_medium, noise_large, uniform_pts], axis=0)

    # compute SDF using pysdf
    print(f"  computing SDF for {len(all_pts):,} points...")
    t0 = time.time()
    sdf_fn = make_sdf_fn(mesh)

    chunk = 200_000
    sdf = np.zeros(len(all_pts), dtype=np.float32)
    for i in range(0, len(all_pts), chunk):
        sdf[i:i + chunk] = sdf_fn(all_pts[i:i + chunk]).astype(np.float32)
        print(f"    chunk {i // chunk + 1}/{math.ceil(len(all_pts) / chunk)}")

    print(f"  SDF done in {time.time() - t0:.1f}s")
    return all_pts, sdf


def main():
    d = CFG.data
    d.sdf_dir.mkdir(parents=True, exist_ok=True)

    for obj_i in range(d.num_objects):
        out_path = d.sdf_dir / f"obj_{obj_i:02d}.npz"
        if out_path.exists():
            print(f"[sdf-sample] skip obj_{obj_i:02d}")
            continue

        mesh_path = d.watertight_dir / f"obj_{obj_i:02d}.obj"
        assert mesh_path.exists(), f"missing {mesh_path}"

        print(f"[sdf-sample] obj_{obj_i:02d}")
        pts, sdf = sample_object(mesh_path)

        # clamp
        sdf = np.clip(sdf, -d.sdf_truncation, d.sdf_truncation)

        np.savez_compressed(out_path, points=pts, sdf=sdf)
        print(f"[sdf-sample] saved {out_path}  pts={pts.shape}  sdf range=[{sdf.min():.4f}, {sdf.max():.4f}]")

    print(f"\n[sdf-sample] done")


if __name__ == "__main__":
    main()