"""Watertight meshes via scatter + VDB-from-particles approach.

Replicates the Houdini pipeline:
    1. Scatter points on mesh surface (high density)
    2. VDB from particles (each point creates a small sphere of density)
    3. Morphological closing to seal any remaining pinholes
    4. Convert to polygons (marching cubes)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure


def _normalize_mesh(mesh: trimesh.Trimesh, padding: float = 0.85) -> trimesh.Trimesh:
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.apply_translation(-center)
    extent = np.abs(mesh.vertices).max()
    if extent > 0:
        mesh.apply_scale(padding / extent)
    return mesh


def _scatter_points(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, count)
    return pts.astype(np.float32)


def _vdb_from_particles(
    points: np.ndarray,
    resolution: int,
    particle_radius: float,
) -> np.ndarray:
    """Splat each point as a sphere into a binary voxel grid."""
    # convert points from [-1,1] to voxel indices
    voxel_pts = (points + 1.0) / 2.0 * (resolution - 1)

    # mark occupied voxels
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    ix = np.clip(np.round(voxel_pts[:, 0]).astype(np.int32), 0, resolution - 1)
    iy = np.clip(np.round(voxel_pts[:, 1]).astype(np.int32), 0, resolution - 1)
    iz = np.clip(np.round(voxel_pts[:, 2]).astype(np.int32), 0, resolution - 1)
    grid[ix, iy, iz] = 1.0

    # dilate with sphere kernel (= particle radius)
    r = int(np.ceil(particle_radius))
    kernel_size = 2 * r + 1
    zz, yy, xx = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    sphere = (xx**2 + yy**2 + zz**2 <= particle_radius**2).astype(np.float32)

    # binary dilation — guarantees every occupied voxel spreads uniformly
    grid_binary = (grid > 0).astype(np.float32)
    grid_dilated = ndimage.binary_dilation(grid_binary, structure=sphere).astype(np.float32)

    # morphological closing: dilate then erode — seals pinholes
    close_r = max(1, int(particle_radius * 0.5))
    close_kernel_size = 2 * close_r + 1
    cz, cy, cx = np.mgrid[-close_r:close_r+1, -close_r:close_r+1, -close_r:close_r+1]
    close_sphere = (cx**2 + cy**2 + cz**2 <= close_r**2)
    grid_closed = ndimage.binary_closing(grid_dilated, structure=close_sphere).astype(np.float32)

    return grid_closed


def watertight(
    in_path: Path,
    out_path: Path,
    resolution: int = 384,
    scatter_count: int = 3_000_000,
    particle_radius: float = 2.0,
    smooth_sigma: float = 0.8,
) -> trimesh.Trimesh:
    """Load mesh, scatter points, build volume, extract watertight surface."""
    mesh = trimesh.load(in_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"could not load as single mesh: {in_path}")
    mesh = _normalize_mesh(mesh)

    print(f"    scattering {scatter_count:,} points...")
    points = _scatter_points(mesh, count=scatter_count)

    print(f"    building density grid @ {resolution}^3 (radius={particle_radius})...")
    grid = _vdb_from_particles(points, resolution, particle_radius)

    # smooth for clean surface
    if smooth_sigma > 0:
        grid = ndimage.gaussian_filter(grid.astype(np.float32), sigma=smooth_sigma)

    # extract at 0.5 threshold (binary grid smoothed — 0.5 is the natural boundary)
    print(f"    marching cubes...")
    verts, faces, normals, _ = measure.marching_cubes(grid, level=0.5)
    verts = verts / (resolution - 1) * 2.0 - 1.0

    wt = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wt.export(out_path)
    print(f"    exported: {len(wt.vertices):,} verts, {len(wt.faces):,} faces")
    return wt