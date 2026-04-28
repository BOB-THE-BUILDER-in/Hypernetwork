"""Multi-view renderer — 3-ring scan-rig layout (24 views).

Three stacked rings of 8 cameras each:
    - upper ring: elevation +45°,  8 azimuths starting at 0°
    - middle ring: elevation  0°,  8 azimuths offset by 22.5°
    - lower ring: elevation -45°,  8 azimuths offset by 11.25°

The azimuth offsets between rings mean consecutive-elevation cameras are
never vertically aligned, so every "pillar" of the object gets multiple
distinct looks from different heights.

Same 24 poses for every object — non-negotiable for the hypernet pipeline.

Uses pyrender + EGL (Linux headless). Black background, auto-fit distance,
strong lighting.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import trimesh

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

FOV_DEG = 40.0

# (elevation_deg, azimuth_offset_deg) per ring. 8 cameras per ring, 45° step.
RING_SPEC: tuple[tuple[float, float], ...] = (
    ( 45.0, 0.0),     # upper
    (  0.0, 22.5),    # middle (offset 1/2 ring-step)
    (-45.0, 11.25),   # lower  (offset 1/4 ring-step)
)
PER_RING = 8


def _fit_radius(mesh: trimesh.Trimesh, fov_deg: float = FOV_DEG, padding: float = 1.3) -> float:
    r_obj = np.linalg.norm(mesh.vertices, axis=1).max()
    half_fov = np.deg2rad(fov_deg) / 2.0
    d = r_obj / np.tan(half_fov)
    return float(d * padding)


def scan_rig_directions(
    per_ring: int = PER_RING,
    rings: tuple[tuple[float, float], ...] = RING_SPEC,
) -> np.ndarray:
    """Returns (num_rings * per_ring, 3) unit direction vectors.
    Order: ring 0 (upper), ring 1 (middle), ring 2 (lower). Azimuth ascends
    within each ring starting from that ring's offset.
    """
    dirs: list[tuple[float, float, float]] = []
    az_step = 2.0 * np.pi / per_ring
    for elev_deg, az_off_deg in rings:
        elev = np.deg2rad(elev_deg)
        az_off = np.deg2rad(az_off_deg)
        cos_e, sin_e = np.cos(elev), np.sin(elev)
        for i in range(per_ring):
            az = az_off + i * az_step
            dirs.append((cos_e * np.cos(az), sin_e, cos_e * np.sin(az)))
    return np.asarray(dirs, dtype=np.float32)


def camera_poses(radius: float) -> np.ndarray:
    directions = scan_rig_directions()
    poses = []
    for d in directions:
        eye = d * radius
        target = np.zeros(3, dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)

        f = target - eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, up)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            up = np.array([0, 0, 1], dtype=np.float32)
            r = np.cross(f, up)
            r_norm = np.linalg.norm(r)
        r = r / r_norm
        u = np.cross(r, f)

        M = np.eye(4, dtype=np.float32)
        M[:3, 0] = r
        M[:3, 1] = u
        M[:3, 2] = -f
        M[:3, 3] = eye
        poses.append(M)
    return np.stack(poses)


def _prepare_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    center = m.bounds.mean(axis=0)
    m.apply_translation(-center)
    return m


def render_views(
    mesh_path: Path,
    out_dir: Path,
    num_views: int = 24,   # signature compat; count determined by RING_SPEC * PER_RING
    img_res: int = 256,
) -> list[Path]:
    import pyrender
    from PIL import Image

    raw = trimesh.load(mesh_path, force="mesh")
    mesh = _prepare_mesh(raw)
    radius = _fit_radius(mesh, fov_deg=FOV_DEG, padding=1.3)

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 1.0],
        ambient_light=[0.25, 0.25, 0.25],
    )
    scene.add(pr_mesh)

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(FOV_DEG))
    cam_node = scene.add(cam, pose=np.eye(4))

    light = pyrender.PointLight(color=np.ones(3), intensity=radius ** 2 * 8.0)
    light_node = scene.add(light, pose=np.eye(4))

    r = pyrender.OffscreenRenderer(img_res, img_res)

    out_dir.mkdir(parents=True, exist_ok=True)
    poses = camera_poses(radius=radius)
    expected = PER_RING * len(RING_SPEC)
    assert poses.shape[0] == expected, f"pose count mismatch: {poses.shape[0]} vs {expected}"

    saved: list[Path] = []
    for i, pose in enumerate(poses):
        scene.set_pose(cam_node, pose)
        scene.set_pose(light_node, pose)
        color, _ = r.render(scene)
        p = out_dir / f"view_{i:02d}.png"
        Image.fromarray(color).save(p)
        saved.append(p)
    r.delete()
    return saved