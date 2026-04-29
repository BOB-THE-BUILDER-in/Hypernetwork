"""Run watertight + render + image-SIRENs for primitive OOD test meshes.
Skips Objaverse download — uses meshes already at data/ood_test/obj_NNN.obj.
"""
import sys
sys.path.insert(0, "/workspace/hypernet")
sys.path.insert(0, "/workspace/hypernet/scripts")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--obj_idx", type=int, required=True)
args = ap.parse_args()

import ood_test_full as oot
from pathlib import Path

oot.OBJ_IDX = args.obj_idx
oot.OBJ_TAG = f"obj_{args.obj_idx:03d}"
oot.OOD_GLB   = oot.OOD_DIR / f"{oot.OBJ_TAG}.glb"  # not used
oot.OOD_WT    = oot.OOD_DIR / f"{oot.OBJ_TAG}.obj"  # already exists
oot.OOD_VIEWS = oot.OOD_DIR / oot.OBJ_TAG / "views"
oot.OOD_SDF   = oot.OOD_DIR / f"{oot.OBJ_TAG}.npz"
oot.OOD_IMG_SIRENS = oot.OOD_DIR / oot.OBJ_TAG / "image_sirens"

assert oot.OOD_WT.exists(), f"missing {oot.OOD_WT}"
print(f"Preparing {oot.OBJ_TAG}: {oot.OOD_WT}")

# Skip download + watertight (already have valid mesh)
# Run render + sdf + image-SIRENs
oot.render_stage()
oot.sdf_stage()
oot.image_siren_stage()

import os
n = len([f for f in os.listdir(oot.OOD_IMG_SIRENS) if f.endswith('.pt')]) if oot.OOD_IMG_SIRENS.exists() else 0
print(f"\n[done] {oot.OBJ_TAG}: {n}/24 image-SIRENs at {oot.OOD_IMG_SIRENS}")
