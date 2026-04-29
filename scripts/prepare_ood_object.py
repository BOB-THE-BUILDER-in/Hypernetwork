import sys
sys.path.insert(0, "/workspace/hypernet")
sys.path.insert(0, "/workspace/hypernet/scripts")
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--obj_idx", type=int, default=101)
args = ap.parse_args()

import ood_test_full as oot

oot.OBJ_IDX = args.obj_idx
oot.OBJ_TAG = f"obj_{args.obj_idx:03d}"
oot.OOD_GLB   = oot.OOD_DIR / f"{oot.OBJ_TAG}.glb"
oot.OOD_WT    = oot.OOD_DIR / f"{oot.OBJ_TAG}.obj"
oot.OOD_VIEWS = oot.OOD_DIR / oot.OBJ_TAG / "views"
oot.OOD_SDF   = oot.OOD_DIR / f"{oot.OBJ_TAG}.npz"
oot.OOD_IMG_SIRENS = oot.OOD_DIR / oot.OBJ_TAG / "image_sirens"

print(f"Preparing {oot.OBJ_TAG}")
oot.download()
oot.watertight_stage()
oot.render_stage()
oot.sdf_stage()
oot.image_siren_stage()

import os
n = len([f for f in os.listdir(oot.OOD_IMG_SIRENS) if f.endswith('.pt')]) if oot.OOD_IMG_SIRENS.exists() else 0
print(f"\n[done] {oot.OBJ_TAG}: {n}/24 image-SIRENs at {oot.OOD_IMG_SIRENS}")
