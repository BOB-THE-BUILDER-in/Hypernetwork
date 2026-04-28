"""
Scale the pipeline from 10 to 100 shapes.

What this does:
    1. Patches CFG.data.num_objects = 100 in-memory (does NOT edit config.py)
    2. Downloads 90 new Objaverse meshes, naming them obj_10.glb through obj_99.glb
       so the existing obj_00..obj_09 are preserved untouched
    3. Runs watertight, SDF sampling, rendering, image-SIRENs, hypernets,
       shape-SIRENs in sequence. Each stage skips already-processed objects.
    4. Does NOT touch the anchors (anchor_image_siren, anchor_hypernet,
       anchor_shape_siren). They stay as originally trained.
    5. Builds manifest_n100.pt pointing to all 100 pairs.

Designed to be restart-safe. If Vast.ai reboots, just rerun this script;
skip-if-exists on every stage means it picks up where it left off.

Logging: everything goes to BOTH stdout and a log file so you can tail -f
overnight.
"""
from __future__ import annotations

import argparse
import random
import runpy
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/workspace/hypernet")
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(PROJECT))

from configs.config import CFG                                # noqa: E402


TARGET_N = 100
DOWNLOAD_TIMEOUT_SEC = 60
CANDIDATE_POOL = 1200  # uids to try (90 new shapes, many will fail/be bad)


# ---------------------------------------------------------------------------
# Logging helper: tee to file
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "a", buffering=1)
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)
    def flush(self):
        self.stdout.flush()
        self.fh.flush()


def log_header(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = "=" * 70
    print(f"\n{bar}\n[{ts}] {msg}\n{bar}")


# ---------------------------------------------------------------------------
# Stage 1: download new shapes into slots 10..99
# ---------------------------------------------------------------------------

class _Timeout:
    def __init__(self, sec): self.sec = sec
    def __enter__(self):
        def handler(signum, frame): raise TimeoutError("download timeout")
        self._prev = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.sec)
    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._prev)


def fetch_one(uid):
    import objaverse
    try:
        with _Timeout(DOWNLOAD_TIMEOUT_SEC):
            result = objaverse.load_objects(uids=[uid])
        src_str = result.get(uid)
        if not src_str:
            return None
        src = Path(src_str)
        return src if src.exists() else None
    except Exception as e:
        print(f"    skip {uid[:12]}: {type(e).__name__} {e}")
        return None


def download_stage():
    log_header("STAGE 1: download new Objaverse shapes into slots 10..99")
    meshes_dir = CFG.data.meshes_dir
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Find which slots already have files
    existing = set()
    for p in meshes_dir.glob("obj_*.glb"):
        try:
            idx = int(p.stem.split("_")[-1])
            existing.add(idx)
        except ValueError:
            pass
    print(f"[download] existing meshes: {sorted(existing)}")

    missing = [i for i in range(TARGET_N) if i not in existing]
    if not missing:
        print(f"[download] all {TARGET_N} slots already filled. skipping.")
        return

    print(f"[download] need to fill slots: {missing[:5]}..{missing[-5:]} ({len(missing)} total)")

    import objaverse
    all_uids = objaverse.load_uids()
    print(f"[download] {len(all_uids)} uids available in Objaverse")

    rng = random.Random(CFG.seed + 1)  # different seed from original download
    candidates = rng.sample(all_uids, min(CANDIDATE_POOL, len(all_uids)))

    # Skip uids that map to slots we've already filled? No — Objaverse doesn't
    # know about our slot indices. Any new uid maps to the next free slot.
    next_slot_iter = iter(missing)
    try:
        target_slot = next(next_slot_iter)
    except StopIteration:
        return

    for i, uid in enumerate(candidates):
        print(f"[download] candidate {i+1}/{len(candidates)}  slot={target_slot:02d}  uid={uid[:12]}", flush=True)
        src = fetch_one(uid)
        if src is None:
            continue
        dst = meshes_dir / f"obj_{target_slot:02d}.glb"
        try:
            shutil.copyfile(src, dst)
        except Exception as e:
            print(f"    copy failed: {e}")
            continue
        size_mb = dst.stat().st_size / 1e6
        print(f"    ok -> {dst.name}  ({size_mb:.1f} MB)")

        try:
            target_slot = next(next_slot_iter)
        except StopIteration:
            print(f"[download] all slots filled")
            break

    final_existing = sorted({int(p.stem.split("_")[-1])
                             for p in meshes_dir.glob("obj_*.glb")
                             if p.stem.split("_")[-1].isdigit()})
    print(f"[download] done. filled slots: {final_existing[:5]}..{final_existing[-5:]}  total={len(final_existing)}")


# ---------------------------------------------------------------------------
# Stage helpers: run an existing script with num_objects patched
# ---------------------------------------------------------------------------

def run_script(script_name: str):
    path = SCRIPTS / script_name
    log_header(f"STAGE: {script_name}")
    CFG.data.num_objects = TARGET_N
    t0 = time.time()
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"[orchestrator] {script_name} exited with code {e.code}")
            raise
    dt = time.time() - t0
    print(f"[orchestrator] {script_name} finished in {dt/60:.1f} minutes")


# ---------------------------------------------------------------------------
# Anchor guard: refuse to run if it would overwrite anchors
# ---------------------------------------------------------------------------

def guard_anchors():
    """Verify anchors exist. If they do, we must never run the anchor-training scripts."""
    anchors = [
        PROJECT / "checkpoints" / "anchor_image_siren.pt",
        PROJECT / "checkpoints" / "anchor_shape_siren.pt",
        PROJECT / "data" / "checkpoints" / "anchor_hypernet.pt",
    ]
    for a in anchors:
        if not a.exists():
            raise RuntimeError(f"anchor missing: {a}. Aborting to avoid retraining anchors with new shapes.")
        print(f"[guard] anchor present: {a}")


# ---------------------------------------------------------------------------
# Manifest for the N=100 mapper training
# ---------------------------------------------------------------------------

def build_manifest():
    log_header("STAGE: build manifest_n100.pt")
    import re
    import torch

    hyp_dir = PROJECT / "data" / "hypernets"
    shp_dir = PROJECT / "data" / "shape_sirens"

    key_re = re.compile(r"obj_(\d+)")
    hyp_files, shp_files = {}, {}
    for p in sorted(hyp_dir.glob("obj_*.pt")):
        m = key_re.search(p.name)
        if m: hyp_files[m.group(1)] = p
    for p in sorted(shp_dir.glob("obj_*.pt")):
        m = key_re.search(p.name)
        if m: shp_files[m.group(1)] = p

    keys = sorted(set(hyp_files) & set(shp_files), key=lambda k: int(k))
    if not keys:
        print("[manifest] no paired files found!")
        return None

    hyp_paths = [hyp_files[k] for k in keys]
    shp_paths = [shp_files[k] for k in keys]

    print(f"[manifest] paired {len(keys)} objects")
    print(f"[manifest] first: {keys[0]}   last: {keys[-1]}")

    out = SCRIPTS / "manifest_n100.pt"
    torch.save({"hypernet_paths": hyp_paths, "shape_paths": shp_paths}, out)
    print(f"[manifest] saved -> {out}")
    return out


# ---------------------------------------------------------------------------
# Train the mapper on all N pairs
# ---------------------------------------------------------------------------

def train_mapper(manifest_path, out_dir):
    log_header("STAGE: train mapper on N=100 pairs")
    cmd = [
        sys.executable,
        str(SCRIPTS / "hypernet_to_shape_mapper.py"),
        "--manifest", str(manifest_path),
        "--anchor_hyp", str(PROJECT / "data" / "checkpoints" / "anchor_hypernet.pt"),
        "--anchor_shp", str(PROJECT / "checkpoints" / "anchor_shape_siren.pt"),
        "--out", str(out_dir),
        "--lr", "1e-3",
        "--wd", "0",
        "--grad_clip", "0",
        "--steps", "5000",
    ]
    print("[orchestrator] launching mapper train:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(PROJECT / "data" / "scale_n100.log"))
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_mapper", action="store_true",
                    help="just build/run data pipeline; don't train mapper")
    args = ap.parse_args()

    # Tee stdout to log file
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_path)

    log_header(f"ORCHESTRATOR START  target N={TARGET_N}")
    print(f"[orchestrator] log file: {log_path}")
    print(f"[orchestrator] disk: {subprocess.check_output(['df','-h',str(PROJECT)]).decode()}")

    guard_anchors()

    # Patch num_objects so every downstream script sees N=100
    CFG.data.num_objects = TARGET_N
    print(f"[orchestrator] CFG.data.num_objects = {CFG.data.num_objects}")

    # --- Stage 1: download new shapes ---
    if not args.skip_download:
        download_stage()

    # --- Stage 2: watertight ---
    run_script("01_watertight.py")

    # --- Stage 3: render views ---
    run_script("03_render_views.py")

    # --- Stage 4: SDF sampling ---
    run_script("60_sample_sdf.py")

    # --- Stage 5: image SIRENs (this is the slow one: N * 24 trainings) ---
    run_script("20_train_image_sirens.py")

    # --- Stage 6: hypernets ---
    run_script("50_train_hypernets.py")

    # --- Stage 7: shape SIRENs ---
    run_script("80_train_shape_sirens.py")

    # --- Stage 8: manifest + mapper ---
    manifest_path = build_manifest()

    if not args.skip_mapper and manifest_path is not None:
        mapper_out = SCRIPTS / "t1_n100"
        train_mapper(manifest_path, mapper_out)

    log_header("ORCHESTRATOR DONE")


if __name__ == "__main__":
    main()
