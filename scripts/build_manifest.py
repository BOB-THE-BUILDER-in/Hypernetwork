"""
Build the manifest .pt consumed by rectified_flow_transformer.py.

Scans --hypernet_dir and --shape_dir, pairs files by a shared stem pattern.
Default assumption (adjust --hyp_glob / --shape_glob if needed):

    hypernet_dir/hypernet_obj{k}.pt   <->   shape_dir/sdf_obj{k}.pt

Saves manifest.pt with:
    { 'hypernet_paths': [PosixPath, ...],
      'shape_paths':    [PosixPath, ...] }
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def extract_key(path: Path, pattern: re.Pattern) -> str | None:
    m = pattern.search(path.name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypernet_dir", required=True)
    ap.add_argument("--shape_dir", required=True)
    ap.add_argument("--hyp_glob", default="hypernet_obj*.pt")
    ap.add_argument("--shape_glob", default="sdf_obj*.pt")
    ap.add_argument("--key_regex", default=r"obj(\d+)",
                    help="regex with one capture group extracting the shared key")
    ap.add_argument("--out", default="manifest.pt")
    args = ap.parse_args()

    pattern = re.compile(args.key_regex)
    hyp_dir = Path(args.hypernet_dir)
    shp_dir = Path(args.shape_dir)

    hyp_files = {extract_key(p, pattern): p for p in sorted(hyp_dir.glob(args.hyp_glob))}
    shp_files = {extract_key(p, pattern): p for p in sorted(shp_dir.glob(args.shape_glob))}
    hyp_files.pop(None, None)
    shp_files.pop(None, None)

    keys = sorted(set(hyp_files) & set(shp_files), key=lambda k: int(k) if k.isdigit() else k)
    if not keys:
        raise RuntimeError("no paired files matched. Check --hyp_glob, --shape_glob, --key_regex.")

    missing_shp = sorted(set(hyp_files) - set(shp_files))
    missing_hyp = sorted(set(shp_files) - set(hyp_files))
    if missing_shp:
        print(f"[warn] hypernet has no shape pair for keys: {missing_shp}")
    if missing_hyp:
        print(f"[warn] shape has no hypernet pair for keys: {missing_hyp}")

    hyp_paths = [hyp_files[k] for k in keys]
    shp_paths = [shp_files[k] for k in keys]

    print(f"[manifest] paired {len(keys)} objects")
    for k, hp, sp in zip(keys, hyp_paths, shp_paths):
        print(f"  key={k}  hyp={hp.name}  shape={sp.name}")

    torch.save({"hypernet_paths": hyp_paths, "shape_paths": shp_paths}, args.out)
    print(f"[save] -> {args.out}")


if __name__ == "__main__":
    main()
