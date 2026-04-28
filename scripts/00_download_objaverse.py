"""Download 10 Objaverse objects — one at a time, stop as soon as 10 succeed.

objaverse.load_objects() batches all downloads before returning, which means
one slow/hung file blocks everything. We bypass that by calling the per-uid
fetch directly with a timeout, and we stop the moment we have 10 good meshes.
"""
from __future__ import annotations

import random
import shutil
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG  # noqa: E402

import objaverse  # noqa: E402


DOWNLOAD_TIMEOUT_SEC = 60  # per-object timeout; raises TimeoutError if exceeded


class _Timeout:
    def __init__(self, sec: int):
        self.sec = sec

    def __enter__(self):
        def handler(signum, frame):
            raise TimeoutError("download timeout")
        self._prev = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.sec)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._prev)


def fetch_one(uid: str) -> Path | None:
    """Download a single uid with timeout. Returns local path or None on failure."""
    try:
        with _Timeout(DOWNLOAD_TIMEOUT_SEC):
            result = objaverse.load_objects(uids=[uid])
        src_str = result.get(uid)
        if not src_str:
            return None
        src = Path(src_str)
        return src if src.exists() else None
    except Exception as e:
        print(f"    skip {uid[:12]}: {type(e).__name__}")
        return None


def main():
    CFG.data.meshes_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] target: {CFG.data.meshes_dir}")

    all_uids = objaverse.load_uids()
    print(f"[download] {len(all_uids)} uids available")

    rng = random.Random(CFG.seed)
    candidates = rng.sample(all_uids, 400)  # big pool in case many fail

    chosen: list[Path] = []
    for i, uid in enumerate(candidates):
        if len(chosen) >= CFG.data.num_objects:
            break
        print(f"[download] ({len(chosen)+1}/{CFG.data.num_objects}) trying {uid[:12]}...", flush=True)
        src = fetch_one(uid)
        if src is None:
            continue
        dst = CFG.data.meshes_dir / f"obj_{len(chosen):02d}.glb"
        try:
            shutil.copyfile(src, dst)
        except Exception as e:
            print(f"    copy failed: {e}")
            continue
        print(f"    ok -> {dst.name}  ({dst.stat().st_size/1e6:.1f} MB)")
        chosen.append(dst)

    print(f"\n[download] saved {len(chosen)} meshes")
    if len(chosen) < CFG.data.num_objects:
        print("[download] WARNING: did not reach target. Re-run to try more uids.")


if __name__ == "__main__":
    main()