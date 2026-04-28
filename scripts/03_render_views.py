"""Render 10 fixed views per watertight mesh."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.render import render_views


def main():
    CFG.data.views_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(CFG.data.watertight_dir.glob("*.obj")):
        out = CFG.data.views_dir / p.stem
        if out.exists() and len(list(out.glob("view_*.png"))) == CFG.data.num_views:
            print(f"[render] skip {p.stem}")
            continue
        print(f"[render] {p.stem}")
        render_views(p, out, num_views=CFG.data.num_views, img_res=CFG.data.img_res)


if __name__ == "__main__":
    main()
