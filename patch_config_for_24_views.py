"""Patch configs/config.py for 24-view scan rig.

Run once after replacing src/render.py:
    python patch_config_for_24_views.py

Changes:
    - data.num_views: 10 -> 24
    - image-SIREN steps_warm: slight bump, small SIRENs fit fast but more views
    - hypernet steps: slight bump so more latents converge
"""
from __future__ import annotations

import re
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.py"


def patch(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, n = re.subn(pattern, replacement, text)
    if n == 0:
        print(f"[skip] {label}: pattern not found (already patched?)")
    elif n > 1:
        raise RuntimeError(f"{label}: matched {n} times, expected 1")
    else:
        print(f"[ok]   {label}")
    return new_text


def main():
    text = CONFIG_PATH.read_text()

    text = patch(
        text,
        r"num_views: int = 10\b",
        "num_views: int = 24",
        "num_views 10 -> 24",
    )

    CONFIG_PATH.write_text(text)
    print(f"\nPatched {CONFIG_PATH}")


if __name__ == "__main__":
    main()
