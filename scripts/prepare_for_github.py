"""
Prepare /workspace/hypernet/ for GitHub upload.

Creates a .gitignore that excludes data and large checkpoints, leaving only:
  - configs/        (config files)
  - src/            (core modules)
  - scripts/        (training and analysis scripts)
  - README.md       (a generated overview)

Does NOT run git itself. You do that yourself after reviewing what would be
tracked, so you can decide what (if anything) to LFS.

After this script:
    cd /workspace/hypernet
    git init
    git add .
    git status     # review what would be committed
    git commit -m "initial commit"
    git remote add origin <YOUR_REPO_URL>
    git push -u origin main
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("/workspace/hypernet")
GITIGNORE = ROOT / ".gitignore"
README = ROOT / "README.md"


GITIGNORE_CONTENT = """# Data files - these are large and reproducible from the scripts
data/

# Checkpoints - too large for git, often >500MB
checkpoints/

# Logs from training runs
*.log

# Python build artifacts
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Backup files left by patches
*.py.bak

# Environment / cache
.venv/
venv/
.cache/
.pytest_cache/

# Editor / OS
.vscode/
.idea/
.DS_Store

# Rendered views (huge, regenerable)
*.png
"""


README_CONTENT = """# hypernet → shape pipeline

Image-to-3D experiments via per-shape SIRENs and learned weight-space latent codes.

## Pipeline overview

```
24 images → 24 image-SIRENs → 1 hypernet (per object)

hypernet weights → tiny mapper → 128-dim latent → AE decoder → shape-SIREN weights → SDF → mesh
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  weight-space autoencoder rescues mesh quality
                                  vs predicting 264K weights directly
```

## Layout

- `configs/` — pipeline config (CFG.data, CFG.shape_siren, etc.)
- `src/` — core modules (siren, hypernet, render, watertight)
- `scripts/` — numbered pipeline stages plus experiment scripts:
  - `01_watertight.py` … `80_train_shape_sirens.py` — main pipeline
  - `hypernet_to_shape_mapper.py` — direct weight-prediction mapper (baseline)
  - `autoencoder_pipeline_n100_mlp.py` — latent autoencoder + tiny mapper (current)
  - `scale_to_n100.py` — orchestrator for scaling experiment
  - `ood_test_full.py` — OOD generalization test on a held-out shape

## Reproducing

1. Run `00_download_objaverse.py` … `80_train_shape_sirens.py` in order.
   Set `CFG.data.num_objects = 100` first if you want N=100.
2. `python scripts/autoencoder_pipeline_n100_mlp.py` to train the
   weight-space autoencoder and tiny latent mapper.
3. `python scripts/ood_test_full.py` to test generalization on a new
   Objaverse shape.

## Key findings (April 2026)

- **Direct weight prediction** at N=10 plateaus at MSE 0.143; at N=100
  reaches best-case 0.139 but interpolation between training points
  collapses, and OOD novel shapes produce floating fragments.
- **Weight-space autoencoder** (128-dim latent, MLP architecture) achieves
  reconstruction MSE 0.0026, near-perfect tiny-mapper (MSE 1.4e-5,
  scrambled-cond ratio 9M×), and OOD prediction produces coherent geometry
  rather than fragments.
- Conclusion: at small/medium N, direct weight prediction is too brittle
  due to SIREN sensitivity; learned latent compression rescues OOD
  generalization.
"""


def main():
    print(f"[prep] root: {ROOT}")
    if not ROOT.exists():
        print("ERROR: project not found")
        return

    # Write .gitignore
    GITIGNORE.write_text(GITIGNORE_CONTENT)
    print(f"[prep] wrote {GITIGNORE}")

    # Write README only if not present
    if not README.exists():
        README.write_text(README_CONTENT)
        print(f"[prep] wrote {README}")
    else:
        print(f"[prep] {README} already exists, leaving it")

    # Show what would be tracked vs ignored
    print("\n[prep] Files at top level of repo:")
    for p in sorted(ROOT.iterdir()):
        marker = "  IGNORED " if any(p.match(pat) for pat in [
            "data", "checkpoints", "*.log", "__pycache__", "*.py.bak"
        ]) else "  TRACKED "
        size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.is_dir() else p.stat().st_size / 1e6
        print(f"{marker} {p.name:30s}  {size_mb:>10.1f} MB")

    # Estimate tracked size
    total = 0
    for p in [ROOT / "configs", ROOT / "src", ROOT / "scripts"]:
        if p.exists():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    if README.exists():
        total += README.stat().st_size

    print(f"\n[prep] Estimated tracked size: {total/1e6:.2f} MB (well within GitHub limits)")
    print("\n[prep] Next steps:")
    print("  cd /workspace/hypernet")
    print("  git init")
    print("  git add .")
    print("  git status      # double-check what's staged")
    print("  git commit -m 'initial commit'")
    print("  git remote add origin <YOUR_GITHUB_REPO_URL>")
    print("  git branch -M main")
    print("  git push -u origin main")


if __name__ == "__main__":
    main()
