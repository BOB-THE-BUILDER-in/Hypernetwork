"""
Patch hypernet_to_shape_mapper.py to fix CUDA OOM at N=100.

Edits:
  1. Removes .to(device) on the 100 x 17.9M hyp/shp stacks at load time
  2. Removes .to(device) on anchor tensors
  3. Makes reconstruct_shape_weights device-aware
  4. Replaces full-batch train loop with mini-batch sampling
  5. Adds --batch_size CLI arg (default 16)

Idempotent. Backs up original to .py.bak on first run.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SRC = Path("/workspace/hypernet/scripts/hypernet_to_shape_mapper.py")
BACKUP = SRC.with_suffix(".py.bak")


PATCHES = [
    {
        "name": "stacks-cpu",
        "old": '        hyp = torch.stack(hyp_flats).to(device)\n        shp = torch.stack(shp_flats).to(device)',
        "new": '        # Keep stacks on CPU; mini-batches moved to GPU per step.\n        hyp = torch.stack(hyp_flats)\n        shp = torch.stack(shp_flats)',
    },
    {
        "name": "anchors-cpu",
        "old": '        self.anchor_hyp = anchor_hyp.to(device)\n        self.anchor_shp = anchor_shp.to(device)',
        "new": '        # Anchors stay on CPU; reconstruct_shape_weights moves them on demand.\n        self.anchor_hyp = anchor_hyp\n        self.anchor_shp = anchor_shp',
    },
    {
        "name": "reconstruct-device-aware",
        "old": '    def reconstruct_shape_weights(self, x_norm):\n        return x_norm * self.shp_std + self.shp_mean + self.anchor_shp[None]',
        "new": (
            '    def reconstruct_shape_weights(self, x_norm):\n'
            '        dev = x_norm.device\n'
            '        return x_norm * self.shp_std.to(dev) + self.shp_mean.to(dev) + self.anchor_shp[None].to(dev)'
        ),
    },
    {
        "name": "train-loop-minibatch",
        "old": (
            '    hyp_all = ds.hyp_norm\n'
            '    shp_all = ds.shp_norm\n'
            '    print(f"[train] full-batch on N={len(ds.shp_norm)} for {args.steps} steps (direct MSE)")\n'
            '    for step in range(1, args.steps + 1):\n'
            '        model.train()\n'
            '        pred = model(hyp_all)\n'
            '        loss = F.mse_loss(pred, shp_all)\n'
            '        opt.zero_grad(set_to_none=True)\n'
            '        loss.backward()\n'
            '        if args.grad_clip > 0:\n'
            '            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)\n'
            '        opt.step()\n'
            '        sched.step()\n'
            '        if step % log_every == 0 or step == 1:\n'
            '            print(f"step {step:6d} / {args.steps}   loss={loss.item():.4e}   lr={sched.get_last_lr()[0]:.2e}")'
        ),
        "new": (
            '    hyp_all = ds.hyp_norm  # CPU\n'
            '    shp_all = ds.shp_norm  # CPU\n'
            '    N = hyp_all.shape[0]\n'
            '    bs = min(getattr(args, "batch_size", 16), N)\n'
            '    print(f"[train] mini-batch on N={N}  bs={bs}  for {args.steps} steps "\n'
            '          f"(~{args.steps * bs / N:.0f} passes/shape, direct MSE)")\n'
            '    import random as _random\n'
            '    _rng = _random.Random(0)\n'
            '    for step in range(1, args.steps + 1):\n'
            '        model.train()\n'
            '        idx = _rng.sample(range(N), bs)\n'
            '        hyp_b = hyp_all[idx].to(device, non_blocking=True)\n'
            '        shp_b = shp_all[idx].to(device, non_blocking=True)\n'
            '        pred = model(hyp_b)\n'
            '        loss = F.mse_loss(pred, shp_b)\n'
            '        opt.zero_grad(set_to_none=True)\n'
            '        loss.backward()\n'
            '        if args.grad_clip > 0:\n'
            '            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)\n'
            '        opt.step()\n'
            '        sched.step()\n'
            '        if step % log_every == 0 or step == 1:\n'
            '            print(f"step {step:6d} / {args.steps}   loss={loss.item():.4e}   lr={sched.get_last_lr()[0]:.2e}")'
        ),
    },
]


def add_batch_size_arg(text: str) -> str:
    if "--batch_size" in text:
        return text
    candidates = [
        '    p.add_argument("--steps"',
        "    p.add_argument('--steps'",
    ]
    for c in candidates:
        if c in text:
            insertion = '    p.add_argument("--batch_size", type=int, default=16, help="N=100 mini-batch size")\n'
            return text.replace(c, insertion + c, 1)
    print("[patch] WARNING: could not find --steps arg to anchor batch_size insertion. Add manually.")
    return text


def main():
    if not SRC.exists():
        print(f"ERROR: {SRC} not found")
        sys.exit(1)

    text = SRC.read_text()

    if not BACKUP.exists():
        shutil.copy(SRC, BACKUP)
        print(f"[backup] -> {BACKUP}")
    else:
        print(f"[backup] {BACKUP} already exists, leaving it")

    applied, skipped = [], []
    for p in PATCHES:
        if p["old"] in text:
            text = text.replace(p["old"], p["new"])
            applied.append(p["name"])
        elif p["new"] in text:
            skipped.append(f'{p["name"]} (already applied)')
        else:
            skipped.append(f'{p["name"]} (PATTERN NOT FOUND - file may have changed)')

    text = add_batch_size_arg(text)
    SRC.write_text(text)

    print("\n=== Patch summary ===")
    print(f"applied: {applied}")
    print(f"skipped: {skipped}")
    print()
    if any("PATTERN NOT FOUND" in s for s in skipped):
        print("WARNING: at least one pattern was not found. Restore from .py.bak and inspect manually.")
        sys.exit(2)
    print("Done. Now run:")
    print()
    print("  cd /workspace/hypernet/scripts")
    print("  python hypernet_to_shape_mapper.py \\")
    print("    --manifest manifest_n100.pt \\")
    print("    --anchor_hyp /workspace/hypernet/data/checkpoints/anchor_hypernet.pt \\")
    print("    --anchor_shp /workspace/hypernet/checkpoints/anchor_shape_siren.pt \\")
    print("    --out t1_n100 \\")
    print("    --lr 1e-3 --wd 0 --grad_clip 0 \\")
    print("    --batch_size 16 --steps 15000")


if __name__ == "__main__":
    main()
