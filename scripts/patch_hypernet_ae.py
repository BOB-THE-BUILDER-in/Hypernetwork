"""
Patch for hypernet_ae_pipeline.py to stabilize training.

Three changes:
1. Gradient accumulation (4 sub-batches of 8 = effective batch=32, reduces gradient variance)
2. Tighter gradient clipping (0.1 instead of 1.0)
3. Linear warmup over first 200 steps (gradients normalize before full lr kicks in)

Plus an --accum_steps argument to control accumulation count.
"""
from pathlib import Path

src = Path("/workspace/hypernet/scripts/hypernet_ae_pipeline.py")
text = src.read_text()

if not (src.with_suffix(".py.bak").exists()):
    src.with_suffix(".py.bak").write_text(text)
    print("[backup] saved .py.bak")
else:
    print("[backup] .py.bak exists, leaving")

# ---- Change 1: replace the training loop with grad accumulation + warmup
OLD = '''    print(f"[hyp-ae] mini-batch on N={N}, bs={bs}, {args.steps} steps, lr={args.lr}")

    import random as _random
    rng = _random.Random(0)
    for step in range(1, args.steps + 1):
        model.train()
        idx = rng.sample(range(N), bs)
        x = H_norm[idx].to(device, non_blocking=True)

        recon = model(x)
        loss = F.mse_loss(recon, x)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {loss.item():.4e}  lr {sched.get_last_lr()[0]:.2e}")'''

NEW = '''    accum = args.accum_steps
    eff_bs = bs * accum
    warmup = args.warmup_steps
    print(f"[hyp-ae] mini-batch on N={N}, bs={bs}, accum={accum}, eff_bs={eff_bs}, "
          f"{args.steps} steps, lr={args.lr}, warmup={warmup}")

    import random as _random
    rng = _random.Random(0)
    for step in range(1, args.steps + 1):
        model.train()
        # Linear warmup on top of cosine schedule
        if step <= warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step / warmup)
        else:
            sched.step()

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for k in range(accum):
            idx = rng.sample(range(N), bs)
            x = H_norm[idx].to(device, non_blocking=True)
            recon = model(x)
            loss = F.mse_loss(recon, x) / accum
            loss.backward()
            total_loss += loss.item() * accum  # un-scale for logging

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            print(f"  step {step:5d}  mse {total_loss/accum:.4e}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  gn {gn.item():.2e}")'''

if OLD in text:
    text = text.replace(OLD, NEW)
    print("[patch] replaced training loop")
else:
    print("[patch] training loop pattern NOT FOUND — did the file change?")

# ---- Change 2: add new argparse args
OLD2 = '''    p.add_argument("--device", default="cuda")
    return p.parse_args()'''
NEW2 = '''    p.add_argument("--accum_steps", type=int, default=4,
                   help="gradient accumulation: effective batch = batch_size * accum_steps")
    p.add_argument("--grad_clip", type=float, default=0.1,
                   help="tight clipping for stability with 2.3B params")
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="linear LR warmup over first N steps")
    p.add_argument("--device", default="cuda")
    return p.parse_args()'''

if OLD2 in text:
    text = text.replace(OLD2, NEW2)
    print("[patch] added accum/grad_clip/warmup args")
else:
    print("[patch] argparse pattern NOT FOUND — did the file change?")

src.write_text(text)
print("[patch] done. Re-run with same command.")
