"""
Final fix for hypernet AE: CPU-offloaded AdamW optimizer.

Replaces both torch.AdamW and bitsandbytes 8bit AdamW with a manual
implementation where optimizer state lives on CPU. This is the standard
fix for very large parameter counts.

Memory budget at h=128:
  GPU:
    Model fp32:    18.5 GB
    Gradients:     18.5 GB
    Activations:   ~3 GB
    Total:         ~40 GB  (fits in 48 GB)
  CPU:
    Optimizer m:   18.5 GB
    Optimizer v:   18.5 GB
    Total:         ~37 GB  (you have 750 GB RAM)

Each step copies gradients GPU->CPU, runs optimizer on CPU, copies updated
params CPU->GPU. Slower per step (~2x) but runs.
"""
from pathlib import Path

src = Path("/workspace/hypernet/scripts/hypernet_ae_pipeline.py")
text = src.read_text()


# ---- Add CPUOffloadedAdamW class right before HypernetAE
INSERT_CLASS = '''
class CPUOffloadedAdamW:
    """Manual AdamW where optimizer state (m, v) lives on CPU.

    Per-step flow:
        1. Read gradients from GPU model (already there from .backward())
        2. For each param: copy grad to CPU, update CPU-resident m/v, compute step
        3. Copy updated param back to GPU
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # CPU-resident optimizer state, one per param
        self.m = []
        self.v = []
        for p in self.params:
            self.m.append(torch.zeros_like(p, device="cpu", dtype=torch.float32))
            self.v.append(torch.zeros_like(p, device="cpu", dtype=torch.float32))

        # Group view for compatibility with LR schedulers
        self.param_groups = [{"params": self.params, "lr": lr}]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        bc1 = 1 - self.beta1 ** self.step_count
        bc2 = 1 - self.beta2 ** self.step_count
        lr = self.param_groups[0]["lr"]

        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            # Move grad to CPU for the update math
            g_cpu = p.grad.detach().to("cpu", non_blocking=True).float()

            # AdamW: weight decay applied directly to param
            if self.weight_decay > 0:
                p.data.mul_(1 - lr * self.weight_decay)

            # Update first and second moments on CPU
            m.mul_(self.beta1).add_(g_cpu, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g_cpu, g_cpu, value=1 - self.beta2)

            # Bias-corrected step: -lr * (m_hat) / (sqrt(v_hat) + eps)
            m_hat = m / bc1
            v_hat = v / bc2
            update = m_hat / (v_hat.sqrt() + self.eps)

            # Apply update on GPU
            p.data.add_(update.to(p.device, non_blocking=True), alpha=-lr)


'''

OLD_CLASS_HEADER = '''class HypernetAE(nn.Module):'''

if OLD_CLASS_HEADER in text and "class CPUOffloadedAdamW" not in text:
    text = text.replace(OLD_CLASS_HEADER, INSERT_CLASS + OLD_CLASS_HEADER)
    print("[patch] inserted CPUOffloadedAdamW class")
else:
    print("[patch] HypernetAE marker not found OR class already inserted")


# ---- Replace optimizer construction
OLD_OPT = '''    if HAS_BNB and not args.no_8bit:
        print("[hyp-ae] using bitsandbytes 8-bit AdamW (saves ~14 GB optimizer state)")
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:
        print("[hyp-ae] using torch AdamW (full fp32 optimizer state)")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)'''

NEW_OPT = '''    if args.use_cpu_optim:
        print("[hyp-ae] using CPU-offloaded AdamW (optimizer state on CPU)")
        opt = CPUOffloadedAdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    elif HAS_BNB and not args.no_8bit:
        print("[hyp-ae] using bitsandbytes 8-bit AdamW")
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:
        print("[hyp-ae] using torch AdamW (full fp32 optimizer state)")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Manual cosine schedule (don't use torch scheduler with custom optimizer)
    def get_lr(step):
        import math
        if step <= args.warmup_steps:
            return args.lr * (step / max(args.warmup_steps, 1))
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * progress))'''

if OLD_OPT in text:
    text = text.replace(OLD_OPT, NEW_OPT)
    print("[patch] switched to CPUOffloadedAdamW with manual scheduler")
else:
    print("[patch] optimizer block not found")


# ---- Replace the training loop's lr/sched logic
OLD_LOOP_LR = '''    for step in range(1, args.steps + 1):
        model.train()
        # Linear warmup on top of cosine schedule
        if step <= warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step / warmup)
        else:
            sched.step()'''

NEW_LOOP_LR = '''    for step in range(1, args.steps + 1):
        model.train()
        # Manual cosine + warmup
        cur_lr = get_lr(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr'''

if OLD_LOOP_LR in text:
    text = text.replace(OLD_LOOP_LR, NEW_LOOP_LR)
    print("[patch] replaced training loop LR logic")
else:
    print("[patch] training loop LR block not found")


# ---- Add --use_cpu_optim flag
OLD_FLAG = '''    p.add_argument("--no_8bit", action="store_true",
                   help="disable bitsandbytes 8-bit Adam, use full fp32 optimizer")'''

NEW_FLAG = '''    p.add_argument("--no_8bit", action="store_true",
                   help="disable bitsandbytes 8-bit Adam, use full fp32 optimizer")
    p.add_argument("--use_cpu_optim", action="store_true", default=True,
                   help="use CPU-offloaded AdamW (default; needed for h>=128)")'''

if OLD_FLAG in text:
    text = text.replace(OLD_FLAG, NEW_FLAG)
    print("[patch] added --use_cpu_optim flag (default True)")
else:
    print("[patch] no_8bit flag not found")


src.write_text(text)
print("\n[done] CPU-offloaded optimizer patch applied.")
print("Run with default settings:")
print("  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\")
print("    python hypernet_ae_pipeline.py --hidden 128")
