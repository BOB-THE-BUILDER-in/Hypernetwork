"""
Real fix for hypernet AE training.

Changes:
1. Switch to bitsandbytes 8-bit AdamW: saves ~14 GB optimizer memory
2. Bump hidden=64 -> hidden=256: enough capacity for 17.9M-dim compression
3. Loosen grad clip 0.1 -> 1.0 (no longer fighting OOM)
4. Keep gradient accumulation and warmup (still useful)

After this patch, default invocation should train without OOM and without
collapsing to mean-prediction.
"""
from pathlib import Path

src = Path("/workspace/hypernet/scripts/hypernet_ae_pipeline.py")
text = src.read_text()

# ---- Change 1: import bitsandbytes
OLD1 = '''import torch
import torch.nn as nn
import torch.nn.functional as F'''
NEW1 = '''import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("[warn] bitsandbytes not installed; falling back to torch.AdamW")'''

if OLD1 in text:
    text = text.replace(OLD1, NEW1)
    print("[patch] added bitsandbytes import")
else:
    print("[patch] import block NOT FOUND")

# ---- Change 2: replace optimizer construction
OLD2 = '''    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)'''

NEW2 = '''    if HAS_BNB and not args.no_8bit:
        print("[hyp-ae] using bitsandbytes 8-bit AdamW (saves ~14 GB optimizer state)")
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:
        print("[hyp-ae] using torch AdamW (full fp32 optimizer state)")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)'''

if OLD2 in text:
    text = text.replace(OLD2, NEW2)
    print("[patch] switched to 8-bit AdamW")
else:
    print("[patch] optimizer construction NOT FOUND")

# ---- Change 3: bump hidden default 64 -> 256
OLD3 = '''    p.add_argument("--hidden", type=int, default=64,
                   help="encoder hidden dim. 64 fits in 16GB; bump to 128 if reconstruction is poor")'''
NEW3 = '''    p.add_argument("--hidden", type=int, default=256,
                   help="encoder hidden dim. 256 needs 8-bit AdamW + 48GB GPU")'''

if OLD3 in text:
    text = text.replace(OLD3, NEW3)
    print("[patch] hidden default 64 -> 256")
else:
    print("[patch] hidden arg NOT FOUND")

# ---- Change 4: loosen grad clip default
OLD4 = '''    p.add_argument("--grad_clip", type=float, default=0.1,
                   help="tight clipping for stability with 2.3B params")'''
NEW4 = '''    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="standard clipping with bigger hidden (no longer fighting OOM)")'''

if OLD4 in text:
    text = text.replace(OLD4, NEW4)
    print("[patch] grad_clip default 0.1 -> 1.0")
else:
    print("[patch] grad_clip arg NOT FOUND")

# ---- Change 5: add --no_8bit flag for testing
OLD5 = '''    p.add_argument("--device", default="cuda")
    return p.parse_args()'''
NEW5 = '''    p.add_argument("--no_8bit", action="store_true",
                   help="disable bitsandbytes 8-bit Adam, use full fp32 optimizer")
    p.add_argument("--device", default="cuda")
    return p.parse_args()'''

if OLD5 in text:
    text = text.replace(OLD5, NEW5)
    print("[patch] added --no_8bit flag")
else:
    print("[patch] device arg NOT FOUND")

src.write_text(text)
print("[done] patch applied. Re-run with no flags (defaults are now correct).")
