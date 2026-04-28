"""Conditional Rectified Flow on shape-SIREN weight space.

The flow model learns to denoise shape-SIREN weight vectors,
conditioned on hypernetwork weights (which encode 24 views of the object).

Architecture:
    - Shape-SIREN weights tokenized by layer (each SIREN layer = 1 token)
    - Hypernetwork condition: per-head chunked encoder → 12 condition tokens
    - Transformer with self-attention on shape tokens + cross-attention to condition
    - Rectified flow: straight-line interpolation between noise and clean weights

Training:
    t ~ U(0,1)
    x_t = (1-t) * noise + t * clean_weights
    loss = MSE(model(x_t, t, condition), clean_weights - noise)

Inference:
    Start from noise, integrate flow for N steps → clean shape-SIREN weights
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN, flatten_weights
from src.hypernet import HyperNet


# ─── Condition Encoder (hypernetwork → tokens) ────────────────────────

class ConditionEncoder(nn.Module):
    """Encode each hypernetwork head into a d_model token via chunking + mean-pool."""
    def __init__(self, n_heads: int, chunk_size: int = 512,
                 hidden: int = 256, d_model: int = 256):
        super().__init__()
        self.chunk_size = chunk_size
        self.proj = nn.Sequential(
            nn.Linear(chunk_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.head_bias = nn.Parameter(torch.randn(n_heads, d_model) * 0.01)

    def forward(self, head_tensors: list[torch.Tensor]) -> torch.Tensor:
        """head_tensors: list of (B, head_size_i) → (B, n_heads, d_model)"""
        tokens = []
        for i, ht in enumerate(head_tensors):
            B, n = ht.shape
            pad_n = math.ceil(n / self.chunk_size) * self.chunk_size
            padded = F.pad(ht, (0, pad_n - n))
            chunks = padded.view(B, -1, self.chunk_size)
            encoded = self.proj(chunks).mean(dim=1)  # (B, d_model)
            tokens.append(encoded + self.head_bias[i])
        return torch.stack(tokens, dim=1)  # (B, n_heads, d_model)


# ─── Flow Transformer ─────────────────────────────────────────────────

class FlowTransformerBlock(nn.Module):
    """Self-attention on shape tokens + cross-attention to condition tokens."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # cross-attention to condition
        x = x + self.cross_attn(self.norm2(x), cond, cond)[0]
        # feedforward
        x = x + self.ff(self.norm3(x))
        return x


class RectifiedFlowTransformer(nn.Module):
    """Predicts the flow velocity v = (clean - noise) given noisy weights x_t,
    time t, and condition tokens from hypernetwork."""
    def __init__(
        self,
        siren_layer_sizes: list[int],  # param count per SIREN layer
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_ff: int = 512,
    ):
        super().__init__()
        self.siren_layer_sizes = siren_layer_sizes
        self.n_tokens = len(siren_layer_sizes)
        self.d_model = d_model

        # per-layer projection: each SIREN layer's weights → d_model
        self.in_projs = nn.ModuleList([
            nn.Linear(sz, d_model) for sz in siren_layer_sizes
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(d_model, sz) for sz in siren_layer_sizes
        ])

        # positional embedding for shape tokens
        self.pos_embed = nn.Parameter(torch.randn(self.n_tokens, d_model) * 0.02)

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # transformer blocks
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(d_model, nhead, dim_ff)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x_t_layers: list[torch.Tensor], t: torch.Tensor,
                cond: torch.Tensor) -> list[torch.Tensor]:
        """
        x_t_layers: list of (B, layer_size_i) — noisy SIREN layer weights
        t: (B, 1) — time step
        cond: (B, n_cond_tokens, d_model) — condition from hypernetwork
        Returns: list of (B, layer_size_i) — predicted velocity per layer
        """
        B = t.shape[0]

        # tokenize each SIREN layer
        tokens = []
        for i, (proj, x_layer) in enumerate(zip(self.in_projs, x_t_layers)):
            tokens.append(proj(x_layer))
        x = torch.stack(tokens, dim=1)  # (B, n_tokens, d_model)

        # add positional embedding
        x = x + self.pos_embed.unsqueeze(0)

        # add time embedding (broadcast to all tokens)
        t_emb = self.time_mlp(t)  # (B, d_model)
        x = x + t_emb.unsqueeze(1)

        # transformer blocks with cross-attention to condition
        for block in self.blocks:
            x = block(x, cond)

        x = self.final_norm(x)

        # project back to per-layer weight sizes
        velocities = []
        for i, proj in enumerate(self.out_projs):
            velocities.append(proj(x[:, i]))

        return velocities

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Data Loading ─────────────────────────────────────────────────────

def get_siren_layer_sizes(siren: SIREN) -> list[int]:
    """Get the parameter count per named-parameter group (each weight/bias)."""
    sizes = []
    for name, p in siren.named_parameters():
        sizes.append(p.numel())
    return sizes


def flatten_per_layer(siren: SIREN) -> list[torch.Tensor]:
    """Return each parameter as a flat tensor."""
    return [p.data.reshape(-1) for p in siren.parameters()]


def extract_head_params(hypernet: HyperNet) -> list[torch.Tensor]:
    heads = []
    for name, head in hypernet.heads.items():
        flat = torch.cat([p.data.reshape(-1) for p in head.parameters()])
        heads.append(flat)
    return heads


def load_all_data(device):
    d = CFG.data
    c = CFG.img_siren
    hc = CFG.hypernet
    sc = CFG.shape_siren

    ref_img_siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                          c.w0_first, c.w0_hidden)

    # load shape SIRENs per-layer
    ref_shape_siren = SIREN(sc.in_dim, sc.out_dim, sc.hidden_dim, sc.num_layers,
                            sc.w0_first, sc.w0_hidden)
    layer_sizes = get_siren_layer_sizes(ref_shape_siren)

    all_shape_layers = []  # list of 10 lists, each inner list has N_layers tensors
    all_hypernet_heads = []  # list of 10 lists

    for obj_i in range(d.num_objects):
        # shape SIREN
        ss = SIREN(sc.in_dim, sc.out_dim, sc.hidden_dim, sc.num_layers,
                   sc.w0_first, sc.w0_hidden)
        ss.load_state_dict(torch.load(
            sc.out_dir / f"obj_{obj_i:02d}.pt", map_location="cpu", weights_only=True
        ))
        all_shape_layers.append(flatten_per_layer(ss))

        # hypernetwork
        hp = HyperNet(
            target_siren=ref_img_siren, in_dim=3,
            head_hidden=hc.head_hidden, head_layers=hc.head_layers,
            final_init_scale=hc.final_init_scale,
        )
        hp.load_state_dict(torch.load(
            hc.out_dir / f"obj_{obj_i:02d}.pt", map_location="cpu", weights_only=True
        ))
        all_hypernet_heads.append(extract_head_params(hp))

    # batch shape layers: for each layer index, stack across objects
    n_layers = len(layer_sizes)
    batched_shape = []
    for l in range(n_layers):
        stacked = torch.stack([all_shape_layers[obj][l] for obj in range(d.num_objects)]).to(device)
        batched_shape.append(stacked)

    # batch hypernet heads
    n_heads = len(all_hypernet_heads[0])
    batched_heads = []
    for h in range(n_heads):
        stacked = torch.stack([all_hypernet_heads[obj][h] for obj in range(d.num_objects)]).to(device)
        batched_heads.append(stacked)

    return batched_shape, batched_heads, layer_sizes, n_heads


# ─── Training ─────────────────────────────────────────────────────────

def main():
    dev = CFG.device
    tc = CFG.transformer
    d = CFG.data

    print("[flow] loading all data...")
    batched_shape, batched_heads, layer_sizes, n_heads = load_all_data(dev)

    print(f"[flow] shape-SIREN: {len(layer_sizes)} layers, sizes: {layer_sizes}")
    print(f"[flow] total shape params: {sum(layer_sizes):,}")
    print(f"[flow] hypernetwork: {n_heads} heads")

    # normalize shape layers
    shape_stats = []
    batched_shape_norm = []
    for bs in batched_shape:
        sm, ss = bs.mean(), bs.std()
        shape_stats.append((sm, ss))
        batched_shape_norm.append((bs - sm) / (ss + 1e-8))

    # normalize hypernet heads
    head_stats = []
    batched_heads_norm = []
    for bh in batched_heads:
        hm, hs = bh.mean(), bh.std()
        head_stats.append((hm, hs))
        batched_heads_norm.append((bh - hm) / (hs + 1e-8))

    # build condition encoder
    cond_encoder = ConditionEncoder(
        n_heads=n_heads, chunk_size=512,
        hidden=256, d_model=tc.d_model,
    ).to(dev)

    # build flow transformer
    flow_model = RectifiedFlowTransformer(
        siren_layer_sizes=layer_sizes,
        d_model=tc.d_model,
        nhead=tc.nhead,
        num_layers=tc.num_encoder_layers,
        dim_ff=tc.dim_feedforward,
    ).to(dev)

    cond_params = sum(p.numel() for p in cond_encoder.parameters())
    flow_params = flow_model.num_params()
    print(f"[flow] condition encoder params: {cond_params:,}")
    print(f"[flow] flow transformer params: {flow_params:,}")
    print(f"[flow] total trainable: {cond_params + flow_params:,}")

    all_params = list(cond_encoder.parameters()) + list(flow_model.parameters())
    opt = torch.optim.Adam(all_params, lr=tc.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.steps, eta_min=1e-6)

    N = d.num_objects
    print(f"[flow] training for {tc.steps} steps on {N} objects")
    best_loss = float("inf")
    best_state = None

    for step in range(tc.steps):
        # sample random t ~ U(0,1)
        t = torch.rand(N, 1, device=dev)

        # sample noise (same structure as shape weights)
        noise_layers = [torch.randn_like(bs) for bs in batched_shape_norm]

        # interpolate: x_t = (1-t) * noise + t * clean
        x_t_layers = []
        for noise, clean in zip(noise_layers, batched_shape_norm):
            x_t = (1 - t) * noise + t * clean
            x_t_layers.append(x_t)

        # target velocity: v = clean - noise
        v_target = [clean - noise for clean, noise in zip(batched_shape_norm, noise_layers)]

        # condition: encode hypernetwork
        cond = cond_encoder(batched_heads_norm)  # (N, n_heads, d_model)

        # predict velocity
        v_pred = flow_model(x_t_layers, t, cond)

        # loss: MSE on velocity per layer, averaged
        loss = sum(F.mse_loss(vp, vt) for vp, vt in zip(v_pred, v_target)) / len(layer_sizes)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                "cond_encoder": {k: v.clone() for k, v in cond_encoder.state_dict().items()},
                "flow_model": {k: v.clone() for k, v in flow_model.state_dict().items()},
            }

        if step % 500 == 0 or step == tc.steps - 1:
            lr = scheduler.get_last_lr()[0]
            print(f"  step {step:5d}  loss {loss.item():.6f}  best {best_loss:.6f}  lr {lr:.2e}")

    # save
    tc.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **best_state,
        "layer_sizes": layer_sizes,
        "shape_stats": shape_stats,
        "head_stats": head_stats,
        "n_heads": n_heads,
    }, tc.ckpt_path)
    print(f"\n[flow] saved -> {tc.ckpt_path}")


if __name__ == "__main__":
    main()
