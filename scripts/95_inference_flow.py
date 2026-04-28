"""Inference: Conditional rectified flow sampling.

For each object:
    1. Load its hypernetwork → encode to condition tokens
    2. Start from random noise in shape-SIREN weight space
    3. Integrate flow for N steps (Euler method)
    4. Unnormalize → load into SIREN → marching cubes → mesh
"""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CFG
from src.siren import SIREN, unflatten_weights
from src.hypernet import HyperNet


# ─── Model classes (same as training) ─────────────────────────────────

class ConditionEncoder(nn.Module):
    def __init__(self, n_heads, chunk_size=512, hidden=256, d_model=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.proj = nn.Sequential(
            nn.Linear(chunk_size, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.head_bias = nn.Parameter(torch.randn(n_heads, d_model) * 0.01)

    def forward(self, head_tensors):
        tokens = []
        for i, ht in enumerate(head_tensors):
            B, n = ht.shape
            pad_n = math.ceil(n / self.chunk_size) * self.chunk_size
            padded = F.pad(ht, (0, pad_n - n))
            chunks = padded.view(B, -1, self.chunk_size)
            encoded = self.proj(chunks).mean(dim=1)
            tokens.append(encoded + self.head_bias[i])
        return torch.stack(tokens, dim=1)


class FlowTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, cond):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.cross_attn(self.norm2(x), cond, cond)[0]
        x = x + self.ff(self.norm3(x))
        return x


class RectifiedFlowTransformer(nn.Module):
    def __init__(self, siren_layer_sizes, d_model=256, nhead=4, num_layers=6, dim_ff=512):
        super().__init__()
        self.siren_layer_sizes = siren_layer_sizes
        self.n_tokens = len(siren_layer_sizes)
        self.d_model = d_model
        self.in_projs = nn.ModuleList([nn.Linear(sz, d_model) for sz in siren_layer_sizes])
        self.out_projs = nn.ModuleList([nn.Linear(d_model, sz) for sz in siren_layer_sizes])
        self.pos_embed = nn.Parameter(torch.randn(self.n_tokens, d_model) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([FlowTransformerBlock(d_model, nhead, dim_ff) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x_t_layers, t, cond):
        B = t.shape[0]
        tokens = [proj(xl) for proj, xl in zip(self.in_projs, x_t_layers)]
        x = torch.stack(tokens, dim=1)
        x = x + self.pos_embed.unsqueeze(0) + self.time_mlp(t).unsqueeze(1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x)
        return [proj(x[:, i]) for i, proj in enumerate(self.out_projs)]


# ─── Mesh extraction ──────────────────────────────────────────────────

def extract_mesh(siren, resolution, device):
    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    xs, ys, zs = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.from_numpy(np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)).to(device)
    chunk = 100_000
    sdf = np.empty(pts.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            sdf[i:i + chunk] = siren(pts[i:i + chunk]).squeeze(-1).cpu().numpy()
    sdf = sdf.reshape(resolution, resolution, resolution)
    try:
        verts, faces, normals, _ = measure.marching_cubes(sdf, level=0.0)
        verts = verts / (resolution - 1) * 2.0 - 1.0
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    except:
        return None


# ─── Flow sampling (Euler integration) ────────────────────────────────

@torch.no_grad()
def sample_flow(flow_model, cond, layer_sizes, shape_stats, device,
                n_steps=50):
    """Euler integration from t=0 (noise) to t=1 (clean)."""
    # start from random noise (normalized scale)
    x_layers = [torch.randn(1, sz, device=device) for sz in layer_sizes]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.tensor([[step / n_steps]], device=device)
        v_pred = flow_model(x_layers, t, cond)
        x_layers = [x + v * dt for x, v in zip(x_layers, v_pred)]

    # unnormalize
    clean_layers = []
    for x, (sm, ss) in zip(x_layers, shape_stats):
        clean_layers.append(x * (ss + 1e-8) + sm)

    return clean_layers


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    dev = CFG.device
    tc = CFG.transformer
    c = CFG.img_siren
    hc = CFG.hypernet
    sc = CFG.shape_siren
    d = CFG.data

    ckpt = torch.load(tc.ckpt_path, map_location=dev, weights_only=False)
    layer_sizes = ckpt["layer_sizes"]
    shape_stats = ckpt["shape_stats"]
    head_stats = ckpt["head_stats"]
    n_heads = ckpt["n_heads"]

    cond_encoder = ConditionEncoder(n_heads, chunk_size=512, hidden=256, d_model=tc.d_model).to(dev)
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    cond_encoder.eval()

    flow_model = RectifiedFlowTransformer(
        siren_layer_sizes=layer_sizes, d_model=tc.d_model, nhead=tc.nhead,
        num_layers=tc.num_encoder_layers, dim_ff=tc.dim_feedforward,
    ).to(dev)
    flow_model.load_state_dict(ckpt["flow_model"])
    flow_model.eval()

    print(f"[inference] loaded flow model + condition encoder")

    ref_img_siren = SIREN(c.in_dim, c.out_dim, c.hidden_dim, c.num_layers,
                          c.w0_first, c.w0_hidden)

    out_dir = Path(d.views_dir).parent / "flow_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Reconstruct all 10 objects (conditioned on their hypernetwork) ===
    print("\n=== Conditional reconstruction ===")
    for obj_i in range(d.num_objects):
        print(f"[inference] obj_{obj_i:02d}")

        # load hypernetwork
        hp = HyperNet(
            target_siren=ref_img_siren, in_dim=3,
            head_hidden=hc.head_hidden, head_layers=hc.head_layers,
            final_init_scale=hc.final_init_scale,
        )
        hp.load_state_dict(torch.load(
            hc.out_dir / f"obj_{obj_i:02d}.pt", map_location="cpu", weights_only=True
        ))

        # extract and normalize heads
        head_tensors = []
        for name, head in hp.heads.items():
            flat = torch.cat([p.data.reshape(-1) for p in head.parameters()])
            head_tensors.append(flat.unsqueeze(0).to(dev))

        head_tensors_norm = [
            (ht - hm.to(dev)) / (hs.to(dev) + 1e-8)
            for ht, (hm, hs) in zip(head_tensors, head_stats)
        ]

        # encode condition
        cond = cond_encoder(head_tensors_norm)

        # sample via flow
        pred_layers = sample_flow(flow_model, cond, layer_sizes, shape_stats, dev, n_steps=50)

        # assemble into SIREN
        flat = torch.cat([pl.squeeze(0) for pl in pred_layers])
        siren = SIREN(sc.in_dim, sc.out_dim, sc.hidden_dim, sc.num_layers,
                      sc.w0_first, sc.w0_hidden).to(dev)
        unflatten_weights(siren, flat)
        siren.eval()

        # extract mesh
        mesh = extract_mesh(siren, 256, dev)
        if mesh is not None:
            path = out_dir / f"obj_{obj_i:02d}_pred.obj"
            mesh.export(path)
            print(f"  saved {path}  verts={len(mesh.vertices)}")
        else:
            print(f"  FAILED")

        # copy GT
        gt = sc.out_dir / f"obj_{obj_i:02d}.obj"
        if gt.exists():
            shutil.copy(gt, out_dir / f"obj_{obj_i:02d}_gt.obj")

    print(f"\n[inference] done — check {out_dir}")


if __name__ == "__main__":
    main()
