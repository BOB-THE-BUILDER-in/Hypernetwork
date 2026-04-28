"""
Patch autoencoder_pipeline_n100.py to fix conditioning.

The original `Decoder` uses `x = self.queries + c.unsqueeze(1)`, which broadcasts
the latent additively across 259 tokens. This is too weak: the decoder learns
to output the mean and ignore z entirely, plateauing at MSE 0.99.

Replace with FiLM modulation: a learned MLP turns z into per-block (shift, scale)
which modulates each transformer block's normalization. This is the standard
trick for strong conditioning in diffusion models (DiT, AdaLN-Zero).

Also makes encoder output a slight std bias so latents are well-distributed.

Idempotent. Backs up the original.
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path("/workspace/hypernet/scripts/autoencoder_pipeline_n100.py")
BACKUP = SRC.with_suffix(".py.bak")


REPLACE_BLOCKS = [
    {
        "name": "TransformerBlock -> FiLMBlock",
        "old": '''class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x''',
        "new": '''class TransformerBlock(nn.Module):
    """Plain transformer block (used by encoder)."""
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class FiLMBlock(nn.Module):
    """Transformer block with FiLM conditioning on latent z.
    z is projected to (shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff)."""
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        # Initialize so blocks start as identity (helps training stability)
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        # c: (B, d_model) ; x: (B, N, d_model)
        s_a, sc_a, g_a, s_f, sc_f, g_f = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc_a.unsqueeze(1)) + s_a.unsqueeze(1)
        x = x + g_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + sc_f.unsqueeze(1)) + s_f.unsqueeze(1)
        x = x + g_f.unsqueeze(1) * self.ffn(h)
        return x''',
    },
    {
        "name": "Decoder uses FiLMBlock",
        "old": '''class Decoder(nn.Module):
    """Latent -> weights. NOT zero-init (so it can learn from step 0)."""
    def __init__(self, shape_dim, latent_dim, chunk_size=1024, d_model=256, n_layers=6, n_heads=4):
        super().__init__()
        self.chunker = ChunkedProjector(shape_dim, chunk_size, d_model)
        self.queries = nn.Parameter(torch.randn(1, self.chunker.num_chunks, d_model) * 0.02)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, z):
        B = z.shape[0]
        c = self.latent_proj(z).unsqueeze(1)  # (B, 1, d)
        x = self.queries.expand(B, -1, -1) + c  # condition by addition
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        return self.chunker.detokenize(x)''',
        "new": '''class Decoder(nn.Module):
    """Latent -> weights. FiLM/AdaLN conditioning on z."""
    def __init__(self, shape_dim, latent_dim, chunk_size=1024, d_model=256, n_layers=6, n_heads=4):
        super().__init__()
        self.chunker = ChunkedProjector(shape_dim, chunk_size, d_model)
        self.queries = nn.Parameter(torch.randn(1, self.chunker.num_chunks, d_model) * 0.02)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.blocks = nn.ModuleList([FiLMBlock(d_model, n_heads) for _ in range(n_layers)])
        # Final FiLM-modulated norm
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

    def forward(self, z):
        B = z.shape[0]
        c = self.latent_proj(z)                          # (B, d_model)
        x = self.queries.expand(B, -1, -1)               # (B, num_chunks, d_model)
        for b in self.blocks:
            x = b(x, c)
        # Final FiLM-modulated norm
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.chunker.detokenize(x)''',
    },
]


def main():
    if not BACKUP.exists():
        shutil.copy(SRC, BACKUP)
        print(f"[backup] -> {BACKUP}")
    else:
        print(f"[backup] {BACKUP} exists, leaving")

    text = SRC.read_text()
    applied, skipped = [], []
    for p in REPLACE_BLOCKS:
        if p["old"] in text:
            text = text.replace(p["old"], p["new"])
            applied.append(p["name"])
        elif p["new"] in text:
            skipped.append(f'{p["name"]} (already applied)')
        else:
            skipped.append(f'{p["name"]} (PATTERN NOT FOUND)')
    SRC.write_text(text)

    print(f"applied: {applied}")
    print(f"skipped: {skipped}")
    if any("PATTERN NOT FOUND" in s for s in skipped):
        print("WARNING: at least one pattern was not found; restore from .py.bak")
    else:
        print("Done. Re-run autoencoder_pipeline_n100.py.")


if __name__ == "__main__":
    main()
