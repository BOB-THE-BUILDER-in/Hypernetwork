"""Central config — edit here, don't scatter magic numbers.

Sizes chosen to keep RTX 3060 (12GB) comfortable.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"


@dataclass
class DataCfg:
    num_objects: int = 10
    num_views: int = 24
    img_res: int = 256
    sdf_surface_samples: int = 250_000
    sdf_volume_samples: int = 250_000
    sdf_truncation: float = 0.1

    meshes_dir: Path = DATA / "meshes"
    watertight_dir: Path = DATA / "watertight"
    views_dir: Path = DATA / "views"
    sdf_dir: Path = DATA / "sdf_samples"


@dataclass
class ImageSirenCfg:
    in_dim: int = 2
    out_dim: int = 3
    hidden_dim: int = 128
    num_layers: int = 5
    w0_first: float = 30.0
    w0_hidden: float = 30.0

    lr: float = 1e-4
    steps_anchor: int = 5_000
    steps_warm: int = 2_000
    batch_pixels: int = 16_384

    anchor_path: Path = CKPT / "anchor_image_siren.pt"
    out_dir: Path = DATA / "image_sirens"


@dataclass
class ShapeSirenCfg:
    in_dim: int = 3
    out_dim: int = 1
    hidden_dim: int = 256
    num_layers: int = 5
    w0_first: float = 30.0
    w0_hidden: float = 30.0

    lr: float = 1e-4
    steps_anchor: int = 5_000
    steps_warm: int = 1_000
    batch_points: int = 16_384

    truncation: float = 0.1

    anchor_path: Path = CKPT / "anchor_shape_siren.pt"
    out_dir: Path = DATA / "shape_sirens"


@dataclass
class HyperNetCfg:
    latent_dim: int = 64
    view_embed_dim: int = 32
    head_hidden: int = 64  # shrunken from 256 -> ~5M param hypernets
    head_layers: int = 3
    final_init_scale: float = 1e-2

    lr_hypernet: float = 1e-4
    lr_latents: float = 1e-3
    steps: int = 2_000
    l2_latent: float = 1e-4

    out_dir: Path = DATA / "hypernets"


@dataclass
class TransformerCfg:
    chunk_size: int = 512
    d_model: int = 256               # was 512 — smaller for 10 pairs
    nhead: int = 4                   # was 8
    num_encoder_layers: int = 4      # was 6
    num_decoder_layers: int = 4      # was 6
    dim_feedforward: int = 512       # was 2048
    dropout: float = 0.0

    lr: float = 3e-4
    steps: int = 10_000              # was 20000
    weight_decay: float = 0.0

    w_weight_mse: float = 1.0
    w_sdf_aux: float = 0.1

    ckpt_path: Path = CKPT / "transformer.pt"


@dataclass
class Cfg:
    data: DataCfg = field(default_factory=DataCfg)
    img_siren: ImageSirenCfg = field(default_factory=ImageSirenCfg)
    shape_siren: ShapeSirenCfg = field(default_factory=ShapeSirenCfg)
    hypernet: HyperNetCfg = field(default_factory=HyperNetCfg)
    transformer: TransformerCfg = field(default_factory=TransformerCfg)

    device: str = "cuda"
    seed: int = 42


CFG = Cfg()