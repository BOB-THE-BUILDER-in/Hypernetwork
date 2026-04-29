"""Cross-modal pipeline with latent noise injection.

Same architecture as cross_modal_pipeline.py with L2 alignment.
The new piece: during training, inject Gaussian noise to z before
decoding so the decoders learn to be smooth in a neighborhood around
each training latent. Test-time inference: no noise.

Output dir: data/cross_modal_noise_n100/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path("/workspace/hypernet")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from configs.config import CFG
from src.siren import SIREN
from src.render import scan_rig_directions

# Reuse encoders/decoders from base pipeline
from cross_modal_pipeline import (
    EncoderImg, EncoderShp, DecoderImg, DecoderShp,
    flatten_sd, load_image_sirens, load_shape_sirens,
)

OUT_DIR = PROJECT / "data" / "cross_modal_noise_n100"
MESH_DIR = OUT_DIR / "meshes"


def train(args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] image-SIRENs...")
    W_img, img_keys = load_image_sirens(PROJECT / "data" / "image_sirens", device)
    img_siren_dim = W_img.shape[-1]

    print("[load] shape-SIRENs...")
    W_shp, shp_keys = load_shape_sirens(PROJECT / "data" / "shape_sirens", device)
    shp_siren_dim = W_shp.shape[-1]

    img_anchor = W_img.mean(dim=(0, 1))
    shp_anchor = W_shp.mean(dim=0)
    W_img_res = W_img - img_anchor[None, None, :]
    W_shp_res = W_shp - shp_anchor[None, :]
    img_std = W_img_res.reshape(-1, img_siren_dim).std(0).clamp_min(1e-6)
    shp_std = W_shp_res.std(0).clamp_min(1e-6)
    W_img_norm = W_img_res / img_std[None, None, :]
    W_shp_norm = W_shp_res / shp_std[None, :]
    print(f"[load] normalized variance: img={W_img_norm.var().item():.3f}  "
          f"shp={W_shp_norm.var().item():.3f}")

    cam_dirs = torch.from_numpy(scan_rig_directions()).float().to(device)

    enc_img = EncoderImg(img_siren_dim, z_dim=args.z_dim, hidden=args.enc_hidden).to(device)
    enc_shp = EncoderShp(shp_siren_dim, z_dim=args.z_dim, hidden=args.enc_hidden).to(device)
    dec_img = DecoderImg(args.z_dim, cam_dim=3, out_dim=img_siren_dim,
                         hidden=args.dec_hidden, num_layers=args.dec_layers).to(device)
    dec_shp = DecoderShp(args.z_dim, out_dim=shp_siren_dim,
                         hidden=args.dec_hidden, num_layers=args.dec_layers).to(device)

    n_params = sum(sum(p.numel() for p in m.parameters())
                   for m in [enc_img, enc_shp, dec_img, dec_shp])
    print(f"[arch] total params: {n_params:,}")

    opt = torch.optim.AdamW(
        list(enc_img.parameters()) + list(enc_shp.parameters())
        + list(dec_img.parameters()) + list(dec_shp.parameters()),
        lr=args.lr, weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    print(f"[train] {args.steps} steps  batch={args.batch_size}  noise_std={args.noise_std}")
    print(f"[train] losses: img={args.w_img:.2f} shp={args.w_shp:.2f} align={args.w_align:.4f}")

    import random as _rnd
    rng = _rnd.Random(0)

    for step in range(1, args.steps + 1):
        for m in [enc_img, enc_shp, dec_img, dec_shp]:
            m.train()

        idx = rng.sample(range(100), args.batch_size)
        bs = len(idx)

        x_img = W_img_norm[idx]
        x_shp = W_shp_norm[idx]

        z_img = enc_img(x_img)
        z_shp = enc_shp(x_shp)

        # Latent noise injection — scaled by current latent magnitude
        with torch.no_grad():
            z_img_norm_avg = z_img.norm(dim=-1, keepdim=True).mean()
            z_shp_norm_avg = z_shp.norm(dim=-1, keepdim=True).mean()
        noise_img = torch.randn_like(z_img) * args.noise_std * z_img_norm_avg / (args.z_dim ** 0.5)
        noise_shp = torch.randn_like(z_shp) * args.noise_std * z_shp_norm_avg / (args.z_dim ** 0.5)

        z_img_noisy = z_img + noise_img
        z_shp_noisy = z_shp + noise_shp

        # Image reconstruction with noisy z
        z_img_b = z_img_noisy.unsqueeze(1).expand(bs, 24, -1).reshape(bs * 24, -1)
        cam_b = cam_dirs.unsqueeze(0).expand(bs, 24, -1).reshape(bs * 24, -1)
        target_img = x_img.reshape(bs * 24, img_siren_dim)
        pred_img = dec_img(z_img_b, cam_b)
        L_img = F.mse_loss(pred_img, target_img)

        # Shape reconstruction with noisy z
        pred_shp = dec_shp(z_shp_noisy)
        L_shp = F.mse_loss(pred_shp, x_shp)

        # ALSO: dec_shp on z_img_noisy should produce shape (cross-modal robustness)
        pred_shp_from_img = dec_shp(z_img_noisy)
        L_shp_xm = F.mse_loss(pred_shp_from_img, x_shp)

        # ALSO: dec_img on z_shp_noisy should produce images
        z_shp_b = z_shp_noisy.unsqueeze(1).expand(bs, 24, -1).reshape(bs * 24, -1)
        pred_img_from_shp = dec_img(z_shp_b, cam_b)
        L_img_xm = F.mse_loss(pred_img_from_shp, target_img)

        # Alignment (no noise)
        L_align = F.mse_loss(z_img, z_shp)

        loss = (args.w_img * L_img + args.w_shp * L_shp
                + args.w_xm * (L_shp_xm + L_img_xm)
                + args.w_align * L_align)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        for m in [enc_img, enc_shp, dec_img, dec_shp]:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % max(1, args.steps // 100) == 0:
            with torch.no_grad():
                z_img_n = F.normalize(z_img, dim=-1)
                z_shp_n = F.normalize(z_shp, dim=-1)
                cos_sim = (z_img_n * z_shp_n).sum(-1).mean().item()
            print(f"  step {step:5d}  L_img {L_img.item():.3e}  L_shp {L_shp.item():.3e}  "
                  f"L_shp_xm {L_shp_xm.item():.3e}  L_img_xm {L_img_xm.item():.3e}  "
                  f"L_align {L_align.item():.3e}  cos {cos_sim:.3f}")

    torch.save({
        "enc_img": enc_img.state_dict(),
        "enc_shp": enc_shp.state_dict(),
        "dec_img": dec_img.state_dict(),
        "dec_shp": dec_shp.state_dict(),
        "args": vars(args),
        "img_keys": img_keys,
        "shp_keys": shp_keys,
        "img_anchor": img_anchor.cpu(),
        "shp_anchor": shp_anchor.cpu(),
        "img_std": img_std.cpu(),
        "shp_std": shp_std.cpu(),
        "img_siren_dim": img_siren_dim,
        "shp_siren_dim": shp_siren_dim,
        "cam_dirs": cam_dirs.cpu(),
    }, OUT_DIR / "cross_modal.pt")
    print(f"[save] -> {OUT_DIR/'cross_modal.pt'}")

    return enc_img, enc_shp, dec_img, dec_shp


def reconstruct_and_ood(enc_img, dec_shp, ckpt, device):
    print("\n[recon] training sanity check (no noise at inference)")
    img_anchor = ckpt["img_anchor"].to(device)
    img_std = ckpt["img_std"].to(device)
    shp_anchor = ckpt["shp_anchor"].cpu()
    shp_std = ckpt["shp_std"].cpu()
    img_keys = ckpt["img_keys"]

    enc_img.eval(); dec_shp.eval()
    sc = CFG.shape_siren

    def mesh_from_z(z, name):
        with torch.no_grad():
            recon_norm = dec_shp(z).cpu()
        recon_abs = recon_norm * shp_std[None, :] + shp_anchor[None, :]
        siren = SIREN(sc.in_dim, sc.out_dim, sc.hidden_dim, sc.num_layers,
                      sc.w0_first, sc.w0_hidden).to(device)
        sd = {}
        off = 0
        for k in ckpt["shp_keys"]:
            shape = siren.state_dict()[k].shape
            n = 1
            for dd in shape: n *= dd
            sd[k] = recon_abs[0, off:off+n].view(*shape).to(device)
            off += n
        siren.load_state_dict(sd); siren.eval()
        res, bound = 192, 1.0
        lin = torch.linspace(-bound, bound, res, device=device)
        xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
        out = torch.empty(pts.shape[0], device=device)
        with torch.no_grad():
            for i in range(0, pts.shape[0], 65536):
                s = siren(pts[i:i+65536])
                if s.dim() > 1: s = s.squeeze(-1)
                out[i:i+65536] = s
        vol = out.reshape(res, res, res).cpu().numpy()
        from skimage.measure import marching_cubes
        if not (vol.min() <= 0 <= vol.max()):
            print(f"  {name}: no zero crossing")
            return
        spacing = (2*bound/(res-1),)*3
        v, f, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
        v = v - bound
        out_path = MESH_DIR / f"{name}.obj"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            for vv in v: fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
            for tri in f: fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        print(f"  {name}: {v.shape[0]} v / {f.shape[0]} f")

    def encode_obj(image_sirens_dir):
        flats = []
        for j in range(CFG.data.num_views):
            sd = torch.load(image_sirens_dir / f"view_{j:02d}.pt",
                            map_location="cpu", weights_only=True)
            flat, _ = flatten_sd(sd, img_keys)
            flats.append(flat)
        sirens = torch.stack(flats).to(device)
        sirens_norm = ((sirens - img_anchor) / img_std).unsqueeze(0)
        with torch.no_grad():
            return enc_img(sirens_norm)

    # Training reconstructions
    for obj_idx in [0, 42, 67, 88]:
        z = encode_obj(PROJECT / "data" / "image_sirens" / f"obj_{obj_idx:02d}")
        mesh_from_z(z, f"train_recon_obj_{obj_idx:02d}")

    # OOD: obj_101
    z = encode_obj(PROJECT / "data" / "ood_test" / "obj_101" / "image_sirens")
    print(f"  obj_101 z norm: {z.norm().item():.2f}")
    mesh_from_z(z, "ood_obj_101")

    # OOD: 6 primitives
    primitive_names = {200: "sphere", 201: "cube", 202: "two_spheres",
                       203: "cylinder", 204: "torus", 205: "chair"}
    for idx, name in primitive_names.items():
        d = PROJECT / "data" / "ood_test" / f"obj_{idx:03d}" / "image_sirens"
        if not d.exists(): continue
        z = encode_obj(d)
        print(f"  obj_{idx:03d} ({name}) z norm: {z.norm().item():.2f}")
        mesh_from_z(z, f"primitive_obj_{idx:03d}_{name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--enc_hidden", type=int, default=512)
    p.add_argument("--dec_hidden", type=int, default=512)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise_std", type=float, default=0.3,
                   help="Noise injected to latents, as fraction of latent norm")
    p.add_argument("--w_img", type=float, default=1.0)
    p.add_argument("--w_shp", type=float, default=1.0)
    p.add_argument("--w_xm", type=float, default=0.5,
                   help="Weight on cross-modal reconstruction (z_img -> shape, z_shp -> images)")
    p.add_argument("--w_align", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    train(args, device)
    ckpt = torch.load(OUT_DIR / "cross_modal.pt", map_location=device, weights_only=False)
    enc_img = EncoderImg(ckpt["img_siren_dim"], z_dim=args.z_dim, hidden=args.enc_hidden).to(device)
    enc_img.load_state_dict(ckpt["enc_img"])
    dec_shp = DecoderShp(args.z_dim, out_dim=ckpt["shp_siren_dim"],
                         hidden=args.dec_hidden, num_layers=args.dec_layers).to(device)
    dec_shp.load_state_dict(ckpt["dec_shp"])
    reconstruct_and_ood(enc_img, dec_shp, ckpt, device)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
