"""
Diagnostic: find the minimum omega_0 needed to represent the shape-SIRENs'
actual frequency content.

Logic:
    1. For each of the 10 trained shape-SIRENs, query the SDF on a 3D grid
       (GRID_RES^3). This gives us the ground-truth SDF volume as a 3D tensor.
    2. Compute the 3D FFT. Take the magnitude, average over angular bins,
       giving a radial power spectrum indexed by frequency k.
    3. Find k95 and k99 — the frequencies that enclose 95% / 99% of spectral
       power. These are the highest frequencies we actually need to represent.
    4. Translate k into the omega_0 a SIREN needs. For a SIREN on the domain
       [-1, 1], the first layer produces activations at frequencies bounded
       roughly by omega_0 (since it does sin(omega_0 * W * x) with W small
       initial values). Deep layers can *combine* frequencies but cannot
       create frequencies exceeding roughly omega_0 * sqrt(num_layers) in
       the standard SIREN analysis (Sitzmann et al.).

    So the rule of thumb is:
        omega_0_min ≈ k99 / sqrt(num_layers)

    Current omega_0 = 30, num_layers = 5, so the max frequency it can
    represent is ~30 * sqrt(5) ≈ 67 cycles per unit. If k99 << 67, we have
    headroom and can safely reduce omega_0 substantially.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _unwrap(o):
    if isinstance(o, dict) and "state_dict" in o and isinstance(o["state_dict"], dict):
        return o["state_dict"]
    return o


def build_siren(sd, device, hidden=256, n_layers=5, omega=30.0):
    from src.siren import SIREN
    sd = _unwrap(sd)
    for kw in [
        dict(in_dim=3, out_dim=1, hidden=hidden, n_layers=n_layers, omega=omega),
        dict(in_features=3, out_features=1, hidden_features=hidden,
             hidden_layers=n_layers, first_omega_0=omega, hidden_omega_0=omega),
    ]:
        try:
            net = SIREN(**kw); break
        except TypeError:
            net = None
    if net is None:
        net = SIREN(3, 1, hidden, n_layers)
    net.load_state_dict(sd)
    net.to(device).eval()
    return net


@torch.no_grad()
def query_sdf_grid(net, device, res, bound=1.0, chunk=65536):
    lin = torch.linspace(-bound, bound, res, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    for i in range(0, pts.shape[0], chunk):
        s = net(pts[i:i + chunk])
        if s.dim() > 1:
            s = s.squeeze(-1)
        out[i:i + chunk] = s
    return out.reshape(res, res, res)  # keep on GPU


# ----------------------------------------------------------------------------
# Radial power spectrum
# ----------------------------------------------------------------------------

def radial_power_spectrum(volume: torch.Tensor, bound: float = 1.0):
    """Compute the radial (angle-averaged) power spectrum of a 3D volume.

    Returns (k_vals, power) where:
      - k_vals: cycles per unit length (not per pixel)
      - power: sum of |FFT|^2 within each radial bin

    The domain is [-bound, bound]^3, length 2*bound. Nyquist = res / (2 * 2*bound) = res / (4*bound).
    """
    res = volume.shape[0]
    L = 2.0 * bound                  # domain length
    dx = L / res                     # voxel size

    # 3D FFT
    F = torch.fft.fftn(volume)
    P = (F.real ** 2 + F.imag ** 2)  # power

    # Frequency grids: torch.fft.fftfreq returns cycles/sample, scale by 1/dx
    # to get cycles/unit length.
    fx = torch.fft.fftfreq(res, d=dx, device=volume.device)
    fy = torch.fft.fftfreq(res, d=dx, device=volume.device)
    fz = torch.fft.fftfreq(res, d=dx, device=volume.device)
    FX, FY, FZ = torch.meshgrid(fx, fy, fz, indexing="ij")
    K = torch.sqrt(FX ** 2 + FY ** 2 + FZ ** 2)  # (res, res, res)

    # Radial bins
    kmax = K.max().item()
    n_bins = res // 2
    bin_edges = torch.linspace(0.0, kmax, n_bins + 1, device=volume.device)
    # For each voxel, find its bin index
    bin_idx = torch.bucketize(K.flatten(), bin_edges) - 1
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    # Sum power per bin
    radial = torch.zeros(n_bins, device=volume.device)
    radial.scatter_add_(0, bin_idx, P.flatten())

    # Bin centers
    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return k_centers.cpu().numpy(), radial.cpu().numpy()


def energy_quantile_freq(k, power, q: float):
    """Return the frequency below which fraction q of the total energy lies."""
    cum = np.cumsum(power)
    total = cum[-1]
    if total <= 0:
        return 0.0
    idx = int(np.searchsorted(cum, q * total))
    idx = min(idx, len(k) - 1)
    return float(k[idx])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/workspace/hypernet/scripts/manifest.pt")
    ap.add_argument("--grid_res", type=int, default=128,
                    help="resolution for the SDF volume FFT (128 gives Nyquist=32 cycles/unit)")
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=5)
    ap.add_argument("--omega", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/workspace/hypernet/data/omega_diagnostic.npz")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/hypernet")
    sys.path.insert(0, "/workspace/hypernet/scripts")

    device = torch.device(args.device)
    manifest = torch.load(args.manifest, map_location="cpu", weights_only=False)

    print(f"[diag] grid_res={args.grid_res}  bound=[-{args.bound},{args.bound}]")
    print(f"[diag] spatial Nyquist = {args.grid_res / (4 * args.bound):.2f} cycles/unit")
    print(f"[diag] current SIREN: omega_0={args.omega}, n_layers={args.n_layers}")
    print(f"[diag] current max representable freq (omega_0 * sqrt(L)) ≈ "
          f"{args.omega * np.sqrt(args.n_layers):.2f} cycles/unit\n")

    all_k95 = []
    all_k99 = []
    all_k_peak = []
    all_power = []

    for i, p in enumerate(manifest["shape_paths"]):
        p = Path(p)
        sd = torch.load(p, map_location="cpu")
        net = build_siren(sd, device,
                          hidden=args.hidden, n_layers=args.n_layers, omega=args.omega)
        vol = query_sdf_grid(net, device, res=args.grid_res, bound=args.bound)

        k, power = radial_power_spectrum(vol, bound=args.bound)
        # Skip DC bin (k=0) when measuring frequency content — it's just the mean SDF
        power_ac = power.copy()
        power_ac[0] = 0.0

        k95 = energy_quantile_freq(k, power_ac, 0.95)
        k99 = energy_quantile_freq(k, power_ac, 0.99)
        k_peak = float(k[np.argmax(power_ac)])

        all_k95.append(k95)
        all_k99.append(k99)
        all_k_peak.append(k_peak)
        all_power.append(power_ac)

        print(f"  {p.stem}  peak={k_peak:.2f}  k95={k95:.2f}  k99={k99:.2f}  "
              f"(cycles/unit over domain of length {2*args.bound})")

    k95_arr = np.array(all_k95)
    k99_arr = np.array(all_k99)

    print(f"\n[summary across 10 shapes]")
    print(f"  k95 across shapes:  mean={k95_arr.mean():.2f}  max={k95_arr.max():.2f}")
    print(f"  k99 across shapes:  mean={k99_arr.mean():.2f}  max={k99_arr.max():.2f}")

    # ----- Recommendation -----
    # A SIREN with n_layers and omega_0 can represent frequencies up to
    # roughly omega_0 * sqrt(n_layers).  So to represent k_max, we need:
    #     omega_0 >= k_max / sqrt(n_layers)
    # Use k99 (worst case across shapes) and a 1.5x safety margin.
    k_needed = k99_arr.max()
    omega_min = k_needed / np.sqrt(args.n_layers)
    omega_rec = 1.5 * omega_min  # safety margin

    print(f"\n[recommendation]")
    print(f"  max k99 across shapes = {k_needed:.2f} cycles/unit")
    print(f"  minimum omega_0 to represent this = {omega_min:.2f}")
    print(f"  recommended omega_0 with 1.5x margin = {omega_rec:.2f}")
    print(f"  current omega_0 = {args.omega}")
    print(f"  headroom factor = {args.omega / omega_min:.2f}x")

    # Interpret
    if args.omega / omega_min > 3:
        print("\n  --> SIGNIFICANT HEADROOM. You can safely reduce omega_0 to around")
        print(f"      {omega_rec:.0f}-{min(args.omega, omega_rec * 2):.0f} without losing representational power.")
    elif args.omega / omega_min > 1.5:
        print("\n  --> MODERATE HEADROOM. A small reduction to ~{:.0f} is probably safe;".format(omega_rec))
        print("      going lower risks losing detail.")
    else:
        print("\n  --> LITTLE HEADROOM. Current omega_0 is near the minimum;")
        print("      reducing would lose detail.")

    # Also suggest concrete round-number choices
    print("\n[concrete suggestions]")
    for candidate in [20, 15, 10, 5]:
        max_freq = candidate * np.sqrt(args.n_layers)
        coverage_frac = (k99_arr <= max_freq).mean()
        status = "safe" if max_freq > k_needed * 1.2 else ("tight" if max_freq > k_needed else "risky")
        print(f"  omega_0 = {candidate:3d}  -> max representable ~{max_freq:.1f} cycles/unit  "
              f"[{status}]  (covers k99 of {coverage_frac*100:.0f}% of shapes)")

    # Save for later reference
    np.savez(args.out,
             k_centers=k,
             power_per_shape=np.stack(all_power),
             k95=k95_arr,
             k99=k99_arr,
             k_peak=np.array(all_k_peak))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
