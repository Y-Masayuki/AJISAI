#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sigma_clip_rms_test.py
======================
AJISAI off-source RMS estimator comparison (standalone, no CASA required).

Purpose:
    Compare five RMS estimation methods on a CASA-exported FITS image to
    decide whether sigma-clipping based methods can safely replace the
    legacy annulus-based method that required a user-specified
    ``target_radius``. Both numerical and visual diagnostics are produced.

Dependencies:
    numpy, matplotlib, astropy  (all standard; should be present in any
    CASA environment).

Usage:
    python sigma_clip_rms_test.py <fitsfile>
    python sigma_clip_rms_test.py <fitsfile> --target-radius 2.0
    python sigma_clip_rms_test.py <fitsfile> --target-radius 2.0 --exclude-factor 5.0

Input:
    A CASA-exported continuum FITS image.
    Shape may be (1, 1, ny, nx) or (ny, nx); extra axes are squeezed.

Output:
    1. Standard output: numerical comparison of the five methods plus
       diagnostic information.
    2. PNG:  <fitsname>_rms_methods.png  (four-panel visualization).

The five RMS estimators:
    (1) Whole image std          - naive std over all finite pixels
                                    (includes source contamination)
    (2) Sigma-clip (3 sigma, 5 iters) - astropy.stats.sigma_clipped_stats
    (3) Sigma-clip + center exclusion - sigma-clip after masking out a
                                    central circle of radius N*beam
    (4) Off-source annulus       - legacy AJISAI: std over an annulus
                                    target_radius < r < 0.4*FoV
    (5) MAD std                  - median absolute deviation based std

Interpretation guide:
    - If method (2) or (5) agrees with (4) to within ~10%, sigma-clipping
      can safely replace the annulus method.
    - If method (2) is smaller than (4): possible source contamination
      in the annulus -> annulus may overestimate noise.
    - If method (2) is larger than (4): possible extended emission outside
      the annulus -> sigma-clip may underestimate noise.
    - Method (3) with center exclusion should match (4) most closely.

Author: Masayuki Yamaguchi (Kyushu Univ./NAOJ) and AJISAI development team
License: MIT
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, mad_std


# ============================================================================
# I/O
# ============================================================================
def load_image(fitsfile):
    """Load a CASA-exported FITS and return the 2D data plus coordinate info."""
    with fits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
    while data.ndim > 2:
        data = data[0]
    cdelt_x = abs(hdr["CDELT1"]) * 3600.0  # arcsec/pix
    # CDELT2 is read for symmetry but not used downstream (image is assumed
    # to be square in pixel scale).
    bmaj = hdr.get("BMAJ", None)
    bmin = hdr.get("BMIN", None)
    bmaj_as = bmaj * 3600.0 if bmaj is not None else None
    bmin_as = bmin * 3600.0 if bmin is not None else None
    beam_pix = np.sqrt(bmaj_as * bmin_as) / cdelt_x if bmaj is not None else None
    return {
        "data": data,
        "cdelt_arcsec": cdelt_x,
        "bmaj_arcsec": bmaj_as,
        "bmin_arcsec": bmin_as,
        "beam_pix": beam_pix,
        "crpix": (hdr["CRPIX1"] - 1, hdr["CRPIX2"] - 1),  # 0-indexed
        "shape": data.shape,
    }


# ============================================================================
# RMS estimators
# ============================================================================
def rms_whole_image(data):
    """Naive: std of all finite pixels (heavy source contamination expected)."""
    valid = np.isfinite(data)
    return float(np.std(data[valid]))


def rms_sigma_clip(data, sigma=3.0, maxiters=5):
    """Iterative sigma-clipping with median centering and std dispersion."""
    _, _, std = sigma_clipped_stats(
        data, sigma=sigma, maxiters=maxiters,
        cenfunc="median", stdfunc="std"
    )
    return float(std)


def rms_sigma_clip_with_center_exclusion(data, crpix, beam_pix, exclude_factor=5.0):
    """Sigma-clip after masking out a central circle of radius exclude_factor * beam."""
    if beam_pix is None:
        return None, None
    cx, cy = crpix
    ny, nx = data.shape
    yy, xx = np.mgrid[:ny, :nx]
    r_pix = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask_off = r_pix > exclude_factor * beam_pix
    masked = np.where(mask_off, data, np.nan)
    _, _, std = sigma_clipped_stats(
        masked, sigma=3.0, maxiters=5,
        cenfunc="median", stdfunc="std"
    )
    return float(std), mask_off


def rms_offsource_annulus(data, crpix, cdelt_arcsec, target_radius_arcsec, fov_arcsec):
    """Legacy AJISAI: std of pixels in the annulus target_radius < r < 0.4*FoV."""
    cx, cy = crpix
    ny, nx = data.shape
    yy, xx = np.mgrid[:ny, :nx]
    x_as = (xx - cx) * cdelt_arcsec
    y_as = (yy - cy) * cdelt_arcsec
    r_as = np.sqrt(x_as ** 2 + y_as ** 2)
    mask = (r_as > target_radius_arcsec) & (r_as < 0.4 * fov_arcsec)
    region = data[mask]
    region = region[np.isfinite(region)]
    return float(region.std()), mask


def rms_mad(data):
    """Median absolute deviation based std (robust to outliers)."""
    return float(mad_std(data, ignore_nan=True))


# ============================================================================
# Diagnostics for sigma-clip mask reconstruction (for visualization)
# ============================================================================
def reconstruct_sclip_mask(data, sigma=3.0, maxiters=5):
    """Reconstruct the boolean mask of pixels retained by sigma-clipping."""
    arr = data.copy()
    mask_valid = np.isfinite(arr)
    arr_v = arr[mask_valid]
    median = np.median(arr_v)
    std = np.std(arr_v)
    keep = np.ones_like(arr_v, dtype=bool)
    for _ in range(maxiters):
        dev = np.abs(arr_v - median)
        new_keep = dev < sigma * std
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
        if keep.sum() == 0:
            break
        median = np.median(arr_v[keep])
        std = np.std(arr_v[keep])
    full_mask = np.zeros_like(data, dtype=bool)
    full_mask[mask_valid] = keep
    return full_mask


# ============================================================================
# Visualization
# ============================================================================
def make_comparison_plot(img, results, masks, args, out_png):
    """Four-panel diagnostic plot."""
    data = img["data"]
    rms_sclip = results["sigma_clip"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # --- Panel (a): original image ---
    ax = axes[0, 0]
    vmax = np.nanmax(data) * 0.5
    vmin = -3 * rms_sclip
    im = ax.imshow(data, origin="lower", cmap="afmhot", vmin=vmin, vmax=vmax)
    ax.set_title("(a) Original image (stretched +/-5 sigma_sclip about peak/2)")
    ax.set_xlabel("pix")
    ax.set_ylabel("pix")
    plt.colorbar(im, ax=ax, label="Jy/beam", fraction=0.046)

    # --- Panel (b): pixels retained by sigma-clip ---
    ax = axes[0, 1]
    kept = masks["sigma_clip_kept"]
    overlay = np.where(kept, data, np.nan)
    im = ax.imshow(overlay, origin="lower", cmap="viridis",
                   vmin=-3 * rms_sclip, vmax=3 * rms_sclip)
    n_kept = kept.sum()
    n_finite = np.isfinite(data).sum()
    ax.set_title(f"(b) Pixels kept by 3-sigma clip ({100*n_kept/n_finite:.1f}%) "
                 f"-> RMS = {rms_sclip*1e6:.1f} uJy/beam")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # Overlay annulus boundary if available
    if args.target_radius and img["beam_pix"]:
        from matplotlib.patches import Circle
        cx, cy = img["crpix"]
        r_in_pix = args.target_radius / img["cdelt_arcsec"]
        r_out_pix = 0.4 * (img["shape"][1] * img["cdelt_arcsec"]) / img["cdelt_arcsec"]
        for rp, ls in [(r_in_pix, "--"), (r_out_pix, ":")]:
            c = Circle((cx, cy), rp, fill=False, edgecolor="red", linestyle=ls, lw=1.2)
            ax.add_patch(c)
        ax.plot([], [], "r--", label=f"annulus inner ({args.target_radius}\")")
        ax.plot([], [], "r:",  label="annulus outer (0.4 * FoV)")
        ax.legend(loc="upper right", fontsize=8)

    # --- Panel (c): histogram with thresholds ---
    ax = axes[1, 0]
    flat = data[np.isfinite(data)].ravel() * 1e6
    ax.hist(flat, bins=300, log=True, color="steelblue", alpha=0.7, label="all pixels")
    for s, c, label in [(3, "red", "3 sigma"), (5, "darkorange", "5 sigma")]:
        ax.axvline(s * rms_sclip * 1e6, color=c, ls="--",
                   label=f"+/-{label}_sclip = +/-{s*rms_sclip*1e6:.1f} uJy")
        ax.axvline(-s * rms_sclip * 1e6, color=c, ls="--")
    ax.set_xlabel("Pixel value [uJy/beam]")
    ax.set_ylabel("Count (log)")
    ax.set_title("(c) Pixel value histogram")
    ax.set_xlim(-10 * rms_sclip * 1e6, 10 * rms_sclip * 1e6)
    ax.legend(fontsize=8)

    # --- Panel (d): bar chart of methods ---
    ax = axes[1, 1]
    labels = []
    values = []
    colors = []
    for key, label, color in [
        ("whole", "(1) whole img", "gray"),
        ("sigma_clip", "(2) sigma-clip", "steelblue"),
        ("sigma_clip_excl", "(3) sigma-clip\n+center excl", "seagreen"),
        ("annulus", "(4) annulus\n(AJISAI)", "orange"),
        ("mad", "(5) MAD", "purple"),
    ]:
        v = results.get(key)
        if v is not None:
            labels.append(label)
            values.append(v * 1e6)
            colors.append(color)
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(rms_sclip * 1e6, color="steelblue", ls="--", alpha=0.5,
               label="sigma-clip reference")
    ax.set_ylabel("RMS [uJy/beam]")
    ax.set_title("(d) RMS comparison across methods")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=8)

    plt.suptitle(f"AJISAI RMS method comparison: {os.path.basename(args.fitsfile)}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fitsfile", help="CASA-exported FITS image file")
    parser.add_argument("--target-radius", type=float, default=None,
                        help="Inner radius [arcsec] for off-source annulus method "
                             "(omit to skip method 4)")
    parser.add_argument("--exclude-factor", type=float, default=5.0,
                        help="Beam factor for center exclusion in method 3 (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.fitsfile):
        sys.exit(f"FITS file not found: {args.fitsfile}")

    img = load_image(args.fitsfile)
    data = img["data"]

    print("\n" + "=" * 70)
    print("  AJISAI sigma-clip RMS test")
    print("=" * 70)
    print(f"  File         : {args.fitsfile}")
    print(f"  Shape        : {img['shape']}")
    print(f"  Pixel scale  : {img['cdelt_arcsec']:.5f} arcsec/pix")
    if img["bmaj_arcsec"]:
        print(f"  Beam         : {img['bmaj_arcsec']:.4f} x {img['bmin_arcsec']:.4f} arcsec "
              f"({img['beam_pix']:.2f} pix)")
    fov_arcsec = img["shape"][1] * img["cdelt_arcsec"]
    print(f"  FoV          : {fov_arcsec:.2f} arcsec")
    print(f"  Image peak   : {np.nanmax(data)*1e3:.4f} mJy/beam "
          f"(max abs = {np.nanmax(np.abs(data))*1e3:.4f} mJy/beam)")

    # Compute all RMS values
    results = {}
    masks = {}

    results["whole"] = rms_whole_image(data)
    results["sigma_clip"] = rms_sigma_clip(data)
    masks["sigma_clip_kept"] = reconstruct_sclip_mask(data)

    if img["beam_pix"]:
        rms3, mask3 = rms_sigma_clip_with_center_exclusion(
            data, img["crpix"], img["beam_pix"], args.exclude_factor
        )
        results["sigma_clip_excl"] = rms3
        masks["sigma_clip_excl"] = mask3
    else:
        results["sigma_clip_excl"] = None
        print("  [WARN] No BMAJ/BMIN in header -> skipping method (3)")

    if args.target_radius:
        rms4, mask4 = rms_offsource_annulus(
            data, img["crpix"], img["cdelt_arcsec"],
            args.target_radius, fov_arcsec
        )
        results["annulus"] = rms4
        masks["annulus"] = mask4
    else:
        results["annulus"] = None

    results["mad"] = rms_mad(data)

    # Print comparison table
    print()
    print("=" * 70)
    print("  RMS estimates (lower = more conservative; larger = more contamination)")
    print("=" * 70)
    ref = results["sigma_clip"]
    rows = [
        ("(1) Whole image std (naive)", results["whole"]),
        ("(2) Sigma-clip 3-sigma x 5 iters (astropy)", results["sigma_clip"]),
        (f"(3) Sigma-clip + exclude r<{args.exclude_factor}*beam", results["sigma_clip_excl"]),
        (f"(4) Annulus {args.target_radius}\" < r < 0.4*FoV (AJISAI)"
         if args.target_radius else "(4) Annulus (SKIPPED: pass --target-radius)",
         results["annulus"]),
        ("(5) MAD std (robust median-based)", results["mad"]),
    ]
    print(f"  {'Method':<50s}  {'RMS [uJy/bm]':>13s}  {'ratio':>7s}")
    print("  " + "-" * 75)
    for name, v in rows:
        if v is None:
            print(f"  {name:<50s}  {'(n/a)':>13s}  {'-':>7s}")
        else:
            r = v / ref
            print(f"  {name:<50s}  {v*1e6:>13.2f}  {r:>7.3f}")

    # Diagnostics
    n_finite = np.isfinite(data).sum()
    n_kept = masks["sigma_clip_kept"].sum()
    print()
    print("=" * 70)
    print("  Diagnostics")
    print("=" * 70)
    print(f"  Finite pixels        : {n_finite:,} / {data.size:,} "
          f"({100*n_finite/data.size:.1f}%)")
    print(f"  Kept by sigma-clip   : {n_kept:,} ({100*n_kept/n_finite:.2f}% of finite)")
    print(f"  Rejected             : {n_finite - n_kept:,} "
          f"({100*(n_finite-n_kept)/n_finite:.2f}% of finite)")
    if img["beam_pix"]:
        beam_area_pix = np.pi * img["beam_pix"] ** 2
        print(f"  -> approx {(n_finite - n_kept) / beam_area_pix:.1f} beam-areas of "
              f"source/artifact rejected by sigma-clip")

    # Interpretation
    print()
    print("=" * 70)
    print("  Interpretation")
    print("=" * 70)
    if results["annulus"] is not None:
        delta = abs(results["sigma_clip"] - results["annulus"]) / results["annulus"] * 100
        print(f"  sigma-clip (2) vs AJISAI annulus (4): differ by {delta:.1f}%")
        if delta < 10:
            print("  -> sigma-clip adoption is SAFE (within 10%)")
        elif delta < 25:
            print("  -> sigma-clip adoption is acceptable (within 25%); "
                  "worth considering")
        else:
            print("  -> sigma-clip and annulus differ significantly. "
                  "Inspect image character.")
            if results["sigma_clip"] < results["annulus"]:
                print("     sigma-clip smaller: extended emission may bias "
                      "sigma-clip downward")
            else:
                print("     sigma-clip larger: source flux leaking into the "
                      "annulus may bias annulus downward")

    # Save plot
    out_png = os.path.splitext(args.fitsfile)[0] + "_rms_methods.png"
    make_comparison_plot(img, results, masks, args, out_png)
    print()
    print(f"  Saved comparison plot: {out_png}")
    print()


if __name__ == "__main__":
    main()
