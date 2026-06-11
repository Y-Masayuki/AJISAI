#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ajisai.core
===========
Core implementation of the AJISAI self-calibration pipeline.

This module is normally not imported directly; use::

    from ajisai import AJISAI, AJISAIConfig

instead. Internal use only::

    from ajisai.core import _safe_rmtree_path  # internal helper

Typical usage (run from within CASA or with modular CASA installed)::

    from ajisai import AJISAI, AJISAIConfig

    cfg = AJISAIConfig(vis="/path/to/data.ms")
    aj = AJISAI(cfg).run()
    print(aj.best_image, aj.best_metric_value)

Multi-field usage::

    from ajisai import AJISAI, AJISAIConfig, list_target_fields

    vis = "/path/to/multi_field.ms"
    for f in list_target_fields(vis):
        cfg = AJISAIConfig(vis=vis, field=f["id"], projname=f["name"])
        AJISAI(cfg).run()

Copyright (c) 2026 Masayuki Yamaguchi (Kyushu Univ./NAOJ).
Released under the MIT License.
"""

# ============================================================================
# Imports
# ============================================================================
from __future__ import annotations

import json
import math
import os
import re
import shutil
import warnings
from dataclasses import asdict, dataclass, field as _dc_field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# CASA imports are guarded so that this module can be imported in test
# environments without CASA installed. Methods that actually need CASA
# will raise a clear error when called.
try:
    import casatools  # type: ignore
    import casatasks  # type: ignore
    import casaplotms  # type: ignore
    _HAS_CASA = True
except ImportError:  # pragma: no cover
    casatools = None  # type: ignore
    casatasks = None  # type: ignore
    casaplotms = None  # type: ignore
    _HAS_CASA = False

# AJISAI MS query helpers (native casatools-based replacement for analysisUtils).
from .ms_utils import (
    get_on_source_time,
    get_median_frequency,
    get_baseline_at_percentile,
    get_antenna_positions,
    get_antenna_flag_stats,
    pick_cell_imsize,
    icrs_to_j2000,
)
_HAS_MS_UTILS = True


# Version is defined in ajisai/__init__.py. This local copy is used for
# the justification log and for the standalone __main__ smoke test.
__version__ = "0.1.0"


# ============================================================================
# Configuration dataclasses (immutable for reproducibility)
# ============================================================================

@dataclass(frozen=True)
class ImagingConfig:
    """tclean-related parameters. Extend by adding fields here."""
    robust: float = 0.5
    weighting: str = "briggs"
    niter: int = 50000
    gain: float = 0.05
    cyclefactor: float = 1.5
    threshold_factor: float = 1.0      # threshold = threshold_factor * RMS
    deconvolver: str = "multiscale"
    gridder: str = "standard"
    pbcor: bool = False
    parallel: bool = False
    uvtaper: Tuple[str, ...] = ()
    nterms: int = 1
    cellpix: int = 10                  # pixels per beam
    # Mask strategy. Supported values:
    #   "auto-multithresh" (default): CASA's automatic threshold-based masking.
    #       Works in any CASA version. Robust for typical ALMA continuum sources.
    #   "interactive": let the user draw a mask in the CASA viewer at the first
    #       tclean call. The saved mask is reused by all subsequent self-cal
    #       iterations. REQUIRES CASA <= 6.6; the casaviewer GUI was removed
    #       in CASA 6.7+, so interactive tclean is non-functional there.
    #   "user": use a pre-made mask file supplied via `user_mask`. This is the
    #       portable option for CASA 6.7+ users: prepare a mask externally
    #       (e.g. with CARTA, an older CASA installation, or any FITS-aware
    #       tool) and point AJISAI at it.
    #   "none": no mask at all (cleans without restriction). Rarely useful.
    mask_mode: str = "auto-multithresh"
    user_mask: Optional[str] = None
    # auto-multithresh tunables (CASA defaults are conservative for ALMA continuum)
    sidelobethreshold: float = 2.0
    noisethreshold: float = 4.25
    lownoisethreshold: float = 1.5
    minbeamfrac: float = 0.3
    growiterations: int = 75


@dataclass(frozen=True)
class GainCalConfig:
    """gaincal/applycal-related parameters."""
    minblperant: int = 4
    minsnr: float = 1.5
    gaintype: str = "T"
    combine: str = "scan"
    gaincal_interp: str = "linear,linear"
    applycal_interp: str = "linearPD"
    applymode: str = "calonly"
    calwt: bool = True


@dataclass(frozen=True)
class SelfcalStep:
    """A single self-calibration step in the schedule."""
    calmode: str                   # "p" (phase) or "a" (amp; phase is preserved as well)
    solint: str                    # CASA solint string OR "N*IT" where IT = integration time
    solnorm: bool = False          # set True for amp calibration to preserve flux scale
    label: Optional[str] = None    # human-readable name for justification log


@dataclass(frozen=True)
class SelfcalSchedule:
    """Ordered sequence of self-cal steps. Default = AJISAI Description spec."""
    steps: Tuple[SelfcalStep, ...] = _dc_field(default_factory=lambda: (
        SelfcalStep("p", "inf",  label="phase_inf"),
        SelfcalStep("p", "6*IT", label="phase_6IT"),
        SelfcalStep("p", "3*IT", label="phase_3IT"),
        SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
    ))


@dataclass(frozen=True)
class AJISAIConfig:
    """
    Top-level user-facing configuration.

    The minimum required parameter is ``vis``. Everything else has a
    sensible default or is auto-derived from the data.
    """
    # === Required ===
    vis: str

    # === Recommended / auto-derived if None ===
    field: Optional[str] = None
    projname: Optional[str] = None
    results_dir: Optional[str] = None

    # === Sub-configs (composition) ===
    imaging: ImagingConfig = _dc_field(default_factory=ImagingConfig)
    gaincal: GainCalConfig = _dc_field(default_factory=GainCalConfig)
    schedule: SelfcalSchedule = _dc_field(default_factory=SelfcalSchedule)

    # === Phase shift ===
    # Off by default. When enabled, AJISAI shifts the MS so that the target
    # source lies at the phase center, which is helpful (though not strictly
    # required) for self-calibration model accuracy.
    phase_shift: bool = False
    # If phase_center is None and phase_shift is True, AJISAI uses the
    # brightest pixel in the dirty image as the new phase center (method A:
    # pixel-accurate, never fails).
    # If phase_center is given, it must be a 3-tuple (ra, dec, frame) where
    # frame is either "ICRS" or "J2000". This lets advanced users supply
    # sub-pixel-accurate coordinates from their own imfit/Gaussian fit.
    # Example: phase_center=("16:25:45.0", "-24:12:23.0", "ICRS")
    phase_center: Optional[Tuple[str, str, str]] = None

    # === Strategy switches (future-proofing for new algorithms) ===
    refant_strategy: str = "hybrid"        # "hybrid"|"geometric_center"|"flag_stats"|"manual"
    refant_manual: Optional[str] = None    # antenna name when strategy="manual"
    refant_flag_threshold: float = 0.25    # fraction; antennas with > this are excluded
    rms_method: str = "sigma_clip_excl"    # "sigma_clip_excl"|"sigma_clip"|"annulus"|"mad"
    rms_exclude_beam_factor: float = 5.0   # for sigma_clip_excl: radius = N*beam
    rms_sigma: float = 3.0
    rms_maxiters: int = 5
    quality_metric: str = "dynamic_range"  # "dynamic_range"|"peak_snr"
    # NOTE: AJISAI runs ALL iterations to completion and selects the best by DR
    # at the end. Anomalies (e.g. catastrophic DR drop) are logged for the
    # justification but do NOT alter pipeline flow. This preserves the
    # deterministic-fixed-schedule identity vs. auto_selfcal's adaptive rollback.
    on_iter_anomaly: str = "log_only"      # "log_only" (default) | future: "warn_loudly"

    # === Misc ===
    verbose: bool = True
    # Print the AJISAI ASCII banner at the start of run(). Disable for
    # automated batch runs where the banner clutters the log.
    show_banner: bool = True


# ============================================================================
# Pure utility functions (CASA-independent)
# ============================================================================

def load_fits_image(fitsfile: str) -> Dict[str, Any]:
    """
    Replacement for ``imdata.IMFITS``.

    Returns a dict with:
        data2d        : 2D ndarray (last two axes squeezed)
        cdelt_arcsec  : pixel scale in arcsec (positive)
        bmaj_arcsec   : beam major in arcsec (or None)
        bmin_arcsec   : beam minor in arcsec (or None)
        beam_pix      : geometric mean beam in pixels (or None)
        crpix         : (x, y) reference pixel, 0-indexed
        shape         : (ny, nx)
        x_arcsec      : 1D array of x offsets (arcsec) from reference pixel
        y_arcsec      : 1D array of y offsets (arcsec) from reference pixel
        header        : full FITS header

    This intentionally returns a plain dict (no class wrapper) to avoid the
    14k-line imdata.py dependency that the original AJISAI.py used merely for
    .data[0,0] and .get_xygrid().
    """
    with fits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
    while data.ndim > 2:
        data = data[0]

    cdelt_x = abs(hdr["CDELT1"]) * 3600.0  # arcsec/pix
    cdelt_y = abs(hdr["CDELT2"]) * 3600.0

    bmaj = hdr.get("BMAJ")
    bmin = hdr.get("BMIN")
    bmaj_as = bmaj * 3600.0 if bmaj is not None else None
    bmin_as = bmin * 3600.0 if bmin is not None else None
    beam_pix = (np.sqrt(bmaj_as * bmin_as) / cdelt_x) if bmaj is not None else None

    nx, ny = data.shape[1], data.shape[0]
    crpix1 = hdr["CRPIX1"] - 1  # FITS is 1-indexed; convert to 0-indexed
    crpix2 = hdr["CRPIX2"] - 1

    # Offset grids (arcsec). RA grows leftward in standard image convention.
    x_arcsec = (np.arange(nx) - crpix1) * (-cdelt_x)
    y_arcsec = (np.arange(ny) - crpix2) * cdelt_y

    return {
        "data2d": data,
        "cdelt_arcsec": cdelt_x,
        "bmaj_arcsec": bmaj_as,
        "bmin_arcsec": bmin_as,
        "beam_pix": beam_pix,
        "crpix": (crpix1, crpix2),
        "shape": data.shape,
        "x_arcsec": x_arcsec,
        "y_arcsec": y_arcsec,
        "header": hdr,
    }


def compute_rms(
    fitsfile: str,
    method: str = "sigma_clip_excl",
    exclude_factor: float = 5.0,
    sigma: float = 3.0,
    maxiters: int = 5,
    target_radius_arcsec: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute off-source RMS noise of a CASA-exported FITS image.

    Validated against ``G204SW_TM12.clim_sc2.fits`` where ``sigma_clip_excl``
    agrees with the legacy annulus method to within 1 percent, while
    requiring no source-dependent ``target_radius`` input.

    Parameters
    ----------
    fitsfile : str
        Path to a CASA-exported FITS image.
    method : str
        One of:

        - ``"sigma_clip_excl"`` (default, recommended): sigma-clipping
          after excluding a circle of radius ``exclude_factor * beam``
          from the image center. Robust to bright sources and PB-masked
          regions.
        - ``"sigma_clip"``: sigma-clipping over the entire image.
        - ``"annulus"``: legacy AJISAI method; requires
          ``target_radius_arcsec``.
        - ``"mad"``: median absolute deviation based std.

    Returns
    -------
    dict
        Keys: ``rms`` (RMS in Jy/beam), ``method``, ``n_kept`` (pixels used
        in the estimate), ``n_total_finite`` (finite pixels in the image),
        ``justification`` (structured rationale for the justification log).
    """
    img = load_fits_image(fitsfile)
    data = img["data2d"]
    n_finite = int(np.isfinite(data).sum())

    if method == "sigma_clip_excl":
        if img["beam_pix"] is None:
            warnings.warn("No BMAJ/BMIN in FITS header; falling back to sigma_clip without center exclusion")
            return compute_rms(fitsfile, method="sigma_clip", sigma=sigma, maxiters=maxiters)
        cx, cy = img["crpix"]
        ny, nx = data.shape
        yy, xx = np.mgrid[:ny, :nx]
        r_pix = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask_off = r_pix > exclude_factor * img["beam_pix"]
        masked = np.where(mask_off, data, np.nan)
        _, _, std = sigma_clipped_stats(
            masked, sigma=sigma, maxiters=maxiters,
            cenfunc="median", stdfunc="std",
        )
        n_kept = int(np.isfinite(masked).sum())  # upper bound; sigma_clip may reject more
        return {
            "rms": float(std),
            "method": method,
            "n_kept": n_kept,
            "n_total_finite": n_finite,
            "justification": {
                "method": "sigma_clip_excl",
                "sigma": sigma,
                "maxiters": maxiters,
                "exclude_factor_beam": exclude_factor,
                "exclude_radius_pix": float(exclude_factor * img["beam_pix"]),
                "beam_pix": float(img["beam_pix"]),
                "description": (
                    f"3-sigma iterative clipping (astropy.stats.sigma_clipped_stats) "
                    f"after excluding a central circle of radius "
                    f"{exclude_factor:.1f} x beam from the image. "
                    f"Validated to agree with annulus method within 1% on "
                    f"G204SW_TM12 test image."
                ),
            },
        }

    elif method == "sigma_clip":
        _, _, std = sigma_clipped_stats(
            data, sigma=sigma, maxiters=maxiters,
            cenfunc="median", stdfunc="std",
        )
        return {
            "rms": float(std),
            "method": method,
            "n_kept": n_finite,
            "n_total_finite": n_finite,
            "justification": {
                "method": "sigma_clip",
                "sigma": sigma,
                "maxiters": maxiters,
                "description": "sigma-clipping over the entire image (no center exclusion).",
            },
        }

    elif method == "annulus":
        if target_radius_arcsec is None:
            raise ValueError("method='annulus' requires target_radius_arcsec to be set")
        cx, cy = img["crpix"]
        ny, nx = data.shape
        yy, xx = np.mgrid[:ny, :nx]
        x_as = (xx - cx) * img["cdelt_arcsec"]
        y_as = (yy - cy) * img["cdelt_arcsec"]
        r_as = np.sqrt(x_as ** 2 + y_as ** 2)
        fov_as = nx * img["cdelt_arcsec"]
        mask = (r_as > target_radius_arcsec) & (r_as < 0.4 * fov_as)
        region = data[mask & np.isfinite(data)]
        return {
            "rms": float(region.std()),
            "method": method,
            "n_kept": int(region.size),
            "n_total_finite": n_finite,
            "justification": {
                "method": "annulus",
                "target_radius_arcsec": target_radius_arcsec,
                "outer_radius_arcsec": 0.4 * fov_as,
                "description": "legacy AJISAI: std of pixels in annulus around phase center.",
            },
        }

    elif method == "mad":
        from astropy.stats import mad_std
        return {
            "rms": float(mad_std(data, ignore_nan=True)),
            "method": method,
            "n_kept": n_finite,
            "n_total_finite": n_finite,
            "justification": {"method": "mad", "description": "MAD-based std (robust)."},
        }

    else:
        raise ValueError(f"Unknown RMS method: {method!r}")


def compute_image_stats(
    fitsfile: str,
    rms_method: str = "sigma_clip_excl",
    **rms_kwargs,
) -> Dict[str, Any]:
    """
    Compute peak, RMS, SNR, dynamic range, and beam from a FITS image.

    Dynamic range (DR) = peak / RMS_offsource is the recommended quality
    metric, more robust than peak SNR because RMS is measured off-source
    rather than including the source pixels.
    """
    img = load_fits_image(fitsfile)
    data = img["data2d"]
    peak = float(np.nanmax(data))
    rms_info = compute_rms(fitsfile, method=rms_method, **rms_kwargs)
    rms = rms_info["rms"]
    return {
        "peak_jy_beam": peak,
        "rms_jy_beam": rms,
        "snr": peak / rms if rms > 0 else np.nan,
        "dynamic_range": peak / rms if rms > 0 else np.nan,
        "bmaj_arcsec": img["bmaj_arcsec"],
        "bmin_arcsec": img["bmin_arcsec"],
        "bpa_deg": float(img["header"].get("BPA", np.nan)),
        "rms_info": rms_info,
    }


# ============================================================================
# MS-querying utilities (CASA-dependent)
# ============================================================================

def _require_casa():
    if not _HAS_CASA:
        raise RuntimeError(
            "casatools/casatasks are not available in this Python environment. "
            "AJISAI MS-level operations require CASA. Run from within CASA or "
            "install modular CASA (pip install casatools casatasks)."
        )


# Phase 1.4: analysisUtils dependency removed. All MS-level queries are now
# implemented natively in ajisai_ms_utils.py.


def _safe_rmtree_path(path) -> None:
    """Remove a path (file or directory) within the workdir safely.

    Refuses to operate on paths outside any obvious working directory (i.e.,
    paths that resolve to something like '/' or contain no parent directory
    in the workspace). This is the replacement for the dangerous
    ``os.system("rm -rf sc*")`` calls in the original AJISAI.py.
    """
    p = Path(path)
    if not p.exists():
        return
    # Refuse paths whose absolute resolution has fewer than 3 parts (e.g. '/', '/tmp')
    resolved = p.resolve()
    if len(resolved.parts) < 3:
        warnings.warn(f"Refusing to rmtree suspicious path: {resolved}")
        return
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except OSError:
            pass


def _safe_rmtree_glob(parent, pattern: str) -> None:
    """Remove paths matching ``pattern`` under ``parent`` (no path traversal)."""
    parent = Path(parent)
    if not parent.is_dir():
        return
    # Strict: pattern must NOT contain '..' or absolute paths
    if ".." in pattern or pattern.startswith("/"):
        warnings.warn(f"Refusing dangerous pattern: {pattern!r}")
        return
    for p in parent.glob(pattern):
        _safe_rmtree_path(p)


def relabel_J2000_to_ICRS(msfile: str, verbose: bool = False) -> None:
    """
    Relabel J2000 frame entries in FIELD and SOURCE tables to ICRS.

    Ported from AJISAI.py. Addresses the J2000/ICRS confusion that has
    historically caused CASA imaging tasks to error on archival ALMA data.
    Operates in place on the MS; safe to call repeatedly (idempotent).
    """
    _require_casa()
    tb = casatools.table()
    # FIELD table: PhaseDir_Ref / DelayDir_Ref / RefDir_Ref columns use integer
    # codes; 0 = J2000, 21 = ICRS in CASA's MEASINFO convention.
    tb.open(msfile + "/FIELD", nomodify=False)
    try:
        for colname in ("PhaseDir_Ref", "DelayDir_Ref", "RefDir_Ref"):
            if colname in tb.colnames():
                a = tb.getcol(colname)
                changed = False
                for i in range(len(a)):
                    if a[i] == 0:
                        a[i] = 21
                        changed = True
                if changed:
                    tb.putcol(colname, a)
                    if verbose:
                        print(f"[AJISAI] FIELD.{colname}: J2000→ICRS")
    finally:
        tb.close()
    # SOURCE table: DIRECTION column has a frame keyword
    tb.open(msfile + "/SOURCE", nomodify=False)
    try:
        try:
            keys = tb.getcolkeywords("DIRECTION")
            if keys.get("MEASINFO", {}).get("Ref") == "J2000":
                keys["MEASINFO"]["Ref"] = "ICRS"
                tb.putcolkeywords("DIRECTION", keys)
                if verbose:
                    print("[AJISAI] SOURCE.DIRECTION: J2000→ICRS")
        except Exception:
            pass
    finally:
        tb.close()


def _max_pixel_world_coords(imagename: str) -> Dict[str, str]:
    """World coordinates of the brightest pixel in a CASA image.

    Uses casatasks.imstat (always succeeds; the max pixel always exists)
    and reads the image's reference frame via casatools.image.

    Returns
    -------
    dict with keys:
        ra    : RA sexagesimal string (no whitespace)
        dec   : Dec sexagesimal string (no whitespace)
        frame : "ICRS", "J2000", "B1950", etc. (whatever the image declares)
    """
    _require_casa()
    stats = casatasks.imstat(imagename=imagename, stokes="I")
    maxposf = stats["maxposf"]
    # maxposf is comma-separated: "RA, Dec, Stokes, Frequency"
    parts = [p.strip() for p in maxposf.split(",")]
    if len(parts) < 2:
        raise RuntimeError(
            f"imstat returned unexpected maxposf format: {maxposf!r}"
        )
    ra = parts[0].replace(" ", "")
    dec = parts[1].replace(" ", "")
    # Read the image's reference frame
    ia = casatools.image()
    ia.open(imagename)
    try:
        cs = ia.coordsys()
        try:
            ref = cs.referencecode("direction")
            # referencecode returns either a string or a list of strings
            frame = ref[0] if isinstance(ref, (list, tuple)) else ref
            frame = str(frame).upper()
        finally:
            cs.done()
    finally:
        ia.close()
    return {"ra": ra, "dec": dec, "frame": frame}


def _compute_avg_integration_time(vis: str, field: Optional[str] = None) -> float:
    """Average per-integration (dump) time from listobs metadata.

    Used to resolve solint expressions like '6*IT'. Returns seconds.
    Note: despite its position in CASA's listobs output (under each scan),
    'IntegrationTime' is the per-dump time, not the scan duration.
    """
    _require_casa()
    metadata = casatasks.listobs(vis=vis, verbose=True, intent="OBSERVE_TARGET#ON_SOURCE")
    times = []
    for key, val in metadata.items():
        if not key.startswith("scan"):
            continue
        # listobs scan entries: {scan_id: {field_id_str: {..., 'IntegrationTime': N}}}
        for subkey, info in val.items():
            if isinstance(info, dict) and "IntegrationTime" in info:
                if field is None or str(info.get("FieldId", subkey)) == str(field):
                    times.append(info["IntegrationTime"])
    if not times:
        # Fallback: read INTERVAL column directly from MAIN
        tb = casatools.table()
        tb.open(vis)
        try:
            intervals = tb.getcol("INTERVAL")
        finally:
            tb.close()
        if intervals.size == 0:
            raise RuntimeError(f"Could not determine integration time for {vis}")
        return float(np.median(intervals))
    return float(np.mean(times))


def list_target_fields(vis: str, intent: str = "OBSERVE_TARGET#ON_SOURCE") -> List[Dict]:
    """
    Enumerate target field IDs and names in an MS.

    Used by users who want to loop over multiple targets:

        for f in list_target_fields(vis):
            cfg = AJISAIConfig(vis=vis, field=f["id"], projname=f["name"])
            AJISAI(cfg).run()

    Returns a list of dicts: [{"id": "0", "name": "V883_Ori"}, ...]
    """
    _require_casa()
    msmd = casatools.msmetadata()
    msmd.open(vis)
    try:
        try:
            field_ids = msmd.fieldsforintent(intent)
        except Exception:
            # Fallback: any field that has rows in MAIN
            field_ids = msmd.fieldsforscans(msmd.scannumbers())
        names = msmd.fieldnames()
        out = []
        for fid in sorted(set(int(f) for f in field_ids)):
            out.append({"id": str(fid), "name": str(names[fid])})
        return out
    finally:
        msmd.close()


# Antenna queries (get_antenna_flag_stats, get_antenna_positions) are
# provided by ajisai_ms_utils and re-exported above for backward compatibility.


def select_refant(
    vis: str,
    strategy: str = "hybrid",
    flag_threshold: float = 0.25,
    manual: Optional[str] = None,
    field: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Select a reference antenna using one of four strategies.

    Strategies:
        "hybrid" (default):
            1. Compute per-antenna flag fraction.
            2. Keep antennas with flag_fraction < flag_threshold.
            3. Among the kept set, pick the one closest to the XY geometric
               center of the array.
            4. If the kept set is empty, fall back to "geometric_center"
               over all antennas.
        "geometric_center":
            Pure XY centroid nearest (current AJISAI behaviour).
        "flag_stats":
            Lowest flag fraction (ignoring position).
        "manual":
            Use the antenna name provided in ``manual``.

    Returns a structured dict with the selected antenna and full justification
    information suitable for the justification log and the visualization plot.
    """
    _require_casa()
    positions = get_antenna_positions(vis)
    names = list(positions.keys())
    xs = np.array([positions[n][0] for n in names])
    ys = np.array([positions[n][1] for n in names])
    cx, cy = float(xs.mean()), float(ys.mean())
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    # Flag stats are needed by hybrid and flag_stats
    if strategy in ("hybrid", "flag_stats"):
        flag_frac = get_antenna_flag_stats(vis, field=field)
    else:
        flag_frac = {n: 0.0 for n in names}

    flags = np.array([flag_frac.get(n, 1.0) for n in names])

    if strategy == "manual":
        if manual is None or manual not in names:
            raise ValueError(f"strategy='manual' requires a valid antenna name from {names!r}")
        idx = names.index(manual)
        chosen = manual
        reason = "user-specified (strategy='manual')"
        fallback_used = False

    elif strategy == "geometric_center":
        idx = int(np.argmin(dist))
        chosen = names[idx]
        reason = "geometric center of XY-projected antenna positions (no flag filter)"
        fallback_used = False

    elif strategy == "flag_stats":
        idx = int(np.argmin(flags))
        chosen = names[idx]
        reason = f"minimum flag fraction = {flags[idx]:.3f}"
        fallback_used = False

    elif strategy == "hybrid":
        candidate_mask = flags < flag_threshold
        if candidate_mask.sum() == 0:
            # Fallback: pure geometric center
            idx = int(np.argmin(dist))
            chosen = names[idx]
            reason = (f"no antenna has flag_fraction < {flag_threshold:.2f}; "
                      f"fell back to geometric_center over all antennas")
            fallback_used = True
        else:
            # Among candidates: nearest to geometric center
            cand_dist = np.where(candidate_mask, dist, np.inf)
            idx = int(np.argmin(cand_dist))
            chosen = names[idx]
            reason = (f"hybrid: nearest to XY geometric center among "
                      f"{int(candidate_mask.sum())} antennas with "
                      f"flag_fraction < {flag_threshold:.2f}")
            fallback_used = False
    else:
        raise ValueError(f"Unknown refant strategy: {strategy!r}")

    return {
        "refant": chosen,
        "strategy": strategy,
        "flag_threshold": flag_threshold,
        "fallback_used": fallback_used,
        "reason": reason,
        "antenna_names": names,
        "antenna_x": xs.tolist(),
        "antenna_y": ys.tolist(),
        "antenna_flag_frac": flags.tolist(),
        "antenna_dist_from_center": dist.tolist(),
        "center_xy": (cx, cy),
        "chosen_index": idx,
    }


def plot_refant_selection(
    refant_info: Dict[str, Any],
    outpath: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot A: single scatter plot showing refant selection rationale.

    - Antennas plotted at their XY positions (units = meters from array origin).
    - Color = flagged fraction (viridis colormap, 0 to 1).
    - Antennas above the flag threshold are crossed out with red 'x'.
    - Selected refant is circled in green.
    - Geometric center is marked with a black cross.
    - Antenna names are annotated.

    Intended as a justification artifact: one PNG that visually defends the
    refant choice. Use as supplementary material in publications.
    """
    info = refant_info
    fig, ax = plt.subplots(figsize=(9, 9))

    xs = np.array(info["antenna_x"])
    ys = np.array(info["antenna_y"])
    flags = np.array(info["antenna_flag_frac"])
    names = info["antenna_names"]
    cx, cy = info["center_xy"]
    idx = info["chosen_index"]
    threshold = info["flag_threshold"]

    # Scatter: color by flag fraction
    sc = ax.scatter(
        xs - cx, ys - cy, c=flags, cmap="viridis",
        s=120, edgecolors="black", linewidth=0.5,
        vmin=0.0, vmax=max(0.5, float(flags.max())),
    )
    plt.colorbar(sc, ax=ax, label="flagged fraction", shrink=0.7)

    # Mark excluded antennas (above threshold)
    excluded = flags >= threshold
    if excluded.any():
        ax.scatter(
            (xs - cx)[excluded], (ys - cy)[excluded],
            marker="x", s=180, c="red", linewidth=2.5,
            label=f"excluded (flag ≥ {threshold:.2f})",
        )

    # Circle the chosen refant
    ax.scatter(
        xs[idx] - cx, ys[idx] - cy,
        s=380, facecolors="none", edgecolors="lime", linewidth=3,
        label=f"refant = {info['refant']}",
    )

    # Mark geometric center
    ax.plot(0, 0, "k+", markersize=18, markeredgewidth=2, label="XY geometric center")

    # Annotate antenna names
    for x, y, n in zip(xs, ys, names):
        ax.annotate(str(n), (x - cx, y - cy), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    # Cosmetics
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlabel("X offset from array center [m]")
    ax.set_ylabel("Y offset from array center [m]")
    ttl = title or (
        f"AJISAI refant selection (strategy={info['strategy']}) → {info['refant']}\n"
        f"{info['reason']}"
    )
    ax.set_title(ttl, fontsize=10)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main class
# ============================================================================

class AJISAI:
    """Automated Justification-based Imaging and Self-calibration.

    **Design philosophy**

    - Fire-and-forget: user provides minimal input, AJISAI does the rest.
    - Justification-based: every parameter choice is logged with rationale.
    - Subclassable: each strategy step is an overridable method.

    **Typical usage**

    .. code-block:: python

       cfg = AJISAIConfig(vis="data.ms")
       aj = AJISAI(cfg).run()
       print(aj.best_image, aj.justification)

    **Overriding a strategy** (e.g. custom refant chooser):

    .. code-block:: python

       class MyAJISAI(AJISAI):
           def select_refant(self):
               return "DA46"
       MyAJISAI(cfg).run()
    """

    def __init__(self, config: AJISAIConfig):
        self.cfg: AJISAIConfig = config
        # Mutable state populated during run()
        self.metrics: List[Dict[str, Any]] = []
        self.justification: Dict[str, Any] = {
            "ajisai_version": __version__,
            "casa_version": _casa_version_string(),
            "started_at": None,
            "finished_at": None,
            "input_config": asdict(config),
            "derived": {},
            "iterations": [],
            "best": None,
        }
        self._derived: Dict[str, Any] = {}
        self.workdir: Optional[Path] = None
        self.best_image: Optional[str] = None
        self.best_fits: Optional[str] = None
        self.best_metric_value: Optional[float] = None

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def run(self) -> "AJISAI":
        """Top-level pipeline orchestrator.

        **Pipeline order**

        1. Validate inputs.
        2. Set up working directory.
        3. Inspect MS (refant, frequency, integration time, etc.).
        4. Compute imaging parameters (cell, imsize, scales).
        5. Make dirty image (initial RMS reference).
        6. Apply phase shift if requested (off by default).
        7. Make initial CLEAN image (iteration 0).
        8. Run each self-cal iteration in the schedule.
        9. Select the best iteration by quality metric.
        10. Write summary CSV, summary PNG, and justification JSON.
        """
        self.justification["started_at"] = datetime.utcnow().isoformat() + "Z"
        if self.cfg.show_banner:
            self._print_banner()
        try:
            self._validate_inputs()
            self._setup_workdir()
            self._inspect_ms()
            self._compute_imaging_params()
            self._make_dirty_image()
            self._apply_phase_shift_if_requested()
            self._make_initial_clean()
            for i, step in enumerate(self.cfg.schedule.steps, 1):
                self._run_iteration(i, step)
            self._select_best()
            self._write_summary()
        finally:
            self.justification["finished_at"] = datetime.utcnow().isoformat() + "Z"
            self._write_justification_json()
        return self

    # ----------------------------------------------------------------------
    # Banner
    # ----------------------------------------------------------------------
    @staticmethod
    def _print_banner() -> None:
        """Print the AJISAI ASCII banner. Called from run() by default."""
        print("--------------------------------------")
        print(r"""
    _         _  ___  ____      _     ___
   / \       | ||_ _|/ ___|    / \   |_ _|
  / _ \   _  | | | | \___ \   / _ \   | |
 / ___ \ | |_| | | |  ___) | / ___ \  | |
/_/   \_\ \___/ |___||____/ /_/   \_\|___|

ALMA
Justification-based
Imaging
Self-calibration and
Automation
Infrastructure
        """)
        print("--------------------------------------")
        print(f"  AJISAI version {__version__}")
        print("--------------------------------------")

    # ----------------------------------------------------------------------
    # Overridable strategy hooks (Phase 1 surface)
    # ----------------------------------------------------------------------
    def select_refant(self) -> Dict[str, Any]:
        """Choose reference antenna. Override for custom strategies."""
        return select_refant(
            vis=self.cfg.vis,
            strategy=self.cfg.refant_strategy,
            flag_threshold=self.cfg.refant_flag_threshold,
            manual=self.cfg.refant_manual,
            field=self.cfg.field,
        )

    def compute_image_stats(self, fitsfile: str) -> Dict[str, Any]:
        """Compute peak/RMS/SNR/DR for an image. Override for custom metrics."""
        return compute_image_stats(
            fitsfile=fitsfile,
            rms_method=self.cfg.rms_method,
            exclude_factor=self.cfg.rms_exclude_beam_factor,
            sigma=self.cfg.rms_sigma,
            maxiters=self.cfg.rms_maxiters,
        )

    def select_best_iteration(self) -> int:
        """Pick the best iteration index by DR (or SNR). Skipped iterations
        (no image produced) are excluded automatically since their metric is None.

        Anomalous iterations (status='anomaly') are NOT filtered out — if they
        happen to have the highest DR, they are still selected. The user can
        inspect the justification log to understand the anomaly. This is
        consistent with the "all iterations complete + DR best + log only" design.
        """
        metric_key = "dynamic_range" if self.cfg.quality_metric == "dynamic_range" else "snr"
        best_i = -1
        best_v = -np.inf
        for i, m in enumerate(self.metrics):
            v = m.get(metric_key)
            if v is None or not np.isfinite(v):
                continue
            if v > best_v:
                best_v = v
                best_i = i
        return best_i

    # ----------------------------------------------------------------------
    # Internal pipeline methods
    # ----------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        cfg = self.cfg
        if not os.path.exists(cfg.vis):
            raise FileNotFoundError(f"vis not found: {cfg.vis}")
        if cfg.refant_strategy == "manual" and cfg.refant_manual is None:
            raise ValueError("refant_strategy='manual' requires refant_manual to be set")
        if cfg.rms_method == "annulus":
            warnings.warn(
                "rms_method='annulus' is the legacy method and requires a target_radius; "
                "consider switching to 'sigma_clip_excl' (the validated default)."
            )
        # mask_mode validation
        valid_modes = ("auto-multithresh", "interactive", "user", "none")
        if cfg.imaging.mask_mode not in valid_modes:
            raise ValueError(
                f"imaging.mask_mode must be one of {valid_modes}; "
                f"got {cfg.imaging.mask_mode!r}"
            )
        if cfg.imaging.mask_mode == "interactive":
            ver = _casa_version_string()
            warnings.warn(
                f"mask_mode='interactive' requires CASA <= 6.6 because the "
                f"casaviewer GUI was removed in CASA 6.7+. Detected CASA "
                f"version: {ver}. If interactive tclean fails to open, switch "
                f"to mask_mode='auto-multithresh' (default) or 'user' with a "
                f"pre-made mask file."
            )
        if cfg.imaging.mask_mode == "user" and not cfg.imaging.user_mask:
            raise ValueError(
                "imaging.mask_mode='user' requires imaging.user_mask to be set "
                "to the path of a pre-made mask file"
            )
        # phase_center validation
        if cfg.phase_center is not None:
            if not isinstance(cfg.phase_center, tuple) or len(cfg.phase_center) != 3:
                raise ValueError(
                    "phase_center must be a 3-tuple (ra, dec, frame); "
                    f"got {cfg.phase_center!r}"
                )
            ra, dec, frame = cfg.phase_center
            if not isinstance(ra, str) or not isinstance(dec, str):
                raise ValueError(
                    "phase_center RA and Dec must be sexagesimal strings, "
                    f"e.g. ('16:25:45.0', '-24:12:23.0', 'ICRS'); got ({ra!r}, {dec!r})"
                )
            if frame not in ("ICRS", "J2000"):
                raise ValueError(
                    f"phase_center frame must be 'ICRS' or 'J2000'; got {frame!r}"
                )
            if not cfg.phase_shift:
                warnings.warn(
                    "phase_center is set but phase_shift=False; the coordinates "
                    "will be ignored. Set phase_shift=True to apply the shift."
                )

    def _setup_workdir(self) -> None:
        """Create a unique output directory per field/projname."""
        cfg = self.cfg
        # Determine projname
        if cfg.projname is None:
            base = os.path.basename(cfg.vis).replace(".ms", "")
            field_part = f"_field{cfg.field}" if cfg.field is not None else ""
            projname = f"{base}{field_part}"
        else:
            projname = cfg.projname
        # Determine output dir
        if cfg.results_dir is None:
            workdir = Path.cwd() / f"ajisai_{projname}"
        else:
            workdir = Path(cfg.results_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "intermediate").mkdir(exist_ok=True)
        self.workdir = workdir
        self._derived["projname"] = projname
        self._derived["workdir"] = str(workdir)
        if self.cfg.verbose:
            print(f"[AJISAI] working directory: {workdir}")

    # ----------------------------------------------------------------------
    # _inspect_ms: refant, scan/integration time, frequency, on-source time,
    # J2000→ICRS relabel, target field validation
    # ----------------------------------------------------------------------
    def _inspect_ms(self) -> None:
        """Inspect MS and record derived parameters with full justification."""
        cfg = self.cfg
        # --- Refant selection (validated, with visualization) ---
        refant_info = self.select_refant()
        self._derived["refant"] = refant_info["refant"]
        self.justification["derived"]["refant"] = refant_info
        if self.workdir is not None:
            png_path = self.workdir / "ajisai_refant_selection.png"
            plot_refant_selection(
                refant_info, outpath=str(png_path),
                title=f"AJISAI refant: {self._derived.get('projname', '')}",
            )
            if cfg.verbose:
                print(f"[AJISAI] refant = {refant_info['refant']} "
                      f"({refant_info['reason']})")

        # --- Field validation ---
        self._validate_field()

        # --- J2000 → ICRS relabel (CASA hygiene; preserves AJISAI behavior) ---
        relabel_J2000_to_ICRS(cfg.vis, verbose=cfg.verbose)

        # --- Integration time (per-dump time, used to resolve "N*IT" solints) ---
        # NOTE: original AJISAI named this 'avg_scantime_sec' but it is in fact
        # the per-integration dump time (CASA's listobs returns this in the
        # 'IntegrationTime' field of each scan).
        inttime_sec = _compute_avg_integration_time(cfg.vis, cfg.field)
        self._derived["avg_int_time_sec"] = inttime_sec
        self.justification["derived"]["avg_int_time_sec"] = {
            "value": inttime_sec,
            "reason": ("per-integration (dump) time averaged across target scans; "
                       "used to resolve solint expressions like '6*IT'"),
        }

        # --- On-source time (Phase 1.4: native casatools-based) ---
        try:
            onsource_sec = get_on_source_time(cfg.vis, field=cfg.field)
            self._derived["onsource_time_sec"] = onsource_sec
            self.justification["derived"]["onsource_time_sec"] = {
                "value": onsource_sec,
                "reason": "sum of scan time intervals for OBSERVE_TARGET#ON_SOURCE "
                          "scans (ajisai_ms_utils.get_on_source_time)",
            }
        except Exception as e:
            warnings.warn(f"get_on_source_time failed: {e}")
            self._derived["onsource_time_sec"] = None

        # --- Median frequency (Phase 1.4: native casatools-based) ---
        try:
            freq_hz = get_median_frequency(cfg.vis)
            self._derived["median_freq_hz"] = float(freq_hz)
            self.justification["derived"]["median_freq_hz"] = {
                "value": float(freq_hz),
                "reason": "median of mean frequency over target-intent SPWs "
                          "(ajisai_ms_utils.get_median_frequency)",
            }
        except Exception as e:
            warnings.warn(f"get_median_frequency failed: {e}")
            self._derived["median_freq_hz"] = None

        # --- SPW info (needed for split width) ---
        tb = casatools.table()
        tb.open(cfg.vis + "/SPECTRAL_WINDOW")
        try:
            num_chan = tb.getcol("NUM_CHAN").tolist()
            nspw = len(num_chan)
        finally:
            tb.close()
        self._derived["num_chan_per_spw"] = num_chan
        self._derived["nspw"] = nspw
        # Use the first spw's channel count for chavg width (matches original AJISAI)
        self._derived["chavg_width"] = num_chan[0]

        if cfg.verbose:
            print(f"[AJISAI] integration time      : {inttime_sec:.2f} s")
            if self._derived.get("onsource_time_sec"):
                print(f"[AJISAI] on-source time        : {self._derived['onsource_time_sec']/60:.2f} min")
            print(f"[AJISAI] median frequency      : "
                  f"{(self._derived.get('median_freq_hz') or 0)/1e9:.3f} GHz")
            print(f"[AJISAI] spectral windows      : {nspw} (channels: {num_chan})")

    def _validate_field(self) -> None:
        """Validate cfg.field exists in MAIN table; auto-detect if None."""
        cfg = self.cfg
        tb = casatools.table()
        tb.open(cfg.vis + "/FIELD")
        try:
            field_names = list(tb.getcol("NAME"))
        finally:
            tb.close()
        tb.open(cfg.vis)
        try:
            present_fids = sorted(set(int(x) for x in np.unique(tb.getcol("FIELD_ID"))))
        finally:
            tb.close()

        if cfg.field is None:
            if len(present_fids) == 0:
                raise RuntimeError(f"No FIELD_ID found in MAIN of {cfg.vis}")
            if len(present_fids) > 1:
                warnings.warn(f"Multiple fields {present_fids} present and "
                              f"cfg.field=None; using first ({present_fids[0]}). "
                              f"For multi-field MS, set cfg.field explicitly or "
                              f"loop with list_target_fields().")
            chosen_fid = present_fids[0]
            self._derived["resolved_field_id"] = str(chosen_fid)
            self._derived["resolved_field_name"] = str(field_names[chosen_fid])
        else:
            # Allow either integer ID or NAME
            try:
                fid = int(cfg.field)
                if fid not in present_fids:
                    raise ValueError(f"field id {fid} not present in MAIN "
                                     f"(present: {present_fids})")
                self._derived["resolved_field_id"] = str(fid)
                self._derived["resolved_field_name"] = str(field_names[fid])
            except ValueError:
                # treat as NAME
                matches = [i for i, n in enumerate(field_names) if n == cfg.field]
                if not matches:
                    raise ValueError(f"field name {cfg.field!r} not found in "
                                     f"{cfg.vis}/FIELD")
                if matches[0] not in present_fids:
                    raise ValueError(f"field {cfg.field!r} has no rows in MAIN")
                self._derived["resolved_field_id"] = str(matches[0])
                self._derived["resolved_field_name"] = str(cfg.field)

        if cfg.verbose:
            print(f"[AJISAI] field                 : "
                  f"id={self._derived['resolved_field_id']}, "
                  f"name={self._derived['resolved_field_name']}")

    # ----------------------------------------------------------------------
    # _compute_imaging_params: cellsize, imsize, multiscale scales, MRS
    # ----------------------------------------------------------------------
    def _compute_imaging_params(self) -> None:
        """Determine cellsize, imsize, multiscale scales from baseline statistics.

        Uses ajisai_ms_utils (native casatools-based replacement for
        analysisUtils.pickCellSize and getBaselineStats).
        """
        if not _HAS_MS_UTILS:
            raise RuntimeError(
                "ajisai_ms_utils is not importable. Place ajisai_ms_utils.py "
                "alongside AJISAI_v2.py on sys.path."
            )
        cfg = self.cfg

        # --- MRS from 5th-percentile shortest baseline (ALMA Tech Handbook Eq 7.7) ---
        try:
            short_baseline_m = get_baseline_at_percentile(cfg.vis, percentile=5)
            freq_hz = self._derived["median_freq_hz"]
            wavelength_m = 2.99792458e8 / freq_hz
            mrs_arcsec = (0.983 * 180 * 3600 / np.pi) * wavelength_m / short_baseline_m
        except Exception as e:
            warnings.warn(f"MRS computation failed: {e}")
            mrs_arcsec = None
        self._derived["mrs_arcsec"] = mrs_arcsec
        self.justification["derived"]["mrs_arcsec"] = {
            "value": mrs_arcsec,
            "reason": ("0.983 * lambda / (5th-percentile shortest baseline); "
                       "ALMA Technical Handbook Eq 7.7"),
        }

        # --- cellsize, imsize (uvtaper-aware) ---
        cellpix = cfg.imaging.cellpix
        cell_info = pick_cell_imsize(
            cfg.vis,
            npix_per_beam=cellpix,
            max_baseline_percentile=90,
            pb_level=0.2,
            uvtaper=cfg.imaging.uvtaper,
        )
        cell_arcsec = cell_info["cell_arcsec"]
        imsize_xy = cell_info["imsize_xy"]
        self._derived["cell_arcsec"] = cell_arcsec
        self._derived["cell"] = cell_info["cell"]
        self._derived["imsize"] = imsize_xy
        self._derived["fov_arcsec"] = cell_info["fov_arcsec"]
        self._derived["primary_beam_arcsec"] = cell_info["primary_beam_arcsec"]
        self._derived["untapered_beam_arcsec"] = cell_info["beam_fwhm_arcsec"]
        self._derived["taper_image_fwhm_arcsec"] = cell_info["taper_image_fwhm_arcsec"]
        self._derived["predicted_effective_beam_arcsec"] = (
            cell_info["predicted_effective_beam_arcsec"]
        )
        self.justification["derived"]["cellsize"] = {
            "value_arcsec": cell_arcsec,
            "reason": cell_info["justification"]["description"],
        }
        self.justification["derived"]["imsize"] = {
            "value": imsize_xy,
            "reason": (f"covers down to {0.2:.0%} primary-beam attenuation "
                       f"(Baars taper); rounded to smallest 5-smooth integer"),
        }

        # --- multiscale scales: [0, 1, 3] * effective beam in pixels ---
        # When uvtaper widens the beam, the scales (in pixels) stay the same
        # because cell size scales with the beam.
        scales = [0, cellpix, 3 * cellpix]
        self._derived["scales"] = scales
        self.justification["derived"]["scales"] = {
            "value": scales,
            "reason": "[0, 1, 3] * synthesized beam (in pixels); "
                      "Cornwell 2008 multiscale recommendation",
        }

        if cfg.verbose:
            print(f"[AJISAI] cell                  : {cell_arcsec:.5f} arcsec")
            print(f"[AJISAI] imsize                : {imsize_xy}")
            print(f"[AJISAI] FoV                   : {self._derived['fov_arcsec']:.2f} arcsec")
            print(f"[AJISAI] primary beam (FWHM)   : {cell_info['primary_beam_arcsec']:.2f} arcsec")
            print(f"[AJISAI] untapered beam        : {cell_info['beam_fwhm_arcsec']:.4f} arcsec")
            if cell_info["taper_image_fwhm_arcsec"] > 0.0:
                print(f"[AJISAI] uvtaper input         : {list(cfg.imaging.uvtaper)}")
                print(f"[AJISAI] taper image FWHM      : "
                      f"{cell_info['taper_image_fwhm_arcsec']:.4f} arcsec")
                print(f"[AJISAI] predicted eff. beam   : "
                      f"{cell_info['predicted_effective_beam_arcsec']:.4f} arcsec "
                      f"(actual will be verified after dirty image)")
            print(f"[AJISAI] multiscale scales     : {scales} pix")
            if mrs_arcsec is not None:
                print(f"[AJISAI] MRS                   : {mrs_arcsec:.2f} arcsec")

    # ----------------------------------------------------------------------
    # _apply_phase_shift_if_requested: optional MS shift to bring source on-center
    # ----------------------------------------------------------------------
    def _apply_phase_shift_if_requested(self) -> None:
        """Optionally shift the MS so the target source is at the phase center.

        Default behaviour (``phase_shift=False``): this method is a no-op
        and the pipeline operates on the original input MS.

        With ``phase_shift=True``:

        * If ``cfg.phase_center is None``: the source location is auto-detected
          by reading the world coordinates of the brightest pixel of the
          dirty image (method A: pixel-accurate, never fails, no user input).

        * If ``cfg.phase_center`` is a ``(ra, dec, frame)`` tuple where
          ``frame`` is ``"ICRS"`` or ``"J2000"``: the user-supplied
          coordinates are used. This route is intended for advanced users
          who need sub-pixel accuracy (e.g. from their own imfit on the
          dirty image, or from VLBI / multi-wavelength astrometry).

        Frame handling:
            ``casatasks.fixplanets`` only supports the ``J2000`` frame.
            ICRS inputs are converted to J2000 with astropy
            (``icrs_to_j2000``) before fixplanets is called. The MS frame
            label is restored to ICRS afterwards via
            ``relabel_J2000_to_ICRS``, matching the original AJISAI behavior.

        The shifted MS is written to
        ``<workdir>/intermediate/phaseshifted.ms``, and
        ``self._derived['working_vis']`` is updated to point to it. All
        subsequent CASA tasks (initial CLEAN, self-cal iterations) operate
        on the shifted MS.
        """
        cfg = self.cfg
        # Default: working_vis is the original input MS.
        # (Set unconditionally here so downstream methods can always use it.)
        self._derived.setdefault("working_vis", cfg.vis)

        if not cfg.phase_shift:
            self._derived["phase_shift_applied"] = False
            self.justification["derived"]["phase_shift"] = {
                "applied": False,
                "reason": "phase_shift=False (default; no correction applied)",
            }
            if cfg.verbose:
                print("[AJISAI] phase shift           : skipped (phase_shift=False)")
            return

        # === Resolve the target coordinates ===
        if cfg.phase_center is not None:
            ra, dec, frame = cfg.phase_center
            frame = frame.upper()
            source_label = "user-specified"
        else:
            dirty_image = self.workdir / "intermediate" / "dirty.image"
            if not dirty_image.exists():
                raise RuntimeError(
                    "Phase shift auto-detection requires the dirty image, "
                    "but dirty.image was not found at "
                    f"{dirty_image}. _make_dirty_image must run first."
                )
            coords = _max_pixel_world_coords(str(dirty_image))
            ra, dec, frame = coords["ra"], coords["dec"], coords["frame"]
            source_label = (
                f"auto: max pixel of dirty image (frame={frame})"
            )

        # === Convert to J2000 for fixplanets ===
        if frame == "ICRS":
            j2000_radec = icrs_to_j2000(f"{ra} {dec}")
            # j2000_radec is "HH:MM:SS.s DD:MM:SS.s"
        elif frame == "J2000":
            j2000_radec = f"{ra} {dec}"
        else:
            # Image was in some other frame (e.g. B1950); refuse rather than guess
            raise RuntimeError(
                f"Auto-detected image frame is {frame!r}, but phase shifting "
                f"only supports ICRS or J2000. Specify cfg.phase_center "
                f"explicitly with a supported frame."
            )

        # === Phaseshift to new MS ===
        shifted_vis = str(self.workdir / "intermediate" / "phaseshifted.ms")
        _safe_rmtree_path(shifted_vis)
        casatasks.phaseshift(
            vis=cfg.vis,
            outputvis=shifted_vis,
            keepmms=True,
            field=self._derived["resolved_field_id"],
            spw="",
            scan="",
            intent="",
            array="",
            observation="",
            datacolumn="all",
            phasecenter=f"{frame} {ra} {dec}",
        )

        # === Fix planet labels (J2000 only) ===
        casatasks.fixplanets(
            vis=shifted_vis,
            field=self._derived["resolved_field_id"],
            direction=f"J2000 {j2000_radec}",
        )

        # === Restore ICRS label on the shifted MS (CASA hygiene) ===
        relabel_J2000_to_ICRS(shifted_vis, verbose=cfg.verbose)

        # === Update state ===
        self._derived["working_vis"] = shifted_vis
        self._derived["phase_shift_applied"] = True
        self.justification["derived"]["phase_shift"] = {
            "applied": True,
            "source": source_label,
            "input_frame": frame,
            "input_ra": ra,
            "input_dec": dec,
            "j2000_for_fixplanets": j2000_radec,
            "shifted_vis": shifted_vis,
            "reason": (
                f"shifted source to phase center; source location was "
                f"{source_label}; fixplanets called in J2000 then MS frame "
                f"label restored to ICRS"
            ),
        }

        if cfg.verbose:
            print(f"[AJISAI] phase shift applied   : {source_label}")
            print(f"[AJISAI]   target {frame:<5s}        : {ra}, {dec}")
            print(f"[AJISAI]   J2000 (fixplanets)   : {j2000_radec}")
            print(f"[AJISAI]   shifted vis          : {shifted_vis}")

    # ----------------------------------------------------------------------
    # _make_dirty_image: initial RMS reference (no CLEAN, niter=0)
    # ----------------------------------------------------------------------
    def _make_dirty_image(self) -> None:
        """Run tclean(niter=0) to estimate initial RMS for threshold setting."""
        cfg = self.cfg
        assert self.workdir is not None
        base = self.workdir / "intermediate" / "dirty"
        _safe_rmtree_glob(self.workdir / "intermediate", "dirty.*")

        casatasks.tclean(
            vis=cfg.vis,
            imagename=str(base),
            datacolumn="data",
            specmode="mfs",
            niter=0,
            threshold="0mJy",
            deconvolver="hogbom",
            gridder=cfg.imaging.gridder,
            interactive=False,
            imsize=self._derived["imsize"],
            cell=self._derived["cell"],
            weighting=cfg.imaging.weighting,
            robust=cfg.imaging.robust,
            uvtaper=list(cfg.imaging.uvtaper),
            pbcor=cfg.imaging.pbcor,
            parallel=cfg.imaging.parallel,
            field=self._derived["resolved_field_id"],
        )
        fits_path = self.workdir / "intermediate" / "dirty.fits"
        if fits_path.exists():
            fits_path.unlink()
        casatasks.exportfits(imagename=str(base) + ".image",
                             fitsimage=str(fits_path), history=False)
        stats = self.compute_image_stats(str(fits_path))
        self._derived["dirty_rms"] = stats["rms_jy_beam"]
        self._derived["dirty_peak"] = stats["peak_jy_beam"]
        self._derived["dirty_dr"] = stats["dynamic_range"]
        self.justification["derived"]["dirty_image"] = stats

        # --- Verify actual beam vs predicted (especially important with uvtaper) ---
        actual_beam_arcsec = math.sqrt(
            (stats["bmaj_arcsec"] or 0.0) * (stats["bmin_arcsec"] or 0.0)
        )
        predicted = self._derived.get("predicted_effective_beam_arcsec", 0.0)
        beam_check = {
            "actual_beam_arcsec": actual_beam_arcsec,
            "predicted_effective_beam_arcsec": predicted,
            "ratio_actual_over_predicted": (
                actual_beam_arcsec / predicted if predicted > 0 else None
            ),
            "mismatch_warning": False,
        }
        if predicted > 0 and actual_beam_arcsec > 0:
            rel_diff = abs(actual_beam_arcsec - predicted) / predicted
            if rel_diff > 0.20:
                beam_check["mismatch_warning"] = True
                beam_check["mismatch_pct"] = rel_diff * 100
                msg = (
                    f"actual dirty-image beam ({actual_beam_arcsec:.4f} arcsec) "
                    f"differs from predicted effective beam "
                    f"({predicted:.4f} arcsec) by {rel_diff*100:.0f}%. "
                    f"Current cell size = {self._derived['cell_arcsec']:.4f} arcsec "
                    f"is based on the prediction. For tighter pixel sampling, "
                    f"consider overriding ImagingConfig.cellpix or providing "
                    f"explicit cell/imsize."
                )
                warnings.warn(f"[AJISAI] {msg}")
                if cfg.verbose:
                    print(f"[AJISAI]   WARN: {msg}")
        self.justification["derived"]["beam_check"] = beam_check

        if cfg.verbose:
            print(f"[AJISAI] dirty image: peak={stats['peak_jy_beam']*1e3:.3f} mJy/bm, "
                  f"RMS={stats['rms_jy_beam']*1e6:.2f} uJy/bm, "
                  f"DR={stats['dynamic_range']:.1f}")
            print(f"[AJISAI]   actual beam     : {actual_beam_arcsec:.4f} arcsec")
            if predicted > 0:
                print(f"[AJISAI]   predicted beam  : {predicted:.4f} arcsec "
                      f"({100*actual_beam_arcsec/predicted:.0f}% of prediction)")

    # ----------------------------------------------------------------------
    # _make_initial_clean: pre-selfcal CLEAN image (Iter 0), saves model
    # ----------------------------------------------------------------------
    def _make_initial_clean(self) -> None:
        """Run the first CLEAN to establish the model for the first gaincal.

        Operates on ``self._derived['working_vis']``, which is either the
        original input MS (default) or the phase-shifted MS produced by
        :meth:`_apply_phase_shift_if_requested`.
        """
        cfg = self.cfg
        assert self.workdir is not None
        working_vis = self._derived["working_vis"]
        base = self.workdir / "intermediate" / "clean_iter0"
        _safe_rmtree_glob(self.workdir / "intermediate", "clean_iter0.*")

        threshold_jy = cfg.imaging.threshold_factor * self._derived["dirty_rms"]
        # ``usemask_kwargs`` carries everything mask-related INCLUDING the
        # ``interactive`` flag. For mask_mode='interactive', interactive=True
        # is set here; in all other modes (and in subsequent self-cal
        # iterations), interactive=False.
        usemask_kwargs = self._build_usemask_kwargs()

        # First run: actual CLEAN with model save
        casatasks.tclean(
            vis=working_vis,
            imagename=str(base),
            datacolumn="data",
            specmode="mfs",
            niter=cfg.imaging.niter,
            gain=cfg.imaging.gain,
            threshold=f"{threshold_jy}Jy",
            deconvolver=cfg.imaging.deconvolver,
            scales=self._derived["scales"] if cfg.imaging.deconvolver == "multiscale" else [],
            gridder=cfg.imaging.gridder,
            cyclefactor=cfg.imaging.cyclefactor,
            imsize=self._derived["imsize"],
            cell=self._derived["cell"],
            weighting=cfg.imaging.weighting,
            robust=cfg.imaging.robust,
            uvtaper=list(cfg.imaging.uvtaper),
            pbcor=cfg.imaging.pbcor,
            savemodel="modelcolumn",
            parallel=cfg.imaging.parallel,
            field=self._derived["resolved_field_id"],
            **usemask_kwargs,
        )
        # Verify model column was populated; if not, rerun with niter=0 + savemodel
        self._ensure_model_column(working_vis, base, threshold_jy)

        # Export FITS + compute stats
        fits_path = self.workdir / "intermediate" / "clean_iter0.fits"
        if fits_path.exists():
            fits_path.unlink()
        casatasks.exportfits(imagename=str(base) + ".image",
                             fitsimage=str(fits_path), history=False)
        stats = self.compute_image_stats(str(fits_path))

        # Record as iteration 0
        rec = {
            "iteration": 0,
            "label": "no_selfcal",
            "calmode": "-",
            "solint": "-",
            "peak_jy_beam": stats["peak_jy_beam"],
            "rms_jy_beam": stats["rms_jy_beam"],
            "snr": stats["snr"],
            "dynamic_range": stats["dynamic_range"],
            "bmaj_arcsec": stats["bmaj_arcsec"],
            "bmin_arcsec": stats["bmin_arcsec"],
            "n_gaincal_solutions": None,
            "status": "ok",
            "failure_reason": None,
            "image_path": str(base) + ".image",
            "fits_path": str(fits_path),
        }
        self.metrics.append(rec)
        self.justification["iterations"].append(rec)

        # Initial state for the iteration chain. good_vis is the MS that the
        # first gaincal will run against (= the working MS, possibly shifted).
        self._derived["good_vis"] = str(working_vis)
        self._derived["last_dr"] = stats["dynamic_range"]
        self._derived["last_rms"] = stats["rms_jy_beam"]
        self._derived["initial_mask_path"] = str(base) + ".mask"

        if cfg.verbose:
            print(f"[AJISAI] iter 0 (no selfcal)   : DR={stats['dynamic_range']:.1f}, "
                  f"peak={stats['peak_jy_beam']*1e3:.3f} mJy/bm, "
                  f"RMS={stats['rms_jy_beam']*1e6:.2f} uJy/bm")

    # ----------------------------------------------------------------------
    # _run_iteration: one self-cal iteration (gaincal + applycal + split + tclean)
    # ----------------------------------------------------------------------
    def _run_iteration(self, idx: int, step: SelfcalStep) -> None:
        """Run one self-cal iteration with failure detection."""
        cfg = self.cfg
        assert self.workdir is not None
        interm = self.workdir / "intermediate"
        caltable = interm / f"sc{idx}.pcal"
        out_vis = interm / f"selfcal_{idx}.ms"
        clean_base = interm / f"clean_sc{idx}"
        fits_path = interm / f"clean_sc{idx}.fits"

        # Cleanup any stale files from previous run (safe: only within workdir)
        _safe_rmtree_path(caltable)
        _safe_rmtree_path(out_vis)
        _safe_rmtree_glob(interm, f"clean_sc{idx}.*")

        # Resolve solint expression like "6*IT"
        solint_str = self._resolve_solint(step.solint)
        field_id = self._derived["resolved_field_id"]
        refant = self._derived["refant"]
        input_vis = self._derived["good_vis"]

        if cfg.verbose:
            print(f"[AJISAI] --- iter {idx} ({step.label}) "
                  f"calmode={step.calmode} solint={solint_str} ---")

        # === 1. gaincal ===
        n_solutions, gaincal_ok = self._run_gaincal(
            input_vis=input_vis,
            caltable=str(caltable),
            solint=solint_str,
            step=step,
            field_id=field_id,
            refant=refant,
        )
        if not gaincal_ok or n_solutions == 0:
            # No usable solutions: applycal cannot run, no new MS can be made.
            # This is the one case where the iteration is structurally unable
            # to produce output. Record it and move on; next iteration tries
            # with its own solint against the same input_vis (good_vis unchanged
            # because nothing was produced to update it to).
            self._record_skipped_iteration(
                idx, step, solint_str,
                reason=f"gaincal returned 0 solutions (all below minsnr={cfg.gaincal.minsnr})",
            )
            return

        # === 2. plot the gain table (justification artifact) ===
        self._plot_gain_table(str(caltable), idx, step)

        # === 3. applycal ===
        casatasks.applycal(
            vis=input_vis,
            gaintable=[str(caltable)],
            spwmap=[],
            calwt=cfg.gaincal.calwt,
            interp=cfg.gaincal.applycal_interp,
            applymode=cfg.gaincal.applymode,
        )

        # === 4. split corrected → new MS ===
        casatasks.split(
            vis=input_vis,
            outputvis=str(out_vis),
            datacolumn="corrected",
        )

        # === 5. tclean of the new MS, save model for next iteration ===
        # Self-cal iterations ALWAYS reuse the mask produced by the initial
        # CLEAN (interactive=False even when mask_mode='interactive').
        threshold_jy = cfg.imaging.threshold_factor * self._derived["last_rms"]
        usemask_kwargs = self._build_usemask_kwargs(
            existing_mask=self._derived.get("initial_mask_path"),
        )
        casatasks.clearcal(vis=str(out_vis))
        casatasks.tclean(
            vis=str(out_vis),
            imagename=str(clean_base),
            datacolumn="data",
            specmode="mfs",
            niter=cfg.imaging.niter,
            gain=cfg.imaging.gain,
            threshold=f"{threshold_jy}Jy",
            deconvolver=cfg.imaging.deconvolver,
            scales=self._derived["scales"] if cfg.imaging.deconvolver == "multiscale" else [],
            gridder=cfg.imaging.gridder,
            cyclefactor=cfg.imaging.cyclefactor,
            imsize=self._derived["imsize"],
            cell=self._derived["cell"],
            weighting=cfg.imaging.weighting,
            robust=cfg.imaging.robust,
            uvtaper=list(cfg.imaging.uvtaper),
            pbcor=cfg.imaging.pbcor,
            savemodel="modelcolumn",
            parallel=cfg.imaging.parallel,
            field=field_id,
            **usemask_kwargs,
        )
        self._ensure_model_column(str(out_vis), clean_base, threshold_jy)

        # === 6. compute image stats (DR-based) ===
        if fits_path.exists():
            fits_path.unlink()
        casatasks.exportfits(imagename=str(clean_base) + ".image",
                             fitsimage=str(fits_path), history=False)
        stats = self.compute_image_stats(str(fits_path))

        # === 7. anomaly detection (informational only — does NOT alter flow) ===
        is_anomaly, anomaly_reason = self._detect_iter_anomaly(stats, n_solutions)

        # === 8. record metrics ===
        rec = {
            "iteration": idx,
            "label": step.label,
            "calmode": step.calmode,
            "solint": solint_str,
            "peak_jy_beam": stats["peak_jy_beam"],
            "rms_jy_beam": stats["rms_jy_beam"],
            "snr": stats["snr"],
            "dynamic_range": stats["dynamic_range"],
            "bmaj_arcsec": stats["bmaj_arcsec"],
            "bmin_arcsec": stats["bmin_arcsec"],
            "n_gaincal_solutions": n_solutions,
            "status": "anomaly" if is_anomaly else "ok",
            "anomaly_reason": anomaly_reason,
            "image_path": str(clean_base) + ".image",
            "fits_path": str(fits_path),
        }
        self.metrics.append(rec)
        self.justification["iterations"].append(rec)

        # === 9. ALWAYS update good_vis (all iterations complete by design) ===
        # AJISAI design: anomalies are logged but do NOT trigger fallback.
        # The MS chain is always continuous. Best-iteration selection at the
        # end handles cases where an anomaly degraded the result.
        self._derived["good_vis"] = str(out_vis)
        self._derived["last_dr"] = stats["dynamic_range"]
        self._derived["last_rms"] = stats["rms_jy_beam"]

        if is_anomaly:
            warnings.warn(
                f"[AJISAI] iter {idx} ({step.label}) anomaly detected: "
                f"{anomaly_reason}. Pipeline continues; see justification.json."
            )
            if cfg.verbose:
                print(f"[AJISAI]   → ANOMALY ({anomaly_reason}). "
                      f"Pipeline continues.")
        else:
            if cfg.verbose:
                print(f"[AJISAI]   → ok. DR={stats['dynamic_range']:.1f}, "
                      f"RMS={stats['rms_jy_beam']*1e6:.2f} uJy/bm")

    # ----------------------------------------------------------------------
    # Helper methods for iteration
    # ----------------------------------------------------------------------
    def _resolve_solint(self, solint: str) -> str:
        """Resolve solint expressions: '6*IT' → '36.30s' (6 * avg integration time)."""
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*\*\s*IT\s*$", solint)
        if m:
            mult = float(m.group(1))
            it = self._derived.get("avg_int_time_sec")
            if it is None:
                raise RuntimeError("solint uses 'IT' but avg_int_time_sec not computed yet")
            return f"{mult * it:.3f}s"
        return solint  # pass-through (e.g., "inf", "60s")

    def _run_gaincal(
        self,
        input_vis: str,
        caltable: str,
        solint: str,
        step: SelfcalStep,
        field_id: str,
        refant: str,
    ) -> Tuple[int, bool]:
        """Run gaincal and return (n_solutions, success)."""
        cfg = self.cfg
        try:
            casatasks.gaincal(
                vis=input_vis,
                caltable=caltable,
                spw="*",
                field=field_id,
                selectdata=True,
                combine=cfg.gaincal.combine,
                timerange="",
                observation="",
                spwmap=[],
                solint=solint,
                refant=refant,
                minblperant=cfg.gaincal.minblperant,
                minsnr=cfg.gaincal.minsnr,
                gaintype=cfg.gaincal.gaintype,
                calmode=step.calmode,
                interp=cfg.gaincal.gaincal_interp,
                append=False,
                solnorm=step.solnorm,
            )
        except Exception as e:
            warnings.warn(f"gaincal raised: {e}")
            return 0, False

        # Count the number of solutions in the resulting caltable
        if not os.path.exists(caltable):
            return 0, False
        tb = casatools.table()
        tb.open(caltable)
        try:
            n_rows = tb.nrows()
            if n_rows > 0:
                # Some rows may be all-flagged; count un-flagged solutions
                flags = tb.getcol("FLAG")  # shape (npol, nchan, nrow)
                n_unflagged = int((~flags).any(axis=(0, 1)).sum()) if flags.size > 0 else n_rows
            else:
                n_unflagged = 0
        finally:
            tb.close()
        return n_unflagged, n_unflagged > 0

    def _plot_gain_table(self, caltable: str, idx: int, step: SelfcalStep) -> None:
        """Save a per-iteration gain plot for the justification log."""
        assert self.workdir is not None
        diag_dir = self.workdir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        phase_png = diag_dir / f"iter{idx}_{step.label}_phasegain.png"
        casaplotms.plotms(
            vis=caltable,
            xaxis="time",
            yaxis="phase",
            iteraxis="antenna",
            gridrows=9, gridcols=9,
            coloraxis="scan",
            plotrange=[0, 0, -180, 180],
            showgui=False,
            overwrite=True,
            height=2500, width=3000,
            plotfile=str(phase_png),
        )
        if step.calmode == "a":
            amp_png = diag_dir / f"iter{idx}_{step.label}_ampgain.png"
            casaplotms.plotms(
                vis=caltable,
                xaxis="time",
                yaxis="amp",
                iteraxis="antenna",
                gridrows=9, gridcols=9,
                coloraxis="scan",
                plotrange=[0, 0, 0.5, 1.5],
                showgui=False,
                overwrite=True,
                height=2500, width=3000,
                plotfile=str(amp_png),
            )

    def _ensure_model_column(self, vis: str, imagebase, threshold_jy: float) -> None:
        """Verify MODEL_DATA was written; if not, rerun tclean(niter=0, savemodel)."""
        cfg = self.cfg
        tb = casatools.table()
        tb.open(vis)
        try:
            if "MODEL_DATA" not in tb.colnames():
                model_amp = None
            else:
                dats = tb.getcol("MODEL_DATA")
                model_amp = float(np.abs(dats).mean()) if dats.size else None
        finally:
            tb.close()
        # Original AJISAI's heuristic: amp ~ 1.0 or 0.0 → not populated
        needs_rerun = model_amp is None or np.isclose(model_amp, 1.0) or np.isclose(model_amp, 0.0)
        if needs_rerun:
            if cfg.verbose:
                print("[AJISAI]   model column not populated; rerunning tclean(niter=0)")
            casatasks.tclean(
                vis=vis,
                imagename=str(imagebase),
                datacolumn="data",
                specmode="mfs",
                niter=0,
                calcpsf=False,
                calcres=False,
                gain=cfg.imaging.gain,
                threshold=f"{threshold_jy}Jy",
                deconvolver=cfg.imaging.deconvolver,
                scales=self._derived["scales"] if cfg.imaging.deconvolver == "multiscale" else [],
                gridder=cfg.imaging.gridder,
                cyclefactor=cfg.imaging.cyclefactor,
                interactive=False,
                imsize=self._derived["imsize"],
                cell=self._derived["cell"],
                weighting=cfg.imaging.weighting,
                robust=cfg.imaging.robust,
                uvtaper=list(cfg.imaging.uvtaper),
                pbcor=cfg.imaging.pbcor,
                savemodel="modelcolumn",
                parallel=cfg.imaging.parallel,
                field=self._derived["resolved_field_id"],
            )

    def _build_usemask_kwargs(self, existing_mask: Optional[str] = None) -> Dict[str, Any]:
        """Construct tclean mask kwargs based on cfg.imaging.mask_mode.

        Returns a dict that is splatted into tclean() and includes the
        ``interactive`` flag. For all non-interactive modes ``interactive``
        is False; only the FIRST call in ``mask_mode='interactive'`` returns
        ``interactive=True`` -- subsequent iterations always receive a
        pre-made mask via ``existing_mask`` and run non-interactively.
        """
        cfg = self.cfg
        mode = cfg.imaging.mask_mode
        # If a mask file already exists from a previous step (e.g. the
        # initial CLEAN saved one), reuse it as a user mask. This branch
        # also covers the second-and-later iterations of mask_mode='interactive'.
        if existing_mask and os.path.exists(existing_mask):
            return {"mask": existing_mask, "usemask": "user", "interactive": False}

        if mode == "auto-multithresh":
            return {
                "usemask": "auto-multithresh",
                "interactive": False,
                "sidelobethreshold": cfg.imaging.sidelobethreshold,
                "noisethreshold": cfg.imaging.noisethreshold,
                "lownoisethreshold": cfg.imaging.lownoisethreshold,
                "minbeamfrac": cfg.imaging.minbeamfrac,
                "growiterations": cfg.imaging.growiterations,
            }
        elif mode == "interactive":
            # First tclean call only: open the CASA viewer for the user to
            # draw a mask. tclean saves it as <imagebase>.mask, which we
            # then reuse for every subsequent self-cal iteration via the
            # existing_mask branch above.
            #
            # Requires CASA <= 6.6: the casaviewer GUI was removed in CASA
            # 6.7+, making interactive tclean non-functional. Users on
            # CASA 6.7+ should use 'auto-multithresh' (default) or
            # 'user' with a pre-made mask file.
            if cfg.verbose:
                print("[AJISAI] mask_mode='interactive': opening CASA viewer "
                      "for mask drawing. Requires CASA <= 6.6.")
            return {"usemask": "user", "interactive": True}
        elif mode == "user":
            if not cfg.imaging.user_mask:
                raise ValueError("mask_mode='user' but user_mask is None")
            return {"mask": cfg.imaging.user_mask, "usemask": "user",
                    "interactive": False}
        elif mode == "none":
            return {"mask": "", "usemask": "user", "interactive": False}
        else:
            raise ValueError(
                f"unknown mask_mode: {mode!r}. "
                f"Valid values: 'auto-multithresh', 'interactive', 'user', 'none'."
            )

    def _detect_iter_anomaly(
        self,
        stats: Dict[str, Any],
        n_solutions: int,
    ) -> Tuple[bool, Optional[str]]:
        """Flag anomalous iteration outcomes for the justification log.

        AJISAI runs all iterations to completion (deterministic fixed schedule).
        This method does NOT alter pipeline flow — it only marks iterations
        whose outcome looks unusual, so the user can inspect them after the run.

        Anomaly conditions (any one triggers the flag):
            1. RMS / peak / DR is non-finite (numerical pathology in tclean)
            2. DR drops to < 50% of the previous good iteration's DR
               (catastrophic, e.g. solint too short for available SNR)

        Note: small fluctuations (e.g. DR slightly worse) are NOT anomalies.
        ALMA phase selfcal often plateaus before amp selfcal recovers, which is
        normal behavior and would produce false positives if we flagged it.
        """
        new_dr = stats["dynamic_range"]
        if not np.isfinite(new_dr) or not np.isfinite(stats["rms_jy_beam"]):
            return True, "non-finite DR/RMS (tclean numerical pathology?)"
        prev_dr = self._derived.get("last_dr", 0)
        if prev_dr > 0 and new_dr < 0.5 * prev_dr:
            return True, (f"catastrophic DR drop: {new_dr:.1f} < 0.5 × "
                          f"{prev_dr:.1f} (previous iteration)")
        return False, None

    def _record_skipped_iteration(
        self,
        idx: int,
        step: SelfcalStep,
        solint: str,
        reason: str,
    ) -> None:
        """Record an iteration that produced no output (gaincal failed entirely).

        This is the ONLY case where an iteration produces no image. The MS
        state (good_vis) is not advanced because no new MS exists; the next
        iteration tries again from the same MS with its own solint. This is
        a structural consequence of gaincal failing, not a 'fallback' policy.
        """
        rec = {
            "iteration": idx,
            "label": step.label,
            "calmode": step.calmode,
            "solint": solint,
            "peak_jy_beam": None,
            "rms_jy_beam": None,
            "snr": None,
            "dynamic_range": None,
            "bmaj_arcsec": None,
            "bmin_arcsec": None,
            "n_gaincal_solutions": 0,
            "status": "skipped",
            "anomaly_reason": reason,
            "image_path": None,
            "fits_path": None,
        }
        self.metrics.append(rec)
        self.justification["iterations"].append(rec)
        warnings.warn(f"[AJISAI] iter {idx} ({step.label}) SKIPPED: {reason}. "
                      f"Pipeline continues to next iteration.")

    def _select_best(self) -> None:
        """Pick best iteration and set best_image / best_fits / best_metric_value."""
        if not self.metrics:
            warnings.warn("_select_best called with empty metrics list; nothing to choose.")
            return
        best_i = self.select_best_iteration()
        best_metric = self.metrics[best_i]
        self.best_image = best_metric.get("image_path")
        self.best_fits = best_metric.get("fits_path")
        metric_key = "dynamic_range" if self.cfg.quality_metric == "dynamic_range" else "snr"
        self.best_metric_value = best_metric.get(metric_key)
        self.justification["best"] = {
            "iteration": best_i,
            "metric_key": metric_key,
            "value": self.best_metric_value,
            "image": self.best_image,
            "all_iterations": self.metrics,
        }
        if self.cfg.verbose:
            print(f"[AJISAI] best iteration: #{best_i}, "
                  f"{metric_key} = {self.best_metric_value}")

    def _write_summary(self) -> None:
        """Write metrics CSV, 4-panel summary PNG, and copy best image to top of workdir."""
        if not self.metrics or self.workdir is None:
            return

        # --- metrics.csv ---
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.workdir / "metrics.csv", index=False)

        # --- summary plot (DR, peak, RMS, beam over iterations) ---
        self._plot_summary(df)

        # --- Copy best image to top-level workdir for easy access ---
        if self.best_fits is not None and os.path.exists(self.best_fits):
            try:
                final_fits = self.workdir / "final.fits"
                if final_fits.exists():
                    final_fits.unlink()
                shutil.copy2(self.best_fits, final_fits)
            except Exception as e:
                warnings.warn(f"Could not copy best FITS: {e}")
        if self.best_image is not None and os.path.exists(self.best_image):
            try:
                final_image = self.workdir / "final.image"
                if final_image.exists():
                    shutil.rmtree(final_image, ignore_errors=True)
                shutil.copytree(self.best_image, final_image)
            except Exception as e:
                warnings.warn(f"Could not copy best image: {e}")

    def _plot_summary(self, df: pd.DataFrame) -> None:
        """4-panel plot: DR / peak / RMS / beam over iterations."""
        if self.workdir is None or df.empty:
            return
        fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        x = df["iteration"].values
        # Use status to color-code points
        status = df["status"].values
        color_map = {"ok": "steelblue", "anomaly": "orange", "skipped": "red"}
        colors = [color_map.get(s, "gray") for s in status]

        # DR
        ax = axes[0]
        ax.scatter(x, df["dynamic_range"], c=colors, s=80, zorder=3)
        ax.plot(x, df["dynamic_range"], "-", color="gray", alpha=0.5)
        ax.set_ylabel("Dynamic Range\n(peak / RMS_offsrc)")
        ax.grid(True, ls=":", alpha=0.5)
        if self.best_metric_value:
            best_iter = self.justification["best"]["iteration"]
            ax.axvline(best_iter, color="green", ls="--", alpha=0.5,
                       label=f"best = iter {best_iter}")
            ax.legend(loc="best", fontsize=8)

        # Peak
        ax = axes[1]
        peak_mjy = df["peak_jy_beam"].astype(float) * 1e3
        ax.scatter(x, peak_mjy, c=colors, s=80, zorder=3)
        ax.plot(x, peak_mjy, "-", color="gray", alpha=0.5)
        ax.set_ylabel("Peak [mJy/beam]")
        ax.grid(True, ls=":", alpha=0.5)

        # RMS
        ax = axes[2]
        rms_ujy = df["rms_jy_beam"].astype(float) * 1e6
        ax.scatter(x, rms_ujy, c=colors, s=80, zorder=3)
        ax.plot(x, rms_ujy, "-", color="gray", alpha=0.5)
        ax.set_ylabel("RMS [μJy/beam]")
        ax.grid(True, ls=":", alpha=0.5)

        # Beam (geometric mean)
        ax = axes[3]
        bmaj = df["bmaj_arcsec"].astype(float)
        bmin = df["bmin_arcsec"].astype(float)
        beam_mas = np.sqrt(bmaj * bmin) * 1e3
        ax.scatter(x, beam_mas, c=colors, s=80, zorder=3)
        ax.plot(x, beam_mas, "-", color="gray", alpha=0.5)
        ax.set_ylabel("Beam (sqrt(bmaj×bmin)) [mas]")
        ax.set_xlabel("Iteration")
        ax.grid(True, ls=":", alpha=0.5)

        # Legend for status colors
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=s) for s, c in color_map.items()]
        axes[0].legend(handles=handles + axes[0].get_legend_handles_labels()[0],
                       loc="best", fontsize=8)

        projname = self._derived.get("projname", "AJISAI")
        fig.suptitle(f"AJISAI self-calibration summary: {projname}", fontsize=12)
        fig.tight_layout()
        plt.savefig(self.workdir / "selfcal_summary.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _write_justification_json(self) -> None:
        if self.workdir is None:
            return
        # Make justification JSON-serializable (numpy types, paths)
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj
        with open(self.workdir / "justification.json", "w") as f:
            json.dump(_convert(self.justification), f, indent=2)


# ============================================================================
# Module-level helpers
# ============================================================================

def _casa_version_string() -> str:
    if not _HAS_CASA:
        return "not-installed"
    try:
        return casatasks.version_string()
    except Exception:
        return "unknown"


# ============================================================================
# CLI / debugging entry point
# ============================================================================
if __name__ == "__main__":  # pragma: no cover
    # Minimal smoke test: instantiate Config classes and print their fields.
    cfg = AJISAIConfig(vis="/tmp/fake.ms")
    print("AJISAIConfig fields:")
    for k, v in asdict(cfg).items():
        print(f"  {k:<20s} = {v}")
    print(f"\nAJISAI version: {__version__}")
    print(f"CASA available: {_HAS_CASA}")
