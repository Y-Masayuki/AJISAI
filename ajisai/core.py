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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    _HAS_CASA = True
except ImportError:  # pragma: no cover
    casatools = None  # type: ignore
    casatasks = None  # type: ignore
    _HAS_CASA = False
# casaplotms is only needed for the per-iteration gain plots; without it (e.g.
# a headless server or a CASA build that lacks it) the plots are skipped.
try:
    import casaplotms  # type: ignore
except ImportError:  # pragma: no cover
    casaplotms = None  # type: ignore

# AJISAI MS query helpers (native casatools-based replacement for analysisUtils).
from .plotting import (
    panels_from_metrics,
    plot_refant_selection,
    plot_selfcal_images,
    plot_selfcal_summary,
)
from .ms_utils import (
    get_on_source_time,
    get_median_frequency,
    get_baseline_at_percentile,
    get_antenna_positions,
    get_antenna_flag_stats,
    get_dish_diameter_m,
    pick_cell_imsize,
    read_vis_summary,
    SKY_FRAMES,
    parse_ra_dec,
    rad_to_radec,
    casa_direction_string,
    convert_direction,
    code_of_frame,
    field_direction_frame,
    field_id_by_name,
    write_field_direction,
    field_direction_offsets,
    _column_frames,
    _column_measinfo,
    CELL_RULES,
    _CHUNK_SAMPLES,
    _primary_beam_fwhm_arcsec,
    _smallest_5_smooth_at_least,
)
_HAS_MS_UTILS = True
# Data-driven solution intervals (pure numpy; ASEBI SPEC v0.3 section 10.2).
from . import solint as _solint


# Version is defined in ajisai/__init__.py. This local copy is used for
# the justification log and for the standalone __main__ smoke test.
__version__ = "0.3.0.dev0"


# ============================================================================
# Configuration dataclasses (immutable for reproducibility)
# ============================================================================

@dataclass(frozen=True)
class ImagingConfig:
    """tclean-related parameters. Extend by adding fields here.

    Every tclean call of a run (dirty image, initial CLEAN, each self-cal
    iteration, the final image) uses the same ``cell``, ``imsize``,
    ``nterms``, ``reffreq``, ``scales`` and mask. ``None`` means "derived
    from the data" (the derivation is recorded in ``justification.json``).
    """
    robust: float = 0.5
    weighting: str = "briggs"
    niter: int = 50000
    gain: float = 0.05
    cyclefactor: float = 1.5
    deconvolver: str = "multiscale"    # "mtmfs" is used automatically when nterms > 1
    gridder: str = "standard"
    pbcor: bool = False
    parallel: bool = False
    uvtaper: Tuple[str, ...] = ()
    nterms: int = 1                    # > 1: deconvolver="mtmfs" (multiscale scales kept)
    # --- Image geometry (None = derived from the data) ---
    # cell: tclean cell string, e.g. "0.0062arcsec" or "6.2mas".
    cell: Optional[str] = None
    # imsize: N or (Nx, Ny) pixels.
    imsize: Optional[Tuple[int, ...]] = None
    # Rule for the derived cell:
    #   "umax" (default): cell = 1/(5 u_max), u_max from the longest unflagged
    #       baseline at the highest frequency.
    #   "percentile" (legacy, v0.2): beam from the 90th-percentile baseline at
    #       the median frequency, divided by ``cellpix``.
    cell_rule: str = "umax"
    cellpix: int = 10                  # pixels per beam; used by cell_rule="percentile" only
    # reffreq: Taylor-expansion reference frequency, e.g. "227.0GHz". None =
    # CASA's default (the middle of the frequency range). The value actually
    # used is read back from the image and recorded.
    reffreq: Optional[str] = None
    # scales: multiscale scales in pixels. None = from the measured beam b:
    # 0, b, 3b, 9b, ... up to the largest that fits inside the user mask
    # region (factor-3 ladder); without a user mask region, (0, b, 3b).
    scales: Optional[Tuple[int, ...]] = None
    # --- CLEAN depth of the self-cal loop ---
    #   "pure_noise" (default): absolute threshold = threshold_nsigma x sigma_th,
    #       sigma_th measured once on a signal-free dirty image ((XX-YY)/2 or
    #       (RR-LL)/2), the same for every iteration.
    #   "image_rms" (legacy, v0.2): threshold = threshold_factor x the
    #       off-source RMS of the previous image.
    threshold_mode: str = "pure_noise"
    threshold_nsigma: float = 3.0
    threshold_factor: float = 1.0      # threshold_mode="image_rms" only
    # --- Final image (made once, on the selected iteration's data) ---
    # Depth schedule in units of sigma_th: CLEAN to 3 sigma, then continue to
    # 2, 1, 0.5 sigma, stopping as soon as the aperture flux has converged
    # (see AJISAI._make_final_image). () = no final image (v0.2 behaviour:
    # final.image is the selected iteration's image).
    final_depth: Tuple[float, ...] = (3.0, 2.0, 1.0, 0.5)
    # Radius (arcsec, about the phase center) of the aperture used for the
    # flux convergence test. None = the CLEAN mask region, or 5 beams
    # (rms_exclude_beam_factor) when there is no mask.
    final_aperture_arcsec: Optional[float] = None
    # Mask strategy. Supported values:
    #   "auto-multithresh" (default): CASA's automatic threshold-based masking.
    #       Works in any CASA version. Robust for typical ALMA continuum sources.
    #   "interactive": let the user draw a mask in the CASA viewer at the first
    #       tclean call. The saved mask is reused by all subsequent self-cal
    #       iterations. REQUIRES CASA <= 6.6; the casaviewer GUI was removed
    #       in CASA 6.7+, so interactive tclean is non-functional there.
    #       Refused when AJISAIConfig.caller == "asebi".
    #   "user": use a pre-made mask supplied via `user_mask`: a CRTF region
    #       file (text, first line "#CRTF") or a CASA mask image. It is passed
    #       to every tclean (initial, each iteration, final).
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
    # Solutions below this SNR are dropped (the data stay unflagged: calonly).
    minsnr: float = 3.0
    gaintype: str = "T"
    combine: str = "scan"
    gaincal_interp: str = "linear,linear"
    # "perobs": time interpolation never crosses an observation (execution
    # block) boundary of a concatenated MS; ignored for a single observation.
    applycal_interp: str = "linearPDperobs"
    applymode: str = "calonly"
    # False: the self-cal tables never change WEIGHT (an amplitude step with
    # calwt=True would rescale the weights by |g|^2).
    calwt: bool = False
    # spwmap passed to applycal for every table, e.g. (0, 0, 0, 0) when the
    # solve combines spws; () = identity.
    spwmap: Tuple[int, ...] = ()


@dataclass(frozen=True)
class SelfcalStep:
    """A single self-calibration step in the schedule."""
    calmode: str                   # "p" (phase) or "a" (amp; phase is preserved as well)
    solint: str                    # CASA solint string OR "N*IT" where IT = integration time
    solnorm: bool = False          # set True for amp calibration to preserve flux scale
    label: Optional[str] = None    # human-readable name for justification log


#: The fixed ladder of AJISAI v0.2 (``SelfcalSchedule(mode="fixed")``).
FIXED_SCHEDULE_STEPS: Tuple[SelfcalStep, ...] = (
    SelfcalStep("p", "inf",  label="phase_inf"),
    SelfcalStep("p", "6*IT", label="phase_6IT"),
    SelfcalStep("p", "3*IT", label="phase_3IT"),
    SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
)

#: Self-calibration modes: "data" (stages and solints from the data) and "fixed".
SELFCAL_MODES = ("data", "fixed")


@dataclass(frozen=True)
class SelfcalSchedule:
    """How the self-calibration stages are chosen.

    ``mode="data"`` (default, v0.3): phase stages with solution intervals
    decided from the data (ASEBI SPEC v0.3 section 10.2; ``ajisai.solint``):
    stage 1 is one solution per execution block; every later stage measures
    the noise scale, the solution-SNR rate and the phase structure function,
    predicts the residual phase variance of candidate intervals (divisors of
    the scan length, the scan, the EB), gates them on the predicted
    solution SNR (G1) and chooses and accepts one on held-out integrations
    (G2). At most three phase stages; the run stops at the first stage that
    is not accepted, and the best stage is the last accepted one. ``steps``
    is not used.

    ``mode="fixed"`` (v0.2): every step of ``steps`` is run (default: the
    ladder ``inf -> 6*IT -> 3*IT -> amplitude inf``) and the best iteration
    is chosen by ``AJISAIConfig.quality_metric``.

    ``amplitude``: the amplitude stage of ``mode="data"``. Only ``"never"``
    in this version; the conditional amplitude stage (P18 item 6, P24) will
    be added later.
    """
    steps: Tuple[SelfcalStep, ...] = FIXED_SCHEDULE_STEPS
    mode: str = "data"
    amplitude: str = "never"


@dataclass(frozen=True)
class PlotConfig:
    """Options for the diagnostic figures written at the end of ``run()``.

    The fields prefixed ``images_`` control ``selfcal_images.png``, the
    gallery of the CLEAN image of every self-cal iteration. The summary and
    reference-antenna figures have no options.
    """
    images_cmap: str = "inferno"
    images_gamma: float = 1.0                    # top row: power-law stretch (1 = linear)
    images_gamma2: Optional[float] = 0.3         # second row; None draws one row only
    images_fov_beam_factor: float = 10.0         # half-width of the field shown = N * beam
    images_fov_radius_arcsec: Optional[float] = None  # overrides images_fov_beam_factor
    images_center: str = "phase"                 # "phase" (phase center) | "peak" (brightest pixel)


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
    plots: PlotConfig = _dc_field(default_factory=PlotConfig)

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
    # RA is read as hours and Dec as degrees whatever the separators
    # (':' '.' 'h m s' 'd m s' or spaces) and converted to radians at
    # validation; CASA only ever receives radians.
    # Example: phase_center=("16:25:45.0", "-24:12:23.0", "ICRS")
    phase_center: Optional[Tuple[str, str, str]] = None

    # === Strategy switches (future-proofing for new algorithms) ===
    refant_strategy: str = "hybrid"        # "hybrid"|"geometric_center"|"flag_stats"|"manual"
    refant_manual: Optional[str] = None    # antenna name when strategy="manual"
    refant_flag_threshold: float = 0.25    # fraction; antennas with > this are excluded
    rms_method: str = "sigma_clip_excl"    # "sigma_clip_excl"|"sigma_clip"|"annulus"|"mad"
    rms_exclude_beam_factor: float = 5.0   # for sigma_clip_excl: radius = N*beam
    # For rms_method="annulus" (legacy): inner radius of the off-source annulus
    # in arcsec (the target_radius of the legacy AJISAI script). Required, and
    # must be positive, when rms_method="annulus"; ignored otherwise.
    rms_target_radius_arcsec: Optional[float] = None
    rms_sigma: float = 3.0
    rms_maxiters: int = 5
    # Best iteration of schedule.mode="fixed": "dynamic_range" | "peak_snr".
    # With mode="data" the best stage is the last accepted one, and peak,
    # RMS and DR of every image are recorded as diagnostics only.
    quality_metric: str = "dynamic_range"
    # Anomalies (e.g. catastrophic DR drop) are logged for the justification
    # but do NOT alter pipeline flow.
    on_iter_anomaly: str = "log_only"      # "log_only" (default) | future: "warn_loudly"

    # === Data hygiene ===
    # AJISAI never writes to the input MS (``vis``): every MODEL/CORRECTED
    # column and every relabelling lives in copies under results_dir. When
    # this is True and the input carries J2000 frame codes on ALMA data
    # observed before Cycle 3 (ICRS numbers labelled J2000, ALMA
    # knowledgebase), a copy (intermediate/input_icrs.ms) is relabelled
    # J2000 -> ICRS (frame code only; coordinates are not converted) and used
    # as the working MS. Any other J2000 label is genuine J2000: it is kept
    # and the reason is warned about and recorded (relabel_caveat). Same rule
    # as ASEBI SPEC 4.1 item 10 (D37, decided by the author 2026-10-16).
    relabel_j2000_to_icrs: bool = True
    # Who drives this run: "user" (standalone) or "asebi" (the ASEBI pipeline).
    # With "asebi", interactive masking is refused (viewer-free operation).
    caller: str = "user"

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
          ``target_radius_arcsec`` (in the pipeline, set
          ``AJISAIConfig.rms_target_radius_arcsec``).
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
# Pure helpers for imaging, self-cal bookkeeping and checks (CASA-independent)
# ============================================================================

# tclean products that carry a Taylor-term suffix (".tt0", ".tt1", ...) when
# deconvolver="mtmfs" (nterms > 1). Others (mask, alpha, alpha.error) do not.
_TAYLOR_PRODUCTS = ("image", "model", "residual", "psf", "sumwt", "pb", "weight")


def _image_name(base, nterms: int = 1, product: str = "image", term: int = 0) -> str:
    """Name of a tclean product for an image base name.

    ``nterms == 1``: ``<base>.<product>`` (``.image``, ``.model``, ...).
    ``nterms > 1`` (mtmfs): ``<base>.<product>.tt<term>`` for the Taylor
    products (image, model, residual, psf, sumwt, pb, weight); the other
    products (``mask``, ``alpha``, ``alpha.error``) never take a suffix.

    Every place that reads an image (statistics, FITS export, model
    prediction, checks) goes through this helper, so that statistics are
    always taken on the Taylor-0 (``tt0``) image.
    """
    base = str(base)
    if int(nterms) > 1 and product in _TAYLOR_PRODUCTS:
        return f"{base}.{product}.tt{int(term)}"
    return f"{base}.{product}"


def _model_images(base, nterms: int = 1) -> List[str]:
    """Model image(s) of a tclean run: one per Taylor term for mtmfs."""
    if int(nterms) > 1:
        return [_image_name(base, nterms, "model", t) for t in range(int(nterms))]
    return [_image_name(base, 1, "model")]


def _deconvolver_kwargs(deconvolver: str, nterms: int, scales) -> Dict[str, Any]:
    """tclean deconvolver/nterms/scales for an ImagingConfig.

    ``nterms > 1`` always selects ``mtmfs``; the multiscale scales are kept
    (``[0]`` when the configured deconvolver is not a multiscale one).
    """
    nterms = int(nterms)
    sc = [int(x) for x in scales] if scales else [0]
    if nterms > 1:
        if deconvolver not in ("multiscale", "mtmfs"):
            sc = [0]
        return {"deconvolver": "mtmfs", "nterms": nterms, "scales": sc}
    if deconvolver in ("multiscale", "mtmfs"):
        out: Dict[str, Any] = {"deconvolver": deconvolver, "scales": sc}
        if deconvolver == "mtmfs":
            out["nterms"] = 1
        return out
    return {"deconvolver": deconvolver}


def _cell_to_arcsec(cell) -> float:
    """Parse a tclean cell ("0.0062arcsec", "6.2mas", '0.1"', 0.1) to arcsec."""
    if isinstance(cell, (list, tuple)):
        cell = cell[0]
    if isinstance(cell, (int, float)):
        return float(cell)
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*(arcsec|\"|mas|arcmin|deg|rad)?\s*", str(cell))
    if not m:
        raise ValueError(f"cannot parse cell {cell!r}; use e.g. '0.0062arcsec' or '6.2mas'")
    value = float(m.group(1))
    unit = m.group(2) or "arcsec"
    factor = {"arcsec": 1.0, '"': 1.0, "mas": 1e-3, "arcmin": 60.0,
              "deg": 3600.0, "rad": 180.0 * 3600.0 / math.pi}[unit]
    return value * factor


def _imsize_list(imsize) -> List[int]:
    """Normalise an imsize (N or (Nx, Ny)) to tclean's [Nx, Ny]."""
    if isinstance(imsize, (int, np.integer)):
        return [int(imsize), int(imsize)]
    vals = [int(x) for x in imsize]
    if len(vals) == 1:
        return [vals[0], vals[0]]
    if len(vals) != 2:
        raise ValueError(f"imsize must be N or (Nx, Ny); got {imsize!r}")
    return vals


def scale_ladder(beam_pix: float, max_scale_pix: float, ratio: float = 3.0) -> List[int]:
    """Multiscale scales ``0, b, ratio*b, ratio^2*b, ...`` not larger than ``max_scale_pix``.

    ``b`` is the beam FWHM in pixels. Scales are rounded to whole pixels and
    duplicates removed; 0 (the point component) is always first.
    """
    if not (beam_pix > 0 and math.isfinite(beam_pix)):
        return [0]
    out = [0]
    s = float(beam_pix)
    while s <= max_scale_pix + 1e-9:
        v = int(round(s))
        if v > out[-1]:
            out.append(v)
        s *= ratio
    return out


def _disk_fits(mask, radius_pix: float) -> bool:
    """True if a disk of radius ``radius_pix`` fits entirely inside ``mask``.

    This mirrors CASA's own test (MatrixCleaner "Ignoring scale N since it is
    too large to fit within the mask"): a multiscale component of scale size
    ``s`` pixels has a support of radius ``s``. Computed by an FFT
    correlation of the mask's bounding box with the disk.
    """
    mask = np.asarray(mask, dtype=bool)
    if radius_pix <= 0:
        return bool(mask.any())
    idx = np.argwhere(mask)
    if idx.size == 0:
        return False
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    sub = mask[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(float)
    r = int(math.ceil(radius_pix))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= radius_pix ** 2
    if disk.shape[0] > sub.shape[0] or disk.shape[1] > sub.shape[1]:
        return False
    shape = [sub.shape[0] + disk.shape[0], sub.shape[1] + disk.shape[1]]
    conv = np.fft.irfft2(np.fft.rfft2(sub, shape) * np.fft.rfft2(disk.astype(float), shape),
                         shape)
    return bool((conv > disk.sum() - 0.5).any())


def multiscale_scales(
    beam_pix: float,
    region_mask=None,
    ratio: float = 3.0,
) -> Tuple[List[int], str]:
    """Scales for multiscale/mtmfs CLEAN from the beam and the mask region.

    With a mask region (the user's CRTF rasterised on the image grid): the
    factor-``ratio`` ladder ``0, b, 3b, 9b, ...`` up to the largest scale
    whose support (radius = scale) fits inside the region, i.e. half the
    region size for a round region. Larger scales would be ignored by CASA
    ("Ignoring scale ... too large to fit within the mask").
    Without a region: ``0, b, 3b`` (the v0.2 rule ``[0, 1, 3] x beam``).

    Returns ``(scales, reason)``.
    """
    b = float(beam_pix)
    if region_mask is None or not np.asarray(region_mask, dtype=bool).any():
        scales = scale_ladder(b, 3.0 * b * (1 + 1e-9), ratio=ratio)
        return scales, (f"[0, 1, 3] x beam (beam = {b:.2f} pix); no mask region to "
                        f"bound larger scales")
    mask = np.asarray(region_mask, dtype=bool)
    # Candidate ladder up to the region's bounding size, then keep those that fit.
    extent = float(max(np.ptp(np.argwhere(mask), axis=0)) + 1)
    candidates = scale_ladder(b, extent, ratio=ratio)
    scales = [0]
    for s in candidates[1:]:
        if _disk_fits(mask, s):
            scales.append(s)
        else:
            break
    return scales, (f"0, b, {ratio:g}b, ... (b = beam = {b:.2f} pix) up to the largest scale "
                    f"whose support fits inside the mask region ({int(mask.sum())} pix, "
                    f"extent {extent:.0f} pix)")


# Rational approximation of the prolate spheroidal function used by CASA's
# multiscale components (MatrixCleaner::spheroidal): numerator p, denominator
# q, for 0 <= nu < 0.75 (column 0) and 0.75 <= nu <= 1 (column 1).
_SPHEROIDAL_P = np.array([[8.203343e-2, 4.028559e-3],
                          [-3.644705e-1, -3.697768e-2],
                          [6.278660e-1, 1.021332e-1],
                          [-5.335581e-1, -1.201436e-1],
                          [2.312756e-1, 6.412774e-2]])
_SPHEROIDAL_Q = np.array([[1.0, 1.0],
                          [8.212018e-1, 9.599102e-1],
                          [2.078043e-1, 2.918724e-1]])


def _casa_spheroidal(nu) -> np.ndarray:
    """CASA's ``MatrixCleaner::spheroidal``: 1 at nu <= 0, 0 at nu >= 1."""
    nu = np.asarray(nu, dtype=float)
    out = np.zeros_like(nu)
    parts = (((nu >= 0.0) & (nu < 0.75), 0.75), ((nu >= 0.75) & (nu < 1.0), 1.0))
    for part, (inside, end) in enumerate(parts):
        d = nu[inside] ** 2 - end ** 2
        top = sum(_SPHEROIDAL_P[k, part] * d ** k for k in range(_SPHEROIDAL_P.shape[0]))
        bot = sum(_SPHEROIDAL_Q[k, part] * d ** k for k in range(_SPHEROIDAL_Q.shape[0]))
        out[inside] = top / bot
    out[nu <= 0.0] = 1.0
    return out


def casa_scale_kernel(scale_pix: float, shape) -> np.ndarray:
    """CASA's multiscale component of size ``scale_pix`` on a grid of ``shape``.

    Mirrors ``MatrixCleaner::makeScale``: ``(1 - r^2) x spheroidal(r)`` with
    ``r`` = distance / scale (support radius = scale), centred on pixel
    ``(n0 // 2, n1 // 2)`` and normalised to unit sum; scale 0 is a single
    pixel.
    """
    n0, n1 = (int(x) for x in shape)
    out = np.zeros((n0, n1))
    c0, c1 = n0 // 2, n1 // 2
    s = float(scale_pix)
    if s <= 0.0:
        out[c0, c1] = 1.0
        return out
    r = int(math.ceil(s))
    i0, i1 = max(0, c0 - r), min(n0, c0 + r + 1)
    j0, j1 = max(0, c1 - r), min(n1, c1 + r + 1)
    ii, jj = np.mgrid[i0:i1, j0:j1]
    rad2 = ((ii - c0) / s) ** 2 + ((jj - c1) / s) ** 2
    inside = rad2 < 1.0
    out[i0:i1, j0:j1] = np.where(inside, (1.0 - rad2) * _casa_spheroidal(np.sqrt(rad2)), 0.0)
    return out / out.sum()


def scale_psf_response(psf, scales) -> List[Dict[str, Any]]:
    """The scale-PSF of each multiscale component, as CASA's minor cycle computes it.

    ``psf`` is the PSF plane of the run (peak 1). The scale-PSF is the PSF
    convolved twice with the component (CASA's ``MatrixCleaner::makePsfScales``:
    FFT(psf) x FFT(scale)^2, a circular convolution on the image grid): the
    response of the scale-smoothed residual to a component of that scale.
    For each scale: ``centre``, its value at the PSF peak, i.e. the component's
    own position (= sum over the samples of w |FT(scale)|^2 / sum w: the part
    of the scale's power that the uv coverage samples; the diagonal of the
    Hessian that CASA's mtmfs logs); ``max_abs``, the value of largest
    magnitude anywhere, with its sign, and its pixel ``max_abs_at``;
    ``ok``: the scale-PSF has its largest magnitude at the centre.

    A scale is not ``ok`` when its sampled power falls below the smoothed
    sidelobes of the PSF, i.e. no baseline is short enough to see it.
    CASA normalises each scale by ``max_abs`` and its ``MatrixCleaner::clean``
    refuses to start when ``max_abs`` of any scale is negative ("As Peak of
    PSF is negative, you should setscales again with a smaller scale size"):
    tclean then ends with no CLEAN component at all. When ``max_abs`` is
    positive but away from the centre, CASA runs but normalises the scale by
    a sidelobe, not by the component's own response. Which of the two
    happens depends on the sign of the largest sidelobe; ``ok`` covers both.
    """
    psf = np.asarray(psf, dtype=float)
    while psf.ndim > 2:
        psf = psf[..., 0]
    c = np.unravel_index(int(np.argmax(psf)), psf.shape)
    psf_ft = None
    out: List[Dict[str, Any]] = []
    for s in scales:
        if float(s) <= 0.0:
            conv = psf
        else:
            if psf_ft is None:
                psf_ft = np.fft.rfft2(psf)
            kernel_ft = np.fft.rfft2(np.fft.ifftshift(casa_scale_kernel(s, psf.shape)))
            conv = np.fft.irfft2(psf_ft * kernel_ft * kernel_ft, psf.shape)
        mag = np.abs(conv)
        k = np.unravel_index(int(np.argmax(mag)), conv.shape)
        centre = float(conv[c])
        out.append({"scale": int(s), "centre": centre, "max_abs": float(conv[k]),
                    "max_abs_at": [int(k[0]), int(k[1])],
                    "ok": bool(centre > 0.0 and centre >= float(mag[k]) * (1.0 - 1e-9))})
    return out


def psf_supported_scales(psf, scales) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Scales whose scale-PSF peaks at the component's own position (:func:`scale_psf_response`).

    Returns ``(kept, response)``; scale 0 is always kept.
    """
    response = scale_psf_response(psf, scales)
    kept = [int(r["scale"]) for r in response if r["ok"] or int(r["scale"]) == 0]
    return kept, response


# CASA Stokes codes of the correlation products (casacore Stokes enum).
_CORR_RR, _CORR_LL = 5, 8
_CORR_XX, _CORR_YY = 9, 12


def _noise_stokes(corr_types_per_pol) -> Optional[str]:
    """Stokes parameter whose image is signal-free noise, or None.

    ``(XX - YY)/2`` is Stokes Q for linear feeds, ``(RR - LL)/2`` Stokes V for
    circular feeds; both have the same thermal noise as Stokes I and (to the
    level of the source polarisation) no signal. Every polarisation setup
    of the MS must contain the parallel-hand pair.
    """
    setups = [set(int(c) for c in np.asarray(ct).ravel()) for ct in corr_types_per_pol]
    if not setups:
        return None
    if all({_CORR_XX, _CORR_YY} <= s for s in setups):
        return "Q"
    if all({_CORR_RR, _CORR_LL} <= s for s in setups):
        return "V"
    return None


def solint_seconds(solint: str, int_time_s: Optional[float] = None) -> Optional[float]:
    """Solution interval in seconds from a resolved CASA solint string.

    ``"inf"`` -> None (one solution per scan, or per observation with
    ``combine='scan'``); ``"int"`` -> the integration time; ``"36.300s"``,
    ``"1.5min"``, ``"0.5h"``, ``"60"`` -> seconds.
    """
    s = str(solint).strip().lower()
    if s == "inf":
        return None
    if s == "int":
        return float(int_time_s) if int_time_s is not None else None
    m = re.fullmatch(r"([0-9.]+(?:e[+-]?[0-9]+)?)\s*(s|sec|min|m|h)?", s)
    if not m:
        raise ValueError(f"cannot convert solint {solint!r} to seconds")
    value = float(m.group(1))
    unit = m.group(2) or "s"
    return value * {"s": 1.0, "sec": 1.0, "min": 60.0, "m": 60.0, "h": 3600.0}[unit]


def flagged_solution_fractions(obs_ids, spw_ids, flags) -> List[Dict[str, Any]]:
    """Fraction of flagged gain solutions per (observation, spw).

    ``flags`` is the caltable FLAG column (npol, nchan, nrow); a solution
    (row) counts as flagged when all its entries are flagged (e.g. below
    ``minsnr``).
    """
    obs = np.asarray(obs_ids).ravel()
    spw = np.asarray(spw_ids).ravel()
    f = np.asarray(flags, dtype=bool)
    row_flagged = f.reshape(-1, f.shape[-1]).all(axis=0) if f.size else np.zeros(0, bool)
    out = []
    for o, w in sorted(set(zip(obs.tolist(), spw.tolist()))):
        sel = (obs == o) & (spw == w)
        n = int(sel.sum())
        k = int(row_flagged[sel].sum())
        out.append({"observation": int(o), "spw": int(w), "n_solutions": n,
                    "n_flagged": k, "fraction": (k / n) if n else None})
    return out


# Keys of the per-stage solve record (AJISAI.stages[i]["solve"]); stable API
# for callers that re-use the solutions (e.g. ASEBI's line re-solve).
SOLVE_INFO_KEYS = (
    "table", "solint", "solint_s", "combine", "refant", "field", "calmode", "gaintype",
    "minsnr", "minblperant", "solnorm", "applycal", "model_image", "model_nterms",
    "model_reffreq_hz", "solved_on", "stage_ms", "n_solutions",
    "flagged_solution_fraction",
)


def build_solve_info(
    *,
    table: str,
    solint: str,
    solint_s: Optional[float],
    combine: str,
    refant: str,
    calmode: str,
    gaintype: str,
    minsnr: float,
    minblperant: int,
    solnorm: bool,
    interp: str,
    applymode: str,
    calwt: bool,
    spwmap,
    model_image,
    model_nterms: int,
    model_reffreq_hz: Optional[float],
    solved_on: str,
    stage_ms: Optional[str],
    n_solutions: int,
    flagged_solution_fraction: List[Dict[str, Any]],
    field: str = "",
    spwmap_per_observation: Optional[Dict[int, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Per-stage record of how the gain table was solved and applied.

    Enough to re-solve the same stage later (same solint, combine, refant,
    field, model) and to re-apply the table elsewhere (interp, applymode,
    spwmap). ``field`` is the gaincal field selection ("" = all fields).
    ``spwmap_per_observation`` (combined spws, ASEBI SPEC 9 item 3, D38 d):
    the map of each observation from the table's spw labels; ``spwmap`` is
    then the one map for all observations, or [] when the table needs one
    applycal per observation.
    """
    info = {
        "table": str(table),
        "solint": str(solint),
        "solint_s": solint_s,
        "combine": combine,
        "refant": refant,
        "field": str(field),
        "calmode": calmode,
        "gaintype": gaintype,
        "minsnr": float(minsnr),
        "minblperant": int(minblperant),
        "solnorm": bool(solnorm),
        "applycal": {
            "interp": interp,
            "applymode": applymode,
            "calwt": bool(calwt),
            "spwmap": [int(x) for x in spwmap] if spwmap else [],
        },
        "model_image": [str(m) for m in model_image] if isinstance(model_image, (list, tuple))
        else [str(model_image)],
        "model_nterms": int(model_nterms),
        "model_reffreq_hz": model_reffreq_hz,
        "solved_on": str(solved_on),
        "stage_ms": str(stage_ms) if stage_ms is not None else None,
        "n_solutions": int(n_solutions),
        "flagged_solution_fraction": flagged_solution_fraction,
    }
    if spwmap_per_observation is not None:
        info["applycal"]["spwmap_per_observation"] = {
            str(int(o)): [int(x) for x in m] for o, m in sorted(spwmap_per_observation.items())}
    return info


# Largest offset between the image's spectral frame (LSRK) and the data's
# (TOPO) as a fraction of frequency: 60 km/s (Earth's orbit + rotation + the
# Sun's motion are < 50 km/s).
_DOPPLER_TOLERANCE = 2.0e-4


def spectral_cover_increment(ref_hz: float, inc_hz: float, lo_hz: float, hi_hz: float,
                             rtol: float = _DOPPLER_TOLERANCE) -> Optional[float]:
    """Spectral increment, centred on ``ref_hz``, that covers the data's channel edges.

    tclean sets an mfs image's frequency range from the data, and with a
    descending spw (negative channel width) at an edge of the band it was found
    to drop the outermost channel (DG Tau B, 2015 EB: range from 219.5245 GHz,
    data from 219.5046 GHz); ``ft`` then writes no model into that channel.
    Returns the increment (sign of ``inc_hz``) whose range ``ref +- |inc|/2``
    covers ``[lo (1 - rtol), hi (1 + rtol)]``, the data's TOPO edges widened by
    the largest LSRK/TOPO offset, or None when the axis already covers it.
    Widening the axis does not change the predicted values: the model is the
    same at every frequency for nterms = 1, and the Taylor terms are evaluated
    about the reference frequency given to ``ft``.
    """
    lo_m = float(lo_hz) * (1.0 - rtol)
    hi_m = float(hi_hz) * (1.0 + rtol)
    half = 0.5 * abs(float(inc_hz))
    ref = float(ref_hz)
    if ref - half <= lo_m and ref + half >= hi_m:
        return None
    need = 2.0 * max(ref - lo_m, hi_m - ref)
    return math.copysign(need, float(inc_hz) if float(inc_hz) != 0.0 else 1.0)


def imaging_consistency_problems(
    ref_freq_hz: float,
    width_hz: float,
    sumwt: float,
    data_fmin_hz: Optional[float],
    data_fmax_hz: Optional[float],
    tolerance: float = _DOPPLER_TOLERANCE,
) -> List[str]:
    """Problems with an MFS image made by tclean ([] if consistent).

    1. The frequency axis is finite and positive.
    2. The image's single channel ``[ref - width/2, ref + width/2]`` covers
       the data's frequency range (within the LSRK/TOPO Doppler tolerance).
       A narrower band means tclean dropped channels (seen with spws of
       different channel counts in CASA 6.5/6.7).
    3. ``sumwt > 0`` (some data were gridded).
    """
    problems = []
    if not (np.isfinite(ref_freq_hz) and ref_freq_hz > 0):
        problems.append(f"reference frequency is not finite/positive ({ref_freq_hz!r})")
    if not (np.isfinite(width_hz) and width_hz != 0):
        problems.append(f"frequency width is not finite/non-zero ({width_hz!r})")
    if not problems and data_fmin_hz is not None and data_fmax_hz is not None:
        half = abs(width_hz) / 2.0
        lo, hi = ref_freq_hz - half, ref_freq_hz + half
        tol = tolerance * data_fmax_hz
        if lo > data_fmin_hz + tol or hi < data_fmax_hz - tol:
            problems.append(
                f"image band {lo / 1e9:.6f}-{hi / 1e9:.6f} GHz does not cover the data "
                f"{data_fmin_hz / 1e9:.6f}-{data_fmax_hz / 1e9:.6f} GHz (channels dropped)")
    if not (np.isfinite(sumwt) and sumwt > 0):
        problems.append(f"sumwt is not positive ({sumwt!r}): no data were gridded")
    return problems


def model_zero_channels(absmax_by_chan, n_unflagged_by_chan) -> List[int]:
    """Channels with unflagged data whose MODEL is zero (or not finite) everywhere.

    After ``ft`` (or a tclean model prediction) every unflagged channel must
    carry a non-zero model; zeros mean the channel fell outside the model's
    band and would be dropped from the gain solve.
    """
    a = np.asarray(absmax_by_chan, dtype=float)
    n = np.asarray(n_unflagged_by_chan)
    bad = (n > 0) & ~(np.isfinite(a) & (a > 0))
    return [int(i) for i in np.flatnonzero(bad)]


def empty_model_problem(models, residual, mask, threshold_jy: float,
                        scales=None, casa_log: Optional[str] = None) -> Optional[str]:
    """Why a CLEAN left no component; None when the model is not empty.

    ``models`` are the model planes (one per Taylor term), ``residual`` the
    residual (tt0) plane and ``mask`` the CLEAN mask plane (None: the whole
    image). With the residual peak inside the mask above the threshold the
    minor cycle did not run; otherwise the mask holds no emission above the
    threshold.
    """
    for m in models:
        m = np.asarray(m, dtype=float)
        if np.any(np.isfinite(m) & (m != 0.0)):
            return None
    res = np.abs(np.asarray(residual, dtype=float))
    if mask is None:
        inside, where = np.isfinite(res), "the image"
    else:
        inside, where = (np.asarray(mask, dtype=float) > 0.5) & np.isfinite(res), "the mask"
    if not inside.any():
        return f"the CLEAN model is empty and {where} has no pixel to CLEAN"
    peak = float(res[inside].max())
    thr = float(threshold_jy)
    if peak > thr:
        log = f" ({casa_log})" if casa_log else ""
        return (f"the CLEAN model is empty although the residual peak inside {where} "
                f"({peak * 1e3:.4g} mJy/beam) is {peak / thr:.1f} x the threshold "
                f"({thr * 1e6:.4g} uJy): the minor cycle did not run. See the CASA log{log} "
                f"of this tclean, e.g. MatrixCleaner 'As Peak of PSF is negative' for a "
                f"multiscale scale the uv coverage cannot see (scales {list(scales or [])})")
    return (f"the CLEAN model is empty: the residual peak inside {where} ({peak * 1e6:.4g} uJy/beam) "
            f"is not above the threshold ({thr * 1e6:.4g} uJy); there is no emission to "
            f"self-calibrate on")


def psf_volume_within(psf, radius_pix: float) -> float:
    """Sum of the PSF (peak normalised to 1) within ``radius_pix`` of its peak.

    This is Omega_R, the dirty-beam volume in pixels that converts a sum of
    residual pixels (Jy/beam) inside an aperture of that size into Jy.
    """
    p = np.asarray(psf, dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    peak_idx = np.unravel_index(int(np.argmax(p)), p.shape)
    peak = p[peak_idx]
    if not peak > 0:
        raise ValueError("PSF has no positive peak")
    grids = np.indices(p.shape)
    r2 = sum((g - c) ** 2 for g, c in zip(grids, peak_idx))
    return float(p[r2 <= radius_pix ** 2].sum() / peak)


def aperture_flux_terms(
    model,
    residual,
    psf,
    aperture,
    cell_arcsec: float,
    bmaj_arcsec: float,
    bmin_arcsec: float,
    sigma_jy: float,
) -> Dict[str, Any]:
    """Flux bookkeeping of a CLEAN image inside an aperture.

    - ``F_model`` = sum of the model (Jy/pixel) inside the aperture.
    - ``Omega_R`` = PSF volume within the radius of a circle of the same
      area as the aperture (pixels); ``F_unc = sum(residual)/Omega_R`` is the
      flux not yet CLEANed; ``F_corr = F_model + F_unc``.
    - ``sigma_F = sigma sqrt(N_pix / Omega_R)``: noise of ``F_unc`` for
      residual noise sigma (Jy/beam) correlated over the dirty beam.
    - ``Omega_clean`` = restoring-beam area (pixels); the restored image
      counts the residual with the CLEAN beam, so its aperture flux is biased
      by ``F_unc (Omega_R/Omega_clean - 1)`` (``restored_bias_jy``).
    """
    ap = np.asarray(aperture, dtype=bool)
    n_pix = int(ap.sum())
    if n_pix == 0:
        raise ValueError("empty aperture")
    model = np.asarray(model, dtype=float)
    residual = np.asarray(residual, dtype=float)
    radius = math.sqrt(n_pix / math.pi)
    omega_r = psf_volume_within(psf, radius)
    if not omega_r > 0:
        raise ValueError(f"PSF volume within the aperture is not positive ({omega_r})")
    f_model = float(np.nansum(model[ap]))
    f_unc = float(np.nansum(residual[ap])) / omega_r
    sigma_f = float(sigma_jy) * math.sqrt(n_pix / omega_r)
    omega_clean = math.pi * bmaj_arcsec * bmin_arcsec / (4.0 * math.log(2.0)) / cell_arcsec ** 2
    return {
        "aperture_npix": n_pix,
        "aperture_radius_equiv_pix": radius,
        "omega_r_pix": omega_r,
        "omega_clean_pix": omega_clean,
        "f_model_jy": f_model,
        "f_unc_jy": f_unc,
        "f_corr_jy": f_model + f_unc,
        "sigma_f_jy": sigma_f,
        "restored_bias_jy": f_unc * (omega_r / omega_clean - 1.0),
    }


def final_depth_decision(
    f_model_jy: float,
    f_unc_jy: float,
    sigma_f_jy: float,
    prev_f_corr_jy: Optional[float] = None,
) -> Dict[str, Any]:
    """Stop rule of the final image's CLEAN depth (decided 2026-10-15).

    After each depth stage (3 -> 2 -> 1 -> 0.5 sigma_th) with
    ``F_corr = F_model + F_unc`` inside the aperture and the width
    ``w = max(3 sigma_F, 0.01 F_corr)``, stop when

    - ``|F_unc| <= w`` (the flux left in the residual is within the noise /
      1 % of the total), or
    - ``|F_corr - F_corr(previous stage)| <= w`` (deeper CLEAN no longer
      changes the corrected flux).
    """
    f_corr = f_model_jy + f_unc_jy
    width = max(3.0 * sigma_f_jy, 0.01 * abs(f_corr))
    unc_ok = abs(f_unc_jy) <= width
    change = None if prev_f_corr_jy is None else f_corr - prev_f_corr_jy
    change_ok = change is not None and abs(change) <= width
    if unc_ok:
        reason = (f"|F_unc| = {abs(f_unc_jy) * 1e3:.3g} mJy <= max(3 sigma_F, 1% F_corr) "
                  f"= {width * 1e3:.3g} mJy")
    elif change_ok:
        reason = (f"F_corr changed by {abs(change) * 1e3:.3g} mJy since the previous stage "
                  f"<= max(3 sigma_F, 1% F_corr) = {width * 1e3:.3g} mJy")
    else:
        reason = (f"|F_unc| = {abs(f_unc_jy) * 1e3:.3g} mJy > {width * 1e3:.3g} mJy"
                  + ("" if change is None else
                     f" and F_corr changed by {abs(change) * 1e3:.3g} mJy"))
    return {
        "f_corr_jy": f_corr,
        "width_jy": width,
        "f_unc_within_width": bool(unc_ok),
        "f_corr_change_jy": change,
        "f_corr_change_within_width": bool(change_ok),
        "stop": bool(unc_ok or change_ok),
        "reason": reason,
    }


def _relabel_needed(field_frames, source_frames) -> bool:
    """Would :func:`relabel_J2000_to_ICRS` change anything?

    ``field_frames``: per-row frame names of the FIELD direction columns
    (one sequence per column); ``source_frames``: those of SOURCE.DIRECTION.
    Frame names come from the MEASINFO keywords (variable-reference codes
    mapped through TabRefTypes/TabRefCodes), never from assumed codes.
    """
    for frames in list(field_frames) + [source_frames or []]:
        if any(str(f) == "J2000" for f in frames):
            return True
    return False


# Start of ALMA Cycle 3 (2015-10-01 00:00 UTC) as an MJD in seconds, the unit
# of the MS TIME columns. ALMA data observed before it carry a J2000 label on
# ICRS numbers (ALMA knowledgebase), which a label-only relabel corrects.
ALMA_CYCLE3_START_MJD_S = float(
    (datetime(2015, 10, 1) - datetime(1858, 11, 17)).days * 86400)


def _relabel_caveat(telescope_names, start_mjd_s: Optional[float]) -> Optional[str]:
    """Why a J2000 label must NOT be relabelled for these data (None when the relabel is right).

    The relabel changes the frame label and keeps the numbers. That is the
    correction for ALMA data observed before Cycle 3 (ICRS numbers labelled
    J2000); for any other data with a J2000 label the numbers are genuine
    J2000 and the relabel would move every direction by the frame bias (up
    to about 23 mas), so those labels are kept (ASEBI SPEC 4.1 item 10, D37).
    """
    names = sorted({str(n).strip().upper() for n in telescope_names})
    reasons = []
    if not names or any(n not in ("ALMA", "ACA") for n in names):
        reasons.append(f"TELESCOPE_NAME {names} is not ALMA")
    if start_mjd_s is None:
        reasons.append("the observation start time is unknown")
    elif float(start_mjd_s) >= ALMA_CYCLE3_START_MJD_S:
        reasons.append("the observation starts on or after 2015-10-01 (ALMA Cycle 3)")
    if not reasons:
        return None
    return ("the J2000 label is kept as genuine J2000: the label-only J2000 -> ICRS relabel is "
            "only right for ALMA data observed before Cycle 3 (ICRS numbers labelled J2000); here "
            + " and ".join(reasons) + " (relabelling genuine J2000 numbers would move them by the "
              "frame bias, up to about 23 mas)")


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


def _relabel_column(tb, column: str) -> Dict[str, Any]:
    """Relabel J2000 -> ICRS in one direction column of an open table (numbers unchanged)."""
    mi = _column_measinfo(tb, column)
    if "VarRefCol" in mi:
        refcol = str(mi["VarRefCol"])
        j2000, icrs = code_of_frame(mi, "J2000"), code_of_frame(mi, "ICRS")
        codes = np.asarray(tb.getcol(refcol))
        rows = np.flatnonzero(codes == j2000)
        if rows.size:
            codes = codes.copy()
            codes[rows] = icrs
            tb.putcol(refcol, codes)
        return {"reference": "variable", "ref_column": refcol, "j2000_code": j2000,
                "icrs_code": icrs, "rows_relabelled": [int(r) for r in rows]}
    ref = str(mi.get("Ref", "J2000"))   # casacore's default when no Ref is stored
    if ref == "J2000":
        mi.setdefault("type", "direction")
        mi["Ref"] = "ICRS"
        tb.putcolkeyword(column, "MEASINFO", mi)
        return {"reference": "fixed", "ref": "J2000 -> ICRS (whole column)",
                "rows_relabelled": list(range(tb.nrows()))}
    return {"reference": "fixed", "ref": ref, "rows_relabelled": []}


def relabel_J2000_to_ICRS(msfile: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Relabel J2000 frame entries in FIELD and SOURCE tables to ICRS.

    Ported from AJISAI.py. A label-only change: the numbers are kept, so it
    is right for ALMA data observed before Cycle 3, whose ICRS numbers carry
    a J2000 label, and wrong (by the frame bias, up to about 23 mas) for
    genuine J2000 numbers. Operates in place on the MS; safe to call
    repeatedly (idempotent). The pipeline calls it only on its own copies
    (never on the input MS).

    FIELD PHASE_DIR, DELAY_DIR and REFERENCE_DIR and SOURCE.DIRECTION are
    handled alike: for a variable-reference column the J2000 rows get the
    ICRS code, both codes read from the column's MEASINFO
    TabRefTypes/TabRefCodes; a fixed-reference column labelled J2000 is
    relabelled as a whole. A table that cannot be relabelled is reported
    with a warning and in the returned record (``errors``), never skipped
    silently.

    Returns
    -------
    dict
        ``{"FIELD": {column: record}, "SOURCE": {...}, "errors": [...]}``.
    """
    _require_casa()
    out: Dict[str, Any] = {"FIELD": {}, "SOURCE": {}, "errors": []}
    tb = casatools.table()
    for table, columns in (("FIELD", ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR")),
                           ("SOURCE", ("DIRECTION",))):
        path = msfile + "/" + table
        if not os.path.isdir(path):
            out[table]["absent"] = True
            continue
        try:
            tb.open(path, nomodify=False)
        except Exception as exc:  # recorded and warned, not swallowed
            msg = f"relabel_J2000_to_ICRS: cannot open {path}: {exc}"
            out["errors"].append(msg)
            warnings.warn(msg)
            continue
        try:
            for col in columns:
                if col not in tb.colnames():
                    continue
                try:
                    rec = _relabel_column(tb, col)
                except Exception as exc:  # recorded and warned, not swallowed
                    msg = f"relabel_J2000_to_ICRS: {table}.{col} not relabelled: {exc}"
                    out["errors"].append(msg)
                    warnings.warn(msg)
                    continue
                out[table][col] = rec
                if verbose and rec["rows_relabelled"]:
                    print(f"[AJISAI] {table}.{col}: J2000 -> ICRS "
                          f"({len(rec['rows_relabelled'])} rows, label only)")
        finally:
            tb.close()
    return out


def _max_pixel_world_coords(imagename: str) -> Dict[str, Any]:
    """World coordinates of the brightest pixel in a CASA image.

    Uses casatasks.imstat for the pixel (always succeeds; the max pixel
    always exists) and the image's coordinate system for its direction, in
    radians at the pixel centre (``imstat``'s ``maxposf`` string is rounded
    to milliseconds of time, i.e. up to 15 mas, and is not used).

    Returns
    -------
    dict with keys:
        ra_rad, dec_rad : direction of the pixel centre (radians)
        frame : "ICRS", "J2000", "B1950", etc. (whatever the image declares)
        ra, dec : the same direction as strings (RA hours with ':', Dec
                  degrees with '.'), for the record only
        pixel : the pixel indices from imstat
    """
    _require_casa()
    stats = casatasks.imstat(imagename=imagename, stokes="I")
    pixel = [int(p) for p in np.asarray(stats["maxpos"]).ravel()]
    ia = casatools.image()
    qa = casatools.quanta()
    ia.open(imagename)
    try:
        cs = ia.coordsys()
        try:
            world = cs.toworld([float(p) for p in pixel], "m")["measure"]["direction"]
        finally:
            cs.done()
    finally:
        ia.close()
    ra_rad = float(qa.convert(world["m0"], "rad")["value"]) % (2.0 * math.pi)
    dec_rad = float(qa.convert(world["m1"], "rad")["value"])
    frame = str(world["refer"]).upper()
    ra, dec = rad_to_radec(ra_rad, dec_rad, precision=7).split()
    return {"ra_rad": ra_rad, "dec_rad": dec_rad, "frame": frame, "ra": ra, "dec": dec,
            "pixel": pixel}


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
            "casa_log": _casa_log_path(),
            "started_at": None,
            "finished_at": None,
            "input_config": asdict(config),
            "derived": {},
            "iterations": [],
            "stages": [],
            "checks": [],
            "best": None,
            "final_ms": None,
            "final_image": None,
        }
        self._derived: Dict[str, Any] = {}
        self.workdir: Optional[Path] = None
        self.best_image: Optional[str] = None
        self.best_fits: Optional[str] = None
        self.best_metric_value: Optional[float] = None
        # Per-stage records (stage 0 = no self-cal): data and stage MS, image
        # and model, and for stages >= 1 the solve information ("solve", see
        # SOLVE_INFO_KEYS). Also written to justification.json ("stages").
        self.stages: List[Dict[str, Any]] = []
        # DATA-only MS of the selected stage, and the final (deep) image.
        self.final_ms: Optional[str] = None
        self.final_image: Optional[str] = None
        self.final_fits: Optional[str] = None

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def run(self) -> "AJISAI":
        """Top-level pipeline orchestrator.

        **Pipeline order**

        1. Validate inputs.
        2. Set up working directory.
        3. Inspect MS (refant, frequency, integration time, etc.); the input
           MS is only read (a relabelled copy is made if needed).
        4. Compute imaging parameters (cell from u_max, imsize from the
           primary beam of the smallest dish).
        5. Make dirty image (beam and peak).
        6. Apply phase shift if requested (off by default).
        7. Measure the thermal noise sigma_th on a signal-free dirty image.
        8. Derive the multiscale scales from the beam and the mask region.
        9. Make initial CLEAN image (iteration 0).
        10. Self-calibrate (each stage on a temporary copy of the previous
            stage's MS; stage MSs are kept). ``schedule.mode="data"``
            (default): phase stages decided from the data (stage 1 one
            solution per EB, then G1/G2 on candidate intervals, at most three
            stages); ``mode="fixed"``: every step of ``schedule.steps``.
        11. Select the best stage: the last accepted one (``mode="data"``),
            or by ``quality_metric`` (``mode="fixed"``).
        12. Write the DATA-only ``final_selfcal.ms`` of that iteration and
            make the final (deep) image from it.
        13. Write summary CSV, summary PNG, and justification JSON.
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
            self._measure_noise()
            self._compute_scales()
            self._make_initial_clean()
            if self.cfg.schedule.mode == "fixed":
                for i, step in enumerate(self.cfg.schedule.steps, 1):
                    self._run_iteration(i, step)
            else:
                self._run_data_driven()
            self._select_best()
            self._write_final_ms()
            self._make_final_image()
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
            target_radius_arcsec=self.cfg.rms_target_radius_arcsec,
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
        # rms_method validation
        if cfg.rms_method == "annulus":
            radius = cfg.rms_target_radius_arcsec
            if radius is None:
                raise ValueError(
                    "rms_method='annulus' requires rms_target_radius_arcsec (inner "
                    "radius of the off-source annulus, in arcsec) to be set"
                )
            if not radius > 0:
                raise ValueError(
                    f"rms_target_radius_arcsec must be positive; got {radius!r}"
                )
            warnings.warn(
                "rms_method='annulus' is the legacy method; consider switching to "
                "'sigma_clip_excl' (the validated default, which needs no target radius)."
            )
        elif cfg.rms_target_radius_arcsec is not None:
            warnings.warn(
                f"rms_target_radius_arcsec is set but rms_method={cfg.rms_method!r}; "
                "the radius is only used by rms_method='annulus' and will be ignored."
            )
        # caller
        if cfg.caller not in ("user", "asebi"):
            raise ValueError(f"caller must be 'user' or 'asebi'; got {cfg.caller!r}")
        # mask_mode validation
        valid_modes = ("auto-multithresh", "interactive", "user", "none")
        if cfg.imaging.mask_mode not in valid_modes:
            raise ValueError(
                f"imaging.mask_mode must be one of {valid_modes}; "
                f"got {cfg.imaging.mask_mode!r}"
            )
        if cfg.imaging.mask_mode == "interactive" and cfg.caller == "asebi":
            raise ValueError(
                "mask_mode='interactive' is refused when caller='asebi' (viewer-free "
                "operation); give a CRTF region file with mask_mode='user' instead"
            )
        self._validate_imaging_config()
        self._validate_schedule()
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
        if cfg.imaging.mask_mode == "user" and os.path.isfile(cfg.imaging.user_mask):
            with open(cfg.imaging.user_mask, "r", errors="replace") as fh:
                first = fh.readline()
            if not first.lstrip().startswith("#CRTF"):
                raise ValueError(
                    f"imaging.user_mask {cfg.imaging.user_mask!r} is a file but not a CRTF "
                    f"region file (its first line must start with '#CRTF'); give a CRTF "
                    f"file or a CASA mask image"
                )
        # plot options
        if cfg.plots.images_center not in ("phase", "peak"):
            raise ValueError(
                f"plots.images_center must be 'phase' or 'peak'; got {cfg.plots.images_center!r}"
            )
        for name in ("images_gamma", "images_gamma2"):
            val = getattr(cfg.plots, name)
            if val is not None and not val > 0:
                raise ValueError(f"plots.{name} must be positive; got {val!r}")
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
                    "phase_center RA and Dec must be strings (RA in hours, Dec in "
                    "degrees), e.g. ('16:25:45.0', '-24:12:23.0', 'ICRS') or "
                    f"('16h25m45.0s', '-24d12m23.0s', 'ICRS'); got ({ra!r}, {dec!r})"
                )
            if not isinstance(frame, str) or frame.upper() not in SKY_FRAMES:
                raise ValueError(
                    f"phase_center frame must be 'ICRS' or 'J2000'; got {frame!r}"
                )
            # Parsed here, not by CASA: CASA reads a ':'-separated Dec as hours
            # ('-24:12:23.0' -> Dec -3.1 deg). Only radians are passed on.
            try:
                ra_rad, dec_rad = parse_ra_dec(ra, dec)
            except ValueError as exc:
                raise ValueError(f"phase_center: {exc}") from exc
            self._derived["phase_center"] = {
                "frame": frame.upper(), "ra_rad": ra_rad, "dec_rad": dec_rad,
                "ra": ra, "dec": dec,
            }
            if not cfg.phase_shift:
                warnings.warn(
                    "phase_center is set but phase_shift=False; the coordinates "
                    "will be ignored. Set phase_shift=True to apply the shift."
                )

    def _validate_imaging_config(self) -> None:
        """Checks of the ImagingConfig fields added in v0.3 (geometry, depth)."""
        img = self.cfg.imaging
        if img.cell_rule not in CELL_RULES:
            raise ValueError(f"imaging.cell_rule must be one of {CELL_RULES}; got {img.cell_rule!r}")
        if img.cell is not None and not _cell_to_arcsec(img.cell) > 0:
            raise ValueError(f"imaging.cell must be positive; got {img.cell!r}")
        if img.imsize is not None and min(_imsize_list(img.imsize)) < 1:
            raise ValueError(f"imaging.imsize must be positive; got {img.imsize!r}")
        if int(img.nterms) < 1:
            raise ValueError(f"imaging.nterms must be >= 1; got {img.nterms!r}")
        if img.scales is not None and (len(img.scales) == 0
                                       or any(int(x) < 0 for x in img.scales)):
            raise ValueError(f"imaging.scales must be non-negative pixels; got {img.scales!r}")
        if img.threshold_mode not in ("pure_noise", "image_rms"):
            raise ValueError("imaging.threshold_mode must be 'pure_noise' or 'image_rms'; "
                             f"got {img.threshold_mode!r}")
        if not img.threshold_nsigma > 0:
            raise ValueError(f"imaging.threshold_nsigma must be positive; got {img.threshold_nsigma!r}")
        depth = tuple(img.final_depth)
        if any(not d > 0 for d in depth) or any(b >= a for a, b in zip(depth, depth[1:])):
            raise ValueError("imaging.final_depth must be positive and strictly decreasing "
                             f"(e.g. (3, 2, 1, 0.5)); got {img.final_depth!r}")
        if img.final_aperture_arcsec is not None and not img.final_aperture_arcsec > 0:
            raise ValueError("imaging.final_aperture_arcsec must be positive; "
                             f"got {img.final_aperture_arcsec!r}")

    def _validate_schedule(self) -> None:
        """Checks of SelfcalSchedule (mode, steps, amplitude) and the gaincal options it needs."""
        cfg = self.cfg
        sch = cfg.schedule
        if sch.mode not in SELFCAL_MODES:
            raise ValueError(f"schedule.mode must be one of {SELFCAL_MODES}; got {sch.mode!r}")
        if sch.mode == "fixed":
            return
        if tuple(sch.steps) != FIXED_SCHEDULE_STEPS:
            raise ValueError(
                "SelfcalSchedule.steps is used only with mode='fixed'; mode='data' (the default) "
                "chooses the stages and solution intervals from the data. Pass "
                "SelfcalSchedule(mode='fixed', steps=...) to run a fixed list of steps")
        if sch.amplitude != "never":
            raise ValueError(
                "schedule.amplitude: only 'never' is implemented in this version; the conditional "
                "amplitude stage of mode='data' (P18 item 6, P24) will come later. Use "
                "mode='fixed' with an amplitude step for an unconditional one")
        if cfg.gaincal.gaintype != "T":
            raise ValueError("schedule.mode='data' solves phase-only gains with gaintype='T' "
                             f"(one phase per antenna for both correlations); got "
                             f"gaincal.gaintype={cfg.gaincal.gaintype!r}")
        if cfg.gaincal.applymode != "calonly":
            raise ValueError("schedule.mode='data' needs gaincal.applymode='calonly': the solutions "
                             "below the G1 gate are flagged so that those antennas keep the "
                             f"previous stage (got {cfg.gaincal.applymode!r}, which would flag data)")
        # The configured interpolation is linear; a stage switches itself to the
        # nearest form of it only when G2 chose a nearest-interpolation candidate
        # (ASEBI SPEC 10.2 item 6, D34 F2; solint.applycal_interp_string).
        interp = cfg.gaincal.applycal_interp.split(",")[0].strip().lower()
        if not interp.startswith("linear"):
            raise ValueError("schedule.mode='data' predicts and evaluates the stages with linear time "
                             "interpolation of the solutions (a stage switches to nearest only when "
                             "G2 chooses it); gaincal.applycal_interp must be "
                             f"'linear...' (e.g. 'linearPDperobs'); got {cfg.gaincal.applycal_interp!r}")
        parts = [x.strip() for x in cfg.gaincal.combine.split(",") if x.strip()]
        bad = [x for x in parts if x not in ("scan", "spw")]
        if bad:
            raise ValueError(f"gaincal.combine may contain 'scan' and 'spw' in mode='data' (the "
                             f"time combination is chosen per stage); got {cfg.gaincal.combine!r}")

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

        # --- Working MS: the input is only read; J2000 -> ICRS relabelling
        #     (CASA hygiene) is done on a copy when needed ---
        self._prepare_working_vis()

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

        # --- SPW info ---
        tb = casatools.table()
        tb.open(cfg.vis + "/SPECTRAL_WINDOW")
        try:
            num_chan = tb.getcol("NUM_CHAN").tolist()
            nspw = len(num_chan)
        finally:
            tb.close()
        self._derived["num_chan_per_spw"] = num_chan
        self._derived["nspw"] = nspw

        # --- Frequency range and correlations of the target field (used by
        #     the post-tclean checks and the signal-free noise image) ---
        setup = self._spectral_setup(cfg.vis, int(self._derived["resolved_field_id"]))
        self._derived["data_fmin_hz"] = setup["fmin_hz"]
        self._derived["data_fmax_hz"] = setup["fmax_hz"]
        self._derived["noise_stokes"] = _noise_stokes(setup["corr_types"])
        self.justification["derived"]["spectral_setup"] = {
            **setup,
            "noise_stokes": self._derived["noise_stokes"],
            "reason": ("frequency range = channel edges of the field's spws (the band "
                       "every MFS image must cover); noise_stokes = the signal-free "
                       "Stokes image of the parallel hands ((XX-YY)/2 = Q, (RR-LL)/2 = V)"),
        }

        if cfg.verbose:
            print(f"[AJISAI] integration time      : {inttime_sec:.2f} s")
            if self._derived.get("onsource_time_sec"):
                print(f"[AJISAI] on-source time        : {self._derived['onsource_time_sec']/60:.2f} min")
            print(f"[AJISAI] median frequency      : "
                  f"{(self._derived.get('median_freq_hz') or 0)/1e9:.3f} GHz")
            print(f"[AJISAI] spectral windows      : {nspw} (channels: {num_chan})")

    def _prepare_working_vis(self) -> None:
        """Choose the MS that the pipeline reads; never write to the input MS.

        The input MS (``cfg.vis``) is opened read-only by every task. With
        ``relabel_j2000_to_icrs=True`` (default) and J2000 frame codes in the
        input, a copy ``intermediate/input_icrs.ms`` is relabelled J2000 ->
        ICRS (frame code only) and used instead; otherwise the input itself
        is the working MS (stage 0).
        """
        cfg = self.cfg
        assert self.workdir is not None
        working = cfg.vis
        relabel_record: Optional[Dict[str, Any]] = None
        caveat: Optional[str] = None
        if not cfg.relabel_j2000_to_icrs:
            reason = "relabel_j2000_to_icrs=False: the input MS is used as it is (read-only)"
        elif not _ms_needs_relabel(cfg.vis):
            reason = "no J2000 frame code in the input; it is used as it is (read-only)"
        else:
            names, start = _observation_telescopes_and_start(cfg.vis)
            caveat = _relabel_caveat(names, start)
            if caveat is None:
                copy = self.workdir / "intermediate" / "input_icrs.ms"
                _safe_rmtree_path(copy)
                _copy_ms(cfg.vis, copy)
                relabel_record = relabel_J2000_to_ICRS(str(copy), verbose=cfg.verbose)
                working = str(copy)
                reason = ("ALMA data observed before Cycle 3 carry J2000 frame codes on ICRS "
                          "numbers; a copy was relabelled J2000 -> ICRS (frame code only, "
                          "coordinates unchanged) so that the input MS is never written")
            else:
                reason = ("J2000 frame codes kept (genuine J2000); the input MS is used as it is "
                          "(read-only)")
                warnings.warn(f"[AJISAI] {caveat}")
                if cfg.verbose:
                    print(f"[AJISAI]   WARN: {caveat}")
        self._derived["input_vis"] = cfg.vis
        self._derived["working_vis"] = working
        self.justification["derived"]["working_vis"] = {
            "input_vis": cfg.vis,
            "working_vis": working,
            "relabel_j2000_to_icrs": cfg.relabel_j2000_to_icrs,
            "reason": reason,
        }
        if relabel_record is not None:
            self.justification["derived"]["working_vis"]["relabel"] = relabel_record
        if caveat is not None:
            self.justification["derived"]["working_vis"]["relabel_caveat"] = caveat
        if cfg.verbose:
            print(f"[AJISAI] working MS            : {working}")

    @staticmethod
    def _spectral_setup(vis: str, field_id: int) -> Dict[str, Any]:
        """Frequency range (channel edges) and correlation types of a field's spws."""
        msmd = casatools.msmetadata()
        msmd.open(vis)
        try:
            spws = sorted(int(x) for x in msmd.spwsforfield(field_id))
            lo: List[float] = []
            hi: List[float] = []
            corr: List[List[int]] = []
            for spw in spws:
                f = np.asarray(msmd.chanfreqs(spw), dtype=float)
                w = np.abs(np.asarray(msmd.chanwidths(spw), dtype=float))
                lo.append(float((f - w / 2.0).min()))
                hi.append(float((f + w / 2.0).max()))
                for dd in msmd.datadescids(spw=spw):
                    pol = msmd.polidfordatadesc(int(dd))
                    corr.append([int(c) for c in msmd.corrtypesforpol(int(pol))])
        finally:
            msmd.close()
        return {
            "spws": spws,
            "fmin_hz": min(lo) if lo else None,
            "fmax_hz": max(hi) if hi else None,
            "corr_types": corr,
        }

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
        """Determine cell size and image size (scales follow the dirty image).

        - Cell: ``ImagingConfig.cell`` if given; otherwise ``cell_rule``:
          ``"umax"`` (default) = 1/(5 u_max) with u_max from the longest
          unflagged baseline at the highest frequency, or ``"percentile"``
          (legacy v0.2 rule, beam from the 90th-percentile baseline /
          ``cellpix``).
        - Image size: ``ImagingConfig.imsize`` if given; otherwise the
          primary beam down to 20 % of the smallest dish in
          ``ANTENNA/DISH_DIAMETER`` (12 m only as a fallback), rounded up to
          a 5-smooth size.
        """
        cfg = self.cfg
        img = cfg.imaging
        vis = self._derived["working_vis"]

        # --- MRS from 5th-percentile shortest baseline (ALMA Tech Handbook Eq 7.7) ---
        try:
            short_baseline_m = get_baseline_at_percentile(vis, percentile=5)
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

        # --- cell size and image size ---
        cell_info: Optional[Dict[str, Any]] = None
        if img.cell is None or img.imsize is None:
            cell_info = pick_cell_imsize(
                vis,
                npix_per_beam=img.cellpix,
                max_baseline_percentile=90,
                pb_level=0.2,
                uvtaper=img.uvtaper,
                rule=img.cell_rule,
                field=self._derived["resolved_field_id"],
            )
        if img.cell is not None:
            cell_arcsec = _cell_to_arcsec(img.cell)
            cell = img.cell if isinstance(img.cell, str) else f"{cell_arcsec}arcsec"
            cell_reason = "ImagingConfig.cell given by the caller"
        else:
            assert cell_info is not None
            cell_arcsec = cell_info["cell_arcsec"]
            cell = cell_info["cell"]
            cell_reason = cell_info["justification"]["description"]

        if img.imsize is not None:
            imsize_xy = _imsize_list(img.imsize)
            imsize_reason = "ImagingConfig.imsize given by the caller"
        else:
            assert cell_info is not None
            offset = cell_info["justification"]["gaussian_offset_at_pblevel_arcsec"]
            n = _smallest_5_smooth_at_least(int(math.ceil(2 * offset / cell_arcsec)))
            imsize_xy = [n, n]
            imsize_reason = (
                f"covers down to 20% primary-beam attenuation (Baars taper) of the "
                f"{cell_info['dish_diameter_m']:g} m dish ({cell_info['dish']['source']}) at "
                f"{cell_info['pb_freq_hz'] / 1e9:.4f} GHz; rounded to smallest 5-smooth integer"
            )

        if cell_info is not None:
            dish = cell_info["dish"]
            pb_arcsec = cell_info["primary_beam_arcsec"]
            beam_arcsec = cell_info["beam_fwhm_arcsec"]
            taper_arcsec = cell_info["taper_image_fwhm_arcsec"]
            predicted = cell_info["predicted_effective_beam_arcsec"]
        else:
            dish = get_dish_diameter_m(vis)
            freq_hz = self._derived.get("median_freq_hz") or 0.0
            pb_arcsec = (_primary_beam_fwhm_arcsec(freq_hz, diameter_m=dish["diameter_m"])
                         if freq_hz > 0 else None)
            beam_arcsec = None
            taper_arcsec = 0.0
            predicted = 0.0   # no prediction -> the beam check is skipped

        self._derived["cell_arcsec"] = cell_arcsec
        self._derived["cell"] = cell
        self._derived["imsize"] = imsize_xy
        self._derived["fov_arcsec"] = imsize_xy[0] * cell_arcsec
        self._derived["dish_diameter_m"] = dish["diameter_m"]
        self._derived["primary_beam_arcsec"] = pb_arcsec
        self._derived["untapered_beam_arcsec"] = beam_arcsec
        self._derived["taper_image_fwhm_arcsec"] = taper_arcsec
        self._derived["predicted_effective_beam_arcsec"] = predicted
        self.justification["derived"]["cellsize"] = {
            "value_arcsec": cell_arcsec,
            "rule": "given" if img.cell is not None else img.cell_rule,
            "uv_max": (cell_info or {}).get("uv_max"),
            "reason": cell_reason,
        }
        self.justification["derived"]["imsize"] = {
            "value": imsize_xy,
            "reason": imsize_reason,
        }
        self.justification["derived"]["dish_diameter"] = dish

        if cfg.verbose:
            print(f"[AJISAI] cell                  : {cell_arcsec:.6g} arcsec "
                  f"({self.justification['derived']['cellsize']['rule']})")
            print(f"[AJISAI] imsize                : {imsize_xy}")
            print(f"[AJISAI] FoV                   : {self._derived['fov_arcsec']:.2f} arcsec")
            print(f"[AJISAI] dish diameter         : {dish['diameter_m']:g} m ({dish['source']})")
            if pb_arcsec:
                print(f"[AJISAI] primary beam (FWHM)   : {pb_arcsec:.2f} arcsec")
            if beam_arcsec:
                print(f"[AJISAI] untapered beam        : {beam_arcsec:.4f} arcsec")
            if taper_arcsec > 0.0:
                print(f"[AJISAI] uvtaper input         : {list(img.uvtaper)}")
                print(f"[AJISAI] taper image FWHM      : {taper_arcsec:.4f} arcsec")
                print(f"[AJISAI] predicted eff. beam   : {predicted:.4f} arcsec "
                      f"(actual will be verified after dirty image)")
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
          as the centre of the brightest pixel of the dirty image (method A:
          pixel-accurate, never fails, no user input), read in radians from
          the image's coordinate system.

        * If ``cfg.phase_center`` is a ``(ra, dec, frame)`` tuple where
          ``frame`` is ``"ICRS"`` or ``"J2000"``: the user-supplied
          coordinates are used (parsed to radians at validation: RA in
          hours, Dec in degrees). This route is intended for advanced users
          who need sub-pixel accuracy (e.g. from their own imfit on the
          dirty image, or from VLBI / multi-wavelength astrometry).

        Frame handling (exact in any frame the MS uses):

        1. The frame W of the target field's PHASE_DIR is read from the
           working MS (per row for ALMA's variable-reference columns, codes
           mapped through MEASINFO TabRefTypes/TabRefCodes).
        2. The target is converted into W with ``casatools.measures`` when
           its frame differs (never with astropy, whose FK5 differs from
           CASA's J2000 by 20-25 mas).
        3. ``phaseshift(phasecenter="W <ra>rad <dec>rad")`` shifts the data
           and computes UVW (radians only: CASA reads a ':'-separated Dec
           as hours).
        4. The target field is found in the output by name (CASA 6.5
           ``phaseshift`` renumbers the selected field from 0).
        5. FIELD PHASE_DIR, DELAY_DIR, REFERENCE_DIR and SOURCE.DIRECTION
           are written with :func:`ajisai.ms_utils.write_field_direction`,
           each converted into its own frame. ``phaseshift`` alone is not
           enough: CASA 6.5 writes J2000 numbers into an ICRS-coded
           PHASE_DIR (off by the frame bias, up to about 23 mas) and leaves
           DELAY_DIR, REFERENCE_DIR and SOURCE at the old centre.

        ``fixplanets`` is not used: it writes its numbers unconverted with a
        J2000 code, refuses other frames, and with ``fixuvw=True``
        recomputes UVW from the ANTENNA table (0.5 m wrong for ALMA's DV
        antennas). No frame label is changed here; the read-back offsets of
        all written directions from the target are recorded.

        The shifted MS is written to
        ``<workdir>/intermediate/phaseshifted.ms``, and
        ``self._derived['working_vis']`` (and ``resolved_field_id``, when
        the field was renumbered) is updated to point to it. All
        subsequent CASA tasks (initial CLEAN, self-cal iterations) operate
        on the shifted MS.
        """
        cfg = self.cfg
        # Default: working_vis is the input MS (or its relabelled copy), set
        # by _prepare_working_vis; set here too so this method can run alone.
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

        # === Resolve the target direction (radians, with its frame) ===
        if cfg.phase_center is not None:
            ra, dec, frame = cfg.phase_center
            frame = str(frame).upper()
            ra_rad, dec_rad = parse_ra_dec(ra, dec)
            source_label = "user-specified"
            pixel = None
        else:
            dirty_image = Path(_image_name(self.workdir / "intermediate" / "dirty", 1))
            if not dirty_image.exists():
                raise RuntimeError(
                    "Phase shift auto-detection requires the dirty image, "
                    "but dirty.image was not found at "
                    f"{dirty_image}. _make_dirty_image must run first."
                )
            peak = _max_pixel_world_coords(str(dirty_image))
            ra, dec, frame = peak["ra"], peak["dec"], peak["frame"]
            ra_rad, dec_rad = peak["ra_rad"], peak["dec_rad"]
            pixel = peak["pixel"]
            source_label = f"auto: max pixel of dirty image (frame={frame})"
        if frame not in SKY_FRAMES:
            # Image was in some other frame (e.g. B1950); refuse rather than guess
            raise RuntimeError(
                f"The target frame is {frame!r}, but phase shifting only supports "
                f"{SKY_FRAMES}. Specify cfg.phase_center explicitly with a supported frame."
            )

        # === The target in the frame of the field's PHASE_DIR ===
        working = self._derived["working_vis"]
        field_in = str(self._derived["resolved_field_id"])
        ms_frame = field_direction_frame(working, int(field_in), "PHASE_DIR")
        if ms_frame not in SKY_FRAMES:
            raise RuntimeError(
                f"PHASE_DIR of field {field_in} in {working} is in frame {ms_frame!r}; "
                f"phase shifting supports only {SKY_FRAMES}"
            )
        ra_w, dec_w = convert_direction(ra_rad, dec_rad, frame, ms_frame)
        phasecenter = casa_direction_string(ms_frame, ra_w, dec_w)

        # === Phaseshift to new MS (data and UVW) ===
        shifted_vis = str(self.workdir / "intermediate" / "phaseshifted.ms")
        _safe_rmtree_path(shifted_vis)
        casatasks.phaseshift(
            vis=working,
            outputvis=shifted_vis,
            keepmms=True,
            field=field_in,
            spw="",
            scan="",
            intent="",
            array="",
            observation="",
            datacolumn="all",
            phasecenter=phasecenter,
        )

        # === FIELD and SOURCE directions, each in its own frame ===
        field_name = self._derived.get("resolved_field_name")
        if field_name is not None:
            field_out = field_id_by_name(shifted_vis, str(field_name))
        else:
            field_out = int(field_in)
        written = write_field_direction(shifted_vis, [field_out], ra_w, dec_w, ms_frame)
        check = field_direction_offsets(shifted_vis, field_out, ra_rad, dec_rad, frame)

        # === Update state ===
        self._derived["working_vis"] = shifted_vis
        self._derived["phase_shift_applied"] = True
        if str(field_out) != field_in:
            self._derived["input_field_id"] = field_in
            self._derived["resolved_field_id"] = str(field_out)
        self.justification["derived"]["phase_shift"] = {
            "applied": True,
            "source": source_label,
            "input_frame": frame,
            "input_ra": ra,
            "input_dec": dec,
            "target_rad": {"frame": frame, "ra": ra_rad, "dec": dec_rad},
            "dirty_max_pixel": pixel,
            "ms_frame": ms_frame,
            "phasecenter": phasecenter,
            "field_id_in": int(field_in),
            "field_id_out": int(field_out),
            "shifted_vis": shifted_vis,
            "field_source_write": written,
            "readback_offset_mas": check,
            "reason": (
                f"shifted source to phase center; source location was {source_label}; "
                f"phaseshift to the target in the field's frame {ms_frame} "
                f"(casatools.measures), then FIELD PHASE_DIR/DELAY_DIR/REFERENCE_DIR and "
                f"SOURCE.DIRECTION written in their own frames; no fixplanets, UVW from "
                f"phaseshift, no frame label changed"
            ),
        }

        if cfg.verbose:
            print(f"[AJISAI] phase shift applied   : {source_label}")
            target_text = rad_to_radec(ra_rad, dec_rad, precision=7)
            print(f"[AJISAI]   target {frame:<5s}        : {target_text}")
            print(f"[AJISAI]   phasecenter          : {phasecenter}")
            if check.get("max_mas") is not None:
                print(f"[AJISAI]   read-back offset     : {check['max_mas']:.3g} mas (max over "
                      f"FIELD/SOURCE)")
            if str(field_out) != field_in:
                print(f"[AJISAI]   field id             : {field_in} -> {field_out} "
                      f"(renumbered by phaseshift)")
            print(f"[AJISAI]   shifted vis          : {shifted_vis}")

    # ----------------------------------------------------------------------
    # _make_dirty_image: initial RMS reference (no CLEAN, niter=0)
    # ----------------------------------------------------------------------
    def _make_dirty_image(self) -> None:
        """Run tclean(niter=0): dirty-image peak, RMS and the synthesized beam."""
        cfg = self.cfg
        assert self.workdir is not None
        base = self.workdir / "intermediate" / "dirty"
        _safe_rmtree_glob(self.workdir / "intermediate", "dirty.*")

        self._tclean(self._derived["working_vis"], base, niter=0, threshold_jy=0.0, dirty=True)
        fits_path = self.workdir / "intermediate" / "dirty.fits"
        stats = self._export_and_stats(base, 1, fits_path)
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
                    f"Current cell size = {self._derived['cell_arcsec']:.4g} arcsec. "
                    f"With cell_rule='percentile' the cell is based on the prediction; "
                    f"for tighter pixel sampling use cell_rule='umax' (default) or "
                    f"give ImagingConfig.cell/imsize explicitly."
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
    # _measure_noise: thermal noise from a signal-free dirty image
    # ----------------------------------------------------------------------
    def _measure_noise(self) -> None:
        """Measure sigma_th, the thermal noise, once on a signal-free image.

        The dirty image of ``(XX - YY)/2`` (Stokes Q, linear feeds) or
        ``(RR - LL)/2`` (Stokes V, circular feeds) has the noise of the
        Stokes I image but no total-intensity signal or sidelobes, so its
        off-source RMS is a pure-noise estimate. It sets the absolute CLEAN
        threshold of every self-cal image (``threshold_nsigma x sigma_th``,
        the same in every iteration) and the depth stages of the final
        image. Without a parallel-hand pair the Stokes I dirty-image RMS is
        used and the fallback is recorded.
        """
        cfg = self.cfg
        assert self.workdir is not None
        stokes = self._derived.get("noise_stokes")
        if stokes is None:
            sigma = float(self._derived["dirty_rms"])
            info: Dict[str, Any] = {
                "value_jy_beam": sigma,
                "stokes": "I",
                "image": _image_name(self.workdir / "intermediate" / "dirty", 1),
                "reason": ("no parallel-hand pair (XX,YY or RR,LL) for a signal-free image; "
                           "off-source RMS of the Stokes I dirty image used instead (includes "
                           "source sidelobes, so it can overestimate the noise)"),
            }
            if cfg.imaging.threshold_mode == "pure_noise":
                warnings.warn(f"[AJISAI] {info['reason']}")
        else:
            interm = self.workdir / "intermediate"
            base = interm / f"noise_{stokes}"
            _safe_rmtree_glob(interm, f"noise_{stokes}.*")
            self._tclean(self._derived["working_vis"], base, niter=0, threshold_jy=0.0,
                         dirty=True, stokes=stokes)
            fits_path = interm / f"noise_{stokes}.fits"
            stats = self._export_and_stats(base, 1, fits_path)
            sigma = float(stats["rms_jy_beam"])
            info = {
                "value_jy_beam": sigma,
                "stokes": stokes,
                "image": _image_name(base, 1),
                "rms_info": stats["rms_info"],
                "reason": (f"off-source RMS of the Stokes {stokes} dirty image "
                           f"({'(XX-YY)/2' if stokes == 'Q' else '(RR-LL)/2'}): signal-free, "
                           f"same weighting, cell and imsize as the Stokes I images"),
            }
        if not (np.isfinite(sigma) and sigma > 0):
            raise RuntimeError(f"sigma_th is not positive ({sigma!r}); cannot set CLEAN thresholds")
        self._derived["sigma_th"] = sigma
        self.justification["derived"]["sigma_th"] = info
        if cfg.verbose:
            print(f"[AJISAI] sigma_th              : {sigma * 1e6:.3f} uJy/bm "
                  f"(Stokes {info['stokes']} dirty image)")

    # ----------------------------------------------------------------------
    # _compute_scales: multiscale scales from the beam and the mask region
    # ----------------------------------------------------------------------
    def _compute_scales(self) -> None:
        """Multiscale scales (pixels) used by every multiscale/mtmfs tclean.

        ``ImagingConfig.scales`` if given; otherwise ``[0, 1, 3] x b`` with
        ``b`` the dirty-image beam FWHM in pixels (the v0.2 rule, from CASA's
        multiscale example). Which scales serve a source best depends on its
        structure and there is no unique answer; a caller that knows the
        source (ASEBI's final image) gives its own scales. The mask-bounded
        ladder :func:`multiscale_scales` stays available to callers but is not
        the default: tested on one source only, and on a single long-baseline
        EB it reached scales the uv coverage does not see.

        Then, when the deconvolver uses scales, every scale (given or
        derived) whose scale-PSF, from the dirty image's PSF (the PSF of every
        CLEAN of the run), does not peak at the component's own position is
        dropped: the uv coverage does not see that scale, and when the
        largest sidelobe is negative CASA's multiscale minor cycle refuses to
        start and the CLEAN ends with an empty model
        (:func:`scale_psf_response`). A dropped given scale is warned about;
        both are recorded.
        """
        cfg = self.cfg
        img = cfg.imaging
        given = img.scales is not None
        if given:
            scales = [int(x) for x in img.scales]
            reason = "ImagingConfig.scales given by the caller"
        else:
            dirty = self.justification["derived"].get("dirty_image") or {}
            beam_as = math.sqrt((dirty.get("bmaj_arcsec") or 0.0) * (dirty.get("bmin_arcsec") or 0.0))
            beam_pix = beam_as / self._derived["cell_arcsec"] if beam_as > 0 else 0.0
            scales = scale_ladder(beam_pix, 3.0 * beam_pix * (1 + 1e-9))
            reason = (f"[0, 1, 3] x beam (beam = {beam_pix:.2f} pix; the v0.2 rule); larger scales only "
                      f"when the caller gives them")
        used = int(img.nterms) > 1 or img.deconvolver in ("multiscale", "mtmfs")
        candidates = list(scales)
        psf_check: Optional[Dict[str, Any]] = None
        if used and any(int(s) > 0 for s in scales):
            assert self.workdir is not None
            psf_path = _image_name(self.workdir / "intermediate" / "dirty", 1, "psf")
            scales, response = psf_supported_scales(_read_image_plane(psf_path), scales)
            dropped = [r["scale"] for r in response if not r["ok"] and r["scale"] > 0]
            psf_check = {
                "psf": psf_path,
                "response": response,
                "dropped": dropped,
                "rule": ("a scale is kept when its scale-PSF (the dirty PSF convolved twice with "
                         "CASA's component, MatrixCleaner::makeScale/makePsfScales) has its "
                         "largest magnitude at the PSF peak, the component's own position; "
                         "otherwise the uv coverage does not see the scale, and with a negative "
                         "largest sidelobe CASA's multiscale minor cycle refuses to start and "
                         "no component is CLEANed"),
            }
            if dropped:
                reason += (f"; scales {dropped} dropped: their scale-PSF does not peak at the "
                           f"component's position (no baseline short enough to see them)")
                if given:
                    warnings.warn(f"[AJISAI] ImagingConfig.scales {candidates}: scales {dropped} "
                                  f"dropped, their scale-PSF does not peak at the component's "
                                  f"position (with a negative sidelobe peak CASA's multiscale "
                                  f"minor cycle would not start); using {scales}")
        self._derived["scales"] = scales
        self.justification["derived"]["scales"] = {
            "value": scales,
            "used": used,
            "deconvolver": _deconvolver_kwargs(img.deconvolver, img.nterms, scales)["deconvolver"],
            "reason": reason,
            "candidates": candidates,
            "psf_check": psf_check,
        }
        if cfg.verbose:
            print(f"[AJISAI] multiscale scales     : {scales} pix"
                  + ("" if used else " (not used by this deconvolver)")
                  + (f" (dropped {psf_check['dropped']}: the uv coverage does not see them)"
                     if psf_check and psf_check["dropped"] else ""))

    def _user_mask_region(self) -> Optional[np.ndarray]:
        """The user mask (CRTF or mask image) on the dirty-image grid, or None."""
        cfg = self.cfg
        if cfg.imaging.mask_mode != "user" or not cfg.imaging.user_mask:
            return None
        assert self.workdir is not None
        path = cfg.imaging.user_mask
        dirty = _image_name(self.workdir / "intermediate" / "dirty", 1)
        try:
            if os.path.isdir(path):
                region = _read_image_plane(path)
                if region.shape != _read_image_plane(dirty).shape:
                    warnings.warn(f"[AJISAI] user mask {path} is not on the image grid; "
                                  f"scales are not bounded by it")
                    return None
            else:
                out = self.workdir / "intermediate" / "user_mask_on_grid.mask"
                _safe_rmtree_path(out)
                casatasks.makemask(mode="copy", inpimage=dirty, inpmask=path,
                                   output=str(out), overwrite=True)
                region = _read_image_plane(out)
        except Exception as e:  # scales fall back to (0, b, 3b)
            warnings.warn(f"[AJISAI] could not rasterise user mask {path}: {e}")
            return None
        return np.asarray(region) > 0.5

    # ----------------------------------------------------------------------
    # _make_initial_clean: pre-selfcal CLEAN image (Iter 0)
    # ----------------------------------------------------------------------
    def _loop_threshold_jy(self, previous_rms: float) -> float:
        """CLEAN threshold of a self-cal image (initial CLEAN and every iteration)."""
        img = self.cfg.imaging
        if img.threshold_mode == "pure_noise":
            return float(img.threshold_nsigma * self._derived["sigma_th"])
        return float(img.threshold_factor * previous_rms)

    def _threshold_reason(self) -> str:
        img = self.cfg.imaging
        if img.threshold_mode == "pure_noise":
            return (f"{img.threshold_nsigma:g} x sigma_th (absolute, the same for every "
                    f"self-cal image)")
        return f"{img.threshold_factor:g} x off-source RMS of the previous image (legacy)"

    def _make_initial_clean(self) -> None:
        """Run the first CLEAN to establish the model for the first gaincal.

        Operates on ``self._derived['working_vis']`` (the input MS, its
        relabelled copy, or the phase-shifted MS) with ``savemodel='none'``:
        the working MS is never written. The model is predicted into a
        temporary copy at the first self-cal iteration.
        """
        cfg = self.cfg
        assert self.workdir is not None
        working_vis = self._derived["working_vis"]
        interm = self.workdir / "intermediate"
        base = interm / "clean_iter0"
        _safe_rmtree_glob(interm, "clean_iter0.*")

        threshold_jy = self._loop_threshold_jy(self._derived["dirty_rms"])
        # ``usemask_kwargs`` carries everything mask-related INCLUDING the
        # ``interactive`` flag. For mask_mode='interactive', interactive=True
        # is set here; in all other modes (and in subsequent self-cal
        # iterations), interactive=False.
        usemask_kwargs = self._build_usemask_kwargs()
        nterms = self._tclean(working_vis, base, niter=cfg.imaging.niter,
                              threshold_jy=threshold_jy, mask_kwargs=usemask_kwargs)
        # Self-calibration needs this model: stop here, with the reason, if it is empty.
        # niter = 0 is a dirty image (e.g. ASEBI's bootstrap image): no model is expected.
        if int(cfg.imaging.niter) > 0:
            self._check_clean_model(base, nterms, threshold_jy, "initial CLEAN (iteration 0)")

        fits_path = interm / "clean_iter0.fits"
        stats = self._export_and_stats(base, nterms, fits_path)

        # Record as iteration 0
        rec = {
            "iteration": 0,
            "label": "no_selfcal",
            "calmode": "-",
            "solint": "-",
            "solint_s": None,
            "peak_jy_beam": stats["peak_jy_beam"],
            "rms_jy_beam": stats["rms_jy_beam"],
            "snr": stats["snr"],
            "dynamic_range": stats["dynamic_range"],
            "bmaj_arcsec": stats["bmaj_arcsec"],
            "bmin_arcsec": stats["bmin_arcsec"],
            "threshold_jy": threshold_jy,
            "n_gaincal_solutions": None,
            "status": "ok",
            "failure_reason": None,
            "image_path": _image_name(base, nterms),
            "image_base": str(base),
            "fits_path": str(fits_path),
            "caltable": None,
            "selfcal_ms": None,
        }
        self.metrics.append(rec)
        self.justification["iterations"].append(rec)
        self._add_stage({
            "stage": 0,
            "label": "no_selfcal",
            "status": "ok",
            "data_ms": str(working_vis),
            "stage_ms": None,
            "image_base": str(base),
            "image": _image_name(base, nterms),
            "nterms": nterms,
            "model_image": _model_images(base, nterms),
            "threshold_jy": threshold_jy,
            "threshold_reason": self._threshold_reason(),
            "solve": None,
        })

        # Initial state for the iteration chain. good_vis is the MS whose
        # temporary copy the first gaincal/applycal will run on; its MODEL
        # column does not hold this model yet (the working MS is read-only).
        self._derived["good_vis"] = str(working_vis)
        self._derived["good_vis_model_from"] = None
        self._derived["last_image_base"] = str(base)
        self._derived["last_nterms"] = nterms
        self._derived["last_dr"] = stats["dynamic_range"]
        self._derived["last_rms"] = stats["rms_jy_beam"]
        self._derived["initial_mask_path"] = _image_name(base, nterms, "mask")

        if cfg.verbose:
            print(f"[AJISAI] iter 0 (no selfcal)   : DR={stats['dynamic_range']:.1f}, "
                  f"peak={stats['peak_jy_beam']*1e3:.3f} mJy/bm, "
                  f"RMS={stats['rms_jy_beam']*1e6:.2f} uJy/bm "
                  f"(threshold {threshold_jy*1e6:.2f} uJy)")

    # ----------------------------------------------------------------------
    # _run_iteration: one self-cal iteration
    # ----------------------------------------------------------------------
    def _run_iteration(self, idx: int, step: SelfcalStep) -> None:
        """Run one self-cal iteration with failure detection.

        Data hygiene (the caller's MS and earlier stage MSs are never
        written):

        1. copy the previous stage's MS to a temporary MS;
        2. predict the previous image's model into the copy if the previous
           stage MS does not already carry it (``ft``, checked);
        3. gaincal on the copy; applycal on the copy;
        4. split CORRECTED of the copy -> ``intermediate/selfcal_<idx>.ms``
           (DATA only) and delete the copy;
        5. image ``selfcal_<idx>.ms`` (``savemodel='none'``) and predict that
           image's model into its MODEL column (checked). The stage MS then
           holds DATA + MODEL and is not modified again.
        """
        cfg = self.cfg
        assert self.workdir is not None
        interm = self.workdir / "intermediate"
        caltable = interm / f"sc{idx}.pcal"
        out_vis = interm / f"selfcal_{idx}.ms"
        tmp_vis = interm / f"selfcal_{idx}_applycal.tmp.ms"
        clean_base = interm / f"clean_sc{idx}"
        fits_path = interm / f"clean_sc{idx}.fits"

        # Cleanup any stale files from previous run (safe: only within workdir)
        _safe_rmtree_path(caltable)
        _safe_rmtree_path(out_vis)
        _safe_rmtree_path(tmp_vis)
        _safe_rmtree_glob(interm, f"clean_sc{idx}.*")

        # Resolve solint expression like "6*IT"
        solint_str = self._resolve_solint(step.solint)
        field_id = self._derived["resolved_field_id"]
        refant = self._derived["refant"]
        prev_vis = self._derived["good_vis"]
        prev_base = self._derived["last_image_base"]
        prev_nterms = self._derived["last_nterms"]

        if cfg.verbose:
            print(f"[AJISAI] --- iter {idx} ({step.label}) "
                  f"calmode={step.calmode} solint={solint_str} ---")

        # === 1-4. solve and apply on a temporary copy of the previous stage MS ===
        _copy_ms(prev_vis, tmp_vis)
        try:
            if self._derived.get("good_vis_model_from") != prev_base:
                model_info = self._predict_model(tmp_vis, prev_base, prev_nterms)
            else:
                model_info = {"model_image": _model_images(prev_base, prev_nterms),
                              "nterms": prev_nterms,
                              "reffreq_hz": self._derived.get("good_vis_model_reffreq_hz")}

            n_solutions, gaincal_ok = self._run_gaincal(
                input_vis=str(tmp_vis),
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
                # with its own solint against the same good_vis.
                self._record_skipped_iteration(
                    idx, step, solint_str,
                    reason=(f"gaincal returned 0 solutions (all below "
                            f"minsnr={cfg.gaincal.minsnr})"),
                )
                return

            # === plot the gain table (justification artifact) ===
            self._plot_gain_table(str(caltable), idx, step)

            # === applycal on the copy, then split corrected -> stage MS ===
            self._applycal(tmp_vis, caltable)
            casatasks.split(
                vis=str(tmp_vis),
                outputvis=str(out_vis),
                datacolumn="corrected",
            )
        finally:
            _safe_rmtree_path(tmp_vis)

        solution_stats = self._caltable_solution_stats(str(caltable))

        # === 5. image the new stage MS (read-only), then store its model ===
        # Self-cal iterations reuse the mask of the initial CLEAN (or the
        # user's CRTF in mask_mode='user'); interactive=False always.
        threshold_jy = self._loop_threshold_jy(self._derived["last_rms"])
        usemask_kwargs = self._build_usemask_kwargs(
            existing_mask=self._derived.get("initial_mask_path"),
        )
        nterms = self._tclean(out_vis, clean_base, niter=cfg.imaging.niter,
                              threshold_jy=threshold_jy, mask_kwargs=usemask_kwargs)
        own_model = self._predict_model(out_vis, clean_base, nterms)

        # === 6. compute image stats (DR-based) on the tt0 image ===
        stats = self._export_and_stats(clean_base, nterms, fits_path)

        # === 7. anomaly detection (informational only — does NOT alter flow) ===
        is_anomaly, anomaly_reason = self._detect_iter_anomaly(stats, n_solutions)

        # === 8. record metrics and the solve information ===
        g = cfg.gaincal
        solve = build_solve_info(
            table=str(caltable),
            solint=solint_str,
            solint_s=solint_seconds(solint_str, self._derived.get("avg_int_time_sec")),
            combine=g.combine,
            refant=refant,
            field=str(field_id),
            calmode=step.calmode,
            gaintype=g.gaintype,
            minsnr=g.minsnr,
            minblperant=g.minblperant,
            solnorm=step.solnorm,
            interp=g.applycal_interp,
            applymode=g.applymode,
            calwt=g.calwt,
            spwmap=g.spwmap,
            model_image=model_info["model_image"],
            model_nterms=model_info["nterms"],
            model_reffreq_hz=model_info.get("reffreq_hz"),
            solved_on=str(prev_vis),
            stage_ms=str(out_vis),
            n_solutions=n_solutions,
            flagged_solution_fraction=solution_stats,
        )
        rec = {
            "iteration": idx,
            "label": step.label,
            "calmode": step.calmode,
            "solint": solint_str,
            "solint_s": solve["solint_s"],
            "peak_jy_beam": stats["peak_jy_beam"],
            "rms_jy_beam": stats["rms_jy_beam"],
            "snr": stats["snr"],
            "dynamic_range": stats["dynamic_range"],
            "bmaj_arcsec": stats["bmaj_arcsec"],
            "bmin_arcsec": stats["bmin_arcsec"],
            "threshold_jy": threshold_jy,
            "n_gaincal_solutions": n_solutions,
            "status": "anomaly" if is_anomaly else "ok",
            "anomaly_reason": anomaly_reason,
            "image_path": _image_name(clean_base, nterms),
            "image_base": str(clean_base),
            "fits_path": str(fits_path),
            "caltable": str(caltable),
            "selfcal_ms": str(out_vis),
        }
        self.metrics.append(rec)
        self.justification["iterations"].append(rec)
        self._add_stage({
            "stage": idx,
            "label": step.label,
            "status": rec["status"],
            "data_ms": str(out_vis),
            "stage_ms": str(out_vis),
            "stage_ms_columns": "DATA + MODEL (model of this stage's image)",
            "image_base": str(clean_base),
            "image": _image_name(clean_base, nterms),
            "nterms": nterms,
            "model_image": own_model["model_image"],
            "threshold_jy": threshold_jy,
            "threshold_reason": self._threshold_reason(),
            "solve": solve,
        })

        # === 9. ALWAYS update good_vis (all iterations complete by design) ===
        # AJISAI design: anomalies are logged but do NOT trigger fallback.
        # The MS chain is always continuous. Best-iteration selection at the
        # end handles cases where an anomaly degraded the result.
        self._derived["good_vis"] = str(out_vis)
        self._derived["good_vis_model_from"] = str(clean_base)
        self._derived["good_vis_model_reffreq_hz"] = own_model.get("reffreq_hz")
        self._derived["last_image_base"] = str(clean_base)
        self._derived["last_nterms"] = nterms
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
    # schedule.mode="data": stages and solution intervals from the data
    # ----------------------------------------------------------------------
    def _run_data_driven(self) -> None:
        """Phase self-calibration with data-driven stages (ASEBI SPEC v0.3 section 10.2).

        Stage 1 solves one phase per execution block; its held-out change
        (G2) is computed and the stage is skipped only below -2 sigma. Each
        later stage measures s, r_i, D_eff and kappa on the current data and
        model, builds the candidate intervals (divisors of the median scan
        length in integrations, the scan, the EB), gates them on the
        predicted solution SNR (G1), predicts V(T), evaluates the predicted
        best, its neighbours and the EB on held-out integrations (G2, numpy,
        no gaincal per candidate), chooses with rules A/B and accepts when
        the corrected improvement exceeds max(1 sigma, 0.001); gaincal runs
        only for the chosen interval. At most three phase stages; the run
        stops at the first stage that is not accepted. Every accepted stage
        is re-imaged, so the last accepted stage's image is the final
        self-cal image. All quantities, candidates, gates, decisions and
        reasons are recorded under ``stages[k]["selection"]``.
        """
        cfg = self.cfg
        combine_spw = "spw" in [x.strip() for x in cfg.gaincal.combine.split(",")]
        self._sc: Dict[str, Any] = {
            "state0": None, "geom": None, "halves": False, "chain": [], "best_iteration": 0,
            "combine_spw": combine_spw, "pending_g3": None, "spwmap": None,
        }
        summary: Dict[str, Any] = {
            "mode": "data",
            "spec": ("ASEBI SPEC v0.3 section 10.2, appendices C.2-C.3; DESIGN_REVIEW P17, P18, P23, "
                     "D34 (F2, F3)"),
            "rules": {
                "g1_snr_min": _solint.G1_SNR_MIN,
                "gaincal_minsnr": float(cfg.gaincal.minsnr),
                "accept": f"corrected held-out improvement > max({_solint.ACCEPT_SIGMA:g} sigma, "
                          f"{_solint.ACCEPT_FLOOR:g})",
                "stage1_skip": f"corrected held-out improvement < -{_solint.STAGE1_SKIP_SIGMA:g} sigma",
                "max_phase_stages": _solint.MAX_PHASE_STAGES,
                "loss_table": {"snr": list(_solint.LOSS_TABLE_SNR),
                               "neg_ln_C": list(_solint.LOSS_TABLE_NEG_LN_C)},
                "amplitude": cfg.schedule.amplitude,
                "candidate_step": (f"least common multiple of the integration times (within "
                                   f"{_solint.IT_MATCH_RTOL:.0%}) up to "
                                   f"{_solint.IT_LCM_MAX_FACTOR:g} x the longest, else the "
                                   f"median integration time"),
                "nearest_trigger_two_D_rad2": _solint.NEAREST_TRIGGER_2D,
                "applycal_interp": cfg.gaincal.applycal_interp,
            },
            "combine_spw": combine_spw,
            "stages": [],
        }
        self.justification["selfcal"] = summary
        stop_reason = f"the limit of {_solint.MAX_PHASE_STAGES} phase stages was reached"
        for k in range(1, _solint.MAX_PHASE_STAGES + 1):
            decision, reason = self._data_stage(k)
            summary["stages"].append({"stage": k, "decision": decision, "reason": reason})
            if decision == "stop":
                stop_reason = reason
                break
        self._finish_g3()
        # Hook for the conditional amplitude stage (ASEBI SPEC v0.3 10.2 item 9, P24):
        # sigma_a SNR_EB > 2, SNR_EB >= 10, G2 and G3; 'ap', T = EB, on top of the phase
        # tables, normalised per observation, calwt=False. Not implemented in this version.
        summary["amplitude_stage"] = {
            "attempted": False,
            "reason": ("schedule.amplitude='never': the conditional amplitude stage is not "
                       "implemented in this version"),
        }
        accepted = [st["stage"] for st in self.stages
                    if st["stage"] > 0 and st.get("status") in ("ok", "anomaly")]
        summary.update({
            "accepted_stages": accepted,
            "best_stage": accepted[-1] if accepted else 0,
            "stop_reason": stop_reason,
            "g2_split": (self._sc["geom"] and _solint.split_mode(self._sc["geom"],
                                                                 self._sc["halves"])),
            "spwmap": self._sc["spwmap"],
            "spwmap_record": self._sc.get("spwmap_record"),
            "split_note": self._sc.get("split_note"),
            "it_step": (dict(self._sc["geom"].it_step_record) if self._sc["geom"] is not None
                        else None),
            # how the integrations were found (INTERVAL; ASEBI SPEC 4 item 4, D38 e)
            "time_geometry": (dict(self._sc["geom"].time_record) if self._sc["geom"] is not None
                              else None),
        })
        self._derived["selfcal_summary"] = summary
        if cfg.verbose:
            print(f"[AJISAI] self-cal (mode='data'): accepted stages {accepted or 'none'}; "
                  f"{stop_reason}")

    def _read_state(self, vis) -> "_solint.VisSummary":
        """Reduce DATA/MODEL/WEIGHT/FLAG of an AJISAI MS once; track its phase to state 0."""
        sc = self._sc
        fid = self._derived["resolved_field_id"]
        st = read_vis_summary(str(vis), fid, band_halves=sc["halves"])
        if sc["state0"] is None:
            geom = _solint.geometry(st)
            if _solint.split_mode(geom, True) == "band" and not sc["halves"]:
                try:
                    st = read_vis_summary(str(vis), fid, band_halves=True)
                    sc["halves"] = True
                    geom = _solint.geometry(st)
                except RuntimeError as e:   # a spw with one channel: no band halves
                    sc["split_note"] = (f"scans have fewer than {_solint.BAND_SPLIT_BELOW_NINT} "
                                        f"integrations but the band cannot be halved ({e}); "
                                        f"G2 uses the even/odd split")
                    warnings.warn(f"[AJISAI] {sc['split_note']}")
            sc["state0"] = st
            sc["geom"] = geom
        else:
            st.track_phase(sc["state0"])
        return st

    def _data_spwmap(self, state, caltable: str) -> Optional[Dict[str, Any]]:
        """applycal spwmap of a combined-spw table, from the spw labels gaincal wrote in it.

        ASEBI SPEC 9 item 3 (D38 d; convention audit 2026-10-16, combine item
        3 and time item 3): per observation, every spw with data is mapped to
        the label gaincal wrote for that observation, read from all rows of the
        table (flagged solutions included). The lowest spw of an observation is
        not always the label (EBs of one setup merged by concat: labels 0, 1, 1;
        a lowest spw without rows in some intervals), and a map built from the
        spws with valid data missed a fully flagged lowest spw that still
        labels the solutions ([0, 1] instead of [0, 0]: no spw-1 row
        corrected). An observation with several labels has its other labels'
        rows relabelled to its lowest label (a direct edit of the table AJISAI
        just wrote, recorded); when no single map serves every observation,
        applycal runs per observation. A configured ``GainCalConfig.spwmap``
        that differs from the table's is warned about and not used. Returns
        None without spw combination.
        """
        g = self.cfg.gaincal
        if not self._sc["combine_spw"]:
            return None
        labels = self._caltable_spw_labels(caltable)
        spws: Dict[int, List[int]] = {}
        for o in np.unique(state.obs):
            spws[int(o)] = sorted(int(x) for x in np.unique(state.spw[state.obs == o]))
        nspw = int(self._derived.get("nspw") or (int(state.spw.max()) + 1))
        nspw = max([nspw] + [max(v) + 1 for v in labels.values() if v])
        rec = _solint.spwmap_from_labels(labels, spws, nspw)
        rec["labels_by_observation"] = {str(o): list(v) for o, v in sorted(labels.items())}
        if rec["relabel"]:
            rec["relabelled_rows"] = self._relabel_caltable_spws(caltable, rec["relabel"])
        if g.spwmap:
            conf = [int(x) for x in g.spwmap]
            agrees = rec["global"] is not None and conf == rec["global"]
            rec["configured"] = conf
            if not agrees:
                msg = (f"GainCalConfig.spwmap {conf} does not match the spw labels in {caltable} "
                       f"(per observation {rec['label']}); the table's map is used "
                       f"({rec['global'] if rec['global'] is not None else rec['per_observation']})")
                rec["configured_warning"] = msg
                warnings.warn(f"[AJISAI] {msg}")
        return rec

    @staticmethod
    def _caltable_spw_labels(caltable: str) -> Dict[int, List[int]]:
        """Observation -> the SPECTRAL_WINDOW_ID values of all rows of a gain table."""
        tb = casatools.table()
        tb.open(str(caltable))
        try:
            if tb.nrows() == 0:
                return {}
            obs = tb.getcol("OBSERVATION_ID")
            spw = tb.getcol("SPECTRAL_WINDOW_ID")
        finally:
            tb.close()
        return {int(o): sorted(set(int(x) for x in spw[obs == o])) for o in np.unique(obs)}

    @staticmethod
    def _relabel_caltable_spws(caltable: str, relabel: Dict[int, Any]) -> Dict[str, int]:
        """Give every row of an observation its lowest spw label (one label per observation).

        A combined-spw solution is one solution for all spws of its interval; its
        label only names the spw whose frequency linearPD scales the phase from
        (a ratio of the spw frequencies, < 1 % across a band). Rows that would
        duplicate an existing (time, antenna) solution of the target label are
        left alone. The table is then rewritten in gaincal's row order
        (spw, time, antenna): applycal reads each spw's solutions in row order,
        and relabelled rows left at the end of the table were mis-interpolated
        (CASA 6.5.1, ``tests/test_casa_spwmap.py``). Returns the number of
        relabelled rows per observation.
        """
        out: Dict[str, int] = {}
        caltable = str(caltable)
        tb = casatools.table()
        tb.open(caltable, nomodify=False)
        try:
            obs = tb.getcol("OBSERVATION_ID")
            spw = tb.getcol("SPECTRAL_WINDOW_ID")
            t = tb.getcol("TIME")
            ant = tb.getcol("ANTENNA1")
            new = spw.copy()
            for o, (labs, target) in relabel.items():
                sel = (obs == int(o)) & (spw != int(target))
                have = set(zip(np.round(t[(obs == int(o)) & (spw == int(target))], 6).tolist(),
                               ant[(obs == int(o)) & (spw == int(target))].tolist()))
                idx = [i for i in np.flatnonzero(sel)
                       if (round(float(t[i]), 6), int(ant[i])) not in have]
                new[idx] = int(target)
                out[str(o)] = len(idx)
            changed = bool(np.any(new != spw))
            if changed:
                tb.putcol("SPECTRAL_WINDOW_ID", new)
        finally:
            tb.close()
        if changed:
            tmp = caltable.rstrip("/") + ".sorted.tmp"
            _safe_rmtree_path(tmp)
            tb.open(caltable)
            try:
                srt = tb.query("", sortlist="SPECTRAL_WINDOW_ID,TIME,ANTENNA1")
                try:
                    srt.copy(newtablename=tmp, deep=True, valuecopy=True)
                finally:
                    srt.close()
            finally:
                tb.close()
            _safe_rmtree_path(caltable)
            os.rename(tmp, caltable)
        return out

    def _data_stage(self, k: int) -> Tuple[str, str]:
        """One stage of mode="data"; returns (decision, reason) with decision apply|skip|stop."""
        cfg = self.cfg
        sc = self._sc
        assert self.workdir is not None
        interm = self.workdir / "intermediate"
        caltable = interm / f"sc{k}.pcal"
        out_vis = interm / f"selfcal_{k}.ms"
        tmp_vis = interm / f"selfcal_{k}_applycal.tmp.ms"
        clean_base = interm / f"clean_sc{k}"
        fits_path = interm / f"clean_sc{k}.fits"
        _safe_rmtree_path(caltable)
        _safe_rmtree_path(out_vis)
        _safe_rmtree_path(tmp_vis)
        _safe_rmtree_glob(interm, f"clean_sc{k}.*")

        field_id = self._derived["resolved_field_id"]
        refant = self._derived["refant"]
        prev_vis = self._derived["good_vis"]
        prev_base = self._derived["last_image_base"]
        prev_nterms = self._derived["last_nterms"]
        have_model = self._derived.get("good_vis_model_from") == prev_base
        copied = False
        gate_info: Dict[str, Any] = {}
        try:
            if have_model:
                read_vis = prev_vis
                model_info = {"model_image": _model_images(prev_base, prev_nterms),
                              "nterms": prev_nterms,
                              "reffreq_hz": self._derived.get("good_vis_model_reffreq_hz")}
            else:
                _copy_ms(prev_vis, tmp_vis)
                copied = True
                model_info = self._predict_model(tmp_vis, prev_base, prev_nterms)
                read_vis = tmp_vis
            state = self._read_state(read_vis)
            self._record_pending_g3(state)
            names = state.antenna_names
            refant_idx = names.index(refant) if refant in names else None
            rec = _solint.evaluate_stage(k, state, sc["chain"], sc["geom"],
                                         combine_spw=sc["combine_spw"], refant=refant_idx,
                                         minblperant=cfg.gaincal.minblperant)
            selection = _solint.strip_internal(rec)
            selection["data_read_from"] = str(read_vis)
            if cfg.verbose:
                self._print_selection(selection)
            if rec["decision"] != "apply":
                self._record_unapplied_stage(k, selection)
                return rec["decision"], selection["reason"]

            cand = rec["_"]["chosen"]
            combine = ",".join(x for x in (cand["time_combine"],
                                           "spw" if sc["combine_spw"] else "") if x)
            # the stage's time interpolation: the configured (linear) one, or its nearest
            # form when G2 chose a nearest variant (ASEBI SPEC 10.2 item 6, D34 F2)
            stage_interp = _solint.applycal_interp_string(cfg.gaincal.applycal_interp,
                                                          cand.get("interp", "linear"))
            label = f"phase_{cand['label']}"
            step = SelfcalStep("p", cand["gaincal_solint"], label=label)
            if not copied:
                _copy_ms(prev_vis, tmp_vis)
                copied = True
            n_solutions, gaincal_ok = self._run_gaincal(
                input_vis=str(tmp_vis), caltable=str(caltable), solint=cand["gaincal_solint"],
                step=step, field_id=field_id, refant=refant, combine=combine)
            if gaincal_ok:
                gate_info = self._gate_caltable(str(caltable), state, cand, rec["_"]["s_rows"])
                n_solutions -= int(gate_info.get("n_flagged", 0))
            if not gaincal_ok or n_solutions <= 0:
                selection["decision"] = "stop"
                selection["reason"] = (f"gaincal returned no usable solution for {cand['label']} "
                                       f"(minsnr={cfg.gaincal.minsnr:g} and the G1 gate)")
                self._record_unapplied_stage(k, selection)
                return "stop", selection["reason"]
            self._plot_gain_table(str(caltable), k, step)
            # the spwmap from the spw labels gaincal wrote (ASEBI SPEC 9 item 3, D38 d)
            maps = self._data_spwmap(state, str(caltable))
            spwmap = maps["global"] if maps else None
            per_obs = (maps["per_observation"] if maps and maps["global"] is None else None)
            sc["spwmap"] = spwmap
            sc["spwmap_record"] = maps
            if maps is not None:
                selection["spwmap"] = maps
            self._applycal(tmp_vis, caltable, spwmap=spwmap, interp=stage_interp,
                           per_observation=per_obs)
            casatasks.split(vis=str(tmp_vis), outputvis=str(out_vis), datacolumn="corrected")
        finally:
            if copied:
                _safe_rmtree_path(tmp_vis)

        solution_stats = self._caltable_solution_stats(str(caltable))
        threshold_jy = self._loop_threshold_jy(self._derived["last_rms"])
        usemask_kwargs = self._build_usemask_kwargs(
            existing_mask=self._derived.get("initial_mask_path"))
        nterms = self._tclean(out_vis, clean_base, niter=cfg.imaging.niter,
                              threshold_jy=threshold_jy, mask_kwargs=usemask_kwargs)
        own_model = self._predict_model(out_vis, clean_base, nterms)
        stats = self._export_and_stats(clean_base, nterms, fits_path)
        is_anomaly, anomaly_reason = self._detect_iter_anomaly(stats, n_solutions)

        g = cfg.gaincal
        solve = build_solve_info(
            table=str(caltable), solint=cand["gaincal_solint"],
            solint_s=None if cand["kind"] == "eb" else float(cand["T_s"]),
            combine=combine, refant=refant, field=str(field_id), calmode="p", gaintype=g.gaintype, minsnr=g.minsnr,
            minblperant=g.minblperant, solnorm=False, interp=stage_interp,
            applymode=g.applymode, calwt=g.calwt,
            spwmap=spwmap if maps is not None else g.spwmap,
            spwmap_per_observation=maps["per_observation"] if maps is not None else None,
            model_image=model_info["model_image"], model_nterms=model_info["nterms"],
            model_reffreq_hz=model_info.get("reffreq_hz"), solved_on=str(prev_vis),
            stage_ms=str(out_vis), n_solutions=n_solutions,
            flagged_solution_fraction=solution_stats)
        selection["g1_gate_applied"] = gate_info
        selection["same_data_diagnostics"] = {
            "peak_jy_beam": stats["peak_jy_beam"], "rms_jy_beam": stats["rms_jy_beam"],
            "dynamic_range": stats["dynamic_range"],
            "note": "measured on the data the gains were solved from; never used for decisions",
        }
        metric = {
            "iteration": k, "label": label, "calmode": "p", "solint": cand["label"],
            "solint_s": solve["solint_s"], "peak_jy_beam": stats["peak_jy_beam"],
            "rms_jy_beam": stats["rms_jy_beam"], "snr": stats["snr"],
            "dynamic_range": stats["dynamic_range"], "bmaj_arcsec": stats["bmaj_arcsec"],
            "bmin_arcsec": stats["bmin_arcsec"], "threshold_jy": threshold_jy,
            "n_gaincal_solutions": n_solutions,
            "status": "anomaly" if is_anomaly else "ok", "anomaly_reason": anomaly_reason,
            "image_path": _image_name(clean_base, nterms), "image_base": str(clean_base),
            "fits_path": str(fits_path), "caltable": str(caltable), "selfcal_ms": str(out_vis),
        }
        self.metrics.append(metric)
        self.justification["iterations"].append(metric)
        self._add_stage({
            "stage": k, "label": label, "status": metric["status"],
            "data_ms": str(out_vis), "stage_ms": str(out_vis),
            "stage_ms_columns": "DATA + MODEL (model of this stage's image)",
            "image_base": str(clean_base), "image": _image_name(clean_base, nterms),
            "nterms": nterms, "model_image": own_model["model_image"],
            "threshold_jy": threshold_jy, "threshold_reason": self._threshold_reason(),
            "solve": solve, "selection": selection,
        })
        sc["chain"].append(_solint.StageSolve(k, cand, state, rec["_"]["s_rows"],
                                              float(rec["_"]["chosen_corr"])))
        sc["pending_g3"] = {"stage": k, "before": _solint.g3_short_baseline_ratio(state),
                            "s": float(np.median(rec["_"]["s_rows"])) if state.n_rows else 1.0,
                            "vis": str(out_vis)}
        sc["best_iteration"] = len(self.metrics) - 1
        self._derived["good_vis"] = str(out_vis)
        self._derived["good_vis_model_from"] = str(clean_base)
        self._derived["good_vis_model_reffreq_hz"] = own_model.get("reffreq_hz")
        self._derived["last_image_base"] = str(clean_base)
        self._derived["last_nterms"] = nterms
        self._derived["last_dr"] = stats["dynamic_range"]
        self._derived["last_rms"] = stats["rms_jy_beam"]
        if is_anomaly:
            warnings.warn(f"[AJISAI] stage {k} ({label}) anomaly detected: {anomaly_reason}. "
                          f"Recorded only; see justification.json.")
        if cfg.verbose:
            print(f"[AJISAI]   -> stage {k} applied ({cand['label']}). "
                  f"DR={stats['dynamic_range']:.1f} (diagnostic), "
                  f"RMS={stats['rms_jy_beam'] * 1e6:.2f} uJy/bm")
        return "apply", selection["reason"]

    def _gate_caltable(self, caltable: str, state, cand: Dict[str, Any], s_rows) -> Dict[str, Any]:
        """G1 per antenna and solution: flag table rows whose predicted SNR is below 5.

        ASEBI SPEC v0.3 10.2 item 4: the flagged solutions are not applied
        (applymode 'calonly'), so those antennas keep the previous stage's
        correction. A direct edit of the table AJISAI just wrote; recorded.
        """
        gt = _solint.gate_table(state, self._sc["geom"], cand, s_rows, self._sc["combine_spw"])
        tb = casatools.table()
        tb.open(caltable, nomodify=False)
        try:
            if tb.nrows() == 0:
                return {"n_rows": 0, "n_flagged": 0}
            flag = tb.getcol("FLAG")
            res = _solint.caltable_gate(gt, tb.getcol("TIME"), tb.getcol("OBSERVATION_ID"),
                                        tb.getcol("SPECTRAL_WINDOW_ID"), tb.getcol("ANTENNA1"))
            ant = tb.getcol("ANTENNA1")
            already = flag.reshape(-1, flag.shape[-1]).all(axis=0)
            new = res["flag"] & ~already
            if new.any():
                flag[:, :, new] = True
                tb.putcol("FLAG", flag)
        finally:
            tb.close()
        names = state.antenna_names
        per_ant: Dict[str, int] = {}
        for a in ant[new]:
            nm = names[int(a)] if 0 <= int(a) < len(names) else str(int(a))
            per_ant[nm] = per_ant.get(nm, 0) + 1
        return {
            "table": caltable, "n_rows": res["n_rows"], "n_flagged": int(new.sum()),
            "unmatched": res["unmatched"], "per_antenna": per_ant,
            "rule": (f"solutions with a predicted SNR below {_solint.G1_SNR_MIN:g} are flagged; "
                     f"with applymode='{self.cfg.gaincal.applymode}' those antennas keep the "
                     f"previous stage's correction"),
        }

    def _record_unapplied_stage(self, k: int, selection: Dict[str, Any]) -> None:
        """A stage that was evaluated but not applied (G1, G2 or no solution)."""
        chosen = selection.get("chosen")
        label = f"phase_{chosen}" if chosen else "phase"
        reason = selection.get("reason", "")
        status = "rejected"
        rec = {
            "iteration": k, "label": label, "calmode": "p", "solint": chosen or "-",
            "solint_s": None, "peak_jy_beam": None, "rms_jy_beam": None, "snr": None,
            "dynamic_range": None, "bmaj_arcsec": None, "bmin_arcsec": None,
            "threshold_jy": None, "n_gaincal_solutions": None, "status": status,
            "anomaly_reason": reason, "image_path": None, "image_base": None,
            "fits_path": None, "caltable": None, "selfcal_ms": None,
        }
        self.metrics.append(rec)
        self.justification["iterations"].append(rec)
        self._add_stage({
            "stage": k, "label": label, "status": status, "data_ms": None, "stage_ms": None,
            "image_base": None, "image": None, "model_image": None, "solve": None,
            "reason": reason, "selection": selection,
        })
        if self.cfg.verbose:
            print(f"[AJISAI]   -> stage {k} not applied: {reason}")

    def _record_pending_g3(self, state) -> None:
        """G3 of the last accepted stage: short-baseline data/model ratio before vs after."""
        pend = self._sc.get("pending_g3")
        if not pend:
            return
        after = _solint.g3_short_baseline_ratio(state)
        cmp = _solint.g3_compare(pend["before"], after, pend["s"])
        st = self._stage(pend["stage"])
        if st is not None and st.get("selection") is not None:
            st["selection"]["g3"] = {
                "per_observation": {str(o): v for o, v in cmp.items()},
                "rule": ("data/model flux ratio of the shortest 20% baselines before and after "
                         "the stage; > max(3 sigma, 1%) would reject an amplitude stage and is "
                         "recorded only for phase stages"),
            }
        self._sc["pending_g3"] = None

    def _finish_g3(self) -> None:
        """Read the last accepted stage MS once more for its G3 record, if still pending."""
        pend = self._sc.get("pending_g3")
        if not pend:
            return
        try:
            state = self._read_state(pend["vis"])
        except Exception as e:  # the record is informational; never fail the run on it
            warnings.warn(f"[AJISAI] G3 record of stage {pend['stage']} skipped: {e}")
            self._sc["pending_g3"] = None
            return
        self._record_pending_g3(state)

    def _print_selection(self, sel: Dict[str, Any]) -> None:
        meas = sel.get("measured", {})
        r = meas.get("r", {})
        fit = meas.get("D_fit", {})
        print(f"[AJISAI] --- stage {sel['stage']} (mode='data') ---")
        print(f"[AJISAI]   r median {r.get('median', float('nan')):.3g} s^-1/2 "
              f"(p10 {r.get('p10', float('nan')):.3g}); D_eff fit: "
              + (f"tau0={fit['tau0_s']:.3g} s, beta={fit['beta']:.2f}" if fit.get("significant")
                 else "no significant structure")
              + f"; G2 split: {meas.get('g2_split')}")
        for c in sel.get("candidates", []):
            g2 = c.get("g2") or {}
            line = (f"[AJISAI]   {c['label']:>8s}: G1 {'pass' if c['g1']['pass'] else 'fail'} "
                    f"(SNR {c['g1']['snr_median']:.1f})")
            if "V_pred" in c:
                line += f", V_pred {c['V_pred']:.4f}"
            if g2:
                line += (f", dV_held {g2['dV_raw']:+.4f} +- {g2['sigma']:.4f}, "
                         f"corrected improvement {g2['improvement_corr']:+.4f}")
            print(line)
        ip = sel.get("interpolation") or {}
        if ip.get("computed"):
            print(f"[AJISAI]   adjacent solutions at {ip['candidate']}: 2 D_eff = "
                  f"{ip['two_D_eff']:.3f} rad^2 (raw {ip['two_D_eff_raw']:.3f}) -> "
                  + ("nearest variants evaluated" if ip.get("trigger") else "linear only"))
        print(f"[AJISAI]   rule A {sel.get('rule_A')}, rule B {sel.get('rule_B')} -> "
              f"{sel.get('chosen')}: {sel.get('decision')} ({sel.get('reason')})")

    def selfcal_summary(self) -> Dict[str, Any]:
        """Compact record of the self-calibration for a calling pipeline (e.g. ASEBI).

        Per stage: status, label, decision record (``selection``), gain table
        and how it was applied (``solve``), stage MS, image and model; plus
        the best stage, the final DATA-only MS and the image of the best stage.
        """
        best = self.justification.get("best") or {}
        stages = []
        for st in self.stages:
            stages.append({k: st.get(k) for k in ("stage", "label", "status", "stage_ms", "image",
                                                   "model_image", "solve", "selection", "reason")})
        best_stage = None
        if best.get("iteration") is not None and 0 <= int(best["iteration"]) < len(self.metrics):
            best_stage = int(self.metrics[int(best["iteration"])]["iteration"])
        st = self._stage(best_stage) if best_stage is not None else None
        return {
            "mode": self.cfg.schedule.mode,
            "best_stage": best_stage,
            "best_image": self.best_image,
            "best_model": (st or {}).get("model_image"),
            "final_ms": self.final_ms,
            "final_image": self.final_image,
            "selfcal": self.justification.get("selfcal"),
            "stages": stages,
        }

    # ----------------------------------------------------------------------
    # Helper methods for iteration
    # ----------------------------------------------------------------------
    def _add_stage(self, stage: Dict[str, Any]) -> None:
        """Append a per-stage record (also to justification.json)."""
        self.stages.append(stage)
        self.justification["stages"].append(stage)

    def _stage(self, index: int) -> Optional[Dict[str, Any]]:
        for st in self.stages:
            if st["stage"] == index:
                return st
        return None

    def _applycal(self, vis, caltable, spwmap=None, interp: Optional[str] = None,
                  per_observation: Optional[Dict[int, List[int]]] = None) -> None:
        """applycal of one self-cal table on an AJISAI-owned temporary MS.

        ``spwmap`` overrides ``GainCalConfig.spwmap`` (mode="data" derives it
        from the table's spw labels when spws are combined); ``interp``
        overrides ``GainCalConfig.applycal_interp`` (mode="data": the stage's
        own interpolation, nearest when G2 chose it; ASEBI SPEC 10.2 item 6,
        D34 F2). ``per_observation`` (observation -> spwmap): one applycal per
        observation, selected with ``observation=``, when no single spwmap
        serves every observation (ASEBI SPEC 9 item 3, D38 d); the rows of an
        observation without solutions are left uncorrected (calonly).
        """
        g = self.cfg.gaincal
        kw = dict(vis=str(vis), gaintable=[str(caltable)], calwt=g.calwt,
                  interp=g.applycal_interp if interp is None else interp, applymode=g.applymode)
        if per_observation:
            for o in sorted(per_observation):
                casatasks.applycal(observation=str(int(o)),
                                   spwmap=[[int(x) for x in per_observation[o]]], **kw)
            return
        spwmap = [int(x) for x in (g.spwmap if spwmap is None else spwmap)]
        casatasks.applycal(spwmap=[spwmap] if spwmap else [], **kw)

    @staticmethod
    def _caltable_solution_stats(caltable: str) -> List[Dict[str, Any]]:
        """Flagged-solution fraction per observation and spw of a gain table."""
        tb = casatools.table()
        tb.open(caltable)
        try:
            if tb.nrows() == 0:
                return []
            obs = tb.getcol("OBSERVATION_ID")
            spw = tb.getcol("SPECTRAL_WINDOW_ID")
            flag = tb.getcol("FLAG")
        finally:
            tb.close()
        return flagged_solution_fractions(obs, spw, flag)

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
        combine: Optional[str] = None,
    ) -> Tuple[int, bool]:
        """Run gaincal and return (n_solutions, success).

        ``combine`` overrides ``GainCalConfig.combine`` (mode="data" chooses
        the time combination per stage).
        """
        cfg = self.cfg
        try:
            casatasks.gaincal(
                vis=input_vis,
                caltable=caltable,
                spw="*",
                field=field_id,
                selectdata=True,
                combine=cfg.gaincal.combine if combine is None else combine,
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
        if casaplotms is None:
            if not self._derived.get("warned_no_casaplotms"):
                warnings.warn("[AJISAI] casaplotms is not importable; gain-table plots "
                              "are skipped")
                self._derived["warned_no_casaplotms"] = True
            return
        diag_dir = self.workdir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        try:
            self._plotms_gain_table(caltable, idx, step, diag_dir)
        except Exception as e:  # a diagnostic plot must not fail the run (e.g. no DISPLAY)
            if not self._derived.get("warned_plotms_failed"):
                warnings.warn(f"[AJISAI] gain-table plots are skipped: plotms failed ({e})")
                self._derived["warned_plotms_failed"] = True

    def _plotms_gain_table(self, caltable: str, idx: int, step: SelfcalStep, diag_dir) -> None:
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

    # ----------------------------------------------------------------------
    # tclean, model prediction and their consistency checks
    # ----------------------------------------------------------------------
    def _tclean(
        self,
        vis,
        imagename,
        *,
        niter: int,
        threshold_jy: float,
        mask_kwargs: Optional[Dict[str, Any]] = None,
        savemodel: str = "none",
        dirty: bool = False,
        check: bool = True,
        **extra,
    ) -> int:
        """Every tclean of a run goes through here.

        Uses the run's cell, imsize, weighting, gridder, reffreq and the
        deconvolver/nterms/scales of ``ImagingConfig`` (``mtmfs`` when
        ``nterms > 1``); ``dirty=True`` makes a ``hogbom`` niter=0 image.
        ``savemodel`` defaults to ``'none'``: tclean never writes to the MS
        (models are predicted explicitly by :meth:`_predict_model`). After
        the call the image is checked (:meth:`_check_imaging_consistency`).
        Returns the number of Taylor terms of the products (1 or nterms).
        """
        cfg = self.cfg
        img = cfg.imaging
        kw: Dict[str, Any] = dict(
            vis=str(vis),
            imagename=str(imagename),
            datacolumn="data",
            specmode="mfs",
            niter=int(niter),
            gain=img.gain,
            threshold=f"{threshold_jy}Jy",
            gridder=img.gridder,
            cyclefactor=img.cyclefactor,
            imsize=list(self._derived["imsize"]),
            cell=self._derived["cell"],
            weighting=img.weighting,
            robust=img.robust,
            uvtaper=list(img.uvtaper),
            pbcor=img.pbcor,
            savemodel=savemodel,
            parallel=img.parallel,
            field=self._derived["resolved_field_id"],
            interactive=False,
        )
        if img.reffreq:
            kw["reffreq"] = img.reffreq
        if dirty:
            kw["deconvolver"] = "hogbom"
        else:
            kw.update(_deconvolver_kwargs(img.deconvolver, img.nterms,
                                          self._derived.get("scales")))
        kw.update(mask_kwargs or {})
        kw.update(extra)
        if kw.get("interactive") and cfg.caller == "asebi":
            raise ValueError("tclean(interactive=True) is refused when caller='asebi' "
                             "(viewer-free operation)")
        casatasks.tclean(**kw)
        nterms = int(kw.get("nterms", 1)) if kw.get("deconvolver") == "mtmfs" else 1
        if check:
            self._check_imaging_consistency(imagename, nterms)
        return nterms

    def _check_clean_model(self, imagebase, nterms: int, threshold_jy: float, label: str) -> None:
        """A CLEAN whose model feeds self-calibration must not leave an empty model.

        Run right after the tclean, before anything uses the model; raises
        RuntimeError with the reason (:func:`empty_model_problem`) and records
        the check.
        """
        models = [_read_image_plane(p) for p in _model_images(imagebase, nterms)]
        n_model = int(sum(np.count_nonzero(np.isfinite(m) & (m != 0.0)) for m in models))
        problem = None
        if n_model == 0:
            mask_path = _image_name(imagebase, nterms, "mask")
            problem = empty_model_problem(
                models, _read_image_plane(_image_name(imagebase, nterms, "residual")),
                _read_image_plane(mask_path) if os.path.isdir(mask_path) else None,
                threshold_jy, self._derived.get("scales"), _casa_log_path())
        self.justification["checks"].append({
            "kind": "clean_model",
            "label": label,
            "image_base": str(imagebase),
            "n_model_pixels": n_model,
            "threshold_jy": float(threshold_jy),
            "ok": problem is None,
            "problems": [problem] if problem else [],
        })
        if problem:
            raise RuntimeError(f"[AJISAI] {label} of {imagebase}: {problem}")

    def _check_imaging_consistency(self, imagebase, nterms: int) -> None:
        """Post-tclean check: frequency axis covers the data, sumwt > 0.

        Raises RuntimeError on failure (an image made from a subset of the
        channels, or from no data, must never feed a self-cal solve).
        """
        image = _image_name(imagebase, nterms, "image")
        sumwt_image = _image_name(imagebase, nterms, "sumwt")
        ref_hz, width_hz = _image_spectral_axis(image)
        sumwt = float(np.nansum(_read_image_plane(sumwt_image)))
        problems = imaging_consistency_problems(
            ref_hz, width_hz, sumwt,
            self._derived.get("data_fmin_hz"), self._derived.get("data_fmax_hz"),
        )
        self.justification["checks"].append({
            "kind": "tclean",
            "image": image,
            "ref_freq_hz": ref_hz,
            "width_hz": width_hz,
            "data_fmin_hz": self._derived.get("data_fmin_hz"),
            "data_fmax_hz": self._derived.get("data_fmax_hz"),
            "sumwt": sumwt,
            "ok": not problems,
            "problems": problems,
        })
        if problems:
            raise RuntimeError(f"[AJISAI] imaging consistency check failed for {image}: "
                               + "; ".join(problems))

    def _predict_model(self, vis, imagebase, nterms: int) -> Dict[str, Any]:
        """Write the model of ``imagebase`` into MODEL_DATA of an AJISAI-owned MS.

        Standard gridder: ``ft`` of the Taylor model images with ``nterms`` and
        the model's reference frequency. Other gridders (e.g. mosaic, which
        needs the primary beam that ``ft`` does not apply): tclean prediction
        from the existing model (``niter=0``, ``calcres=calcpsf=False``,
        ``savemodel='modelcolumn'``). Then :meth:`_check_model_column`.
        """
        cfg = self.cfg
        models = _model_images(imagebase, nterms)
        reffreq_hz = _image_spectral_axis(models[0])[0] if nterms > 1 else None
        cover: Optional[Dict[str, Any]] = None
        if cfg.imaging.gridder == "standard":
            ft_models, cover = _models_covering_data(models, vis)
            try:
                casatasks.ft(
                    vis=str(vis),
                    field=self._derived["resolved_field_id"],
                    model=ft_models if nterms > 1 else ft_models[0],
                    nterms=int(nterms),
                    reffreq=f"{reffreq_hz}Hz" if reffreq_hz else "",
                    usescratch=True,
                )
            finally:
                for m in ft_models:
                    if m not in models:
                        _safe_rmtree_path(m)
            method = "ft"
        else:
            self._tclean(vis, imagebase, niter=0, threshold_jy=0.0, savemodel="modelcolumn",
                         calcpsf=False, calcres=False, check=False)
            method = "tclean(savemodel='modelcolumn')"
        problems = self._check_model_column(vis)
        self.justification["checks"].append({
            "kind": "model_prediction",
            "vis": str(vis),
            "method": method,
            "model_image": models,
            "nterms": int(nterms),
            "reffreq_hz": reffreq_hz,
            "spectral_cover": cover,
            "ok": not problems,
            "problems": problems,
        })
        if problems:
            raise RuntimeError(f"[AJISAI] model prediction check failed for {vis}: "
                               + "; ".join(problems))
        return {"model_image": models, "nterms": int(nterms), "reffreq_hz": reffreq_hz,
                "method": method}

    def _check_model_column(self, vis) -> List[str]:
        """MODEL_DATA must be non-zero on every unflagged channel of the field."""
        fid = int(self._derived["resolved_field_id"])
        tb = casatools.table()
        tb.open(str(vis) + "/DATA_DESCRIPTION")
        try:
            dd_spw = tb.getcol("SPECTRAL_WINDOW_ID")
        finally:
            tb.close()
        problems: List[str] = []
        tb.open(str(vis))
        try:
            if "MODEL_DATA" not in tb.colnames():
                return ["no MODEL_DATA column after the model prediction"]
            any_model = False
            for dd in range(len(dd_spw)):
                sub = tb.query(f"DATA_DESC_ID=={dd} && FIELD_ID=={fid}",
                               columns="MODEL_DATA,FLAG")
                try:
                    nrow = sub.nrows()
                    if nrow == 0:
                        continue
                    npol, nchan = np.asarray(sub.getcell("FLAG", 0)).shape
                    chunk = max(1, _CHUNK_SAMPLES // max(1, npol * nchan))
                    absmax = np.zeros(nchan)
                    count = np.zeros(nchan, dtype=np.int64)
                    for start in range(0, nrow, chunk):
                        nr = min(chunk, nrow - start)
                        amp = np.abs(sub.getcol("MODEL_DATA", start, nr))
                        unflagged = ~np.asarray(sub.getcol("FLAG", start, nr), dtype=bool)
                        amp = np.where(np.isfinite(amp), amp, 0.0)
                        absmax = np.maximum(absmax, np.where(unflagged, amp, 0.0).max(axis=(0, 2)))
                        count += unflagged.sum(axis=(0, 2))
                    any_model = any_model or bool((absmax > 0).any())
                    bad = model_zero_channels(absmax, count)
                    if bad:
                        problems.append(f"spw {int(dd_spw[dd])} (data description {dd}): "
                                        f"MODEL_DATA = 0 on unflagged channels {bad}")
                finally:
                    sub.close()
        finally:
            tb.close()
        if problems and not any_model:
            problems.insert(0, "the model is empty (the CLEAN left no component: no emission "
                               "above the threshold, or a minor cycle that did not run; see the "
                               "CASA log); self-calibration needs a model")
        return problems

    def _export_and_stats(self, imagebase, nterms: int, fits_path) -> Dict[str, Any]:
        """Export the (tt0) restored image to FITS and compute its statistics."""
        fits_path = Path(fits_path)
        if fits_path.exists():
            fits_path.unlink()
        casatasks.exportfits(imagename=_image_name(imagebase, nterms),
                             fitsimage=str(fits_path), history=False)
        return self.compute_image_stats(str(fits_path))

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
        # A user mask (CRTF region file or mask image) is passed as it is to
        # every tclean: initial CLEAN, each iteration and the final image.
        if mode == "user":
            if not cfg.imaging.user_mask:
                raise ValueError("mask_mode='user' but user_mask is None")
            return {"mask": cfg.imaging.user_mask, "usemask": "user",
                    "interactive": False}
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
            if cfg.caller == "asebi":
                raise ValueError("mask_mode='interactive' is refused when caller='asebi' "
                                 "(viewer-free operation)")
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
        self._add_stage({
            "stage": idx,
            "label": step.label,
            "status": "skipped",
            "data_ms": None,
            "stage_ms": None,
            "image_base": None,
            "image": None,
            "model_image": None,
            "solve": None,
            "reason": reason,
        })
        warnings.warn(f"[AJISAI] iter {idx} ({step.label}) SKIPPED: {reason}. "
                      f"Pipeline continues to next iteration.")

    def _select_best(self) -> None:
        """Pick best iteration and set best_image / best_fits / best_metric_value.

        ``schedule.mode="data"``: the last accepted stage (the image statistics
        are diagnostics only; ASEBI SPEC v0.3 10.2 item 13). ``mode="fixed"``:
        :meth:`select_best_iteration` (``quality_metric``).
        """
        if not self.metrics:
            warnings.warn("_select_best called with empty metrics list; nothing to choose.")
            return
        data_mode = self.cfg.schedule.mode == "data"
        if data_mode:
            best_i = int(getattr(self, "_sc", {}).get("best_iteration", 0))
            rule = ("last accepted stage (schedule.mode='data'); peak, RMS and dynamic range "
                    "are recorded as diagnostics only")
        else:
            best_i = self.select_best_iteration()
            rule = f"highest {self.cfg.quality_metric} over all iterations (schedule.mode='fixed')"
        best_metric = self.metrics[best_i]
        self.best_image = best_metric.get("image_path")
        self.best_fits = best_metric.get("fits_path")
        metric_key = "dynamic_range" if self.cfg.quality_metric == "dynamic_range" else "snr"
        self.best_metric_value = best_metric.get(metric_key)
        self.justification["best"] = {
            "iteration": best_i,
            "rule": rule,
            "metric_key": metric_key,
            "value": self.best_metric_value,
            "image": self.best_image,
            "all_iterations": self.metrics,
        }
        if self.cfg.verbose:
            print(f"[AJISAI] best iteration: #{best_i}, "
                  f"{metric_key} = {self.best_metric_value}")

    # ----------------------------------------------------------------------
    # Final products: DATA-only MS of the selected stage, final deep image
    # ----------------------------------------------------------------------
    def _selected_stage(self) -> Optional[Dict[str, Any]]:
        """The stage record of the iteration chosen by :meth:`select_best_iteration`."""
        best = (self.justification.get("best") or {}).get("iteration")
        if best is None or not 0 <= int(best) < len(self.metrics):
            return None
        return self._stage(int(self.metrics[int(best)]["iteration"]))

    def _write_final_ms(self) -> None:
        """Split the selected stage's DATA into ``<results_dir>/final_selfcal.ms``.

        The product holds the self-calibrated visibilities in DATA only (no
        MODEL/CORRECTED); for stage 0 it is a DATA-only copy of the working MS.
        """
        assert self.workdir is not None
        stage = self._selected_stage()
        if stage is None or not stage.get("data_ms"):
            self.justification["final_ms"] = {
                "path": None,
                "reason": "no stage with an image was selected; no final MS written",
            }
            return
        src = stage["data_ms"]
        out = self.workdir / "final_selfcal.ms"
        _safe_rmtree_path(out)
        casatasks.split(vis=str(src), outputvis=str(out), datacolumn="data")
        self.final_ms = str(out)
        self.justification["final_ms"] = {
            "path": str(out),
            "stage": stage["stage"],
            "source_ms": str(src),
            "columns": "DATA",
            "reason": ("DATA column of the selected stage (self-calibrated visibilities), "
                       "split without MODEL/CORRECTED"),
        }
        if self.cfg.verbose:
            print(f"[AJISAI] final MS              : {out} (stage {stage['stage']}, DATA only)")

    def _make_final_image(self) -> None:
        """One deeper CLEAN of the selected stage's data, with a flux stop rule.

        Made on ``final_selfcal.ms`` with the run's imaging rules (cell,
        imsize, nterms, reffreq, scales, mask). The threshold steps through
        ``ImagingConfig.final_depth`` x sigma_th (default 3, 2, 1, 0.5),
        continuing the same CLEAN (no new PSF). After each step, inside the
        aperture: ``F_corr = F_model + F_unc`` with ``F_unc = sum(residual) /
        Omega_R``; stop when ``|F_unc| <= max(3 sigma_F, 0.01 F_corr)`` or
        ``F_corr`` changed by no more than that since the previous step
        (:func:`final_depth_decision`). The restored-flux bias
        ``F_unc (Omega_R/Omega_clean - 1)`` is recorded and warned about
        when it exceeds the same width. The self-cal loop images are never
        CLEANed deeper than ``threshold_nsigma x sigma_th``.
        """
        cfg = self.cfg
        depth = tuple(cfg.imaging.final_depth)
        if not depth or self.final_ms is None:
            self.justification["final_image"] = {
                "made": False,
                "reason": ("ImagingConfig.final_depth is empty" if not depth
                           else "no stage was selected"),
            }
            return
        assert self.workdir is not None
        interm = self.workdir / "intermediate"
        base = interm / "final_clean"
        _safe_rmtree_glob(interm, "final_clean.*")
        sigma = float(self._derived["sigma_th"])
        mask_kwargs = self._build_usemask_kwargs(
            existing_mask=self._derived.get("initial_mask_path"),
        )
        steps: List[Dict[str, Any]] = []
        prev_f_corr: Optional[float] = None
        nterms = 1
        converged = False
        for k, nsig in enumerate(depth):
            thr = float(nsig) * sigma
            restart = {} if k == 0 else {"calcpsf": False, "calcres": False}
            step_mask = mask_kwargs
            if k > 0 and mask_kwargs and mask_kwargs.get("mask") and (Path(str(base) + ".mask")).exists():
                # tclean refuses a restart that supplies a mask while the run's .mask image already
                # exists; the first step wrote the same mask into that image, so reuse it
                step_mask = dict(mask_kwargs, mask="")
            nterms = self._tclean(self.final_ms, base, niter=cfg.imaging.niter,
                                  threshold_jy=thr, mask_kwargs=step_mask, **restart)
            terms = self._final_flux_terms(base, nterms)
            decision = final_depth_decision(terms["f_model_jy"], terms["f_unc_jy"],
                                            terms["sigma_f_jy"], prev_f_corr)
            steps.append({"nsigma": float(nsig), "threshold_jy": thr, **terms, **decision})
            if cfg.verbose:
                print(f"[AJISAI] final image {nsig:g} sigma : F_model={terms['f_model_jy']*1e3:.3f}, "
                      f"F_unc={terms['f_unc_jy']*1e3:.3f}, F_corr={decision['f_corr_jy']*1e3:.3f} mJy "
                      f"-> {'stop' if decision['stop'] else 'continue'} ({decision['reason']})")
            prev_f_corr = decision["f_corr_jy"]
            if decision["stop"]:
                converged = True
                break
        last = steps[-1]
        warnings_list: List[str] = []
        if not converged:
            warnings_list.append(
                f"final image: the aperture flux did not converge at the deepest step "
                f"({depth[-1]:g} sigma_th): {last['reason']}")
        if abs(last["restored_bias_jy"]) > last["width_jy"]:
            warnings_list.append(
                f"final image: the restored image over-counts the residual flux by "
                f"{last['restored_bias_jy'] * 1e3:.3g} mJy (F_unc x (Omega_R/Omega_clean - 1)), "
                f"more than max(3 sigma_F, 1% F_corr) = {last['width_jy'] * 1e3:.3g} mJy; "
                f"use F_corr = {last['f_corr_jy'] * 1e3:.4g} mJy for the flux")
        for w in warnings_list:
            warnings.warn(f"[AJISAI] {w}")

        fits_path = self.workdir / "intermediate" / "final_clean.fits"
        stats = self._export_and_stats(base, nterms, fits_path)
        self.final_image = _image_name(base, nterms)
        self.final_fits = str(fits_path)
        products = {name: _image_name(base, nterms, name)
                    for name in ("image", "model", "residual", "psf", "pb", "mask")}
        if nterms > 1:
            products.update({"image_tt1": _image_name(base, nterms, "image", 1),
                             "alpha": _image_name(base, nterms, "alpha"),
                             "alpha_error": _image_name(base, nterms, "alpha.error")})
        self.justification["final_image"] = {
            "made": True,
            "stage": (self._selected_stage() or {}).get("stage"),
            "vis": self.final_ms,
            "image_base": str(base),
            "image": self.final_image,
            "fits": str(fits_path),
            "products": products,
            "nterms": nterms,
            "sigma_th_jy": sigma,
            "depth_schedule": [float(d) for d in depth],
            "steps": steps,
            "converged": converged,
            "stop_nsigma": last["nsigma"],
            "aperture": self._derived.get("final_aperture"),
            "stats": stats,
            "warnings": warnings_list,
            "rule": ("stop when |F_unc| <= max(3 sigma_F, 0.01 F_corr) or |Delta F_corr| <= "
                     "the same width; F_corr = F_model + sum(residual)/Omega_R in the aperture"),
        }

    def _final_flux_terms(self, imagebase, nterms: int) -> Dict[str, Any]:
        """Aperture flux bookkeeping of the final image after one depth step."""
        model = _read_image_plane(_image_name(imagebase, nterms, "model"))
        residual = _read_image_plane(_image_name(imagebase, nterms, "residual"))
        psf = _read_image_plane(_image_name(imagebase, nterms, "psf"))
        geom = _image_beam_cell(_image_name(imagebase, nterms, "image"))
        aperture = self._final_aperture(imagebase, nterms, residual.shape, geom)
        return aperture_flux_terms(model, residual, psf, aperture, geom["cell_arcsec"],
                                   geom["bmaj_arcsec"], geom["bmin_arcsec"],
                                   float(self._derived["sigma_th"]))

    def _final_aperture(self, imagebase, nterms: int, shape, geom: Dict[str, Any]) -> np.ndarray:
        """Aperture for the final-image flux: explicit radius, CLEAN mask, or 5 beams."""
        cfg = self.cfg
        cell = geom["cell_arcsec"]
        mask_path = _image_name(imagebase, nterms, "mask")
        if cfg.imaging.final_aperture_arcsec is None and os.path.isdir(mask_path):
            region = _read_image_plane(mask_path) > 0.5
            if region.shape == tuple(shape) and region.any():
                self._derived["final_aperture"] = {
                    "kind": "clean_mask", "path": mask_path, "npix": int(region.sum())}
                return region
        if cfg.imaging.final_aperture_arcsec is not None:
            radius_as = float(cfg.imaging.final_aperture_arcsec)
            kind = "final_aperture_arcsec"
        else:
            beam = math.sqrt(geom["bmaj_arcsec"] * geom["bmin_arcsec"])
            radius_as = cfg.rms_exclude_beam_factor * beam
            kind = f"circle of rms_exclude_beam_factor ({cfg.rms_exclude_beam_factor:g}) x beam"
        cx, cy = geom["refpix"]
        ix, iy = np.indices(tuple(shape))
        region = np.hypot(ix - cx, iy - cy) <= radius_as / cell
        self._derived["final_aperture"] = {"kind": kind, "radius_arcsec": radius_as,
                                           "npix": int(region.sum())}
        return region

    def _write_summary(self) -> None:
        """Write metrics CSV, summary and gallery PNGs, and copy the final image to the workdir."""
        if not self.metrics or self.workdir is None:
            return

        # --- metrics.csv ---
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.workdir / "metrics.csv", index=False)

        # --- summary plot (DR, peak, RMS, beam over iterations) ---
        self._plot_summary(df)

        # --- Copy the final image (or, without one, the best iteration's
        #     image) to the top-level workdir for easy access (tt0 for mtmfs) ---
        src_fits = self.final_fits or self.best_fits
        src_image = self.final_image or self.best_image
        if src_fits is not None and os.path.exists(src_fits):
            try:
                final_fits = self.workdir / "final.fits"
                if final_fits.exists():
                    final_fits.unlink()
                shutil.copy2(src_fits, final_fits)
            except Exception as e:
                warnings.warn(f"Could not copy final FITS: {e}")
        if src_image is not None and os.path.exists(src_image):
            try:
                final_image = self.workdir / "final.image"
                if final_image.exists():
                    shutil.rmtree(final_image, ignore_errors=True)
                shutil.copytree(src_image, final_image)
            except Exception as e:
                warnings.warn(f"Could not copy final image: {e}")

        # --- gallery of the CLEAN image of every iteration ---
        try:
            self._plot_images(df)
        except Exception as e:  # a plotting problem must not fail the run
            warnings.warn(f"Could not write selfcal_images.png: {e}")

    def _plot_summary(self, df: pd.DataFrame) -> None:
        """4-panel plot: DR / peak / RMS / beam over iterations (selfcal_summary.png)."""
        if self.workdir is None or df.empty:
            return
        fig = plot_selfcal_summary(
            df,
            best_iteration=(self.justification.get("best") or {}).get("iteration"),
            projname=self._derived.get("projname", "AJISAI"),
            outpath=self.workdir / "selfcal_summary.png",
        )
        plt.close(fig)

    def _plot_images(self, df: pd.DataFrame) -> None:
        """Gallery of the CLEAN image of every iteration (selfcal_images.png)."""
        if self.workdir is None or df.empty:
            return
        panels = panels_from_metrics(df)
        if not any(p["fits"] is not None for p in panels):
            return
        opts = self.cfg.plots
        fig = plot_selfcal_images(
            panels,
            fov_radius_arcsec=opts.images_fov_radius_arcsec,
            fov_beam_factor=opts.images_fov_beam_factor,
            cmap=opts.images_cmap,
            gamma=opts.images_gamma,
            gamma2=opts.images_gamma2,
            center=opts.images_center,
            best_iteration=(self.justification.get("best") or {}).get("iteration"),
            projname=self._derived.get("projname", "AJISAI"),
            outpath=self.workdir / "selfcal_images.png",
        )
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
# Entry point for a calling pipeline's measurement self-calibration (ASEBI)
# ============================================================================

def selfcal_measurement(
    vis: str,
    *,
    results_dir: str,
    field: Optional[str] = None,
    imaging: Optional[ImagingConfig] = None,
    gaincal: Optional[GainCalConfig] = None,
    refant: Optional[str] = None,
    projname: Optional[str] = None,
    verbose: bool = False,
    **options: Any,
) -> "AJISAI":
    """Phase self-calibration of one execution block with the data-driven rules.

    The measurement self-cal of ASEBI (ASEBI SPEC v0.3 section 6 and 10;
    DESIGN_REVIEW P23 item 1: the measurement and the production self-cal use
    the same rules): ``schedule.mode="data"`` with no amplitude stage, run on
    ``vis`` (only read) under ``results_dir``. Defaults that suit a calling
    pipeline: ``caller="asebi"`` (viewer-free), ``relabel_j2000_to_icrs=False``
    (the products stay in the input frame), no banner, and
    ``ImagingConfig(final_depth=())`` when ``imaging`` is not given (the image
    of the last accepted stage, CLEANed to the loop threshold, is the result;
    pass an ``ImagingConfig`` with the caller's cell, imsize, nterms, reffreq
    and CRTF mask). ``refant`` selects the reference antenna by name;
    ``options`` are further ``AJISAIConfig`` fields.

    Returns the finished :class:`AJISAI`; :meth:`AJISAI.selfcal_summary` gives
    the per-stage decisions, gain tables (with their applycal arguments),
    stage MSs, the best stage's image and model and the DATA-only
    ``final_selfcal.ms``.
    """
    kw: Dict[str, Any] = dict(
        vis=vis,
        results_dir=results_dir,
        field=field,
        projname=projname,
        imaging=imaging if imaging is not None else ImagingConfig(final_depth=()),
        gaincal=gaincal if gaincal is not None else GainCalConfig(),
        schedule=SelfcalSchedule(mode="data", amplitude="never"),
        caller="asebi",
        relabel_j2000_to_icrs=False,
        show_banner=False,
        verbose=verbose,
    )
    if refant:
        kw.update(refant_strategy="manual", refant_manual=refant)
    kw.update(options)
    if kw["schedule"].mode != "data":
        raise ValueError("selfcal_measurement uses schedule.mode='data'")
    return AJISAI(AJISAIConfig(**kw)).run()


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


def _casa_log_path() -> Optional[str]:
    """The file the CASA logger writes to (None without CASA)."""
    if not _HAS_CASA:
        return None
    try:
        return str(casatasks.casalog.logfile())
    except Exception:
        return None


def _copy_ms(src, dst) -> None:
    """Exact copy of an MS directory (all columns and subtables, no reindexing)."""
    shutil.copytree(str(src), str(dst), symlinks=True)


def _ms_needs_relabel(msfile: str) -> bool:
    """Read-only check: would :func:`relabel_J2000_to_ICRS` change ``msfile``?

    Reads the per-row frames of the FIELD direction columns and of
    SOURCE.DIRECTION from their MEASINFO keywords (codes mapped through
    TabRefTypes/TabRefCodes).
    """
    _require_casa()
    tb = casatools.table()
    tb.open(msfile + "/FIELD")
    try:
        field_frames = [_column_frames(tb, c) for c in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR")
                        if c in tb.colnames()]
    finally:
        tb.close()
    source_frames: List[str] = []
    if os.path.isdir(msfile + "/SOURCE"):
        tb.open(msfile + "/SOURCE")
        try:
            if "DIRECTION" in tb.colnames():
                source_frames = _column_frames(tb, "DIRECTION")
        finally:
            tb.close()
    return _relabel_needed(field_frames, source_frames)


def _observation_telescopes_and_start(msfile: str) -> Tuple[List[str], Optional[float]]:
    """TELESCOPE_NAME values and the earliest TIME_RANGE start (MJD seconds) of OBSERVATION."""
    _require_casa()
    path = msfile + "/OBSERVATION"
    if not os.path.isdir(path):
        return [], None
    tb = casatools.table()
    tb.open(path)
    try:
        if tb.nrows() == 0:
            return [], None
        names = [str(n) for n in tb.getcol("TELESCOPE_NAME")]
        tr = np.asarray(tb.getcol("TIME_RANGE"), dtype=float)
    finally:
        tb.close()
    starts = tr[0][tr[0] > 0] if tr.ndim == 2 else np.asarray([])
    return names, (float(starts.min()) if starts.size else None)


def _read_image_plane(path) -> np.ndarray:
    """Pixel values of a CASA image with degenerate axes dropped ([x, y] for a plane)."""
    _require_casa()
    ia = casatools.image()
    ia.open(str(path))
    try:
        arr = np.asarray(ia.getchunk(dropdeg=True), dtype=float)
    finally:
        ia.close()
    while arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def _image_spectral_axis(path) -> Tuple[float, float]:
    """(reference frequency, increment) of a CASA image's spectral axis, in Hz."""
    _require_casa()
    ia = casatools.image()
    ia.open(str(path))
    try:
        cs = ia.coordsys()
        try:
            names = list(cs.names())
            units = list(cs.units())
            i = names.index("Frequency")
            ref = float(cs.referencevalue()["numeric"][i])
            inc = float(cs.increment()["numeric"][i])
            scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}.get(units[i], 1.0)
        finally:
            cs.done()
    finally:
        ia.close()
    return ref * scale, inc * scale


def _data_frequency_edges(vis) -> Tuple[float, float]:
    """(lowest, highest) channel-edge frequency (Hz, the MS's own frame) of the spws that hold rows."""
    _require_casa()
    tb = casatools.table()
    tb.open(str(vis))
    try:
        ddids = sorted(set(int(x) for x in np.unique(tb.getcol("DATA_DESC_ID"))))
    finally:
        tb.close()
    tb.open(str(vis) + "/DATA_DESCRIPTION")
    try:
        spws = sorted(set(int(tb.getcell("SPECTRAL_WINDOW_ID", d)) for d in ddids))
    finally:
        tb.close()
    tb.open(str(vis) + "/SPECTRAL_WINDOW")
    try:
        lo, hi = math.inf, -math.inf
        for s in spws:
            f = np.asarray(tb.getcell("CHAN_FREQ", s), dtype=float)
            w = np.abs(np.asarray(tb.getcell("CHAN_WIDTH", s), dtype=float))
            lo = min(lo, float(np.min(f - 0.5 * w)))
            hi = max(hi, float(np.max(f + 0.5 * w)))
    finally:
        tb.close()
    return lo, hi


def _models_covering_data(models: List[str], vis) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Model images for ``ft`` whose spectral range covers every channel of ``vis``.

    When the model's range does not cover the data (see
    :func:`spectral_cover_increment`), temporary copies with only the spectral
    increment widened are returned; the originals are never changed (tclean
    restarts and callers re-solving a stage read them). Returns the paths for
    ``ft`` and a record of the check.
    """
    ref, inc = _image_spectral_axis(models[0])
    lo, hi = _data_frequency_edges(vis)
    new_inc = spectral_cover_increment(ref, inc, lo, hi)
    rec: Dict[str, Any] = {"image_ref_hz": ref, "image_increment_hz": inc,
                           "data_lo_hz": lo, "data_hi_hz": hi, "widened_increment_hz": new_inc}
    if new_inc is None:
        return list(models), rec
    out: List[str] = []
    ia = casatools.image()
    for m in models:
        tmp = str(m) + ".ft_cover"
        _safe_rmtree_path(tmp)
        shutil.copytree(str(m), tmp)
        ia.open(tmp)
        try:
            cs = ia.coordsys()
            try:
                names = list(cs.names())
                i = names.index("Frequency")
                incs = cs.increment()
                scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}.get(list(cs.units())[i], 1.0)
                incs["numeric"][i] = new_inc / scale
                cs.setincrement(incs)
                ia.setcoordsys(cs.torecord())
            finally:
                cs.done()
        finally:
            ia.close()
        out.append(tmp)
    rec["note"] = ("the model's spectral range did not cover the data; ft used copies with the spectral "
                   "increment widened (values unchanged)")
    return out, rec


def models_covering_data(models: List[str], vis) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Public form of :func:`_models_covering_data` for callers that run ``ft`` themselves (ASEBI).

    The caller removes the returned paths that are not among ``models`` after ``ft``.
    """
    return _models_covering_data(list(models), vis)


def _image_beam_cell(path) -> Dict[str, Any]:
    """Restoring beam (arcsec), cell (arcsec) and reference pixel of a CASA image."""
    _require_casa()
    ia = casatools.image()
    qa = casatools.quanta()
    ia.open(str(path))
    try:
        beam = ia.restoringbeam()
        if "major" not in beam and "beams" in beam:      # per-plane beams: take the first
            beam = beam["beams"]["*0"]["*0"]
        cs = ia.coordsys()
        try:
            inc = cs.increment(format="q", type="direction")["quantity"]
            cell = abs(float(qa.convert(inc["*1"], "arcsec")["value"]))
            refpix = [float(x) for x in cs.referencepixel()["numeric"][:2]]
        finally:
            cs.done()
    finally:
        ia.close()
    return {
        "bmaj_arcsec": float(qa.convert(beam["major"], "arcsec")["value"]),
        "bmin_arcsec": float(qa.convert(beam["minor"], "arcsec")["value"]),
        "cell_arcsec": cell,
        "refpix": refpix,
    }


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
