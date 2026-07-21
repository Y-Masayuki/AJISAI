#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ajisai_ms_utils.py
==================
Measurement-set query utilities for AJISAI.

This module provides native casatools/astropy implementations of the small
subset of analysisUtils (Todd Hunter / NRAO) functions that AJISAI relies on.
By containing these helpers in a single file, AJISAI removes its dependency
on the full 110,000-line analysisUtils.py module, which is no longer actively
maintained and breaks on newer Python versions.

Each function here is documented with:
    - A reference to the corresponding analysisUtils function name
    - Validated test inputs/outputs where applicable

Replaced analysisUtils functions:
    1. ValueMapping              -> get_array_info()
    2. timeOnSource              -> get_on_source_time()
    3. medianFrequencyOfIntent   -> get_median_frequency()
    4. getBaselineStats          -> get_baseline_at_percentile()
    5. pickCellSize              -> pick_cell_imsize()
    6. rad2radec                 -> rad_to_radec()
    7. ICRSToJ2000               -> icrs_to_j2000()

Validation against the original analysisUtils:
    Test MS: V883Ori 232 GHz, 90-percentile baseline = 3293.48 m
    Expected pickCellSize output: [0.0081 arcsec, [4800, 4800]]
    See test_pick_cell_imsize() at the bottom of this file.

Author: Masayuki Yamaguchi (Kyushu Univ./NAOJ) and AJISAI development team
License: MIT
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import casatools  # type: ignore
    import casatasks  # type: ignore
    _HAS_CASA = True
except ImportError:
    casatools = None  # type: ignore
    casatasks = None  # type: ignore
    _HAS_CASA = False


# ============================================================================
# Physical constants
# ============================================================================
SPEED_OF_LIGHT_M_S = 2.99792458e8
ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi  # ~= 206264.806

# ALMA antenna parameters (12 m dish, central obscuration 0.75 m)
ALMA_DISH_DIAMETER_M = 12.0
ALMA_DISH_OBSCURATION_M = 0.75
ALMA_DEFAULT_TAPER_DB = 10.0


# ============================================================================
# Helpers (internal)
# ============================================================================
def _require_casa() -> None:
    """Raise a clear error if casatools/casatasks are unavailable."""
    if not _HAS_CASA:
        raise RuntimeError(
            "ajisai_ms_utils requires casatools/casatasks. "
            "Run from within CASA or install modular CASA via "
            "'pip install casatools casatasks'."
        )


def _round_to_sig_figs(value: float, n_sig_figs: int = 2) -> float:
    """Round to N significant figures (analysisUtils.roundFigures equivalent).

    Example: _round_to_sig_figs(0.008058, 2) -> 0.0081
    """
    if value == 0.0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(abs(value)))
    factor = 10 ** (n_sig_figs - 1 - exponent)
    return round(value * factor) / factor


def _smallest_5_smooth_at_least(n: int) -> int:
    """Return the smallest integer >= n whose only prime factors are 2, 3, 5.

    Used to choose an FFT-friendly image size. Equivalent to
    cleanhelper.getOptimumSize() in CASA.

    Examples:
        _smallest_5_smooth_at_least(4691) -> 4800
        _smallest_5_smooth_at_least(4691) -> 4800  (= 2^6 * 3 * 5^2)
    """
    if n <= 1:
        return 1
    best = None
    # 2^a up to 2^16 = 65536 is enough for any ALMA image
    for a in range(0, 17):
        pa = 2 ** a
        if pa > n * 8:  # quick prune
            break
        for b in range(0, 11):
            pab = pa * (3 ** b)
            if pab > n * 8:
                break
            for c in range(0, 9):
                v = pab * (5 ** c)
                if v >= n and (best is None or v < best):
                    best = v
    return best if best is not None else n


def _baars_taper_factor(taper_db: float = 10.0) -> float:
    """Baars (2007) tapering factor for ALMA primary beam.

    Returns the constant b in FWHM = b * lambda/D.
    For taper_db=10 dB, b ~= 1.137.

    Reference: Baars 2007 book, Eq 4.13 (used by ALMA pipeline since 2014).
    """
    tau = 10 ** (-0.05 * taper_db)
    return 1.269 - 0.566 * tau + 0.534 * (tau ** 2) - 0.208 * (tau ** 3)


def _central_obscuration_factor(
    diameter_m: float = ALMA_DISH_DIAMETER_M,
    obscuration_m: float = ALMA_DISH_OBSCURATION_M,
) -> float:
    """Spline interpolation from Schroeder, Astronomical Optics, Table 10.1.

    Computes the scale factor of the Airy pattern as a function of the
    central obscuration ratio. Returns 1.0 for no obscuration; smaller
    values as obscuration increases.
    """
    epsilon = obscuration_m / diameter_m
    # Knot values from analysisUtils.centralObstructionFactor:
    eps_knots = np.array([0.0, 0.1, 0.2, 0.33, 0.4])
    factor_knots = np.array([1.22, 1.205, 1.167, 1.098, 1.058])
    # Use a simple linear interpolation (scipy spline gives nearly identical
    # values within the standard ALMA range and adds a dependency).
    factor_eps = np.interp(epsilon, eps_knots, factor_knots)
    return float(factor_eps / 1.22)


def _primary_beam_fwhm_arcsec(
    freq_hz: float,
    diameter_m: float = ALMA_DISH_DIAMETER_M,
    obscuration_m: float = ALMA_DISH_OBSCURATION_M,
    taper_db: float = ALMA_DEFAULT_TAPER_DB,
) -> float:
    """ALMA primary beam FWHM in arcsec at the given frequency.

    FWHM = b * lambda/D, where b combines the Baars taper factor and the
    central-obscuration correction.

    At 232.994 GHz on a 12 m dish with 0.75 m obscuration and 10 dB taper,
    returns ~24.9 arcsec.
    """
    wavelength_m = SPEED_OF_LIGHT_M_S / freq_hz
    b = _baars_taper_factor(taper_db) * _central_obscuration_factor(
        diameter_m, obscuration_m
    )
    return b * wavelength_m / diameter_m * ARCSEC_PER_RAD


def _gaussian_offset_for_pb_level(fwhm_arcsec: float, pb_level: float) -> float:
    """Radius (arcsec) at which a Gaussian beam drops to ``pb_level``.

    Useful for sizing images to cover down to a chosen PB attenuation.
    For pb_level=0.2 and FWHM=24.9, returns ~19.0 arcsec.
    """
    if not (0.0 < pb_level <= 1.0):
        raise ValueError(f"pb_level must be in (0, 1]; got {pb_level}")
    # Gaussian: G(r) = exp(-r^2 / (2 sigma^2)), with sigma = FWHM / 2.3548
    sigma = fwhm_arcsec / 2.3548
    return sigma * math.sqrt(-2.0 * math.log(pb_level))


# Fourier-transform conversion constant: a uv-domain Gaussian with FWHM_uv
# in wavelengths corresponds to an image-domain Gaussian with FWHM_image
# (in radians) such that FWHM_image * FWHM_uv = 4 ln 2 / pi ~= 0.8825.
_UVTAPER_FWHM_PRODUCT = 4.0 * math.log(2.0) / math.pi  # ~= 0.8825


def _parse_uvtaper_to_image_fwhm(uvtaper, wavelength_m: float) -> float:
    """Parse a tclean-style ``uvtaper`` value and return the image-domain
    Gaussian FWHM in arcsec.

    For Gaussian uvtaper with FWHM in the uv-domain = FWHM_uv (wavelengths),
    the resulting image-domain Gaussian has FWHM_image (radians) such that
    FWHM_image * FWHM_uv = 4 ln 2 / pi (approx 0.8825).

    Accepted forms (CASA tclean conventions):
        ``[]`` or ``None`` or ``''``       -> returns 0.0 (no taper)
        ``['X arcsec']`` or ``['X"']``     -> image-domain FWHM = X arcsec
        ``['X klambda']``                  -> uv FWHM = X * 1000 wavelengths
        ``['X lambda']``                   -> uv FWHM = X wavelengths
        ``['X m']``                        -> uv FWHM = X meters, converted
                                              to wavelengths via the given
                                              ``wavelength_m``
        ``['X arcsec', 'Y arcsec', 'PA']`` -> elliptical taper; the geometric
                                              mean sqrt(X*Y) is used (a
                                              circular-equivalent approximation)

    Returns
    -------
    float
        Image-domain Gaussian FWHM in arcsec corresponding to the taper.
        Returns 0.0 when no taper is set.
    """
    if uvtaper is None or uvtaper == "" or (isinstance(uvtaper, (list, tuple)) and len(uvtaper) == 0):
        return 0.0
    if isinstance(uvtaper, str):
        uvtaper = [uvtaper]

    parts = []
    for entry in uvtaper:
        s = str(entry).strip().lower()
        # PA / angle entries (degrees) are ignored for the size estimate
        if s.endswith("deg") or s.endswith("rad"):
            continue
        parts.append(s)
    if not parts:
        return 0.0

    image_fwhms_arcsec = []
    for s in parts:
        if s.endswith("arcsec"):
            image_fwhms_arcsec.append(float(s[:-len("arcsec")].strip()))
        elif s.endswith('"'):
            image_fwhms_arcsec.append(float(s[:-1].strip()))
        elif s.endswith("mas"):
            image_fwhms_arcsec.append(float(s[:-len("mas")].strip()) * 1e-3)
        elif s.endswith("klambda"):
            num = float(s[:-len("klambda")].strip())
            fwhm_uv_wavelengths = num * 1e3
            image_fwhms_arcsec.append(
                _UVTAPER_FWHM_PRODUCT / fwhm_uv_wavelengths * ARCSEC_PER_RAD
            )
        elif s.endswith("mlambda"):
            num = float(s[:-len("mlambda")].strip())
            fwhm_uv_wavelengths = num * 1e6
            image_fwhms_arcsec.append(
                _UVTAPER_FWHM_PRODUCT / fwhm_uv_wavelengths * ARCSEC_PER_RAD
            )
        elif s.endswith("lambda"):
            num = float(s[:-len("lambda")].strip())
            image_fwhms_arcsec.append(
                _UVTAPER_FWHM_PRODUCT / num * ARCSEC_PER_RAD
            )
        elif s.endswith("m"):
            num = float(s[:-1].strip())
            fwhm_uv_wavelengths = num / wavelength_m
            image_fwhms_arcsec.append(
                _UVTAPER_FWHM_PRODUCT / fwhm_uv_wavelengths * ARCSEC_PER_RAD
            )
        else:
            raise ValueError(
                f"Unsupported uvtaper format: {s!r}. Use units 'arcsec', "
                f"'mas', 'klambda', 'mlambda', 'lambda', or 'm'."
            )

    # For elliptical tapers (e.g. ['1arcsec', '0.5arcsec', '30deg']) take
    # the geometric mean as a circular-equivalent FWHM.
    if len(image_fwhms_arcsec) >= 2:
        return float(math.sqrt(image_fwhms_arcsec[0] * image_fwhms_arcsec[1]))
    return float(image_fwhms_arcsec[0])


# ============================================================================
# Public API
# ============================================================================
def get_array_info(vis: str) -> Dict[str, Any]:
    """Return basic array properties of an MS (replacement for ValueMapping).

    Returns
    -------
    dict with keys:
        n_antennas              : int
        n_polarizations         : int (number of correlation products in DATA)
        spw_bandwidths_hz       : list of float (TOTAL_BANDWIDTH per SPW)
        spw_total_bandwidth_hz  : float
        spw_mean_freqs_hz       : list of float (mean frequency per SPW)
    """
    _require_casa()
    info: Dict[str, Any] = {}

    tb = casatools.table()
    tb.open(vis + "/ANTENNA")
    try:
        info["n_antennas"] = int(len(tb.getcol("NAME")))
    finally:
        tb.close()

    tb.open(vis + "/POLARIZATION")
    try:
        corr_types = tb.getcol("NUM_CORR")
        info["n_polarizations"] = int(corr_types[0]) if corr_types.size > 0 else 0
    finally:
        tb.close()

    tb.open(vis + "/SPECTRAL_WINDOW")
    try:
        bw = tb.getcol("TOTAL_BANDWIDTH")
        info["spw_bandwidths_hz"] = [float(x) for x in bw]
        info["spw_total_bandwidth_hz"] = float(np.sum(bw))
        # Mean freq per spw
        means = []
        for ispw in range(len(bw)):
            cf = tb.getcell("CHAN_FREQ", ispw)
            means.append(float(np.mean(cf)))
        info["spw_mean_freqs_hz"] = means
    finally:
        tb.close()

    return info


def get_on_source_time(
    vis: str,
    field: Optional[str] = None,
    intent: str = "OBSERVE_TARGET#ON_SOURCE",
) -> float:
    """Total on-source integration time in seconds (replaces timeOnSource).

    Uses casatools.msmetadata.timesforscan() summed over scans with the
    requested intent (and field, if given).
    """
    _require_casa()
    msmd = casatools.msmetadata()
    msmd.open(vis)
    try:
        try:
            scans = list(msmd.scansforintent(intent))
        except Exception:
            scans = list(msmd.scannumbers())
        # Filter by field if given
        if field is not None:
            try:
                fid = int(field)
            except ValueError:
                fid = int(msmd.fieldsforname(field)[0])
            target_scans = set(msmd.scansforfield(fid))
            scans = [s for s in scans if s in target_scans]

        total_sec = 0.0
        for s in scans:
            times = msmd.timesforscan(s)
            if len(times) < 2:
                continue
            # Sum of integration intervals approximated by (last - first).
            # For sub-scan boundaries, this is a slight under-estimate;
            # for ALMA continuum scans the error is well below 1%.
            total_sec += float(times[-1] - times[0])
        return total_sec
    finally:
        msmd.close()


def get_median_frequency(
    vis: str,
    intent: str = "OBSERVE_TARGET#ON_SOURCE",
    ignore_chanavg_spws: bool = True,
) -> float:
    """Median of the mean frequencies of SPWs with the given intent.

    Replaces analysisUtils.medianFrequencyOfIntent. Returns Hz.
    """
    _require_casa()
    msmd = casatools.msmetadata()
    msmd.open(vis)
    try:
        try:
            spws = list(msmd.spwsforintent(intent))
        except Exception:
            spws = list(range(msmd.nspw()))

        # Exclude WVR spws (these have name containing 'WVR' in ALMA MSes)
        try:
            wvr = set(msmd.wvrspws())
        except Exception:
            wvr = set()
        spws = [s for s in spws if s not in wvr]

        if ignore_chanavg_spws:
            # Single-channel spws are usually chan-averaged WVR-derived spws
            multi = [s for s in spws if msmd.nchan(s) > 1]
            if multi:
                spws = multi

        if not spws:
            raise RuntimeError(f"No SPWs found with intent={intent!r}")

        freqs = [float(msmd.meanfreq(s)) for s in spws]
        return float(np.median(freqs))
    finally:
        msmd.close()


def get_antenna_positions(vis: str) -> Dict[str, np.ndarray]:
    """Return ``{antenna_name: np.array([x, y, z])}`` from the ANTENNA table."""
    _require_casa()
    tb = casatools.table()
    tb.open(vis + "/ANTENNA")
    try:
        names = tb.getcol("NAME")
        positions = tb.getcol("POSITION")  # shape (3, nant)
    finally:
        tb.close()
    return {str(names[i]): positions[:, i].copy() for i in range(len(names))}


def get_antenna_flag_stats(
    vis: str,
    field: Optional[str] = None,
) -> Dict[str, float]:
    """Per-antenna fraction of fully flagged visibilities (0.0 - 1.0).

    A visibility row is counted as 'fully flagged' if every channel and
    every polarization product is flagged. The returned fraction is over
    visibilities involving each antenna (as ANTENNA1 or ANTENNA2). Used by
    the hybrid refant strategy in AJISAI to filter out poorly-performing
    antennas before applying the geometric-center criterion.
    """
    _require_casa()
    tb = casatools.table()
    tb.open(vis)
    try:
        if field is not None:
            try:
                fid = int(field)
                tb_sel = tb.query(f"FIELD_ID == {fid}")
            except ValueError:
                ftb = casatools.table()
                ftb.open(vis + "/FIELD")
                names = ftb.getcol("NAME")
                ftb.close()
                matches = [i for i, n in enumerate(names) if n == field]
                if not matches:
                    raise ValueError(f"Field name {field!r} not found in {vis}/FIELD")
                tb_sel = tb.query(f"FIELD_ID == {matches[0]}")
        else:
            tb_sel = tb
        ant1 = tb_sel.getcol("ANTENNA1")
        ant2 = tb_sel.getcol("ANTENNA2")
        flag = tb_sel.getcol("FLAG")  # shape (npol, nchan, nrow)
    finally:
        tb.close()

    fully_flagged = np.all(flag, axis=(0, 1))  # (nrow,)
    atb = casatools.table()
    atb.open(vis + "/ANTENNA")
    ant_names = list(atb.getcol("NAME"))
    atb.close()

    flag_frac: Dict[str, float] = {}
    for i, name in enumerate(ant_names):
        mask_i = (ant1 == i) | (ant2 == i)
        n_total = int(mask_i.sum())
        if n_total == 0:
            flag_frac[name] = 1.0
            continue
        n_flagged = int(fully_flagged[mask_i].sum())
        flag_frac[name] = n_flagged / n_total
    return flag_frac


def get_baseline_lengths(
    vis: str,
    unflagged: bool = False,
    flag_cutoff: float = 0.99,
) -> np.ndarray:
    """Return a 1D ndarray of physical (unprojected) baseline lengths in meters.

    Computed from the ANTENNA POSITION column. Each unique baseline pair
    appears exactly once. With ``unflagged=True``, antennas whose flagged
    visibility fraction exceeds ``flag_cutoff`` are excluded.
    """
    _require_casa()
    positions = get_antenna_positions(vis)
    names = list(positions.keys())
    n_ant = len(names)
    pos = np.column_stack([positions[n] for n in names])  # shape (3, n_ant)

    if unflagged:
        flag_frac = get_antenna_flag_stats(vis)
        keep = np.array([flag_frac.get(n, 1.0) < flag_cutoff for n in names], dtype=bool)
    else:
        keep = np.ones(n_ant, dtype=bool)

    lengths: List[float] = []
    for i in range(n_ant):
        if not keep[i]:
            continue
        for j in range(i + 1, n_ant):
            if not keep[j]:
                continue
            dx = pos[0, i] - pos[0, j]
            dy = pos[1, i] - pos[1, j]
            dz = pos[2, i] - pos[2, j]
            lengths.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return np.array(lengths)


def get_baseline_at_percentile(
    vis: str,
    percentile: float = 90.0,
    unflagged: bool = False,
) -> float:
    """Return the baseline length at a given percentile, in meters.

    Replaces a focused subset of analysisUtils.getBaselineStats.

    percentile : 0..100; e.g. 5 = 5th percentile (shortest), 90 = 90th
                 percentile (close to longest), 50 = median.
    """
    lengths = get_baseline_lengths(vis, unflagged=unflagged)
    if lengths.size == 0:
        raise RuntimeError(f"No baselines found in {vis}")
    return float(np.percentile(lengths, percentile))


def pick_cell_imsize(
    vis: str,
    npix_per_beam: int = 10,
    max_baseline_percentile: float = 90.0,
    pb_level: float = 0.2,
    intent: str = "OBSERVE_TARGET#ON_SOURCE",
    diameter_m: float = ALMA_DISH_DIAMETER_M,
    obscuration_m: float = ALMA_DISH_OBSCURATION_M,
    taper_db: float = ALMA_DEFAULT_TAPER_DB,
    uvtaper=(),
) -> Dict[str, Any]:
    """Pick cell size and image size from MS baseline statistics.

    Replaces ``analysisUtils.pickCellSize``. Optionally accounts for
    ``uvtaper`` by computing the expected post-taper effective beam.

    Algorithm
        1. Compute the maximum-percentile baseline length.
        2. Compute the (untapered) synthesized beam FWHM as
           ``wavelength / baseline``.
        3. If ``uvtaper`` is non-empty, compute its image-domain Gaussian
           FWHM and take the effective beam as the quadrature sum
           ``effective_beam = sqrt(untapered_beam**2 + taper_image_fwhm**2)``.
           This is an empirical approximation; the actual post-taper beam
           depends on the visibility distribution and is accurate to roughly
           10-20 percent for moderate tapers.
        4. cellsize = effective_beam / npix_per_beam, rounded to 2 sig figs.
        5. Compute ALMA primary beam FWHM using the Baars taper formula
           (primary beam does NOT depend on uvtaper).
        6. Compute Gaussian offset radius at which the PB drops to pb_level.
        7. imsize_raw = ceil(2 * offset / cellsize).
        8. Round imsize up to the smallest 5-smooth integer at least
           imsize_raw (5-smooth means only prime factors 2, 3, 5;
           FFT-friendly).

    Validated against ``analysisUtils`` (with no uvtaper): at 232.994 GHz
    with 90-percentile baseline = 3293.48 m, returns cellsize=0.0081 arcsec
    and imsize=[4800, 4800].

    Returns
    -------
    dict
        Keys include ``cell_arcsec``, ``cell`` (tclean-ready string),
        ``imsize_xy``, ``beam_fwhm_arcsec`` (untapered),
        ``taper_image_fwhm_arcsec`` (0.0 if no uvtaper),
        ``predicted_effective_beam_arcsec``, ``primary_beam_arcsec``,
        ``fov_arcsec``, ``baseline_m``, ``freq_hz``, and ``justification``.
    """
    _require_casa()

    freq_hz = get_median_frequency(vis, intent=intent)
    wavelength_m = SPEED_OF_LIGHT_M_S / freq_hz
    baseline_m = get_baseline_at_percentile(vis, percentile=max_baseline_percentile)

    # 1-2. Untapered synthesized beam from wavelength/baseline.
    beam_fwhm_arcsec = wavelength_m / baseline_m * ARCSEC_PER_RAD

    # 3. Effective beam after applying uvtaper (if any).
    taper_image_fwhm = _parse_uvtaper_to_image_fwhm(uvtaper, wavelength_m)
    effective_beam_arcsec = math.sqrt(
        beam_fwhm_arcsec ** 2 + taper_image_fwhm ** 2
    )

    # 4. Cell size from effective beam.
    cell_arcsec_raw = effective_beam_arcsec / npix_per_beam
    cell_arcsec = _round_to_sig_figs(cell_arcsec_raw, n_sig_figs=2)

    # 5-8. imsize from ALMA primary beam at pb_level.
    # Note: analysisUtils.pickCellSize has two code paths -- the 'asdm path'
    # which does int(ceil(2*offset)/cell) and the 'vis path' which goes via
    # plotmosaic and effectively does ceil(2*offset/cell). For single-field
    # vis input, the vis path is invoked, and the validated test value
    # [0.0081, [4800,4800]] was produced by it. So we replicate the vis-path
    # behaviour (no inner ceil) here.
    pb_fwhm_arcsec = _primary_beam_fwhm_arcsec(
        freq_hz, diameter_m=diameter_m, obscuration_m=obscuration_m, taper_db=taper_db,
    )
    offset_arcsec = _gaussian_offset_for_pb_level(pb_fwhm_arcsec, pb_level)
    raw_size = int(math.ceil(2 * offset_arcsec / cell_arcsec))
    optimum_size = _smallest_5_smooth_at_least(raw_size)
    imsize_xy = [optimum_size, optimum_size]

    has_taper = taper_image_fwhm > 0.0
    if has_taper:
        cell_reason = (
            f"effective_beam = sqrt(beam^2 + taper^2) = "
            f"sqrt({beam_fwhm_arcsec:.4f}^2 + {taper_image_fwhm:.4f}^2) "
            f"= {effective_beam_arcsec:.4f} arcsec; "
            f"cellsize = effective_beam / {npix_per_beam}; "
            f"verify against actual dirty-image beam (approximation accurate "
            f"to ~10-20% for moderate tapers)"
        )
    else:
        cell_reason = (
            f"cellsize = beam/{npix_per_beam} from the "
            f"{max_baseline_percentile:.0f}-percentile baseline "
            f"({baseline_m:.1f} m); no uvtaper applied"
        )

    return {
        "cell_arcsec": cell_arcsec,
        "cell": f"{cell_arcsec}arcsec",
        "imsize_xy": imsize_xy,
        "beam_fwhm_arcsec": beam_fwhm_arcsec,
        "taper_image_fwhm_arcsec": taper_image_fwhm,
        "predicted_effective_beam_arcsec": effective_beam_arcsec,
        "primary_beam_arcsec": pb_fwhm_arcsec,
        "fov_arcsec": optimum_size * cell_arcsec,
        "baseline_m": baseline_m,
        "freq_hz": freq_hz,
        "justification": {
            "method": "ajisai_ms_utils.pick_cell_imsize",
            "max_baseline_percentile": max_baseline_percentile,
            "npix_per_beam": npix_per_beam,
            "pb_level": pb_level,
            "untapered_beam_fwhm_arcsec": beam_fwhm_arcsec,
            "uvtaper_input": list(uvtaper) if uvtaper else [],
            "taper_image_fwhm_arcsec": taper_image_fwhm,
            "predicted_effective_beam_arcsec": effective_beam_arcsec,
            "primary_beam_fwhm_arcsec": pb_fwhm_arcsec,
            "gaussian_offset_at_pblevel_arcsec": offset_arcsec,
            "raw_image_size_pix": raw_size,
            "optimum_size_5smooth_pix": optimum_size,
            "description": cell_reason,
        },
    }


# ============================================================================
# Coordinate utilities (replace rad2radec and ICRSToJ2000 via astropy)
# ============================================================================
def rad_to_radec(
    ra_rad: float,
    dec_rad: float,
    hms_dms: bool = True,
    delimiter: str = " ",
    precision: int = 5,
) -> str:
    """Convert RA/Dec from radians to a sexagesimal string.

    Replaces analysisUtils.rad2radec when called with explicit ra/dec values.
    Uses astropy for portable, well-tested coordinate formatting.

    Parameters
    ----------
    ra_rad, dec_rad : radians
    hms_dms         : if True, output like 'HH:MM:SS.ss DD:MM:SS.ss';
                      if False, output decimal degrees.
    delimiter       : separator between RA and Dec.
    precision       : decimal places on seconds field.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    c = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad, frame="icrs")
    if hms_dms:
        ra_str = c.ra.to_string(unit=u.hour, sep=":", precision=precision, pad=True)
        dec_str = c.dec.to_string(unit=u.deg, sep=":", precision=precision,
                                  pad=True, alwayssign=True)
        return f"{ra_str}{delimiter}{dec_str}"
    return f"{c.ra.deg:.{precision}f}{delimiter}{c.dec.deg:.{precision}f}"


def rad_to_radec_from_imfit(
    imfitdict: Dict[str, Any],
    component: int = 0,
    hms_dms: bool = True,
    delimiter: str = " ",
    precision: int = 5,
) -> str:
    """Extract RA/Dec from an imfit result dict and return as a string.

    Replaces ``analysisUtils.rad2radec(imfitdict=...)`` usage in the
    original AJISAI script. The imfit dict structure is
    ``imfitdict['deconvolved']['component0']['shape']['direction']``
    with ``m0`` = RA (radians) and ``m1`` = Dec (radians).
    """
    comp = imfitdict["deconvolved"][f"component{component}"]
    direction = comp["shape"]["direction"]
    ra_rad = float(direction["m0"]["value"])
    dec_rad = float(direction["m1"]["value"])
    return rad_to_radec(ra_rad, dec_rad, hms_dms=hms_dms,
                         delimiter=delimiter, precision=precision)


def _sexagesimal_to_colon(token: str) -> str:
    """Normalise a single sexagesimal token to ':'-separated form.

    CASA / analysisUtils declination strings use '.' as the *field*
    separator, e.g. ``-34.17.38.348`` (= -34d 17m 38.348s). astropy cannot
    parse that directly, so convert the field separators to ':' while
    preserving the fractional seconds. RA tokens (already ':'-separated) and
    unrecognised tokens are returned unchanged for astropy to handle.
    """
    if ":" in token:
        return token
    sign, body = "", token
    if body[:1] in "+-":
        sign, body = body[0], body[1:]
    fields = body.split(".")
    if len(fields) == 3:          # DD.MM.SS
        d, m, s = fields
    elif len(fields) == 4:        # DD.MM.SS.ffff  -> seconds = SS.ffff
        d, m, s = fields[0], fields[1], f"{fields[2]}.{fields[3]}"
    else:
        return token             # unknown layout; let astropy try as-is
    return f"{sign}{d}:{m}:{s}"


def _split_radec(coord_str: str) -> tuple:
    """Split a CASA/analysisUtils RA-Dec string into (ra_str, dec_str).

    Accepts both real-world input layouts:

    * CASA / analysisUtils form ``'HH:MM:SS.s DD.MM.SS.s'`` -- RA uses ':'
      separators, Dec uses '.' separators. This is what ``imstat`` /
      ``au.rad2radec`` emit and is the layout AJISAI actually feeds in.
    * Space-separated form ``'HH MM SS.s DD MM SS.s'`` (6 whitespace tokens).

    Both are returned with ':'-separated RA and Dec so astropy can parse them.
    """
    tokens = coord_str.split()
    if len(tokens) == 2:
        return _sexagesimal_to_colon(tokens[0]), _sexagesimal_to_colon(tokens[1])
    if len(tokens) == 6:
        return ":".join(tokens[:3]), ":".join(tokens[3:])
    raise ValueError(
        f"Cannot parse RA/Dec string {coord_str!r}; expected CASA form "
        f"'HH:MM:SS.s DD.MM.SS.s' or space form 'HH MM SS.s DD MM SS.s'"
    )


def icrs_to_j2000(coord_str: str) -> str:
    """Convert an ICRS sexagesimal coordinate string to J2000 (FK5).

    Replaces analysisUtils.ICRSToJ2000. Uses astropy. The two frames share
    the same origin at epoch J2000.0, so the on-sky position is unchanged and
    the returned coordinate label is numerically almost identical to the
    input; AJISAI uses this for CASA's ``fixplanets`` / ``phasecenter``
    arguments, which need an explicit J2000 label.

    Input may be either the CASA / analysisUtils layout
    ``'HH:MM:SS.s DD.MM.SS.s'`` (RA ':'-separated, Dec '.'-separated -- the
    format emitted by ``imstat``'s ``maxposf`` and ``au.rad2radec``) or the
    space-separated ``'HH MM SS.s DD MM SS.s'`` layout.

    The output preserves the CASA convention: RA with ':' separators and Dec
    with '.' separators (e.g. ``'15:45:06.32 -34.17.38.34'``). This matters
    because CASA interprets ':' in a declination as a time (hours) value, so
    Dec must use '.' separators to be parsed as an angle downstream.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    ra_str, dec_str = _split_radec(coord_str)
    c_icrs = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame="icrs")
    c_j2000 = c_icrs.transform_to("fk5")  # FK5 J2000.0
    ra_out = c_j2000.ra.to_string(unit=u.hour, sep=":", precision=5, pad=True)
    dec_out = c_j2000.dec.to_string(unit=u.deg, sep=".", precision=5,
                                    pad=True, alwayssign=True)
    return f"{ra_out} {dec_out}"


# ============================================================================
# Self-test (run on a known MS to verify pickCellSize agreement)
# ============================================================================
def test_pick_cell_imsize(vis: str, verbose: bool = True) -> Dict[str, Any]:
    """Smoke test against the validated analysisUtils output.

    Reference (analysisUtils, V883Ori 232 GHz, 90-percentile baseline 3293.48 m):
        cellsize = 0.0081 arcsec
        imsize   = [4800, 4800]

    Run from CASA with the test MS:
        from ajisai_ms_utils import test_pick_cell_imsize
        test_pick_cell_imsize("/path/to/test.ms")
    """
    result = pick_cell_imsize(
        vis,
        npix_per_beam=10,
        max_baseline_percentile=90,
    )
    if verbose:
        print("=" * 60)
        print("  ajisai_ms_utils.pick_cell_imsize self-test")
        print("=" * 60)
        print(f"  cellsize         : {result['cell_arcsec']:.5f} arcsec")
        print(f"  imsize           : {result['imsize_xy']}")
        print(f"  beam FWHM        : {result['beam_fwhm_arcsec']:.5f} arcsec")
        print(f"  primary beam     : {result['primary_beam_arcsec']:.3f} arcsec")
        print(f"  FoV              : {result['fov_arcsec']:.3f} arcsec")
        print(f"  baseline (90%ile): {result['baseline_m']:.2f} m")
        print(f"  median freq      : {result['freq_hz']/1e9:.3f} GHz")
        print()
        print("  Expected (AU)    : cellsize=0.0081, imsize=[4800,4800]")
        # Pass/fail
        cell_ok = abs(result['cell_arcsec'] - 0.0081) < 1e-4
        imsize_ok = result['imsize_xy'] == [4800, 4800]
        print(f"  cellsize match   : {'PASS' if cell_ok else 'FAIL'}")
        print(f"  imsize match     : {'PASS' if imsize_ok else 'FAIL'}")
    return result


# ============================================================================
# Numerical self-checks (no CASA required; pure math)
# ============================================================================
def _self_check_math() -> bool:
    """Run pure-math checks (no MS required). Returns True if all pass."""
    ok = True
    # 1. 5-smooth rounding
    cases = [
        (4691, 4800),
        (1000, 1000),  # 2^3 * 5^3
        (1001, 1024),  # 2^10
        (4938, 5000),  # 2^3 * 5^4
    ]
    for n_in, n_expected in cases:
        n_out = _smallest_5_smooth_at_least(n_in)
        if n_out != n_expected:
            print(f"  FAIL: _smallest_5_smooth_at_least({n_in}) = {n_out}, "
                  f"expected {n_expected}")
            ok = False
    # 2. roundFigures
    for v_in, sf, v_expected in [(0.008058, 2, 0.0081), (123.456, 3, 123.0), (0.0, 2, 0.0)]:
        v_out = _round_to_sig_figs(v_in, sf)
        if abs(v_out - v_expected) > 1e-6:
            print(f"  FAIL: _round_to_sig_figs({v_in}, {sf}) = {v_out}, "
                  f"expected {v_expected}")
            ok = False
    # 3. Baars taper factor at 10 dB ~ 1.137
    b = _baars_taper_factor(10.0)
    if abs(b - 1.137) > 0.005:
        print(f"  FAIL: _baars_taper_factor(10) = {b}, expected ~1.137")
        ok = False
    # 4. Primary beam at 232.994 GHz ~ 24.9 arcsec
    pb = _primary_beam_fwhm_arcsec(232.994e9)
    if abs(pb - 24.9) > 0.5:
        print(f"  FAIL: PB at 232.994 GHz = {pb} arcsec, expected ~24.9")
        ok = False
    # 5. Gaussian offset at FWHM=24.9, pb_level=0.2 ~ 19.0 arcsec
    off = _gaussian_offset_for_pb_level(24.93, 0.2)
    if abs(off - 19.0) > 0.5:
        print(f"  FAIL: offset(FWHM=24.93, pb=0.2) = {off}, expected ~19.0")
        ok = False
    # 6. Full pickCellSize math at 232.994 GHz, 3293.48 m (no uvtaper)
    freq_hz = 232.994e9
    baseline_m = 3293.48
    wavelength_m = SPEED_OF_LIGHT_M_S / freq_hz
    beam = wavelength_m / baseline_m * ARCSEC_PER_RAD
    cell = _round_to_sig_figs(beam / 10, 2)
    pb_fwhm = _primary_beam_fwhm_arcsec(freq_hz)
    off = _gaussian_offset_for_pb_level(pb_fwhm, 0.2)
    raw = int(math.ceil(2 * off / cell))
    final = _smallest_5_smooth_at_least(raw)
    if not (abs(cell - 0.0081) < 1e-4 and final == 4800):
        print(f"  FAIL: end-to-end at V883Ori parameters -> "
              f"cell={cell}, imsize={final}; expected 0.0081 / 4800")
        ok = False

    # 7. uvtaper parser
    # 7a. Empty input -> 0
    if _parse_uvtaper_to_image_fwhm([], wavelength_m) != 0.0:
        print("  FAIL: _parse_uvtaper_to_image_fwhm([]) != 0.0")
        ok = False
    if _parse_uvtaper_to_image_fwhm(None, wavelength_m) != 0.0:
        print("  FAIL: _parse_uvtaper_to_image_fwhm(None) != 0.0")
        ok = False
    # 7b. Direct image-domain FWHM
    v = _parse_uvtaper_to_image_fwhm(["1arcsec"], wavelength_m)
    if abs(v - 1.0) > 1e-6:
        print(f"  FAIL: ['1arcsec'] -> {v}, expected 1.0")
        ok = False
    v = _parse_uvtaper_to_image_fwhm(['0.5"'], wavelength_m)
    if abs(v - 0.5) > 1e-6:
        print(f"  FAIL: ['0.5\"'] -> {v}, expected 0.5")
        ok = False
    v = _parse_uvtaper_to_image_fwhm(["100mas"], wavelength_m)
    if abs(v - 0.1) > 1e-6:
        print(f"  FAIL: ['100mas'] -> {v}, expected 0.1")
        ok = False
    # 7c. uv-domain klambda
    # At 100 klambda: FWHM_image = 0.8825 / (100e3) * 206265 = 1.82 arcsec
    v = _parse_uvtaper_to_image_fwhm(["100klambda"], wavelength_m)
    expected = _UVTAPER_FWHM_PRODUCT / 100e3 * ARCSEC_PER_RAD  # ~1.82
    if abs(v - expected) > 0.01:
        print(f"  FAIL: ['100klambda'] -> {v}, expected ~{expected:.3f}")
        ok = False
    # 7d. Elliptical taper: geometric mean
    v = _parse_uvtaper_to_image_fwhm(["2arcsec", "0.5arcsec", "30deg"], wavelength_m)
    if abs(v - 1.0) > 1e-6:  # sqrt(2 * 0.5) = 1.0
        print(f"  FAIL: ['2arcsec','0.5arcsec','30deg'] -> {v}, expected 1.0")
        ok = False

    # 8. End-to-end with uvtaper: should give larger cell than no-uvtaper
    # At V883Ori conditions with 0.5arcsec image-domain taper, effective
    # beam is sqrt(0.0806^2 + 0.5^2) ~= 0.506 arcsec, so cell ~= 0.051
    # (rounded to 2 sig figs).
    taper = _parse_uvtaper_to_image_fwhm(["0.5arcsec"], wavelength_m)
    eff_beam = math.sqrt(beam ** 2 + taper ** 2)
    cell_tapered = _round_to_sig_figs(eff_beam / 10, 2)
    if not (cell_tapered > cell and abs(cell_tapered - 0.051) < 0.005):
        print(f"  FAIL: uvtaper=0.5arcsec -> cell={cell_tapered}, "
              f"expected ~0.051 (must be > {cell})")
        ok = False
    return ok


if __name__ == "__main__":
    print("ajisai_ms_utils self-checks (math only, no CASA required)")
    ok = _self_check_math()
    print(f"\nResult: {'ALL PASS' if ok else 'SOME FAILURES'}")
