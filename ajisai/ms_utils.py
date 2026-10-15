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
    5. pickCellSize              -> pick_cell_imsize(rule="percentile")
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

# ALMA antenna parameters (12 m dish, central obscuration 0.75 m).
# The dish diameter is read from ANTENNA/DISH_DIAMETER (smallest dish = largest
# primary beam); ALMA_DISH_DIAMETER_M is only the fallback for an MS without
# usable values in that column.
ALMA_DISH_DIAMETER_M = 12.0
ALMA_DISH_OBSCURATION_M = 0.75
ALMA_DEFAULT_TAPER_DB = 10.0

# Default cell rule: cell = 1 / (CELL_SAMPLES_PER_UMAX * u_max), with u_max (in
# wavelengths) from the longest unflagged baseline at the highest frequency.
CELL_SAMPLES_PER_UMAX = 5.0

# Cell rules accepted by pick_cell_imsize / ImagingConfig.cell_rule.
CELL_RULES = ("umax", "percentile")


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


def _floor_to_sig_figs(value: float, n_sig_figs: int = 4) -> float:
    """Round a positive value DOWN to N significant figures.

    Used for the u_max cell rule: rounding must never make the cell larger
    than 1/(5 u_max), so the value is truncated rather than rounded.

    Example: _floor_to_sig_figs(0.0035579, 4) -> 0.003557
    """
    if value <= 0.0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(value))
    factor = 10 ** (n_sig_figs - 1 - exponent)
    # The small epsilon keeps exact decimal values (e.g. 0.0062) from being
    # truncated one step down by binary floating-point representation.
    return math.floor(value * factor * (1.0 + 1e-12)) / factor


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
# Dish diameter and u_max: pure parts (the CASA readers are in the public API)
# ============================================================================
def smallest_dish_diameter(
    diameters,
    fallback_m: float = ALMA_DISH_DIAMETER_M,
) -> Dict[str, Any]:
    """Smallest dish diameter (= largest primary beam) from DISH_DIAMETER values.

    Non-finite and non-positive entries are ignored. When nothing usable is
    left (column missing, empty, or all zero), ``fallback_m`` is returned with
    ``source='fallback'``.

    Returns
    -------
    dict
        ``diameter_m``, ``source`` (``'ANTENNA/DISH_DIAMETER'`` or
        ``'fallback'``), ``n_antennas``, ``unique_diameters_m``, ``reason``.
    """
    vals = np.asarray(diameters if diameters is not None else [], dtype=float).ravel()
    good = vals[np.isfinite(vals) & (vals > 0.0)]
    if good.size == 0:
        return {
            "diameter_m": float(fallback_m),
            "source": "fallback",
            "n_antennas": int(vals.size),
            "unique_diameters_m": [],
            "reason": (f"ANTENNA/DISH_DIAMETER missing or not positive; "
                       f"fallback {fallback_m:g} m"),
        }
    unique = sorted(set(float(x) for x in np.round(good, 3)))
    return {
        "diameter_m": float(good.min()),
        "source": "ANTENNA/DISH_DIAMETER",
        "n_antennas": int(vals.size),
        "unique_diameters_m": unique,
        "reason": (f"smallest dish of {unique} m in ANTENNA/DISH_DIAMETER "
                   f"(largest primary beam)"),
    }


def max_uv_of_unflagged_rows(uvw_m, flag, flag_row=None):
    """Longest projected baseline among rows that carry unflagged data.

    ``uvw_m`` is the UVW column in metres, shape ``(3, nrow)``; ``flag`` the
    FLAG column, shape ``(npol, nchan, nrow)``; ``flag_row`` (optional) the
    FLAG_ROW column. A row counts when it is not FLAG_ROW-flagged and has at
    least one unflagged sample.

    Returns a pair: the ``sqrt(u**2 + v**2)`` of the longest counted row in
    metres (0.0 if none), and a boolean array over channels that is True
    where any counted row has an unflagged sample.
    """
    flag = np.asarray(flag, dtype=bool)
    row_ok = ~flag.all(axis=(0, 1))
    if flag_row is not None:
        row_ok &= ~np.asarray(flag_row, dtype=bool)
    if not row_ok.any():
        return 0.0, np.zeros(flag.shape[1], dtype=bool)
    chan_ok = (~flag[:, :, row_ok]).any(axis=(0, 2))
    uvw_m = np.asarray(uvw_m, dtype=float)
    uv = np.hypot(uvw_m[0, row_ok], uvw_m[1, row_ok])
    return float(uv.max()), chan_ok


def uv_max_from_maxima(
    max_uv_m_by_spw: Dict[int, float],
    fmax_hz_by_spw: Dict[int, float],
) -> Dict[str, Any]:
    """u_max in wavelengths from per-spw maxima.

    ``u_max = max over spw of (longest unflagged |uv| in m) * f_max / c``, where
    ``f_max`` is the highest frequency with unflagged data in that spw.

    Raises
    ------
    ValueError
        If no spw has a positive baseline and frequency (all data flagged).
    """
    best: Optional[Dict[str, Any]] = None
    for spw, uv_m in max_uv_m_by_spw.items():
        f_hz = fmax_hz_by_spw.get(spw)
        if f_hz is None or not (math.isfinite(uv_m) and uv_m > 0.0):
            continue
        if not (math.isfinite(f_hz) and f_hz > 0.0):
            continue
        u = uv_m * f_hz / SPEED_OF_LIGHT_M_S
        if best is None or u > best["u_max_lambda"]:
            best = {"u_max_lambda": float(u), "baseline_m": float(uv_m),
                    "freq_hz": float(f_hz), "spw": int(spw)}
    if best is None:
        raise ValueError("no unflagged baseline: u_max cannot be determined")
    return best


def cell_from_umax(u_max_lambda: float, samples: float = CELL_SAMPLES_PER_UMAX) -> float:
    """Cell size in arcsec, ``1 / (samples * u_max)`` radians.

    ``u_max`` is in wavelengths. With the default ``samples = 5`` the longest
    baseline's fringe is sampled by five pixels.
    """
    if not (u_max_lambda > 0.0 and math.isfinite(u_max_lambda)):
        raise ValueError(f"u_max must be positive and finite; got {u_max_lambda!r}")
    if not samples > 0.0:
        raise ValueError(f"samples must be positive; got {samples!r}")
    return ARCSEC_PER_RAD / (samples * u_max_lambda)


# ============================================================================
# Public API
# ============================================================================
def get_array_info(vis: str) -> Dict[str, Any]:
    """Return basic array properties of an MS (replacement for ValueMapping).

    Returns
    -------
    dict with keys:
        n_antennas              : int
        dish_diameters_m        : list of float (ANTENNA/DISH_DIAMETER; [] if absent)
        min_dish_diameter_m     : float (smallest dish; 12 m fallback if absent)
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
        diam = tb.getcol("DISH_DIAMETER") if "DISH_DIAMETER" in tb.colnames() else []
        info["dish_diameters_m"] = [float(x) for x in diam]
    finally:
        tb.close()
    info["min_dish_diameter_m"] = smallest_dish_diameter(info["dish_diameters_m"])["diameter_m"]

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


def _casa_column_reader(table_path: str, column: str) -> np.ndarray:
    """Read one column of a CASA table (raises KeyError if the column is absent)."""
    _require_casa()
    tb = casatools.table()
    tb.open(table_path)
    try:
        if column not in tb.colnames():
            raise KeyError(f"{table_path} has no column {column!r}")
        return np.asarray(tb.getcol(column))
    finally:
        tb.close()


def get_dish_diameter_m(
    vis: str,
    fallback_m: float = ALMA_DISH_DIAMETER_M,
    table_reader=None,
) -> Dict[str, Any]:
    """Smallest dish diameter of the array, read from ``ANTENNA/DISH_DIAMETER``.

    The smallest dish has the largest primary beam, so it sets the field of
    view that must be imaged. ``fallback_m`` (12 m) is used only when the
    column is missing or holds no positive value.

    ``table_reader(table_path, column)`` returns the column as an array; the
    default reads it with casatools (tests pass a fake reader). The result is
    the dict of :func:`smallest_dish_diameter`.
    """
    reader = table_reader if table_reader is not None else _casa_column_reader
    try:
        diameters = reader(vis + "/ANTENNA", "DISH_DIAMETER")
    except Exception:  # column absent or table unreadable -> fallback
        diameters = None
    return smallest_dish_diameter(diameters, fallback_m=fallback_m)


def _field_id_for(vis: str, field: Optional[str]) -> Optional[int]:
    """FIELD_ID for a field given as an ID string or a NAME (None stays None)."""
    if field is None or str(field).strip() == "":
        return None
    try:
        return int(field)
    except ValueError:
        pass
    names = list(_casa_column_reader(vis + "/FIELD", "NAME"))
    matches = [i for i, n in enumerate(names) if n == field]
    if not matches:
        raise ValueError(f"Field name {field!r} not found in {vis}/FIELD")
    return matches[0]


# Number of visibility samples (npol * nchan * nrow) read per chunk when
# scanning the MAIN table, so that large MSs are never loaded at once.
_CHUNK_SAMPLES = 20_000_000


def get_uv_max(vis: str, field: Optional[str] = None) -> Dict[str, Any]:
    """Longest unflagged baseline at the highest frequency, in wavelengths.

    For every data description the MAIN rows of the field are scanned in
    chunks; a row counts when it has at least one unflagged sample. The
    result is ``u_max = max over spw of max|uv| (m) * f_max / c`` with
    ``f_max`` the highest channel frequency that has unflagged data in the
    counted rows.

    Returns
    -------
    dict
        ``u_max_lambda``, ``baseline_m`` (``|uv|`` of the longest counted row),
        ``freq_hz`` (f_max of the spw giving u_max), ``spw``, ``fmin_hz``
        (lowest unflagged channel frequency over all spws), ``fmax_hz``.
    """
    _require_casa()
    fid = _field_id_for(vis, field)
    dd_spw = _casa_column_reader(vis + "/DATA_DESCRIPTION", "SPECTRAL_WINDOW_ID")
    tb = casatools.table()
    tb.open(vis + "/SPECTRAL_WINDOW")
    try:
        chan_freq = {i: np.asarray(tb.getcell("CHAN_FREQ", i), dtype=float)
                     for i in range(tb.nrows())}
    finally:
        tb.close()

    max_uv: Dict[int, float] = {}
    chan_any: Dict[int, np.ndarray] = {}
    tb.open(vis)
    try:
        for dd in range(len(dd_spw)):
            query = f"DATA_DESC_ID=={dd}"
            if fid is not None:
                query += f" && FIELD_ID=={fid}"
            sub = tb.query(query, columns="UVW,FLAG,FLAG_ROW")
            try:
                nrow = sub.nrows()
                if nrow == 0:
                    continue
                spw = int(dd_spw[dd])
                npol, nchan = np.asarray(sub.getcell("FLAG", 0)).shape
                chunk = max(1, _CHUNK_SAMPLES // max(1, npol * nchan))
                for start in range(0, nrow, chunk):
                    nr = min(chunk, nrow - start)
                    m, ok = max_uv_of_unflagged_rows(
                        sub.getcol("UVW", start, nr),
                        sub.getcol("FLAG", start, nr),
                        sub.getcol("FLAG_ROW", start, nr),
                    )
                    max_uv[spw] = max(max_uv.get(spw, 0.0), m)
                    prev = chan_any.get(spw)
                    chan_any[spw] = ok if prev is None else (prev | ok)
            finally:
                sub.close()
    finally:
        tb.close()

    fmax: Dict[int, float] = {}
    fmin_all: List[float] = []
    for spw, ok in chan_any.items():
        if ok.any():
            freqs = chan_freq[spw][ok]
            fmax[spw] = float(freqs.max())
            fmin_all.append(float(freqs.min()))
    best = uv_max_from_maxima(max_uv, fmax)
    best["fmin_hz"] = float(min(fmin_all))
    best["fmax_hz"] = float(max(fmax.values()))
    return best


def read_vis_summary(vis: str, field: Optional[str] = None, band_halves: bool = False,
                     chunk_samples: int = _CHUNK_SAMPLES):
    """Read DATA, MODEL_DATA, WEIGHT and FLAG of a field once into an ``ajisai.solint.VisSummary``.

    Spec (ASEBI v0.3): 10.2 item 14 (the held-out evaluation reads the four
    columns once and solves in numpy), C.2.

    Cross-correlations of the field only; per row the sums of
    :func:`ajisai.solint.reduce_visibilities` over the parallel hands
    (WEIGHT_SPECTRUM when the column holds values, else WEIGHT), the UV
    distance (m) and, with ``band_halves=True``, the sums over the lower and
    upper half of each spw's channels. Rows without any valid sample are
    dropped. The table is opened read-only.
    """
    _require_casa()
    from .solint import VisSummary, reduce_visibilities

    fid = _field_id_for(vis, field)
    dd_spw = _casa_column_reader(vis + "/DATA_DESCRIPTION", "SPECTRAL_WINDOW_ID")
    dd_pol = _casa_column_reader(vis + "/DATA_DESCRIPTION", "POLARIZATION_ID")
    names = [str(n) for n in _casa_column_reader(vis + "/ANTENNA", "NAME")]
    tb = casatools.table()
    tb.open(vis + "/POLARIZATION")
    try:
        corr = [np.asarray(tb.getcell("CORR_TYPE", i)).ravel() for i in range(tb.nrows())]
    finally:
        tb.close()
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in (
        "time", "obs", "scan", "spw", "a1", "a2", "y", "P", "S", "uv", "yh", "Ph")}
    tb.open(vis)
    try:
        cols = tb.colnames()
        if "MODEL_DATA" not in cols:
            raise RuntimeError(f"{vis} has no MODEL_DATA column; predict the model first")
        has_ws = "WEIGHT_SPECTRUM" in cols
        for dd in range(len(dd_spw)):
            query = f"DATA_DESC_ID=={dd} && ANTENNA1!=ANTENNA2"
            if fid is not None:
                query += f" && FIELD_ID=={fid}"
            sub = tb.query(query)
            try:
                nrow = sub.nrows()
                if nrow == 0:
                    continue
                use_ws = has_ws and sub.iscelldefined("WEIGHT_SPECTRUM", 0)
                npol, nchan = np.asarray(sub.getcell("FLAG", 0)).shape
                chunk = max(1, chunk_samples // max(1, npol * nchan))
                ctypes = corr[int(dd_pol[dd])]
                for start in range(0, nrow, chunk):
                    nr = min(chunk, nrow - start)
                    red = reduce_visibilities(
                        sub.getcol("DATA", start, nr), sub.getcol("MODEL_DATA", start, nr),
                        sub.getcol("FLAG", start, nr),
                        sub.getcol("WEIGHT_SPECTRUM" if use_ws else "WEIGHT", start, nr),
                        sub.getcol("FLAG_ROW", start, nr), ctypes,
                        halves=band_halves and nchan >= 2)
                    keep = red["n_valid"] > 0
                    uvw = sub.getcol("UVW", start, nr)
                    parts["time"].append(sub.getcol("TIME", start, nr)[keep])
                    parts["obs"].append(sub.getcol("OBSERVATION_ID", start, nr)[keep])
                    parts["scan"].append(sub.getcol("SCAN_NUMBER", start, nr)[keep])
                    parts["spw"].append(np.full(int(keep.sum()), int(dd_spw[dd])))
                    parts["a1"].append(sub.getcol("ANTENNA1", start, nr)[keep])
                    parts["a2"].append(sub.getcol("ANTENNA2", start, nr)[keep])
                    parts["y"].append(red["y"][keep])
                    parts["P"].append(red["P"][keep])
                    parts["S"].append(red["S"][keep])
                    parts["uv"].append(np.hypot(uvw[0], uvw[1])[keep])
                    if band_halves:
                        if "y_half" not in red:
                            raise RuntimeError(f"spw {int(dd_spw[dd])} of {vis} has one channel; "
                                               f"the band cannot be split into halves")
                        parts["yh"].append(red["y_half"][:, keep])
                        parts["Ph"].append(red["P_half"][:, keep])
            finally:
                sub.close()
    finally:
        tb.close()
    cat = {k: (np.concatenate(v, axis=-1) if v else np.zeros(0)) for k, v in parts.items()}
    order = np.lexsort((cat["time"], cat["a2"], cat["a1"], cat["spw"], cat["obs"]))

    def pick(x):
        return x[..., order] if x.ndim == 2 else x[order]

    return VisSummary(
        time=cat["time"][order], obs=cat["obs"][order], scan=cat["scan"][order],
        spw=cat["spw"][order], a1=cat["a1"][order], a2=cat["a2"][order], y=cat["y"][order],
        P=cat["P"][order], antenna_names=names, S=cat["S"][order], uvdist_m=cat["uv"][order],
        y_half=pick(cat["yh"]) if band_halves else None,
        P_half=pick(cat["Ph"]) if band_halves else None, source=str(vis))


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
    diameter_m: Optional[float] = None,
    obscuration_m: float = ALMA_DISH_OBSCURATION_M,
    taper_db: float = ALMA_DEFAULT_TAPER_DB,
    uvtaper=(),
    rule: str = "umax",
    field: Optional[str] = None,
    samples_per_umax: float = CELL_SAMPLES_PER_UMAX,
) -> Dict[str, Any]:
    """Pick cell size and image size from the MS.

    Two cell rules are available (``rule``):

    ``"umax"`` (default)
        ``cell = 1 / (samples_per_umax * u_max)`` with ``u_max`` (wavelengths)
        from the longest unflagged baseline at the highest frequency
        (:func:`get_uv_max`), truncated to 4 significant figures so that it
        never exceeds the rule. The image covers the primary beam down to
        ``pb_level`` at the lowest unflagged frequency.
    ``"percentile"`` (legacy, the v0.2 rule)
        Replaces ``analysisUtils.pickCellSize``: the synthesized beam is
        ``wavelength / B`` with ``B`` the ``max_baseline_percentile``-th
        percentile baseline at the median frequency; with ``uvtaper`` the
        effective beam is ``sqrt(beam**2 + taper_fwhm**2)`` (an approximation
        accurate to roughly 10-20 percent for moderate tapers);
        ``cell = effective_beam / npix_per_beam`` rounded to 2 significant
        figures. The image covers the primary beam down to ``pb_level`` at
        the median frequency. Validated against ``analysisUtils`` (no
        uvtaper): at 232.994 GHz with 90-percentile baseline = 3293.48 m,
        cellsize = 0.0081 arcsec and imsize = [4800, 4800].

    In both rules the primary beam uses ``diameter_m``; when it is ``None``
    the smallest dish in ``ANTENNA/DISH_DIAMETER`` is used (largest primary
    beam), with 12 m only as a fallback. The image size is rounded up to the
    smallest 5-smooth integer (FFT-friendly). The predicted (tapered) beam
    from the percentile baseline is returned by both rules for the
    actual-vs-predicted beam check of the dirty image.

    Returns
    -------
    dict
        Keys include ``cell_arcsec``, ``cell`` (tclean-ready string),
        ``imsize_xy``, ``rule``, ``beam_fwhm_arcsec`` (untapered),
        ``taper_image_fwhm_arcsec`` (0.0 if no uvtaper),
        ``predicted_effective_beam_arcsec``, ``primary_beam_arcsec``,
        ``pb_freq_hz``, ``dish_diameter_m``, ``dish``, ``fov_arcsec``,
        ``baseline_m``, ``freq_hz``, ``uv_max`` (None for the percentile
        rule), and ``justification``.
    """
    _require_casa()
    if rule not in CELL_RULES:
        raise ValueError(f"rule must be one of {CELL_RULES}; got {rule!r}")

    freq_hz = get_median_frequency(vis, intent=intent)
    wavelength_m = SPEED_OF_LIGHT_M_S / freq_hz
    baseline_m = get_baseline_at_percentile(vis, percentile=max_baseline_percentile)

    # Predicted (untapered and tapered) synthesized beam from the percentile
    # baseline. It sets the legacy cell and is kept for the beam check.
    beam_fwhm_arcsec = wavelength_m / baseline_m * ARCSEC_PER_RAD
    taper_image_fwhm = _parse_uvtaper_to_image_fwhm(uvtaper, wavelength_m)
    effective_beam_arcsec = math.sqrt(beam_fwhm_arcsec ** 2 + taper_image_fwhm ** 2)

    # Dish diameter: smallest dish in ANTENNA (largest primary beam).
    if diameter_m is None:
        dish = get_dish_diameter_m(vis)
    else:
        dish = {"diameter_m": float(diameter_m), "source": "argument",
                "n_antennas": None, "unique_diameters_m": [float(diameter_m)],
                "reason": f"diameter_m={diameter_m:g} given by the caller"}
    dish_m = dish["diameter_m"]

    uvmax: Optional[Dict[str, Any]] = None
    if rule == "umax":
        uvmax = get_uv_max(vis, field=field)
        cell_arcsec_raw = cell_from_umax(uvmax["u_max_lambda"], samples=samples_per_umax)
        cell_arcsec = _floor_to_sig_figs(cell_arcsec_raw, n_sig_figs=4)
        pb_freq_hz = uvmax["fmin_hz"]
        cell_reason = (
            f"cell = 1/({samples_per_umax:g} u_max) = {cell_arcsec_raw:.6g} arcsec "
            f"(truncated to {cell_arcsec:.4g}); u_max = {uvmax['u_max_lambda']:.6g} lambda "
            f"from the longest unflagged baseline ({uvmax['baseline_m']:.1f} m) at the "
            f"highest unflagged frequency ({uvmax['freq_hz'] / 1e9:.4f} GHz, spw "
            f"{uvmax['spw']})"
        )
        if taper_image_fwhm > 0.0:
            cell_reason += ("; uvtaper widens the beam but the cell still samples the "
                            "untapered u_max (use rule='percentile' or an explicit cell "
                            "for heavily tapered images)")
    else:
        cell_arcsec_raw = effective_beam_arcsec / npix_per_beam
        cell_arcsec = _round_to_sig_figs(cell_arcsec_raw, n_sig_figs=2)
        pb_freq_hz = freq_hz
        if taper_image_fwhm > 0.0:
            cell_reason = (
                f"legacy percentile rule: effective_beam = sqrt(beam^2 + taper^2) = "
                f"sqrt({beam_fwhm_arcsec:.4f}^2 + {taper_image_fwhm:.4f}^2) "
                f"= {effective_beam_arcsec:.4f} arcsec; "
                f"cellsize = effective_beam / {npix_per_beam}; "
                f"verify against actual dirty-image beam (approximation accurate "
                f"to ~10-20% for moderate tapers)"
            )
        else:
            cell_reason = (
                f"legacy percentile rule: cellsize = beam/{npix_per_beam} from the "
                f"{max_baseline_percentile:.0f}-percentile baseline "
                f"({baseline_m:.1f} m); no uvtaper applied"
            )

    # imsize from the primary beam at pb_level.
    # Note: analysisUtils.pickCellSize has two code paths -- the 'asdm path'
    # which does int(ceil(2*offset)/cell) and the 'vis path' which goes via
    # plotmosaic and effectively does ceil(2*offset/cell). For single-field
    # vis input, the vis path is invoked, and the validated test value
    # [0.0081, [4800,4800]] was produced by it. So we replicate the vis-path
    # behaviour (no inner ceil) here.
    pb_fwhm_arcsec = _primary_beam_fwhm_arcsec(
        pb_freq_hz, diameter_m=dish_m, obscuration_m=obscuration_m, taper_db=taper_db,
    )
    offset_arcsec = _gaussian_offset_for_pb_level(pb_fwhm_arcsec, pb_level)
    raw_size = int(math.ceil(2 * offset_arcsec / cell_arcsec))
    optimum_size = _smallest_5_smooth_at_least(raw_size)
    imsize_xy = [optimum_size, optimum_size]

    return {
        "cell_arcsec": cell_arcsec,
        "cell": f"{cell_arcsec}arcsec",
        "imsize_xy": imsize_xy,
        "rule": rule,
        "beam_fwhm_arcsec": beam_fwhm_arcsec,
        "taper_image_fwhm_arcsec": taper_image_fwhm,
        "predicted_effective_beam_arcsec": effective_beam_arcsec,
        "primary_beam_arcsec": pb_fwhm_arcsec,
        "pb_freq_hz": pb_freq_hz,
        "dish_diameter_m": dish_m,
        "dish": dish,
        "fov_arcsec": optimum_size * cell_arcsec,
        "baseline_m": baseline_m,
        "freq_hz": freq_hz,
        "uv_max": uvmax,
        "justification": {
            "method": "ajisai.ms_utils.pick_cell_imsize",
            "rule": rule,
            "samples_per_umax": samples_per_umax if rule == "umax" else None,
            "uv_max": uvmax,
            "max_baseline_percentile": max_baseline_percentile,
            "npix_per_beam": npix_per_beam if rule == "percentile" else None,
            "pb_level": pb_level,
            "pb_freq_hz": pb_freq_hz,
            "dish": dish,
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
        rule="percentile",   # the analysisUtils-equivalent legacy rule
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
