"""
Pure-math tests for ``ajisai.ms_utils``.

These tests do NOT require CASA. They validate the numerical helpers that
underpin AJISAI's cell/imsize selection and uvtaper-aware beam estimation.

The benchmark case throughout is the V883 Ori 232.994 GHz dataset where the
90-percentile baseline is 3293.48 m. With these inputs, ``pick_cell_imsize``
must produce ``cellsize = 0.0081 arcsec`` and ``imsize = [4800, 4800]``,
matching the validated ``analysisUtils.pickCellSize`` output.
"""
from __future__ import annotations

import math

import pytest

from ajisai.ms_utils import (
    ARCSEC_PER_RAD,
    SPEED_OF_LIGHT_M_S,
    _UVTAPER_FWHM_PRODUCT,
    _baars_taper_factor,
    _central_obscuration_factor,
    _gaussian_offset_for_pb_level,
    _parse_uvtaper_to_image_fwhm,
    _primary_beam_fwhm_arcsec,
    _round_to_sig_figs,
    _smallest_5_smooth_at_least,
)


# ---------------------------------------------------------------------------
# _smallest_5_smooth_at_least
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_in, n_expected", [
    (4691, 4800),    # V883Ori imsize_raw -> snaps to 4800 (=2^6 * 3 * 5^2)
    (1000, 1000),    # already 5-smooth (2^3 * 5^3)
    (1001, 1024),    # next 5-smooth above 1001 is 1024 (2^10)
    (4938, 5000),    # next 5-smooth above 4938 is 5000 (2^3 * 5^4)
    (1, 1),
    (2, 2),
    (7, 8),          # next 5-smooth above 7 is 8 (2^3)
    (11, 12),        # next 5-smooth above 11 is 12 (2^2 * 3)
])
def test_smallest_5_smooth_at_least(n_in, n_expected):
    assert _smallest_5_smooth_at_least(n_in) == n_expected


def test_smallest_5_smooth_zero_or_negative():
    """Non-positive inputs return 1 (defensive)."""
    assert _smallest_5_smooth_at_least(0) == 1
    assert _smallest_5_smooth_at_least(-5) == 1


# ---------------------------------------------------------------------------
# _round_to_sig_figs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value, sig_figs, expected", [
    (0.008058, 2, 0.0081),
    (123.456, 3, 123.0),
    (0.0, 2, 0.0),
    (1.0, 1, 1.0),
    (999.5, 3, 1000.0),
])
def test_round_to_sig_figs(value, sig_figs, expected):
    assert _round_to_sig_figs(value, sig_figs) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Baars taper factor and central obscuration factor
# ---------------------------------------------------------------------------
def test_baars_taper_factor_at_10db():
    """At 10 dB taper, b ~= 1.137 (Baars 2007 Eq. 4.13)."""
    b = _baars_taper_factor(taper_db=10.0)
    assert b == pytest.approx(1.137, abs=0.005)


def test_baars_taper_factor_at_0db():
    """At 0 dB (uniform illumination), b ~= 1.029."""
    b = _baars_taper_factor(taper_db=0.0)
    # tau = 1.0, b = 1.269 - 0.566 + 0.534 - 0.208 = 1.029
    assert b == pytest.approx(1.029, abs=0.005)


def test_central_obscuration_factor_alma_12m():
    """ALMA 12m with 0.75m obscuration: epsilon=0.0625, factor ~= 0.992."""
    f = _central_obscuration_factor(diameter_m=12.0, obscuration_m=0.75)
    assert f == pytest.approx(0.992, abs=0.01)


def test_central_obscuration_factor_no_obscuration():
    """Zero obscuration: factor = 1.0 (no correction)."""
    f = _central_obscuration_factor(diameter_m=12.0, obscuration_m=0.0)
    assert f == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Primary beam and Gaussian offset
# ---------------------------------------------------------------------------
def test_primary_beam_at_232ghz():
    """PB FWHM at 232.994 GHz on ALMA 12m ~= 24.9 arcsec."""
    pb = _primary_beam_fwhm_arcsec(232.994e9)
    assert pb == pytest.approx(24.9, abs=0.5)


def test_gaussian_offset_at_pb_level_02():
    """Gaussian offset for FWHM=24.93 at 20% PB level ~= 19.0 arcsec."""
    off = _gaussian_offset_for_pb_level(24.93, 0.2)
    assert off == pytest.approx(19.0, abs=0.5)


def test_gaussian_offset_at_fwhm_half_power():
    """At pb_level=0.5 (half power), offset = FWHM/2."""
    off = _gaussian_offset_for_pb_level(10.0, 0.5)
    assert off == pytest.approx(5.0, abs=0.01)


def test_gaussian_offset_invalid_pb_level():
    """pb_level outside (0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        _gaussian_offset_for_pb_level(10.0, 0.0)
    with pytest.raises(ValueError):
        _gaussian_offset_for_pb_level(10.0, 1.5)


# ---------------------------------------------------------------------------
# uvtaper parser
# ---------------------------------------------------------------------------
WAVELENGTH_232GHZ = SPEED_OF_LIGHT_M_S / 232.994e9


def test_uvtaper_empty_input_returns_zero():
    """No taper -> 0.0 arcsec."""
    assert _parse_uvtaper_to_image_fwhm([], WAVELENGTH_232GHZ) == 0.0
    assert _parse_uvtaper_to_image_fwhm(None, WAVELENGTH_232GHZ) == 0.0
    assert _parse_uvtaper_to_image_fwhm("", WAVELENGTH_232GHZ) == 0.0


@pytest.mark.parametrize("input_str, expected_arcsec", [
    ("1arcsec",     1.0),
    ('0.5"',        0.5),
    ("100mas",      0.1),
    ("2arcsec",     2.0),
])
def test_uvtaper_image_domain(input_str, expected_arcsec):
    """Direct image-domain FWHM (arcsec or mas) passes through."""
    v = _parse_uvtaper_to_image_fwhm([input_str], WAVELENGTH_232GHZ)
    assert v == pytest.approx(expected_arcsec, rel=1e-6)


def test_uvtaper_klambda():
    """100 klambda taper -> Fourier-transformed image-domain FWHM."""
    v = _parse_uvtaper_to_image_fwhm(["100klambda"], WAVELENGTH_232GHZ)
    # FWHM_image_arcsec = 0.8825 / (100e3) * 206265 ~= 1.82
    expected = _UVTAPER_FWHM_PRODUCT / 100e3 * ARCSEC_PER_RAD
    assert v == pytest.approx(expected, rel=1e-4)
    assert v == pytest.approx(1.82, abs=0.01)


def test_uvtaper_mlambda():
    """1 Mlambda taper = 1000 klambda."""
    v = _parse_uvtaper_to_image_fwhm(["1mlambda"], WAVELENGTH_232GHZ)
    expected = _UVTAPER_FWHM_PRODUCT / 1e6 * ARCSEC_PER_RAD
    assert v == pytest.approx(expected, rel=1e-4)


def test_uvtaper_meters():
    """1000 m baseline length, converted via the supplied wavelength."""
    v = _parse_uvtaper_to_image_fwhm(["1000m"], WAVELENGTH_232GHZ)
    fwhm_uv_wavelengths = 1000.0 / WAVELENGTH_232GHZ
    expected = _UVTAPER_FWHM_PRODUCT / fwhm_uv_wavelengths * ARCSEC_PER_RAD
    assert v == pytest.approx(expected, rel=1e-4)


def test_uvtaper_elliptical_takes_geometric_mean():
    """Elliptical taper: AJISAI uses sqrt(major*minor) for cell sizing."""
    v = _parse_uvtaper_to_image_fwhm(
        ["2arcsec", "0.5arcsec", "30deg"], WAVELENGTH_232GHZ
    )
    assert v == pytest.approx(1.0, abs=1e-6)  # sqrt(2 * 0.5) = 1.0


def test_uvtaper_pa_only_ignored():
    """A PA-only entry without a size component yields 0 (no taper)."""
    v = _parse_uvtaper_to_image_fwhm(["30deg"], WAVELENGTH_232GHZ)
    assert v == 0.0


def test_uvtaper_unsupported_unit_raises():
    with pytest.raises(ValueError, match="Unsupported uvtaper format"):
        _parse_uvtaper_to_image_fwhm(["1furlong"], WAVELENGTH_232GHZ)


# ---------------------------------------------------------------------------
# End-to-end pickCellSize math (the golden V883 Ori case)
# ---------------------------------------------------------------------------
def test_pick_cell_imsize_math_v883ori_no_taper():
    """V883 Ori conditions without uvtaper must reproduce AU output exactly.

    Reference (analysisUtils.pickCellSize):
        cellsize = 0.0081 arcsec
        imsize   = [4800, 4800]
    """
    freq_hz = 232.994e9
    baseline_m = 3293.48
    wavelength_m = SPEED_OF_LIGHT_M_S / freq_hz

    beam = wavelength_m / baseline_m * ARCSEC_PER_RAD
    cell = _round_to_sig_figs(beam / 10, 2)

    pb_fwhm = _primary_beam_fwhm_arcsec(freq_hz)
    off = _gaussian_offset_for_pb_level(pb_fwhm, 0.2)
    raw = int(math.ceil(2 * off / cell))
    final = _smallest_5_smooth_at_least(raw)

    assert cell == pytest.approx(0.0081, abs=1e-4)
    assert final == 4800


def test_pick_cell_imsize_math_v883ori_with_taper_increases_cell():
    """Adding uvtaper must produce a larger cell size than without taper."""
    freq_hz = 232.994e9
    baseline_m = 3293.48
    wavelength_m = SPEED_OF_LIGHT_M_S / freq_hz

    beam = wavelength_m / baseline_m * ARCSEC_PER_RAD
    cell_no_taper = _round_to_sig_figs(beam / 10, 2)

    taper_fwhm = _parse_uvtaper_to_image_fwhm(["0.5arcsec"], wavelength_m)
    effective_beam = math.sqrt(beam ** 2 + taper_fwhm ** 2)
    cell_with_taper = _round_to_sig_figs(effective_beam / 10, 2)

    assert cell_with_taper > cell_no_taper
    # 0.5 arcsec taper on a 0.08 arcsec beam -> effective ~0.506 arcsec
    # -> cell ~0.051 (rounded to 2 sig figs)
    assert cell_with_taper == pytest.approx(0.051, abs=0.005)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------
def test_uvtaper_fwhm_product_constant():
    """4 ln 2 / pi ~= 0.8825 (Fourier transform of Gaussian)."""
    assert _UVTAPER_FWHM_PRODUCT == pytest.approx(0.8825, abs=1e-3)


def test_arcsec_per_rad_constant():
    """206264.806 arcsec per radian."""
    assert ARCSEC_PER_RAD == pytest.approx(206264.806, abs=0.01)


# ---------------------------------------------------------------------------
# icrs_to_j2000 / RA-Dec string parsing
#
# Regression: _max_pixel_world_coords (and analysisUtils' rad2radec) emit
# declination with '.' field separators, e.g. '-34.17.38.348'. The original
# icrs_to_j2000 only split on ':', so the CASA-format string collapsed to 4
# tokens and raised "Expected ... (6 parts)". See phase-shift crash on
# 2MASSJ15450634 (LupusI_13).
# ---------------------------------------------------------------------------
from ajisai.ms_utils import (  # noqa: E402
    _sexagesimal_to_colon,
    _split_radec,
    icrs_to_j2000,
)


@pytest.mark.parametrize("token, expected", [
    ("-34.17.38.348", "-34:17:38.348"),   # CASA dec, negative, fractional sec
    ("+34.17.38.348", "+34:17:38.348"),   # CASA dec, explicit positive
    ("34.17.38.348", "34:17:38.348"),     # CASA dec, unsigned
    ("-00.05.30.5", "-00:05:30.5"),       # small negative dec
    ("-34.17.38", "-34:17:38"),           # integer arcsec (3 fields)
    ("15:45:06.322", "15:45:06.322"),     # already ':'-separated -> unchanged
])
def test_sexagesimal_to_colon(token, expected):
    assert _sexagesimal_to_colon(token) == expected


def test_split_radec_casa_form():
    """CASA form: RA ':'-separated, Dec '.'-separated (2 whitespace tokens)."""
    ra, dec = _split_radec("15:45:06.322 -34.17.38.348")
    assert ra == "15:45:06.322"
    assert dec == "-34:17:38.348"


def test_split_radec_space_form():
    """Legacy documented form: 6 whitespace tokens."""
    ra, dec = _split_radec("15 45 06.322 -34 17 38.348")
    assert ra == "15:45:06.322"
    assert dec == "-34:17:38.348"


def test_split_radec_rejects_garbage():
    with pytest.raises(ValueError):
        _split_radec("not a coordinate")


def test_icrs_to_j2000_casa_format_regression():
    """The exact string that crashed the phase-shift step must now convert.

    ICRS and FK5 (J2000.0) share an origin, so the label is essentially
    unchanged; we assert the on-sky position is preserved to < 1 mas and the
    output keeps the CASA convention (RA ':'-sep, Dec '.'-sep).
    """
    pytest.importorskip("astropy")
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    crash_input = "15:45:06.322 -34.17.38.348"
    out = icrs_to_j2000(crash_input)

    # Output layout: RA uses ':', Dec uses '.' (CASA-parseable declination)
    ra_out, dec_out = out.split()
    assert ra_out.count(":") == 2
    assert ":" not in dec_out
    assert dec_out.startswith("-34.17.38")

    # On-sky position is preserved (frame relabel, not a physical move)
    c_in = SkyCoord(ra="15:45:06.322", dec="-34:17:38.348",
                    unit=(u.hourangle, u.deg), frame="icrs")
    c_out = SkyCoord(ra=ra_out.replace(":", " "),
                     dec=dec_out.replace(".", " ", 2),
                     unit=(u.hourangle, u.deg), frame="fk5")
    assert c_in.separation(c_out).arcsec < 1e-3


def test_icrs_to_j2000_accepts_space_form():
    """The legacy 6-token layout still works."""
    pytest.importorskip("astropy")
    out = icrs_to_j2000("15 45 06.322 -34 17 38.348")
    ra_out, dec_out = out.split()
    assert ra_out.count(":") == 2
    assert dec_out.startswith("-34.17.38")
