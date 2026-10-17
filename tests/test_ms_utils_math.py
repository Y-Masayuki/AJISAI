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

import numpy as np

from ajisai.ms_utils import (
    ALMA_DISH_DIAMETER_M,
    ARCSEC_PER_RAD,
    CELL_SAMPLES_PER_UMAX,
    SPEED_OF_LIGHT_M_S,
    _UVTAPER_FWHM_PRODUCT,
    _baars_taper_factor,
    _central_obscuration_factor,
    _floor_to_sig_figs,
    _gaussian_offset_for_pb_level,
    _parse_uvtaper_to_image_fwhm,
    _primary_beam_fwhm_arcsec,
    _round_to_sig_figs,
    _smallest_5_smooth_at_least,
    cell_from_umax,
    get_dish_diameter_m,
    max_uv_of_unflagged_rows,
    smallest_dish_diameter,
    uv_max_from_maxima,
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


# icrs_to_j2000 converts with casatools.measures (CASA's J2000, not astropy's
# FK5, which differs by 20-25 mas); its accuracy and output layout are tested
# against me.measure in tests/test_casa_frames.py.


# ---------------------------------------------------------------------------
# Coordinate strings: RA in hours, Dec in degrees whatever the separators
# (CASA reads a ':'-separated Dec as hours, so AJISAI parses them itself)
# ---------------------------------------------------------------------------
from ajisai.ms_utils import (  # noqa: E402
    SKY_FRAMES,
    angular_offset_mas,
    casa_direction_string,
    code_of_frame,
    convert_direction,
    direction_frames,
    format_dec,
    format_ra,
    frame_of_code,
    parse_dec,
    parse_ra,
    parse_ra_dec,
    rad_to_radec,
)

_DEG = math.pi / 180.0
_MAS = _DEG / 3600e3


@pytest.mark.parametrize("text", ["16:25:45.0", "16h25m45.0s", "16h 25m 45.0s", "16 25 45.0",
                                  "+16:25:45.0", "246.4375deg", "4.301145775383526rad"])
def test_parse_ra_forms(text):
    assert parse_ra(text) == pytest.approx(246.4375 * _DEG, abs=1e-12)


@pytest.mark.parametrize("text", ["-24:12:23.0", "-24.12.23.0", "-24d12m23.0s", "-24 12 23.0",
                                  "-024.12.23.0", "-24d 12m 23s", "-24.20638888888889deg"])
def test_parse_dec_forms_are_degrees(text):
    """'-24:12:23.0' is -24.2064 deg (CASA alone would read -363.1 deg -> Dec -3.1 deg)."""
    assert parse_dec(text) == pytest.approx(-(24 + 12 / 60 + 23 / 3600) * _DEG, abs=1e-12)


def test_parse_dec_sign_of_small_negative_values():
    assert parse_dec("-00:05:30.5") == pytest.approx(-(5 / 60 + 30.5 / 3600) * _DEG)
    assert parse_dec("+00.05.30.5") == pytest.approx((5 / 60 + 30.5 / 3600) * _DEG)
    assert parse_dec("00 05 30.5") > 0


def test_parse_ra_dec_pair():
    ra, dec = parse_ra_dec("16:25:45.0", "-24:12:23.0")
    assert math.degrees(ra) == pytest.approx(246.4375)
    assert math.degrees(dec) == pytest.approx(-24.2063888889)


@pytest.mark.parametrize("bad", ["24:00:00", "12:60:00", "12:00:60", "16.25.45.0",
                                 "246.4375", "-1:00:00", "nonsense", ""])
def test_parse_ra_rejects(bad):
    with pytest.raises(ValueError):
        parse_ra(bad)


@pytest.mark.parametrize("bad", ["-91:00:00", "90:00:01", "-24:60:00", "-24.2063",
                                 "-24:12", "2rad", "-95deg", "nonsense"])
def test_parse_dec_rejects(bad):
    with pytest.raises(ValueError):
        parse_dec(bad)


def test_format_ra_dec_round_trip_and_carry():
    for ra, dec in ((4.301145775, -0.422481186), (0.0, 0.0), (6.283185, 1.5707), (1.0, -1e-9)):
        r, d = format_ra(ra, 7), format_dec(dec, 7)
        assert ":" not in d and d.count(".") == 3 and d[0] in "+-"
        assert angular_offset_mas(parse_ra(r), parse_dec(d), ra % (2 * math.pi), dec) < 0.005
    # 59.99999996 s rounds up into the next minute and hour
    assert format_ra((15.0 * (23 + 59 / 60 + 59.99999996 / 3600)) * _DEG, 7) == "00:00:00.0000000"
    assert format_dec(-(29 + 59 / 60 + 59.99999996 / 3600) * _DEG, 7) == "-30.00.00.0000000"
    assert format_dec(0.5 * _DEG, 0, sep=":") == "+00:30:00"


def test_rad_to_radec_writes_dec_with_dots():
    s = rad_to_radec(246.4375 * _DEG, -24.2063888889 * _DEG)
    ra, dec = s.split()
    assert ra == "16:25:45.00000"
    assert dec == "-24.12.23.00000"
    deg = rad_to_radec(-0.5 * _DEG, 10 * _DEG, hms_dms=False, precision=3)
    assert deg == "359.500 10.000"


def test_casa_direction_string_is_radians_with_single_spaces():
    s = casa_direction_string("ICRS", 4.301145775383526, -1e-9)
    assert s == "ICRS 4.301145775383526rad -0.000000001000000rad"
    assert len(s.split(" ")) == 3 and "e" not in s.split(" ")[2]


def test_convert_direction_same_frame_needs_no_casa_and_wraps_ra():
    assert convert_direction(-2.0, 0.3, "J2000", "j2000") == (pytest.approx(2 * math.pi - 2.0), 0.3)
    assert SKY_FRAMES == ("ICRS", "J2000")
    with pytest.raises(ValueError):
        convert_direction(1.0, 0.3, "B1950", "ICRS")
    with pytest.raises(ValueError):
        convert_direction(1.0, 0.3, "ICRS", "GALACTIC")


def test_angular_offset_mas():
    assert angular_offset_mas(1.0, 0.0, 1.0 + _MAS, 0.0) == pytest.approx(1.0, rel=1e-6)
    assert angular_offset_mas(1.0, math.radians(60), 1.0 + 2 * _MAS, math.radians(60)) == \
        pytest.approx(1.0, rel=1e-6)
    assert angular_offset_mas(0.0, 0.0, 2 * math.pi - _MAS, 0.0) == pytest.approx(1.0, rel=1e-6)


# MEASINFO of the FIELD direction columns of a real ALMA MS (variable reference)
_ALMA_MEASINFO = {
    "type": "direction", "VarRefCol": "PhaseDir_Ref",
    "TabRefTypes": ["J2000", "JMEAN", "JTRUE", "APP", "B1950", "B1950_VLA", "BMEAN", "BTRUE",
                    "GALACTIC", "HADEC", "AZEL", "AZELSW", "AZELNE", "AZELGEO", "AZELSWGEO",
                    "AZELNEGEO", "JNAT", "ECLIPTIC", "MECLIPTIC", "TECLIPTIC", "SUPERGAL",
                    "ITRF", "TOPO", "ICRS", "MERCURY", "VENUS", "MARS", "JUPITER", "SATURN",
                    "URANUS", "NEPTUNE", "PLUTO", "SUN", "MOON", "COMET"],
    "TabRefCodes": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 12, 13, 12, 14, 15, 16,
                             17, 18, 19, 20, 21, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
                            dtype=np.uint64),
}


def test_frame_codes_come_from_the_measinfo():
    assert frame_of_code(_ALMA_MEASINFO, 21) == "ICRS"
    assert frame_of_code(_ALMA_MEASINFO, 0) == "J2000"
    assert frame_of_code(_ALMA_MEASINFO, 10) == "AZEL"      # shared with AZELNE: first name
    assert code_of_frame(_ALMA_MEASINFO, "ICRS") == 21
    assert code_of_frame(_ALMA_MEASINFO, "J2000") == 0
    other = dict(_ALMA_MEASINFO, TabRefTypes=["ICRS", "J2000"], TabRefCodes=[0, 21])
    assert frame_of_code(other, 0) == "ICRS" and code_of_frame(other, "J2000") == 21
    with pytest.raises(ValueError):
        frame_of_code(_ALMA_MEASINFO, 99)
    with pytest.raises(ValueError):
        code_of_frame({"type": "direction", "Ref": "ICRS"}, "ICRS")


def test_direction_frames_variable_and_fixed():
    assert direction_frames(_ALMA_MEASINFO, ref_codes=np.array([21, 0, 21])) == \
        ["ICRS", "J2000", "ICRS"]
    assert direction_frames({"type": "direction", "Ref": "ICRS"}, nrow=2) == ["ICRS", "ICRS"]
    assert direction_frames({"type": "direction"}, nrow=1) == ["J2000"]
    with pytest.raises(ValueError):
        direction_frames(_ALMA_MEASINFO)


# ---------------------------------------------------------------------------
# Cell from u_max (default rule since v0.3)
# ---------------------------------------------------------------------------
def test_cell_from_umax_definition():
    """cell = 1/(5 u_max) radians, in arcsec."""
    assert CELL_SAMPLES_PER_UMAX == 5.0
    u_max = 11_596e3  # J16011549 TM1 spw19: 11 596 klambda
    cell = cell_from_umax(u_max)
    assert cell == pytest.approx(ARCSEC_PER_RAD / (5 * u_max))
    assert cell * 1e3 == pytest.approx(3.557, abs=0.001)   # 3.557 mas (P13 I2)


def test_cell_from_umax_samples_argument():
    assert cell_from_umax(1e6, samples=10) == pytest.approx(cell_from_umax(1e6) / 2)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_cell_from_umax_rejects_bad_umax(bad):
    with pytest.raises(ValueError):
        cell_from_umax(bad)


@pytest.mark.parametrize("value, expected", [
    (0.0035579, 0.003557),     # truncated, never rounded up
    (0.0062, 0.0062),          # exact decimals survive
    (0.35385, 0.3538),
    (123.456, 123.4),
])
def test_floor_to_sig_figs(value, expected):
    assert _floor_to_sig_figs(value, 4) == pytest.approx(expected, rel=1e-12)
    assert _floor_to_sig_figs(value, 4) <= value


def test_max_uv_of_unflagged_rows_ignores_flagged_rows():
    uvw = np.array([[100.0, 3000.0, 400.0],
                    [0.0, 4000.0, 300.0],
                    [5.0, 0.0, 0.0]])
    flag = np.zeros((2, 4, 3), bool)
    flag[:, :, 1] = True              # the longest row (5 km) is fully flagged
    flag[:, 3, :] = True              # last channel flagged everywhere
    m, chan_ok = max_uv_of_unflagged_rows(uvw, flag)
    assert m == pytest.approx(500.0)  # hypot(400, 300)
    assert chan_ok.tolist() == [True, True, True, False]


def test_max_uv_of_unflagged_rows_flag_row_and_all_flagged():
    uvw = np.array([[100.0, 200.0], [0.0, 0.0], [0.0, 0.0]])
    flag = np.zeros((1, 2, 2), bool)
    m, _ = max_uv_of_unflagged_rows(uvw, flag, flag_row=np.array([False, True]))
    assert m == pytest.approx(100.0)
    m, chan_ok = max_uv_of_unflagged_rows(uvw, np.ones((1, 2, 2), bool))
    assert m == 0.0 and not chan_ok.any()


def test_uv_max_from_maxima_uses_each_spw_highest_frequency():
    """u_max = max over spw of |uv|_max * f_max / c: the high-frequency spw can win
    with a shorter baseline."""
    res = uv_max_from_maxima({0: 1000.0, 1: 990.0}, {0: 224e9, 1: 241e9})
    assert res["spw"] == 1
    assert res["u_max_lambda"] == pytest.approx(990.0 * 241e9 / SPEED_OF_LIGHT_M_S)
    assert res["baseline_m"] == 990.0 and res["freq_hz"] == 241e9


def test_uv_max_from_maxima_all_flagged_raises():
    with pytest.raises(ValueError):
        uv_max_from_maxima({0: 0.0}, {0: 230e9})
    with pytest.raises(ValueError):
        uv_max_from_maxima({0: 100.0}, {})


# ---------------------------------------------------------------------------
# Dish diameter from ANTENNA (smallest dish -> largest primary beam)
# ---------------------------------------------------------------------------
def _fake_reader(tables):
    calls = []

    def reader(table_path, column):
        calls.append((table_path, column))
        if (table_path, column) not in tables:
            raise KeyError(column)
        return np.asarray(tables[(table_path, column)])
    reader.calls = calls
    return reader


def test_dish_diameter_from_antenna_takes_smallest():
    reader = _fake_reader({("x.ms/ANTENNA", "DISH_DIAMETER"): [12.0, 12.0, 7.0, 12.0]})
    d = get_dish_diameter_m("x.ms", table_reader=reader)
    assert reader.calls == [("x.ms/ANTENNA", "DISH_DIAMETER")]
    assert d["diameter_m"] == 7.0
    assert d["source"] == "ANTENNA/DISH_DIAMETER"
    assert d["unique_diameters_m"] == [7.0, 12.0]
    assert d["n_antennas"] == 4


def test_dish_diameter_falls_back_to_12m_without_column():
    d = get_dish_diameter_m("x.ms", table_reader=_fake_reader({}))
    assert d["diameter_m"] == ALMA_DISH_DIAMETER_M == 12.0
    assert d["source"] == "fallback"


def test_dish_diameter_ignores_zero_and_nan():
    reader = _fake_reader({("x.ms/ANTENNA", "DISH_DIAMETER"): [0.0, float("nan"), 25.0]})
    assert get_dish_diameter_m("x.ms", table_reader=reader)["diameter_m"] == 25.0
    assert smallest_dish_diameter([0.0, -1.0])["source"] == "fallback"


def test_smaller_dish_gives_larger_primary_beam():
    """7 m dish: the primary beam (and so the imaged field) is 12/7 times larger."""
    pb12 = _primary_beam_fwhm_arcsec(230e9, diameter_m=12.0)
    pb7 = _primary_beam_fwhm_arcsec(230e9, diameter_m=7.0)
    assert pb7 / pb12 == pytest.approx(12.0 / 7.0, rel=0.01)
