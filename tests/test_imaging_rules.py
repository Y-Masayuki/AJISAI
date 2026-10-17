"""
Pure tests of the v0.3 imaging rules in ``ajisai.core``.

No CASA is needed: the functions tested here take numbers or numpy arrays.
They pin the product naming (``.image.tt0`` for mtmfs), the deconvolver
choice, the multiscale scale ladder bounded by the mask region, the
signal-free noise image, the post-tclean / post-ft consistency checks, and
the CLEAN-depth stop rule of the final image decided on 2026-10-15
(checked against the V1094 Sco test numbers).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ajisai.core import (
    _cell_to_arcsec,
    _deconvolver_kwargs,
    _disk_fits,
    _image_name,
    _imsize_list,
    _model_images,
    _noise_stokes,
    _relabel_needed,
    aperture_flux_terms,
    final_depth_decision,
    imaging_consistency_problems,
    model_zero_channels,
    multiscale_scales,
    psf_volume_within,
    scale_ladder,
)


# ---------------------------------------------------------------------------
# Product names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("product, expected", [
    ("image", "b.image"),
    ("model", "b.model"),
    ("residual", "b.residual"),
    ("psf", "b.psf"),
    ("sumwt", "b.sumwt"),
    ("mask", "b.mask"),
])
def test_image_name_single_term(product, expected):
    assert _image_name("b", 1, product) == expected


@pytest.mark.parametrize("product, expected", [
    ("image", "b.image.tt0"),
    ("model", "b.model.tt0"),
    ("residual", "b.residual.tt0"),
    ("psf", "b.psf.tt0"),
    ("sumwt", "b.sumwt.tt0"),
    ("pb", "b.pb.tt0"),
    # products without a Taylor suffix
    ("mask", "b.mask"),
    ("alpha", "b.alpha"),
    ("alpha.error", "b.alpha.error"),
])
def test_image_name_mtmfs(product, expected):
    assert _image_name("b", 2, product) == expected


def test_image_name_default_is_restored_tt0_and_accepts_paths(tmp_path):
    assert _image_name(tmp_path / "clean_sc1", 3) == str(tmp_path / "clean_sc1") + ".image.tt0"
    assert _image_name("b", 2, "image", term=1) == "b.image.tt1"


def test_model_images_one_per_taylor_term():
    assert _model_images("b", 1) == ["b.model"]
    assert _model_images("b", 2) == ["b.model.tt0", "b.model.tt1"]


# ---------------------------------------------------------------------------
# Deconvolver
# ---------------------------------------------------------------------------
def test_mtmfs_when_nterms_above_one_keeps_scales():
    kw = _deconvolver_kwargs("multiscale", 2, [0, 13, 39])
    assert kw == {"deconvolver": "mtmfs", "nterms": 2, "scales": [0, 13, 39]}


def test_mtmfs_from_hogbom_uses_point_scale_only():
    assert _deconvolver_kwargs("hogbom", 2, [0, 5]) == {
        "deconvolver": "mtmfs", "nterms": 2, "scales": [0]}


def test_single_term_multiscale_and_hogbom():
    assert _deconvolver_kwargs("multiscale", 1, [0, 5, 15]) == {
        "deconvolver": "multiscale", "scales": [0, 5, 15]}
    assert _deconvolver_kwargs("hogbom", 1, [0, 5]) == {"deconvolver": "hogbom"}


# ---------------------------------------------------------------------------
# Cell and imsize parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell, arcsec", [
    ("0.0062arcsec", 0.0062),
    ("6.2mas", 0.0062),
    ('0.1"', 0.1),
    (0.25, 0.25),
    (["0.03arcsec"], 0.03),
    ("0.001deg", 3.6),
])
def test_cell_to_arcsec(cell, arcsec):
    assert _cell_to_arcsec(cell) == pytest.approx(arcsec)


def test_cell_to_arcsec_rejects_garbage():
    with pytest.raises(ValueError):
        _cell_to_arcsec("small")


def test_imsize_list():
    assert _imsize_list(2700) == [2700, 2700]
    assert _imsize_list((4320,)) == [4320, 4320]
    assert _imsize_list((512, 256)) == [512, 256]
    with pytest.raises(ValueError):
        _imsize_list((1, 2, 3))


# ---------------------------------------------------------------------------
# Multiscale scales
# ---------------------------------------------------------------------------
def test_scale_ladder_factor_three_v1094():
    """V1094 Sco test: beam 13.07 pix -> 0, 13, 39, 118, 353 up to L/2 ~ 400 pix."""
    assert scale_ladder(13.07, 400.0) == [0, 13, 39, 118, 353]


def test_scale_ladder_no_beam():
    assert scale_ladder(0.0, 100.0) == [0]


def _disk(n, radius):
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2.0
    return np.hypot(xx - c, yy - c) <= radius


def test_disk_fits_inside_round_region():
    region = _disk(101, 30.0)
    assert _disk_fits(region, 25)
    assert _disk_fits(region, 30)
    assert not _disk_fits(region, 32)
    assert not _disk_fits(np.zeros((10, 10), bool), 1)


def test_multiscale_scales_bounded_by_region_half_size():
    """Round region of radius 30 pix (L = 60): with b = 3 -> 0, 3, 9, 27 (81 > L/2)."""
    scales, reason = multiscale_scales(3.0, _disk(101, 30.0))
    assert scales == [0, 3, 9, 27]
    assert "fits inside the mask region" in reason


def test_multiscale_scales_thin_region_limits_scale():
    """A 9-pixel-wide strip cannot hold a scale of radius 9 even though it is long."""
    region = np.zeros((200, 200), bool)
    region[96:105, 10:190] = True
    scales, _ = multiscale_scales(3.0, region)
    assert scales == [0, 3]


def test_multiscale_scales_without_region_is_legacy_0_1_3_beams():
    scales, reason = multiscale_scales(5.0, None)
    assert scales == [0, 5, 15]
    assert "no mask region" in reason


def test_multiscale_scales_region_smaller_than_beam():
    scales, _ = multiscale_scales(10.0, _disk(41, 4.0))
    assert scales == [0]


# ---------------------------------------------------------------------------
# Signal-free noise image
# ---------------------------------------------------------------------------
def test_noise_stokes_linear_and_circular():
    assert _noise_stokes([[9, 12]]) == "Q"            # XX, YY -> (XX-YY)/2
    assert _noise_stokes([[9, 10, 11, 12]]) == "Q"    # full pol, linear
    assert _noise_stokes([[5, 8]]) == "V"             # RR, LL -> (RR-LL)/2
    assert _noise_stokes([[9, 12], [9, 12]]) == "Q"


def test_noise_stokes_without_parallel_pair():
    assert _noise_stokes([[9]]) is None
    assert _noise_stokes([[9, 12], [9]]) is None      # every setup needs the pair
    assert _noise_stokes([]) is None


# ---------------------------------------------------------------------------
# Post-tclean consistency
# ---------------------------------------------------------------------------
def test_consistency_ok_for_lsrk_image_of_topo_data():
    """CASA 6.5.1 image of a 2-spw simulation: LSRK band 15 MHz below the TOPO data."""
    problems = imaging_consistency_problems(
        ref_freq_hz=231.391025e9, width_hz=3.06231486e9, sumwt=3.5e9,
        data_fmin_hz=229.875e9, data_fmax_hz=232.9375e9)
    assert problems == []


def test_consistency_ok_with_reffreq_off_centre():
    """With reffreq, CASA keeps ref = reffreq and widens the band symmetrically."""
    problems = imaging_consistency_problems(
        ref_freq_hz=230.1e9, width_hz=5.644365807e9, sumwt=1.0e10,
        data_fmin_hz=229.875e9, data_fmax_hz=232.9375e9)
    assert problems == []


def test_consistency_detects_dropped_channels():
    problems = imaging_consistency_problems(
        ref_freq_hz=230.5e9, width_hz=1.0e9, sumwt=1.0,
        data_fmin_hz=229.875e9, data_fmax_hz=232.9375e9)
    assert len(problems) == 1 and "does not cover" in problems[0]


@pytest.mark.parametrize("sumwt", [0.0, -1.0, float("nan")])
def test_consistency_detects_empty_gridding(sumwt):
    problems = imaging_consistency_problems(230e9, 2e9, sumwt, 229.5e9, 230.5e9)
    assert any("sumwt" in p for p in problems)


def test_consistency_detects_bad_frequency_axis():
    problems = imaging_consistency_problems(float("nan"), 2e9, 1.0, 229.5e9, 230.5e9)
    assert any("reference frequency" in p for p in problems)


def test_model_zero_channels():
    absmax = [0.06, 0.0, 0.05, 0.0, float("nan")]
    n_unflagged = [10, 10, 0, 0, 3]
    # channel 1: unflagged data, zero model -> bad; channel 3: all flagged -> ignored;
    # channel 4: non-finite model -> bad
    assert model_zero_channels(absmax, n_unflagged) == [1, 4]
    assert model_zero_channels([0.1, 0.2], [5, 5]) == []


def test_relabel_needed():
    assert _relabel_needed([np.array([0, 21])], None)
    assert not _relabel_needed([np.array([21, 21])], "ICRS")
    assert _relabel_needed([], "J2000")
    assert not _relabel_needed([], None)


# ---------------------------------------------------------------------------
# Aperture flux of the final image
# ---------------------------------------------------------------------------
def _gauss(n, sigma_pix, amp=1.0):
    yy, xx = np.mgrid[:n, :n]
    c = n // 2
    return amp * np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma_pix ** 2))


def test_psf_volume_of_gaussian_is_beam_area():
    sigma = 4.0
    psf = _gauss(129, sigma)
    # Whole volume of a Gaussian = 2 pi sigma^2 pixels.
    assert psf_volume_within(psf, 60.0) == pytest.approx(2 * math.pi * sigma ** 2, rel=1e-6)
    # Within 1 sigma: 2 pi sigma^2 (1 - e^-1/2), up to pixelisation.
    assert psf_volume_within(psf, sigma) == pytest.approx(
        2 * math.pi * sigma ** 2 * (1 - math.exp(-0.5)), rel=0.05)


def test_aperture_flux_terms_recovers_residual_flux():
    """A residual that is 2 mJy of point flux convolved with the PSF gives F_unc = 2 mJy."""
    n, sigma = 129, 3.0
    psf = _gauss(n, sigma)
    model = np.zeros((n, n))
    model[n // 2, n // 2] = 0.010                      # 10 mJy CLEAN component
    residual = 0.002 * psf                             # 2 mJy left uncleaned (Jy/beam)
    aperture = _disk(n, 40.0)
    fwhm_pix = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma
    cell = 0.01
    t = aperture_flux_terms(model, residual, psf, aperture, cell,
                            fwhm_pix * cell, fwhm_pix * cell, sigma_jy=1e-5)
    assert t["f_model_jy"] == pytest.approx(0.010)
    assert t["f_unc_jy"] == pytest.approx(0.002, rel=1e-3)
    assert t["f_corr_jy"] == pytest.approx(0.012, rel=1e-4)
    # Gaussian PSF = Gaussian CLEAN beam: no restored-flux bias.
    assert t["omega_clean_pix"] == pytest.approx(t["omega_r_pix"], rel=1e-3)
    assert abs(t["restored_bias_jy"]) < 1e-6
    # sigma_F = sigma sqrt(N_pix / Omega_R)
    assert t["sigma_f_jy"] == pytest.approx(1e-5 * math.sqrt(t["aperture_npix"] / t["omega_r_pix"]))


def test_aperture_flux_terms_empty_aperture_raises():
    with pytest.raises(ValueError):
        aperture_flux_terms(np.zeros((5, 5)), np.zeros((5, 5)), _gauss(5, 1.0),
                            np.zeros((5, 5), bool), 0.1, 0.3, 0.3, 1e-5)


# ---------------------------------------------------------------------------
# CLEAN-depth stop rule of the final image (decided 2026-10-15)
# ---------------------------------------------------------------------------
# V1094 Sco test (DESIGN_REVIEW P13-6): sigma_F = 1.54 mJy; mJy values.
V1094 = [  # (nsigma, F_model, F_unc)
    (3.0, 215.4, 9.9),
    (2.0, 219.1, 6.6),
    (1.0, 221.9, 4.0),
]


def test_depth_rule_v1094_continues_after_3_sigma():
    d = final_depth_decision(215.4e-3, 9.9e-3, 1.54e-3)
    assert d["f_corr_jy"] == pytest.approx(225.3e-3)
    assert d["width_jy"] == pytest.approx(4.62e-3)        # max(3 x 1.54, 1% x 225) mJy
    assert not d["f_unc_within_width"]
    assert d["f_corr_change_jy"] is None
    assert not d["stop"]


def test_depth_rule_v1094_stops_at_2_sigma_on_f_corr_change():
    """F_corr changes by 0.4 mJy from 3 to 2 sigma (< 4.6 mJy): stop at 2 sigma."""
    prev = None
    stopped_at = None
    for nsig, f_model, f_unc in V1094:
        d = final_depth_decision(f_model * 1e-3, f_unc * 1e-3, 1.54e-3, prev)
        prev = d["f_corr_jy"]
        if d["stop"]:
            stopped_at = nsig
            break
    assert stopped_at == 2.0
    assert d["f_corr_change_within_width"] and not d["f_unc_within_width"]
    assert "changed" in d["reason"]


def test_depth_rule_stops_when_f_unc_within_width():
    d = final_depth_decision(221.9e-3, 4.0e-3, 1.54e-3)   # the 1 sigma state of V1094
    assert d["f_unc_within_width"] and d["stop"]


def test_depth_rule_one_percent_floor():
    """For a bright source the 1 % of F_corr dominates 3 sigma_F."""
    d = final_depth_decision(1.0, 0.008, 1e-4)
    assert d["width_jy"] == pytest.approx(0.01008)
    assert d["stop"]


def test_depth_rule_negative_f_unc_uses_absolute_value():
    d = final_depth_decision(0.100, -0.010, 1e-4)
    assert not d["f_unc_within_width"]


def test_restored_bias_v1094_ratio():
    """Omega_R / Omega_clean = 458.6 / 175: restored image over-counts F_unc by 1.62x."""
    ratio = 458.6 / 175.0 - 1.0
    assert ratio == pytest.approx(1.62, abs=0.01)
    assert 6.6 * ratio == pytest.approx(10.7, abs=0.1)     # +11 mJy at 2 sigma
