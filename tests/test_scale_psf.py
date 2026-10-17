"""
Multiscale scales the uv coverage cannot see, and the empty-model stop (no CASA needed).

Regression of the ASEBI v0.3 run of V1094 Sco (EB uid___A002_Xc72427_X2f71,
8.5 km baselines, cell 6.2 mas, 6000^2): the scales derived from the beam and
the mask region were [0, 8, 23, 70, 209, 628]. The scale-PSF of 628 pix has
its largest magnitude at a negative sidelobe, CASA's multiscale minor cycle
refused to start ("As Peak of PSF is negative, you should setscales again with
a smaller scale size"), the initial CLEAN ended with an all-zero model, and
the run failed only later, in the model prediction. On that PSF (CASA 6.7.4)
the boundary is 274 pix (kept) / 276 pix (refused by CASA), as predicted by
:func:`ajisai.core.scale_psf_response`.

Here: CASA's component shape, the scale-PSF and its "peaks at the component's
own position" rule on PSFs made the way tclean makes them, the scales that
reach tclean, and the early stop with a reason when a CLEAN leaves no model.
"""
from __future__ import annotations

import numpy as np
import pytest

from ajisai import ImagingConfig, core
from ajisai.core import (
    _casa_spheroidal,
    casa_scale_kernel,
    empty_model_problem,
    psf_supported_scales,
    scale_psf_response,
)
from tests.test_selfcal_bookkeeping import _pipeline


# ---------------------------------------------------------------------------
# PSFs as tclean makes them
# ---------------------------------------------------------------------------
def _psf_from_weights(n, k_min, k_max, pad=1.0):
    """PSF of uniform weights on the uv annulus k_min <= |k| <= k_max (in cells of an n grid).

    ``pad > 1``: gridded on a ``pad`` times larger grid and cropped to n x n,
    as tclean does (padding 1.2), so the PSF is no longer the transform of
    non-negative weights on the image grid.
    """
    m = int(round(n * pad / 2)) * 2
    f = np.fft.fftfreq(m) * m
    k = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    w = ((k >= k_min * pad) & (k <= k_max * pad)).astype(float)
    full = np.fft.fftshift(np.real(np.fft.ifft2(w)))
    c = m // 2
    psf = full[c - n // 2:c + n // 2, c - n // 2:c + n // 2]
    return psf / psf[n // 2, n // 2], w


# ---------------------------------------------------------------------------
# CASA's multiscale component
# ---------------------------------------------------------------------------
def test_casa_scale_kernel_matches_casa_component():
    """Profile of a 40-pixel component as CASA 6.7.4 writes it into the model (tclean niter=1)."""
    k = casa_scale_kernel(40, (200, 200))
    row = k[100, 100:141] / k[100, 100]
    for r, casa in ((0, 1.0), (10, 0.734366), (20, 0.270799), (30, 0.035890),
                    (38, 0.000935), (39, 0.000317), (40, 0.0)):
        assert row[r] == pytest.approx(casa, abs=2e-6)


def test_casa_scale_kernel_unit_volume_support_and_point_scale():
    k = casa_scale_kernel(7.5, (64, 63))
    assert k.sum() == pytest.approx(1.0)
    assert np.unravel_index(np.argmax(k), k.shape) == (32, 31)
    yy, xx = np.nonzero(k)
    assert np.hypot(yy - 32, xx - 31).max() < 7.5          # support radius = scale
    p = casa_scale_kernel(0, (8, 8))
    assert p[4, 4] == 1.0 and p.sum() == 1.0
    assert _casa_spheroidal([0.0, 1.0, 1.5]).tolist() == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Scale-PSF
# ---------------------------------------------------------------------------
def test_scale_psf_centre_is_the_sampled_power_of_the_scale():
    """centre = sum w |FT(scale)|^2 / sum w; on a PSF of non-negative weights every scale peaks there."""
    psf, w = _psf_from_weights(128, 4, 30)
    response = scale_psf_response(psf, [0, 3, 9, 27])
    for r in response:
        ft = np.fft.fft2(np.fft.ifftshift(casa_scale_kernel(r["scale"], psf.shape)))
        sampled = float((w * np.abs(ft) ** 2).sum() / w.sum())
        assert r["centre"] == pytest.approx(sampled, rel=1e-9)
        assert r["ok"] and r["max_abs_at"] == [64, 64]


def test_scale_psf_rule_drops_scales_with_a_negative_sidelobe_peak():
    """No spacing below 4 cells: 60 and 81 pix have their largest scale-PSF value at a negative
    sidelobe, the case in which CASA's minor cycle refuses to start (V1094 Sco, 628 pix)."""
    psf, _ = _psf_from_weights(256, 4, 40, pad=1.2)
    kept, response = psf_supported_scales(psf, [0, 3, 9, 27, 40, 60, 81])
    assert kept == [0, 3, 9, 27, 40]
    by = {r["scale"]: r for r in response}
    assert by[40]["max_abs"] == pytest.approx(by[40]["centre"])
    for s in (60, 81):
        assert by[s]["max_abs"] < 0 < by[s]["centre"] and not by[s]["ok"]


def test_scale_psf_rule_does_not_depend_on_the_sign_of_the_sidelobe():
    """No spacing below 8 cells: at 40 pix the largest value is a positive sidelobe, 15 x the
    scale's own response; CASA would run but normalise the scale by that sidelobe. Dropped too."""
    psf, _ = _psf_from_weights(256, 8, 40, pad=1.2)
    kept, response = psf_supported_scales(psf, [0, 3, 9, 27, 40])
    assert kept == [0, 3, 9, 27]
    r40 = response[-1]
    assert r40["max_abs"] > 10 * r40["centre"] > 0 and not r40["ok"]


def test_point_scale_is_always_kept():
    kept, _ = psf_supported_scales(-np.ones((16, 16)), [0])
    assert kept == [0]


# ---------------------------------------------------------------------------
# Empty CLEAN model
# ---------------------------------------------------------------------------
def _planes(n=64, peak=1.95e-3):
    residual = np.zeros((n, n))
    residual[30, 34] = peak
    mask = np.zeros((n, n))
    mask[20:44, 20:44] = 1.0
    return residual, mask


def test_empty_model_problem_none_when_the_model_has_a_component():
    residual, mask = _planes()
    model = np.zeros_like(residual)
    model[30, 34] = 1e-4
    assert empty_model_problem([model], residual, mask, 8.5e-5) is None
    # mtmfs: any Taylor term
    assert empty_model_problem([np.zeros_like(model), model], residual, mask, 8.5e-5) is None


def test_empty_model_with_emission_above_threshold_means_minor_cycle_did_not_run():
    """V1094 Sco X2f71: dirty peak 1.95 mJy/beam inside the mask, threshold 85.3 uJy."""
    residual, mask = _planes()
    msg = empty_model_problem([np.zeros_like(residual)], residual, mask, 8.5297e-5,
                              scales=[0, 8, 23, 70, 209, 628], casa_log="/w/casa.log")
    assert "minor cycle did not run" in msg
    assert "22.9 x the threshold" in msg
    assert "As Peak of PSF is negative" in msg and "[0, 8, 23, 70, 209, 628]" in msg
    assert "/w/casa.log" in msg


def test_empty_model_below_threshold_means_no_emission():
    residual, mask = _planes(peak=5e-5)
    msg = empty_model_problem([np.zeros_like(residual)], residual, mask, 8.5e-5)
    assert "not above the threshold" in msg and "no emission" in msg


def test_empty_model_mask_outside_emission_and_empty_mask():
    residual, mask = _planes()
    residual[30, 34] = 0.0
    residual[2, 2] = 1e-2                       # bright, but outside the mask
    msg = empty_model_problem([np.zeros_like(residual)], residual, mask, 8.5e-5)
    assert "not above the threshold" in msg
    assert "minor cycle did not run" in empty_model_problem(
        [np.zeros_like(residual)], residual, None, 8.5e-5)          # no mask: whole image
    assert "no pixel to CLEAN" in empty_model_problem(
        [np.zeros_like(residual)], residual, np.zeros_like(mask), 8.5e-5)


# ---------------------------------------------------------------------------
# In the pipeline: the scales that reach tclean
# ---------------------------------------------------------------------------
def _disk(n, radius):
    yy, xx = np.mgrid[:n, :n]
    return np.hypot(xx - n // 2, yy - n // 2) <= radius


def _scales_pipeline(tmp_path, monkeypatch, psf, imaging):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch, imaging=imaging)
    aj._derived.update({"imsize": list(psf.shape), "cell": "0.05arcsec", "cell_arcsec": 0.05})
    # beam 3 pix (0.15")
    aj.justification["derived"]["dirty_image"] = {"bmaj_arcsec": 0.15, "bmin_arcsec": 0.15}
    aj._user_mask_region = lambda: _disk(psf.shape[0], 120)
    reads = []

    def read(path):
        reads.append(str(path))
        assert str(path).endswith("intermediate/dirty.psf")
        return psf
    monkeypatch.setattr(core, "_read_image_plane", read)
    aj._check_imaging_consistency = lambda base, nterms, **kw: None
    return aj, tasks, reads


def test_default_scales_are_0_1_3_beams_whatever_the_mask(tmp_path, monkeypatch):
    """Without ImagingConfig.scales the v0.2 rule [0, 1, 3] x beam is used (beam 3 pix), not the
    mask-bounded ladder (which would give 0, 3, 9, 27, 81 here); the scale-PSF check still runs."""
    psf, _ = _psf_from_weights(256, 4, 40, pad=1.2)
    aj, tasks, reads = _scales_pipeline(
        tmp_path, monkeypatch, psf,
        ImagingConfig(mask_mode="user", user_mask="/data/region.crtf"))
    aj._compute_scales()
    rec = aj.justification["derived"]["scales"]
    assert rec["candidates"] == [0, 3, 9] and rec["value"] == [0, 3, 9]
    assert rec["psf_check"]["dropped"] == [] and "[0, 1, 3] x beam" in rec["reason"]


def test_given_scales_that_the_data_cannot_see_never_reach_tclean(tmp_path, monkeypatch):
    """The V1094 Sco failure: scales 0, 3, 9, 27, 81 pix (a ladder up to the mask size); 81 has a
    negative scale-PSF peak and is dropped with a warning before tclean."""
    psf, _ = _psf_from_weights(256, 4, 40, pad=1.2)
    aj, tasks, reads = _scales_pipeline(
        tmp_path, monkeypatch, psf,
        ImagingConfig(mask_mode="user", user_mask="/data/region.crtf", scales=(0, 3, 9, 27, 81)))
    with pytest.warns(UserWarning, match=r"scales \[81\] dropped"):
        aj._compute_scales()
    rec = aj.justification["derived"]["scales"]
    assert rec["candidates"] == [0, 3, 9, 27, 81]
    assert rec["value"] == [0, 3, 9, 27] and aj._derived["scales"] == [0, 3, 9, 27]
    assert rec["psf_check"]["dropped"] == [81]
    assert rec["psf_check"]["psf"].endswith("intermediate/dirty.psf")
    assert "[81] dropped" in rec["reason"]
    aj._tclean("a.ms", "clean_iter0", niter=100000, threshold_jy=8.5e-5,
               mask_kwargs=aj._build_usemask_kwargs())
    kw = tasks.named("tclean")[0]
    assert kw["deconvolver"] == "multiscale" and kw["scales"] == [0, 3, 9, 27]
    assert kw["mask"] == "/data/region.crtf" and kw["usemask"] == "user"


def test_given_scales_that_the_data_cannot_see_are_dropped_with_a_warning(tmp_path, monkeypatch):
    psf, _ = _psf_from_weights(256, 4, 40, pad=1.2)
    aj, tasks, reads = _scales_pipeline(tmp_path, monkeypatch, psf,
                                        ImagingConfig(nterms=2, scales=(0, 9, 27, 60)))
    with pytest.warns(UserWarning, match=r"scales \[60\] dropped"):
        aj._compute_scales()
    assert aj._derived["scales"] == [0, 9, 27]
    aj._tclean("a.ms", "img", niter=10, threshold_jy=1e-4, mask_kwargs={})
    kw = tasks.named("tclean")[0]
    assert kw["deconvolver"] == "mtmfs" and kw["scales"] == [0, 9, 27]


def test_scales_not_checked_when_the_deconvolver_uses_none(tmp_path, monkeypatch):
    psf, _ = _psf_from_weights(64, 4, 20, pad=1.2)
    aj, tasks, reads = _scales_pipeline(tmp_path, monkeypatch, psf,
                                        ImagingConfig(deconvolver="hogbom"))
    aj._compute_scales()
    assert reads == [] and aj.justification["derived"]["scales"]["psf_check"] is None


# ---------------------------------------------------------------------------
# The initial CLEAN stops at once when it leaves no model
# ---------------------------------------------------------------------------
def test_initial_clean_with_empty_model_stops_with_the_reason(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(
        tmp_path, monkeypatch, imaging=ImagingConfig(mask_mode="user", user_mask="/data/region.crtf"))
    aj._derived["scales"] = [0, 8, 23, 70, 209, 628]
    aj._derived["sigma_th"] = 2.8432e-5                       # threshold 3 sigma_th = 85.3 uJy
    aj._tclean = lambda vis, base, **kw: 1
    residual, mask = _planes()
    (tmp_path / "out" / "intermediate" / "clean_iter0.mask").mkdir()
    planes = {"model": np.zeros_like(residual), "residual": residual, "mask": mask}
    monkeypatch.setattr(core, "_read_image_plane",
                        lambda path: planes[str(path).rsplit(".", 1)[-1]])

    def no_stats(*a, **k):
        raise AssertionError("statistics of an empty CLEAN must not be computed")
    aj._export_and_stats = no_stats
    with pytest.raises(RuntimeError, match="initial CLEAN .* minor cycle did not run"):
        aj._make_initial_clean()
    check = aj.justification["checks"][-1]
    assert check["kind"] == "clean_model" and not check["ok"]
    assert check["n_model_pixels"] == 0 and check["threshold_jy"] == pytest.approx(8.5296e-5)
    assert aj.metrics == [] and aj.justification["iterations"] == []


def test_initial_clean_with_a_model_passes_the_check(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch)
    aj._tclean = lambda vis, base, **kw: 1
    model = np.zeros((8, 8))
    model[4, 4] = 1e-3
    monkeypatch.setattr(core, "_read_image_plane", lambda path: model)
    aj._export_and_stats = lambda base, nterms, fits: {
        "peak_jy_beam": 1e-2, "rms_jy_beam": 1e-5, "snr": 1e3, "dynamic_range": 1e3,
        "bmaj_arcsec": 0.2, "bmin_arcsec": 0.15}
    aj._make_initial_clean()
    check = aj.justification["checks"][-1]
    assert check["kind"] == "clean_model" and check["ok"] and check["n_model_pixels"] == 1
    assert aj.metrics[0]["iteration"] == 0


def test_initial_dirty_image_with_niter_0_is_not_checked_for_a_model(tmp_path, monkeypatch):
    """niter = 0 makes a dirty image (ASEBI's bootstrap image through image()): no model is expected,
    so the empty-model check must not stop it."""
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch, imaging=ImagingConfig(niter=0))
    aj._tclean = lambda vis, base, **kw: 1
    monkeypatch.setattr(core, "_read_image_plane", lambda path: np.zeros((8, 8)))
    aj._export_and_stats = lambda base, nterms, fits: {
        "peak_jy_beam": 1e-2, "rms_jy_beam": 1e-5, "snr": 1e3, "dynamic_range": 1e3,
        "bmaj_arcsec": 0.2, "bmin_arcsec": 0.15}
    aj._make_initial_clean()
    assert not [c for c in aj.justification["checks"] if c["kind"] == "clean_model"]
    assert aj.metrics[0]["iteration"] == 0
