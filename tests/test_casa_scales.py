"""
Multiscale scales the uv coverage cannot see, on a simulated MS (CASA required).

Marked ``casa``. Pins :func:`ajisai.core.scale_psf_response` against CASA's own
multiscale minor cycle and checks the fix of the V1094 Sco failure (ASEBI v0.3
run, EB uid___A002_Xc72427_X2f71: initial CLEAN with an all-zero model):

* every scale whose scale-PSF has its largest value at a negative sidelobe is
  refused by CASA (tclean leaves an all-zero model), and every scale whose
  scale-PSF peaks at the component's position is CLEANed;
* with the scale ladder bounded only by the mask region (the v0.3.0.dev0
  rule), the initial CLEAN of this set-up leaves an empty model; AJISAI drops
  those scales, records them, and its initial CLEAN has a model.

Set-up: 16 antennas of alma.cycle7.3 (15-150 m baselines), cell 0.35", 128^2
(44.8" field, about two primary-beam FWHM), a CRTF circle of 20" radius.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.casa

CELL = "0.35arcsec"
IMSIZE = 128


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    from tests.casa_sim import simulate_ms
    work = tmp_path_factory.mktemp("ajisai_scales")
    ms = simulate_ms(str(work / "input.ms"), nchan=(2, 2), chan_mhz=(500.0, 500.0), scans=2,
                     phase_rms_deg=0.0)
    crtf = work / "region.crtf"
    crtf.write_text("#CRTFv0\ncircle[[16h00m00.0s, -30d00m00.0s], 20arcsec]\n")
    return {"work": work, "ms": ms, "crtf": str(crtf)}


def _tclean(vis, imagename, **kw):
    import casatasks
    for p in [p for p in os.listdir(os.path.dirname(imagename))
              if p.startswith(os.path.basename(imagename) + ".")]:
        shutil.rmtree(os.path.join(os.path.dirname(imagename), p))
    args = dict(vis=vis, imagename=imagename, specmode="mfs", imsize=[IMSIZE, IMSIZE], cell=CELL,
                weighting="briggs", robust=0.5, gridder="standard", savemodel="none",
                interactive=False)
    args.update(kw)
    casatasks.tclean(**args)


def _model_pixels(base) -> int:
    from ajisai.core import _read_image_plane
    return int(np.count_nonzero(_read_image_plane(base + ".model")))


def test_scale_psf_rule_predicts_casa_refusal(sim):
    from ajisai.core import _read_image_plane, scale_psf_response
    work = str(sim["work"])
    psf_base = os.path.join(work, "psf")
    _tclean(sim["ms"], psf_base, niter=0, deconvolver="hogbom")
    response = scale_psf_response(_read_image_plane(psf_base + ".psf"), [0, 6, 12, 18, 29, 37, 46, 58])
    negative = [r["scale"] for r in response if r["max_abs"] < 0]
    ok = [r["scale"] for r in response if r["ok"] and r["scale"] > 0]
    assert negative and ok, response          # the set-up has both kinds
    for s in ok + negative:
        base = os.path.join(work, f"verdict_{s}")
        _tclean(sim["ms"], base, niter=20, gain=0.1, threshold="0Jy", deconvolver="multiscale",
                scales=[0, s], usemask="user", mask=sim["crtf"])
        n = _model_pixels(base)
        if s in ok:
            assert n > 0, f"scale {s}: predicted usable, CASA left an empty model"
        else:
            assert n == 0, f"scale {s}: predicted refused by CASA, but CASA CLEANed"


def test_initial_clean_has_a_model_with_the_unseen_scales_dropped(sim):
    from ajisai import AJISAI, AJISAIConfig, ImagingConfig, SelfcalSchedule
    work = sim["work"]
    cfg = AJISAIConfig(
        vis=sim["ms"], results_dir=str(work / "out"),
        imaging=ImagingConfig(cell=CELL, imsize=IMSIZE, mask_mode="user", user_mask=sim["crtf"],
                              niter=200, final_depth=()),
        schedule=SelfcalSchedule(mode="fixed", steps=()),
        relabel_j2000_to_icrs=False, show_banner=False, verbose=False)
    aj = AJISAI(cfg).run()
    rec = aj.justification["derived"]["scales"]
    dropped = rec["psf_check"]["dropped"]
    assert dropped and dropped[-1] == rec["candidates"][-1], rec
    assert rec["value"] == [s for s in rec["candidates"] if s not in dropped]
    check = [c for c in aj.justification["checks"] if c["kind"] == "clean_model"]
    assert len(check) == 1 and check[0]["ok"] and check[0]["n_model_pixels"] > 0
    assert aj.justification["casa_log"]

    # The same initial CLEAN with the ladder bounded by the mask only (before the fix): CASA
    # refuses the scales with a negative sidelobe peak and leaves no component.
    negative = [r["scale"] for r in rec["psf_check"]["response"] if r["max_abs"] < 0]
    assert negative, rec["psf_check"]
    base = os.path.join(str(work), "unbounded")
    thr = aj.metrics[0]["threshold_jy"]
    _tclean(sim["ms"], base, niter=200, gain=0.05, threshold=f"{thr}Jy", deconvolver="multiscale",
            scales=rec["candidates"], usemask="user", mask=sim["crtf"], field="0")
    assert _model_pixels(base) == 0
