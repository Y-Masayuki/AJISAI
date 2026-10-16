"""
Post-tclean checks: frequency range without reffreq and sumwt against the data's weights.

Marked ``casa``: skipped automatically when casatools/casatasks are absent.
ASEBI SPEC 10.3 (DESIGN_REVIEW D38 h); convention audit 2026-10-16
(spectral, item 2). tclean centres an image's frequency axis on reffreq and
makes it 2 max|edge - reffreq| wide, so the range check of an image made
with an off-centre reffreq sees only the far edge: with data from 216 to
234 GHz, an image of the 233 GHz spw alone (channels dropped) passed with
reffreq = 218 GHz. AJISAI passes reffreq only to mtmfs images and checks
the range and sumwt on a natural-weighted check image without reffreq.
These tests pin the sumwt formula (tclean's natural-weighted Stokes I sumwt
= the sum of both parallel hands' weights over the rows without FLAG_ROW and
the channels where both are unflagged, WEIGHT_SPECTRUM where defined;
``ms_utils.stokes_i_weight_sum``, the rule of ASEBI's
``engines.common.stokes_i_weight_sum``) and show that the comparison
catches what the range check misses.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.casa


def _sumwt(vis, base, nterms=1, reffreq=None, spw="", weighting="natural"):
    import casatasks
    import casatools
    d = os.path.dirname(base)
    for p in [p for p in os.listdir(d) if p.startswith(os.path.basename(base) + ".")]:
        shutil.rmtree(os.path.join(d, p))
    kw = dict(vis=vis, imagename=base, imsize=[64, 64], cell="0.1arcsec", specmode="mfs", niter=0,
              weighting=weighting, robust=0.5, datacolumn="data", interactive=False, spw=spw)
    if nterms > 1:
        kw.update(deconvolver="mtmfs", nterms=nterms)
        if reffreq:
            kw["reffreq"] = reffreq
    casatasks.tclean(**kw)
    suffix = ".tt0" if nterms > 1 else ""
    ia = casatools.image()
    ia.open(f"{base}.sumwt{suffix}")
    try:
        return float(np.sum(ia.getchunk()))
    finally:
        ia.close()


def _edit(ms, fn):
    """Apply ``fn(table, rows)`` to every data description of ``ms`` (shapes differ between them)."""
    import casatools
    tb = casatools.table()
    tb.open(ms, nomodify=False)
    try:
        dd = tb.getcol("DATA_DESC_ID")
        for d in np.unique(dd):
            sub = tb.selectrows(np.flatnonzero(dd == d).tolist())
            fn(sub, int(d))
            sub.close()
    finally:
        tb.close()


@pytest.fixture(scope="module")
def layouts(tmp_path_factory):
    """Weights and flags that distinguish the candidate formulas."""
    import casatasks
    from tests.casa_sim import simulate_ms
    work = tmp_path_factory.mktemp("ajisai_sumwt")
    base = simulate_ms(str(work / "base.ms"), nant=8, nchan=(4, 8), chan_mhz=(250.0, 125.0),
                       scans=2, scan_s=60.0, noise="20mJy", seed=1)
    out = {"work": work, "base": base}
    # XX weights x3, 20 % of the XX samples flagged, half the channels of 10 % of the rows flagged
    uneq = str(work / "uneq.ms")
    shutil.copytree(base, uneq)
    rng = np.random.default_rng(1)

    def unequal(sub, d):
        w = sub.getcol("WEIGHT")
        w[0] *= 3.0
        sub.putcol("WEIGHT", w)
        sub.putcol("SIGMA", 1.0 / np.sqrt(w))
        f = sub.getcol("FLAG")
        f[0][rng.random(f.shape[1:]) < 0.2] = True
        rows = rng.random(f.shape[2]) < 0.1
        f[:, : f.shape[1] // 2, rows] = True
        sub.putcol("FLAG", f)
    _edit(uneq, unequal)
    out["uneq"] = uneq
    # WEIGHT_SPECTRUM with per-channel structure
    ws = str(work / "ws.ms")
    casatasks.mstransform(vis=base, outputvis=ws, datacolumn="data", usewtspectrum=True)

    def spectrum(sub, d):
        x = sub.getcol("WEIGHT_SPECTRUM")
        x[:, : x.shape[1] // 2, :] *= 2.0
        sub.putcol("WEIGHT_SPECTRUM", x)
        sub.putcol("SIGMA_SPECTRUM", 1.0 / np.sqrt(x))
    _edit(ws, spectrum)
    out["ws"] = ws
    # XX weight 0 (not flagged) on a quarter of the rows: the YY weight still counts
    zero = str(work / "zerow.ms")
    shutil.copytree(base, zero)

    def zero_weight(sub, d):
        w = sub.getcol("WEIGHT")
        w[0, ::4] = 0.0
        sub.putcol("WEIGHT", w)
    _edit(zero, zero_weight)
    out["zerow"] = zero
    # one parallel hand only (XX): w_p
    one = str(work / "onehand.ms")
    casatasks.mstransform(vis=uneq, outputvis=one, datacolumn="data", correlation="XX")
    out["onehand"] = one
    return out


@pytest.mark.parametrize("name", ["base", "uneq", "ws", "zerow", "onehand"])
@pytest.mark.parametrize("nterms, reffreq", [(1, None), (2, None), (2, "229GHz")])
def test_natural_sumwt_is_the_stokes_i_weight_sum(layouts, name, nterms, reffreq):
    from ajisai.ms_utils import stokes_i_weight_sum
    vis = layouts[name]
    exp = stokes_i_weight_sum(vis)["expected_sumwt"]
    got = _sumwt(vis, str(layouts["work"] / f"img_{name}_{nterms}"), nterms, reffreq)
    print(f"\n[sumwt] {name} nterms {nterms} reffreq {reffreq}: sumwt {got:.9g}, expected "
          f"{exp:.9g}, ratio {got / exp:.9f}")
    assert got / exp == pytest.approx(1.0, abs=1e-6)


def test_briggs_sumwt_has_no_closed_form(layouts):
    """Why the comparison needs natural weighting: robust 0.5 grids a third of the weight here."""
    from ajisai.ms_utils import stokes_i_weight_sum
    vis = layouts["base"]
    got = _sumwt(vis, str(layouts["work"] / "img_briggs"), weighting="briggs")
    ratio = got / stokes_i_weight_sum(vis)["expected_sumwt"]
    print(f"\n[sumwt] briggs 0.5: ratio {ratio:.4f}")
    assert ratio < 0.9


def test_sumwt_catches_dropped_channels_an_offcentre_reffreq_hides(tmp_path):
    """Data 216-234 GHz; the image grids only the 233 GHz spw (dropped channels). With reffreq
    218 GHz its axis covers the whole band and the range check passes; the sumwt comparison fails.
    Without reffreq the range check fails as designed."""
    from tests.casa_sim import simulate_ms
    from ajisai.core import (_data_frequency_edges, _image_spectral_axis,
                             imaging_consistency_problems, sumwt_problems)
    from ajisai.ms_utils import stokes_i_weight_sum
    ms = simulate_ms(str(tmp_path / "wide.ms"), nant=8, freq_ghz=(216.5, 233.0), nchan=(8, 8),
                     chan_mhz=(-62.5, 125.0), scans=1, scan_s=60.0, noise="10mJy", seed=2)
    lo, hi = _data_frequency_edges(ms)
    exp = stokes_i_weight_sum(ms)["expected_sumwt"]
    out = {}
    for label, reffreq in (("reffreq 218 GHz", "218GHz"), ("no reffreq", None)):
        base = str(tmp_path / ("img_ref" if reffreq else "img_noref"))
        got = _sumwt(ms, base, nterms=2, reffreq=reffreq, spw="1")
        ref, inc = _image_spectral_axis(base + ".image.tt0")
        rf = 218e9 if reffreq else None
        out[label] = (imaging_consistency_problems(ref, inc, got, lo, hi, reffreq_hz=rf,
                                                   exact_axis=True),
                      sumwt_problems(got, exp))
        print(f"\n[sumwt] {label}: axis {(ref - abs(inc) / 2) / 1e9:.3f}-{(ref + abs(inc) / 2) / 1e9:.3f} "
              f"GHz for data {lo / 1e9:.3f}-{hi / 1e9:.3f} GHz; sumwt ratio {got / exp:.4f}")
    rng_ref, sw_ref = out["reffreq 218 GHz"]
    assert rng_ref == [] and len(sw_ref) == 1          # the axis check is fooled, sumwt is not
    rng_no, sw_no = out["no reffreq"]
    assert len(rng_no) == 2 and len(sw_no) == 1        # does not cover, and is not the data's range
