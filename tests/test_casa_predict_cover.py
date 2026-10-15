"""Model prediction must reach every channel (CASA tests, simulated MS).

DG Tau B (2015 EB, 2026-10-16): the measurement copy had a descending spw
(negative channel width) at the low edge of the band. tclean's mfs image range
began one channel above the data's lowest edge, and ``ft`` of that model wrote
MODEL_DATA = 0 into the lowest channel; AJISAI's model check stopped the run.
These tests pin the CASA behaviour on a simulated MS of the same layout and
check that ``_models_covering_data`` gives ``ft`` a model that reaches every
channel without changing the predicted values.
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
    work = tmp_path_factory.mktemp("ajisai_cover")
    # The DG Tau B layout: a narrow-channel spw at the low edge (19.5 MHz channels, descending) and a
    # wide one at the high edge, observed when the LSRK frequency is ~1.3e-4 above TOPO (+29 MHz at
    # 219.5 GHz, more than half a narrow channel), at the source's transit
    ms = simulate_ms(str(work / "input.ms"), freq_ghz=(219.5534, 231.4173), nchan=(3, 3),
                     chan_mhz=(-19.53125, 578.125), scans=2, phase_rms_deg=0.0,
                     date="2016/02/29/22:00:00", ra="04h27m02.57s", dec="+26d05m30.2s")
    return {"work": work, "ms": ms}


def _model_zero_channels(vis):
    """(spw, channel) pairs with MODEL_DATA == 0 on every unflagged cross-correlation sample."""
    import casatools
    tb = casatools.table()
    tb.open(vis)
    out = []
    try:
        for dd in sorted(set(int(x) for x in np.unique(tb.getcol("DATA_DESC_ID")))):
            q = tb.query(f"DATA_DESC_ID=={dd} && ANTENNA1!=ANTENNA2", columns="MODEL_DATA,FLAG")
            try:
                m = np.abs(q.getcol("MODEL_DATA"))
                f = q.getcol("FLAG")
            finally:
                q.close()
            for c in range(m.shape[1]):
                ok = ~f[:, c, :]
                if ok.any() and not np.any(m[:, c, :][ok] > 0):
                    out.append((dd, c))
    finally:
        tb.close()
    return out


def _clean(vis, base):
    import casatasks
    d = os.path.dirname(base)
    for p in [p for p in os.listdir(d) if p.startswith(os.path.basename(base) + ".")]:
        shutil.rmtree(os.path.join(d, p))
    casatasks.tclean(vis=vis, imagename=base, specmode="mfs", imsize=[IMSIZE, IMSIZE], cell=CELL,
                     weighting="briggs", robust=0.5, gridder="standard", deconvolver="hogbom",
                     niter=50, threshold="0Jy", savemodel="none", interactive=False)


def test_ft_reaches_every_channel_with_the_widened_model(sim):
    import casatasks
    from ajisai.core import _data_frequency_edges, _image_spectral_axis, _models_covering_data
    base = str(sim["work"] / "img")
    _clean(sim["ms"], base)
    ref, inc = _image_spectral_axis(base + ".model")
    lo, hi = _data_frequency_edges(sim["ms"])

    # The CASA behaviour behind the failure (recorded, not required): the image range may miss the
    # descending spw's outermost channel; then ft of the original model leaves it at zero.
    misses = ref - abs(inc) / 2 > lo + 1.0
    print(f"\n[cover] image {ref - abs(inc) / 2:.6e}-{ref + abs(inc) / 2:.6e} Hz, data {lo:.6e}-{hi:.6e} Hz, misses={misses}")
    plain = str(sim["work"] / "plain.ms")
    shutil.copytree(sim["ms"], plain)
    casatasks.ft(vis=plain, model=base + ".model", usescratch=True)
    zero_plain = _model_zero_channels(plain)
    print(f"[cover] ft of the original model: zero channels {zero_plain}")
    # tclean's mfs range is the data's range in LSRK; ft compares the TOPO channel frequencies with it,
    # so with a positive LSRK shift larger than half the lowest channel that channel gets no model
    assert misses and zero_plain, "the layout no longer reproduces the dropped channel"

    # With the widened copy every channel gets the model, the original image is unchanged, and the
    # values on the channels the original reached are the same.
    fixed = str(sim["work"] / "fixed.ms")
    shutil.copytree(sim["ms"], fixed)
    models, rec = _models_covering_data([base + ".model"], fixed)
    try:
        casatasks.ft(vis=fixed, model=models[0], usescratch=True)
    finally:
        for m in models:
            if m != base + ".model":
                shutil.rmtree(m, ignore_errors=True)
    assert _model_zero_channels(fixed) == []
    assert _image_spectral_axis(base + ".model") == (ref, inc)          # original untouched
    assert rec["data_lo_hz"] == pytest.approx(lo) and rec["data_hi_hz"] == pytest.approx(hi)
    import casatools
    tb = casatools.table()
    tb.open(plain)
    a = tb.getcol("MODEL_DATA")
    tb.close()
    tb.open(fixed)
    b = tb.getcol("MODEL_DATA")
    tb.close()
    reached = np.abs(a) > 0
    assert np.allclose(a[reached], b[reached], rtol=1e-6, atol=0.0)
