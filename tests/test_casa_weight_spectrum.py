"""
WEIGHT_SPECTRUM defined on some rows only (CASA tests, simulated MSs).

Marked ``casa``: skipped automatically when casatools/casatasks are absent.
Convention audit 2026-10-16 (weights, item 2; ASEBI SPEC 9 item 3, D38 d):
concat of an MS with WEIGHT_SPECTRUM and one without keeps the column when
the earliest input has it, defined on that input's rows only. Reading an
undefined cell aborts CASA 6.5.1, and ``read_vis_summary`` decided per data
description from the first row, so it aborted the process. Now each row is
read with its own weight column (``ms_utils.weight_column_queries``).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.casa


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    import casatasks
    import casatools
    from tests.casa_sim import simulate_ms
    work = tmp_path_factory.mktemp("ajisai_mixed_ws")
    kw = dict(nant=6, nchan=(4, 4), chan_mhz=(250.0, 250.0), scans=2, scan_s=60.0, noise="1mJy")
    plain = simulate_ms(str(work / "plain.ms"), seed=1, date="2018/05/01/00:00:00", **kw)
    early = simulate_ms(str(work / "early.ms"), seed=2, date="2017/05/01/00:00:00", **kw)
    ws = str(work / "early_ws.ms")
    casatasks.mstransform(vis=early, outputvis=ws, datacolumn="data", usewtspectrum=True)
    tb = casatools.table()
    tb.open(ws, nomodify=False)
    try:
        dd = tb.getcol("DATA_DESC_ID")
        for d in np.unique(dd):
            sub = tb.selectrows(np.flatnonzero(dd == d).tolist())
            x = sub.getcol("WEIGHT_SPECTRUM")
            x[:, 2:, :] *= 4.0                      # per-channel weights WEIGHT cannot carry
            sub.putcol("WEIGHT_SPECTRUM", x)
            sub.putcol("SIGMA_SPECTRUM", 1.0 / np.sqrt(x))
            sub.close()
    finally:
        tb.close()
    # MODEL_DATA before the concat: clearcal on the mixed concat crashes CASA 6.5.1
    for v in (plain, ws):
        casatasks.clearcal(vis=v, addmodel=True)
    cat = str(work / "concat.ms")
    shutil.rmtree(cat, ignore_errors=True)
    casatasks.concat(vis=[plain, ws], concatvis=cat, dirtol="0.01arcsec", copypointing=False)
    return {"plain": plain, "ws": ws, "concat": cat}


def test_weight_spectrum_defined_on_the_earliest_input_only(mixed):
    import casatools
    from ajisai.ms_utils import weight_column_queries
    tb = casatools.table()
    tb.open(mixed["concat"])
    try:
        n = tb.nrows()
        obs = tb.getcol("OBSERVATION_ID")
        defined = np.array([tb.iscelldefined("WEIGHT_SPECTRUM", i) for i in range(n)])
        counts = []
        for q, col in weight_column_queries(tb.colnames(), "ANTENNA1!=ANTENNA2"):
            sub = tb.query(q)
            counts.append((col, sub.nrows()))
            sub.close()
    finally:
        tb.close()
    # observation 0 is the earliest input (with WEIGHT_SPECTRUM), observation 1 has none
    assert defined[obs == 0].all() and not defined[obs == 1].any()
    assert counts == [("WEIGHT_SPECTRUM", int((obs == 0).sum())), ("WEIGHT", int((obs == 1).sum()))]


def test_read_vis_summary_uses_each_rows_weights(mixed):
    """Every row reduced with its own weights: the concat equals its two inputs read alone."""
    from ajisai.ms_utils import read_vis_summary
    cat = read_vis_summary(mixed["concat"])
    alone = {0: read_vis_summary(mixed["ws"]), 1: read_vis_summary(mixed["plain"])}
    for o, v in alone.items():
        sel = cat.obs == o
        assert sel.sum() == v.n_rows
        np.testing.assert_allclose(cat.P[sel].sum(), v.P.sum(), rtol=1e-12)
        np.testing.assert_allclose(cat.w[sel].sum(), v.w.sum(), rtol=1e-12)
    # the WEIGHT_SPECTRUM rows carry the x4 channels: P per row 2.5 x that of WEIGHT alone
    assert cat.P[cat.obs == 0].sum() / cat.P[cat.obs == 1].sum() == pytest.approx(2.5, rel=1e-6)
