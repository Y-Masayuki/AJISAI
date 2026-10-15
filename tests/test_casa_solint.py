"""
The data-driven self-calibration (``schedule.mode="data"``) on a simulated MS.

Marked ``casa``: skipped automatically when casatools/casatasks are absent.
Checks, with real gaincal/applycal/split/tclean (ASEBI SPEC v0.3 10.2):

* the measurement entry point runs the adaptive loop (stage 1 = one solution
  per EB, then data-driven stages) and leaves the input MS untouched;
* the numpy model of gaincal's solution intervals matches the tables
  gaincal writes (every table row is matched to a predicted solution, and
  the solution times agree), so the G1 gate flags the right solutions;
* the phase tracking between stage MSs works (state 0 is recovered);
* the stage records hold the measured quantities, candidates and decisions;
* applycal does not interpolate across a flagged solution (calonly,
  linearPD): the open interval between the neighbouring good solutions is
  left uncorrected, as ``ajisai.solint`` assumes (P18 test results 5).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.casa


def _digest(path: str) -> str:
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if name == "table.lock":
                continue
            full = os.path.join(root, name)
            h.update(os.path.relpath(full, path).encode())
            with open(full, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def _columns(ms):
    import casatools
    tb = casatools.table()
    tb.open(ms)
    try:
        return sorted(c for c in tb.colnames() if c.endswith("DATA"))
    finally:
        tb.close()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from tests.casa_sim import simulate_ms
    from ajisai import ImagingConfig, selfcal_measurement

    work = tmp_path_factory.mktemp("ajisai_solint")
    ms = simulate_ms(str(work / "input.ms"), nchan=(2, 2), chan_mhz=(500.0, 500.0), scans=3,
                     scan_s=120.0, int_s=6.0, noise="60mJy", phase_rms_deg=40.0, seed=3)
    crtf = work / "region.crtf"
    crtf.write_text("#CRTFv0\ncircle[[16h00m00.0s, -30d00m00.0s], 3arcsec]\n")
    before = _digest(ms)
    aj = selfcal_measurement(
        ms, results_dir=str(work / "out"),
        imaging=ImagingConfig(mask_mode="user", user_mask=str(crtf), niter=1000,
                              final_depth=()))
    return {"aj": aj, "ms": ms, "before": before}


def test_input_untouched_and_products(run):
    aj = run["aj"]
    assert _digest(run["ms"]) == run["before"]
    accepted = aj.justification["selfcal"]["accepted_stages"]
    assert accepted and accepted[0] == 1
    for k in accepted:
        assert _columns(aj._stage(k)["stage_ms"]) == ["DATA", "MODEL_DATA"]
    assert _columns(aj.final_ms) == ["DATA"]
    assert aj.justification["final_ms"]["stage"] == accepted[-1]
    interm = os.path.dirname(aj._stage(1)["stage_ms"])
    assert [p for p in os.listdir(interm) if ".tmp.ms" in p] == []
    assert aj.justification["best"]["iteration"] == len(
        [m for m in aj.metrics if m["status"] in ("ok", "anomaly")]) - 1


def test_stage_one_is_one_solution_per_eb(run):
    aj = run["aj"]
    st = aj._stage(1)
    assert st["selection"]["chosen"] == "EB"
    assert st["solve"]["solint"] == "inf" and st["solve"]["combine"] == "scan"
    assert [c["kind"] for c in st["selection"]["candidates"]] == ["eb"]


def test_predicted_solution_intervals_match_the_gaincal_tables(run):
    """Every gaincal row is matched to a predicted solution at the same time."""
    import casatools
    from ajisai import solint as so

    aj = run["aj"]
    geom = aj._sc["geom"]
    tb = casatools.table()
    for k, solve in enumerate(aj._sc["chain"], 1):
        stage = aj._stage(solve.stage)
        gate = stage["selection"]["g1_gate_applied"]
        assert gate["unmatched"] == 0 and gate["n_rows"] > 0
        units = so.solution_units(solve.state, geom, solve.cand, combine_spw=False)
        tb.open(stage["solve"]["table"])
        try:
            t = tb.getcol("TIME")
            spw = tb.getcol("SPECTRAL_WINDOW_ID")
            ant = tb.getcol("ANTENNA1")
        finally:
            tb.close()
        for sp in np.unique(spw):
            got = np.unique(t[(spw == sp) & (ant == 1)])
            want = np.sort(units.time[units.group == sp])
            np.testing.assert_allclose(got, want, atol=1e-3)


def test_phase_tracking_between_stage_mss(run):
    aj = run["aj"]
    chain = aj._sc["chain"]
    state0 = aj._sc["state0"]
    assert np.all(chain[0].state.phi0 == 0)
    if len(chain) > 1:
        # DATA of selfcal_1.ms is DATA_0 times the stage-1 correction: phi0 is not zero,
        # and the model of stage 1 against DATA_0 is recovered exactly
        st1 = chain[1].state
        assert np.std(st1.phi0) > 1e-3
        np.testing.assert_allclose(np.abs(st1.y0()), np.abs(st1.y), rtol=1e-9)
        assert st1.n_rows == state0.n_rows


def test_records_and_justification_file(run):
    aj = run["aj"]
    path = os.path.join(aj._derived["workdir"], "justification.json")
    with open(path) as fh:
        js = json.load(fh)
    assert js["selfcal"]["mode"] == "data"
    stages = [s for s in js["stages"] if s["stage"] > 0]
    assert stages
    for s in stages:
        sel = s["selection"]
        assert {"measured", "candidates", "decision", "reason"} <= set(sel)
        assert {"s", "r", "D_eff", "D_fit", "g2_split"} <= set(sel["measured"])
        for c in sel["candidates"]:
            assert {"label", "T_s", "g1", "evaluated"} <= set(c)
    summary = aj.selfcal_summary()
    assert summary["best_stage"] == js["selfcal"]["best_stage"]
    assert summary["best_model"]


def test_flagged_solution_is_not_interpolated(tmp_path):
    """applycal calonly/linearPD: a flagged solution leaves the open interval uncorrected."""
    import casatasks
    import casatools
    from tests.casa_sim import simulate_ms
    from ajisai import solint as so

    ms = simulate_ms(str(tmp_path / "oi.ms"), nant=8, nchan=(2, 2), chan_mhz=(500.0, 500.0),
                     scans=1, scan_s=60.0, int_s=6.0, noise="1mJy", phase_rms_deg=30.0, seed=2)
    tb = casatools.table()
    tb.open(ms + "/FIELD")
    d = tb.getcol("PHASE_DIR")
    tb.close()
    cl = casatools.componentlist()
    cl.addcomponent(dir=f"J2000 {d[0, 0, 0]}rad {d[1, 0, 0]}rad", flux=0.085, fluxunit="Jy",
                    freq="230GHz", shape="point")
    cl.rename(str(tmp_path / "pt.cl"))
    cl.done()
    casatasks.ft(vis=ms, complist=str(tmp_path / "pt.cl"), usescratch=True)
    ct = str(tmp_path / "g.pcal")
    casatasks.gaincal(vis=ms, caltable=ct, solint="11.000s", combine="", calmode="p", gaintype="T",
                      refant="0", minsnr=0.0, minblperant=4)
    tb.open(ct, nomodify=False)
    try:
        t = tb.getcol("TIME")
        ant = tb.getcol("ANTENNA1")
        spw = tb.getcol("SPECTRAL_WINDOW_ID")
        flag = tb.getcol("FLAG")
        sol_t = np.unique(t[spw == 0])
        assert sol_t.size == 5                    # 2 integrations per solution, 10 per scan
        bad = (ant == 2) & (spw == 0) & (np.abs(t - sol_t[2]) < 1e-3)
        flag[:, :, bad] = True
        tb.putcol("FLAG", flag)
    finally:
        tb.close()
    casatasks.applycal(vis=ms, gaintable=[ct], interp="linearPD", applymode="calonly",
                       calwt=False)
    tb.open(ms)
    try:
        q = tb.query("ANTENNA1==0 && ANTENNA2==2 && DATA_DESC_ID==0")
        try:
            dt = q.getcol("TIME")
            data = q.getcol("DATA")[0, 0]
            corr = q.getcol("CORRECTED_DATA")[0, 0]
        finally:
            q.close()
    finally:
        tb.close()
    corrected = np.abs(corr / data - 1.0) > 1e-6
    units = so.Units(of_row=np.zeros(1, int), time=sol_t, obs=np.zeros(5, int),
                     group=np.zeros(5, int), n_times=np.full(5, 2))
    good = np.ones((5, 8), bool)
    good[2, 2] = False
    lo, hi, f = so._row_brackets(units, dt, np.zeros(dt.size, int), np.zeros(dt.size, int))
    _, ok0 = so._row_values(np.zeros((5, 8)), good, lo, hi, f, np.zeros(dt.size, int), True)
    _, ok2 = so._row_values(np.zeros((5, 8)), good, lo, hi, f, np.full(dt.size, 2), True)
    assert corrected.tolist() == (ok0 & ok2).tolist()
    assert (~corrected).sum() == 4                 # the open interval holds 4 integrations
