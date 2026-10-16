"""
The applycal spwmap of a combined-spw gain table (CASA tests, simulated MSs).

Marked ``casa``: skipped automatically when casatools/casatasks are absent.
ASEBI SPEC 9 item 3 (DESIGN_REVIEW D38 d); convention audit 2026-10-16
(combine item 3, time item 3). gaincal(combine='scan,spw') writes each
solution under one spw label, which is not always the lowest spw of the
observation:

* the lowest spw fully flagged (rows kept): the label is still spw 0; the map
  AJISAI built from the spws with valid data was [0, 1] and left every spw-1
  row uncorrected;
* three EBs of one setup concatenated (concat merges their spws): labels
  0, 1, 1 per observation; the "lowest spw of the observation" map [0, 0]
  left observations 1 and 2 uncorrected. No single map serves all three, so
  applycal runs per observation;
* the lowest spw without rows in part of a scan: some solutions of the
  observation are labelled spw 1; they are relabelled to the observation's
  lowest label so that one map applies all of them.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.casa

POINT = ((0.5, 0.0, 0.0, 0.05, 0.05, 0.0),)
KW = dict(nant=8, freq_ghz=(230.0, 232.0), nchan=(4, 4), chan_mhz=(250.0, 250.0), scans=2,
          scan_s=60.0, int_s=6.0, noise="1mJy", source=POINT, shared_scans=True)


def _aj(vis):
    from ajisai import AJISAI, AJISAIConfig, GainCalConfig
    return AJISAI(AJISAIConfig(vis=str(vis), gaincal=GainCalConfig(combine="scan,spw"),
                               verbose=False))


def _gaincal(vis, ct, solint="inf", combine="scan,spw"):
    import casatasks
    shutil.rmtree(ct, ignore_errors=True)
    casatasks.gaincal(vis=vis, caltable=ct, solint=solint, combine=combine, calmode="p",
                      gaintype="T", refant="0", minsnr=0.0, minblperant=4)
    return ct


def _corrected_fraction(ms):
    """Fraction of the cross-correlation rows of each (observation, spw) with CORRECTED != DATA."""
    import casatools
    tb = casatools.table()
    tb.open(ms + "/DATA_DESCRIPTION")
    dd_spw = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()
    tb.open(ms)
    out = {}
    try:
        for dd in np.unique(tb.getcol("DATA_DESC_ID")):
            q = tb.query(f"DATA_DESC_ID=={int(dd)} && ANTENNA1!=ANTENNA2 && ANTENNA2==5")
            try:
                o = q.getcol("OBSERVATION_ID")
                ratio = q.getcol("CORRECTED_DATA")[0, 0] / q.getcol("DATA")[0, 0]
            finally:
                q.close()
            for x in np.unique(o):
                out[(int(x), int(dd_spw[dd]))] = float(np.mean(np.abs(ratio[o == x] - 1) > 1e-6))
    finally:
        tb.close()
    return out


def _derive(vis, ct):
    """The core's derivation: labels from the table, spws with valid data from the summary."""
    from ajisai import solint as so
    from ajisai.ms_utils import read_vis_summary
    vs = read_vis_summary(vis)
    labels = _aj(vis)._caltable_spw_labels(ct)
    spws = {int(o): sorted(int(x) for x in np.unique(vs.spw[vs.obs == o])) for o in np.unique(vs.obs)}
    return labels, spws, so.spwmap_from_labels(labels, spws, int(vs.spw.max()) + 1)


def test_fully_flagged_lowest_spw_still_labels_the_solutions(tmp_path):
    import casatasks
    import casatools
    from ajisai import solint as so
    from tests.casa_sim import simulate_ms
    ms = simulate_ms(str(tmp_path / "a.ms"), seed=1, phase_rms_deg=30.0, **KW)
    tb = casatools.table()
    tb.open(ms, nomodify=False)
    try:
        sub = tb.selectrows(np.flatnonzero(tb.getcol("DATA_DESC_ID") == 0).tolist())
        f = sub.getcol("FLAG")
        f[:] = True
        sub.putcol("FLAG", f)
        fr = sub.getcol("FLAG_ROW")
        fr[:] = True
        sub.putcol("FLAG_ROW", fr)
        sub.close()
    finally:
        tb.close()
    casatasks.clearcal(vis=ms, addmodel=True)
    ct = _gaincal(ms, str(tmp_path / "a.G"))
    labels, spws, rec = _derive(ms, ct)
    assert labels == {0: [0]} and spws == {0: [1]}
    old = so.spwmap_for_combined_spw(spws, 2)          # the map from the spws with valid data
    assert old == [0, 1] and rec["global"] == [0, 0]
    fracs = {}
    for name, m in (("old", old), ("new", rec["global"])):
        cp = str(tmp_path / f"a_{name}.ms")
        shutil.copytree(ms, cp)
        _aj(cp)._applycal(cp, ct, spwmap=m, interp="linearPDperobs")
        fracs[name] = _corrected_fraction(cp)[(0, 1)]
    print(f"\n[spwmap] fully flagged spw 0: spw-1 rows corrected with {old}: {fracs['old']:.0%}, "
          f"with {rec['global']}: {fracs['new']:.0%}")
    assert fracs["old"] == 0.0 and fracs["new"] == 1.0


def test_concat_merged_setups_need_applycal_per_observation(tmp_path):
    import casatasks
    from ajisai import solint as so
    from tests.casa_sim import simulate_ms
    ebs = [simulate_ms(str(tmp_path / f"b{k}.ms"), seed=10 + k, date=f"{2017 + k}/05/01/00:00:00",
                       phase_rms_deg=40.0, **KW) for k in range(3)]
    cat = str(tmp_path / "b.ms")
    casatasks.concat(vis=ebs, concatvis=cat, dirtol="0.01arcsec", copypointing=False)
    casatasks.clearcal(vis=cat, addmodel=True)
    ct = _gaincal(cat, str(tmp_path / "b.G"))
    labels, spws, rec = _derive(cat, ct)
    print(f"\n[spwmap] merged setups: spws per observation {spws}, labels {labels}")
    assert spws == {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    assert labels == {0: [0], 1: [1], 2: [1]}
    old = so.spwmap_for_combined_spw(spws, 2)
    assert old == [0, 0] and rec["global"] is None
    assert rec["per_observation"] == {0: [0, 0], 1: [1, 1], 2: [1, 1]}
    fracs = {}
    for name in ("old", "new"):
        cp = str(tmp_path / f"b_{name}.ms")
        shutil.copytree(cat, cp)
        if name == "old":
            _aj(cp)._applycal(cp, ct, spwmap=old, interp="linearPDperobs")
        else:
            _aj(cp)._applycal(cp, ct, interp="linearPDperobs", per_observation=rec["per_observation"])
        fracs[name] = _corrected_fraction(cp)
    print(f"[spwmap] rows corrected per (observation, spw): with {old}: {fracs['old']}; "
          f"per observation: {fracs['new']}")
    assert all(fracs["old"][(o, s)] == 0.0 for o in (1, 2) for s in (0, 1))
    assert all(v == 1.0 for v in fracs["new"].values()) and len(fracs["new"]) == 6


def test_several_labels_in_one_observation_are_relabelled(tmp_path):
    """spw 0 lacks rows in the first 4 integrations of scan 2: two solutions labelled spw 1."""
    import casatasks
    import casatools
    from tests.casa_sim import simulate_ms
    ms0 = simulate_ms(str(tmp_path / "c0.ms"), seed=3, phase_rms_deg=30.0, **KW)
    tb = casatools.table()
    tb.open(ms0)
    t = tb.getcol("TIME")
    sc = tb.getcol("SCAN_NUMBER")
    tb.close()
    s2 = np.unique(t[sc == 2])
    ms = str(tmp_path / "c.ms")
    casatasks.mstransform(vis=ms0, outputvis=ms, datacolumn="all",
                          taql=f"!(DATA_DESC_ID==0 && TIME>{s2[0] - 0.5!r} && TIME<{s2[3] + 0.5!r})")
    casatasks.clearcal(vis=ms, addmodel=True)
    ct = _gaincal(ms, str(tmp_path / "c.G"), solint="11.7500s", combine="spw")
    labels, spws, rec = _derive(ms, ct)
    assert labels == {0: [0, 1]} and rec["relabel"] == {0: ([0, 1], 0)}
    assert rec["global"] == [0, 0]
    # with [0, 0] and the labels as written, spw-0 data skip the spw-1-labelled solutions
    plain = str(tmp_path / "c_plain.ms")
    shutil.copytree(ms, plain)
    _aj(plain)._applycal(plain, ct, spwmap=[0, 0], interp="linearPDperobs")
    n = _aj(ms)._relabel_caltable_spws(ct, rec["relabel"])
    assert n == {"0": 16} and _aj(ms)._caltable_spw_labels(ct) == {0: [0]}
    fixed = str(tmp_path / "c_fixed.ms")
    shutil.copytree(ms, fixed)
    _aj(fixed)._applycal(fixed, ct, spwmap=rec["global"], interp="linearPDperobs")
    # the relabelled table, rewritten in gaincal's row order (spw, time, antenna)
    tb.open(ct)
    try:
        sp = tb.getcol("SPECTRAL_WINDOW_ID")
        tt_tab = tb.getcol("TIME")
        an = tb.getcol("ANTENNA1")
        ph = np.angle(tb.getcol("CPARAM")[0, 0])
    finally:
        tb.close()
    rows = list(zip(sp.tolist(), tt_tab.tolist(), an.tolist()))
    assert rows == sorted(rows)
    m5 = an == 5
    sol_t, sol_p = tt_tab[m5], np.unwrap(ph[m5])

    def applied(path):
        tb.open(path)
        try:
            q = tb.query("ANTENNA1==0 && ANTENNA2==5 && DATA_DESC_ID==1")
            try:
                return q.getcol("TIME"), np.angle(q.getcol("CORRECTED_DATA")[0, 0]
                                                  / q.getcol("DATA")[0, 0])
            finally:
                q.close()
        finally:
            tb.close()
    tt, a_plain = applied(plain)
    _, a_fixed = applied(fixed)
    # refant 0: the phase applied to baseline 0-5 is antenna 5's solution, linearly interpolated
    # (linearPD scales it by nu_spw1/nu_spw0 = 1.009, within the tolerance)
    want = np.interp(tt, sol_t, sol_p)
    err_fixed = np.abs(np.angle(np.exp(1j * (a_fixed - want))))
    err_plain = np.abs(np.angle(np.exp(1j * (a_plain - want))))
    early = (tt > s2[0] - 1) & (tt < s2[0] + 25)
    print(f"\n[spwmap] relabelled {n}; |applied - solutions| on spw 1: relabelled table max "
          f"{err_fixed.max():.3f} rad; labels as written, first 25 s of scan 2 max "
          f"{err_plain[early].max():.3f} rad")
    assert err_fixed.max() < 0.02
    assert err_plain[early].max() > 0.5


def test_data_mode_corrects_every_observation_of_merged_setups(tmp_path):
    """End to end (selfcal_measurement, combine='scan,spw'): two EBs of one setup with static
    antenna phase errors, concatenated (spws merged, labels 0 and 1). Stage 1 (one solution per
    EB) must correct both observations: applycal runs per observation with the table's maps."""
    import casatasks
    import casatools
    from ajisai import GainCalConfig, ImagingConfig, selfcal_measurement
    from tests.casa_sim import simulate_ms
    kw = dict(KW, source=((0.06, 0.0, 0.0, 0.3, 0.3, 0.0),), noise="5mJy", scans=3, scan_s=90.0)
    ebs = []
    for k in range(2):
        ms = simulate_ms(str(tmp_path / f"e{k}.ms"), seed=20 + k, date=f"{2017 + k}/05/01/00:00:00",
                         phase_rms_deg=0.0, **kw)
        rng = np.random.default_rng(30 + k)
        tb = casatools.table()
        tb.open(ms, nomodify=False)
        try:
            a1 = tb.getcol("ANTENNA1")
            a2 = tb.getcol("ANTENNA2")
            ph = rng.uniform(-1.5, 1.5, int(max(a1.max(), a2.max())) + 1)
            for d in np.unique(tb.getcol("DATA_DESC_ID")):
                sub = tb.selectrows(np.flatnonzero(tb.getcol("DATA_DESC_ID") == d).tolist())
                s1, s2 = sub.getcol("ANTENNA1"), sub.getcol("ANTENNA2")
                sub.putcol("DATA", sub.getcol("DATA") * np.exp(1j * (ph[s1] - ph[s2]))[None, None, :])
                sub.close()
        finally:
            tb.close()
        ebs.append(ms)
    cat = str(tmp_path / "cat.ms")
    casatasks.concat(vis=ebs, concatvis=cat, dirtol="0.01arcsec", copypointing=False)
    crtf = tmp_path / "region.crtf"
    crtf.write_text("#CRTFv0\ncircle[[16h00m00.0s, -30d00m00.0s], 3arcsec]\n")
    aj = selfcal_measurement(cat, results_dir=str(tmp_path / "out"),
                             gaincal=GainCalConfig(combine="scan,spw"),
                             imaging=ImagingConfig(mask_mode="user", user_mask=str(crtf), niter=500,
                                                   final_depth=()))
    st1 = aj._stage(1)
    assert st1["status"] in ("ok", "anomaly")
    applied = st1["solve"]["applycal"]
    rec = st1["selection"]["spwmap"]
    print(f"\n[spwmap] end to end: labels {rec['labels_by_observation']}, spwmap {applied['spwmap']}, "
          f"per observation {applied['spwmap_per_observation']}")
    assert rec["labels_by_observation"] == {"0": [0], "1": [1]}
    assert applied["spwmap"] == [] and applied["spwmap_per_observation"] == {"0": [0, 0],
                                                                             "1": [1, 1]}

    def scatter(ms):
        """rms phase (rad) of the cross-correlation data per observation (a resolved source at
        the phase centre: zero phase when calibrated)."""
        tb = casatools.table()
        tb.open(ms)
        out = {}
        try:
            for d in np.unique(tb.getcol("DATA_DESC_ID")):
                q = tb.query(f"DATA_DESC_ID=={int(d)} && ANTENNA1!=ANTENNA2")
                try:
                    o = q.getcol("OBSERVATION_ID")
                    ph = np.angle(q.getcol("DATA").mean(axis=(0, 1)))
                finally:
                    q.close()
                for x in np.unique(o):
                    out.setdefault(int(x), []).append(ph[o == x])
        finally:
            tb.close()
        return {x: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for x, v in out.items()}
    before, after = scatter(cat), scatter(st1["stage_ms"])
    print(f"[spwmap] rms phase per observation: input {before}, stage 1 {after}")
    assert all(after[o] < 0.3 * before[o] for o in (0, 1))
