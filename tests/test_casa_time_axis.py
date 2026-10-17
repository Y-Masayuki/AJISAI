"""
The time axis of ``ajisai.solint`` on real measurement sets (CASA tests, simulated MS).

Marked ``casa``: skipped automatically when casatools/casatasks are absent.
Convention audit 2026-10-16 (time, items 1 and 5; ASEBI SPEC 4 item 4 and
DESIGN_REVIEW D38 e):

* a copy that lost 3 % of its rows (split with keepflags=False drops fully
  flagged baseline-integrations) averaged into 18 s bins by mstransform:
  each averaged row's TIME is the mean of the rows present, so one bin
  carries several TIMEs. The distinct-time rule reproduces the audit's
  failure (integration time, scan lengths, even/odd parity, noise-scale
  pairs, SNR rate); the integrations found from INTERVAL give the bins;
* one bin per scan: the integration time was 1 s (SNR rate x 8); now the
  median INTERVAL;
* a scan whose first integration is fully flagged but present: gaincal's
  time grid starts at that row; the model of the solution intervals follows
  it through ``VisSummary.scan_bounds``.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.casa


def _cols(ms, names):
    import casatools
    tb = casatools.table()
    tb.open(ms)
    try:
        return {c: tb.getcol(c) for c in names}
    finally:
        tb.close()


def _without_time_info(vs):
    """The rows of ``vs`` without INTERVAL and scan bounds (the rules before D38 e)."""
    from ajisai import solint as so
    return so.VisSummary(time=vs.time, obs=vs.obs, scan=vs.scan, spw=vs.spw, a1=vs.a1, a2=vs.a2,
                         y=vs.y, P=vs.P, antenna_names=vs.antenna_names, S=vs.S,
                         uvdist_m=vs.uvdist_m)


def _measures(vs):
    from ajisai import solint as so
    geom = so.geometry(vs)
    s_map = so.noise_scale(vs, geom)
    return geom, so.snr_rates(vs, s_map, geom)["median"], s_map[0]["n_pairs"]


def _average(ms, out, timebin_s):
    """ASEBI's time averaging before D38 e (apply.time_average): scan boundaries, keepflags."""
    import casatasks
    shutil.rmtree(out, ignore_errors=True)
    casatasks.mstransform(vis=ms, outputvis=out, datacolumn="data", timeaverage=True,
                          timebin=f"{timebin_s:.3f}s", timespan="", keepflags=True)
    casatasks.clearcal(vis=out, addmodel=True)
    return out


@pytest.fixture(scope="module")
def eb(tmp_path_factory):
    """The audit's EB: 10 antennas, one spw, three 63 s scans of 6 s integrations.

    No phase errors: the noise scale then measures the noise alone, so the SNR
    rates compare the time bookkeeping only (a phase walk between adjacent
    18 s bins would enter ``s`` in every copy alike).
    """
    import casatasks
    from tests.casa_sim import simulate_ms
    work = tmp_path_factory.mktemp("ajisai_time_axis")
    base = simulate_ms(str(work / "base.ms"), nant=10, freq_ghz=(230.0,), nchan=(4,),
                       chan_mhz=(250.0,), scans=3, scan_s=63.0, int_s=6.0, noise="1mJy",
                       phase_rms_deg=0.0, seed=5)
    c = _cols(base, ["TIME", "ANTENNA1", "ANTENNA2", "SCAN_NUMBER"])
    start = {int(s): float(c["TIME"][c["SCAN_NUMBER"] == s].min())
             for s in np.unique(c["SCAN_NUMBER"])}
    # 3 % of the baseline-integrations missing, as split(keepflags=False) leaves a copy
    rng = np.random.default_rng(3)
    keys = np.unique(np.stack([np.round(c["TIME"], 3), c["ANTENNA1"], c["ANTENNA2"]]), axis=1).T
    drop = keys[rng.random(len(keys)) < 0.03]
    cond = " && ".join(f"!(ANTENNA1=={int(a)} && ANTENNA2=={int(b)} && abs(TIME-{float(t)!r})<0.01)"
                       for t, a, b in drop)
    dropped = str(work / "dropped.ms")
    casatasks.mstransform(vis=base, outputvis=dropped, datacolumn="all", taql=cond)
    return {"work": work, "base": base, "dropped": dropped, "start": start,
            "n_dropped": len(drop)}


def test_time_averaged_copy_with_missing_rows(eb):
    """18 s bins: 4 per scan; the rule before D38 e reproduces the audit's failure."""
    from ajisai.ms_utils import read_vis_summary
    work = eb["work"]
    clean = read_vis_summary(_average(eb["base"], str(work / "clean18.ms"), 18.0))
    avg = read_vis_summary(_average(eb["dropped"], str(work / "dropped18.ms"), 18.0))
    true_bin = np.array([int(np.floor((t - eb["start"][int(s)] + 1e-6) / 18.0))
                         for t, s in zip(avg.time, avg.scan)])
    _, r_clean, pairs_clean = _measures(clean)
    # the audit's failure: several TIMEs per bin are taken for integrations
    g_old, r_old, pairs_old = _measures(_without_time_info(avg))
    n_times = [np.unique(avg.time[avg.scan == s]).size for s in np.unique(avg.scan)]
    print(f"\n[time axis] dropped {eb['n_dropped']} baseline-integrations; distinct TIMEs per scan "
          f"{n_times}; before: IT {g_old.int_time[0]:.3f} s, scan_nint {g_old.scan_nint.tolist()}, "
          f"pairs {pairs_old}, r {r_old:.4g} vs clean {r_clean:.4g}, wrong parity "
          f"{np.mean((g_old.k_in_scan % 2) != (true_bin % 2)):.1%}")
    assert min(n_times) > 4
    assert abs(g_old.int_time[0] - 18.0) > 3.0
    assert pairs_old == 0 and r_old > 1.3 * r_clean
    assert np.mean((g_old.k_in_scan % 2) != (true_bin % 2)) > 0.2
    # integrations from INTERVAL
    g, r, pairs = _measures(avg)
    print(f"[time axis] after: IT {g.int_time[0]:.3f} s, scan_nint {g.scan_nint.tolist()}, pairs "
          f"{pairs} (clean {pairs_clean}), r {r:.4g}, record {g.time_record['per_observation']['0']}")
    assert g.int_time[0] == pytest.approx(18.0, abs=1e-3)
    assert g.scan_nint.tolist() == [4, 4, 4]
    assert np.array_equal(g.k_in_scan, true_bin)
    assert pairs > 0.9 * pairs_clean
    assert r == pytest.approx(r_clean, rel=0.03)


def test_one_bin_per_scan(eb):
    """A 66 s bin holds a whole 63 s scan: the integration time is the median INTERVAL.

    Without two integrations in a scan the noise scale is assumed (s = 1), so
    the SNR rates are compared as r sqrt(s), the information per second.
    """
    import casatasks
    from ajisai import solint as so
    from ajisai.ms_utils import read_vis_summary
    work = eb["work"]
    full = str(work / "full_model.ms")
    shutil.rmtree(full, ignore_errors=True)
    shutil.copytree(eb["base"], full)
    casatasks.clearcal(vis=full, addmodel=True)
    vs_full = read_vis_summary(full)
    g_full, r_full, _ = _measures(vs_full)
    rate_full = r_full * np.sqrt(so.noise_scale(vs_full, g_full)[0]["s"])
    one = read_vis_summary(_average(eb["base"], str(work / "one66.ms"), 66.0))
    g_old, r_old, _ = _measures(_without_time_info(one))
    print(f"\n[time axis] one bin per scan, before: IT {g_old.int_time[0]} s, r sqrt(s) {r_old:.4g} "
          f"vs unaveraged {rate_full:.4g} (x{r_old / rate_full:.2f})")
    assert g_old.int_time[0] == 1.0 and r_old > 5.0 * rate_full
    g, r, _ = _measures(one)
    iv = np.unique(_cols(str(work / "one66.ms"), ["INTERVAL"])["INTERVAL"])
    print(f"[time axis] after: IT {g.int_time[0]} s (INTERVAL values {iv.tolist()}), "
          f"r sqrt(s) {r:.4g}")
    assert g.scan_nint.tolist() == [1, 1, 1]
    assert g.int_time[0] == pytest.approx(float(np.median(one.interval)))
    assert g.time_record["per_observation"]["0"]["source"] == "median INTERVAL"
    assert r == pytest.approx(rate_full, rel=0.02)


def test_gaincal_grid_from_a_fully_flagged_first_integration(eb):
    """Scan 3 starts with a fully flagged integration (rows kept): gaincal's grid starts there."""
    import casatasks
    import casatools
    from ajisai import solint as so
    from ajisai.ms_utils import read_vis_summary
    work = eb["work"]
    ms = str(work / "flagfirst.ms")
    shutil.rmtree(ms, ignore_errors=True)
    shutil.copytree(eb["base"], ms)
    first3 = eb["start"][3]
    tb = casatools.table()
    tb.open(ms, nomodify=False)
    try:
        t = tb.getcol("TIME")
        sel = np.abs(t - first3) < 1e-3
        f = tb.getcol("FLAG")
        fr = tb.getcol("FLAG_ROW")
        f[:, :, sel] = True
        fr[sel] = True
        tb.putcol("FLAG", f)
        tb.putcol("FLAG_ROW", fr)
    finally:
        tb.close()
    casatasks.clearcal(vis=ms, addmodel=True)
    vs = read_vis_summary(ms)
    assert vs.time.min() > eb["start"][1] - 1 and not np.any(np.abs(vs.time - first3) < 1e-3)
    assert vs.scan_bounds[(0, 3)][0] == pytest.approx(first3, abs=1e-6)
    geom = so.geometry(vs)
    cand = [c for c in so.geometry_candidates(geom) if c.get("n_int") == 2][0]
    ct = str(work / "flagfirst.G")
    shutil.rmtree(ct, ignore_errors=True)
    casatasks.gaincal(vis=ms, caltable=ct, solint=cand["gaincal_solint"], combine="",
                      calmode="p", gaintype="T", refant="0", minsnr=0.0, minblperant=4)
    tab = _cols(ct, ["TIME", "ANTENNA1"])
    got = np.sort(tab["TIME"][tab["ANTENNA1"] == 1])
    want = np.sort(so.solution_units(vs, geom, cand, combine_spw=False).time)
    old_vs = _without_time_info(vs)
    old = np.sort(so.solution_units(old_vs, so.geometry(old_vs), cand, combine_spw=False).time)
    print(f"\n[time axis] flagged first integration of scan 3, {cand['gaincal_solint']}: table "
          f"{got.size} solutions, model {want.size}, model without the flagged row {old.size}")
    assert got.size == want.size
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-3)
    assert old.size != got.size or not np.allclose(old, got, rtol=0, atol=1e-3)


@pytest.fixture(scope="module")
def steps(tmp_path_factory):
    """A 1 Jy point source, one 60 s scan of 6 s integrations, antenna phases that step every 3
    integrations: inside each 3-integration solution the data are coherent, so gaincal's solution
    time is the weighted mean of its rows' times, and the solutions differ from interval to
    interval (which one a row receives is visible)."""
    import casatasks
    import casatools
    from tests.casa_sim import simulate_ms
    work = tmp_path_factory.mktemp("ajisai_soltime")
    ms = simulate_ms(str(work / "steps.ms"), nant=8, freq_ghz=(230.0,), nchan=(4,),
                     chan_mhz=(250.0,), scans=1, scan_s=60.0, int_s=6.0, noise="1mJy",
                     phase_rms_deg=0.0, seed=4, source=((1.0, 0.0, 0.0, 0.001, 0.001, 0.0),))
    rng = np.random.default_rng(7)
    step_phase = rng.uniform(-1.0, 1.0, (4, 8))            # (interval, antenna) rad
    tb = casatools.table()
    tb.open(ms, nomodify=False)
    try:
        t = tb.getcol("TIME")
        a1 = tb.getcol("ANTENNA1")
        a2 = tb.getcol("ANTENNA2")
        k = np.searchsorted(np.unique(t), t) // 3
        d = tb.getcol("DATA")
        tb.putcol("DATA", d * np.exp(1j * (step_phase[k, a1] - step_phase[k, a2]))[None, None, :])
    finally:
        tb.close()
    casatasks.clearcal(vis=ms, addmodel=True)
    return {"work": work, "ms": ms, "t0": float(t.min())}


def _edit_rows(ms, times, fn):
    import casatools
    tb = casatools.table()
    tb.open(ms, nomodify=False)
    try:
        t = tb.getcol("TIME")
        sel = np.zeros(t.size, bool)
        for x in times:
            sel |= np.abs(t - x) < 1e-3
        fn(tb, sel)
    finally:
        tb.close()


def _solve_and_compare(ms, ct_name, cand_n=3):
    import casatasks
    from ajisai import solint as so
    from ajisai.ms_utils import read_vis_summary
    vs = read_vis_summary(ms)
    geom = so.geometry(vs)
    cand = [c for c in so.geometry_candidates(geom) if c.get("n_int") == cand_n][0]
    ct = ms.replace(".ms", ct_name)
    shutil.rmtree(ct, ignore_errors=True)
    casatasks.gaincal(vis=ms, caltable=ct, solint=cand["gaincal_solint"], combine="", calmode="p",
                      gaintype="T", refant="0", minsnr=0.0, minblperant=4)
    tab = _cols(ct, ["TIME", "ANTENNA1", "CPARAM"])
    got = np.sort(tab["TIME"][tab["ANTENNA1"] == 1])
    units = so.solution_units(vs, geom, cand, combine_spw=False)
    plain = _without_time_info(vs)
    old = np.sort(so.solution_units(plain, so.geometry(plain), cand, combine_spw=False).time)
    return vs, geom, cand, units, ct, tab, got, np.sort(units.time), old


def test_solution_time_is_weighted_and_nearest_follows_it(steps):
    """WEIGHT x10 on integrations 0 and 3 (audit exp16): solutions at 1.5 s and 19.5 s from the
    scan start; applycal 'nearestPD' gives integration 2 the second interval's solution."""
    import casatasks
    from ajisai import solint as so
    ms = str(steps["work"] / "w10.ms")
    shutil.rmtree(ms, ignore_errors=True)
    shutil.copytree(steps["ms"], ms)
    t0 = steps["t0"]

    def up(tb, sel):
        w = tb.getcol("WEIGHT")
        w[:, sel] *= 10.0
        tb.putcol("WEIGHT", w)
        tb.putcol("SIGMA", 1.0 / np.sqrt(w))
    _edit_rows(ms, [t0, t0 + 18.0], up)
    vs, geom, cand, units, ct, tab, got, want, old = _solve_and_compare(ms, "_w10.G")
    print(f"\n[solution time] table {np.round(got - t0, 4).tolist()}; model "
          f"{np.round(want - t0, 4).tolist()}; unweighted model {np.round(old - t0, 4).tolist()}")
    np.testing.assert_allclose(got - t0, [1.5, 19.5, 42.0, 54.0], rtol=0, atol=1e-3)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-3)
    assert np.max(np.abs(old - got)) > 1.0
    # nearest interpolation: the solution each row receives
    casatasks.applycal(vis=ms, gaintable=[ct], interp=["nearestPD"], applymode="calonly",
                       calwt=[False], flagbackup=False)
    import casatools
    tb = casatools.table()
    tb.open(ms)
    try:
        q = tb.query("ANTENNA1==0 && ANTENNA2==5")
        try:
            rt = q.getcol("TIME")
            applied = np.angle(q.getcol("CORRECTED_DATA")[0, 0] / q.getcol("DATA")[0, 0])
        finally:
            q.close()
    finally:
        tb.close()
    sol5 = np.angle(tab["CPARAM"][0, 0, tab["ANTENNA1"] == 5])[np.argsort(tab["TIME"][tab["ANTENNA1"] == 5])]
    # refant 0: CORRECTED = DATA / (g_0 g_5*) = DATA exp(i phi_5)
    took = [int(np.argmin(np.abs(np.angle(np.exp(1j * (a - sol5)))))) for a in applied]
    zero = np.zeros(rt.size, int)
    lo, _, _ = so._row_nearest(units, rt, zero, zero)
    print(f"[solution time] CASA nearest took solutions {took}; model {lo.tolist()}")
    assert took[2] == 1 and took == lo.tolist()


def test_solution_time_drops_channels_with_a_flagged_hand(steps):
    """Half the channels flagged on integration 0 count half; integration 3 with one hand flagged
    counts nothing (gaintype 'T' drops a channel with either hand flagged)."""
    ms = str(steps["work"] / "flags.ms")
    shutil.rmtree(ms, ignore_errors=True)
    shutil.copytree(steps["ms"], ms)
    t0 = steps["t0"]

    def half(tb, sel):
        f = tb.getcol("FLAG")
        f[:, :2, sel] = True
        tb.putcol("FLAG", f)

    def one_hand(tb, sel):
        f = tb.getcol("FLAG")
        f[0, :, sel] = True
        tb.putcol("FLAG", f)
    _edit_rows(ms, [t0], half)
    _edit_rows(ms, [t0 + 18.0], one_hand)
    vs, geom, cand, units, ct, tab, got, want, old = _solve_and_compare(ms, "_flags.G")
    print(f"\n[solution time] flags: table {np.round(got - t0, 4).tolist()}; model "
          f"{np.round(want - t0, 4).tolist()}; unweighted model {np.round(old - t0, 4).tolist()}")
    # (0 x 0.5 + 6 + 12) / 2.5 = 7.2 s; (24 + 30) / 2 = 27 s
    np.testing.assert_allclose(got - t0, [7.2, 27.0, 42.0, 54.0], rtol=0, atol=1e-3)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-3)
