"""
Tests of ``ajisai.solint``: the numbers of the data-driven solint rules.

These are the executable form of ASEBI SPEC v0.3 section 10.2 and appendices
C.2/C.3 (DESIGN_REVIEW P17/P18): the CASA loss table, the closed forms of
T* and V(T), the interpolation geometry kappa, the candidate grid, G1, the
held-out G2 with rules A/B and the conditional two-sided correction, the
acceptance floor, and the stage logic on simulated phase screens. No CASA.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ajisai import solint as so
from tests import solint_sim as sim


# ---------------------------------------------------------------------------
# CASA loss table (Spec C.2)
# ---------------------------------------------------------------------------
def test_loss_table_values_are_the_spec_table():
    assert so.LOSS_TABLE_SNR == (2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0)
    assert so.LOSS_TABLE_NEG_LN_C == (0.844, 0.400, 0.198, 0.0856, 0.0492, 0.0233, 0.0113,
                                      0.0027)
    np.testing.assert_allclose(so.casa_loss(so.LOSS_TABLE_SNR), so.LOSS_TABLE_NEG_LN_C,
                               rtol=1e-12)


def test_loss_table_interpolation_and_ends():
    snr = np.array([1.0, 2.2, 4.5, 6.0, 15.0, 40.0])
    L = so.casa_loss(snr)
    assert np.all(np.diff(L) < 0)                       # loss falls with SNR
    assert so.casa_loss(4.5) < 0.0856 and so.casa_loss(4.5) > 0.0492
    assert so.casa_loss(40.0) == pytest.approx(0.0027 / 4)   # 1/SNR^2 beyond the table
    assert so.casa_loss(1.0) > 0.844                    # noise: steeper than the table start
    assert np.isinf(so.casa_loss(0.0)) and np.isinf(so.casa_loss(-1.0))
    # true SNR 5: ~4.8-5.0 % coherence loss when solved by CASA (Spec 10.2 item 4)
    assert 1 - math.exp(-so.casa_loss(5.0)) == pytest.approx(0.048, abs=0.002)


def test_rule_constants():
    assert so.G1_SNR_MIN == 5.0
    assert so.GAINCAL_MINSNR == 3.0
    assert (so.ACCEPT_SIGMA, so.ACCEPT_FLOOR) == (1.0, 0.001)
    assert so.STAGE1_SKIP_SIGMA == 2.0
    assert so.MAX_PHASE_STAGES == 3
    assert so.BAND_SPLIT_BELOW_NINT == 4


# ---------------------------------------------------------------------------
# Closed forms (Spec C.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta, lo, hi", [(5 / 3, 0.34, 0.36), (1.0, 0.77, 0.80),
                                          (2 / 3, 0.89, 0.92)])
def test_linear_interpolation_coefficient_is_the_spec_fraction_of_the_box(beta, lo, hi):
    box = so.box_coefficient(beta)
    assert box == pytest.approx(1 / ((beta + 1) * (beta + 2)))
    assert so.atmospheric_coefficient(beta, "nearest") == box
    assert lo <= so.atmospheric_coefficient(beta, "linear") / box <= hi


def test_box_coefficient_from_the_variogram_identity():
    """The discrete machinery reproduces a = 1/((beta+1)(beta+2)) for one interval."""
    beta = 1.3
    x = (np.arange(400) + 0.5) / 400
    var = so._interval_variance(x, [x], np.array([0.5]), lambda lag: np.abs(lag) ** beta)
    assert var.mean() == pytest.approx(so.box_coefficient(beta), rel=2e-3)


@pytest.mark.parametrize("x, beta, expected", [(2.0, 1.0, 1.25), (2.0, 5 / 3, 1.50)])
def test_variance_ratio_at_twice_the_optimum(x, beta, expected):
    assert so.variance_ratio(x, beta) == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize("beta", [2 / 3, 1.0, 5 / 3])
def test_t_star_minimises_v_and_the_ratio_formula_holds(beta):
    tau0, r = 300.0, 1.7
    a = so.atmospheric_coefficient(beta, "linear")
    T = so.optimal_solint(tau0, beta, r, a=a)
    V = lambda t: so.residual_variance_closed(t, tau0, beta, r, a)  # noqa: E731
    assert V(T) < V(0.97 * T) and V(T) < V(1.03 * T)
    # V* = (1 + 1/beta) kappa / SNR(T*)^2 and V(xT*)/V* = (x^beta + beta/x)/(1+beta)
    assert V(T) == pytest.approx((1 + 1 / beta) * (2 / 3) / (r ** 2 * T))
    for x in (0.5, 1.5, 2.0, 3.0):
        assert V(x * T) / V(T) == pytest.approx(so.variance_ratio(x, beta))
    # at the optimum the thermal term is beta times the atmospheric term
    assert (2 / 3) / (r ** 2 * T) == pytest.approx(beta * a * (T / tau0) ** beta)


def test_t_star_rule_of_thumb_for_kolmogorov_linear():
    """beta = 5/3, linear interpolation: T* ~ 2.5 tau0^(5/8) r^(-3/4) s."""
    for tau0, r in ((100.0, 1.0), (1000.0, 2.0), (300.0, 0.6)):
        T = so.optimal_solint(tau0, 5 / 3, r)
        assert T / (tau0 ** 0.625 * r ** -0.75) == pytest.approx(2.5, rel=0.03)


# ---------------------------------------------------------------------------
# Interpolation geometry kappa (Spec 10.2 item 1, C.2)
# ---------------------------------------------------------------------------
def test_kappa_uniform_interior_grid_is_two_thirds():
    sol = np.arange(20) * 10.0 + 5.0                  # interval centres
    t = np.linspace(sol[0], sol[-1], 20001)[:-1]      # interior only
    assert so.kappa_geometry(t, sol) == pytest.approx(2 / 3, abs=1e-3)


def test_kappa_one_solution_per_eb_and_per_integration_is_one():
    t = np.arange(100) * 6.0
    assert so.kappa_geometry(t, [t.mean()]) == pytest.approx(1.0)
    assert so.kappa_geometry(t, t) == pytest.approx(1.0)


def test_kappa_of_the_even_odd_halves():
    """G2 halves: the training centres fall between held integrations."""
    t = np.arange(40) * 6.0
    # T = 2 IT: one training integration per interval, every held one halfway: kappa = 1/2
    assert so.kappa_geometry(t[1::2][:-1], t[0::2]) == pytest.approx(0.5)
    # T = 4 IT: centres at the mean of two even integrations; held at c = 1 and f = 1/2
    sol = np.array([t[k:k + 4][0::2].mean() for k in range(0, 40, 4)])
    held = t[1::2][2:-2]
    assert so.kappa_geometry(held, sol) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Candidates (Spec 10.2 item 3)
# ---------------------------------------------------------------------------
def test_candidates_are_divisors_plus_scan_and_eb():
    c = so.candidate_solints(12, 6.0, scans_per_obs=5, eb_span_s=3600.0)
    assert [x["label"] for x in c] == ["6s", "12s", "18s", "24s", "36s", "scan", "EB"]
    assert [x["n_int"] for x in c[:5]] == [1, 2, 3, 4, 6]
    assert all(x["divisor"] for x in c)
    assert c[-1]["T_s"] == 3600.0 and c[-1]["time_combine"] == "scan"
    assert c[-2]["gaincal_solint"] == "inf" and c[-2]["time_combine"] == ""


def test_coarse_divisors_get_intermediate_multiples():
    """10 integrations: 1, 2, 5 are divisors; 5/2 > 2 adds 3 (non-divisor, fractional end)."""
    c = so.candidate_solints(10, 6.05, scans_per_obs=7)
    ints = [x for x in c if x["kind"] == "int"]
    assert [x["n_int"] for x in ints] == [1, 2, 3, 5]
    assert [x["divisor"] for x in ints] == [True, True, False, True]
    n = [x["n_int"] for x in ints] + [10]
    assert all(b / a <= 2.0 for a, b in zip(n[:-1], n[1:]))
    # a prime scan length: 7 -> 1, 2, 3, 5 (ratios <= 2)
    assert [x["n_int"] for x in so.candidate_solints(7, 6.0) if x["kind"] == "int"] == [1, 2, 3, 5]


def test_gaincal_solint_holds_exactly_n_integrations():
    c = [x for x in so.candidate_solints(10, 6.048, scan_nint_max=12) if x["kind"] == "int"]
    assert c[0]["n_int"] == 1 and c[0]["gaincal_solint"] == "int"
    for x in c[1:]:
        n = x["n_int"]
        m = math.ceil(12 / n)
        assert float(x["gaincal_solint"].rstrip("s")) == pytest.approx(n * 6.048 - 6.048 / (4 * m),
                                                                       abs=1e-3)
        assert x["T_s"] == pytest.approx(n * 6.048)
    # with timestamp jitter, every chunk of a 12-integration scan holds exactly n integrations
    rng = np.random.default_rng(0)
    t = np.arange(12) * 6.048 + rng.uniform(-1e-3, 1e-3, 12)
    t[0] = 0.0
    for x in c[1:]:
        w = float(x["gaincal_solint"].rstrip("s"))
        chunk = np.floor((t - t[0]) / w).astype(int)
        assert chunk.tolist() == (np.arange(12) // x["n_int"]).tolist()


def test_one_scan_per_observation_has_no_scan_candidate():
    c = so.candidate_solints(30, 2.0, scans_per_obs=1)
    assert [x["kind"] for x in c][-1] == "eb"
    assert "scan" not in [x["kind"] for x in c]


# ---------------------------------------------------------------------------
# Candidate step for EBs of different integration times (Spec 10.2 item 3, D34 F3)
# ---------------------------------------------------------------------------
def test_integration_step_rule_constants():
    assert so.IT_MATCH_RTOL == 0.01
    assert so.IT_LCM_MAX_FACTOR == 4.0


def test_integration_step_of_dg_tau_b_is_6_048_s():
    st = so.integration_step([2.016, 6.048, 2.016], it_median=2.016)
    assert st["rule"] == "lcm" and st["warning"] is None
    assert st["step_s"] == pytest.approx(6.048, abs=1e-12)
    np.testing.assert_allclose(st["distinct_s"], [2.016, 6.048])
    assert st["multiples"] == [3, 1]


def test_integration_step_single_time_changes_nothing():
    for its, med in (([6.048], 6.048), ([6.048, 6.048, 6.048], 6.048), ([], 2.0)):
        st = so.integration_step(its, it_median=med)
        assert st["rule"] == "single" and st["step_s"] == med and st["warning"] is None


@pytest.mark.parametrize("its, rule, n_distinct, step", [
    ([2.016, 2.0159], "single", 1, None),          # within 1 %: one value, step = the median
    ([2.016, 2.0159, 6.048], "lcm", 2, 6.048),     # 6.048 / 2.01595 = 3.00007
    ([2.0, 6.05], "lcm", 2, 6.05),                 # 3.025: an integer multiple within 1 %
    ([2.0, 2.03], "median_fallback", 2, None),     # 1.5 % apart: two values, no multiple <= 4x
    ([2.0, 6.15], "median_fallback", 2, None),     # 3.075: 2.5 % off any integer multiple
])
def test_integration_step_one_percent_tolerance(its, rule, n_distinct, step):
    st = so.integration_step(its, it_median=its[0])
    assert st["rule"] == rule and len(st["distinct_s"]) == n_distinct
    assert st["step_s"] == pytest.approx(step if step is not None else its[0])


@pytest.mark.parametrize("its, lcm, rule", [
    ([2.0, 7.0], 14.0, "lcm"),              # 14 <= 4 x 7
    ([3.0, 7.0], 21.0, "lcm"),
    ([4.0, 7.0], 28.0, "lcm"),              # exactly 4 x the longest: accepted
    ([5.0, 7.0], 35.0, "median_fallback"),  # 35 > 28
])
def test_integration_step_lcm_up_to_four_times_the_longest(its, lcm, rule):
    st = so.integration_step(its, it_median=6.0)
    assert st["lcm_s"] == pytest.approx(lcm) and st["rule"] == rule
    if rule == "lcm":
        assert st["step_s"] == pytest.approx(lcm) and st["warning"] is None
    else:
        assert st["step_s"] == 6.0
        assert "35" in st["warning"] and "median" in st["warning"]


def _mixed_summary(n_ant=5, jitter=2e-4, seed=0):
    """Two EBs: 2.016 s (scans of 30 and 31 integrations) and 6.048 s (two scans of 10)."""
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(n_ant, 1)
    nb = a1.size
    cols = {k: [] for k in ("time", "obs", "scan")}
    for o, it, nints in ((0, 2.016, (30, 31)), (1, 6.048, (10, 10))):
        t0 = 1e5 * o
        for sc, n in enumerate(nints, 1):
            t = t0 + np.arange(n) * it + rng.uniform(-jitter, jitter, n)
            t0 = t[-1] + 120.0
            cols["time"].append(np.repeat(t, nb))
            cols["obs"].append(np.full(n * nb, o))
            cols["scan"].append(np.full(n * nb, sc))
    time = np.concatenate(cols["time"])
    n = time.size
    return so.VisSummary(time=time, obs=np.concatenate(cols["obs"]),
                         scan=np.concatenate(cols["scan"]), spw=np.zeros(n, int),
                         a1=np.tile(a1, n // nb), a2=np.tile(a2, n // nb), y=np.ones(n, complex),
                         P=np.ones(n), antenna_names=[f"A{k}" for k in range(n_ant)])


def test_mixed_integration_times_step_by_the_least_common_multiple():
    vs = _mixed_summary()
    geom = so.geometry(vs)
    assert geom.int_time[0] == pytest.approx(2.016, abs=1e-3)
    assert geom.int_time[1] == pytest.approx(6.048, abs=1e-3)
    assert geom.it_step_record["rule"] == "lcm"
    assert geom.it_step == pytest.approx(6.048, abs=1e-3)
    # scan lengths in steps: 10, 10.33 (31 x 2.016 s), 10, 10
    np.testing.assert_allclose(sorted(geom.scan_nstep), [10, 10, 10, 31 / 3], atol=1e-3)
    assert geom.scan_nstep_median == pytest.approx(10.0, abs=1e-3) and geom.scan_nstep_max == 11
    cands = so.geometry_candidates(geom)
    ints = [c for c in cands if c["kind"] == "int"]
    assert [c["n_int"] for c in ints] == [1, 2, 3, 5]
    assert [c["kind"] for c in cands[len(ints):]] == ["scan", "eb"]
    for c in ints:
        d = c["n_int"]
        assert c["T_s"] == pytest.approx(d * geom.it_step)
        assert c["n_int_by_it_s"] == {"2.016": 3 * d, "6.048": d}
        assert c["gaincal_solint"] != "int"
        # one solint for both EBs: below n IT_i by a quarter of the common window
        m = math.ceil(11 / d)
        assert float(c["gaincal_solint"].rstrip("s")) == pytest.approx(
            d * geom.it_step - geom.int_time[0] / (4 * m), abs=2e-3)
    assert cands[len(ints)]["n_int_by_it_s"] == {"2.016": 30, "6.048": 10}
    # T = 1 step solves single integrations of the 6.048 s EB: not scorable by even/odd
    assert so._single_integration(ints[0]) and not so._single_integration(ints[1])


def test_mixed_integration_times_line_up_with_every_eb():
    """gaincal's chunks hold exactly n_i = T / IT_i integrations of every EB (last one fractional)."""
    vs = _mixed_summary()
    geom = so.geometry(vs)
    for c in [x for x in so.geometry_candidates(geom) if x["kind"] == "int"]:
        units = so.solution_units(vs, geom, c, combine_spw=False)
        for o, key in ((0, "2.016"), (1, "6.048")):
            n_i = c["n_int_by_it_s"][key]
            sel = vs.obs == o
            for sc in np.unique(vs.scan[sel]):
                rows = sel & (vs.scan == sc)
                k = geom.k_in_scan[rows]
                u = units.of_row[rows]
                # rows of the same unit are exactly the integrations k // n_i of the scan
                _, chunk = np.unique(u, return_inverse=True)
                assert np.array_equal(np.asarray(chunk).ravel(), k // n_i)
                nt = units.n_times[np.unique(u)]
                assert np.all(nt[:-1] == n_i) and 1 <= nt[-1] <= n_i


def test_single_integration_time_gives_the_previous_candidates():
    vs, _ = sim.simulate(n_ant=8, n_scans=3, n_int=12, it=6.048, seed=1)
    geom = so.geometry(vs)
    assert geom.it_step_record["rule"] == "single" and geom.it_step == geom.it_ref
    before = so.candidate_solints(geom.scan_nint_median, geom.it_ref, geom.max_scans_per_obs,
                                  sum(geom.obs_span.values()), int(geom.scan_nint.max()))
    assert so.geometry_candidates(geom) == before
    assert [c["gaincal_solint"] for c in before][:3] == ["int", "11.8440s", "17.7660s"]
    # two EBs whose integration times agree within 1 %: one value, candidates as before
    a, pa = sim.simulate(n_ant=8, n_scans=3, n_int=12, it=6.048, seed=1)
    b, pb = sim.simulate(n_ant=8, n_scans=3, n_int=12, it=6.0479, seed=2, obs_offset=1e5)
    vs2, _ = sim.concat([(a, pa), (b, pb)])
    g2 = so.geometry(vs2)
    assert g2.it_step_record["rule"] == "single" and g2.it_step == g2.it_ref
    assert so.geometry_candidates(g2) == so.candidate_solints(
        g2.scan_nint_median, g2.it_ref, g2.max_scans_per_obs, sum(g2.obs_span.values()),
        int(g2.scan_nint.max()))


def test_no_common_multiple_falls_back_to_the_median_with_a_warning():
    a, pa = sim.simulate(n_ant=6, n_scans=2, n_int=12, it=5.0, seed=1)
    b, pb = sim.simulate(n_ant=6, n_scans=2, n_int=12, it=7.0, seed=2, obs_offset=1e5)
    vs, _ = sim.concat([(a, pa), (b, pb)])
    with pytest.warns(UserWarning, match="least common multiple is 35"):
        geom = so.geometry(vs)
    assert geom.it_step_record["rule"] == "median_fallback"
    assert geom.it_step == geom.it_ref
    assert so.geometry_candidates(geom) == so.candidate_solints(
        geom.scan_nint_median, geom.it_ref, geom.max_scans_per_obs, sum(geom.obs_span.values()),
        int(geom.scan_nint.max()))


def test_mixed_integration_times_end_to_end():
    """DG Tau B-like concatenation (2.016 s and 6.048 s EBs): the stages run on the 6.048 s step."""
    a, pa = sim.simulate(n_ant=12, n_scans=3, n_int=30, it=2.016, gap=60.0, r=3.0, tau0=300.0,
                         seed=31)
    b, pb = sim.simulate(n_ant=12, n_scans=3, n_int=10, it=6.048, gap=60.0, r=3.0, tau0=300.0,
                         seed=32, obs_offset=1e5)
    vs, _ = sim.concat([(a, pa), (b, pb)])
    recs, chain, final = sim.run_loop(vs)
    assert recs[0]["decision"] == "apply"
    for r in recs:
        meas = r["measured"]
        assert meas["it_step"]["rule"] == "lcm" and meas["it_step_s"] == pytest.approx(6.048, abs=1e-3)
        for c in r["candidates"]:
            if c["kind"] == "int":
                assert c["T_s"] / 6.048 == pytest.approx(round(c["T_s"] / 6.048), abs=1e-6)
                assert c["n_int_by_it_s"]["2.016"] == 3 * c["n_int_by_it_s"]["6.048"]
    assert len(chain) >= 2 and chain[1].cand["kind"] in ("int", "scan")


# ---------------------------------------------------------------------------
# Solution units reproduce gaincal's chunking (CASA 6.5.1, combine='')
# ---------------------------------------------------------------------------
def _two_scan_summary(n_ant=5):
    """Two scans of 10 integrations (6 s), 66 s apart, like the CASA experiment."""
    t = np.concatenate([np.arange(10) * 6.0, 120.0 + np.arange(10) * 6.0])
    scan = np.repeat([1, 2], 10)
    a1, a2 = np.triu_indices(n_ant, 1)
    nb = a1.size
    vs = so.VisSummary(time=np.repeat(t, nb), obs=np.zeros(t.size * nb, int),
                       scan=np.repeat(scan, nb), spw=np.zeros(t.size * nb, int),
                       a1=np.tile(a1, t.size), a2=np.tile(a2, t.size),
                       y=np.ones(t.size * nb, complex), P=np.ones(t.size * nb),
                       antenna_names=[f"A{k}" for k in range(n_ant)])
    return vs


@pytest.mark.parametrize("solint, times", [
    # gaincal(combine='') on two 10-integration scans (6 s), CASA 6.5.1 and 6.5.6
    ("9.000s", [3.0, 12.0, 21.0, 30.0, 39.0, 48.0, 54.0,
                123.0, 132.0, 141.0, 150.0, 159.0, 168.0, 174.0]),
    ("15.000s", [6.0, 21.0, 36.0, 51.0, 126.0, 141.0, 156.0, 171.0]),
    ("17.9s", [6.0, 24.0, 42.0, 54.0, 126.0, 144.0, 162.0, 174.0]),
    ("27.000s", [12.0, 39.0, 54.0, 132.0, 159.0, 174.0]),
    ("int", [float(x) for x in list(range(0, 60, 6)) + list(range(120, 180, 6))]),
])
def test_units_follow_gaincal_time_grid(solint, times):
    vs = _two_scan_summary()
    geom = so.geometry(vs)
    assert geom.int_time[0] == pytest.approx(6.0) and geom.scan_nint_median == 10
    u = so.solution_units(vs, geom, {"kind": "int", "n_int": 0, "gaincal_solint": solint},
                          combine_spw=False)
    assert sorted(u.time.tolist()) == times


def test_scan_and_eb_units():
    vs = _two_scan_summary()
    geom = so.geometry(vs)
    scan = so.solution_units(vs, geom, {"kind": "scan"}, combine_spw=False)
    assert sorted(scan.time.tolist()) == [27.0, 147.0]
    eb = so.solution_units(vs, geom, {"kind": "eb"}, combine_spw=False)
    assert eb.time.tolist() == [87.0]


def test_training_half_units_use_the_training_times():
    vs = _two_scan_summary()
    geom = so.geometry(vs)
    even = (geom.k_in_scan % 2) == 0
    u = so.solution_units(vs, geom, {"kind": "int", "n_int": 2, "gaincal_solint": "11.250s"},
                          combine_spw=False, rows=even)
    assert sorted(u.time.tolist())[:5] == [0.0, 12.0, 24.0, 36.0, 48.0]


# ---------------------------------------------------------------------------
# applycal behaviour: open interval around flagged solutions, constant ends
# ---------------------------------------------------------------------------
def test_flagged_solution_removes_the_open_interval_and_ends_are_constant():
    units = so.Units(of_row=np.zeros(1, int), time=np.array([3.0, 15.0, 27.0, 39.0]),
                     obs=np.zeros(4, int), group=np.zeros(4, int), n_times=np.full(4, 2))
    t = np.arange(0, 45, 6.0)       # 0, 6, ..., 42
    lo, hi, f = so._row_brackets(units, t, np.zeros(t.size, int), np.zeros(t.size, int))
    values = np.array([[0.1], [0.2], [0.3], [0.4]])
    good = np.array([[True], [True], [False], [True]])
    val, ok = so._row_values(values, good, lo, hi, f, np.zeros(t.size, int), phase=True)
    # the 27 s solution is flagged: 18, 24, 30, 36 s (between 15 and 39) lose the correction
    assert ok.tolist() == [True, True, True, False, False, False, False, True]
    assert val[0] == pytest.approx(0.1)          # before the first solution: constant
    assert val[-1] == pytest.approx(0.4)         # after the last: constant
    assert val[2] == pytest.approx(0.1 + 0.1 * (12 - 3) / 12)


def test_phase_interpolation_takes_the_short_way_round():
    units = so.Units(of_row=np.zeros(1, int), time=np.array([0.0, 10.0]), obs=np.zeros(2, int),
                     group=np.zeros(2, int), n_times=np.ones(2, int))
    t = np.array([5.0])
    lo, hi, f = so._row_brackets(units, t, np.zeros(1, int), np.zeros(1, int))
    values = np.array([[3.0], [-3.0]])           # 6 rad apart the long way, 0.28 rad short way
    val, ok = so._row_values(values, np.ones((2, 1), bool), lo, hi, f, np.zeros(1, int), True)
    assert abs(np.angle(np.exp(1j * (val[0] - 3.0)))) < 0.2


def test_nearest_weights_ties_go_to_the_earlier_solution():
    """CASA 6.5.1 applycal 'nearest': the nearest solution time, ties to the earlier one."""
    st = np.array([0.0, 18.0, 36.0, 48.0])
    t = np.array([-6.0, 0.0, 6.0, 9.0, 12.0, 18.0, 24.0, 27.0, 30.0, 42.0, 48.0, 54.0])
    lo, hi, f = so.nearest_weights(t, st)
    assert lo.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]
    assert hi.tolist() == lo.tolist() and f.tolist() == [0.0] * t.size


def test_nearest_rows_take_their_own_interval_and_a_flagged_one_is_not_skipped():
    # four intervals; interval 2 has no solution time (no training row in it)
    units = so.Units(of_row=np.array([0, 0, 1, 1, 1, 2, 2, 3]),
                     time=np.array([3.0, 15.0, np.nan, 39.0]), obs=np.zeros(4, int),
                     group=np.zeros(4, int), n_times=np.array([2, 3, 2, 1]))
    t = np.arange(0, 48, 6.0)       # 0, 6, ..., 42
    zero = np.zeros(t.size, int)
    lo, hi, f = so._row_nearest(units, t, zero, zero)
    # own interval; rows of interval 2 (30, 36 s) take the nearest solution time (39 s)
    assert lo.tolist() == [0, 0, 1, 1, 1, 3, 3, 3] and np.all(f == 0)
    values = np.array([[0.1], [0.2], [0.3], [0.4]])
    good = np.array([[True], [False], [True], [True]])
    val, ok = so._row_values(values, good, lo, hi, f, zero, phase=True)
    # the flagged solution of interval 1 leaves its own rows uncorrected (calonly)
    assert ok.tolist() == [True, True, False, False, False, True, True, True]
    np.testing.assert_allclose(val[ok], [0.1, 0.1, 0.4, 0.4, 0.4])


def test_kappa_and_the_box_coefficient_of_nearest_interpolation():
    sol = np.arange(20) * 10.0 + 5.0
    t = np.linspace(sol[0], sol[-1], 2001)[:-1]
    assert so.kappa_geometry(t, sol, "nearest") == 1.0
    # the discrete prediction with nearest weights is a = 1/((beta+1)(beta+2)) (Spec C.2)
    beta = 1.3
    members = [k + (np.arange(400) + 0.5) / 400 for k in range(5)]
    centres = np.array([m.mean() for m in members])
    ev = np.linspace(0.5, 4.5, 801)[:-1]
    var = so._interval_variance(ev, members, centres, lambda lag: np.abs(lag) ** beta, "nearest")
    assert var.mean() == pytest.approx(so.box_coefficient(beta), rel=3e-3)
    lin = so._interval_variance(ev, members, centres, lambda lag: np.abs(lag) ** beta)
    assert lin.mean() < var.mean()
    with pytest.raises(ValueError):
        so.kappa_geometry(t, sol, "cubic")


def test_applycal_interp_string():
    assert so.applycal_interp_string("linearPDperobs", "nearest") == "nearestPDperobs"
    assert so.applycal_interp_string("linearperobs", "nearest") == "nearestperobs"
    assert so.applycal_interp_string("linear,linear", "nearest") == "nearest,linear"
    assert so.applycal_interp_string("linearPDperobs") == "linearPDperobs"
    with pytest.raises(ValueError):
        so.applycal_interp_string("nearestperobs", "nearest")
    with pytest.raises(ValueError):
        so.applycal_interp_string("linearPDperobs", "cubic")


def test_interpolation_never_crosses_observations():
    units = so.Units(of_row=np.zeros(1, int), time=np.array([0.0, 100.0]), obs=np.array([0, 1]),
                     group=np.zeros(2, int), n_times=np.ones(2, int))
    t = np.array([50.0, 50.0])
    lo, hi, f = so._row_brackets(units, t, np.array([0, 1]), np.zeros(2, int))
    assert lo.tolist() == [0, 1] and hi.tolist() == [0, 1] and f.tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# G1 (Spec 10.2 item 4)
# ---------------------------------------------------------------------------
def test_g1_gate_per_candidate_and_per_antenna():
    snr = np.array([[8.0, 6.0, 4.0, 0.0],
                    [9.0, 5.5, 4.5, 7.0],
                    [7.0, 5.0, 3.0, 6.0]])
    g = so.g1_gate(snr)
    assert g["pass"] is True                     # median of the 11 solutions is >= 5
    assert g["failing_antennas"] == [2]          # antenna 2 never reaches 5
    assert g["good"][0, 2] == False and g["good"][2, 1] == True  # noqa: E712
    assert not g["good"][0, 3]                   # no data: not a solution
    assert so.g1_gate(snr * 0.5)["pass"] is False


def test_caltable_gate_flags_rows_below_the_gate_by_nearest_time():
    gt = so.GateTable(time=np.array([10.0, 30.0, 10.0]), obs=np.array([0, 0, 0]),
                      group=np.array([0, 0, 1]), snr=np.array([[6.0, 4.0], [3.0, 9.0],
                                                               [5.0, 5.0]]),
                      n_times=np.array([3, 3, 3]), int_time={0: 6.0})
    res = so.caltable_gate(gt, cal_time=[10.2, 10.2, 29.9, 29.9, 10.0, 70.0],
                           cal_obs=[0] * 6, cal_spw=[0, 0, 0, 0, 1, 0], cal_ant=[0, 1, 0, 1, 1, 0])
    assert res["flag"].tolist() == [False, True, True, False, False, False]
    assert res["unmatched"] == 1                 # 70 s is not a predicted solution time


def test_caltable_gate_combined_spw():
    gt = so.GateTable(time=np.array([10.0]), obs=np.array([0]), group=np.array([-1]),
                      snr=np.array([[2.0, 8.0]]), n_times=np.array([3]), int_time={0: 6.0})
    res = so.caltable_gate(gt, [10.0, 10.0], [0, 0], [3, 3], [0, 1])
    assert res["flag"].tolist() == [True, False] and res["unmatched"] == 0


# ---------------------------------------------------------------------------
# Measured quantities on simulated data (Spec C.2, P18 3.2)
# ---------------------------------------------------------------------------
def test_noise_scale_and_snr_rate_are_recovered():
    vs, _ = sim.simulate(n_ant=12, n_scans=3, n_int=30, r=2.0, s=1.7, tau0=400.0, seed=11)
    geom = so.geometry(vs)
    s = so.noise_scale(vs, geom)
    assert s[0]["s"] == pytest.approx(1.7, rel=0.04)
    rates = so.snr_rates(vs, s, geom)
    assert rates["median"] == pytest.approx(2.0, rel=0.04)
    assert set(rates["per_antenna"]) == set(vs.antenna_names)


def test_lag_structure_function_recovers_the_screen():
    vs, _ = sim.simulate(n_ant=16, n_scans=4, n_int=30, r=4.0, tau0=200.0, beta=1.0, seed=2)
    geom = so.geometry(vs)
    lag = so.lag_structure_function(vs, so.noise_scale(vs, geom), geom)
    tau = np.array(lag["tau_s"])
    D = np.array(lag["D"])
    short = tau <= 60
    np.testing.assert_allclose(D[short], tau[short] / 200.0, atol=0.06)
    fit = so.fit_structure_function(lag["tau_s"], lag["D"], lag["sigma_D"])
    assert fit["significant"]
    assert 0.6 <= fit["beta"] <= 1.4
    assert 120 <= fit["tau0_s"] <= 330


def test_no_phase_structure_is_not_significant():
    vs, _ = sim.simulate(n_ant=12, n_scans=3, n_int=20, r=1.0, tau0=None, seed=4)
    geom = so.geometry(vs)
    lag = so.lag_structure_function(vs, so.noise_scale(vs, geom), geom)
    fit = so.fit_structure_function(lag["tau_s"], lag["D"], lag["sigma_D"])
    assert not fit["significant"]
    assert so.structure_model(fit)(np.array([10.0, 100.0])).tolist() == [0.0, 0.0]


def test_reduce_visibilities_parallel_hands_flags_and_halves():
    rng = np.random.default_rng(0)
    npol, nchan, nrow = 4, 4, 3
    data = rng.normal(size=(npol, nchan, nrow)) + 1j * rng.normal(size=(npol, nchan, nrow))
    model = np.full((npol, nchan, nrow), 2.0 + 0j)
    flag = np.zeros((npol, nchan, nrow), bool)
    flag[0, 0, 0] = True
    weight = np.full((npol, nrow), 0.5)
    corr = [9, 10, 11, 12]                      # XX XY YX YY
    out = so.reduce_visibilities(data, model, flag, weight, np.array([False, False, True]),
                                 corr, halves=True)
    ok = ~flag[[0, 3]]
    ok[:, :, 2] = False
    y = (0.5 * data[[0, 3]] * 2.0 * ok).sum(axis=(0, 1))
    np.testing.assert_allclose(out["y"], y)
    np.testing.assert_allclose(out["P"], (0.5 * 4.0 * ok).sum(axis=(0, 1)))
    np.testing.assert_allclose(out["S"], (data[[0, 3]] * ok).sum(axis=(0, 1)))
    assert out["n_valid"].tolist() == [7, 8, 0]
    np.testing.assert_allclose(out["y_half"].sum(axis=0), out["y"])
    np.testing.assert_allclose(out["P_half"].sum(axis=0), out["P"])


def test_phase_tracking_recovers_the_applied_phase():
    vs, _ = sim.simulate(n_ant=6, n_scans=2, n_int=5, r=2.0, seed=5)
    rp = np.random.default_rng(1).uniform(-3, 3, vs.n_rows)
    nxt = sim.rotate(vs, rp, vs)
    np.testing.assert_allclose(np.angle(np.exp(1j * (nxt.phi0 - rp))), 0.0, atol=1e-9)
    np.testing.assert_allclose(nxt.y0(), vs.y, rtol=1e-12)
    other, _ = sim.simulate(n_ant=6, n_scans=2, n_int=4, r=2.0, seed=5)
    with pytest.raises(ValueError, match="rows"):
        other.track_phase(vs)


# ---------------------------------------------------------------------------
# Rules A/B, choice, correction and acceptance (Spec 10.2 items 6-7, C.3)
# ---------------------------------------------------------------------------
def test_rule_a_b_and_the_longer_when_they_disagree():
    ev = [{"index": 1, "V_held": 0.050, "improvement_corr": 0.030},
          {"index": 2, "V_held": 0.046, "improvement_corr": 0.026},
          {"index": 3, "V_held": 0.044, "improvement_corr": 0.020},
          {"index": 5, "V_held": 0.060, "improvement_corr": 0.001}]
    passing = [0, 1, 2, 3, 4, 5]
    assert so.rule_a(ev) == 1
    assert so.rule_b(ev, passing) == 2           # argmin 3 -> one shorter
    T = {0: 6.0, 1: 12.0, 2: 24.0, 3: 36.0, 4: 60.0, 5: 3600.0}
    k, why = so.choose_solint(1, 2, T.get)
    assert k == 2 and "longer" in why
    assert so.choose_solint(3, 3, T.get) == (3, "rules A and B agree")


def test_choose_between_a_linear_candidate_and_its_nearest_variant():
    """Same interval, different interpolation (D34 F2): rule A's answer."""
    T = {3: 24.0, 7: 24.0, 8: 30.0}
    k, why = so.choose_solint(7, 3, T.get)
    assert k == 7 and "different interpolation" in why
    assert so.choose_solint(3, 8, T.get)[0] == 8          # otherwise the longer, as before


def test_rule_b_at_the_shortest_passing_candidate_returns_it():
    ev = [{"index": 2, "V_held": 0.01}, {"index": 3, "V_held": 0.02}]
    assert so.rule_b(ev, [2, 3]) == 2


def test_conditional_two_sided_correction():
    # candidate not longer than the previous stage: both corrections
    c = so.corrected_improvement(dV_raw=-0.002, corr_cand=0.006, T_cand=24.0, T_prev=30.0,
                                 corr_prev=0.005)
    assert c["two_sided"] and c["improvement_corr"] == pytest.approx(0.003)
    # re-solving the same T: the corrections cancel
    c = so.corrected_improvement(-0.0001, 0.004, 60.0, 60.0, 0.004)
    assert c["improvement_corr"] == pytest.approx(0.0001)
    # longer than the previous stage: the candidate's correction only
    c = so.corrected_improvement(dV_raw=0.0004, corr_cand=0.00112, T_cand=3600.0, T_prev=60.0,
                                 corr_prev=0.004)
    assert not c["two_sided"] and c["improvement_corr"] == pytest.approx(0.00072)
    # stage 1 (no previous stage): candidate only
    c = so.corrected_improvement(-0.01, 0.002, 3600.0, None)
    assert c["improvement_corr"] == pytest.approx(0.012)


def test_acceptance_floor_2016_iteration_3_stops():
    """P18 test results 10: 2016 it3 (EB on top of scan) +0.00072 stops at the floor."""
    d = so.acceptance(3, 0.00072, sigma=0.00016)
    assert d["accept"] is False and d["threshold"] == pytest.approx(0.001)
    assert "floor" in d["reason"]
    # TM2 it4 candidate +0.00167 exceeds the floor (stopped there only by the stage limit)
    assert so.acceptance(3, 0.00167, sigma=0.0004)["accept"] is True
    # when 1 sigma is larger than the floor, 1 sigma decides
    d = so.acceptance(2, 0.004, sigma=0.0045)
    assert d["accept"] is False and d["threshold"] == pytest.approx(0.0045)
    assert so.acceptance(2, 0.0203, sigma=0.0046)["accept"] is True


def test_stage1_is_skipped_only_below_minus_two_sigma():
    # TM1-like: corrected change +0.0062 +- 0.0045 worse (-1.4 sigma) -> still applied
    assert so.acceptance(1, -0.0062, 0.0045)["accept"] is True
    assert so.acceptance(1, -0.0095, 0.0045)["accept"] is False
    assert so.acceptance(1, 0.0, 0.0)["accept"] is True


def test_combine_spw_decision():
    assert so.combine_spw_decision(4.0, 6.9)["combine_spw"] is True
    assert so.combine_spw_decision(6.0, 10.0)["combine_spw"] is False
    assert so.combine_spw_decision(2.0, 4.0)["combine_spw"] is False


def test_spwmap_for_combined_spw():
    assert so.spwmap_for_combined_spw({0: [0, 1, 2, 3]}, 4) == [0, 0, 0, 0]
    assert so.spwmap_for_combined_spw({0: [0, 1], 1: [2, 3]}, 4) == [0, 0, 2, 2]
    with pytest.raises(ValueError):
        so.spwmap_for_combined_spw({0: [0, 1], 1: [1, 2]}, 3)


def test_g3_ratio_and_comparison():
    vs, _ = sim.simulate(n_ant=8, n_scans=2, n_int=6, r=3.0, tau0=None, seed=3)
    before = so.g3_short_baseline_ratio(vs)
    assert before[0]["ratio"] == pytest.approx(1.0, abs=0.05)
    lim = so.g3_compare(before, before)[0]["limit"]
    assert lim == pytest.approx(max(3 * math.sqrt(2) * before[0]["sigma"] / before[0]["ratio"],
                                    0.01))
    after = {0: dict(before[0], ratio=before[0]["ratio"] * (1 + 1.5 * lim))}
    cmp = so.g3_compare(before, after)
    assert cmp[0]["exceeds"] and cmp[0]["relative_change"] == pytest.approx(1.5 * lim)
    assert not so.g3_compare(before, before)[0]["exceeds"]


# ---------------------------------------------------------------------------
# G2 on simulated phase screens (Spec 10.2 item 6, C.3, App. A)
# ---------------------------------------------------------------------------
def _g2_short_interval(r, tau0, seed=7):
    vs, _ = sim.simulate(n_ant=16, n_scans=4, n_int=20, r=r, tau0=tau0, beta=1.0, seed=seed)
    geom = so.geometry(vs)
    cands = so.candidate_solints(geom.scan_nint_median, geom.it_ref, geom.max_scans_per_obs)
    eb = cands[-1]
    c2 = [c for c in cands if c.get("n_int") == 2][0]
    s_rows = so._s_rows(vs, so.noise_scale(vs, geom))
    st1 = sim.production(vs, geom, eb, vs)
    chain = [so.StageSolve(1, eb, vs, s_rows, 0.0)]
    splits = so.make_splits(st1, geom, "time")
    bases = so.reference_bases(chain, geom, splits, False, 0, 4)
    g2 = so.holdout_coherence(st1, geom, c2, bases, splits, s_rows, False, 0, 4, gate=False)
    lc = so.loss_correction(st1, geom, c2, splits, s_rows, False)
    imp = so.corrected_improvement(g2["dV_raw"], lc["corr"], c2["T_s"], eb["T_s"], 0.0)
    # same-data metric: solve on all rows, evaluate on the same rows
    units = so.solution_units(st1, geom, c2, False)
    ph, ok = so._solve_units(st1, units, st1.y0(), st1.P, np.ones(st1.n_rows, bool), 0, 4)
    rp, okr = so.apply_units(st1, units, ph, ok)
    same = (abs(np.sum(st1.y0() * np.exp(-1j * rp))) / st1.P.sum(),
            abs(np.sum(st1.y)) / st1.P.sum())
    return g2, imp, same


def test_g2_rejects_noise_fitting_short_intervals_at_low_snr():
    g2, imp, (same_after, same_before) = _g2_short_interval(r=0.5, tau0=3000.0)
    assert same_after > same_before + 0.1        # the same data look much better ...
    assert g2["dV_raw"] > 3 * g2["sigma"]         # ... while the held-out data get worse
    assert so.acceptance(2, imp["improvement_corr"], g2["sigma"])["accept"] is False


def test_g2_accepts_short_intervals_at_high_snr():
    g2, imp, _ = _g2_short_interval(r=6.0, tau0=150.0)
    assert g2["dV_raw"] < -3 * g2["sigma"]
    assert so.acceptance(2, imp["improvement_corr"], g2["sigma"])["accept"] is True


# ---------------------------------------------------------------------------
# The stage logic end to end on simulated data
# ---------------------------------------------------------------------------
def test_bright_source_with_fast_phases_gets_a_short_interval():
    vs, ph = sim.simulate(n_ant=16, n_scans=4, n_int=20, r=3.0, tau0=200.0, beta=1.0, seed=3)
    times = np.unique(vs.time)
    recs, chain, final = sim.run_loop(vs)
    assert [r["decision"] for r in recs][:2] == ["apply", "apply"]
    assert recs[0]["chosen"] == "EB" and len(recs[0]["candidates"]) == 1
    second = recs[1]
    chosen = [c for c in second["candidates"] if c["label"] == second["chosen"]][0]
    assert chosen["kind"] == "int" and chosen["T_s"] <= 30.0
    assert chosen["g2"]["improvement_corr"] > chosen["g2"]["sigma"]
    # T_pred near the closed-form optimum (~12 s for tau0 200 s, beta 1, r 3)
    assert 6.0 <= second["measured"]["T_star_closed_s"] <= 25.0
    assert len(recs) <= so.MAX_PHASE_STAGES
    assert sim.true_coherence(final, ph, times) > 0.97
    after_eb = sim.production(vs, so.geometry(vs), recs[0]["_"]["chosen"], vs)
    assert sim.true_coherence(final, ph, times) > sim.true_coherence(after_eb, ph, times) + 0.05
    # records carry every number the spec lists
    for key in ("measured", "candidates", "rule_A", "rule_B", "chosen", "decision", "reason"):
        assert key in second
    assert {"s", "r", "D_eff", "D_fit", "g2_split"} <= set(second["measured"])


@pytest.mark.parametrize("seed", [1, 3])
def test_faint_source_stays_with_long_intervals(seed):
    """r = 0.6: G1 keeps everything shorter than 60 s out; only scan/EB solutions are applied.

    (Seed 1 also applies a third, neutral stage, EB on top of scan, through the
    one-sided correction; seed 3 stops after EB.)
    """
    vs, ph = sim.simulate(n_ant=16, n_scans=4, n_int=20, r=0.6, tau0=2000.0, beta=1.0, seed=seed)
    times = np.unique(vs.time)
    recs, chain, final = sim.run_loop(vs)
    assert recs[0]["decision"] == "apply"
    assert all(c.cand["kind"] in ("eb", "scan") for c in chain)
    g1 = {c["label"]: c["g1"]["pass"] for c in recs[1]["candidates"]}
    assert not g1["6s"] and not g1["12s"] and not g1["30s"] and g1["scan"] and g1["EB"]
    assert sim.true_coherence(final, ph, times) > sim.true_coherence(vs, ph, times)


def test_too_faint_for_any_solution_stops_at_g1():
    vs, _ = sim.simulate(n_ant=10, n_scans=2, n_int=10, r=0.15, tau0=None, seed=8)
    recs, chain, final = sim.run_loop(vs)
    assert len(recs) == 1 and recs[0]["decision"] == "stop" and not chain
    assert "G1" in recs[0]["reason"]


def test_short_scans_use_the_band_halves():
    vs, ph = sim.simulate(n_ant=12, n_scans=10, n_int=3, gap=30, r=3.0, tau0=300.0, seed=5,
                          halves=True, n_spw=2)
    geom = so.geometry(vs)
    assert so.split_mode(geom, True) == "band"
    assert so.split_mode(geom, False) == "time"
    recs, chain, final = sim.run_loop(vs)
    assert all(r["measured"]["g2_split"] == "band" for r in recs)
    assert recs[1]["decision"] == "apply"
    times = np.unique(vs.time)
    assert sim.true_coherence(final, ph, times) > 0.95


def test_records_are_json_serialisable():
    import json
    vs, _ = sim.simulate(n_ant=8, n_scans=2, n_int=8, r=3.0, tau0=200.0, seed=9)
    recs, _, _ = sim.run_loop(vs)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    for r in recs:
        json.dumps(so.strip_internal(r), default=conv)


# ---------------------------------------------------------------------------
# Breakdown of linear interpolation: nearest variants in G2 (Spec 10.2 item 6, D34 F2)
# ---------------------------------------------------------------------------
def test_nearest_trigger_constant():
    assert so.NEAREST_TRIGGER_2D == 1.0


@pytest.mark.parametrize("n_int, two_d", [
    (1, 2 * 6.0 / 60.0),            # point solutions: 2 D(6 s) = 0.2
    (2, 2 * (12.0 - 3.0) / 60.0),   # means of 2 integrations: 2 (S12 - S11) = 0.3
])
def test_adjacent_solution_structure_recovers_the_screen(n_int, two_d):
    """tau0 = 60 s, beta = 1, IT = 6 s: 2 D_eff from adjacent solutions after the noise is removed."""
    vs, _ = sim.simulate(n_ant=12, n_scans=3, n_int=24, r=4.0, tau0=60.0, beta=1.0, seed=22)
    geom = so.geometry(vs)
    s_rows = so._s_rows(vs, so.noise_scale(vs, geom))
    cand = [c for c in so.geometry_candidates(geom) if c.get("n_int") == n_int][0]
    rec = so.adjacent_solution_structure(vs, geom, cand, s_rows, False, 0, 4,
                                         dfun=lambda lag: np.abs(lag) / 60.0)
    assert rec["computed"] and not rec["trigger"]
    assert rec["lag_s"] == pytest.approx(6.0 * n_int)
    assert rec["n_pairs"] == 3 * (24 // n_int - 1)
    # noise: eps^2 = (N-1)/N / SNR^2 of both antennas in both solutions, SNR^2 = r^2 n IT
    assert rec["two_D_noise"] == pytest.approx(4 * (11 / 12) / (16 * 6.0 * n_int), rel=0.1)
    assert rec["two_D_eff"] == pytest.approx(two_d, rel=0.2)
    assert rec["two_D_eff_raw"] == pytest.approx(rec["two_D_eff"] + rec["two_D_noise"])
    assert rec["rms_adjacent_rad"] == pytest.approx(math.sqrt(rec["two_D_eff_raw"]))
    assert rec["two_D_model"] == pytest.approx(2 * 6.0 * n_int / 60.0)


def test_one_solution_per_eb_has_no_adjacent_solutions():
    vs, _ = sim.simulate(n_ant=8, n_scans=2, n_int=8, r=3.0, tau0=200.0, seed=9)
    geom = so.geometry(vs)
    eb = so.geometry_candidates(geom)[-1]
    rec = so.adjacent_solution_structure(vs, geom, eb, np.ones(vs.n_rows), False)
    assert rec["computed"] is False and rec["trigger"] is False and "EB" in rec["reason"]


def test_small_phase_steps_add_no_nearest_candidates():
    """2 D_eff(T_hat) well below 1 rad^2: linear only, the decisions of the v0.3 rules."""
    vs, _ = sim.simulate(n_ant=16, n_scans=4, n_int=20, r=3.0, tau0=200.0, beta=1.0, seed=3)
    recs, chain, final = sim.run_loop(vs)
    assert [r["decision"] for r in recs] == ["apply", "apply", "stop"]
    assert [r["chosen"] for r in recs] == ["EB", "12s", "scan"]
    assert recs[0]["interpolation"]["computed"] is False
    ip = recs[1]["interpolation"]
    assert ip["computed"] and not ip["trigger"] and ip["two_D_eff"] < 0.3
    assert {"rms_adjacent_rad", "rms_adjacent_corr_rad", "two_D_eff_raw", "two_D_noise",
            "two_D_eff", "two_D_model", "lag_s", "threshold_two_D"} <= set(ip)
    assert all("interp" not in c for r in recs for c in r["candidates"])
    assert all(r["chosen_interp"] == "linear" for r in recs)


def test_large_phase_jumps_add_nearest_candidates_to_g2():
    """Antenna phase jumps (2 D_eff(T_hat) = 2.3 rad^2): nearest variants evaluated like the others.

    The held-out coherence decides; the test does not force a winner (here the
    nearest variant of 24 s is chosen by both rules).
    """
    vs, _ = sim.simulate(n_ant=16, n_scans=4, n_int=20, r=1.5, tau0=2000.0, beta=1.0, seed=2,
                         jumps=(1 / 30.0, 2.0))
    recs, chain, final = sim.run_loop(vs)
    second = recs[1]
    ip = second["interpolation"]
    assert ip["trigger"] and 1.5 < ip["two_D_eff"] < 3.5
    assert ip["rms_adjacent_rad"] > 1.0 and ip["candidate"] == second["T_pred"]
    cands = second["candidates"]
    near = [c for c in cands if c.get("interp") == "nearest"]
    assert near and all(c["label"].endswith("_nearest") for c in near)
    for c in near:
        lin = cands[c["linear_index"]]
        assert lin["kind"] in ("int", "scan") and lin["g1"]["pass"]
        assert (c["T_s"], c["gaincal_solint"], c["g1"]) == (lin["T_s"], lin["gaincal_solint"],
                                                             lin["g1"])
        assert "V_pred" in c
    # every interval G2 evaluated with linear interpolation was evaluated with nearest too
    lin_eval = {c["label"] for c in cands if c["evaluated"] and "interp" not in c
                and c["kind"] in ("int", "scan")}
    near_eval = {c["label"][:-len("_nearest")] for c in near if c["evaluated"]}
    assert lin_eval and lin_eval <= near_eval
    for c in near:
        if c["evaluated"]:
            assert c["g2"]["kappa_held"] == 1.0
            assert {"C_held", "sigma", "improvement_corr"} <= set(c["g2"])
    assert (second["chosen_interp"] == "nearest") == second["chosen"].endswith("_nearest")
    # the chosen interpolation is carried by the stage into production and the next G2 reference
    if second["decision"] == "apply":
        assert chain[1].cand.get("interp", "linear") == second["chosen_interp"]
