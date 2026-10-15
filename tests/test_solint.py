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
