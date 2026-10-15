"""
Tests of ``schedule.mode="data"`` inside :class:`ajisai.AJISAI`, without CASA.

CASA tasks are recorders and the measurement-set reads return simulated
reduced visibilities (``tests/solint_sim.py``): the state read from
``selfcal_<k>.ms`` is the previous state with the chosen solution applied by
a numpy stand-in for gaincal + applycal. So the stage decisions are the real
ones of ``ajisai.solint`` and the bookkeeping around them (copies, gaincal
arguments, the G1 gate, stage records, best stage, the justification) can be
checked. ASEBI SPEC v0.3 section 10.2; DESIGN_REVIEW P18, P23.
"""
from __future__ import annotations

import pytest

from ajisai import (AJISAI, AJISAIConfig, FIXED_SCHEDULE_STEPS, GainCalConfig, ImagingConfig,
                    SelfcalSchedule, SelfcalStep, core, selfcal_measurement)
from ajisai import solint as so
from tests import solint_sim as sim
from tests.test_selfcal_bookkeeping import _fake_imaging, _pipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@pytest.fixture
def existing_vis(tmp_path):
    p = tmp_path / "fake.ms"
    p.mkdir()
    return str(p)


def test_default_schedule_is_data_driven():
    sch = SelfcalSchedule()
    assert sch.mode == "data" and sch.amplitude == "never"
    assert sch.steps == FIXED_SCHEDULE_STEPS          # used only by mode="fixed"


def test_fixed_mode_keeps_the_v02_ladder(existing_vis):
    sch = SelfcalSchedule(mode="fixed")
    assert [s.solint for s in sch.steps] == ["inf", "6*IT", "3*IT", "inf"]
    AJISAI(AJISAIConfig(vis=existing_vis, schedule=sch))._validate_inputs()
    custom = SelfcalSchedule(mode="fixed", steps=(SelfcalStep("p", "30s"),))
    AJISAI(AJISAIConfig(vis=existing_vis, schedule=custom))._validate_inputs()


@pytest.mark.parametrize("schedule, gaincal, match", [
    (SelfcalSchedule(steps=(SelfcalStep("p", "inf"),)), GainCalConfig(), "mode='fixed'"),
    (SelfcalSchedule(mode="adaptive"), GainCalConfig(), "schedule.mode"),
    (SelfcalSchedule(amplitude="conditional"), GainCalConfig(), "amplitude"),
    (SelfcalSchedule(), GainCalConfig(gaintype="G"), "gaintype"),
    (SelfcalSchedule(), GainCalConfig(combine="scan,field"), "combine"),
    (SelfcalSchedule(), GainCalConfig(applymode="calflag"), "calonly"),
    (SelfcalSchedule(), GainCalConfig(applycal_interp="nearestperobs"), "linear"),
])
def test_invalid_data_mode_options(existing_vis, schedule, gaincal, match):
    with pytest.raises(ValueError, match=match):
        AJISAI(AJISAIConfig(vis=existing_vis, schedule=schedule, gaincal=gaincal))._validate_inputs()


def test_fixed_mode_allows_other_gaintypes(existing_vis):
    AJISAI(AJISAIConfig(vis=existing_vis, schedule=SelfcalSchedule(mode="fixed"),
                        gaincal=GainCalConfig(gaintype="G")))._validate_inputs()


# ---------------------------------------------------------------------------
# The data-driven loop with fakes
# ---------------------------------------------------------------------------
def _data_run(tmp_path, monkeypatch, state0, gaincal=None, n_gated=2):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch, gaincal=gaincal)
    _fake_imaging(aj, events)
    geom = so.geometry(state0)
    reads = []
    cache = {}

    def read(path, fid, band_halves=False):
        reads.append(str(path))
        if str(path).endswith("_applycal.tmp.ms") or str(path) == vis:
            return state0
        if str(path) not in cache:
            last = aj._sc["chain"][-1]
            cache[str(path)] = sim.production(last.state, geom, last.cand, state0,
                                              aj._sc["combine_spw"])
        st = cache[str(path)]
        return so.VisSummary(time=st.time, obs=st.obs, scan=st.scan, spw=st.spw, a1=st.a1,
                             a2=st.a2, y=st.y, P=st.P, antenna_names=st.antenna_names, S=st.S,
                             uvdist_m=st.uvdist_m, y_half=st.y_half, P_half=st.P_half)

    monkeypatch.setattr(core, "read_vis_summary", read)
    gaincal_calls = []

    def run_gaincal(**kw):
        gaincal_calls.append(kw)
        return 40, True
    aj._run_gaincal = run_gaincal
    gates = []

    def gate(caltable, state, cand, s_rows):
        gates.append((caltable, cand["label"]))
        return {"table": caltable, "n_rows": 40, "n_flagged": n_gated, "unmatched": 0,
                "per_antenna": {"A00": n_gated}}
    aj._gate_caltable = gate
    # stage 0 (made by _make_initial_clean in a real run)
    aj.metrics.append({"iteration": 0, "status": "ok", "dynamic_range": 50.0,
                       "image_path": "clean_iter0.image", "fits_path": "clean_iter0.fits"})
    aj._add_stage({"stage": 0, "label": "no_selfcal", "status": "ok", "data_ms": vis,
                   "stage_ms": None, "solve": None})
    aj._run_data_driven()
    aj._select_best()
    return aj, tasks, events, vis, reads, gaincal_calls, gates


def test_bright_source_runs_eb_then_a_short_interval_then_stops(tmp_path, monkeypatch):
    state0, _ = sim.simulate(n_ant=16, n_scans=4, n_int=20, r=3.0, tau0=200.0, beta=1.0, seed=3)
    aj, tasks, events, vis, reads, calls, gates = _data_run(tmp_path, monkeypatch, state0)
    interm = aj.workdir / "intermediate"
    # stage 1: copy of the working MS, model predicted into it, read, one solution per EB
    assert ("copy", vis, str(interm / "selfcal_1_applycal.tmp.ms")) in events
    assert reads[0] == str(interm / "selfcal_1_applycal.tmp.ms")
    assert calls[0]["solint"] == "inf" and calls[0]["combine"] == "scan"
    # stage 2: read straight from selfcal_1.ms (it holds its model), a short interval
    assert reads[1] == str(interm / "selfcal_1.ms")
    assert calls[1]["combine"] == "" and calls[1]["solint"].endswith("s")
    assert float(calls[1]["solint"][:-1]) == pytest.approx(12.0 - 6.0 / 40)   # 2 integrations
    # stage 3 evaluated on selfcal_2.ms and not applied
    assert reads[2] == str(interm / "selfcal_2.ms") and len(calls) == 2
    # applycal / split only on the temporary copies; copies removed
    for kw in tasks.named("applycal"):
        assert kw["vis"].endswith("_applycal.tmp.ms")
    assert [kw["outputvis"] for kw in tasks.named("split")] == [
        str(interm / "selfcal_1.ms"), str(interm / "selfcal_2.ms")]
    assert ("rm", str(interm / "selfcal_2_applycal.tmp.ms")) in events
    # records
    assert [m["status"] for m in aj.metrics] == ["ok", "ok", "ok", "rejected"]
    assert aj.justification["best"]["iteration"] == 2
    assert "last accepted stage" in aj.justification["best"]["rule"]
    summary = aj.justification["selfcal"]
    assert summary["accepted_stages"] == [1, 2] and summary["best_stage"] == 2
    assert [s["decision"] for s in summary["stages"]] == ["apply", "apply", "stop"]
    st2 = aj._stage(2)
    assert st2["solve"]["solint_s"] == pytest.approx(12.0)
    assert st2["solve"]["calmode"] == "p" and st2["solve"]["combine"] == ""
    assert st2["label"] == "phase_12s"
    sel = st2["selection"]
    assert sel["decision"] == "apply" and sel["chosen"] == "12s"
    assert {"measured", "candidates", "rule_A", "rule_B", "reason", "g1_gate_applied"} <= set(sel)
    assert sel["g1_gate_applied"]["n_flagged"] == 2
    assert sel["same_data_diagnostics"]["dynamic_range"] == 1e3
    assert set(sel["candidates"][0]["g1"]["antenna_snr_median"]) == set(state0.antenna_names)
    assert "_" not in sel
    # G3 recorded for the accepted stages
    assert set(aj._stage(1)["selection"]["g3"]["per_observation"]) == {"0"}
    assert "g3" in st2["selection"]
    rej = aj._stage(3)
    assert rej["status"] == "rejected" and rej["selection"]["decision"] == "stop"
    assert gates == [(str(interm / "sc1.pcal"), "EB"), (str(interm / "sc2.pcal"), "12s")]
    # the compact summary for a calling pipeline
    cs = aj.selfcal_summary()
    assert cs["mode"] == "data" and cs["best_stage"] == 2
    assert [s["status"] for s in cs["stages"]] == ["ok", "ok", "ok", "rejected"]


def test_too_faint_source_gets_no_self_calibration(tmp_path, monkeypatch):
    state0, _ = sim.simulate(n_ant=10, n_scans=2, n_int=10, r=0.15, tau0=None, seed=8)
    aj, tasks, events, vis, reads, calls, gates = _data_run(tmp_path, monkeypatch, state0)
    assert calls == [] and tasks.named("applycal") == []
    assert [m["status"] for m in aj.metrics] == ["ok", "rejected"]
    assert aj.justification["best"]["iteration"] == 0
    assert aj.justification["selfcal"]["accepted_stages"] == []
    assert "G1" in aj.justification["selfcal"]["stop_reason"]
    assert aj.justification["selfcal"]["amplitude_stage"]["attempted"] is False


def test_combined_spws_get_a_derived_spwmap(tmp_path, monkeypatch):
    state0, _ = sim.simulate(n_ant=12, n_scans=3, n_int=12, n_spw=2, r=3.0, tau0=300.0, seed=4)
    aj, tasks, events, vis, reads, calls, gates = _data_run(
        tmp_path, monkeypatch, state0, gaincal=GainCalConfig(combine="scan,spw"))
    assert calls[0]["combine"] == "scan,spw"
    assert tasks.named("applycal")[0]["spwmap"] == [[0, 0]]
    assert aj._stage(1)["solve"]["applycal"]["spwmap"] == [0, 0]
    assert aj.justification["selfcal"]["spwmap"] == [0, 0]


def test_all_solutions_gated_stops_the_loop(tmp_path, monkeypatch):
    state0, _ = sim.simulate(n_ant=16, n_scans=4, n_int=20, r=3.0, tau0=200.0, seed=3)
    aj, tasks, events, vis, reads, calls, gates = _data_run(tmp_path, monkeypatch, state0,
                                                            n_gated=40)
    assert len(calls) == 1 and tasks.named("applycal") == []
    assert [m["status"] for m in aj.metrics] == ["ok", "rejected"]
    assert "no usable solution" in aj.justification["selfcal"]["stop_reason"]


# ---------------------------------------------------------------------------
# Entry point for a calling pipeline's measurement self-cal
# ---------------------------------------------------------------------------
def test_selfcal_measurement_configuration(existing_vis, tmp_path, monkeypatch):
    seen = {}

    def fake_run(self):
        seen["cfg"] = self.cfg
        return self
    monkeypatch.setattr(AJISAI, "run", fake_run)
    aj = selfcal_measurement(existing_vis, results_dir=str(tmp_path / "m"), refant="DA41",
                             imaging=ImagingConfig(cell="6.2mas", imsize=(512,), final_depth=()),
                             gaincal=GainCalConfig(combine="scan,spw", spwmap=(0, 0, 0, 0)))
    cfg = seen["cfg"]
    assert isinstance(aj, AJISAI)
    assert cfg.schedule.mode == "data" and cfg.schedule.amplitude == "never"
    assert cfg.caller == "asebi" and cfg.relabel_j2000_to_icrs is False
    assert cfg.refant_strategy == "manual" and cfg.refant_manual == "DA41"
    assert cfg.gaincal.spwmap == (0, 0, 0, 0) and cfg.imaging.cell == "6.2mas"
    default = selfcal_measurement(existing_vis, results_dir=str(tmp_path / "n"))
    assert default.cfg.imaging.final_depth == ()
    AJISAI(cfg)._validate_inputs()


def test_gain_plot_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    """plotms needs a DISPLAY; on a headless server it raises: warn and go on."""
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch)

    class NoDisplay:
        @staticmethod
        def plotms(**kw):
            raise RuntimeError("ERROR: DISPLAY environment variable is not set!")
    monkeypatch.setattr(core, "casaplotms", NoDisplay)
    with pytest.warns(UserWarning, match="plotms failed"):
        aj._plot_gain_table("sc1.pcal", 1, SelfcalStep("p", "inf", label="phase_EB"))
    aj._plot_gain_table("sc2.pcal", 2, SelfcalStep("p", "inf", label="phase_EB"))  # warned once
