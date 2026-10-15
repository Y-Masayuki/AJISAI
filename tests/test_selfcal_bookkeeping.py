"""
Tests of the v0.3 self-calibration bookkeeping and data hygiene, without CASA.

``core.casatasks`` and the CASA-touching helpers are replaced by recorders, so
the order and the arguments of the CASA calls can be checked:

* the caller's MS is never the target of applycal/split/ft/clearcal; every
  stage solves and applies on a temporary copy of the previous stage's MS,
  splits CORRECTED into ``selfcal_<N>.ms`` and deletes the copy;
* every tclean of a run gets the same cell/imsize/nterms/reffreq/scales and,
  with ``mask_mode="user"``, the user's CRTF file;
* the per-stage solve information has a fixed structure;
* the final image steps through the depth schedule and stops by the
  2026-10-15 rule; the final MS is a DATA-only split of the selected stage.
"""
from __future__ import annotations

import numpy as np
import pytest

from ajisai import AJISAI, AJISAIConfig, GainCalConfig, ImagingConfig, SelfcalStep
from ajisai import core
from ajisai.core import (
    SOLVE_INFO_KEYS,
    build_solve_info,
    flagged_solution_fractions,
    solint_seconds,
)


# ---------------------------------------------------------------------------
# solint in seconds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solint, expected", [
    ("inf", None),
    ("int", 6.05),
    ("36.300s", 36.3),
    ("60s", 60.0),
    ("1.5min", 90.0),
    ("0.5h", 1800.0),
    ("30", 30.0),
])
def test_solint_seconds(solint, expected):
    assert solint_seconds(solint, int_time_s=6.05) == (
        pytest.approx(expected) if expected is not None else None)


def test_solint_seconds_rejects_garbage():
    with pytest.raises(ValueError):
        solint_seconds("scan")


# ---------------------------------------------------------------------------
# Flagged-solution fraction per observation / spw
# ---------------------------------------------------------------------------
def test_flagged_solution_fractions_per_obs_and_spw():
    obs = [0, 0, 0, 0, 1, 1]
    spw = [0, 0, 1, 1, 0, 0]
    flags = np.zeros((1, 1, 6), bool)
    flags[0, 0, 1] = True          # obs 0 / spw 0: 1 of 2 flagged
    flags[0, 0, 4:] = True         # obs 1 / spw 0: 2 of 2 flagged
    out = flagged_solution_fractions(obs, spw, flags)
    assert [(r["observation"], r["spw"]) for r in out] == [(0, 0), (0, 1), (1, 0)]
    assert [r["fraction"] for r in out] == [0.5, 0.0, 1.0]
    assert [r["n_solutions"] for r in out] == [2, 2, 2]


def test_flagged_solution_needs_all_entries_flagged():
    """A 'G' solution with one polarization flagged is not a flagged solution."""
    flags = np.array([[[True, True]], [[False, True]]])   # (npol=2, nchan=1, nrow=2)
    out = flagged_solution_fractions([0, 0], [3, 3], flags)
    assert out == [{"observation": 0, "spw": 3, "n_solutions": 2, "n_flagged": 1,
                    "fraction": 0.5}]


# ---------------------------------------------------------------------------
# Solve-info structure
# ---------------------------------------------------------------------------
def _solve(**over):
    kw = dict(
        table="sc1.pcal", solint="36.300s", solint_s=36.3, combine="scan", refant="DA42",
        calmode="p", gaintype="T", minsnr=3.0, minblperant=4, solnorm=False,
        interp="linearPDperobs", applymode="calonly", calwt=False, spwmap=(),
        model_image=["c.model.tt0", "c.model.tt1"], model_nterms=2,
        model_reffreq_hz=230e9, solved_on="in.ms", stage_ms="selfcal_1.ms",
        n_solutions=40, flagged_solution_fraction=[],
    )
    kw.update(over)
    return build_solve_info(**kw)


def test_solve_info_has_the_stable_keys_in_order():
    info = _solve()
    assert tuple(info) == SOLVE_INFO_KEYS
    assert info["applycal"] == {"interp": "linearPDperobs", "applymode": "calonly",
                                "calwt": False, "spwmap": []}
    assert info["model_image"] == ["c.model.tt0", "c.model.tt1"]


def test_solve_info_spwmap_and_single_model():
    info = _solve(spwmap=(0, 0, 0, 0), model_image="c.model", model_nterms=1)
    assert info["applycal"]["spwmap"] == [0, 0, 0, 0]
    assert info["model_image"] == ["c.model"]


def test_solve_info_is_json_serialisable():
    import json
    json.dumps(_solve(flagged_solution_fraction=flagged_solution_fractions(
        [0], [0], np.zeros((1, 1, 1), bool))))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class Recorder:
    """Stand-in for casatasks: records every call as (name, kwargs)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def task(**kw):
            self.calls.append((name, kw))
        return task

    def named(self, name):
        return [kw for n, kw in self.calls if n == name]


def _pipeline(tmp_path, monkeypatch, imaging=None, gaincal=None, caller="user"):
    vis = tmp_path / "input.ms"
    vis.mkdir()
    cfg = AJISAIConfig(vis=str(vis), results_dir=str(tmp_path / "out"),
                       imaging=imaging or ImagingConfig(), gaincal=gaincal or GainCalConfig(),
                       caller=caller, verbose=False)
    aj = AJISAI(cfg)
    aj.workdir = tmp_path / "out"
    (aj.workdir / "intermediate").mkdir(parents=True)
    aj._derived.update({
        "resolved_field_id": "0", "refant": "DA42", "avg_int_time_sec": 6.05,
        "imsize": [256, 256], "cell": "0.05arcsec", "cell_arcsec": 0.05,
        "scales": [0, 5, 15], "sigma_th": 1e-5, "dirty_rms": 2e-5,
        "working_vis": str(vis), "good_vis": str(vis), "good_vis_model_from": None,
        "last_image_base": str(aj.workdir / "intermediate" / "clean_iter0"),
        "last_nterms": int(cfg.imaging.nterms), "last_dr": 50.0, "last_rms": 2e-5,
        "initial_mask_path": str(aj.workdir / "intermediate" / "clean_iter0.mask"),
    })
    tasks = Recorder()
    monkeypatch.setattr(core, "casatasks", tasks)
    events = []
    monkeypatch.setattr(core, "_copy_ms", lambda src, dst: events.append(("copy", str(src), str(dst))))
    monkeypatch.setattr(core, "_safe_rmtree_path", lambda p: events.append(("rm", str(p))))
    monkeypatch.setattr(core, "_safe_rmtree_glob", lambda parent, pat: None)
    return aj, tasks, events, str(vis)


def _fake_imaging(aj, events, n_solutions=12):
    """Replace the CASA-reading methods of one AJISAI instance."""
    def tclean(vis, imagename, **kw):
        events.append(("tclean", str(vis), str(imagename), kw))
        return int(aj.cfg.imaging.nterms)

    def predict(vis, base, nterms):
        events.append(("predict", str(vis), str(base)))
        return {"model_image": core._model_images(base, nterms), "nterms": nterms,
                "reffreq_hz": 230e9 if nterms > 1 else None, "method": "ft"}

    stats = {"peak_jy_beam": 1e-2, "rms_jy_beam": 1e-5, "snr": 1e3, "dynamic_range": 1e3,
             "bmaj_arcsec": 0.2, "bmin_arcsec": 0.15, "rms_info": {}}
    aj._tclean = tclean
    aj._predict_model = predict
    aj._export_and_stats = lambda base, nterms, fits: dict(stats)
    aj._run_gaincal = lambda **kw: (n_solutions, n_solutions > 0)
    aj._plot_gain_table = lambda *a, **k: None
    aj._caltable_solution_stats = lambda table: [
        {"observation": 0, "spw": 0, "n_solutions": 12, "n_flagged": 0, "fraction": 0.0}]


# ---------------------------------------------------------------------------
# Data hygiene of one self-cal iteration
# ---------------------------------------------------------------------------
def test_iteration_never_writes_the_callers_ms(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch)
    _fake_imaging(aj, events)
    aj._run_iteration(1, SelfcalStep("p", "inf", label="p_inf"))

    interm = aj.workdir / "intermediate"
    tmp = str(interm / "selfcal_1_applycal.tmp.ms")
    out = str(interm / "selfcal_1.ms")
    # 1. the caller's MS is only the source of the temporary copy
    assert ("copy", vis, tmp) in events
    # 2. the stage-0 model is predicted into the copy (the input has none)
    assert ("predict", tmp, str(interm / "clean_iter0")) in events
    # 3. applycal on the copy, never on the input; no clearcal anywhere
    applycal = tasks.named("applycal")
    assert len(applycal) == 1 and applycal[0]["vis"] == tmp
    assert applycal[0]["calwt"] is False
    assert applycal[0]["interp"] == "linearPDperobs"
    assert applycal[0]["spwmap"] == []
    assert tasks.named("clearcal") == []
    # 4. split CORRECTED of the copy into the stage MS; the copy is deleted after
    split = tasks.named("split")
    assert split == [{"vis": tmp, "outputvis": out, "datacolumn": "corrected"}]
    assert ("rm", tmp) in events
    # 5. the stage MS is imaged, then its own model is stored in it
    assert [e[1] for e in events if e[0] == "tclean"] == [out]
    assert ("predict", out, str(interm / "clean_sc1")) in events
    # nothing at all was called on the input MS
    for name, kw in tasks.calls:
        assert kw.get("vis") != vis, name
    assert not [e for e in events if e[0] in ("tclean", "predict") and e[1] == vis]


def test_second_iteration_reuses_the_model_in_the_stage_ms(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch)
    _fake_imaging(aj, events)
    aj._run_iteration(1, SelfcalStep("p", "inf"))
    events.clear()
    aj._run_iteration(2, SelfcalStep("p", "6*IT"))
    interm = aj.workdir / "intermediate"
    tmp2 = str(interm / "selfcal_2_applycal.tmp.ms")
    assert ("copy", str(interm / "selfcal_1.ms"), tmp2) in events
    # selfcal_1.ms already holds the model of clean_sc1: no prediction into the copy
    assert ("predict", tmp2, str(interm / "clean_sc1")) not in events
    assert [e for e in events if e[0] == "predict"] == [
        ("predict", str(interm / "selfcal_2.ms"), str(interm / "clean_sc2"))]
    solve = aj.stages[-1]["solve"]
    assert solve["solved_on"] == str(interm / "selfcal_1.ms")
    assert solve["model_image"] == [str(interm / "clean_sc1") + ".model"]
    assert solve["solint_s"] == pytest.approx(36.3)


def test_iteration_records_the_solve_information(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(
        tmp_path, monkeypatch, imaging=ImagingConfig(nterms=2),
        gaincal=GainCalConfig(spwmap=(0, 0, 0, 0)))
    _fake_imaging(aj, events)
    aj._run_iteration(1, SelfcalStep("p", "inf", label="p_inf"))
    stage = aj.stages[-1]
    assert stage["stage"] == 1 and stage["status"] == "ok"
    solve = stage["solve"]
    assert tuple(solve) == SOLVE_INFO_KEYS
    interm = aj.workdir / "intermediate"
    assert solve["table"] == str(interm / "sc1.pcal")
    assert solve["solint"] == "inf" and solve["solint_s"] is None
    assert solve["combine"] == "scan" and solve["refant"] == "DA42"
    assert (solve["calmode"], solve["gaintype"], solve["minsnr"]) == ("p", "T", 3.0)
    assert solve["applycal"]["spwmap"] == [0, 0, 0, 0]
    assert "perobs" in solve["applycal"]["interp"]
    assert solve["stage_ms"] == str(interm / "selfcal_1.ms")
    assert solve["model_image"] == [str(interm / "clean_iter0") + ".model.tt0",
                                    str(interm / "clean_iter0") + ".model.tt1"]
    assert solve["flagged_solution_fraction"][0]["fraction"] == 0.0
    # applycal received the configured spwmap
    assert tasks.named("applycal")[0]["spwmap"] == [[0, 0, 0, 0]]
    # metrics row stays flat and points at the tt0 image
    rec = aj.metrics[-1]
    assert rec["image_path"].endswith("clean_sc1.image.tt0")
    assert not any(isinstance(v, dict) for v in rec.values())
    assert aj.justification["stages"][-1] is stage


def test_skipped_iteration_leaves_no_copy_and_keeps_the_chain(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch)
    _fake_imaging(aj, events, n_solutions=0)
    with pytest.warns(UserWarning, match="SKIPPED"):
        aj._run_iteration(1, SelfcalStep("p", "inf"))
    tmp = str(aj.workdir / "intermediate" / "selfcal_1_applycal.tmp.ms")
    assert ("rm", tmp) in events
    assert tasks.named("applycal") == [] and tasks.named("split") == []
    assert aj._derived["good_vis"] == vis
    assert aj.stages[-1]["status"] == "skipped"


# ---------------------------------------------------------------------------
# tclean arguments: the same imaging rules for every call
# ---------------------------------------------------------------------------
def test_tclean_passes_geometry_nterms_reffreq_and_never_saves_a_model(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(
        tmp_path, monkeypatch,
        imaging=ImagingConfig(nterms=2, reffreq="227.0GHz", mask_mode="user",
                              user_mask="/data/region.crtf"))
    aj._check_imaging_consistency = lambda base, nterms: events.append(("check", str(base), nterms))
    mask = aj._build_usemask_kwargs()
    nterms = aj._tclean("a.ms", "img", niter=100, threshold_jy=3e-5, mask_kwargs=mask)
    kw = tasks.named("tclean")[0]
    assert nterms == 2
    assert kw["deconvolver"] == "mtmfs" and kw["nterms"] == 2 and kw["scales"] == [0, 5, 15]
    assert kw["reffreq"] == "227.0GHz"
    assert kw["cell"] == "0.05arcsec" and kw["imsize"] == [256, 256]
    assert kw["savemodel"] == "none" and kw["interactive"] is False
    assert kw["threshold"] == "3e-05Jy"
    assert kw["mask"] == "/data/region.crtf" and kw["usemask"] == "user"
    assert ("check", "img", 2) in events


def test_dirty_image_is_hogbom(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch, imaging=ImagingConfig(nterms=2))
    aj._check_imaging_consistency = lambda base, nterms: events.append(("check", nterms))
    assert aj._tclean("a.ms", "dirty", niter=0, threshold_jy=0.0, dirty=True) == 1
    kw = tasks.named("tclean")[0]
    assert kw["deconvolver"] == "hogbom" and "nterms" not in kw
    assert ("check", 1) in events


def test_user_crtf_mask_for_every_tclean(tmp_path, monkeypatch):
    """mask_mode='user' passes the CRTF itself, also when an earlier .mask exists."""
    aj, tasks, events, vis = _pipeline(
        tmp_path, monkeypatch,
        imaging=ImagingConfig(mask_mode="user", user_mask="/data/region.crtf"))
    existing = tmp_path / "clean_iter0.mask"
    existing.mkdir()
    for kw in (aj._build_usemask_kwargs(), aj._build_usemask_kwargs(existing_mask=str(existing))):
        assert kw == {"mask": "/data/region.crtf", "usemask": "user", "interactive": False}


def test_auto_mask_reuses_the_initial_mask(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch)
    existing = tmp_path / "clean_iter0.mask"
    existing.mkdir()
    assert aj._build_usemask_kwargs()["usemask"] == "auto-multithresh"
    assert aj._build_usemask_kwargs(existing_mask=str(existing))["mask"] == str(existing)


def test_interactive_refused_for_asebi(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(
        tmp_path, monkeypatch, imaging=ImagingConfig(mask_mode="interactive"), caller="asebi")
    with pytest.raises(ValueError, match="asebi"):
        aj._build_usemask_kwargs()
    with pytest.raises(ValueError, match="asebi"):
        aj._tclean("a.ms", "img", niter=1, threshold_jy=0.0, mask_kwargs={"interactive": True})
    assert tasks.named("tclean") == []


# ---------------------------------------------------------------------------
# Final MS and final image
# ---------------------------------------------------------------------------
def _with_stages(aj, best_stage):
    interm = aj.workdir / "intermediate"
    for i in range(best_stage + 1):
        aj.metrics.append({"iteration": i, "dynamic_range": 10.0 + i})
        aj._add_stage({"stage": i, "data_ms": str(interm / f"selfcal_{i}.ms") if i else "input.ms",
                       "stage_ms": str(interm / f"selfcal_{i}.ms") if i else None})
    aj.justification["best"] = {"iteration": best_stage}


def test_final_ms_is_a_data_only_split_of_the_selected_stage(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch)
    _with_stages(aj, 2)
    aj._write_final_ms()
    assert tasks.named("split") == [{
        "vis": str(aj.workdir / "intermediate" / "selfcal_2.ms"),
        "outputvis": str(aj.workdir / "final_selfcal.ms"),
        "datacolumn": "data"}]
    assert aj.final_ms == str(aj.workdir / "final_selfcal.ms")
    assert aj.justification["final_ms"]["stage"] == 2


def test_final_image_steps_through_depths_and_stops(tmp_path, monkeypatch):
    """V1094-like sequence: continue after 3 sigma, stop at 2 sigma; no new PSF."""
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch)
    _fake_imaging(aj, events)
    _with_stages(aj, 1)
    aj.final_ms = str(aj.workdir / "final_selfcal.ms")
    aj._derived["sigma_th"] = 40.9e-6
    seq = iter([(215.4e-3, 9.9e-3), (219.1e-3, 6.6e-3), (221.9e-3, 4.0e-3)])

    def terms(base, nterms):
        f_model, f_unc = next(seq)
        return {"f_model_jy": f_model, "f_unc_jy": f_unc, "f_corr_jy": f_model + f_unc,
                "sigma_f_jy": 1.54e-3, "restored_bias_jy": f_unc * 1.62,
                "omega_r_pix": 458.6, "omega_clean_pix": 175.0, "aperture_npix": 650000,
                "aperture_radius_equiv_pix": 455.0}
    aj._final_flux_terms = terms
    with pytest.warns(UserWarning, match="over-counts"):
        aj._make_final_image()
    calls = [e for e in events if e[0] == "tclean"]
    assert len(calls) == 2
    assert calls[0][1] == aj.final_ms
    assert calls[0][3]["threshold_jy"] == pytest.approx(3 * 40.9e-6)
    assert calls[1][3]["threshold_jy"] == pytest.approx(2 * 40.9e-6)
    assert "calcpsf" not in calls[0][3]
    assert calls[1][3]["calcpsf"] is False and calls[1][3]["calcres"] is False
    fi = aj.justification["final_image"]
    assert fi["made"] and fi["converged"] and fi["stop_nsigma"] == 2.0
    assert [s["stop"] for s in fi["steps"]] == [False, True]
    assert aj.final_image.endswith("final_clean.image")


def test_final_image_can_be_disabled(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch,
                                       imaging=ImagingConfig(final_depth=()))
    aj.final_ms = "final_selfcal.ms"
    aj._make_final_image()
    assert aj.justification["final_image"]["made"] is False
    assert aj.final_image is None


def test_final_image_warns_when_not_converged(tmp_path, monkeypatch):
    aj, tasks, events, vis = _pipeline(tmp_path, monkeypatch,
                                       imaging=ImagingConfig(final_depth=(3.0, 2.0)))
    _fake_imaging(aj, events)
    _with_stages(aj, 0)
    aj.final_ms = "final_selfcal.ms"
    seq = iter([(0.100, 0.020), (0.150, 0.015)])

    def terms(base, nterms):
        f_model, f_unc = next(seq)
        return {"f_model_jy": f_model, "f_unc_jy": f_unc, "f_corr_jy": f_model + f_unc,
                "sigma_f_jy": 1e-4, "restored_bias_jy": 0.0}
    aj._final_flux_terms = terms
    with pytest.warns(UserWarning, match="did not converge"):
        aj._make_final_image()
    assert aj.justification["final_image"]["converged"] is False

