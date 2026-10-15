"""
End-to-end run of AJISAI on a tiny simulated MS (CASA required).

Marked ``casa``: skipped automatically when casatools/casatasks are absent.
Run with ``python -m pytest -m casa`` in a CASA Python (e.g. CASA 6.5.1).
Checks the v0.3 data hygiene and products (with a one-step ``mode="fixed"``
schedule; the data-driven stages are tested in ``test_casa_solint.py``):

* the input MS is byte-for-byte unchanged after the run;
* each stage MS holds DATA + MODEL only (no CORRECTED), the final
  ``final_selfcal.ms`` DATA only, and no temporary copy is left behind;
* mtmfs products are named ``.image.tt0`` and statistics are taken on them;
* every tclean and model prediction passed its consistency check;
* the per-stage solve information has the documented keys;
* the final image went through the depth rule.
"""
from __future__ import annotations

import hashlib
import os

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


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from tests.casa_sim import simulate_ms
    from ajisai import (AJISAI, AJISAIConfig, ImagingConfig, SelfcalSchedule,
                        SelfcalStep)

    work = tmp_path_factory.mktemp("ajisai_casa")
    ms = simulate_ms(str(work / "input.ms"), nchan=(2, 4), chan_mhz=(500.0, 250.0),
                     scans=2, phase_rms_deg=20.0)
    crtf = work / "region.crtf"
    crtf.write_text("#CRTFv0\ncircle[[16h00m00.0s, -30d00m00.0s], 4arcsec]\n")
    before = _digest(ms)
    cfg = AJISAIConfig(
        vis=ms,
        results_dir=str(work / "out"),
        imaging=ImagingConfig(nterms=2, mask_mode="user", user_mask=str(crtf), niter=2000),
        schedule=SelfcalSchedule(mode="fixed", steps=(SelfcalStep("p", "inf", label="p_inf"),)),
        # Read the input directly (no relabelled copy), so that the dirty,
        # noise and initial images and the first copy all touch the input MS.
        relabel_j2000_to_icrs=False,
        show_banner=False,
        verbose=False,
    )
    aj = AJISAI(cfg).run()
    return {"aj": aj, "ms": ms, "before": before}


def _data_columns(ms: str):
    import casatools
    tb = casatools.table()
    tb.open(ms)
    try:
        return sorted(c for c in tb.colnames() if c.endswith("DATA"))
    finally:
        tb.close()


def test_input_ms_is_not_modified(run):
    aj = run["aj"]
    assert aj.justification["derived"]["working_vis"]["working_vis"] == run["ms"]
    assert aj.stages[0]["data_ms"] == run["ms"]
    assert _digest(run["ms"]) == run["before"]


def test_relabelling_works_on_a_copy(run, tmp_path):
    """J2000 codes in the input: the relabelled copy is used, the input is untouched."""
    import casatools
    from ajisai import AJISAI, AJISAIConfig

    ms = run["ms"]
    aj = AJISAI(AJISAIConfig(vis=ms, results_dir=str(tmp_path), verbose=False))
    aj.workdir = tmp_path
    (tmp_path / "intermediate").mkdir()
    aj._prepare_working_vis()
    copy = aj._derived["working_vis"]
    assert copy == str(tmp_path / "intermediate" / "input_icrs.ms")
    tb = casatools.table()
    tb.open(copy + "/SOURCE")
    try:
        assert tb.getcolkeywords("DIRECTION")["MEASINFO"]["Ref"] == "ICRS"
    finally:
        tb.close()
    assert _digest(ms) == run["before"]


def test_stage_and_final_ms_columns(run):
    aj = run["aj"]
    stage = aj.stages[1]
    assert stage["status"] in ("ok", "anomaly")
    assert _data_columns(stage["stage_ms"]) == ["DATA", "MODEL_DATA"]
    assert _data_columns(aj.final_ms) == ["DATA"]
    leftovers = [p for p in os.listdir(os.path.dirname(stage["stage_ms"])) if ".tmp.ms" in p]
    assert leftovers == []


def test_mtmfs_names_and_checks(run):
    aj = run["aj"]
    assert all(st["image"].endswith(".image.tt0") for st in aj.stages)
    assert all(os.path.isdir(st["image"]) for st in aj.stages)
    checks = aj.justification["checks"]
    assert {c["kind"] for c in checks} == {"tclean", "clean_model", "model_prediction"}
    assert all(c["ok"] for c in checks)
    # the initial CLEAN (mtmfs) left a model: checked once, on its Taylor terms
    model_check = [c for c in checks if c["kind"] == "clean_model"]
    assert len(model_check) == 1 and model_check[0]["n_model_pixels"] > 0


def test_solve_information(run):
    from ajisai import SOLVE_INFO_KEYS
    solve = run["aj"].stages[1]["solve"]
    assert tuple(solve) == SOLVE_INFO_KEYS
    assert solve["applycal"]["interp"] == "linearPDperobs"
    assert solve["applycal"]["calwt"] is False
    assert solve["minsnr"] == 3.0
    assert solve["model_nterms"] == 2 and solve["model_reffreq_hz"] > 0
    assert solve["flagged_solution_fraction"]


def test_final_image_and_geometry(run):
    aj = run["aj"]
    fi = aj.justification["final_image"]
    assert fi["made"] and fi["steps"]
    assert os.path.isdir(aj.final_image)
    derived = aj.justification["derived"]
    assert derived["cellsize"]["rule"] == "umax"
    assert derived["dish_diameter"]["source"] == "ANTENNA/DISH_DIAMETER"
    assert derived["sigma_th"]["stokes"] == "Q"
    assert derived["scales"]["value"][0] == 0
