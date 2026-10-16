"""
Sky frames: coordinate conversions and the phase-shift path (CASA required).

Marked ``casa``: skipped automatically when casatools/casatasks are absent.
Run with ``python -m pytest -m casa`` in a CASA Python (e.g. CASA 6.5.1).

The simulated MSs have the FIELD layout of a real ALMA MS (variable-reference
PhaseDir_Ref/DelayDir_Ref/RefDir_Ref columns; ``tests/casa_sim.py``). Two of
them hold the same physical directions: one with genuine J2000 numbers (as
simulated), one converted to ICRS. Every check converts the stored directions
into ICRS with casatools.measures, reading each cell's frame from the table
itself, and compares them with the target:

* conversions (``convert_direction``, ``icrs_to_j2000``, ``rad_to_radec``)
  agree with casatools.measures to < 0.01 mas, and a ':'-separated Dec is
  read as degrees;
* the phase shift (ICRS / genuine-J2000 MS x target in ICRS / J2000 x
  relabel True / False) leaves FIELD PHASE_DIR, DELAY_DIR, REFERENCE_DIR and
  SOURCE.DIRECTION within 0.01 mas of the target, the frame codes as they
  were, and UVW and DATA identical to those of ``phaseshift`` alone;
* the field renumbered by ``phaseshift`` is found by name;
* the auto-detected target (dirty-image peak) is the pixel centre;
* the input relabel (label only) reads the codes from MEASINFO, for ALMA's
  layout, fixed-reference columns and a code table other than 0/21.

The measured residuals are printed (run with ``-s`` to see them).
"""
from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.casa

MAS = math.pi / 180.0 / 3600.0 / 1000.0
TOL_MAS = 0.01
# A target 0.21" east and 0.23" south of the simulated phase centre
# (16h00m00s -30d00m00s, J2000), as strings with a ':'-separated Dec.
TARGET_STR = ("16:00:00.0161234", "-30:00:00.2345678")
TARGET_CASA = ("16h00m00.0161234s", "-30d00m00.2345678s")
# A real ALMA MS to compare the simulated FIELD layout with (read only); optional.
REAL_ALMA_MS = os.environ.get("AJISAI_TEST_ALMA_MS", "")


def _tools():
    import casatools
    return casatools.measures(), casatools.quanta(), casatools.table()


def _to_icrs(ra, dec, frame):
    me, qa, _ = _tools()
    if frame == "ICRS":
        return float(ra) % (2 * math.pi), float(dec)
    m = me.measure(me.direction(frame, qa.quantity(float(ra), "rad"),
                                qa.quantity(float(dec), "rad")), "ICRS")
    return float(m["m0"]["value"]) % (2 * math.pi), float(m["m1"]["value"])


def _offset_mas(a1, d1, a2, d2):
    da = ((a1 - a2 + math.pi) % (2 * math.pi) - math.pi) * math.cos(d2)
    return math.hypot(da, d1 - d2) / MAS


def _frames(tb, col):
    """Per-row frames of a direction column, read independently of ajisai."""
    mi = tb.getcolkeyword(col, "MEASINFO")
    if "VarRefCol" in mi:
        types = [str(t) for t in mi["TabRefTypes"]]
        codes = [int(c) for c in mi["TabRefCodes"]]
        return [types[codes.index(int(c))] for c in tb.getcol(mi["VarRefCol"])]
    return [str(mi.get("Ref", "J2000"))] * tb.nrows()


def _stored_icrs(ms, field_id):
    """FIELD (three columns) and SOURCE directions of a field, converted to ICRS."""
    _, _, tb = _tools()
    out = {}
    tb.open(ms + "/FIELD")
    try:
        sid = int(tb.getcell("SOURCE_ID", field_id))
        for col in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
            fr = _frames(tb, col)[field_id]
            cell = tb.getcell(col, field_id)
            out[col] = _to_icrs(cell[0, 0], cell[1, 0], fr)
    finally:
        tb.close()
    tb.open(ms + "/SOURCE")
    try:
        frames = _frames(tb, "DIRECTION")
        for r, s in enumerate(tb.getcol("SOURCE_ID")):
            if int(s) == sid:
                cell = tb.getcell("DIRECTION", r)
                out[f"SOURCE[{r}]"] = _to_icrs(cell[0], cell[1], frames[r])
    finally:
        tb.close()
    return out


def _target_icrs(frame):
    """The test target (the strings read by CASA in their unambiguous h/d form) in ICRS."""
    me, _, _ = _tools()
    d = me.direction(frame, *TARGET_CASA)
    return _to_icrs(d["m0"]["value"], d["m1"]["value"], frame)


@pytest.fixture(scope="module")
def sims(tmp_path_factory):
    from tests.casa_sim import set_direction_frame, simulate_ms

    work = tmp_path_factory.mktemp("frames")
    j2000 = simulate_ms(str(work / "sim_j2000.ms"), nchan=(2, 2), chan_mhz=(500.0, 500.0),
                        scans=1, scan_s=60.0, phase_rms_deg=0.0)
    icrs = str(work / "sim_icrs.ms")
    shutil.copytree(j2000, icrs)
    set_direction_frame(icrs, "ICRS")
    return {"work": work, "J2000": j2000, "ICRS": icrs}


# ---------------------------------------------------------------------------
# The simulated MS has ALMA's FIELD layout, and CASA reads it per row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frame", ["J2000", "ICRS"])
def test_simulated_ms_has_the_alma_field_layout(sims, frame):
    import casatools
    from tests.casa_sim import ALMA_TAB_REF_CODES, ALMA_TAB_REF_TYPES

    me, qa, tb = _tools()
    ms = sims[frame]
    tb.open(ms + "/FIELD")
    try:
        for col, ref in (("PHASE_DIR", "PhaseDir_Ref"), ("DELAY_DIR", "DelayDir_Ref"),
                         ("REFERENCE_DIR", "RefDir_Ref")):
            mi = tb.getcolkeyword(col, "MEASINFO")
            assert mi["VarRefCol"] == ref and mi["type"] == "direction"
            assert [str(t) for t in mi["TabRefTypes"]] == list(ALMA_TAB_REF_TYPES)
            assert [int(c) for c in mi["TabRefCodes"]] == list(ALMA_TAB_REF_CODES)
            assert tb.getcoldesc(ref)["valueType"] == "int"
            assert _frames(tb, col) == [frame]
    finally:
        tb.close()
    msmd = casatools.msmetadata()
    msmd.open(ms)
    try:
        pc = msmd.phasecenter(0)
    finally:
        msmd.close()
    assert pc["refer"] == frame
    # The two simulated MSs hold the same physical direction.
    ref = me.direction("J2000", "16h00m00.0s", "-30d00m00.0s")
    a, d = _to_icrs(pc["m0"]["value"], pc["m1"]["value"], pc["refer"])
    a0, d0 = _to_icrs(ref["m0"]["value"], ref["m1"]["value"], "J2000")
    assert _offset_mas(a, d, a0, d0) < TOL_MAS


@pytest.mark.skipif(not os.path.isdir(REAL_ALMA_MS),
                    reason="set AJISAI_TEST_ALMA_MS to a real ALMA MS to compare the layout")
def test_simulated_layout_matches_a_real_alma_ms(sims):
    _, _, tb = _tools()
    real, sim = {}, {}
    for path, store in ((REAL_ALMA_MS, real), (sims["ICRS"], sim)):
        tb.open(path + "/FIELD")
        try:
            for col in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
                mi = tb.getcolkeyword(col, "MEASINFO")
                store[col] = (mi["VarRefCol"], mi["type"], [str(t) for t in mi["TabRefTypes"]],
                              [int(c) for c in mi["TabRefCodes"]])
            for col in ("PhaseDir_Ref", "DelayDir_Ref", "RefDir_Ref"):
                d = tb.getcoldesc(col)
                store[col] = (d["valueType"], d["dataManagerType"], d["option"])
        finally:
            tb.close()
    assert sim == real


# ---------------------------------------------------------------------------
# Conversions against casatools.measures
# ---------------------------------------------------------------------------
def test_convert_direction_matches_measures():
    from ajisai.ms_utils import convert_direction

    me, qa, _ = _tools()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        ra, dec = rng.uniform(0, 2 * math.pi), math.asin(rng.uniform(-0.99, 0.99))
        for f_in, f_out in (("ICRS", "J2000"), ("J2000", "ICRS")):
            a, d = convert_direction(ra, dec, f_in, f_out)
            m = me.measure(me.direction(f_in, f"{ra!r}rad", f"{dec!r}rad"), f_out)
            worst = max(worst, _offset_mas(a, d, m["m0"]["value"], m["m1"]["value"]))
            # the frame bias is a rotation of about 23 mas at most
            assert 0.0 < _offset_mas(a, d, ra, dec) < 25.0
            back = convert_direction(a, d, f_out, f_in)
            assert _offset_mas(back[0], back[1], ra, dec) < 1e-6
    print(f"\n[frames] convert_direction vs me.measure: max {worst:.2e} mas")
    assert worst < TOL_MAS


@pytest.mark.parametrize("text", ["15:45:06.322 -34.17.38.348", "15 45 06.322 -34 17 38.348",
                                  "04:27:02.5730000 +26.05.30.1700000"])
def test_icrs_to_j2000_matches_measures(text):
    from ajisai.ms_utils import _split_radec, icrs_to_j2000

    me, qa, _ = _tools()
    out = icrs_to_j2000(text)
    ra_out, dec_out = out.split()
    assert ra_out.count(":") == 2 and ":" not in dec_out and dec_out.count(".") == 3
    ra_in, dec_in = _split_radec(text)
    dec_in_casa = dec_in.replace(":", "d", 1).replace(":", "m", 1) + "s"
    want = me.measure(me.direction("ICRS", ra_in, dec_in_casa), "J2000")
    got = me.direction("J2000", ra_out, dec_out)   # CASA reads the '.'-separated Dec as degrees
    off = _offset_mas(got["m0"]["value"], got["m1"]["value"],
                      want["m0"]["value"], want["m1"]["value"])
    print(f"\n[frames] icrs_to_j2000({text!r}) vs me.measure: {off:.2e} mas")
    assert off < TOL_MAS


def test_rad_to_radec_is_read_by_casa_as_the_same_direction():
    from ajisai.ms_utils import rad_to_radec

    me, qa, _ = _tools()
    for ra, dec in ((4.30114, -0.42248), (1.2, 0.45), (0.001, -0.0000001), (6.28, -1.5)):
        ra_s, dec_s = rad_to_radec(ra, dec, precision=7).split()
        assert ":" not in dec_s
        got = me.direction("ICRS", ra_s, dec_s)
        assert _offset_mas(got["m0"]["value"], got["m1"]["value"], ra, dec) < TOL_MAS


def test_colon_dec_is_degrees_for_ajisai_and_hours_for_casa():
    """The documented form ('-24:12:23.0') is read as degrees by AJISAI; CASA reads it as hours."""
    from ajisai.ms_utils import parse_ra_dec

    me, qa, _ = _tools()
    ra, dec = parse_ra_dec("16:25:45.0", "-24:12:23.0")
    want = me.direction("ICRS", "16h25m45.0s", "-24d12m23.0s")
    assert _offset_mas(ra, dec, want["m0"]["value"], want["m1"]["value"]) < TOL_MAS
    # The hazard the parser avoids: CASA itself reads the ':' Dec as a time.
    assert qa.convert(qa.quantity("-24:12:23.0"), "deg")["value"] == pytest.approx(-363.0958333)


# ---------------------------------------------------------------------------
# The phase-shift path
# ---------------------------------------------------------------------------
def _run_phase_shift(ms, results_dir, phase_center, relabel, field=None):
    """AJISAI's steps up to and including the phase shift (no imaging)."""
    from ajisai import AJISAI, AJISAIConfig

    cfg = AJISAIConfig(vis=ms, results_dir=str(results_dir), field=field, phase_shift=True,
                       phase_center=phase_center, relabel_j2000_to_icrs=relabel,
                       show_banner=False, verbose=False)
    aj = AJISAI(cfg)
    aj._validate_inputs()
    aj._setup_workdir()
    aj._validate_field()
    aj._prepare_working_vis()
    return aj


def _reference_phaseshift(aj, out):
    """phaseshift alone, on the same working MS with the phasecenter AJISAI used."""
    import casatasks

    rec = aj.justification["derived"]["phase_shift"]
    working = aj.justification["derived"]["working_vis"]["working_vis"]
    if os.path.exists(out):
        shutil.rmtree(out)
    casatasks.phaseshift(vis=working, outputvis=out, field=str(rec["field_id_in"]),
                         datacolumn="all", phasecenter=rec["phasecenter"])
    return out


def _main_columns(ms, cols=("UVW", "DATA")):
    _, _, tb = _tools()
    tb.open(ms)
    try:
        return {c: tb.getcol(c) for c in cols}
    finally:
        tb.close()


def _ref_codes(ms):
    _, _, tb = _tools()
    tb.open(ms + "/FIELD")
    try:
        return {c: tb.getcol(c).tolist() for c in ("PhaseDir_Ref", "DelayDir_Ref", "RefDir_Ref")
                if c in tb.colnames()}
    finally:
        tb.close()


@pytest.mark.parametrize("relabel", [False, True])
@pytest.mark.parametrize("target_frame", ["ICRS", "J2000"])
@pytest.mark.parametrize("ms_frame", ["ICRS", "J2000"])
def test_phase_shift_is_exact_in_every_frame(sims, tmp_path, ms_frame, target_frame, relabel):
    import warnings

    me, qa, _ = _tools()
    ms = sims[ms_frame]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aj = _run_phase_shift(ms, tmp_path / "out", TARGET_STR + (target_frame,), relabel)
    working = aj._derived["working_vis"]
    # The simulated MSs are dated 2017: a J2000 label on them is genuine J2000 and is never relabelled
    # (ASEBI SPEC 4.1 item 10, D37); with relabel=True the kept label is warned about and recorded.
    assert working == ms
    if relabel and ms_frame == "J2000":
        caveat = aj.justification["derived"]["working_vis"]["relabel_caveat"]
        assert caveat and "Cycle 3" in caveat and "kept" in caveat
        assert any("Cycle 3" in str(w.message) for w in caught)
    codes_before = _ref_codes(working)

    aj._apply_phase_shift_if_requested()
    shifted = aj._derived["working_vis"]
    rec = aj.justification["derived"]["phase_shift"]
    assert shifted.endswith("phaseshifted.ms") and rec["applied"]
    fid = int(aj._derived["resolved_field_id"])

    # FIELD (three columns) and SOURCE, converted to ICRS, on the target.
    ta, td = _target_icrs(target_frame)
    stored = _stored_icrs(shifted, fid)
    offsets = {k: _offset_mas(a, d, ta, td) for k, (a, d) in stored.items()}
    assert set(offsets) >= {"PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"}
    assert any(k.startswith("SOURCE[") for k in offsets)
    # Frame codes are as in the working MS: no label was changed by the shift.
    assert _ref_codes(shifted) == codes_before

    # UVW and DATA identical to phaseshift's own output (UVW not recomputed).
    ref = _reference_phaseshift(aj, str(tmp_path / "reference.ms"))
    got, want = _main_columns(shifted), _main_columns(ref)
    duvw = float(np.max(np.abs(got["UVW"] - want["UVW"])))
    assert np.array_equal(got["UVW"], want["UVW"])
    assert np.array_equal(got["DATA"], want["DATA"])
    # The phasecenter AJISAI passed is the target, in the field's frame.
    ps = rec["phasecenter"].split()
    pc = me.direction(ps[0], ps[1], ps[2])
    pa, pd = _to_icrs(pc["m0"]["value"], pc["m1"]["value"], ps[0])
    pc_off = _offset_mas(pa, pd, ta, td)
    worst = max(offsets.values())
    print(f"\n[frames] MS {ms_frame:5s} target {target_frame:5s} relabel {relabel!s:5s}: "
          f"max FIELD/SOURCE offset {worst:.2e} mas "
          + " ".join(f"{k}={v:.1e}" for k, v in sorted(offsets.items()))
          + f"; phasecenter {pc_off:.1e} mas; max|dUVW| {duvw:.1e} m; "
            f"recorded read-back {rec['readback_offset_mas']['max_mas']:.1e} mas")
    assert worst < TOL_MAS
    assert pc_off < TOL_MAS
    assert rec["readback_offset_mas"]["max_mas"] < TOL_MAS
    # The records are written to justification.json.
    aj._write_justification_json()
    with open(os.path.join(str(tmp_path / "out"), "justification.json")) as fh:
        saved = json.load(fh)["derived"]
    assert saved["phase_shift"]["field_source_write"]["field_ids"] == [fid]
    assert "relabel" not in saved["working_vis"]                      # never relabelled (2017 data, D37)


def test_phase_shift_finds_the_renumbered_field(sims, tmp_path):
    """phaseshift renumbers the selected field from 0; AJISAI follows it by name."""
    import casatasks
    from ajisai.ms_utils import field_id_by_name
    from tests.casa_sim import simulate_ms

    _, _, tb = _tools()
    other = simulate_ms(str(tmp_path / "other.ms"), nchan=(2, 2), chan_mhz=(500.0, 500.0),
                        scans=1, scan_s=60.0, phase_rms_deg=0.0, ra="16h10m00.0s",
                        dec="-31d00m00.0s", date="2017/05/01/01:00:00", frame="ICRS")
    for sub in ("FIELD", "SOURCE"):
        tb.open(other + "/" + sub, nomodify=False)
        try:
            tb.putcol("NAME", np.array(["OTHER"] * tb.nrows()))
        finally:
            tb.close()
    two = str(tmp_path / "two.ms")
    casatasks.concat(vis=[other, sims["ICRS"]], concatvis=two)
    fid_in = field_id_by_name(two, "SIM")
    if fid_in == 0:  # make the target the second field whatever concat's order
        shutil.rmtree(two)
        casatasks.concat(vis=[sims["ICRS"], other], concatvis=two)
        fid_in = field_id_by_name(two, "OTHER")
        name = "OTHER"
    else:
        name = "SIM"
    assert fid_in != 0
    aj = _run_phase_shift(two, tmp_path / "out", TARGET_STR + ("ICRS",), False, field=name)
    aj._apply_phase_shift_if_requested()
    shifted = aj._derived["working_vis"]
    rec = aj.justification["derived"]["phase_shift"]
    fid_out = field_id_by_name(shifted, name)
    assert rec["field_id_in"] == fid_in and rec["field_id_out"] == fid_out
    assert aj._derived["resolved_field_id"] == str(fid_out)
    ta, td = _target_icrs("ICRS")
    offsets = {k: _offset_mas(a, d, ta, td) for k, (a, d) in _stored_icrs(shifted, fid_out).items()}
    print(f"\n[frames] renumbered field {fid_in} -> {fid_out}: max offset "
          f"{max(offsets.values()):.2e} mas")
    assert max(offsets.values()) < TOL_MAS


def test_phase_shift_auto_target_is_the_peak_pixel_centre(sims, tmp_path):
    import casatasks
    import casatools

    aj = _run_phase_shift(sims["ICRS"], tmp_path / "out", None, False)
    dirty = str(tmp_path / "out" / "intermediate" / "dirty")
    casatasks.tclean(vis=sims["ICRS"], imagename=dirty, imsize=[64, 64], cell="0.1arcsec",
                     specmode="mfs", niter=0, weighting="natural", interactive=False)
    ia = casatools.image()
    ia.open(dirty + ".image")
    try:
        pix = ia.statistics()["maxpos"]
        world = ia.toworld(pix, "n")["numeric"]
        frame = ia.coordsys().referencecode("direction")[0]
        units = ia.coordsys().units()[:2]
    finally:
        ia.close()
    _, qa, _ = _tools()
    pa = qa.convert(qa.quantity(world[0], units[0]), "rad")["value"]
    pd = qa.convert(qa.quantity(world[1], units[1]), "rad")["value"]
    ta, td = _to_icrs(pa, pd, frame)
    aj._apply_phase_shift_if_requested()
    rec = aj.justification["derived"]["phase_shift"]
    assert rec["source"].startswith("auto") and rec["dirty_max_pixel"][:2] == list(pix[:2])
    fid = int(aj._derived["resolved_field_id"])
    offsets = {k: _offset_mas(a, d, ta, td)
               for k, (a, d) in _stored_icrs(aj._derived["working_vis"], fid).items()}
    print(f"\n[frames] auto target (pixel {list(pix[:2])}): max offset "
          f"{max(offsets.values()):.2e} mas")
    assert max(offsets.values()) < TOL_MAS


# ---------------------------------------------------------------------------
# The input relabel (label only) maps codes through the MEASINFO keywords
# ---------------------------------------------------------------------------
def _numbers(ms):
    _, _, tb = _tools()
    out = []
    for table, cols in (("FIELD", ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR")),
                        ("SOURCE", ("DIRECTION",))):
        tb.open(ms + "/" + table)
        try:
            out.extend(np.asarray(tb.getcol(c)).ravel().tolist() for c in cols)
        finally:
            tb.close()
    return out


def _all_frames(ms):
    _, _, tb = _tools()
    out = {}
    for table, cols in (("FIELD", ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR")),
                        ("SOURCE", ("DIRECTION",))):
        tb.open(ms + "/" + table)
        try:
            for c in cols:
                out[f"{table}.{c}"] = _frames(tb, c)
        finally:
            tb.close()
    return out


@pytest.mark.parametrize("layout", ["alma", "fixed", "other_codes"])
def test_relabel_maps_codes_through_measinfo(sims, tmp_path, layout):
    from ajisai.core import _ms_needs_relabel, relabel_J2000_to_ICRS
    from tests.casa_sim import FIELD_REF_COLUMNS, simulate_ms

    _, _, tb = _tools()
    ms = str(tmp_path / "relabel.ms")
    if layout == "fixed":
        simulate_ms(ms, nchan=(2, 2), chan_mhz=(500.0, 500.0), scans=1, scan_s=60.0,
                    phase_rms_deg=0.0, alma_field_layout=False)
    else:
        shutil.copytree(sims["J2000"], ms)
    if layout == "other_codes":
        # A code table where J2000 is not 0 and ICRS is not 21: only the
        # MEASINFO says which code is which.
        tb.open(ms + "/FIELD", nomodify=False)
        try:
            for col, ref in FIELD_REF_COLUMNS:
                tb.putcolkeyword(col, "MEASINFO", {
                    "type": "direction", "VarRefCol": ref,
                    "TabRefTypes": np.array(["ICRS", "J2000"]),
                    "TabRefCodes": np.array([5, 7], dtype=np.uint64)})
                tb.putcol(ref, np.full(tb.nrows(), 7, dtype=np.int32))
        finally:
            tb.close()
    assert all(f == ["J2000"] for f in _all_frames(ms).values())
    numbers = _numbers(ms)
    assert _ms_needs_relabel(ms)
    rec = relabel_J2000_to_ICRS(ms)
    assert rec["errors"] == []
    assert all(f == ["ICRS"] for f in _all_frames(ms).values())
    assert _numbers(ms) == numbers                       # label only
    assert not _ms_needs_relabel(ms)
    import casatools
    msmd = casatools.msmetadata()
    msmd.open(ms)
    try:
        assert msmd.phasecenter(0)["refer"] == "ICRS"
    finally:
        msmd.close()
    if layout == "other_codes":
        assert rec["FIELD"]["PHASE_DIR"]["icrs_code"] == 5
    if layout == "fixed":
        assert rec["FIELD"]["PHASE_DIR"]["reference"] == "fixed"
    again = relabel_J2000_to_ICRS(ms)                     # idempotent
    assert all(r["rows_relabelled"] == [] for r in again["FIELD"].values())
    assert _numbers(ms) == numbers and not _ms_needs_relabel(ms)
