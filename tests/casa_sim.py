"""
Tiny simulated ALMA measurement sets for the CASA-marked tests.

Built with ``casatools.simulator`` from an ALMA configuration file that ships
with casadata. The source is a few Gaussian components; antenna-based phase
errors (a random walk per antenna) are written into DATA so that phase
self-calibration has something to correct. The result has a DATA column only,
like a delivered ``*.ms.split.cal``. Only the casa tests import this module.
"""
from __future__ import annotations

import math
import os
import shutil
from typing import Optional, Sequence, Tuple

import numpy as np

ARCSEC = math.pi / 180.0 / 3600.0

# FIELD direction columns of a real ALMA MS (2016.1.00459.S_TM1, read with
# casatools): PHASE_DIR, DELAY_DIR and REFERENCE_DIR have a variable reference,
# an int column named in MEASINFO['VarRefCol'] whose codes map to frames through
# TabRefTypes/TabRefCodes (AZEL/AZELNE and AZELGEO/AZELNEGEO share codes).
ALMA_TAB_REF_TYPES = (
    "J2000", "JMEAN", "JTRUE", "APP", "B1950", "B1950_VLA", "BMEAN", "BTRUE",
    "GALACTIC", "HADEC", "AZEL", "AZELSW", "AZELNE", "AZELGEO", "AZELSWGEO",
    "AZELNEGEO", "JNAT", "ECLIPTIC", "MECLIPTIC", "TECLIPTIC", "SUPERGAL", "ITRF",
    "TOPO", "ICRS", "MERCURY", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS",
    "NEPTUNE", "PLUTO", "SUN", "MOON", "COMET",
)
ALMA_TAB_REF_CODES = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 12, 13, 12, 14, 15, 16, 17, 18, 19,
    20, 21, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
)
FIELD_REF_COLUMNS = (("PHASE_DIR", "PhaseDir_Ref"), ("DELAY_DIR", "DelayDir_Ref"),
                     ("REFERENCE_DIR", "RefDir_Ref"))

DEFAULT_SOURCE = (
    # flux (Jy), dx ("), dy ("), major ("), minor ("), pa (deg)
    (0.060, 0.00, 0.00, 0.50, 0.35, 30.0),
    (0.025, 0.35, -0.20, 0.30, 0.30, 0.0),
)


def _offset_dir(ra0: float, dec0: float, dx: float, dy: float) -> Tuple[float, float]:
    d = math.atan2(math.sin(dec0) + dy * math.cos(dec0),
                   math.hypot(dx, math.cos(dec0) - dy * math.sin(dec0)))
    a = ra0 + math.atan2(dx, math.cos(dec0) - dy * math.sin(dec0))
    return a, d


def simulate_ms(
    msname: str,
    *,
    cfg: str = "alma.cycle7.3.cfg",
    nant: Optional[int] = 16,
    freq_ghz: Sequence[float] = (230.0, 232.0),
    nchan: Sequence[int] = (4, 8),
    chan_mhz: Sequence[float] = (250.0, 125.0),
    scans: int = 3,
    scan_s: float = 180.0,
    int_s: float = 6.0,
    noise: str = "5mJy",
    phase_rms_deg: float = 40.0,
    source: Sequence[Sequence[float]] = DEFAULT_SOURCE,
    seed: int = 1,
    date: str = "2017/05/01/00:00:00",
    ra: str = "16h00m00.0s",
    dec: str = "-30d00m00.0s",
    alma_field_layout: bool = True,
    frame: str = "J2000",
) -> str:
    """Simulate one execution block into ``msname`` (DATA only).

    ``nchan`` and ``chan_mhz`` are per spw, so spws with different channel
    counts (the case that makes tclean drop data, see the consistency checks)
    can be built. ``ra``/``dec`` (and the source) are J2000 directions.

    ``alma_field_layout=True`` (default) gives FIELD the layout of a real
    ALMA MS: int columns PhaseDir_Ref, DelayDir_Ref, RefDir_Ref and the
    variable-reference MEASINFO keywords (``give_alma_field_layout``). The
    simulator writes fixed-reference columns, on which CASA tasks such as
    ``phaseshift`` and ``fixplanets`` behave differently.

    ``frame`` is the frame of the stored FIELD and SOURCE directions: "J2000"
    (genuine J2000 numbers, as simulated) or "ICRS" (converted with
    casatools.measures, so the directions are physically the same).
    """
    import casatasks
    import casatools
    from casatasks.private import simutil

    raw = msname.rstrip("/") + ".sim"
    for p in (msname, raw):
        if os.path.exists(p):
            shutil.rmtree(p)
    sm = casatools.simulator()
    me = casatools.measures()
    cl = casatools.componentlist()
    tb = casatools.table()
    u = simutil.simutil()
    xx, yy, zz, diam, pads, _ants, _tel, _pos = u.readantenna(
        casatools.ctsys.resolve(f"alma/simmos/{cfg}"))
    if nant is not None:
        xx, yy, zz, diam, pads = xx[:nant], yy[:nant], zz[:nant], diam[:nant], pads[:nant]
    sm.open(raw)
    sm.setconfig(telescopename="ALMA", x=list(xx), y=list(yy), z=list(zz),
                 dishdiameter=list(map(float, diam)), mount=["alt-az"],
                 antname=list(pads), padname=list(pads), coordsystem="global",
                 referencelocation=me.observatory("ALMA"))
    for i, f in enumerate(freq_ghz):
        sm.setspwindow(spwname=f"spw{i}", freq=f"{f}GHz", deltafreq=f"{chan_mhz[i]}MHz",
                       freqresolution=f"{chan_mhz[i]}MHz", nchannels=int(nchan[i]),
                       stokes="XX YY")
    sm.setfeed("perfect X Y")
    pc = me.direction("J2000", ra, dec)
    sm.setfield(sourcename="SIM", sourcedirection=pc)
    sm.setlimits(shadowlimit=0.001, elevationlimit="8deg")
    sm.setauto(autocorrwt=0.0)
    sm.settimes(integrationtime=f"{int_s}s", usehourangle=True,
                referencetime=me.epoch("UTC", date))
    for k in range(scans):
        t0 = -600.0 + k * (scan_s + 60.0)
        for i in range(len(freq_ghz)):
            sm.observe("SIM", f"spw{i}", starttime=f"{t0}s", stoptime=f"{t0 + scan_s}s",
                       state_obs_mode="OBSERVE_TARGET#ON_SOURCE")
    clname = raw + ".cl"
    if os.path.exists(clname):
        shutil.rmtree(clname)
    ra0 = pc["m0"]["value"]
    dec0 = pc["m1"]["value"]
    for flux, dx, dy, bmaj, bmin, pa in source:
        a, d = _offset_dir(ra0, dec0, dx * ARCSEC, dy * ARCSEC)
        cl.addcomponent(dir=f"J2000 {a}rad {d}rad", flux=flux, fluxunit="Jy",
                        freq=f"{freq_ghz[0]}GHz", shape="Gaussian",
                        majoraxis=f"{bmaj}arcsec", minoraxis=f"{bmin}arcsec",
                        positionangle=f"{pa}deg")
    cl.rename(clname)
    cl.done()
    sm.predict(complist=clname)
    sm.close()
    shutil.rmtree(clname)

    sm.openfromms(raw)
    sm.setseed(seed)
    sm.setnoise(mode="simplenoise", simplenoise=noise)
    sm.corrupt()
    sm.close()

    # Antenna-based phase random walk, identical for both polarizations.
    rng = np.random.default_rng(seed)
    tb.open(raw, nomodify=False)
    try:
        time = tb.getcol("TIME")
        a1 = tb.getcol("ANTENNA1")
        a2 = tb.getcol("ANTENNA2")
        ddid = tb.getcol("DATA_DESC_ID")
        utimes = np.unique(time)
        nant_tot = int(max(a1.max(), a2.max())) + 1
        steps = rng.normal(0.0, 1.0, (len(utimes), nant_tot))
        walk = np.cumsum(steps, axis=0)
        walk = walk - walk.mean(axis=0)
        walk *= math.radians(phase_rms_deg) / max(float(walk.std()), 1e-12)
        tindex = np.searchsorted(utimes, time)
        for dd in np.unique(ddid):
            rows = np.where(ddid == dd)[0]
            sub = tb.selectrows(rows.tolist())
            data = sub.getcol("DATA")
            ph = walk[tindex[rows], a1[rows]] - walk[tindex[rows], a2[rows]]
            data = data * np.exp(1j * ph)[None, None, :]
            sub.putcol("DATA", data)
            sig = float(casatools.quanta().convert(noise, "Jy")["value"])
            w = np.full(sub.getcol("WEIGHT").shape, 1.0 / sig ** 2)
            sub.putcol("WEIGHT", w)
            sub.putcol("SIGMA", np.full(w.shape, sig))
            sub.close()
    finally:
        tb.close()
    casatasks.split(vis=raw, outputvis=msname, datacolumn="data")
    shutil.rmtree(raw)
    me.done()
    if alma_field_layout:
        give_alma_field_layout(msname)
    if frame != "J2000":
        set_direction_frame(msname, frame)
    return msname


def give_alma_field_layout(msname: str) -> None:
    """Give the FIELD direction columns ALMA's variable-reference layout.

    For each of PHASE_DIR, DELAY_DIR and REFERENCE_DIR with a fixed reference,
    an int column (``PhaseDir_Ref`` etc., described as in a real ALMA MS) is
    added and filled with the code of the column's current frame, and the
    MEASINFO keyword becomes ``{'type': 'direction', 'VarRefCol': ...,
    'TabRefTypes': ..., 'TabRefCodes': ...}``. The stored numbers and the
    physical directions are unchanged.
    """
    import casatools

    types = list(ALMA_TAB_REF_TYPES)
    tb = casatools.table()
    tb.open(msname + "/FIELD", nomodify=False)
    try:
        nrow = tb.nrows()
        for col, refcol in FIELD_REF_COLUMNS:
            mi = tb.getcolkeyword(col, "MEASINFO")
            if "VarRefCol" in mi:
                continue
            code = ALMA_TAB_REF_CODES[types.index(str(mi.get("Ref", "J2000")))]
            tb.addcols({refcol: {"valueType": "int", "dataManagerType": "StandardStMan",
                                 "dataManagerGroup": "StandardStMan", "option": 0,
                                 "maxlen": 0, "comment": ""}})
            tb.putcol(refcol, np.full(nrow, code, dtype=np.int32))
            tb.putcolkeyword(col, "MEASINFO", {
                "type": "direction", "VarRefCol": refcol,
                "TabRefTypes": np.array(types),
                "TabRefCodes": np.array(ALMA_TAB_REF_CODES, dtype=np.uint64),
            })
    finally:
        tb.close()


def _direction_frames(tb, col: str):
    mi = tb.getcolkeyword(col, "MEASINFO")
    if "VarRefCol" in mi:
        types = [str(t) for t in mi["TabRefTypes"]]
        codes = [int(c) for c in mi["TabRefCodes"]]
        return [types[codes.index(int(c))] for c in tb.getcol(mi["VarRefCol"])]
    return [str(mi.get("Ref", "J2000"))] * tb.nrows()


def set_direction_frame(msname: str, frame: str) -> None:
    """Store every FIELD and SOURCE direction in ``frame`` (numbers converted, labels set).

    The numbers are converted with casatools.measures from each cell's own
    frame, and the reference codes (variable reference) or MEASINFO Ref
    (fixed) are set to ``frame``: the physical directions are unchanged.
    """
    import casatools

    me = casatools.measures()
    qa = casatools.quanta()
    tb = casatools.table()

    def conv(ra, dec, f_in):
        m = me.measure(me.direction(f_in, qa.quantity(float(ra), "rad"),
                                    qa.quantity(float(dec), "rad")), frame)
        return m["m0"]["value"], m["m1"]["value"]

    try:
        tables = (("FIELD", [c for c, _ in FIELD_REF_COLUMNS]), ("SOURCE", ["DIRECTION"]))
        for table, cols in tables:
            if not os.path.isdir(msname + "/" + table):
                continue
            tb.open(msname + "/" + table, nomodify=False)
            try:
                for col in cols:
                    frames = _direction_frames(tb, col)
                    for r in range(tb.nrows()):
                        cell = np.array(tb.getcell(col, r), dtype=float)
                        if cell.ndim == 2:
                            cell[0, 0], cell[1, 0] = conv(cell[0, 0], cell[1, 0], frames[r])
                        else:
                            cell[0], cell[1] = conv(cell[0], cell[1], frames[r])
                        tb.putcell(col, r, cell)
                    mi = tb.getcolkeyword(col, "MEASINFO")
                    if "VarRefCol" in mi:
                        types = [str(t) for t in mi["TabRefTypes"]]
                        code = int(mi["TabRefCodes"][types.index(frame)])
                        tb.putcol(mi["VarRefCol"], np.full(tb.nrows(), code, dtype=np.int32))
                    else:
                        mi["Ref"] = frame
                        tb.putcolkeyword(col, "MEASINFO", mi)
            finally:
                tb.close()
    finally:
        me.done()
