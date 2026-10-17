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
) -> str:
    """Simulate one execution block into ``msname`` (DATA only; FIELD labelled J2000).

    ``nchan`` and ``chan_mhz`` are per spw, so spws with different channel
    counts (the case that makes tclean drop data, see the consistency checks)
    can be built.
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
    return msname
