"""
Simulated reduced visibilities for the tests of ``ajisai.solint`` (numpy only).

A point source observed by ``n_ant`` antennas in scans of integrations: every
row (baseline, integration, spw) has the model power ``P`` and
``y = P exp(i(phi_1 - phi_2)) + noise`` with ``Var(y) = 2 s P``, which is the
reduction ``y = sum w V M*``, ``P = sum w |M|^2`` of a measurement set whose
weights are wrong by the factor ``s``. Antenna phases follow a Gaussian
process with the structure function ``D(tau) = (tau/tau0)^beta`` (per
antenna), optionally with random phase jumps. ``production`` applies a
numpy full-data solution the way gaincal + applycal (calonly, linearPD or
nearest as the candidate says, G1 gate) would, so that the stage logic can
be run end to end without CASA. ``concat`` joins execution blocks (e.g.
with different integration times) into one run.
"""
from __future__ import annotations

import numpy as np

from ajisai import solint as so


def phase_screen(times, n_ant, tau0, beta, rng):
    """Antenna phases (n_times, n_ant) with D(tau) = (tau/tau0)^beta (fBm, H = beta/2)."""
    t = np.asarray(times, dtype=float) - float(times[0]) + 1.0
    if tau0 is None or not np.isfinite(tau0):
        return np.zeros((t.size, n_ant))
    D = lambda x: (np.abs(x) / tau0) ** beta  # noqa: E731
    cov = 0.5 * (D(t)[:, None] + D(t)[None, :] - D(t[:, None] - t[None, :]))
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(t.size))
    ph = L @ rng.standard_normal((t.size, n_ant))
    return ph


def phase_jumps(times, n_ant, rate, rms, rng):
    """Antenna phase steps: Poisson times (``rate`` per s and antenna), Gaussian sizes (``rms`` rad)."""
    t = np.asarray(times, dtype=float)
    out = np.zeros((t.size, n_ant))
    span = float(t[-1] - t[0]) if t.size else 0.0
    for a in range(n_ant):
        n = rng.poisson(rate * span)
        at = np.sort(rng.uniform(t[0], t[-1], n))
        size = rng.normal(0.0, rms, n)
        out[:, a] = np.array([size[at <= x].sum() for x in t])
    return out


def simulate(*, n_ant=16, n_scans=4, n_int=20, it=6.0, gap=60.0, n_spw=1, r=1.0, s=1.0,
             tau0=300.0, beta=1.0, seed=1, halves=False, obs_offset=0.0, jumps=None):
    """One execution block of reduced rows; ``r`` is the per-antenna SNR rate (all spws).

    ``jumps=(rate_per_s, rms_rad)`` adds antenna phase steps (:func:`phase_jumps`)
    drawn from a separate random stream, so the rest of the simulation is the
    same with and without them.
    """
    rng = np.random.default_rng(seed)
    times = []
    scans = []
    t = 0.0
    for k in range(n_scans):
        for j in range(n_int):
            times.append(t + obs_offset)
            scans.append(k + 1)
            t += it
        t += gap - it
    times = np.asarray(times)
    scans = np.asarray(scans)
    phases = phase_screen(times, n_ant, tau0, beta, rng)
    if jumps is not None:
        phases = phases + phase_jumps(times, n_ant, jumps[0], jumps[1],
                                      np.random.default_rng(seed + 1000))
    a1, a2 = np.triu_indices(n_ant, 1)
    nbl = a1.size
    # per-antenna information per integration: r^2 IT = (n_ant - 1) P / s (summed over spws)
    p_row = r ** 2 * it * s / (n_ant - 1) / n_spw
    rows = []
    for ti in range(times.size):
        for sp in range(n_spw):
            rows.append((ti, sp))
    ti = np.repeat([r_[0] for r_ in rows], nbl)
    sp = np.repeat([r_[1] for r_ in rows], nbl)
    A1 = np.tile(a1, len(rows))
    A2 = np.tile(a2, len(rows))
    n = ti.size
    P = np.full(n, p_row)
    sig = np.exp(1j * (phases[ti, A1] - phases[ti, A2]))
    if halves:
        yh = np.zeros((2, n), complex)
        for h in range(2):
            noise = np.sqrt(s * P / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            yh[h] = P / 2 * sig + noise
        y = yh[0] + yh[1]
        Ph = np.stack([P / 2, P / 2])
    else:
        y = P * sig + np.sqrt(s * P) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        yh = Ph = None
    uv = np.tile(rng.uniform(10, 1000, nbl), len(rows))
    vs = so.VisSummary(time=times[ti], obs=np.zeros(n, int), scan=scans[ti], spw=sp, a1=A1, a2=A2,
                       y=y, P=P, antenna_names=[f"A{k:02d}" for k in range(n_ant)], S=y.copy(),
                       uvdist_m=uv, y_half=yh, P_half=Ph)
    order = np.lexsort((vs.time, vs.a2, vs.a1, vs.spw, vs.obs))
    return reorder(vs, order), phases


def concat(parts):
    """Concatenate simulated execution blocks: observation k is ``parts[k]`` (rows re-sorted).

    ``parts``: (VisSummary, phases) pairs as returned by :func:`simulate`
    (each with its own ``obs_offset``). Returns the joint VisSummary and the
    phases of each part (a list).
    """
    cat = {}
    for name in ("time", "scan", "spw", "a1", "a2", "y", "P", "uvdist_m"):
        cat[name] = np.concatenate([np.asarray(getattr(v, name)) for v, _ in parts])
    obs = np.concatenate([np.full(v.n_rows, k, int) for k, (v, _) in enumerate(parts)])
    halves = all(v.y_half is not None for v, _ in parts)
    vs = so.VisSummary(time=cat["time"], obs=obs, scan=cat["scan"], spw=cat["spw"], a1=cat["a1"],
                       a2=cat["a2"], y=cat["y"], P=cat["P"],
                       antenna_names=list(parts[0][0].antenna_names), S=cat["y"].copy(),
                       uvdist_m=cat["uvdist_m"],
                       y_half=np.concatenate([v.y_half for v, _ in parts], axis=1) if halves else None,
                       P_half=np.concatenate([v.P_half for v, _ in parts], axis=1) if halves else None)
    order = np.lexsort((vs.time, vs.a2, vs.a1, vs.spw, vs.obs))
    return reorder(vs, order), [p for _, p in parts]


def reorder(vs, order):
    def pick(x):
        if x is None:
            return None
        x = np.asarray(x)
        return x[..., order] if x.ndim == 2 else x[order]
    return so.VisSummary(time=vs.time[order], obs=vs.obs[order], scan=vs.scan[order],
                         spw=vs.spw[order], a1=vs.a1[order], a2=vs.a2[order], y=vs.y[order],
                         P=vs.P[order], antenna_names=list(vs.antenna_names), S=pick(vs.S),
                         uvdist_m=pick(vs.uvdist_m), y_half=pick(vs.y_half),
                         P_half=pick(vs.P_half))


def rotate(vs, rowphase, state0):
    """The next state: every row multiplied by exp(-i rowphase); phi0 tracked to state 0."""
    e = np.exp(-1j * rowphase)
    new = so.VisSummary(time=vs.time, obs=vs.obs, scan=vs.scan, spw=vs.spw, a1=vs.a1, a2=vs.a2,
                        y=vs.y * e, P=vs.P, antenna_names=vs.antenna_names, S=vs.S * e,
                        uvdist_m=vs.uvdist_m,
                        y_half=None if vs.y_half is None else vs.y_half * e[None, :],
                        P_half=vs.P_half)
    new.track_phase(state0)
    return new


def production(vs, geom, cand, state0, combine_spw=False, refant=0):
    """Full-data solve of ``cand`` with the G1 gate, applied to every row (gaincal stand-in)."""
    units = so.solution_units(vs, geom, cand, combine_spw)
    s_rows = so._s_rows(vs, so.noise_scale(vs, geom))
    ph, valid = so._solve_units(vs, units, vs.y, vs.P, np.ones(vs.n_rows, bool), refant, 4)
    good = valid & (so.unit_snr(vs, units, s_rows) >= so.G1_SNR_MIN)
    rowphase, ok = so.apply_units(vs, units, ph, good, cand.get("interp", "linear"))
    return rotate(vs, rowphase, state0)


def run_loop(state0, refant=0, combine_spw=False):
    """The adaptive phase loop of AJISAI (the model never changes: point source)."""
    geom = so.geometry(state0)
    state = state0
    chain = []
    records = []
    for stage in range(1, so.MAX_PHASE_STAGES + 1):
        rec = so.evaluate_stage(stage, state, chain, geom, combine_spw=combine_spw, refant=refant)
        records.append(rec)
        if rec["decision"] == "apply":
            cand = rec["_"]["chosen"]
            chain.append(so.StageSolve(stage, cand, state, rec["_"]["s_rows"],
                                       rec["_"]["chosen_corr"]))
            state = production(state, geom, cand, state0, combine_spw, refant)
        elif rec["decision"] == "stop" or rec.get("chosen") is None:
            break
    return records, chain, state


def true_coherence(vs, phases, times):
    """Noise-free coherence ``|sum P e^{i resid}| / sum P`` of a state (residual phases known)."""
    ti = np.searchsorted(np.asarray(times), vs.time)
    resid = phases[ti, vs.a1] - phases[ti, vs.a2] - vs.phi0
    return float(np.abs(np.sum(vs.P * np.exp(1j * resid))) / np.sum(vs.P))
