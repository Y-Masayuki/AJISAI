#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ajisai.solint
=============
Data-driven solution intervals for phase self-calibration (pure numpy).

The rules are those of the ASEBI specification v0.3 (SPEC section 10.2,
appendices C.2 and C.3) and of the design-review records P17/P18 (decision
block, sections 3.1-3.5, test results sections 3-5, 7, 9 and 10), P23 (the
measurement self-cal uses the same rules) and P27 (no iteration loop on the
ASEBI side). Every public function names its section in a ``Spec:`` line.

What is measured, from the visibilities, the model and the weights only
(no gain solutions, no images):

* M1 ``s``: noise scale per spw from adjacent integrations
  (:func:`noise_scale`);
* M2 ``r_i``: solution-SNR rate per antenna, ``SNR_i(T) = r_i sqrt(T)``
  (:func:`snr_rates`);
* M3 ``D_eff(tau)``: model-weighted lag structure function of the phase,
  unbiased by thermal noise in its numerator (:func:`lag_structure_function`);
* M4 ``kappa``: interpolation geometry of the solutions
  (:func:`kappa_geometry`).

How a stage is decided (:func:`evaluate_stage`):

1. candidate solints: divisors of the median scan length in integrations,
   the scan and the execution block (:func:`candidate_solints`);
2. G1: predicted true SNR >= 5 per candidate and per antenna and solution
   (:func:`g1_gate`);
3. prediction of the residual phase variance ``V(T)`` on the candidate grid
   with the actual times and the interpolation weights
   (:func:`predict_residual_variance`);
4. G2: held-out coherence with an even/odd split of the integrations in
   each scan (band halves when scans have fewer than 4 integrations),
   antenna delete-one paired jackknife (:func:`holdout_coherence`);
   rule A (loss-table corrected argmin) and rule B (one candidate shorter
   than the held-out argmin), the longer when they disagree
   (:func:`choose_solint`);
5. acceptance: corrected held-out improvement > max(1 sigma, 0.001) with the
   conditional two-sided correction (:func:`acceptance`); stage 1 (one
   solution per EB) is skipped only when its corrected change is below
   -2 sigma.

Data model. Every quantity is computed from a per-row reduction of a
measurement set (one row = one baseline, integration and spw):
``y = sum w V M*`` and ``P = sum w |M|^2`` over the unflagged parallel-hand
correlations and channels (:class:`VisSummary`). A phase-only gain with
gaintype 'T' multiplies every channel and correlation of a row by the same
factor, so applying a solution to the reduced row is exact.

CASA is never imported here; ``ajisai.ms_utils.read_vis_summary`` fills a
:class:`VisSummary` from a measurement set.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================================
# Rules (every number is pinned by tests/test_solint.py)
# ============================================================================

#: True SNR of the CASA loss table (Spec C.2).
LOSS_TABLE_SNR = (2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0)
#: CASA gaincal phase-solution loss -ln C at that true SNR (per antenna; the
#: coherence of a baseline between two such antennas is exp(-L)).
LOSS_TABLE_NEG_LN_C = (0.844, 0.400, 0.198, 0.0856, 0.0492, 0.0233, 0.0113, 0.0027)

#: G1: predicted true SNR of a solution (Spec 10.2 item 4).
G1_SNR_MIN = 5.0
#: gaincal minsnr: the floor on the SNR column (never 5; Spec 10.2 item 4).
GAINCAL_MINSNR = 3.0
#: Acceptance of a stage: corrected improvement > max(ACCEPT_SIGMA sigma, ACCEPT_FLOOR)
#: (Spec 10.2 item 7).
ACCEPT_SIGMA = 1.0
ACCEPT_FLOOR = 0.001
#: Stage 1 (one solution per EB) is skipped only below -STAGE1_SKIP_SIGMA sigma
#: (Spec 10.2 item 2).
STAGE1_SKIP_SIGMA = 2.0
#: At most this many phase stages, stage 1 included (Spec 10.2 item 8).
MAX_PHASE_STAGES = 3
#: G2 splits the band into halves when a scan has fewer integrations (Spec 10.2 item 6).
BAND_SPLIT_BELOW_NINT = 4
#: Neighbouring divisor candidates further apart than this ratio get
#: non-divisor multiples in between (Spec 10.2 item 3).
CANDIDATE_MAX_RATIO = 2.0
#: Structure-function exponent range of the power-law fit (D(tau) = (tau/tau0)^beta).
BETA_RANGE = (0.1, 2.0)
#: A fitted structure function counts only when its amplitude exceeds this many sigma.
FIT_MIN_SIGNIFICANCE = 2.0

# Sample limits for the discrete prediction (accuracy is set by the smoothness
# of D, not by the number of integrations).
_MAX_EVAL_TIMES = 600
_MAX_MEMBERS_PER_UNIT = 300


# ============================================================================
# Loss table and closed forms (Spec C.2)
# ============================================================================

def casa_loss(snr) -> np.ndarray:
    """CASA phase-solution loss ``L = -ln C`` per antenna at a true SNR.

    Spec: C.2 (the CASA loss table), 10.2 item 6 (rule A), P18 test results 4-5.

    Log-log interpolation in the table; above SNR 20 the thermal limit
    ``L = 0.0027 (20/SNR)^2``; below SNR 2 the first segment is extended
    (the solution is noise: the loss grows steeply). ``SNR <= 0`` gives inf.
    """
    snr = np.asarray(snr, dtype=float)
    out = np.full(snr.shape, np.inf)
    xs = np.log(np.asarray(LOSS_TABLE_SNR))
    ys = np.log(np.asarray(LOSS_TABLE_NEG_LN_C))
    pos = np.isfinite(snr) & (snr > 0)
    ls = np.log(np.where(pos, snr, 1.0))
    inside = pos & (snr >= LOSS_TABLE_SNR[0]) & (snr <= LOSS_TABLE_SNR[-1])
    out[inside] = np.exp(np.interp(ls[inside], xs, ys))
    high = pos & (snr > LOSS_TABLE_SNR[-1])
    out[high] = LOSS_TABLE_NEG_LN_C[-1] * (LOSS_TABLE_SNR[-1] / snr[high]) ** 2
    low = pos & (snr < LOSS_TABLE_SNR[0])
    slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    out[low] = np.exp(ys[0] + slope * (ls[low] - xs[0]))
    return out


def box_coefficient(beta: float) -> float:
    """``a`` of nearest (piecewise-constant) interpolation: ``1/((beta+1)(beta+2))``.

    Spec: C.2 (``V = a (T/tau0)^beta + kappa/(r^2 T)``), P18 3.1.
    """
    return 1.0 / ((beta + 1.0) * (beta + 2.0))


def atmospheric_coefficient(beta: float, interp: str = "linear", n_sub: int = 400,
                            n_eval: int = 201) -> float:
    """``a(beta, interp)``: residual atmospheric phase variance per ``(T/tau0)^beta``.

    Spec: C.2 (a = 1/((beta+1)(beta+2)) for nearest; 0.34-0.36, 0.77-0.80 and
    0.89-0.92 of that for linear interpolation at beta = 5/3, 1, 2/3).

    ``interp="nearest"``: the closed form. ``"linear"``: CASA's linear
    interpolation between the means of two neighbouring intervals of length
    T on a uniform interior grid, computed numerically with the variogram
    identity (:func:`_interval_variance`), with ``D(tau) = |tau|^beta``.
    """
    if interp == "nearest":
        return box_coefficient(beta)
    if interp != "linear":
        raise ValueError(f"interp must be 'linear' or 'nearest'; got {interp!r}")
    x0 = -1.0 + (np.arange(n_sub) + 0.5) / n_sub
    x1 = (np.arange(n_sub) + 0.5) / n_sub
    t = -0.5 + (np.arange(n_eval) + 0.5) / n_eval
    f = t + 0.5

    def dfun(lag):
        return np.abs(lag) ** beta

    s00 = dfun(x0[:, None] - x0[None, :]).mean()
    s01 = dfun(x0[:, None] - x1[None, :]).mean()
    m0 = dfun(t[:, None] - x0[None, :]).mean(axis=1)
    m1 = dfun(t[:, None] - x1[None, :]).mean(axis=1)
    var = (1 - f) * m0 + f * m1 - 0.5 * ((1 - f) ** 2 * s00 + f ** 2 * s00
                                          + 2 * f * (1 - f) * s01)
    return float(var.mean())


#: kappa of linear interpolation on a uniform interior grid (mean of (1-f)^2 + f^2).
KAPPA_LINEAR_INTERIOR = 2.0 / 3.0


def residual_variance_closed(T, tau0: float, beta: float, r: float, a: float,
                             kappa: float = KAPPA_LINEAR_INTERIOR):
    """``V(T) = a (T/tau0)^beta + kappa/(r^2 T)`` (per antenna; baseline coherence exp(-V)).

    Spec: C.2.
    """
    T = np.asarray(T, dtype=float)
    return a * (T / tau0) ** beta + kappa / (r ** 2 * T)


def optimal_solint(tau0: float, beta: float, r: float, a: Optional[float] = None,
                   kappa: float = KAPPA_LINEAR_INTERIOR, interp: str = "linear") -> float:
    """``T* = [kappa tau0^beta / (a beta r^2)]^(1/(beta+1))``.

    Spec: C.2 (T*, V* = (1 + 1/beta) kappa / SNR(T*)^2), P18 3.1.
    """
    if a is None:
        a = atmospheric_coefficient(beta, interp)
    return float((kappa * tau0 ** beta / (a * beta * r ** 2)) ** (1.0 / (beta + 1.0)))


def variance_ratio(x, beta: float):
    """``V(x T*) / V* = (x^beta + beta/x) / (1 + beta)``.

    Spec: C.2 (1.5x the optimum costs 8-15 %, 2x costs 25-50 %).
    """
    x = np.asarray(x, dtype=float)
    return (x ** beta + beta / x) / (1.0 + beta)


# ============================================================================
# Reduced visibilities
# ============================================================================

@dataclass
class VisSummary:
    """Per-row reduction of one self-calibration state.

    Spec: C.2 (``z = sum w V M* / sum w |M|^2``, ``P = sum w |M|^2``), 10.2 item 14
    (DATA, MODEL_DATA, WEIGHT and FLAG are read once).

    One row = one cross-correlation baseline, integration and spw of the
    target field. ``y = sum w V M*`` and ``P = sum w |M|^2`` over the unflagged
    parallel-hand correlations and channels; ``y_half``/``P_half`` the same
    over the lower and upper half of each spw's channels (band split of G2).
    ``S = sum V`` over the same elements tracks the phase applied between
    two states of the same rows. ``phi0`` is the per-row phase that maps
    this state's data back to state 0 (``DATA_0 = DATA e^{i phi0}``), set by
    :meth:`track_phase`.
    """
    time: np.ndarray
    obs: np.ndarray
    scan: np.ndarray
    spw: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    y: np.ndarray
    P: np.ndarray
    antenna_names: List[str]
    S: Optional[np.ndarray] = None
    uvdist_m: Optional[np.ndarray] = None
    y_half: Optional[np.ndarray] = None
    P_half: Optional[np.ndarray] = None
    phi0: Optional[np.ndarray] = None
    source: str = ""

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        for name in ("obs", "scan", "spw", "a1", "a2"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.int64))
        self.y = np.asarray(self.y, dtype=complex)
        self.P = np.asarray(self.P, dtype=float)
        n = self.time.size
        for name in ("obs", "scan", "spw", "a1", "a2", "y", "P"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"VisSummary.{name} must have shape ({n},)")
        if self.phi0 is None:
            self.phi0 = np.zeros(n)

    @property
    def n_rows(self) -> int:
        return int(self.time.size)

    @property
    def n_ant(self) -> int:
        return len(self.antenna_names)

    def keys(self) -> Tuple[np.ndarray, ...]:
        """Row identity (observation, spw, antenna1, antenna2, time)."""
        return (self.obs, self.spw, self.a1, self.a2, self.time)

    def y0(self, half: Optional[int] = None) -> np.ndarray:
        """``sum w DATA_0 M*`` of this state's model: the data of state 0 against this model."""
        y = self.y if half is None else self.y_half[half]
        return y * np.exp(1j * self.phi0)

    def track_phase(self, state0: "VisSummary") -> None:
        """Set ``phi0`` from the phase-tracking sums of state 0 (exact for phase-only gains).

        Spec: 10.2 item 7 (the reference state is re-solved from state 0 on the
        training half, so every state's model must be compared with DATA_0).
        """
        if self.S is None or state0.S is None:
            raise ValueError("phase tracking needs the S sums of both states")
        if not all(np.array_equal(a, b) for a, b in zip(self.keys(), state0.keys())):
            raise ValueError("the rows of the two states differ; cannot track the applied phase")
        prod = state0.S * np.conj(self.S)
        self.phi0 = np.where(np.abs(prod) > 0, np.angle(prod), 0.0)


#: casacore Stokes codes of the parallel hands (RR, LL, XX, YY): gaintype 'T' uses these.
PARALLEL_HANDS = (5, 8, 9, 12)


def reduce_visibilities(data, model, flag, weight, flag_row=None, corr_types=None,
                        halves: bool = False) -> Dict[str, np.ndarray]:
    """Per-row ``y = sum w V M*``, ``P = sum w |M|^2`` and ``S = sum V`` of one chunk.

    Spec: C.2 (``z = sum w V M* / sum w |M|^2``), 10.2 item 14 (read DATA,
    MODEL_DATA, WEIGHT and FLAG once).

    ``data``, ``model``, ``flag``: (npol, nchan, nrow) as casatools returns
    them; ``weight``: (npol, nrow) (WEIGHT) or (npol, nchan, nrow)
    (WEIGHT_SPECTRUM). Only the parallel hands (``corr_types`` in
    :data:`PARALLEL_HANDS`; all correlations when ``None``) of unflagged,
    finite, positively weighted samples enter. ``halves=True`` adds the sums
    over the lower and upper half of the channels (``nchan >= 2``).
    """
    data = np.asarray(data)
    model = np.asarray(model)
    flag = np.asarray(flag, dtype=bool)
    w = np.asarray(weight, dtype=float)
    if w.ndim == 2:
        w = np.broadcast_to(w[:, None, :], data.shape)
    if corr_types is not None:
        sel = [i for i, c in enumerate(corr_types) if int(c) in PARALLEL_HANDS]
        if not sel:
            sel = list(range(data.shape[0]))
        data, model, flag, w = data[sel], model[sel], flag[sel], w[sel]
    ok = ~flag & np.isfinite(data) & np.isfinite(w) & (w > 0)
    if flag_row is not None:
        ok &= ~np.asarray(flag_row, dtype=bool)[None, None, :]
    okm = ok & np.isfinite(model)
    v = np.where(okm, data, 0.0)
    m = np.where(okm, model, 0.0)
    ww = np.where(okm, w, 0.0)
    prod = ww * v * np.conj(m)
    pw = ww * np.abs(m) ** 2
    out = {
        "y": prod.sum(axis=(0, 1)),
        "P": pw.sum(axis=(0, 1)),
        "S": np.where(ok, data, 0.0).sum(axis=(0, 1)),
        "n_valid": ok.sum(axis=(0, 1)),
    }
    nchan = data.shape[1]
    if halves and nchan >= 2:
        h = nchan // 2
        out["y_half"] = np.stack([prod[:, :h].sum(axis=(0, 1)), prod[:, h:].sum(axis=(0, 1))])
        out["P_half"] = np.stack([pw[:, :h].sum(axis=(0, 1)), pw[:, h:].sum(axis=(0, 1))])
    return out


@dataclass
class Geometry:
    """Time structure of the rows of a run (identical for every state).

    Spec: 10.2 items 3 and 6 (scan lengths in integrations, even/odd split).
    """
    int_time: Dict[int, float]          # integration time per observation (s)
    it_ref: float                       # median integration time of all rows (s)
    k_in_scan: np.ndarray               # index of the row's integration within its scan
    scan_nint: np.ndarray               # integrations per scan (one entry per scan)
    scan_nint_median: float
    n_scans: int
    max_scans_per_obs: int
    obs_span: Dict[int, float]          # first to last integration + one IT (s)
    utimes: Dict[int, np.ndarray]       # unique times per observation
    tgrid: np.ndarray                   # row -> index on the observation's IT grid
    cache: Dict[Any, Any] = field(default_factory=dict)   # row -> unit assignments


def geometry(vs: VisSummary) -> Geometry:
    """Integration time per observation, scan positions and the even/odd parity.

    Spec: 10.2 items 3 and 6; 10 item 5 (solints are integer multiples of the time bin).
    """
    int_time: Dict[int, float] = {}
    utimes: Dict[int, np.ndarray] = {}
    obs_span: Dict[int, float] = {}
    k_in_scan = np.zeros(vs.n_rows, dtype=np.int64)
    tgrid = np.zeros(vs.n_rows, dtype=np.int64)
    scan_nint: List[int] = []
    max_spo = 0
    for o in np.unique(vs.obs):
        sel_o = vs.obs == o
        ut = np.unique(vs.time[sel_o])
        utimes[int(o)] = ut
        diffs = []
        max_spo = max(max_spo, int(np.unique(vs.scan[sel_o]).size))
        for sc in np.unique(vs.scan[sel_o]):
            sel = sel_o & (vs.scan == sc)
            st = np.unique(vs.time[sel])
            scan_nint.append(int(st.size))
            if st.size > 1:
                diffs.append(np.diff(st))
            k_in_scan[sel] = np.searchsorted(st, vs.time[sel])
        d = np.concatenate(diffs) if diffs else np.zeros(0)
        it = float(np.median(d)) if d.size else 1.0
        int_time[int(o)] = it
        tgrid[sel_o] = np.rint((vs.time[sel_o] - ut[0]) / it).astype(np.int64)
        obs_span[int(o)] = float(ut[-1] - ut[0] + it)
    all_it = np.array([int_time[int(o)] for o in vs.obs]) if vs.n_rows else np.ones(1)
    nint = np.asarray(scan_nint, dtype=np.int64)
    return Geometry(int_time=int_time, it_ref=float(np.median(all_it)), k_in_scan=k_in_scan,
                    scan_nint=nint, scan_nint_median=float(np.median(nint)) if nint.size else 0.0,
                    n_scans=int(nint.size), max_scans_per_obs=max_spo, obs_span=obs_span, utimes=utimes, tgrid=tgrid)


def _sort_order(vs: VisSummary) -> np.ndarray:
    return np.lexsort((vs.time, vs.a2, vs.a1, vs.spw, vs.obs))


# ============================================================================
# M1-M3: measured quantities
# ============================================================================

def noise_scale(vs: VisSummary, geom: Geometry) -> Dict[int, Dict[str, Any]]:
    """M1: noise scale ``s`` per spw from adjacent integrations of the solve unit.

    Spec: 10.2 item 1, C.2 (``s`` from adjacent integrations, the quantity of
    Spec 5), Spec 5 item 5 (the ratio of the summed unit, which includes the
    channel-to-channel noise correlation), P18 3.2 (M1).

    For each baseline and spw, consecutive integrations of a scan give
    ``d = z_{k+1} - z_k`` with ``z = y/P``. Its component parallel to the
    pair's mean ``y_k + y_{k+1}`` (whose noise is independent of the noise
    of ``d``) carries the noise ``s (1/P_k + 1/P_{k+1})`` and only the
    second-order part of the signal-phase change between the two
    integrations. ``s`` is the mean ratio; ``s = 1`` means WEIGHT is right
    for the summed unit. Without any pair the spw gets ``s = 1``
    (``"method": "assumed"``).
    """
    order = _sort_order(vs)
    t = vs.time[order]
    same = ((vs.obs[order][1:] == vs.obs[order][:-1]) & (vs.spw[order][1:] == vs.spw[order][:-1])
            & (vs.a1[order][1:] == vs.a1[order][:-1]) & (vs.a2[order][1:] == vs.a2[order][:-1])
            & (vs.scan[order][1:] == vs.scan[order][:-1]))
    it = np.array([geom.int_time[int(o)] for o in vs.obs[order][1:]]) if vs.n_rows > 1 else np.ones(0)
    adjacent = same & (np.diff(t) < 1.5 * it) & (np.diff(t) > 0)
    P = vs.P[order]
    y = vs.y[order]
    ok = adjacent & (P[1:] > 0) & (P[:-1] > 0)
    i0 = np.flatnonzero(ok)
    i1 = i0 + 1
    z0 = y[i0] / P[i0]
    z1 = y[i1] / P[i1]
    theta = np.angle(y[i0] + y[i1])
    par = np.real((z1 - z0) * np.exp(-1j * theta))
    ratio = par ** 2 / (1.0 / P[i0] + 1.0 / P[i1])
    spw_pair = vs.spw[order][i0]
    out: Dict[int, Dict[str, Any]] = {}
    for sp in np.unique(vs.spw):
        sel = spw_pair == sp
        n = int(sel.sum())
        if n > 0:
            s = float(ratio[sel].mean())
            out[int(sp)] = {"s": s, "sigma": s * math.sqrt(2.0 / n), "n_pairs": n,
                            "method": "adjacent integrations, component parallel to the pair mean"}
        else:
            out[int(sp)] = {"s": 1.0, "sigma": None, "n_pairs": 0,
                            "method": "assumed (no adjacent integrations)"}
    return out


def _s_rows(vs: VisSummary, s_map: Dict[int, Dict[str, Any]]) -> np.ndarray:
    if not vs.n_rows:
        return np.zeros(0)
    lut = np.ones(int(vs.spw.max()) + 1)
    for k, v in s_map.items():
        if 0 <= int(k) < lut.size:
            lut[int(k)] = float(v["s"])
    return lut[vs.spw]


def snr_rates(vs: VisSummary, s_map: Dict[int, Dict[str, Any]], geom: Geometry,
              combine_spw: bool = False) -> Dict[str, Any]:
    """M2: solution-SNR rate ``r_i`` per antenna, ``SNR_i(T) = r_i sqrt(T)``.

    Spec: 10.2 item 1, C.2 (``r_i^2 = sum_j sum w |M_ij|^2 / (s dt_int)``), P18 3.1-3.2 (M2).

    ``r_i^2`` = information ``sum P/s`` of antenna i's rows divided by the
    time it has data. With ``combine_spw=False`` (one solution per spw) the
    rate of the solve unit is per spw; the record gives both.
    """
    info = vs.P / _s_rows(vs, s_map)
    n_ant = vs.n_ant
    spws = [int(s) for s in np.unique(vs.spw)]
    r_spw = np.full((len(spws), n_ant), np.nan)
    tot_info = np.zeros(n_ant)
    tot_time = np.zeros(n_ant)
    for k, sp in enumerate(spws):
        sel = vs.spw == sp
        inf_a = (np.bincount(vs.a1[sel], info[sel], n_ant) + np.bincount(vs.a2[sel], info[sel], n_ant))
        tm = np.zeros(n_ant)
        for o in np.unique(vs.obs[sel]):
            so_ = sel & (vs.obs == o)
            g = geom.tgrid[so_]
            present = np.zeros((n_ant, int(g.max()) + 1), bool)
            present[vs.a1[so_], g] = True
            present[vs.a2[so_], g] = True
            tm += present.sum(axis=1) * geom.int_time[int(o)]
        with np.errstate(invalid="ignore", divide="ignore"):
            r_spw[k] = np.where(tm > 0, np.sqrt(inf_a / np.where(tm > 0, tm, 1.0)), np.nan)
        tot_info += inf_a
        tot_time = np.maximum(tot_time, tm)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_all = np.where(tot_time > 0, np.sqrt(tot_info / np.where(tot_time > 0, tot_time, 1.0)),
                         np.nan)
    unit = r_all if combine_spw or len(spws) == 1 else np.nanmedian(r_spw, axis=0)
    good = np.isfinite(unit)
    return {
        "per_antenna": {vs.antenna_names[a]: float(r_all[a]) for a in range(n_ant)
                        if np.isfinite(r_all[a])},
        "per_spw_median": {int(sp): float(np.nanmedian(r_spw[k])) for k, sp in enumerate(spws)
                           if np.isfinite(r_spw[k]).any()},
        "solve_unit": "all spws" if combine_spw or len(spws) == 1 else "one spw",
        "unit_per_antenna": unit,
        "median": float(np.median(unit[good])) if good.any() else float("nan"),
        "p10": float(np.percentile(unit[good], 10)) if good.any() else float("nan"),
    }


def _lag_bins(it_ref: float, max_lag_s: float) -> np.ndarray:
    """Lag-bin edges (s): single lags 1..8 IT, then x1.25 (geometric)."""
    edges = [(m + 0.5) * it_ref for m in range(0, 9)]
    while edges[-1] < max_lag_s + it_ref:
        edges.append(edges[-1] * 1.25)
    return np.asarray(edges)


def lag_structure_function(vs: VisSummary, s_map: Dict[int, Dict[str, Any]], geom: Geometry,
                           batch: int = 256) -> Dict[str, Any]:
    """M3: ``D_eff(tau) = -ln C(tau)`` from the model-weighted lag correlation.

    Spec: 10.2 item 1, C.2 (``C(tau) = sum P Re[z(t+tau) z*(t)] / sum P (|z|^2 - 2s/P)``),
    P18 3.2 (M3) and test results 2 (agrees with the flux-weighted pair
    structure function within 0.018 on TM2).

    Pairs of integrations of the same baseline and spw in one observation,
    weighted by ``sqrt(P_t P_{t+tau})``; the numerator is a product of
    noises at different times (unbiased), the denominator subtracts the noise
    power ``2s/P`` (needs ``s``). Computed with FFTs on the integration grid
    of each observation; lags are binned (single lags up to 8 IT, then
    geometric). The uncertainty is an antenna delete-one jackknife.
    """
    n_ant = vs.n_ant
    max_span = max(geom.obs_span.values()) if geom.obs_span else geom.it_ref
    edges = _lag_bins(geom.it_ref, max_span)
    nb = edges.size - 1
    num_ant = np.zeros((n_ant, nb))
    den_ant = np.zeros((n_ant, nb))
    num = np.zeros(nb)
    den = np.zeros(nb)
    tau_w = np.zeros(nb)
    npairs = np.zeros(nb)
    s_rows = _s_rows(vs, s_map)
    order = _sort_order(vs)
    for o in np.unique(vs.obs):
        sel = order[vs.obs[order] == o]
        if sel.size == 0:
            continue
        it = geom.int_time[int(o)]
        G = int(geom.tgrid[sel].max()) + 1
        if G < 2:
            continue
        nfft = 1 << int(math.ceil(math.log2(2 * G)))
        lag_s = np.arange(1, G) * it
        bin_of_lag = np.searchsorted(edges, lag_s) - 1
        valid_lag = (bin_of_lag >= 0) & (bin_of_lag < nb)
        onehot = np.zeros((G - 1, nb))
        onehot[np.flatnonzero(valid_lag), bin_of_lag[valid_lag]] = 1.0
        key = np.stack([vs.spw[sel], vs.a1[sel], vs.a2[sel]])
        change = np.flatnonzero(np.any(key[:, 1:] != key[:, :-1], axis=0)) + 1
        starts = np.concatenate([[0], change])
        stops = np.concatenate([change, [sel.size]])
        for b0 in range(0, starts.size, batch):
            st = starts[b0:b0 + batch]
            sp = stops[b0:b0 + batch]
            ns = st.size
            U = np.zeros((ns, nfft), dtype=complex)
            A = np.zeros((ns, nfft))
            B = np.zeros((ns, nfft))
            ant1 = np.empty(ns, dtype=np.int64)
            ant2 = np.empty(ns, dtype=np.int64)
            rows = sel[st[0]:sp[-1]]
            ser = np.repeat(np.arange(ns), sp - st)
            okp = vs.P[rows] > 0
            rows, ser = rows[okp], ser[okp]
            g = geom.tgrid[rows]
            sq = np.sqrt(vs.P[rows])
            u = vs.y[rows] / sq
            U[ser, g] = u
            A[ser, g] = (np.abs(u) ** 2 - 2.0 * s_rows[rows]) / sq
            B[ser, g] = sq
            ant1[:] = vs.a1[sel[st]]
            ant2[:] = vs.a2[sel[st]]
            FU = np.fft.fft(U, axis=1)
            nu = np.fft.ifft(np.abs(FU) ** 2, axis=1).real[:, 1:G]
            FA = np.fft.rfft(A, axis=1)
            FB = np.fft.rfft(B, axis=1)
            dn = 0.5 * (np.fft.irfft(np.conj(FA) * FB, n=nfft, axis=1)[:, 1:G]
                        + np.fft.irfft(np.conj(FB) * FA, n=nfft, axis=1)[:, 1:G])
            cnt = np.fft.irfft(np.abs(np.fft.rfft((B > 0).astype(float), axis=1)) ** 2,
                               n=nfft, axis=1)[:, 1:G]
            for arr, tot, per in ((nu, num, num_ant), (dn, den, den_ant)):
                binned = arr @ onehot
                tot += binned.sum(axis=0)
                np.add.at(per, ant1, binned)
                np.add.at(per, ant2, binned)
            tau_w += (dn.sum(axis=0) * lag_s) @ onehot
            npairs += np.rint(cnt.sum(axis=0)) @ onehot
    present = np.flatnonzero((np.bincount(vs.a1, minlength=n_ant)
                              + np.bincount(vs.a2, minlength=n_ant)) > 0)
    keep = (den > 0) & (npairs > 0)
    tau = np.where(keep, tau_w / np.where(den > 0, den, 1.0), 0.5 * (edges[:-1] + edges[1:]))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(keep, num / np.where(den > 0, den, 1.0), np.nan)
        n = present.size
        if n > 1:
            Cj = (num[None, :] - num_ant[present]) / (den[None, :] - den_ant[present])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sigC = np.sqrt((n - 1) / n * np.nansum((Cj - np.nanmean(Cj, axis=0)) ** 2,
                                                       axis=0))
        else:
            sigC = np.full(nb, np.nan)
        D = np.where(C > 0, -np.log(np.where(C > 0, C, 1.0)), np.nan)
        sigD = np.where(C > 0, sigC / np.where(C > 0, C, 1.0), np.nan)
    keep &= np.isfinite(C)
    return {
        "tau_s": tau[keep].tolist(),
        "C": C[keep].tolist(),
        "D": D[keep].tolist(),
        "sigma_D": sigD[keep].tolist(),
        "n_pairs": npairs[keep].astype(int).tolist(),
        "estimator": ("C(tau) = sum sqrt(P_t P_t+tau) Re[z(t+tau) z*(t)] / "
                      "sum sqrt(P_t P_t+tau) (|z|^2 - 2s/P); D_eff = -ln C; "
                      "sigma from an antenna delete-one jackknife"),
    }


def fit_structure_function(tau, D, sigma_D) -> Dict[str, Any]:
    """Power-law fit ``D_eff(tau) = (tau/tau0)^beta``.

    Spec: C.2 (``D(tau) = (tau/tau0)^beta``), P18 3.1 and test results 2
    (beta at the upper limit 2 for 2016; no measurable structure for TM1).

    Weighted least squares in D (all lags with finite sigma), amplitude
    linear for each beta on a grid over :data:`BETA_RANGE`. Without at least
    two lags or with an amplitude below :data:`FIT_MIN_SIGNIFICANCE` sigma
    the structure is "not significant" and the atmospheric term is zero.
    """
    tau = np.asarray(tau, dtype=float)
    D = np.asarray(D, dtype=float)
    sg = np.asarray(sigma_D, dtype=float)
    ok = np.isfinite(tau) & np.isfinite(D) & np.isfinite(sg) & (sg > 0) & (tau > 0)
    if ok.sum() < 2:
        return {"significant": False, "tau0_s": None, "beta": None, "amplitude": 0.0,
                "reason": "fewer than two lags with a measured D_eff"}
    tau, D, sg = tau[ok], D[ok], sg[ok]
    best = None
    for beta in np.arange(BETA_RANGE[0], BETA_RANGE[1] + 1e-9, 0.01):
        x = tau ** beta
        wx = x / sg ** 2
        amp = float(np.sum(wx * D) / np.sum(wx * x))
        chi2 = float(np.sum(((D - amp * x) / sg) ** 2))
        sig_amp = 1.0 / math.sqrt(float(np.sum(x * x / sg ** 2)))
        if best is None or chi2 < best[0]:
            best = (chi2, float(beta), amp, sig_amp)
    chi2, beta, amp, sig_amp = best
    significant = amp > FIT_MIN_SIGNIFICANCE * sig_amp
    out = {"significant": bool(significant), "beta": beta, "amplitude": amp,
           "amplitude_sigma": sig_amp, "chi2": chi2, "n_lags": int(tau.size),
           "tau0_s": float(amp ** (-1.0 / beta)) if amp > 0 else None,
           "beta_at_limit": bool(beta <= BETA_RANGE[0] + 1e-9 or beta >= BETA_RANGE[1] - 1e-9)}
    out["reason"] = ("power law fitted" if significant else
                     f"amplitude {amp:.3g} <= {FIT_MIN_SIGNIFICANCE:g} sigma ({sig_amp:.3g}): no "
                     f"significant phase structure; the atmospheric term is zero")
    return out


def structure_model(fit: Dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """``D(tau)`` of a fit (zero when the fit is not significant)."""
    if not fit.get("significant"):
        return lambda lag: np.zeros_like(np.asarray(lag, dtype=float))
    amp = float(fit["amplitude"])
    beta = float(fit["beta"])
    return lambda lag: amp * np.abs(np.asarray(lag, dtype=float)) ** beta


# ============================================================================
# Candidates and solution units
# ============================================================================

def gaincal_solint_string(n_int: int, it: float, scan_nint_max: int) -> str:
    """gaincal ``solint`` that makes chunks of exactly ``n_int`` integrations.

    Spec: 10 item 5 (solints in seconds, integer multiples of the time bin),
    10.2 item 3. gaincal bins the data of a scan on a fixed time grid from its
    first integration, ``floor((t - t_first)/solint)`` (CASA 6.5.1 and 6.5.6,
    checked with ``tests/test_casa_solint.py``). ``solint = n IT - delta``
    with ``delta = IT/(4M)`` (``M`` chunks in the longest scan) keeps every
    grid edge at least ``delta`` after an integration and ``3 IT/4`` before the
    next, so timestamp jitter cannot move an integration across an edge;
    ``n = 1`` is ``'int'``.
    """
    n = int(n_int)
    if n <= 1:
        return "int"
    m = max(1, int(math.ceil(max(int(scan_nint_max), n) / n)))
    return f"{n * it - it / (4.0 * m):.4f}s"


def candidate_solints(scan_nint_median: float, it_ref: float, scans_per_obs: int = 2,
                      eb_span_s: Optional[float] = None,
                      scan_nint_max: Optional[int] = None) -> List[Dict[str, Any]]:
    """Candidate solution intervals, shortest first.

    Spec: 10.2 item 3 (divisors of the median scan length in integrations as
    integer multiples of the integration time, the scan and the EB;
    non-divisor multiples where neighbouring divisors are more than a factor
    2 apart; no "N x IT" schedule), 10 item 5 (seconds, integer multiples of
    the time bin).

    Each candidate: ``label``, ``kind`` ("int" | "scan" | "eb"), ``n_int``
    (integrations per solution for "int"), ``T_s`` (nominal length) and the
    gaincal arguments ``gaincal_solint`` / ``time_combine``. For "int",
    ``gaincal_solint`` (:func:`gaincal_solint_string`) makes gaincal's chunks
    hold exactly ``n`` integrations, whatever the timestamp jitter.
    """
    n_scan = int(round(scan_nint_median))
    n_max = int(scan_nint_max) if scan_nint_max else n_scan
    out: List[Dict[str, Any]] = []
    if n_scan >= 2:
        divs = [d for d in range(1, n_scan) if n_scan % d == 0]
        grid = sorted(set(divs + [n_scan]))
        changed = True
        while changed:
            changed = False
            for lo, hi in zip(grid[:-1], grid[1:]):
                if hi / lo > CANDIDATE_MAX_RATIO + 1e-9:
                    mid = int(round(math.sqrt(lo * hi)))
                    mid = min(max(mid, lo + 1), hi - 1)
                    grid = sorted(set(grid + [mid]))
                    changed = True
                    break
        for d in grid:
            if d >= n_scan:
                continue
            out.append({"label": f"{d * it_ref:.4g}s", "kind": "int", "n_int": int(d),
                        "divisor": bool(n_scan % d == 0), "T_s": float(d * it_ref),
                        "gaincal_solint": gaincal_solint_string(d, it_ref, n_max),
                        "time_combine": ""})
    if scans_per_obs > 1:
        out.append({"label": "scan", "kind": "scan", "n_int": n_scan, "divisor": True,
                    "T_s": float(max(n_scan, 1) * it_ref), "gaincal_solint": "inf",
                    "time_combine": ""})
    eb_T = float(eb_span_s) if eb_span_s else float(max(n_scan, 1) * it_ref * max(scans_per_obs, 1))
    out.append({"label": "EB", "kind": "eb", "n_int": None, "divisor": True, "T_s": eb_T,
                "gaincal_solint": "inf", "time_combine": "scan"})
    return out


@dataclass
class Units:
    """Solution intervals of one candidate: row -> unit, and per-unit time and group."""
    of_row: np.ndarray        # unit index of every row (-1: none)
    time: np.ndarray          # solution time per unit (mean of its integration times)
    obs: np.ndarray           # observation per unit
    group: np.ndarray         # spw per unit, or -1 when spws are combined
    n_times: np.ndarray       # integrations per unit


def solution_units(vs: VisSummary, geom: Geometry, cand: Dict[str, Any], combine_spw: bool,
                   rows: Optional[np.ndarray] = None) -> Units:
    """Assign rows to the solution intervals gaincal makes for a candidate.

    Spec: 10.2 item 3 (fractional intervals at scan ends), P18 3.3 (gaincal
    re-partitions at scan boundaries), 10.2 item 14 (table and data times
    matched by nearest).

    "eb": one interval per observation (``combine='scan'``); "scan": one
    per scan; "int": gaincal's fixed time grid inside each scan,
    ``floor((t - t_first)/solint)`` (``'int'``: every integration).
    The solution time is the mean of the integration times of the unit
    (that of the rows in ``rows`` when given, e.g. the training half).
    """
    key = (cand["kind"], cand.get("n_int"), cand.get("gaincal_solint"), bool(combine_spw),
           vs.n_rows)
    base = geom.cache.get(key)
    if base is None:
        base = _unit_assignment(vs, cand, combine_spw)
        geom.cache[key] = base
    of_row, ukey, pair_unit, pair_time, pair_row, n_times = base
    nu = ukey.shape[1]
    if rows is None:
        sel = np.ones(pair_unit.size, bool)
    else:
        sel = np.asarray(rows, bool)[pair_row]
    t_sum = np.bincount(pair_unit[sel], pair_time[sel], minlength=nu)
    t_cnt = np.bincount(pair_unit[sel], minlength=nu).astype(float)
    with np.errstate(invalid="ignore"):
        utime = np.where(t_cnt > 0, t_sum / np.where(t_cnt > 0, t_cnt, 1.0), np.nan)
    return Units(of_row=of_row, time=utime, obs=ukey[0], group=ukey[1], n_times=n_times)


def _unit_assignment(vs: VisSummary, cand: Dict[str, Any], combine_spw: bool):
    """Row -> unit of a candidate and the (unit, integration) pairs (cached per run)."""
    group = vs.spw.copy() if not combine_spw else np.full(vs.n_rows, -1, dtype=np.int64)
    kind = cand["kind"]
    if kind == "eb":
        sub = np.zeros(vs.n_rows, dtype=np.int64)
        scan_key = np.zeros(vs.n_rows, dtype=np.int64)
    elif kind == "scan":
        sub = np.zeros(vs.n_rows, dtype=np.int64)
        scan_key = vs.scan
    elif kind == "int":
        scan_key = vs.scan
        sub = np.zeros(vs.n_rows, dtype=np.int64)
        per_int = cand["gaincal_solint"] == "int"
        solint = None if per_int else float(cand["gaincal_solint"].rstrip("s"))
        for o in np.unique(vs.obs):
            for sc in np.unique(vs.scan[vs.obs == o]):
                sel = (vs.obs == o) & (vs.scan == sc)
                st = np.unique(vs.time[sel])
                if per_int:
                    chunk = np.arange(st.size, dtype=np.int64)
                else:
                    chunk = np.floor((st - st[0]) / solint).astype(np.int64)
                sub[sel] = chunk[np.searchsorted(st, vs.time[sel])]
    else:
        raise ValueError(f"unknown candidate kind {kind!r}")
    cols = [vs.obs, group, scan_key, sub]
    codes = np.zeros(vs.n_rows, dtype=np.int64)
    for c in cols:
        _, inv = np.unique(c, return_inverse=True)
        inv = np.asarray(inv).ravel()
        codes = codes * (int(inv.max()) + 1 if inv.size else 1) + inv
    ucode, first_u, of_row = np.unique(codes, return_index=True, return_inverse=True)
    of_row = np.asarray(of_row).ravel()
    ukey = np.stack([c[first_u] for c in cols])
    utime, tinv = np.unique(vs.time, return_inverse=True)
    tinv = np.asarray(tinv).ravel()
    pcode = of_row.astype(np.int64) * max(utime.size, 1) + tinv
    _, first = np.unique(pcode, return_index=True)
    pair_unit = of_row[first]
    pair_time = vs.time[first]
    n_times = np.bincount(pair_unit, minlength=ucode.size).astype(np.int64)
    return of_row, ukey, pair_unit, pair_time, first, n_times


def unit_snr(vs: VisSummary, units: Units, s_rows: np.ndarray, rows: Optional[np.ndarray] = None,
             P: Optional[np.ndarray] = None) -> np.ndarray:
    """Predicted true SNR per (unit, antenna): ``sqrt(sum P/s)`` of the antenna's rows.

    Spec: 10.2 item 4 (G1 per antenna and per solution on predicted values), C.2.
    """
    P = vs.P if P is None else P
    use = np.ones(vs.n_rows, bool) if rows is None else np.asarray(rows, bool)
    info = np.where(use, P / s_rows, 0.0)
    nu = units.time.size
    acc = np.zeros(nu * vs.n_ant)
    np.add.at(acc, units.of_row * vs.n_ant + vs.a1, info)
    np.add.at(acc, units.of_row * vs.n_ant + vs.a2, info)
    return np.sqrt(acc.reshape(nu, vs.n_ant))


def g1_gate(snr: np.ndarray, threshold: float = G1_SNR_MIN) -> Dict[str, Any]:
    """G1: the candidate passes when the typical (median) solution's predicted SNR >= 5.

    Spec: 10.2 item 4 (gate on the predicted true SNR r_i sqrt(T), per
    antenna and per solution; antennas and solutions below it keep the
    previous stage with applymode 'calonly'), P18 test results 2 (the
    candidate floor (5/r_median)^2) and 7 (item 2).

    ``snr``: (n_units, n_ant) predicted SNR, 0 where the antenna has no data
    in the unit. Returns the candidate decision, the per-(unit, antenna)
    pass mask ``good`` and the antennas whose median solution fails.
    """
    present = snr > 0
    vals = snr[present]
    med = float(np.median(vals)) if vals.size else 0.0
    good = present & (snr >= threshold)
    ant_med = np.array([np.median(snr[present[:, a], a]) if present[:, a].any() else np.nan
                        for a in range(snr.shape[1])])
    return {
        "pass": bool(med >= threshold),
        "snr_median": med,
        "snr_p10": float(np.percentile(vals, 10)) if vals.size else 0.0,
        "solution_pass_fraction": float(good.sum() / max(int(present.sum()), 1)),
        "good": good,
        "antenna_median": ant_med,
        "failing_antennas": [int(a) for a in np.flatnonzero(np.isfinite(ant_med)
                                                            & (ant_med < threshold))],
    }


# ============================================================================
# Interpolation (applycal linearPD, perobs) and kappa
# ============================================================================

def _row_brackets(units: Units, row_time: np.ndarray, row_obs: np.ndarray,
                  row_group: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bracketing units (lo, hi) and the fraction f of every row's time.

    Spec: 10 item 3 ('perobs': never across an observation), P18 test results 5
    (constant outside the first/last solution).
    """
    n = row_time.size
    lo = np.full(n, -1, dtype=np.int64)
    hi = np.full(n, -1, dtype=np.int64)
    f = np.zeros(n)
    for o, g in set(zip(units.obs.tolist(), units.group.tolist())):
        uidx = np.flatnonzero((units.obs == o) & (units.group == g) & np.isfinite(units.time))
        if uidx.size == 0:
            continue
        uidx = uidx[np.argsort(units.time[uidx])]
        ut = units.time[uidx]
        sel = np.flatnonzero((row_obs == o) & (row_group == g))
        t = row_time[sel]
        k = np.searchsorted(ut, t, side="right") - 1
        before = k < 0
        after = k >= ut.size - 1
        kk = np.clip(k, 0, ut.size - 1)
        kh = np.clip(k + 1, 0, ut.size - 1)
        frac = np.zeros(t.size)
        mid = ~before & ~after
        frac[mid] = (t[mid] - ut[kk[mid]]) / (ut[kh[mid]] - ut[kk[mid]])
        lo[sel] = np.where(before, uidx[0], uidx[kk])
        hi[sel] = np.where(before | after, np.where(before, uidx[0], uidx[kk]), uidx[kh])
        f[sel] = frac
    return lo, hi, f


def interpolation_weights(eval_times, sol_times) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lo, hi, f)`` of linear interpolation between sorted solution times.

    Spec: 10.2 item 5 (actual times and interpolation coefficients), P18 3.1.
    Constant before the first and after the last solution.
    """
    st = np.asarray(sol_times, dtype=float)
    t = np.asarray(eval_times, dtype=float)
    k = np.searchsorted(st, t, side="right") - 1
    lo = np.clip(k, 0, st.size - 1)
    hi = np.clip(k + 1, 0, st.size - 1)
    f = np.zeros(t.size)
    mid = (k >= 0) & (k < st.size - 1)
    f[mid] = (t[mid] - st[lo[mid]]) / (st[hi[mid]] - st[lo[mid]])
    hi = np.where(mid, hi, lo)
    return lo, hi, f


def kappa_geometry(eval_times, sol_times) -> float:
    """M4: ``kappa`` = mean over the evaluated integrations of ``sum_k c_k(t)^2``.

    Spec: 10.2 item 1, C.2 (2/3 for a uniform interior grid with linear
    interpolation; 1 for T = IT or one solution per EB; 0.95 for one per
    scan with long gaps; 0.54-0.60 for the even/odd halves of G2).
    """
    lo, hi, f = interpolation_weights(eval_times, sol_times)
    return float(np.mean((1 - f) ** 2 + f ** 2)) if f.size else float("nan")


def _row_values(values: np.ndarray, good: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                f: np.ndarray, ant: np.ndarray, phase: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolated per-antenna value at each row and whether the antenna is corrected.

    Spec: P18 test results 5 (flagged solutions are not interpolated: the
    open interval between the previous and the next good solution gets no
    correction; constant outside the table).
    """
    has = lo >= 0
    l = np.where(has, lo, 0)
    h = np.where(has, hi, 0)
    g_lo = good[l, ant] & has
    g_hi = good[h, ant] & has
    ok = g_lo & ((f == 0) | g_hi)
    v_lo = values[l, ant]
    v_hi = values[h, ant]
    if phase:
        d = np.angle(np.exp(1j * (v_hi - v_lo)))
        val = v_lo + f * d
    else:
        val = (1 - f) ** 2 * v_lo + f ** 2 * v_hi
    return np.where(ok, val, 0.0), ok


# ============================================================================
# Discrete prediction of V(T) (Spec 10.2 item 5, C.2)
# ============================================================================

def _stride_sample(x: np.ndarray, nmax: int) -> np.ndarray:
    if x.size <= nmax:
        return x
    step = int(math.ceil(x.size / nmax))
    return x[::step]


def _interval_variance(eval_t: np.ndarray, members: List[np.ndarray], sol_t: np.ndarray,
                       dfun: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Atmospheric residual variance at ``eval_t`` after linear interpolation of interval means.

    Spec: C.2 (variogram identity ``Var(sum a phi) = -1/2 sum a_m a_n D(t_m - t_n)``
    for weights summing to zero; ``a = 1/((beta+1)(beta+2))`` for nearest).
    """
    order = np.argsort(sol_t)
    sol_t = sol_t[order]
    members = [members[k] for k in order]
    lo, hi, f = interpolation_weights(eval_t, sol_t)
    mem = [_stride_sample(m, _MAX_MEMBERS_PER_UNIT) for m in members]
    nu = len(mem)
    m_k = np.zeros((eval_t.size, nu))
    for k in range(nu):
        m_k[:, k] = dfun(eval_t[:, None] - mem[k][None, :]).mean(axis=1)
    need = set()
    for a, b in zip(lo.tolist(), hi.tolist()):
        need.add((a, a))
        need.add((b, b))
        need.add((min(a, b), max(a, b)))
    S = {}
    for a, b in need:
        S[(a, b)] = float(dfun(mem[a][:, None] - mem[b][None, :]).mean())
    s_ll = np.array([S[(a, a)] for a in lo])
    s_hh = np.array([S[(b, b)] for b in hi])
    s_lh = np.array([S[(min(a, b), max(a, b))] for a, b in zip(lo, hi)])
    idx = np.arange(eval_t.size)
    return ((1 - f) * m_k[idx, lo] + f * m_k[idx, hi]
            - 0.5 * ((1 - f) ** 2 * s_ll + f ** 2 * s_hh + 2 * f * (1 - f) * s_lh))


def predict_residual_variance(vs: VisSummary, geom: Geometry, cand: Dict[str, Any],
                              dfun: Callable[[np.ndarray], np.ndarray], r_unit: float,
                              combine_spw: bool) -> Dict[str, float]:
    """``V_pred(T)`` of a candidate on the actual times: atmospheric + thermal terms.

    Spec: 10.2 item 5, C.2 (``V(T) = a (T/tau0)^beta + kappa/(r^2 T)`` with the
    thermal term from the CASA loss table below SNR 5), P18 3.1 (discrete
    prediction on the candidate grid).

    For the typical antenna (rate ``r_unit``): the atmospheric term uses the
    fitted ``D_eff`` with the variogram identity for linear interpolation of
    interval means; the thermal term is ``mean_t sum_k c_k(t)^2 L(r sqrt(T_k))``.
    Observations are weighted by their number of integrations.
    """
    units = solution_units(vs, geom, cand, combine_spw=True)
    _, _, pair_unit, pair_time, _, _ = geom.cache[(cand["kind"], cand.get("n_int"),
                                                   cand.get("gaincal_solint"), True, vs.n_rows)]
    cut = np.searchsorted(pair_unit, np.arange(units.time.size + 1))
    atm_sum = th_sum = k_sum = w_sum = 0.0
    for o in np.unique(units.obs):
        uidx = np.flatnonzero(units.obs == o)
        it = geom.int_time[int(o)]
        times = geom.utimes[int(o)]
        members = [pair_time[cut[u]:cut[u + 1]] for u in uidx]
        sol_t = units.time[uidx]
        ev = _stride_sample(times, _MAX_EVAL_TIMES)
        atm = _interval_variance(ev, members, sol_t, dfun)
        order = np.argsort(sol_t)
        lo, hi, f = interpolation_weights(ev, sol_t[order])
        snr = r_unit * np.sqrt(np.array([m.size * it for m in members]))[order]
        L = casa_loss(snr)
        th = (1 - f) ** 2 * L[lo] + f ** 2 * L[hi]
        w = float(times.size)
        atm_sum += w * float(np.mean(atm))
        th_sum += w * float(np.mean(th))
        k_sum += w * float(np.mean((1 - f) ** 2 + f ** 2))
        w_sum += w
    return {"V_atm": atm_sum / w_sum, "V_th": th_sum / w_sum,
            "V_pred": (atm_sum + th_sum) / w_sum, "kappa_full": k_sum / w_sum}


# ============================================================================
# G2: phase solves in numpy and the held-out coherence (Spec 10.2 items 6, 14; C.3)
# ============================================================================

def solve_phases(Y: np.ndarray, Q: np.ndarray, refant: Optional[int] = None,
                 minblperant: int = 4, n_iter: int = 200, tol: float = 1e-10
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Phase-only antenna solutions maximising ``Re sum_ij Y_ij e^{-i(phi_i - phi_j)}``.

    Spec: C.1 (``phi_i <- arg sum_j sum_k w V M* e^{i phi_j}``), 10.2 item 14.

    ``Y``: (n_units, n_ant, n_ant) Hermitian sums ``sum w V M*`` (row i,
    column j = baseline i-j); ``Q``: the same shape, ``sum w |M|^2`` (0 where
    there is no data). Antennas with fewer than ``minblperant`` baselines get
    no solution. Phases are referred to ``refant`` (or the antenna with the
    most baselines). Returns ``(phases, valid)``.
    """
    Y = np.array(Y, dtype=complex)
    nb = (Q > 0).sum(axis=2)
    valid = nb >= minblperant
    for _ in range(3):
        mask = valid[:, :, None] & valid[:, None, :]
        nb = ((Q > 0) & mask).sum(axis=2)
        valid = valid & (nb >= minblperant)
    mask = valid[:, :, None] & valid[:, None, :]
    Y = np.where(mask, Y, 0.0)
    g = np.where(valid, 1.0 + 0j, 0.0)
    for it in range(n_iter):
        h = np.einsum("uij,uj->ui", Y, g)
        mag = np.abs(h)
        new = np.where(mag > 0, h / np.where(mag > 0, mag, 1.0), 0.0)
        if it % 2 == 1:
            avg = g + new
            am = np.abs(avg)
            new = np.where(am > 0, avg / np.where(am > 0, am, 1.0), new)
        delta = np.max(np.abs(new - g)) if new.size else 0.0
        g = new
        if delta < tol:
            break
    ph = np.angle(g)
    nu = Y.shape[0]
    for u in range(nu):
        if not valid[u].any():
            continue
        ref = refant if (refant is not None and valid[u, refant]) else int(np.argmax(nb[u]))
        ph[u] = np.where(valid[u], np.angle(np.exp(1j * (ph[u] - ph[u, ref]))), 0.0)
    return ph, valid


def _solve_units(vs: VisSummary, units: Units, y: np.ndarray, P: np.ndarray, rows: np.ndarray,
                 refant: Optional[int], minblperant: int, batch: int = 256
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Solve every unit on the selected rows; returns (phases, valid) of shape (n_units, n_ant)."""
    n_ant = vs.n_ant
    nu = units.time.size
    ph = np.zeros((nu, n_ant))
    valid = np.zeros((nu, n_ant), bool)
    sel = np.flatnonzero(rows & (units.of_row >= 0))
    u_of = units.of_row[sel]
    for u0 in range(0, nu, batch):
        u1 = min(nu, u0 + batch)
        inb = (u_of >= u0) & (u_of < u1)
        r = sel[inb]
        uu = u_of[inb] - u0
        Ym = np.zeros((u1 - u0, n_ant, n_ant), dtype=complex)
        Qm = np.zeros((u1 - u0, n_ant, n_ant))
        np.add.at(Ym, (uu, vs.a1[r], vs.a2[r]), y[r])
        np.add.at(Ym, (uu, vs.a2[r], vs.a1[r]), np.conj(y[r]))
        np.add.at(Qm, (uu, vs.a1[r], vs.a2[r]), P[r])
        np.add.at(Qm, (uu, vs.a2[r], vs.a1[r]), P[r])
        p, v = solve_phases(Ym, Qm, refant=refant, minblperant=minblperant)
        ph[u0:u1] = p
        valid[u0:u1] = v
    return ph, valid


def apply_units(vs: VisSummary, units: Units, phases: np.ndarray, good: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row baseline phase ``psi_1 - psi_2`` of interpolated solutions and its mask.

    Spec: 10.2 items 4 and 14 (calonly: a row whose antennas lack a good
    solution passes uncorrected; table and data times by nearest).
    """
    lo, hi, f = _row_brackets(units, vs.time, vs.obs,
                              units.group[np.maximum(units.of_row, 0)]
                              if units.of_row.size else np.zeros(0, np.int64))
    p1, ok1 = _row_values(phases, good, lo, hi, f, vs.a1, phase=True)
    p2, ok2 = _row_values(phases, good, lo, hi, f, vs.a2, phase=True)
    ok = ok1 & ok2
    return np.where(ok, p1 - p2, 0.0), ok


def thermal_rows(vs: VisSummary, units: Units, snr: np.ndarray, good: np.ndarray,
                 rows: np.ndarray) -> float:
    """P-weighted mean over ``rows`` of the solution-noise loss ``(v_1 + v_2)/2``.

    Spec: C.3 (rule A: ``kappa L(SNR)``), C.2 (loss table), 10.2 item 7.
    ``v_a(t) = sum_k c_k(t)^2 L(SNR_ak)`` of the interpolated solutions.
    """
    L = casa_loss(np.where(snr > 0, snr, np.nan))
    L = np.where(np.isfinite(L), L, 0.0)
    lo, hi, f = _row_brackets(units, vs.time, vs.obs,
                              units.group[np.maximum(units.of_row, 0)])
    v1, ok1 = _row_values(L, good, lo, hi, f, vs.a1, phase=False)
    v2, ok2 = _row_values(L, good, lo, hi, f, vs.a2, phase=False)
    ok = ok1 & ok2 & rows
    w = np.where(rows, vs.P, 0.0)
    tot = w.sum()
    return float(np.sum(np.where(ok, 0.5 * (v1 + v2), 0.0) * w) / tot) if tot > 0 else 0.0


@dataclass
class Split:
    """One G2 split: the rows that train and the rows held out, and their data."""
    name: str
    train: np.ndarray
    held: np.ndarray
    half_train: Optional[int] = None   # band split: index of the half used for training
    half_held: Optional[int] = None


def make_splits(vs: VisSummary, geom: Geometry, mode: str) -> List[Split]:
    """G2 splits: even/odd integrations inside each scan, or the band halves.

    Spec: 10.2 item 6 (even/odd within scans; band halves when a scan has fewer
    than 4 integrations), C.3 (roles swapped and averaged).
    """
    if mode == "time":
        even = (geom.k_in_scan % 2) == 0
        return [Split("train even / hold odd", even, ~even),
                Split("train odd / hold even", ~even, even)]
    if mode == "band":
        if vs.y_half is None or vs.P_half is None:
            raise ValueError("band split needs y_half/P_half")
        allr = np.ones(vs.n_rows, bool)
        return [Split("train lower / hold upper half", allr, allr, 0, 1),
                Split("train upper / hold lower half", allr, allr, 1, 0)]
    raise ValueError(f"unknown split mode {mode!r}")


def split_mode(geom: Geometry, have_halves: bool) -> str:
    """Even/odd unless the median scan has fewer than 4 integrations (then band halves).

    Spec: 10.2 item 6.
    """
    if geom.scan_nint_median < BAND_SPLIT_BELOW_NINT and have_halves:
        return "band"
    return "time"


def _data(vs: VisSummary, half: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """``(y_0, P)``: state-0 data against this state's model, on a band half or the full band."""
    if half is None:
        return vs.y0(), vs.P
    return vs.y0(half), vs.P_half[half]


@dataclass
class StageSolve:
    """An accepted stage as the G2 reference chain re-solves it (Spec 10.2 item 7)."""
    stage: int
    cand: Dict[str, Any]
    state: VisSummary            # the state the stage was solved on (its model)
    s_rows: np.ndarray
    corr: float                  # loss-table correction of this stage (rule A term)


def _train_solution(vs: VisSummary, geom: Geometry, cand: Dict[str, Any], combine_spw: bool,
                    split: Split, base_phase: np.ndarray, s_rows: np.ndarray,
                    refant: Optional[int], minblperant: int, gate: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Solve ``cand`` on the training rows of ``split`` on top of ``base_phase``.

    Returns the per-row phase ``psi_1 - psi_2`` (all rows) and its mask.
    The gate is G1 on the full-data predicted SNR (the same as production).
    """
    units_tr = solution_units(vs, geom, cand, combine_spw, rows=split.train)
    y0, P = _data(vs, split.half_train)
    y_tr = y0 * np.exp(-1j * base_phase)
    ph, valid = _solve_units(vs, units_tr, y_tr, P, split.train, refant, minblperant)
    good = valid
    if gate:
        snr_full = unit_snr(vs, units_tr, s_rows)
        good = good & (snr_full >= G1_SNR_MIN)
    return apply_units(vs, units_tr, ph, good)


def _reference_phase(chain: List[StageSolve], geom: Geometry, combine_spw: bool, split: Split,
                     refant: Optional[int], minblperant: int) -> np.ndarray:
    """Cumulative row phase of the accepted stages re-solved on the training half.

    Spec: 10.2 item 7 (each stage re-solved with the model it was solved
    with in production, on the training half).
    """
    base = np.zeros(chain[0].state.n_rows) if chain else None
    for st in chain:
        dphi, ok = _train_solution(st.state, geom, st.cand, combine_spw, split, base, st.s_rows,
                                   refant, minblperant)
        base = base + dphi
    return base


def _held_sums(vs: VisSummary, split: Split, phase: np.ndarray) -> Dict[str, Any]:
    y0, P = _data(vs, split.half_held)
    yc = np.where(split.held, y0 * np.exp(-1j * phase), 0.0)
    w = np.where(split.held, P, 0.0)
    n_ant = vs.n_ant
    N = yc.sum()
    D = w.sum()
    Na = (np.bincount(vs.a1, yc.real, n_ant) + np.bincount(vs.a2, yc.real, n_ant)
          + 1j * (np.bincount(vs.a1, yc.imag, n_ant) + np.bincount(vs.a2, yc.imag, n_ant)))
    Da = np.bincount(vs.a1, w, n_ant) + np.bincount(vs.a2, w, n_ant)
    return {"N": N, "D": D, "Na": Na, "Da": Da}


def _coherence(sums: List[Dict[str, Any]], drop: Optional[int] = None) -> float:
    num = 0.0
    den = 0.0
    for s in sums:
        if drop is None:
            num += abs(s["N"])
            den += s["D"]
        else:
            num += abs(s["N"] - s["Na"][drop])
            den += s["D"] - s["Da"][drop]
    return num / den if den > 0 else float("nan")


def reference_bases(chain: List[StageSolve], geom: Geometry, splits: List[Split],
                    combine_spw: bool, refant: Optional[int] = None, minblperant: int = 4
                    ) -> List[np.ndarray]:
    """Row phase of the reference state for every split (zeros before any stage).

    Spec: 10.2 item 7.
    """
    out = []
    for sp in splits:
        base = _reference_phase(chain, geom, combine_spw, sp, refant, minblperant)
        out.append(base if base is not None else np.zeros(sp.train.size))
    return out


def holdout_coherence(vs: VisSummary, geom: Geometry, cand: Optional[Dict[str, Any]],
                      bases: List[np.ndarray], splits: List[Split], s_rows: np.ndarray,
                      combine_spw: bool, refant: Optional[int] = None, minblperant: int = 4,
                      gate: bool = True) -> Dict[str, Any]:
    """G2: held-out coherence of a candidate against the re-solved reference state.

    Spec: 10.2 item 6 (``C_held = |sum w V_corr M*| / sum w |M|^2`` with an
    antenna delete-one paired jackknife), item 7 (reference re-solved on the
    training half, :func:`reference_bases`), C.3 (roles swapped and
    averaged), 2.4 (never evaluate on the data the quantity was solved from).

    ``vs`` is the current state (its model is the model of the candidate
    solve and of the evaluation); ``bases[k]`` the reference row phase of
    split ``k``. ``cand=None`` evaluates only the reference. Returns
    ``V_held`` of the candidate and ``V_ref`` of the reference, the raw
    difference ``dV_raw = V_held - V_ref`` (negative = better) and its paired
    jackknife ``sigma``. ``gate=False`` switches G1 off (tests of G2 alone).
    """
    present = np.flatnonzero((np.bincount(vs.a1, minlength=vs.n_ant)
                              + np.bincount(vs.a2, minlength=vs.n_ant)) > 0)
    rsums = []
    csums = []
    for sp, base in zip(splits, bases):
        rsums.append(_held_sums(vs, sp, base))
        if cand is not None:
            dphi, ok = _train_solution(vs, geom, cand, combine_spw, sp, base, s_rows, refant,
                                       minblperant, gate=gate)
            csums.append(_held_sums(vs, sp, base + dphi))
    v_ref = -math.log(_coherence(rsums))
    out: Dict[str, Any] = {"V_ref": v_ref, "C_ref": math.exp(-v_ref)}
    if cand is None:
        return out
    v_c = -math.log(_coherence(csums))
    dv = np.array([-math.log(_coherence(csums, a)) + math.log(_coherence(rsums, a))
                   for a in present])
    n = dv.size
    sigma = float(math.sqrt((n - 1) / n * np.sum((dv - dv.mean()) ** 2))) if n > 1 else float("nan")
    out.update({"V_held": v_c, "C_held": math.exp(-v_c), "dV_raw": v_c - v_ref, "sigma": sigma,
                "n_jackknife": int(n)})
    return out


def loss_correction(vs: VisSummary, geom: Geometry, cand: Dict[str, Any], splits: List[Split],
                    s_rows: np.ndarray, combine_spw: bool) -> Dict[str, float]:
    """Rule-A term ``kappa_held L(SNR_half) - kappa_full L(SNR_full)`` of a candidate.

    Spec: C.3 (``V_full = V_held - kappa_held L(SNR_1/2) + kappa_full L(SNR_full)``),
    10.2 items 6-7, P18 test results 4 (CASA loss table).

    Both terms are P-weighted means of the solution-noise loss with the
    actual interpolation geometry: the full-data solutions on all rows, and
    the training-half solutions (SNR of the training rows) on the held rows,
    averaged over the splits. Solutions failing G1 (full-data predicted SNR)
    carry no correction in either.
    """
    units = solution_units(vs, geom, cand, combine_spw)
    snr_full = unit_snr(vs, units, s_rows)
    good_full = snr_full >= G1_SNR_MIN
    full = thermal_rows(vs, units, snr_full, good_full, np.ones(vs.n_rows, bool))
    held_vals = []
    held_w = []
    kap_h = []
    for sp in splits:
        units_tr = solution_units(vs, geom, cand, combine_spw, rows=sp.train)
        P_tr = vs.P if sp.half_train is None else vs.P_half[sp.half_train]
        snr_half = unit_snr(vs, units_tr, s_rows, rows=sp.train, P=P_tr)
        good = unit_snr(vs, units_tr, s_rows) >= G1_SNR_MIN
        held_vals.append(thermal_rows(vs, units_tr, snr_half, good, sp.held))
        held_w.append(float(np.sum(np.where(sp.held, vs.P, 0.0))))
        for o in np.unique(vs.obs):
            uo = (units_tr.obs == o) & np.isfinite(units_tr.time)
            if not uo.any():
                continue
            st = np.unique(units_tr.time[uo])
            ev = np.unique(vs.time[(vs.obs == o) & sp.held])
            kap_h.append(kappa_geometry(ev, st))
    held = float(np.average(held_vals, weights=held_w)) if sum(held_w) > 0 else 0.0
    return {"thermal_full": full, "thermal_held": held, "corr": held - full,
            "kappa_held": float(np.mean(kap_h)) if kap_h else float("nan")}


# ============================================================================
# Rules A/B, choice of T and acceptance (Spec 10.2 items 6-8, C.3)
# ============================================================================

def corrected_improvement(dV_raw: float, corr_cand: float, T_cand: float,
                          T_prev: Optional[float], corr_prev: float = 0.0) -> Dict[str, Any]:
    """Loss-table corrected held-out improvement with the conditional two-sided correction.

    Spec: 10.2 item 7, C.3 (``L(T_cand) - L(T_ref)`` when ``T_cand <= T_prev``,
    ``L(T_cand)`` when longer), P18 decision and test results 10.

    ``improvement = -dV_raw + corr_cand - (corr_prev if T_cand <= T_prev)``.
    Stage 1 (no previous stage, ``T_prev=None``) corrects the candidate only.
    """
    two_sided = T_prev is not None and T_cand <= T_prev + 1e-9
    imp = -dV_raw + corr_cand - (corr_prev if two_sided else 0.0)
    return {"improvement_corr": float(imp), "two_sided": bool(two_sided),
            "correction_cand": float(corr_cand),
            "correction_ref": float(corr_prev) if two_sided else 0.0}


def rule_a(evaluated: List[Dict[str, Any]]) -> Optional[int]:
    """Rule A: the candidate with the largest corrected improvement (smallest V_full).

    Spec: 10.2 item 6 (rule A), C.3. ``evaluated`` holds dicts with ``index``
    and ``improvement_corr``.
    """
    ok = [e for e in evaluated if e.get("improvement_corr") is not None]
    if not ok:
        return None
    return int(max(ok, key=lambda e: e["improvement_corr"])["index"])


def rule_b(evaluated: List[Dict[str, Any]], passing: Sequence[int]) -> Optional[int]:
    """Rule B: one candidate shorter than the held-out argmin (no correction).

    Spec: 10.2 item 6 (rule B), C.3, P18 test results 4. ``passing`` are the
    indices (shortest first) of the candidates that pass G1; when the argmin
    is the shortest of them, rule B returns the argmin itself.
    """
    ok = [e for e in evaluated if e.get("V_held") is not None]
    if not ok:
        return None
    arg = int(min(ok, key=lambda e: e["V_held"])["index"])
    order = list(passing)
    if arg not in order:
        return arg
    pos = order.index(arg)
    return int(order[pos - 1]) if pos > 0 else arg


def choose_solint(a: Optional[int], b: Optional[int], T_of: Callable[[int], float]
                  ) -> Tuple[Optional[int], str]:
    """The longer of the rule-A and rule-B answers when they disagree.

    Spec: 10.2 item 6 ("一致しなければ長い方を選ぶ"), C.3 (the safe side
    against fitting noise).
    """
    if a is None and b is None:
        return None, "no candidate was evaluated"
    if a is None or b is None:
        k = a if a is not None else b
        return k, "only one rule gave an answer"
    if a == b:
        return a, "rules A and B agree"
    k = a if T_of(a) >= T_of(b) else b
    return k, "rules A and B disagree; the longer interval is taken"


def acceptance(stage: int, improvement_corr: float, sigma: float) -> Dict[str, Any]:
    """Apply the stage or stop.

    Spec: 10.2 item 2 (stage 1 is skipped only when the corrected improvement
    is below -2 sigma), item 7 (later stages need > max(1 sigma, 0.001); the
    floor 0.001 is below a tenth of the flux-ratio precision), P18 test
    results 10 (2016 it3: +0.00072 stops at the floor).
    """
    sig = float(sigma) if np.isfinite(sigma) else 0.0
    if stage == 1:
        thr = -STAGE1_SKIP_SIGMA * sig
        ok = improvement_corr >= thr
        reason = (f"stage 1 (one solution per EB): corrected held-out improvement "
                  f"{improvement_corr:+.5f} {'>=' if ok else '<'} -{STAGE1_SKIP_SIGMA:g} sigma "
                  f"({thr:+.5f}); {'applied' if ok else 'skipped (it makes the held-out data worse)'}")
        return {"accept": bool(ok), "threshold": thr, "reason": reason}
    thr = max(ACCEPT_SIGMA * sig, ACCEPT_FLOOR)
    ok = improvement_corr > thr
    which = "1 sigma" if ACCEPT_SIGMA * sig >= ACCEPT_FLOOR else "the floor 0.001"
    reason = (f"corrected held-out improvement {improvement_corr:+.5f} "
              f"{'>' if ok else '<='} max(1 sigma = {sig:.5f}, 0.001) [{which}]; "
              f"{'accepted' if ok else 'stop'}")
    return {"accept": bool(ok), "threshold": thr, "reason": reason}


def combine_spw_decision(snr_per_spw: float, snr_combined: float,
                         threshold: float = G1_SNR_MIN) -> Dict[str, Any]:
    """Whether to add 'spw' to combine for the chosen solint (recorded; not applied here).

    Spec: 10.2 item 11 (combine only when the per-spw SNR(T) < 5 and the
    combined SNR(T) >= 5), P18 3.6.
    """
    combine = snr_per_spw < threshold <= snr_combined
    return {"combine_spw": bool(combine), "snr_per_spw": float(snr_per_spw),
            "snr_combined": float(snr_combined),
            "reason": ("per-spw SNR below the G1 gate and the combined SNR above it"
                       if combine else "no gain from combining spws at the G1 gate")}


def spwmap_for_combined_spw(spws_per_obs: Dict[int, Sequence[int]], nspw: int) -> List[int]:
    """applycal spwmap for a ``combine='spw'`` table: each spw -> lowest spw of its observation.

    Spec: 10.2 item 11 (spw combination), ASEBI Spec 9 item 3. gaincal labels
    a combined solution with the lowest spw of its interval, and an interval
    never spans observations. A spw that would need two targets raises.
    """
    m = list(range(int(nspw)))
    owner: Dict[int, int] = {}
    for obs in sorted(spws_per_obs):
        spws = sorted(int(x) for x in spws_per_obs[obs])
        if not spws:
            continue
        lo = spws[0]
        for sp in spws:
            if sp in owner and owner[sp] != lo:
                raise ValueError(f"spw {sp} would map to both {owner[sp]} and {lo}")
            owner[sp] = lo
            m[sp] = lo
    return m


def g3_short_baseline_ratio(vs: VisSummary, fraction: float = 0.2) -> Dict[int, Dict[str, float]]:
    """G3 quantity: data/model flux ratio on the shortest 20 % of the baselines, per EB.

    Spec: 10.2 item 10 (compared before and after a stage; > max(3 sigma, 1 %)
    rejects an amplitude stage and is recorded for phase stages), P24.
    """
    out: Dict[int, Dict[str, float]] = {}
    if vs.uvdist_m is None:
        return out
    for o in np.unique(vs.obs):
        sel = (vs.obs == o) & (vs.P > 0)
        if not sel.any():
            continue
        lim = np.percentile(vs.uvdist_m[sel], 100 * fraction)
        rows = sel & (vs.uvdist_m <= lim)
        Psum = float(vs.P[rows].sum())
        if Psum <= 0:
            continue
        ratio = float(np.real(vs.y[rows].sum()) / Psum)
        out[int(o)] = {"ratio": ratio, "sigma": math.sqrt(Psum) / Psum,
                       "uvdist_limit_m": float(lim)}
    return out


def g3_compare(before: Dict[int, Dict[str, float]], after: Dict[int, Dict[str, float]],
               s: float = 1.0) -> Dict[int, Dict[str, Any]]:
    """G3: change of the short-baseline ratio, flagged when > max(3 sigma, 1 %).

    Spec: 10.2 item 10.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for o in sorted(set(before) & set(after)):
        b, a = before[o], after[o]
        sig = math.sqrt(s) * math.hypot(b["sigma"], a["sigma"])
        diff = a["ratio"] - b["ratio"]
        rel = diff / b["ratio"] if b["ratio"] else float("nan")
        lim = max(3 * sig / abs(b["ratio"]) if b["ratio"] else float("inf"), 0.01)
        out[o] = {"before": b["ratio"], "after": a["ratio"], "relative_change": rel,
                  "limit": lim, "exceeds": bool(abs(rel) > lim)}
    return out


# ============================================================================
# One stage (Spec 10.2 items 1-8)
# ============================================================================

def _neighbours(passing: List[int], k: int) -> List[int]:
    if k not in passing:
        return [k]
    p = passing.index(k)
    return passing[max(p - 1, 0):p + 2]


def evaluate_stage(stage: int, state: VisSummary, chain: List[StageSolve], geom: Geometry, *,
                   combine_spw: bool = False, refant: Optional[int] = None, minblperant: int = 4,
                   split: Optional[str] = None) -> Dict[str, Any]:
    """Measure, gate, predict, evaluate (G2) and decide one phase stage.

    Spec: 10.2 items 1-8 and 14 (and the record listed after item 14), C.2,
    C.3; P18 decision; P23 (measurement and production self-cal use the same
    rules).

    ``state``: the current data (DATA of the last accepted stage, MODEL of
    its image) with ``phi0`` tracked to state 0. ``chain``: the accepted
    stages (for the reference state re-solved on the training half). Stage 1
    evaluates one solution per EB only. Returns a JSON-able record with the
    measured quantities, every candidate (G1, prediction, G2), rules A/B,
    the chosen candidate and the decision with its reason; the internal
    objects needed by the caller are under ``"_"``.
    """
    rec: Dict[str, Any] = {"stage": int(stage)}
    s_map = noise_scale(state, geom)
    s_rows = _s_rows(state, s_map)
    rates = snr_rates(state, s_map, geom, combine_spw)
    lag = lag_structure_function(state, s_map, geom)
    fit = fit_structure_function(lag["tau_s"], lag["D"], lag["sigma_D"])
    dfun = structure_model(fit)
    mode = split or split_mode(geom, state.y_half is not None)
    splits = make_splits(state, geom, mode)
    r_med = rates["median"]
    rec["measured"] = {
        "s": {str(k): v for k, v in s_map.items()},
        "r": {k: v for k, v in rates.items() if k != "unit_per_antenna"},
        "D_eff": lag,
        "D_fit": fit,
        "int_time_s": {str(k): v for k, v in geom.int_time.items()},
        "it_ref_s": geom.it_ref,
        "scan_nint_median": geom.scan_nint_median,
        "n_scans": geom.n_scans,
        "g2_split": mode,
        "combine_spw": bool(combine_spw),
    }
    if fit.get("significant") and np.isfinite(r_med) and r_med > 0:
        a_lin = atmospheric_coefficient(fit["beta"], "linear")
        rec["measured"]["T_star_closed_s"] = optimal_solint(fit["tau0_s"], fit["beta"], r_med, a=a_lin)
        rec["measured"]["a_linear"] = a_lin
    else:
        rec["measured"]["T_star_closed_s"] = None

    eb_span = sum(geom.obs_span.values())
    cands = candidate_solints(geom.scan_nint_median, geom.it_ref, geom.max_scans_per_obs, eb_span,
                              int(geom.scan_nint.max()) if geom.scan_nint.size else None)
    if stage == 1:
        cands = [c for c in cands if c["kind"] == "eb"]
    for i, c in enumerate(cands):
        c["index"] = i
        units = solution_units(state, geom, c, combine_spw)
        g1 = g1_gate(unit_snr(state, units, s_rows))
        c["g1"] = {k: v for k, v in g1.items() if k not in ("good", "antenna_median")}
        c["g1"]["failing_antennas"] = [state.antenna_names[a] for a in g1["failing_antennas"]]
        c["g1"]["antenna_snr_median"] = {state.antenna_names[a]: round(float(v), 3)
                                         for a, v in enumerate(g1["antenna_median"])
                                         if np.isfinite(v)}
        c["n_solutions"] = int(units.time.size)
        if np.isfinite(r_med) and r_med > 0:
            c.update(predict_residual_variance(state, geom, c, dfun, r_med, combine_spw))
        c["evaluated"] = False
    passing = [c["index"] for c in cands if c["g1"]["pass"]]
    rec["candidates"] = cands
    T_prev = chain[-1].cand["T_s"] if chain else None
    corr_prev = chain[-1].corr if chain else 0.0
    rec["previous"] = ({"stage": chain[-1].stage, "label": chain[-1].cand["label"], "T_s": T_prev,
                        "correction": corr_prev} if chain else None)
    internal: Dict[str, Any] = {"s_rows": s_rows, "cands": cands}
    rec["_"] = internal

    if not passing:
        rec.update({"T_pred": None, "rule_A": None, "rule_B": None, "chosen": None,
                    "decision": "stop",
                    "reason": (f"G1: no candidate reaches a predicted solution SNR of "
                               f"{G1_SNR_MIN:g} (EB median {cands[-1]['g1']['snr_median']:.2f}); "
                               f"no self-calibration stage")})
        return rec

    if stage == 1:
        to_eval = passing
        k_pred = passing[0]
    else:
        scored = [k for k in passing
                  if not (mode == "time" and cands[k]["kind"] == "int" and cands[k]["n_int"] == 1)]
        with_pred = [k for k in passing if "V_pred" in cands[k]]
        k_pred = (min(with_pred, key=lambda k: cands[k]["V_pred"]) if with_pred else passing[-1])
        eb_k = [c["index"] for c in cands if c["kind"] == "eb" and c["index"] in passing]
        to_eval = sorted(set([k for k in _neighbours(passing, k_pred) if k in scored] + eb_k))
    rec["T_pred"] = cands[k_pred]["label"]

    bases = reference_bases(chain, geom, splits, combine_spw, refant, minblperant)
    results: Dict[int, Dict[str, Any]] = {}

    def run_g2(k: int) -> Dict[str, Any]:
        c = cands[k]
        g2 = holdout_coherence(state, geom, c, bases, splits, s_rows, combine_spw, refant,
                               minblperant)
        lc = loss_correction(state, geom, c, splits, s_rows, combine_spw)
        ci = corrected_improvement(g2["dV_raw"], lc["corr"], c["T_s"], T_prev, corr_prev)
        res = {"index": k, **g2, **lc, **ci}
        c["g2"] = {kk: vv for kk, vv in res.items() if kk != "index"}
        c["evaluated"] = True
        results[k] = res
        return res

    for k in to_eval:
        run_g2(k)
    evald = [dict(results[k], V_held=results[k]["V_held"]) for k in sorted(results)]
    ka = rule_a(evald)
    kb = rule_b(evald, passing)
    if kb is not None and kb not in results:
        if not (mode == "time" and cands[kb]["kind"] == "int" and cands[kb]["n_int"] == 1):
            run_g2(kb)
    kc, why = choose_solint(ka, kb, lambda k: cands[k]["T_s"])
    if kc is not None and kc not in results:
        kc, why = ka, why + "; rule B's interval cannot be scored by the even/odd split, rule A taken"
    rec["rule_A"] = cands[ka]["label"] if ka is not None else None
    rec["rule_B"] = cands[kb]["label"] if kb is not None else None
    rec["chosen"] = cands[kc]["label"] if kc is not None else None
    rec["chosen_reason"] = why
    res = results[kc]
    dec = acceptance(stage, res["improvement_corr"], res["sigma"])
    rec["decision"] = "apply" if dec["accept"] else ("skip" if stage == 1 else "stop")
    rec["reason"] = dec["reason"]
    rec["threshold"] = dec["threshold"]
    internal["chosen"] = cands[kc]
    internal["chosen_corr"] = res["corr"]
    if np.isfinite(r_med) and r_med > 0 and cands[kc]["kind"] in ("int", "scan", "eb"):
        n_spw = max(int(np.unique(state.spw).size), 1)
        snr_c = cands[kc]["g1"]["snr_median"]
        per = snr_c / math.sqrt(n_spw) if combine_spw else snr_c
        comb = snr_c if combine_spw else snr_c * math.sqrt(n_spw)
        rec["spw_combine"] = combine_spw_decision(per, comb) if n_spw > 1 else None
    return rec


def strip_internal(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of a stage record without the internal objects (for justification.json)."""
    out = {k: v for k, v in rec.items() if k != "_"}
    return out


@dataclass
class GateTable:
    """Predicted SNR of every solution of a chosen candidate, for gating a gaincal table."""
    time: np.ndarray        # (n_units,)
    obs: np.ndarray
    group: np.ndarray       # spw, or -1 when spws are combined
    snr: np.ndarray         # (n_units, n_ant)
    n_times: Optional[np.ndarray] = None   # integrations per unit (matching tolerance)
    int_time: Dict[int, float] = field(default_factory=dict)


def gate_table(vs: VisSummary, geom: Geometry, cand: Dict[str, Any], s_rows: np.ndarray,
               combine_spw: bool) -> GateTable:
    """Predicted SNR per solution and antenna of the candidate chosen for production.

    Spec: 10.2 item 4 (G1 per antenna and per solution on predicted values).
    """
    units = solution_units(vs, geom, cand, combine_spw)
    return GateTable(time=units.time, obs=units.obs, group=units.group,
                     snr=unit_snr(vs, units, s_rows), n_times=units.n_times,
                     int_time=dict(geom.int_time))


def caltable_gate(gt: GateTable, cal_time, cal_obs, cal_spw, cal_ant,
                  threshold: float = G1_SNR_MIN) -> Dict[str, Any]:
    """Rows of a gaincal table to flag because their predicted SNR is below G1.

    Spec: 10.2 item 4 (antennas and solutions below the gate keep the previous
    stage: applymode 'calonly'), item 14 (table and data times matched by
    nearest). A table row farther from the nearest predicted solution than
    half its length (at least half an integration) is left alone and counted
    as ``unmatched``.
    """
    cal_time = np.asarray(cal_time, dtype=float)
    cal_obs = np.asarray(cal_obs, dtype=np.int64)
    cal_spw = np.asarray(cal_spw, dtype=np.int64)
    cal_ant = np.asarray(cal_ant, dtype=np.int64)
    flag = np.zeros(cal_time.size, bool)
    matched = np.zeros(cal_time.size, bool)
    combined = bool(np.all(gt.group == -1))
    grp = np.full(cal_time.size, -1, np.int64) if combined else cal_spw
    for o, g in set(zip(cal_obs.tolist(), grp.tolist())):
        rows = np.flatnonzero((cal_obs == o) & (grp == g))
        cand = np.flatnonzero((gt.obs == o) & (gt.group == g) & np.isfinite(gt.time))
        if cand.size == 0:
            continue
        cand = cand[np.argsort(gt.time[cand])]
        ut = gt.time[cand]
        k = np.clip(np.searchsorted(ut, cal_time[rows]), 0, ut.size - 1)
        km = np.clip(k - 1, 0, ut.size - 1)
        pick = np.where(np.abs(ut[km] - cal_time[rows]) < np.abs(ut[k] - cal_time[rows]), km, k)
        j = cand[pick]
        it = gt.int_time.get(int(o), 1.0)
        nt = gt.n_times[j] if gt.n_times is not None else np.ones(j.size)
        tol = 0.5 * it * np.maximum(nt, 1)
        ant = cal_ant[rows]
        ok = (np.abs(gt.time[j] - cal_time[rows]) <= tol) & (ant >= 0) & (ant < gt.snr.shape[1])
        matched[rows] = ok
        snr = gt.snr[j, np.clip(ant, 0, gt.snr.shape[1] - 1)]
        flag[rows] = ok & (snr < threshold)
    return {"flag": flag, "n_flagged": int(flag.sum()), "n_rows": int(cal_time.size),
            "unmatched": int((~matched).sum())}
