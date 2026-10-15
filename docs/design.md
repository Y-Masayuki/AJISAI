# Design

This page explains *why* AJISAI exists, the principles that govern its
behavior, and the design choices that define it.

## Why AJISAI

ALMA self-calibration is well understood algorithmically, but writing a
script that reliably performs it on a new dataset is still tedious and
error-prone, and the choices made in such scripts are rarely recorded in
a form that others can check.

AJISAI does not aim for feature breadth. Instead, it is a tool optimized
for **reproducibility and explainability**. Specifically:

- AJISAI's parameter choices are deterministic given the input MS and
  the config. The same MS + same config always produces the same
  stages, the same gain tables, and the same final image.
- Every parameter choice is recorded with its rationale in a structured
  `justification.json` file, so the reduction is auditable in publications.
- The codebase is small and focused on ALMA continuum imaging, so it is
  easy to read, test, and extend.

The intended use cases are:

1. **Reproducible production self-calibration** for users who want a
   deterministic, well-documented pipeline and an audit trail.
2. **Cross-validation of other self-cal results**, since AJISAI's
   recorded decisions and explicit logging make it a useful reference.
3. **Pedagogy.** Reading AJISAI's code and justification log is an
   accessible introduction to the practical art of self-calibration.

## Three design principles

### Fire-and-forget

The minimum required input is the path to a measurement set. Every other
parameter has a sensible default that AJISAI derives from the data
itself: the reference antenna comes from a flag-stats-plus-geometric-
center hybrid; the cell size and image size come from the baseline
distribution and the ALMA primary beam; the noise threshold comes from
the dirty image; and so on. Users who want to override any default can
do so via the config; users who do not need to read about every option
to get started.

### Justification-based

Every parameter choice is recorded in `justification.json` with the
reason. For example, the refant entry contains not only the chosen
antenna but also the strategy, the flag threshold, the candidate list
with positions and flag fractions, and the geometric center. This is
intended to make the reduction reproducible by anyone with the same MS
and the published `justification.json`.

This is the "J" in AJISAI's name. It is the feature that most clearly
distinguishes AJISAI from black-box pipelines.

### Deterministic and data-driven

AJISAI is deterministic: the same MS and the same config always give the
same stages, solution intervals, gain tables and final image. Since v0.3
the self-calibration stages are no longer a fixed ladder; they are decided
from quantities measured in the data and checked on data that the gains
were not solved from (`SelfcalSchedule(mode="data")`, the default). The
rules follow the ASEBI design specification v0.3 (section 10.2 and
appendices C.2-C.3), where they were derived and tested.

**Why not a fixed ladder.** The v0.2 ladder (`inf -> 6*IT -> 3*IT ->
amplitude inf`) ran every step whatever the signal-to-noise ratio of the
solutions, and chose the best iteration by the dynamic range of images made
from the same data the gains were solved from. Gains fitted to noise make
those images look better (a spurious point source of about
`1.6 sqrt(N_ant N_solint)` times the image noise appears even on pure
noise), so the same-data dynamic range rewards the shortest intervals. On a
faint long-baseline execution block, the two short steps of the ladder can
reach a solution SNR of only 1-2 and lose more than half of the coherence on
held-out data while the dynamic range reads them as an improvement. "N x IT"
is also not a physical quantity: the same `6*IT` is 36 s on one EB and 12 s
on another.

**What decides a solution interval.** A phase solution of interval `T`
leaves the atmospheric phase that varies within `T` and adds the thermal
noise of the solution. Per antenna, the residual phase variance is

    V(T) = a (T/tau0)^beta + kappa L(SNR(T)),   SNR_i(T) = r_i sqrt(T)

and the coherence of a long baseline is `exp(-V)`. Every term is measured
from the visibilities, the model and the weights, without any gain
solution or image:

- `s`, the noise scale per spw from adjacent integrations (the weights'
  error for the summed solve unit);
- `r_i`, the solution-SNR rate of each antenna from the model and the
  weights (`r_i^2 = sum_j sum w |M_ij|^2 / (s dt)`), the quantity gaincal
  writes in its SNR column;
- `D_eff(tau) = (tau/tau0)^beta`, the phase structure function from the
  model-weighted lag correlation of the visibilities (its numerator is a
  product of noises at different times, so it is not biased by noise);
- `kappa`, the interpolation geometry of the solutions (2/3 for linear
  interpolation on a uniform interior grid, 1 for one solution per EB) and
  `a(beta, interp)`, the atmospheric coefficient for gaincal's linear
  interpolation (0.34 of the piecewise-constant value at beta = 5/3);
- `L(SNR) = -ln C`, the loss of a CASA phase solution at a true SNR,
  from a table measured with gaincal (2 -> 0.844, 3 -> 0.198, 5 -> 0.0492,
  10 -> 0.0113, 20 -> 0.0027).

The optimum is `T* = [kappa tau0^beta / (a beta r^2)]^(1/(beta+1))`, the
same bias-variance optimisation as the optimal averaging time of mm-VLBI
phase tracking (Blackburn et al. 2019, ApJ 882, 23, Eq. 21; Pesce et al.
2024, ApJ 968, 69, App. G); AJISAI adds the coefficients of gaincal's
interpolation, measures the structure function and the SNR rate directly
from the model and the weights, and checks the choice on held-out
integrations. The curve is flat near the optimum
(`V(2 T*)/V* = (2^beta + beta/2)/(1 + beta)`, 1.25 at beta = 1), so a
coarse grid of candidates suffices.

**The stages.**

1. *Stage 1*: one phase solution per execution block. It is applied unless
   its corrected change on held-out integrations is below -2 sigma, and the
   data are re-imaged so that the model improves.
2. *Candidates*: integer multiples of the integration time that divide the
   median scan length (with non-divisors added where neighbours differ by
   more than a factor 2), the scan, and the EB.
3. *G1 (gate)*: a candidate needs a predicted true solution SNR of at least
   5 for the typical solution; antennas and solutions below 5 are flagged in
   the table and keep the previous stage's correction (`applymode='calonly'`;
   gaincal's own `minsnr` stays at 3, as a floor).
4. *Prediction*: `V(T)` of every candidate on the actual times and the
   interpolation weights.
5. *G2 (held-out choice)*: the predicted best, its neighbours and the EB
   are solved on the even integrations of each scan and evaluated on the odd
   ones, and vice versa (band halves when scans have fewer than four
   integrations): the held-out coherence `|sum w V M*| / sum w |M|^2`, with
   an antenna delete-one paired jackknife. The solves are done in numpy from
   one read of DATA, MODEL_DATA, WEIGHT and FLAG; gaincal runs only for the
   chosen interval. Rule A takes the candidate with the best loss-corrected
   value, rule B the candidate one step shorter than the raw held-out
   optimum; when they disagree the longer interval is taken.
6. *Acceptance*: the stage is applied only when the loss-corrected held-out
   improvement exceeds `max(1 sigma, 0.001)`. From stage 2 on, the
   reference is the state re-solved on the training half with each stage's
   own model, and the correction is two-sided when the candidate is not
   longer than the previous stage's interval (the candidate re-solves the
   previous stage's noise) and one-sided otherwise. The floor 0.001 (0.1 %
   coherence) is below a tenth of the flux-ratio precision a combining
   pipeline needs, so stages that cannot matter are not added.
7. *At most three phase stages*, each re-imaged; the run stops at the first
   stage that is not accepted, and the **best stage is the last accepted
   one**. Peak, RMS and dynamic range are recorded as diagnostics only.

Every measured quantity, candidate, gate result, rule answer, decision and
reason is written to `justification.json` (`stages[k]["selection"]`). A
stage that is not applied is a result, recorded with its reason.

The amplitude stage of the specification (only when the antenna amplitude
scatter times the EB SNR exceeds 2, the EB SNR is at least 10 and G2 and the
short-baseline flux check G3 pass; `'ap'`, one solution per EB, normalised
per observation) is not part of this version: `SelfcalSchedule.amplitude`
accepts only `"never"`.

**The fixed ladder** of v0.2 is kept as `SelfcalSchedule(mode="fixed")`:
every step runs, and the best iteration is chosen by `quality_metric`
(dynamic range). Anomalies (e.g. a dynamic range drop of more than 50 %, or
gaincal returning no solutions) are logged but do not change the flow in
either mode.

## Anatomy of a run

```
AJISAI(cfg).run() pipeline order:

 1. validate inputs
 2. set up working directory
 3. inspect MS (read-only)
    - choose refant (hybrid strategy)
    - average integration time
    - on-source time, median frequency
    - resolve target field; frequency range and correlations
    - relabel J2000 frames to ICRS on a copy, if needed (CASA hygiene)
 4. compute imaging parameters (or take cell/imsize from ImagingConfig)
    - cell = 1/(5 u_max), u_max from the longest unflagged baseline
      at the highest frequency (legacy: beam from 90-percentile baseline / cellpix)
    - imsize covers down to 20% primary-beam attenuation of the
      smallest dish in ANTENNA/DISH_DIAMETER
 5. make dirty image (niter=0)
    - measure dirty-image peak, off-source RMS and beam
    - verify actual vs predicted beam (uvtaper check)
 6. apply phase shift if requested (off by default)
 7. measure sigma_th on the (XX-YY)/2 dirty image (signal-free noise)
 8. multiscale scales: 0, b, 3b, ... up to the largest that fits the mask;
    then only those whose scale-PSF peaks at the component's position
 9. make initial CLEAN (iteration 0) to 3 sigma_th; no model is saved;
    stop with the reason if it leaves an empty model
10. self-calibration stages (schedule.mode="data", at most 3):
    - read DATA, MODEL, WEIGHT, FLAG of the current stage once
      (a temporary copy with the model predicted, for stage 1)
    - measure s, r_i, D_eff(tau), kappa; candidates; G1; predict V(T)
    - G2 on held-out integrations (numpy); rules A/B; accept or stop
      (stage 1: one solution per EB, skipped only below -2 sigma)
    - if accepted: copy the previous stage's MS to a temporary MS;
      gaincal of the chosen interval; flag the solutions below G1;
      applycal (calonly, calwt=False, linearPDperobs) on the copy;
      split corrected data -> selfcal_<N>.ms; delete the copy
    - tclean of selfcal_<N>.ms to 3 sigma_th; predict its model into it;
      peak/RMS/DR as diagnostics
    (schedule.mode="fixed": every step of schedule.steps, as in v0.2)
    (every tclean and prediction is followed by a consistency check)
11. best stage = the last accepted one (mode="fixed": by DR)
12. split its DATA -> final_selfcal.ms; final image from it, CLEANed
    3 -> 2 -> 1 -> 0.5 sigma_th until the aperture flux converges
13. write summary CSV, PNG, and justification.json
```

## At a glance

| Aspect                          | AJISAI                                  |
|---------------------------------|-----------------------------------------|
| Self-cal stages                 | From the data: EB, then up to 2 phase stages with solints chosen on held-out data (fixed ladder as an option) |
| Behaviour on anomalies          | Log only; pipeline continues            |
| Stages per run                  | Always the same for a given MS and config |
| Reference antenna selection     | Flag-stats + geometric center (hybrid)  |
| RMS estimator                   | Sigma-clip + center exclusion (default) |
| Best-stage criterion            | Last accepted stage (DR is a diagnostic) |
| Decision audit log              | `justification.json` (structured)       |
| Supported targets               | ALMA continuum, single field, multi-EB  |
| Codebase size (LOC)             | ~5000 (focused)                         |
| Primary use                     | Reproducible production + audit         |

Use AJISAI when reproducibility, explainability, or educational
transparency matter more than raw image quality on marginal data.

## Locked-in design choices

These choices were debated during AJISAI's design and are now considered
load-bearing. Changing them would alter AJISAI's identity:

- **Decisions on held-out data, never on the data the gains were solved
  from.** Same-data metrics (peak SNR, dynamic range) always favour the
  shortest solution interval; they are recorded as diagnostics only.
  Anomalies stay log-only.

- **Sigma-clipping with center exclusion for off-source RMS.** Eliminates
  the user-supplied `target_radius` of the legacy AJISAI script while
  agreeing with the legacy annulus method to within 1%.

- **Hybrid reference-antenna selection.** Flag-fraction filter + nearest
  to XY geometric center. Robust on combined / re-imaged MS files where
  pipeline-derived refant logs are not available.

- **Pixel-accurate phase center via `imstat` max pixel.** Sub-pixel
  Gaussian fits introduce failure modes for faint or resolved sources;
  pixel accuracy is sufficient because the first phase self-cal absorbs
  the residual sub-pixel offset into per-antenna phases.

- **English everywhere in code and docs.** AJISAI is aimed at worldwide
  ALMA users.

## What AJISAI is not

- It is **not** a general radio-astronomy imaging package. It targets
  ALMA continuum imaging.

- It is **not** a free-form optimizer. Its stages follow fixed, recorded
  rules applied to measured quantities, and it does not search for the
  image that looks best.

- It does **not** perform continuum-line separation, flagging, or any
  step that should be done before self-cal. Bring AJISAI a clean,
  continuum-ready MS.
