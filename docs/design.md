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
  number of iterations, the same gain tables, and the same final image.
- Every parameter choice is recorded with its rationale in a structured
  `justification.json` file, so the reduction is auditable in publications.
- The codebase is small and focused on ALMA continuum imaging, so it is
  easy to read, test, and extend.

The intended use cases are:

1. **Reproducible production self-calibration** for users who want a
   deterministic, well-documented pipeline and an audit trail.
2. **Cross-validation of other self-cal results**, since AJISAI's fixed
   schedule and explicit logging make it a useful reference.
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

### Deterministic

AJISAI runs a fixed self-cal schedule to completion. The default
schedule is three phase-only iterations (`solint = inf`, `6*IT`,
`3*IT`) followed by one amplitude iteration (`solint = inf`), where
`IT` is the average integration time of the MS. The schedule never
shortens or extends based on intermediate image quality. After all
iterations complete, AJISAI selects the best one by dynamic range
(peak / RMS_offsource).

AJISAI deliberately does not use adaptive rollback, in which each
iteration's outcome is evaluated and the iteration is discarded if the
noise gets worse. Adaptive rollback can produce better images for
marginal datasets, but it makes the behaviour data-dependent at run time
— the same config can produce different numbers of iterations on
different MSes, which complicates publication-grade reproducibility.
AJISAI chooses determinism over adaptivity.

Anomalies (iterations where, e.g., the dynamic range drops by more
than 50%, or gaincal returns no solutions) are still detected, but
they are *logged* into the justification file rather than altering the
pipeline flow.

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
 8. multiscale scales: 0, b, 3b, ... up to the largest that fits the mask
 9. make initial CLEAN (iteration 0) to 3 sigma_th; no model is saved
10. for each iteration in the schedule:
    - copy the previous stage's MS to a temporary MS
    - predict the previous model into it if needed (ft, Taylor terms)
    - gaincal -> caltable; applycal (calonly, calwt=False) on the copy
    - split out corrected data -> selfcal_<N>.ms; delete the copy
    - tclean of selfcal_<N>.ms to 3 sigma_th; predict its model into it
    - compute peak/RMS/SNR/DR on the tt0 image
    - detect anomalies (log only)
    (every tclean and prediction is followed by a consistency check)
11. select best iteration by DR
12. split its DATA -> final_selfcal.ms; final image from it, CLEANed
    3 -> 2 -> 1 -> 0.5 sigma_th until the aperture flux converges
13. write summary CSV, PNG, and justification.json
```

## At a glance

| Aspect                          | AJISAI                                  |
|---------------------------------|-----------------------------------------|
| Self-cal schedule               | Fixed (3 phase + 1 amp by default)      |
| Behaviour on anomalies          | Log only; pipeline continues            |
| Number of iterations per run    | Always the same for a given config      |
| Reference antenna selection     | Flag-stats + geometric center (hybrid)  |
| RMS estimator                   | Sigma-clip + center exclusion (default) |
| Best-image criterion            | Dynamic range over all iterations       |
| Decision audit log              | `justification.json` (structured)       |
| Supported targets               | ALMA continuum, single field, multi-EB  |
| Codebase size (LOC)             | ~3000 (focused)                         |
| Primary use                     | Reproducible production + audit         |

Use AJISAI when reproducibility, explainability, or educational
transparency matter more than raw image quality on marginal data.

## Locked-in design choices

These choices were debated during AJISAI's design and are now considered
load-bearing. Changing them would alter AJISAI's identity:

- **All iterations complete; best chosen by DR; anomalies log-only.**
  Phase plateau followed by amp recovery is a common pattern on ALMA;
  rollback would prematurely stop the pipeline in that case.

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

- It is **not** an adaptive optimizer. It does not try to squeeze the
  best possible image out of a marginal dataset at the expense of
  reproducibility.

- It does **not** perform continuum-line separation, flagging, or any
  step that should be done before self-cal. Bring AJISAI a clean,
  continuum-ready MS.
