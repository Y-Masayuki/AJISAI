# Changelog

All notable changes to AJISAI are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Development version `0.3.0.dev0`. The changes below make AJISAI usable as the
self-calibration engine of a larger pipeline (ASEBI), decide the phase
self-calibration stages from the data, and fix data-hygiene issues; defaults
stay usable for standalone runs, and the previous rules remain available as
explicit options (`SelfcalSchedule(mode="fixed")`, `cell_rule="percentile"`,
`threshold_mode="image_rms"`, `final_depth=()`).

### Added

- **Data-driven phase self-calibration (`SelfcalSchedule(mode="data")`, the
  new default).** The stages and their solution intervals follow the ASEBI
  specification v0.3 (section 10.2, appendices C.2-C.3), implemented in the
  new pure-numpy module `ajisai.solint`:
  - measured from DATA, MODEL_DATA, WEIGHT and FLAG (read once per stage,
    no gain solution and no image): the noise scale `s` per spw from adjacent
    integrations, the solution-SNR rate `r_i` of every antenna
    (`SNR_i(T) = r_i sqrt(T)`), the phase structure function `D_eff(tau)`
    from the model-weighted lag correlation (power-law fit), and the
    interpolation geometry `kappa`;
  - stage 1 is one solution per execution block, applied unless its
    loss-corrected held-out change is below -2 sigma;
  - later stages: candidate intervals (divisors of the median scan length in
    integrations, with non-divisors where neighbours differ by more than a
    factor 2, the scan, the EB), G1 (predicted true SNR >= 5 for the typical
    solution; solutions below 5 are flagged in the gain table so those
    antennas keep the previous stage with `applymode="calonly"`), the
    predicted residual phase variance `V(T)` on the actual times with the
    coefficients of gaincal's linear interpolation and the CASA loss table
    (`-ln C`: SNR 2 -> 0.844, ..., 20 -> 0.0027), and G2: the predicted
    best, its neighbours and the EB solved in numpy on the even (odd)
    integrations of each scan and evaluated on the odd (even) ones (band
    halves when scans have fewer than four integrations), with an antenna
    delete-one paired jackknife; rule A (loss-corrected argmin) and rule B
    (one candidate shorter than the held-out argmin), the longer when they
    disagree;
  - acceptance when the corrected held-out improvement exceeds
    `max(1 sigma, 0.001)`, against the accepted stages re-solved on the
    training half with their own models, with the conditional two-sided
    correction; at most three phase stages; the run stops at the first stage
    that is not accepted; gaincal runs only for the chosen interval, with
    `solint = n x IT - IT/(4M)` (gaincal bins each scan on a fixed time grid
    from its first integration; the small offset keeps the grid edges clear of
    the integrations, so every chunk holds exactly `n`);
  - the best stage is the last accepted one; peak, RMS and dynamic range are
    diagnostics only;
  - every measured quantity, candidate, gate, rule answer, decision and
    reason is recorded in `justification.json["stages"][k]["selection"]`,
    with a summary in `justification.json["selfcal"]`; stages that were
    evaluated but not applied appear with `status: "rejected"`.
  The amplitude stage of the specification is not included yet
  (`SelfcalSchedule.amplitude` accepts only `"never"`).
- `ajisai.selfcal_measurement(vis, results_dir=..., ...)`: the measurement
  self-calibration of one execution block for a calling pipeline (ASEBI),
  with the same rules and no amplitude stage; `AJISAI.selfcal_summary()`
  returns the stage decisions, tables, stage MSs and the best stage's image
  and model.
- `ajisai.ms_utils.read_vis_summary`: one read-only pass over DATA,
  MODEL_DATA, WEIGHT and FLAG reduced per row (`sum w V M*`, `sum w |M|^2`).
- `ajisai.FIXED_SCHEDULE_STEPS`: the v0.2 ladder.
- **Image geometry and Taylor terms from the caller.** `ImagingConfig` gains
  `cell`, `imsize`, `reffreq` and `scales`; when given they are used for
  every `tclean` of the run (dirty image, initial CLEAN, each iteration,
  final image). `nterms > 1` selects `deconvolver="mtmfs"` (multiscale scales
  kept), and `reffreq` is passed to every `tclean`.
- **Final deep image.** After the best iteration is selected, its data are
  split (DATA only) to `<results_dir>/final_selfcal.ms` and imaged once more
  with a deeper CLEAN: the threshold steps through
  `ImagingConfig.final_depth` × sigma_th (default 3, 2, 1, 0.5), continuing
  the same CLEAN without a new PSF. After each step, inside the aperture
  (the CLEAN mask; `final_aperture_arcsec` or 5 beams otherwise),
  `F_corr = F_model + sum(residual)/Omega_R` is measured and the CLEAN stops
  when `|F_unc| <= max(3 sigma_F, 0.01 F_corr)` or when `F_corr` changed by
  no more than that width since the previous step. The restored-image flux
  bias `F_unc (Omega_R/Omega_clean - 1)` is recorded and warned about when it
  exceeds the same width. `final.image` / `final.fits` now come from this
  image (Taylor term 0 for mtmfs); `final_depth=()` restores the previous
  behaviour. Steps, fluxes and the stop reason are in
  `justification.json["final_image"]`.
- **Per-stage solve information.** `AJISAI.stages` (and
  `justification.json["stages"]`) records for every stage the data MS, the
  stage MS, the image, the model and, for self-cal stages, `solve`: gain
  table, solint (string and seconds), combine, refant, calmode, gaintype,
  minsnr, minblperant, solnorm, the applycal interp/applymode/calwt/spwmap,
  the model image(s) the solve used (with nterms and reference frequency),
  the MS solved on, the stage MS, and the flagged-solution fraction per
  observation and spw (`ajisai.SOLVE_INFO_KEYS`).
- **Post-tclean and post-prediction checks.** After every `tclean` the MFS
  frequency axis must be finite and cover the data's frequency range (within
  the LSRK/TOPO Doppler offset) and `sumwt` must be positive; after every
  model prediction `MODEL_DATA` must be non-zero on every unflagged channel
  of the field. A failure raises `RuntimeError`; all results are listed in
  `justification.json["checks"]`.
- **Thermal noise from a signal-free image.** sigma_th is measured once on
  the dirty image of `(XX-YY)/2` (Stokes Q; `(RR-LL)/2` = V for circular
  feeds), which has the noise of the Stokes I image but no source signal or
  sidelobes (`justification.json["derived"]["sigma_th"]`).
- `GainCalConfig.spwmap` (passed to `applycal`), `AJISAIConfig.caller`
  (`"user"` or `"asebi"`; interactive masking is refused for `"asebi"`),
  `AJISAIConfig.relabel_j2000_to_icrs`.
- `ms_utils.get_dish_diameter_m`, `ms_utils.get_uv_max`,
  `ms_utils.cell_from_umax`; `get_array_info` also returns the dish
  diameters.
- `examples/full_config_template.py`: every configuration option with its
  default value and the allowed alternatives in the comments. Running it
  unchanged (after setting `vis`) is equivalent to the defaults. Shown on
  the configuration page of the documentation and linked from the README.
- `AJISAIConfig.rms_target_radius_arcsec` (default `None`): inner radius,
  in arcsec, of the off-source annulus for the legacy
  `rms_method="annulus"`. The pipeline passes it to `compute_rms` for every
  image, and it is recorded in the `rms_info` of `justification.json`.

### Changed

- **Default self-calibration is `SelfcalSchedule(mode="data")`.** The v0.2
  ladder (`inf -> 6*IT -> 3*IT -> amplitude inf`, best by dynamic range) is
  `SelfcalSchedule(mode="fixed")`. Passing `steps` without `mode="fixed"`
  raises `ValueError` instead of being silently ignored. `mode="data"`
  requires `gaintype="T"`, `applymode="calonly"` and a linear
  `applycal_interp` (the defaults); from `GainCalConfig.combine` it takes
  only `"spw"` (and derives the applycal `spwmap` when none is given).
- A failing `plotms` (e.g. no DISPLAY on a headless server) no longer fails
  the run: the gain-table plots are skipped with a warning.
- The summary figure draws stages that were not applied (`rejected`) in gray.
- **The input MS is never written.** Previously the initial CLEAN saved its
  model into the input MS, the first iteration ran `applycal` on it (creating
  CORRECTED_DATA) and the J2000 -> ICRS relabelling edited it in place. Now
  every task opens the input read-only: each iteration copies the previous
  stage's MS to a temporary MS, predicts the model into it when needed,
  runs `gaincal` and `applycal` on the copy, splits CORRECTED into
  `intermediate/selfcal_<N>.ms` and deletes the copy. The relabelling is
  applied to a copy (`intermediate/input_icrs.ms`) only when the input has
  J2000 frame codes. `tclean` never writes a model (`savemodel="none"`); the
  `clearcal` on the new MS is gone.
- **Stage MSs are kept and immutable.** `intermediate/selfcal_<N>.ms` holds
  DATA (the self-calibrated visibilities) and MODEL (the model of that
  stage's image, which the next stage solves against) and is not modified
  after it is made.
- **Model prediction from the Taylor models.** Models are written with `ft`
  of all Taylor-term model images, with `nterms` and the model's reference
  frequency (standard gridder); other gridders use a `tclean` prediction.
  This replaces the old heuristic that re-ran `tclean` when the mean model
  amplitude looked like 0 or 1.
- **Cell from u_max.** The default cell is `1/(5 u_max)`, with `u_max` from
  the longest unflagged baseline at the highest frequency (truncated to 4
  significant figures); the image covers the primary beam down to 20 % at
  the lowest frequency. The v0.2 rule (90th-percentile baseline at the
  median frequency, beam / `cellpix`) is kept as
  `ImagingConfig.cell_rule="percentile"` and in `pick_cell_imsize(rule=...)`.
- **Dish diameter from the MS.** The primary beam uses the smallest dish in
  `ANTENNA/DISH_DIAMETER` (the largest primary beam); 12 m is only the
  fallback for an MS without the column (`pick_cell_imsize(diameter_m=None)`).
- **Multiscale scales.** The default stays `[0, 1, 3] x b` (b = the measured
  dirty-image beam in pixels; v0.2 used the configured pixels per beam).
  `multiscale_scales()` builds the ladder `0, b, 3b, 9b, ...` up to the
  largest scale whose support fits inside a mask region, for callers that
  choose it (ASEBI's final image); it is not the default, because which
  scales serve a source best depends on its structure, the ladder was tested
  on one source only, and on a single long-baseline EB it reached scales the
  uv coverage does not see (see Fixed).
- **Absolute CLEAN threshold in the self-cal loop.** Every self-cal image is
  CLEANed to `threshold_nsigma × sigma_th` (3 sigma_th), the same absolute
  threshold in every iteration, instead of `threshold_factor ×` the RMS of
  the previous image (still available as `threshold_mode="image_rms"`).
- **Statistics on Taylor term 0.** All image names go through one helper;
  with `nterms > 1` the statistics, FITS exports and gallery use
  `.image.tt0`. Metrics rows gain `image_base`, `threshold_jy`, `solint_s`
  and `caltable`.
- **CRTF user masks.** `mask_mode="user"` accepts a CRTF region file (first
  line `#CRTF...`) or a CASA mask image, and passes it to every `tclean`
  (initial, each iteration, final) instead of reusing the initial `.mask`.
- `GainCalConfig.calwt` default `False` (an amplitude step must not rescale
  the weights), `GainCalConfig.minsnr` default `3` (was 1.5), and
  `GainCalConfig.applycal_interp` default `"linearPDperobs"` (time
  interpolation never crosses an observation boundary of a concatenated MS;
  ignored by CASA for a single observation).
- `casaplotms` is optional: without it the gain-table plots are skipped with
  a warning instead of disabling CASA support.
- The install instructions (README, documentation) give a single command
  that installs the latest version from the `main` branch; the per-release
  tag is no longer quoted. Updating on a compute server uses
  `pip install --upgrade --force-reinstall --no-deps "git+https://...@main"`
  (the version string does not change between commits on `main`).

### Removed

- The channel-averaged by-product `intermediate/selfcal_<idx>_avg.ms`
  (`AJISAI._make_chanavg_ms`, the `selfcal_avg_ms` metrics column and its
  tests). Averaging every spw to one channel smears the longest baselines,
  and nothing used the copy.
- `AJISAI._ensure_model_column` (replaced by the explicit model prediction
  and its check).

### Fixed

- **Multiscale scales the data cannot see; empty initial CLEAN.** A scale is
  now used only when its scale-PSF (the dirty PSF convolved twice with CASA's
  multiscale component, as `MatrixCleaner` computes it) has its largest
  magnitude at the component's own position; other scales, derived or given
  in `ImagingConfig.scales`, are dropped (a given one with a warning) and
  recorded in `derived.scales` (`candidates`, `psf_check`). With the
  development default of that time (the ladder bounded by the mask region
  only, since withdrawn), on the 8.5 km baselines of
  V1094 Sco (2017 TM1, cell 6.2 mas) it reached 628 pix, whose scale-PSF
  peaks at a negative sidelobe, CASA's multiscale minor cycle refused to
  start ("As Peak of PSF is negative"), the initial CLEAN ended with an
  all-zero model, and the run failed only at the model prediction of the
  first stage with a misleading "no CLEAN components above the threshold".
  The initial CLEAN (with `niter > 0`; a `niter = 0` dirty image expects no
  model) now stops at once (`RuntimeError`, check `clean_model`) when its
  model is empty, saying whether the minor cycle did not run (peak
  inside the mask above the threshold) or there is no emission to CLEAN.
  `justification.json` records the CASA log file (`casa_log`).
- `rms_method="annulus"` could not be used from `AJISAIConfig`: the pipeline
  never passed a target radius to `compute_rms`, so `_validate_inputs` only
  warned and the run then failed with `ValueError` at the dirty-image
  statistics, after `tclean` had already run. Input validation now raises
  `ValueError` before any imaging when `rms_method="annulus"` is used
  without a positive `rms_target_radius_arcsec`, and warns when the radius
  is set for a method that does not use it.

## [0.2.1] - 2026-10-02

### Added

- `selfcal_images.png`: the CLEAN image of every self-calibration round in
  one row, left to right in the same order as the summary plot, on a shared
  `inferno` colour scale. Two rows by default (linear, and a `gamma = 0.3`
  power-law stretch that brings up faint emission), each with its own
  colour bar; the field shown is 10 beams wide by default; the best round
  is framed. A run that stopped early simply has fewer panels.
- `PlotConfig` (`AJISAIConfig(plots=PlotConfig(...))`) with the gallery
  options: colour map, the two `gamma` values, field of view (in beams or
  arcsec), and centring on the phase center or the source peak.
- `ajisai.plotting` module holding the figure code, and
  `plot_from_workdir(workdir, ...)` to regenerate the three PNGs from an
  existing output directory without CASA, e.g. with a different `gamma`.
- JOSS paper draft under `paper/` (`paper.md`, `paper.bib`) and a workflow
  that compiles it with the Open Journals toolchain on every change
  (`.github/workflows/draft-pdf.yml`).

### Changed

- `selfcal_summary.png`: one tick per round, labelled with what the round
  did (`no self-cal`, `1st: phase / inf`, `2nd: phase / 36 s`, ...)
  instead of fractional iteration numbers; the best round is marked in
  every panel and named in the legend; integer tick labels; the beam panel
  keeps a minimum y range so that a sub-percent beam change is not drawn as
  a jump; `ok` points and the best line are drawn in AJISAI purple.
- `ajisai_refant_selection.png`: square frame with the same X/Y scale and
  a colour bar of the same height; flagged fraction in 0.25-wide bins of
  the `Purples` colour map centred on the flag threshold; excluded antennas
  crossed out in grey; the chosen antenna ringed in light purple.
- All figures use a serif font (Times New Roman where installed, with
  fallbacks) and inward ticks, and are saved at 150 dpi (was 120). The
  style is applied per figure and does not change the caller's matplotlib
  settings.
- `plot_refant_selection()` moved to `ajisai.plotting` (still importable
  from `ajisai`); `outpath` is now optional and the figure is returned.

## [0.2.0] - 2026-10-01

### Added

- Each self-calibration iteration now also writes a channel-averaged copy of
  its measurement set, `intermediate/selfcal_<idx>_avg.ms`, in which every
  spectral window is averaged to a single channel. The copy is meant for
  quick inspection and downstream continuum analysis and is not fed back
  into the self-calibration loop; its path is recorded as `selfcal_avg_ms`
  in the per-iteration metrics and in `justification.json`.
- **TW Hya demo validated end-to-end on real CASA.** The pipeline ran on
  the public ALMA TW Hya Band 7 dataset (Project 2011.0.00340.S) with all
  defaults and completed all 4 self-cal iterations with `status="ok"`.
  Reference results:
  - Dynamic range: 81.06 (iter 0) -> 246.49 (iter 4), a **3.04x improvement**.
  - Off-source RMS: 4115.18 -> 1443.90 uJy/beam, a **2.85x reduction**.
  - Peak: 333.59 -> 355.91 mJy/beam (+22.32 mJy, +6.7%).
  - Beam: 482.8 -> 480.1 mas (essentially unchanged).
  - Hybrid refant selection chose DA42 after filtering out 7 of 26 antennas
    with flagged fraction >= 25% (19 antennas passed the filter).
  - Pipeline-derived parameters: median freq 372.65 GHz, integration time
    6.048 s, on-source 30.64 min, cellsize 0.066", imsize [375, 375],
    MRS 4.51".
- `docs/_static/selfcal_summary.png`, `docs/_static/ajisai_refant_selection.png`:
  reference figures from the TW Hya demo run, embedded in README and tutorial.
- `docs/_static/metrics.csv`, `docs/_static/justification.json`: reference
  outputs from the demo run committed to the repo for byte-by-byte
  comparison against users' own runs.
- TW Hya tutorial updated with precise reference numbers, per-iteration
  metrics table, and embedded figures.
- README has a new "Example results: TW Hya Band 7 demo" section.
- Community files: `CONTRIBUTING.md` (development setup, tests, lint,
  pull-request workflow, and design policy), GitHub issue templates for bug
  reports and feature requests, a pull-request template, and a "Support and
  contributing" section in the README.
- Citation metadata: `CITATION.cff`, `.zenodo.json`, and a Zenodo DOI badge
  in the README. The README citation section points to the published ApJ
  paper and the Zenodo record.
- Project logo and the name and design rationale in the README.

### Changed

- The design page of the documentation describes AJISAI's design on its own
  terms; its comparison table was replaced by an "At a glance" table.
- Installation instructions install from GitHub; a PyPI release is planned.

### Fixed

- `phase_shift=True` no longer crashes on ICRS datasets. `icrs_to_j2000`
  now accepts CASA's declination format with `.` separators
  (e.g. `-34.17.38.348`) as well as the space-separated form, and writes the
  J2000 declination in a form that CASA `fixplanets` parses as an angle.
- `.zenodo.json` described the final image as the one with the highest peak
  signal-to-noise ratio; it now says dynamic range, matching the default
  `quality_metric`.
- `examples/run_twhya_demo.py`: SSL verification failure on some monolithic
  CASA distributions (`CERTIFICATE_VERIFY_FAILED`) is now avoided by using
  `certifi`'s CA bundle when available.
- `examples/run_twhya_demo.py`: switched plotms invocation from
  `casatasks.plotms` to `casaplotms.plotms` to match modern CASA where
  `plotms` was moved out of `casatasks`.
- `examples/run_twhya_demo.py`: made the demo runnable from inside a CASA
  interactive prompt (`exec(open(...).read())`) without losing the
  `prepare_data` entry point.

### Added (test/CI infrastructure)

- pytest test suite under `tests/` with 93 unit tests covering:
  - `ms_utils` pure-math helpers (`_smallest_5_smooth_at_least`,
    `_round_to_sig_figs`, `_baars_taper_factor`,
    `_parse_uvtaper_to_image_fwhm`, etc.)
  - `pick_cell_imsize` math reproducing the V883 Ori golden output
  - `AJISAIConfig` validation (defaults, immutability, phase_center
    3-tuple semantics, mask_mode validation, rms_method warnings)
  - FITS loader (`load_fits_image`) on a synthetic CASA-style image
  - RMS estimators (`compute_rms`) on noise-only and source+noise
    synthetic FITS images
  - the channel-averaged MS step and the RA/Dec string parsing used by
    `icrs_to_j2000`
- `tests/conftest.py` with reusable synthetic FITS fixtures and a
  `@pytest.mark.casa` marker that auto-skips when CASA is unavailable.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`):
  - pytest matrix across Python 3.10, 3.11, 3.12
  - Sphinx HTML build sanity check
  - ruff lint job (advisory only in v0.1)
- pyproject.toml `[tool.pytest.ini_options]` and `[tool.coverage]`
  configuration.
- CI status badge in README.

## [0.1.0] - 2026-05-25

Initial public release.

### Added

- `ajisai` package with class-based architecture:
  - `AJISAI` orchestrator class
  - `AJISAIConfig` hierarchy (`ImagingConfig`, `GainCalConfig`,
    `SelfcalStep`, `SelfcalSchedule`)
- Native MS-query utilities in `ajisai.ms_utils` that replace seven
  `analysisUtils` functions: `get_array_info`, `get_on_source_time`,
  `get_median_frequency`, `get_baseline_at_percentile`,
  `pick_cell_imsize`, `rad_to_radec`, `icrs_to_j2000`. Validated against
  the analysisUtils golden output `[0.0081, [4800, 4800]]` for the
  V883 Ori 232 GHz test case.
- Sigma-clipping + center exclusion as the default off-source RMS
  estimator (`rms_method="sigma_clip_excl"`), validated to agree with
  the legacy annulus method within 1% on G204SW_TM12.
- Hybrid reference-antenna selection (flag-fraction filter + nearest to
  geometric center) with single-panel justification PNG.
- Phase shift functionality (off by default). Supports auto-detection
  via `imstat` max pixel and manual `phase_center=(ra, dec, frame)`
  with `"ICRS"` or `"J2000"` frames; ICRS inputs are converted to
  J2000 for `fixplanets` and the MS frame label is restored to ICRS
  afterwards.
- uvtaper-aware cell size: `pick_cell_imsize` computes the predicted
  effective beam as the quadrature sum of the untapered beam and the
  taper image-domain FWHM. After `_make_dirty_image`, AJISAI compares
  the actual beam against the prediction and warns if they differ by
  more than 20%.
- CLEAN mask modes: `auto-multithresh` (default), `interactive` (CASA
  ≤ 6.6 only), `user` (pre-made mask file), `none`.
- Structured `justification.json` output recording every parameter
  choice and its rationale.
- Four-panel self-cal summary plot (`selfcal_summary.png`).
- Per-iteration gain diagnostics under `diagnostics/`.
- Multi-field workflow via `list_target_fields()` helper.
- ASCII banner displayed at the start of `run()`; controllable via
  `cfg.show_banner`.
- Standalone TW Hya demo (`examples/run_twhya_demo.py`) that performs
  download, pre-processing, and AJISAI run end-to-end.
- Standalone RMS-method comparison tool (`tools/sigma_clip_rms_test.py`).
- Sphinx documentation with installation, quickstart, tutorial,
  configuration reference, output artifacts, design philosophy, and
  auto-generated API reference. ReadTheDocs configuration included.

### Design decisions

- Deterministic fixed schedule (3 phase + 1 amp by default); all
  iterations run to completion. Best image is selected by dynamic range
  at the end. Anomalies are logged but never alter pipeline flow.
- Pixel-accurate phase center via `imstat` max pixel (no Gaussian fit).
- ALMA continuum focus; line/mosaic/spectral-scan support is out of scope
  for v0.1.

### Removed / Replaced

- `analysisUtils` runtime dependency. AJISAI no longer requires the
  110,000-line analysisUtils source code.
- `imdata` runtime dependency. FITS image loading is implemented
  directly on top of astropy.

[Unreleased]: https://github.com/Y-Masayuki/AJISAI/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Y-Masayuki/AJISAI/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Y-Masayuki/AJISAI/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Y-Masayuki/AJISAI/releases/tag/v0.1.0
