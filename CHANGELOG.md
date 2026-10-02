# Changelog

All notable changes to AJISAI are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `examples/full_config_template.py`: every configuration option with its
  default value and the allowed alternatives in the comments. Running it
  unchanged (after setting `vis`) is equivalent to the defaults. Shown on
  the configuration page of the documentation and linked from the README.
- `AJISAIConfig.rms_target_radius_arcsec` (default `None`): inner radius,
  in arcsec, of the off-source annulus for the legacy
  `rms_method="annulus"`. The pipeline passes it to `compute_rms` for every
  image, and it is recorded in the `rms_info` of `justification.json`.

### Fixed

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
