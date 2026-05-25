# Changelog

All notable changes to AJISAI are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/masayuki-yamaguchi/AJISAI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/masayuki-yamaguchi/AJISAI/releases/tag/v0.1.0
