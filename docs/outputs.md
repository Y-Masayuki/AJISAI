# Output artifacts

After `AJISAI(cfg).run()` completes, the working directory contains the
files described below. The directory is `<cfg.results_dir>` if specified,
otherwise `./ajisai_<projname>/`.

```
ajisai_<projname>/
├── final.image                    ← best self-cal image (CASA format)
├── final.fits                     ← best self-cal image (FITS export)
├── metrics.csv                    ← one row per iteration
├── selfcal_summary.png            ← 4-panel diagnostic plot
├── justification.json             ← every decision with its reason
├── ajisai_refant_selection.png    ← refant choice visualization
├── intermediate/                  ← all intermediate MS and image products
│   ├── dirty.image,  dirty.fits
│   ├── clean_iter0.image / .fits  ← pre-selfcal CLEAN
│   ├── clean_sc1.image / .fits    ← after iteration 1
│   ├── clean_sc2.image / .fits    ← after iteration 2
│   ├── selfcal_1.ms, selfcal_2.ms, ...
│   ├── sc1.pcal, sc2.pcal, ...    ← gain tables
│   └── phaseshifted.ms            ← if phase_shift=True
└── diagnostics/
    ├── iter1_phase_inf_phasegain.png
    ├── iter4_amp_inf_phasegain.png
    └── iter4_amp_inf_ampgain.png
```

## `final.image` / `final.fits`

The best CLEAN image AJISAI produced, copied from
`intermediate/clean_iter<N>` where `N` is the iteration that achieved the
highest dynamic range.

This is the primary deliverable. Use as you would any CASA image: open in
CARTA, run `imstat` on it, etc.

## `metrics.csv`

Per-iteration metrics in CSV form. Columns:

| Column                | Meaning                                           |
| --------------------- | ------------------------------------------------- |
| `iteration`           | 0 = no-selfcal CLEAN; 1-4 = self-cal iterations   |
| `label`               | from `SelfcalStep.label`                          |
| `calmode`             | `"p"` or `"a"` (or `"-"` for iteration 0)          |
| `solint`              | actual solint string used (`"6*IT"` resolved)     |
| `peak_jy_beam`        | image peak in Jy/beam                              |
| `rms_jy_beam`         | off-source RMS (method per `cfg.rms_method`)      |
| `snr`                 | peak / rms                                         |
| `dynamic_range`       | also peak / rms (same numerator/denominator)      |
| `bmaj_arcsec` / `bmin_arcsec` | synthesized beam axes                     |
| `n_gaincal_solutions` | number of un-flagged solutions returned by gaincal |
| `status`              | `"ok"`, `"anomaly"`, or `"skipped"`                |
| `anomaly_reason`      | text reason when status is not `"ok"`             |
| `image_path`          | absolute path to that iteration's image           |
| `fits_path`           | absolute path to that iteration's FITS           |

Useful for plotting your own trends or comparing AJISAI runs across data
sets.

## `selfcal_summary.png`

Four-panel plot showing how the image quality evolved across iterations:

1. **Dynamic Range** (peak / RMS_offsource) — should generally climb.
2. **Peak** (mJy/beam) — climbs as self-cal sharpens the source.
3. **RMS** (μJy/beam) — should decrease as calibration improves.
4. **Beam** (geometric mean, milliarcsec) — typically nearly constant.

Points are color-coded by `status`:

- **Steel blue** : `ok`
- **Orange**     : `anomaly` (catastrophic DR drop; logged but not rolled back)
- **Red**        : `skipped` (gaincal returned no solutions)

A green dashed vertical line marks the iteration that was selected as
`best` by the chosen quality metric.

## `justification.json`

The most important file for reproducibility. A structured record of every
decision AJISAI made. Top-level keys:

`ajisai_version`, `casa_version`
:   Software versions used.

`started_at`, `finished_at`
:   ISO 8601 timestamps in UTC.

`input_config`
:   Complete dump of the `AJISAIConfig` that was used, including all
    sub-configs. Recreate the exact same run by feeding this dict back
    into `AJISAIConfig(**input_config)`.

`derived`
:   The auto-determined parameters AJISAI computed from the data, each
    with the reason. Keys include:

    - `refant`: chosen antenna, strategy, flag thresholds, full candidate
      list with positions and flag fractions.
    - `cellsize`, `imsize`: cell size and image size with the formula
      used.
    - `mrs_arcsec`: maximum recoverable scale.
    - `avg_int_time_sec`, `onsource_time_sec`, `median_freq_hz`: MS
      properties relevant to imaging and self-cal.
    - `untapered_beam_arcsec`, `taper_image_fwhm_arcsec`,
      `predicted_effective_beam_arcsec`: when `uvtaper` is set.
    - `beam_check`: comparison of actual dirty-image beam vs predicted.
    - `dirty_image`: peak/RMS/DR of the dirty image.
    - `phase_shift`: whether applied, target coordinates, source frame.

`iterations`
:   List of per-iteration records (same as `metrics.csv` rows, but with
    nested `rms_info` dicts that describe how RMS was computed).

`best`
:   `{"iteration": int, "metric_key": "dynamic_range", "value": float,
       "image": "/path/to/final.image"}`

This file is designed to be **citable in publications**: it records
exactly what AJISAI did to your data, so the reduction is reproducible by
anyone with the same MS.

## `ajisai_refant_selection.png`

Single-panel scatter plot showing why AJISAI chose its reference antenna:

- Each antenna is plotted at its XY position (meters from the array
  geometric center).
- Color encodes the flagged fraction (viridis colormap, 0..1).
- Antennas above the `refant_flag_threshold` are crossed out with a red
  "x" — these are excluded from the candidate set.
- The chosen refant is circled in green.
- The array geometric center is marked with a black "+".

This is intended as a justification artifact suitable for supplementary
material in publications.

## `intermediate/` — keep or delete

The intermediate directory contains every MS and image AJISAI generated.
It is preserved by default so you can debug or re-image. If disk space is
tight, you can safely delete it after taking note of any path you might
want from `metrics.csv`.

## `diagnostics/` — per-iteration gain plots

Standard CASA `plotms` outputs of phase (and amplitude, for the amp
iteration) versus time for each antenna. Inspect these if you suspect
gaincal misbehavior in a specific iteration.
