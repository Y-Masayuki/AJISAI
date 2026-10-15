# Output artifacts

After `AJISAI(cfg).run()` completes, the working directory contains the
files described below. The directory is `<cfg.results_dir>` if specified,
otherwise `./ajisai_<projname>/`.

```
ajisai_<projname>/
├── final.image                    ← final image (CASA format; Taylor term 0 if nterms > 1)
├── final.fits                     ← final image (FITS export)
├── final_selfcal.ms               ← self-calibrated visibilities of the best stage (DATA only)
├── metrics.csv                    ← one row per stage (also the stages that were not applied)
├── selfcal_summary.png            ← 4-panel diagnostic plot
├── selfcal_images.png             ← CLEAN image of every iteration, side by side
├── justification.json             ← every decision with its reason
├── ajisai_refant_selection.png    ← refant choice visualization
├── intermediate/                  ← all intermediate MS and image products
│   ├── dirty.image,  dirty.fits
│   ├── noise_Q.image / .fits      ← signal-free (XX-YY)/2 dirty image (sigma_th)
│   ├── clean_iter0.image / .fits  ← pre-selfcal CLEAN (.image.tt0, .tt1, .alpha... if nterms > 1)
│   ├── clean_sc1.image / .fits    ← after iteration 1
│   ├── clean_sc2.image / .fits    ← after iteration 2
│   ├── selfcal_1.ms, selfcal_2.ms, ...  ← stage MSs (applied stages): DATA + MODEL, never modified
│   ├── sc1.pcal, sc2.pcal, ...    ← gain tables (solutions below the G1 gate flagged)
│   ├── final_clean.*              ← the final image and all its products
│   ├── input_icrs.ms              ← relabelled copy of the input, only if it had J2000 codes
│   ├── user_mask_on_grid.mask     ← the user CRTF on the image grid (for the scales)
│   └── phaseshifted.ms            ← if phase_shift=True
└── diagnostics/
    ├── iter1_phase_EB_phasegain.png       ← mode="data": iter<k>_phase_<interval>_phasegain.png
    ├── iter2_phase_12.1s_phasegain.png
    └── ...                                ← mode="fixed": iter4_amp_inf_ampgain.png etc.
```

The input measurement set is only read. AJISAI never writes MODEL or
CORRECTED columns into it and never relabels it: each iteration copies the
previous stage's MS to a temporary MS (`selfcal_<N>_applycal.tmp.ms`), solves
and applies the gains on the copy, splits the corrected data into
`selfcal_<N>.ms`, and deletes the copy.

## `final.image` / `final.fits`

The final image: the data of the best stage (with the default
`SelfcalSchedule(mode="data")` the last accepted stage; with `mode="fixed"`
the iteration with the highest dynamic range), written to
`final_selfcal.ms`, imaged once more with the same cell, image
size, nterms, reffreq, scales and mask, and CLEANed deeper than the
self-calibration images. The threshold steps through
`ImagingConfig.final_depth` × sigma_th (default 3, 2, 1, 0.5) and stops when
the aperture flux has converged (see `final_image` below). With
`nterms > 1` these are copies of Taylor term 0 (`.image.tt0`); the other
products (`.image.tt1`, `.alpha`, `.alpha.error`, `.residual.tt0`,
`.model.tt*`, `.pb.tt0`, `.mask`) are in `intermediate/final_clean.*` and
listed in `justification.json`.

With `ImagingConfig(final_depth=())` no final image is made and `final.*`
is the best stage's image, as in v0.2.

This is the primary deliverable. Use as you would any CASA image: open in
CARTA, run `imstat` on it, etc.

## `final_selfcal.ms`

The self-calibrated visibilities of the best stage, split from its stage
MS with `datacolumn="data"`: a DATA column only (no MODEL, no CORRECTED), so
it can be imaged or combined without carrying self-calibration scratch
columns.

## `metrics.csv`

Per-iteration metrics in CSV form. Columns:

| Column                | Meaning                                           |
| --------------------- | ------------------------------------------------- |
| `iteration`           | 0 = no-selfcal CLEAN; 1-3 = stages (`mode="data"`), 1-4 = iterations (`mode="fixed"`) |
| `label`               | `phase_EB`, `phase_<interval>` (`mode="data"`) or `SelfcalStep.label` |
| `calmode`             | `"p"` or `"a"` (or `"-"` for iteration 0)          |
| `solint`              | `mode="data"`: the interval (`"EB"`, `"scan"`, `"12.1s"`); `mode="fixed"`: the solint string used (`"6*IT"` resolved) |
| `peak_jy_beam`        | image peak in Jy/beam                              |
| `rms_jy_beam`         | off-source RMS (method per `cfg.rms_method`; parameters in `rms_info`, see below) |
| `snr`                 | peak / rms                                         |
| `dynamic_range`       | also peak / rms (same numerator/denominator)      |
| `bmaj_arcsec` / `bmin_arcsec` | synthesized beam axes                     |
| `solint_s`            | solint in seconds (empty for one solution per EB / `"inf"`) |
| `threshold_jy`        | absolute CLEAN threshold of that image            |
| `n_gaincal_solutions` | number of un-flagged solutions returned by gaincal |
| `status`              | `"ok"`, `"anomaly"`, `"skipped"` (gaincal gave no solution, `mode="fixed"`) or `"rejected"` (evaluated but not applied, `mode="data"`) |
| `anomaly_reason`      | text reason when status is not `"ok"` (for `"rejected"`, the G1/G2 decision) |
| `image_path`          | absolute path to that iteration's image (`.image.tt0` if nterms > 1) |
| `image_base`          | image name without the product suffix             |
| `fits_path`           | absolute path to that iteration's FITS           |
| `caltable`            | gain table of that iteration                       |
| `selfcal_ms`          | the iteration's stage MS (`selfcal_<N>.ms`)       |

Useful for plotting your own trends or comparing AJISAI runs across data
sets.

## `selfcal_summary.png`

Four-panel plot showing how the image quality evolved across iterations:

1. **Dynamic range** (peak / RMS_offsource) — should generally climb. With
   `mode="data"` it is a diagnostic only: the stages are decided on held-out
   data.
2. **Peak** (mJy/beam) — climbs as self-cal sharpens the source.
3. **RMS** (μJy/beam) — should decrease as calibration improves.
4. **Beam** (geometric mean, milliarcsec) — typically nearly constant. The
   panel keeps a minimum y range of 2% of the beam, so a sub-percent change
   is not drawn as a large jump.

The x axis has one tick per round, labelled with what the round did, e.g.
`no self-cal`, `1st: phase / EB`, `2nd: phase / 12 s`, `3rd: phase / scan`
(`mode="data"`) or `1st: phase / inf`, `2nd: phase / 36 s`, ...,
`4th: amp / inf` (`mode="fixed"`; the solution interval is the resolved
value, rounded to whole seconds).

Points are color-coded by `status`:

- **Purple** (`#280a69`) : `ok`
- **Orange**             : `anomaly` (catastrophic DR drop; logged but not rolled back)
- **Red**                : `skipped` (gaincal returned no solutions)
- **Gray**               : `rejected` (`mode="data"`: evaluated, not applied; no image)

A purple dashed vertical line in every panel marks the stage selected as
`best` (the last accepted stage, or by the quality metric with
`mode="fixed"`); the legend names that round.

## `selfcal_images.png`

The CLEAN image of every iteration in one row, left to right in the same
order as `selfcal_summary.png`, so the effect of each round can be seen
directly. A run that stopped early simply has fewer panels; an iteration
without an image (e.g. `skipped`) is drawn as a grey placeholder.

- All panels share one colour scale (`inferno`, 0 to the highest peak in
  the field shown), so the rounds are directly comparable.
- Two rows by default: the top row is linear (`gamma = 1`), the second row
  uses a power-law stretch with `gamma = 0.3` to bring up the faint
  emission and the noise pattern. Each row has its own colour bar.
- The field shown is a square of half-width 10 × beam, centred on the
  phase center.
- Each panel shows the synthesized beam (bottom left), the dynamic range
  and off-source RMS of that round, and the round's label as title. The
  panel of the `best` iteration is framed in light purple.

The colour map, the two `gamma` values, the field of view, and the centring
(`"phase"` or the source `"peak"`) are set with
[`PlotConfig`](configuration.md#plotconfig), e.g.
`AJISAIConfig(vis=..., plots=PlotConfig(images_gamma2=0.5))`.

### Re-plotting a finished run

The three PNGs are drawn from `metrics.csv`, `justification.json`, and the
per-iteration FITS images, so they can be regenerated without CASA, for
example with a different stretch or a wider field:

```python
from ajisai import plot_from_workdir

plot_from_workdir("ajisai_<projname>", gamma2=0.5, fov_beam_factor=15)
```

Keyword arguments are passed to `ajisai.plotting.plot_selfcal_images`; see
the [API reference](api/plotting.rst).

## `justification.json`

The most important file for reproducibility. A structured record of every
decision AJISAI made. Top-level keys:

`ajisai_version`, `casa_version`
:   Software versions used.

`casa_log`
:   The file the CASA logger wrote to during the run.

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
    - `working_vis`: the MS the pipeline read (the input, or its relabelled
      copy) and why.
    - `cellsize`, `imsize`, `dish_diameter`: cell size (rule, u_max) and
      image size with the formula used; the dish diameter read from
      `ANTENNA/DISH_DIAMETER`.
    - `spectral_setup`: spws, frequency range and correlations of the field.
    - `sigma_th`: thermal noise from the signal-free dirty image.
    - `scales`: multiscale scales and how they were bounded: `candidates`
      (from the beam and the mask, or given) and `psf_check` (per scale the
      scale-PSF `centre` and `max_abs`, the scales `dropped` and the rule).
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
    nested `rms_info` dicts that describe how RMS was computed: the
    method, its parameters, such as the exclusion radius of
    `sigma_clip_excl` or the inner radius `rms_target_radius_arcsec` and
    the outer radius of `annulus`, and the number of pixels used).

`selfcal`
:   (`mode="data"`) Summary of the self-calibration: `rules` (the G1 gate,
    gaincal minsnr, the acceptance rule, the stage-1 rule, the stage limit,
    the CASA loss table), `combine_spw`, `stages` (decision and reason of
    each stage), `accepted_stages`, `best_stage`, `stop_reason`,
    `g2_split` (`"time"`: even/odd integrations; `"band"`: band halves) and
    the `spwmap` used for combined-spw tables.

`stages`
:   One record per stage (0 = no self-cal): `data_ms` (the MS imaged),
    `stage_ms` (`selfcal_<N>.ms`, DATA + MODEL), `image`, `image_base`,
    `nterms`, `model_image` (the model stored in the stage MS),
    `threshold_jy`, and for self-cal stages `solve`: the gain `table`,
    `solint` and `solint_s`, `combine`, `refant`, `calmode`, `gaintype`,
    `minsnr`, `minblperant`, `solnorm`, `applycal` (`interp`, `applymode`,
    `calwt`, `spwmap`), `model_image` / `model_nterms` / `model_reffreq_hz`
    (the model the solve used), `solved_on`, `stage_ms`, `n_solutions`, and
    `flagged_solution_fraction` per observation and spw. The same list is
    `AJISAI.stages`; the keys of `solve` are `ajisai.SOLVE_INFO_KEYS`.
    With `mode="data"`, `solve["solint"]` is the string given to gaincal and
    `solve["solint_s"]` the nominal `n x IT`. gaincal bins a scan on a fixed
    time grid from its first integration, so for `n` integrations the string
    is `n x IT - delta` with `delta = IT/(4M)` (`M` chunks in the longest
    scan), which keeps every grid edge clear of the integrations (`'int'`
    for `n = 1`).

    With `mode="data"` every stage also has `selection`, the record of how
    it was decided (stages that were not applied have `status: "rejected"`,
    no MS or image, and the same record):

    - `measured`: `s` per spw (noise scale from adjacent integrations, with
      its sigma and number of pairs), `r` (solution-SNR rate per antenna,
      median, 10th percentile, per spw), `D_eff` (`tau_s`, `C`, `D`,
      `sigma_D`, `n_pairs` of the lag structure function) and `D_fit`
      (`tau0_s`, `beta`, significance), `T_star_closed_s` (the closed-form
      optimum, a diagnostic), `int_time_s`, `scan_nint_median`, `g2_split`;
    - `candidates`: per candidate `label`, `kind` (`int`, `scan`, `eb`),
      `n_int`, `T_s`, `gaincal_solint`, `g1` (`pass`, `snr_median`,
      `snr_p10`, `solution_pass_fraction`, `failing_antennas`,
      `antenna_snr_median`), `V_pred`
      (`V_atm` + `V_th`), `kappa_full`, `evaluated` and, when evaluated,
      `g2`: `V_held` and `V_ref` (held-out `-ln C` of the candidate and the
      reference), `dV_raw` and its paired jackknife `sigma`, the loss
      correction (`thermal_held`, `thermal_full`, `corr`, `kappa_held`),
      `improvement_corr`, `two_sided`, `correction_cand`, `correction_ref`;
    - `previous` (the stage the correction refers to), `T_pred`, `rule_A`,
      `rule_B`, `chosen`, `chosen_reason`, `decision` (`apply`, `skip` for
      stage 1, `stop`), `reason`, `threshold`, `spw_combine` (whether
      combining spws would lift the chosen interval over G1; recorded only);
    - for applied stages, `g1_gate_applied` (table rows flagged below the
      gate, per antenna), `same_data_diagnostics` (peak, RMS and dynamic
      range of the stage's image; never used for decisions) and `g3` (data/model flux ratio of the shortest
      20 % baselines before and after the stage, per observation; recorded
      only for phase stages).

`checks`
:   The consistency check after every `tclean` (the image's frequency band
    covers the data's frequency range; `sumwt > 0`), the model check of the
    initial CLEAN (`clean_model`: the model has a component) and every model
    prediction (`MODEL_DATA` non-zero on every unflagged channel). A failed
    check stops the run with `RuntimeError`.

`best`
:   `{"iteration": int, "rule": "...", "metric_key": "dynamic_range",
       "value": float, "image": "/path/to/clean_sc<N>.image"}`. `rule`
    says how the stage was chosen (the last accepted stage with
    `mode="data"`; `value` is then the stage's dynamic range, a diagnostic).

`final_ms`
:   Path, stage and source of `final_selfcal.ms`.

`final_image`
:   The final image: `vis`, `image`, `products`, `sigma_th_jy`, the
    `depth_schedule` and one entry per CLEAN depth step in `steps`
    (`nsigma`, `threshold_jy`, `f_model_jy`, `f_unc_jy`, `f_corr_jy`,
    `sigma_f_jy`, `width_jy`, `omega_r_pix`, `omega_clean_pix`,
    `restored_bias_jy`, `stop`, `reason`), `converged`, `stop_nsigma`, the
    `aperture` used, and `warnings`. Here `F_model` is the CLEAN model flux
    in the aperture, `F_unc = sum(residual)/Omega_R` the flux left in the
    residual (Omega_R = dirty-beam volume in pixels), `F_corr = F_model +
    F_unc`, `sigma_F` the noise of `F_unc`, and `restored_bias_jy =
    F_unc (Omega_R/Omega_clean - 1)` the amount by which the restored image
    over-counts the residual flux. The CLEAN stops when `|F_unc| <=
    max(3 sigma_F, 0.01 F_corr)` or when `F_corr` changed by no more than
    that since the previous step; quote `F_corr` as the flux.

This file is designed to be **citable in publications**: it records
exactly what AJISAI did to your data, so the reduction is reproducible by
anyone with the same MS.

## `ajisai_refant_selection.png`

Square scatter plot (same scale on both axes) showing why AJISAI chose its
reference antenna:

- Each antenna is plotted at its XY position (meters from the array
  geometric center).
- Color encodes the flagged fraction in 0.25-wide bins (`Purples`
  colormap, centred on the `refant_flag_threshold`, which is also marked
  on the colour bar), so antennas on either side of the selection rule
  are clearly separated.
- Antennas at or above the `refant_flag_threshold` are crossed out in
  grey — these are excluded from the candidate set.
- The chosen refant is ringed in light purple.
- The array geometric center is marked with a black "+".

This is intended as a justification artifact suitable for supplementary
material in publications.

## `intermediate/` — keep or delete

The intermediate directory contains every MS and image AJISAI generated.
The stage MSs `selfcal_<N>.ms` (DATA + MODEL) together with the gain tables
and the `solve` records in `justification.json` are enough to re-solve or
re-apply any stage later without re-running the self-calibration. It is
preserved by default so you can debug or re-image. If disk space is tight,
you can delete it; `final_selfcal.ms` and `final.*` at the top level do not
depend on it.

## `diagnostics/` — per-iteration gain plots

Standard CASA `plotms` outputs of phase (and amplitude, for the amp
iteration) versus time for each antenna. Inspect these if you suspect
gaincal misbehavior in a specific iteration.
