# Configuration reference

AJISAI's behavior is controlled by a hierarchy of frozen dataclasses. The
top-level container is [`AJISAIConfig`](#ajisaiconfig), which composes
sub-configs for imaging, gain calibration, the self-cal stages
(data-driven by default, a fixed ladder as an option), and the diagnostic
figures.

Pattern:

```python
from ajisai import AJISAI, AJISAIConfig, ImagingConfig, GainCalConfig, SelfcalSchedule, SelfcalStep, PlotConfig

cfg = AJISAIConfig(
    vis     = "...",
    imaging = ImagingConfig(...),     # optional, default is fine
    gaincal = GainCalConfig(...),     # optional
    schedule= SelfcalSchedule(...),   # optional
    plots   = PlotConfig(...),        # optional (figure options)
)
AJISAI(cfg).run()
```

For the complete auto-generated reference, see
[API reference](api/index.rst). The page below highlights the parameters
you are most likely to want to override.

## Full configuration template

The file below, [`examples/full_config_template.py`](https://github.com/Y-Masayuki/AJISAI/blob/main/examples/full_config_template.py)
in the repository, lists every option with its default value and the allowed
alternatives in the comments. Run as it is (after setting `vis`) it is
equivalent to `AJISAI(AJISAIConfig(vis=...)).run()`; copy it and change only
the lines you need.

```{literalinclude} ../examples/full_config_template.py
:language: python
```

## `AJISAIConfig`

### Required

`vis: str`
:   Path to the input measurement set. Must be a CASA MS directory.

### Recommended (auto-derived if None)

`field: Optional[str] = None`
:   FIELD_ID (string) or FIELD NAME of the target. If `None`, AJISAI uses
    the first FIELD_ID present in MAIN and warns if there are multiple.

`projname: Optional[str] = None`
:   Project name. Used for the output directory and result file names.
    If `None`, AJISAI derives it from the basename of `vis`.

`results_dir: Optional[str] = None`
:   Output directory. If `None`, AJISAI uses
    `./ajisai_<projname>/` under the current working directory.

### Phase shift (off by default)

`phase_shift: bool = False`
:   If `True`, AJISAI shifts the MS so that the target source lies at the
    phase center before self-cal. Otherwise the original phase center is
    preserved.

`phase_center: Optional[Tuple[str, str, str]] = None`
:   Manual phase-center override. When provided alongside
    `phase_shift=True`, AJISAI uses these coordinates instead of
    auto-detecting from the dirty image. Form: `(ra, dec, frame)`,
    where `frame` is `"ICRS"` or `"J2000"`. RA is read as hours and Dec
    as degrees, whichever separators are used (`:`, `.`, `h m s` /
    `d m s`, or spaces); an angle with a unit (`"241.4deg"`, `"-0.42rad"`)
    is also accepted. Examples:

    ```python
    phase_center=("16:25:45.0", "-24:12:23.0", "ICRS")
    phase_center=("16h25m45.0s", "-24d12m23.0s", "J2000")
    ```

    The strings are converted to radians when the configuration is
    validated, and CASA receives only radians: CASA itself would read a
    `:`-separated declination as a time (`-24:12:23.0` as -363.1 deg, i.e.
    Dec -3.1 deg).

    :::{note}
    How the shift is done: the target is converted (with
    `casatools.measures`) into the frame of the field's `PHASE_DIR` in the
    working MS, `phaseshift` shifts the data and computes UVW, and then
    FIELD `PHASE_DIR`, `DELAY_DIR`, `REFERENCE_DIR` and `SOURCE.DIRECTION`
    are written with the target, each converted into its own frame
    (`ajisai.ms_utils.write_field_direction`). `fixplanets` is not used
    and no frame label is changed, so the result is exact (well below
    0.01 mas) for ICRS data and for genuine J2000 data alike. The
    read-back offsets are recorded in `justification.json`
    (`derived.phase_shift.readback_offset_mas`).
    :::

### Strategy switches

`refant_strategy: str = "hybrid"`
:   Reference-antenna selection method. One of:

    - `"hybrid"` (default): keep antennas with flagged-fraction below
      `refant_flag_threshold`, pick the one nearest to the XY array
      geometric center.
    - `"geometric_center"`: pure geometric center, ignoring flag stats.
    - `"flag_stats"`: lowest flagged-fraction, ignoring position.
    - `"manual"`: use `refant_manual` as the chosen antenna name.

`refant_flag_threshold: float = 0.25`
:   Maximum flagged-fraction (0..1) for an antenna to be eligible under
    `refant_strategy="hybrid"`.

`refant_manual: Optional[str] = None`
:   Antenna name to use when `refant_strategy="manual"`.

`rms_method: str = "sigma_clip_excl"`
:   How AJISAI computes the off-source RMS used for thresholds and
    dynamic range. The default sigma-clipping with center exclusion has
    been validated to agree with the legacy annulus method to within 1%
    and requires no user-supplied source radius. Alternatives are
    `"sigma_clip"`, `"annulus"` (legacy; requires
    `rms_target_radius_arcsec`, see below), and `"mad"`.

`rms_target_radius_arcsec: Optional[float] = None`
:   Inner radius, in arcsec, of the off-source annulus used by
    `rms_method="annulus"`: the RMS is the standard deviation of the
    pixels farther than this from the phase center and within 0.4 × the
    image width. Choose it to enclose the source emission; it is the
    `target_radius` of the legacy AJISAI script. Required, and must be
    positive, when `rms_method="annulus"`: `run()` raises a `ValueError`
    during input validation otherwise, before any imaging is done. For
    the other methods the value is ignored (with a warning). The radius
    is recorded in `justification.json` with the RMS of every image.

`quality_metric: str = "dynamic_range"`
:   How AJISAI picks the best iteration with `SelfcalSchedule(mode="fixed")`:
    `"dynamic_range"` (peak / RMS_offsource) or `"peak_snr"`. With the
    default `mode="data"` the best stage is the last accepted one and these
    image statistics are recorded as diagnostics only: they are measured on
    the data the gains were solved from, which always favours the shortest
    solution interval.

`on_iter_anomaly: str = "log_only"`
:   What to do when an iteration looks anomalous (e.g. DR drop > 50%).
    Currently only `"log_only"` is implemented: anomalies are recorded
    in the justification log but do NOT alter pipeline flow.

### Data hygiene

`relabel_j2000_to_icrs: bool = True`
:   AJISAI never writes to the input MS. When this is `True` and the input
    carries J2000 frame codes, a copy (`intermediate/input_icrs.ms`) is
    relabelled J2000 -> ICRS (the frame code only; coordinates are not
    converted) and used as the working MS, but only for ALMA data observed
    before Cycle 3 (2015-10-01), whose J2000 label sits on ICRS numbers
    (ALMA knowledgebase). Any other J2000 label (not ALMA, observed on or
    after 2015-10-01, or with an unknown start) is genuine J2000: it is
    kept, the input is read directly, and AJISAI warns and records why
    (`derived.working_vis.relabel_caveat`); relabelling genuine J2000
    numbers as ICRS would move them by the frame bias (up to about 23 mas).
    `False` never relabels. Same rule as ASEBI (SPEC 4.1 item 10).

`caller: str = "user"`
:   Who drives the run: `"user"` (standalone) or `"asebi"` (the ASEBI
    pipeline). With `"asebi"`, `mask_mode="interactive"` and any
    `tclean(interactive=True)` are refused (viewer-free operation).

### Misc

`verbose: bool = True`
:   Print progress to stdout while running.

`show_banner: bool = True`
:   Print the AJISAI ASCII banner at the start of `run()`. Disable for
    batch jobs that loop over many fields.

## `ImagingConfig`

### CLEAN parameters

`robust: float = 0.5`
:   Briggs robust parameter (passed to `tclean`).

`weighting: str = "briggs"`
:   Visibility weighting scheme.

`niter: int = 50000`
:   Maximum CLEAN iterations per tclean call.

`gain: float = 0.05`
:   CLEAN loop gain.

`cyclefactor: float = 1.5`
:   tclean cycle factor.

`threshold_mode: str = "pure_noise"`
:   CLEAN depth of the self-calibration images (initial CLEAN and every
    iteration). `"pure_noise"` (default): an absolute threshold
    `threshold_nsigma × sigma_th`, the same in every iteration, where
    sigma_th is the off-source RMS of the dirty image of `(XX-YY)/2`
    (Stokes Q; `(RR-LL)/2` for circular feeds), measured once. That image
    has the noise of the Stokes I image but no source signal or sidelobes.
    `"image_rms"` (legacy, v0.2): `threshold_factor ×` the off-source RMS of
    the previous image.

`threshold_nsigma: float = 3.0`
:   Multiple of sigma_th for `threshold_mode="pure_noise"`. CLEANing the
    self-calibration models deeper puts noise into the model, which the next
    gain solve then fits.

`threshold_factor: float = 1.0`
:   Multiple of the previous image's RMS for `threshold_mode="image_rms"`.

`final_depth: Tuple[float, ...] = (3.0, 2.0, 1.0, 0.5)`
:   Depth schedule of the final image, in units of sigma_th. The final image
    is made once, from the best iteration's data (`final_selfcal.ms`), and
    its CLEAN continues step by step (no new PSF) until the flux in the
    aperture has converged: after each step, `F_corr = F_model + F_unc` with
    `F_unc = sum(residual)/Omega_R` (Omega_R: the dirty beam over the
    aperture's shape) and `sigma_F = sigma sqrt(N_pix Omega_beam)/Omega_R`
    (sigma: the residual rms outside the aperture); the CLEAN stops when
    `|F_unc| <= max(3 sigma_F, 0.01 F_corr)` or when `F_corr` changed by no
    more than that width since the previous step. A warning is issued if
    the last step is reached without convergence, and if the restored image
    over-counts the residual flux (`F_unc (Omega_R/Omega_clean - 1)`) by more
    than the width. `()` disables the final image (`final.*` is then the best
    iteration's image, as in v0.2).

`final_aperture_arcsec: Optional[float] = None`
:   Radius (arcsec, about the phase center) of the aperture for the final
    image's flux test. `None`: the CLEAN mask region, or a circle of
    `rms_exclude_beam_factor` beams when there is no mask.

`deconvolver: str = "multiscale"`
:   tclean deconvolver. Multiscale recommended for extended sources. With
    `nterms > 1`, `"mtmfs"` is used (with the multiscale scales).

`gridder: str = "standard"`
:   tclean gridder.

`pbcor: bool = False`
:   Whether to apply primary-beam correction in tclean.

`parallel: bool = False`
:   Run tclean in parallel mode (mpicasa).

`uvtaper: Tuple[str, ...] = ()`
:   uvtaper passed to tclean, e.g. `("1arcsec",)`, `("100klambda",)`,
    `("2arcsec", "0.5arcsec", "30deg")`. When set, AJISAI computes the
    expected post-taper effective beam and adjusts the cell size
    accordingly; see [Output artifacts](outputs.md) for the
    actual-vs-predicted beam check.

`nterms: int = 1`
:   tclean nterms (number of Taylor terms). With `nterms > 1` every tclean
    uses `deconvolver="mtmfs"`, the products are named `.image.tt0`,
    `.image.tt1`, ..., statistics are taken on `.image.tt0`, and the model is
    predicted from all Taylor terms (`ft` with `nterms` and the model's
    reference frequency).

`reffreq: Optional[str] = None`
:   Reference frequency of the Taylor expansion, e.g. `"227.0GHz"`, passed
    to every mtmfs tclean (`nterms > 1`); nterms = 1 images and the dirty
    image are made without it, so that their frequency axis is the data's
    band (tclean centres the axis on reffreq). `None` uses CASA's default
    (the middle of the frequency range); the value used is recorded with
    each solve.

### Image geometry

`cell: Optional[str] = None`
:   Cell size for every tclean, e.g. `"0.0062arcsec"` or `"6.2mas"`.
    `None`: derived with `cell_rule`.

`imsize: Optional[Tuple[int, ...]] = None`
:   Image size, `N` or `(Nx, Ny)` pixels. `None`: the primary beam down to
    20 % attenuation, for the smallest dish in `ANTENNA/DISH_DIAMETER` (12 m
    only as a fallback), rounded up to a 5-smooth size.

`cell_rule: str = "umax"`
:   Rule for the derived cell. `"umax"` (default): `cell = 1/(5 u_max)`,
    with `u_max` (wavelengths) from the longest unflagged baseline at the
    highest frequency, truncated to 4 significant figures; the image then
    covers the primary beam at the lowest frequency. `"percentile"` (legacy,
    v0.2): the synthesized beam from the 90th-percentile baseline at the
    median frequency (tapered by `uvtaper`), divided by `cellpix` and rounded
    to 2 significant figures. With a strong `uvtaper` the `"umax"` cell
    still samples the untapered `u_max`; use `"percentile"` or an explicit
    `cell` for heavily tapered images.

`cellpix: int = 10`
:   Number of pixels per synthesized beam for `cell_rule="percentile"`.

`scales: Optional[Tuple[int, ...]] = None`
:   Multiscale scales in pixels for every multiscale/mtmfs tclean. `None`:
    `[0, 1, 3] x b` with `b` the dirty-image beam FWHM in pixels. Which
    scales serve a source best depends on its structure; a caller that knows
    the source gives its own (e.g. `ajisai.core.multiscale_scales`, the ladder
    `0, b, 3b, 9b, ...` up to the largest scale that fits a mask region; CASA
    ignores scales too large for the mask). Then, given or derived,
    every scale whose scale-PSF (the dirty PSF convolved twice with CASA's
    component) does not peak at the component's own position is dropped:
    the uv coverage has no baseline short enough to see it, and when its
    largest sidelobe is negative CASA's multiscale minor cycle refuses to
    start and the CLEAN leaves no component. Dropped scales are recorded
    (`derived.scales.psf_check`); a dropped given scale is warned about.

### Masking

`mask_mode: str = "auto-multithresh"`
:   How AJISAI builds the CLEAN mask.

    `"auto-multithresh"`
    :   CASA's automatic threshold-based masking. Default. Works in any
        CASA version. Robust for typical ALMA continuum sources.

    `"interactive"`
    :   Let the user draw a mask in the CASA viewer at the FIRST tclean
        call. The saved mask is then reused by all subsequent self-cal
        iterations.

        **Requires CASA <= 6.6.** The casaviewer GUI was removed in CASA
        6.7+, so interactive tclean is non-functional there. AJISAI will
        warn (but not fail) if you request interactive mode on CASA 6.7+.
        Refused (`ValueError`) when `AJISAIConfig.caller="asebi"`.

    `"user"`
    :   Use a pre-made mask supplied via `user_mask`: a CRTF region file
        (text whose first line starts with `#CRTF`) or a CASA mask image. It
        is passed to every tclean (initial CLEAN, each iteration, final
        image). This is the portable, viewer-free option for CASA 6.7+
        users: prepare a region with CARTA or a text editor and point
        AJISAI at it.

    `"none"`
    :   No mask at all. Rarely useful.

`user_mask: Optional[str] = None`
:   Path to a CRTF region file or a CASA mask image. Required when
    `mask_mode="user"`.

### auto-multithresh tunables

`sidelobethreshold: float = 2.0`
`noisethreshold: float = 4.25`
`lownoisethreshold: float = 1.5`
`minbeamfrac: float = 0.3`
`growiterations: int = 75`
:   These are CASA's auto-multithresh parameters. The defaults follow
    the ALMA pipeline recommendations for continuum imaging.

## `GainCalConfig`

`minblperant: int = 4`
:   Minimum number of baselines required per antenna for a gain solution.

`minsnr: float = 3.0`
:   Minimum SNR for a gain solution to be accepted. Solutions below this
    are discarded; the corresponding visibilities are preserved unflagged
    because AJISAI uses `applymode="calonly"`. (v0.2: 1.5.) The fraction of
    flagged solutions per observation and spw is recorded for every
    iteration. With `SelfcalSchedule(mode="data")` this is only the floor:
    the gate is G1, a predicted true SNR of 5 per antenna and solution, and
    the solutions below it are flagged in the table after gaincal. Do not
    raise `minsnr` to 5: gaincal's SNR column scatters, so near the gate it
    would flag about half of the solutions, and with linear interpolation a
    flagged solution removes the correction from the whole span between its
    good neighbours.

`gaintype: str = "T"`
:   gaincal gain type. `"T"` averages parallel-hand polarizations for
    improved SNR; `"G"` solves them independently. `mode="data"` requires
    `"T"`.

`combine: str = "scan"`
:   With `mode="fixed"`: passed to gaincal (solutions are combined across
    scan boundaries within a solution interval). With `mode="data"` the time
    combination is chosen per stage (`"scan"` for one solution per EB, none
    for the scan and shorter intervals); only `"spw"` is taken from here:
    `combine="scan,spw"` solves all spws together, and when `spwmap` is
    empty AJISAI derives it (every spw of an observation mapped to its
    lowest spw).

`gaincal_interp: str = "linear,linear"`
:   Interpolation in time and frequency for the gain table.

`applycal_interp: str = "linearPDperobs"`
:   Interpolation for applycal. `"linearPD"` preserves phase unwrapping;
    `perobs` keeps the time interpolation from crossing an observation
    (execution block) boundary of a concatenated MS. CASA ignores `perobs`
    (with a log warning) when the table has one observation. `mode="data"`
    requires a linear interpolation: its predictions and held-out
    evaluation model applycal's linear interpolation between solutions. A
    stage switches to the nearest form of this string (`nearestPDperobs`)
    only when adjacent solutions differ by more than 1 rad rms
    (`2 D_eff > 1 rad^2`) and nearest interpolation beats linear at the
    chosen interval on held-out integrations by more than 2 sigma.

`applymode: str = "calonly"`
:   applycal mode. `"calonly"` ensures that flagged solutions do NOT
    propagate to flagging the visibilities themselves — important for
    preserving data on low-SNR antennas. Required by `mode="data"`: the
    solutions below the G1 gate are flagged so that those antennas keep the
    previous stage's correction (a flagged solution is not interpolated:
    the span between its good neighbours passes uncorrected).

`calwt: bool = False`
:   Calibrate the weights when applying. `False` (v0.3 default; v0.2:
    `True`): the self-calibration tables never change WEIGHT, so an
    amplitude step cannot rescale the weights by |g|².

`spwmap: Tuple[int, ...] = ()`
:   spwmap passed to applycal for every gain table, e.g. `(0, 0, 0, 0)` when
    the schedule's `combine` includes `"spw"`. `()`: identity.

## `SelfcalSchedule`

`mode: str = "data"`
:   How the self-calibration stages are chosen.

    `"data"` (default, v0.3)
    :   Phase stages whose solution intervals are decided from the data, with
        the rules of the ASEBI specification v0.3 (section 10.2; see
        [Design](design.md)): stage 1 is one solution per execution block;
        each later stage measures the noise scale, the solution-SNR rate of
        every antenna and the phase structure function, predicts the
        residual phase variance of candidate intervals (divisors of the
        median scan length, the scan, the EB), gates them on the predicted
        solution SNR (G1: 5), and chooses and accepts one on held-out
        integrations (G2: corrected improvement > max(1 sigma, 0.001)). At
        most three phase stages; the run stops at the first stage that is
        not accepted, and the best stage is the last accepted one.
        Solution intervals are integer multiples of the integration time,
        in seconds; there is no `"N*IT"` schedule.

    `"fixed"` (v0.2)
    :   Every step of `steps` runs, and the best iteration is chosen by
        `AJISAIConfig.quality_metric`.

`steps: Tuple[SelfcalStep, ...]`
:   The steps of `mode="fixed"` (ignored, and refused if changed, with
    `mode="data"`). Default, the v0.2 ladder `ajisai.FIXED_SCHEDULE_STEPS`:

    ```python
    SelfcalSchedule(mode="fixed", steps=(
        SelfcalStep("p", "inf",  label="phase_inf"),
        SelfcalStep("p", "6*IT", label="phase_6IT"),
        SelfcalStep("p", "3*IT", label="phase_3IT"),
        SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
    ))
    ```

    The string `"N*IT"` in a `solint` means "N times the average
    integration (dump) time" of the MS — AJISAI resolves it at runtime.

`amplitude: str = "never"`
:   The amplitude stage of `mode="data"`. Only `"never"` in this version;
    the conditional amplitude stage of the specification (antenna amplitude
    scatter times the EB SNR > 2, EB SNR >= 10, held-out and short-baseline
    flux checks; `'ap'` on top of the phase tables, normalised per
    observation) will be added later. For an unconditional amplitude step
    use `mode="fixed"`.

`SelfcalStep`
:   A single step of `mode="fixed"`. Fields:

    - `calmode`: `"p"` for phase, `"a"` for amplitude (which also
      preserves phase). Passed straight through to `gaincal`.
    - `solint`: solution interval, e.g. `"inf"`, `"60s"`, or `"6*IT"`.
    - `solnorm`: when `True`, the gain solutions are normalized to unit
      mean. Always set this for amplitude self-cal to preserve the
      absolute flux scale.
    - `label`: human-readable name used in the justification log and
      output filenames.

To run a custom fixed schedule:

```python
from ajisai import SelfcalSchedule, SelfcalStep

custom_schedule = SelfcalSchedule(mode="fixed", steps=(
    SelfcalStep("p", "inf",  label="phase_inf"),
    SelfcalStep("p", "30s",  label="phase_30s"),
    SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
))

cfg = AJISAIConfig(vis="...", schedule=custom_schedule)
```

### Measurement self-calibration for a calling pipeline

`ajisai.selfcal_measurement(vis, results_dir=..., imaging=..., gaincal=...,
refant=None, **options)` runs `mode="data"` without an amplitude stage on
one execution block, with defaults for a calling pipeline such as ASEBI
(`caller="asebi"`, `relabel_j2000_to_icrs=False`, no banner, and
`ImagingConfig(final_depth=())` unless an `ImagingConfig` is given), and
returns the finished `AJISAI`; `AJISAI.selfcal_summary()` gives the stage
decisions, gain tables, stage MSs, the best stage's image and model and
`final_selfcal.ms`.

## `PlotConfig`

Options for the diagnostic figures written at the end of `run()`. Only the
image gallery (`selfcal_images.png`, see [Output artifacts](outputs.md))
has options; the summary and reference-antenna figures do not.

`images_cmap: str = "inferno"`
:   Matplotlib colour map of the gallery.

`images_gamma: float = 1.0`
:   Power-law stretch of the top row (`1` = linear; values below 1 bring
    up faint emission).

`images_gamma2: Optional[float] = 0.3`
:   Stretch of a second row drawn below the first, with its own colour
    bar. `None` draws the top row only.

`images_fov_beam_factor: float = 10.0`
:   Half-width of the field shown, in units of the geometric-mean beam.

`images_fov_radius_arcsec: Optional[float] = None`
:   Half-width of the field shown in arcsec; overrides
    `images_fov_beam_factor` when set.

`images_center: str = "phase"`
:   `"phase"` centres the panels on the phase center (the reference
    pixel); `"peak"` on the brightest pixel of the last image.

```python
from ajisai import AJISAIConfig, PlotConfig

cfg = AJISAIConfig(vis="...", plots=PlotConfig(images_gamma2=0.5, images_fov_beam_factor=15))
```

A finished run can be re-plotted with other options, without CASA, using
`ajisai.plot_from_workdir("ajisai_<projname>", gamma2=0.5, fov_beam_factor=15)`.
