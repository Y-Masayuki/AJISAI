# Configuration reference

AJISAI's behavior is controlled by a hierarchy of frozen dataclasses. The
top-level container is [`AJISAIConfig`](#ajisaiconfig), which composes
sub-configs for imaging, gain calibration, and the self-cal schedule.

Pattern:

```python
from ajisai import AJISAI, AJISAIConfig, ImagingConfig, GainCalConfig, SelfcalSchedule, SelfcalStep

cfg = AJISAIConfig(
    vis     = "...",
    imaging = ImagingConfig(...),     # optional, default is fine
    gaincal = GainCalConfig(...),     # optional
    schedule= SelfcalSchedule(...),   # optional
)
AJISAI(cfg).run()
```

For the complete auto-generated reference, see
[API reference](api/index.rst). The page below highlights the parameters
you are most likely to want to override.

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
    where `frame` is `"ICRS"` or `"J2000"`. Example:

    ```python
    phase_center=("16:25:45.0", "-24:12:23.0", "ICRS")
    ```

    :::{note}
    `casatasks.fixplanets` only supports `"J2000"`, so AJISAI converts
    `"ICRS"` inputs to J2000 via astropy before calling fixplanets and
    restores the MS frame label to ICRS afterwards. This is transparent
    to the user.
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
    `"sigma_clip"`, `"annulus"` (legacy; requires target_radius), and
    `"mad"`.

`quality_metric: str = "dynamic_range"`
:   How AJISAI picks the best iteration at the end. Default is
    `dynamic_range = peak / RMS_offsource`, which is more robust than
    peak SNR for detecting over-cleaning.

`on_iter_anomaly: str = "log_only"`
:   What to do when an iteration looks anomalous (e.g. DR drop > 50%).
    Currently only `"log_only"` is implemented: anomalies are recorded
    in the justification log but do NOT alter pipeline flow. This
    preserves AJISAI's deterministic-fixed-schedule identity.

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

`threshold_factor: float = 1.0`
:   Threshold for CLEAN is `threshold_factor * RMS_offsource`. Set higher
    for cleaner stops, lower for deeper cleaning.

`deconvolver: str = "multiscale"`
:   tclean deconvolver. Multiscale recommended for extended sources.

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
:   tclean nterms (number of Taylor terms).

`cellpix: int = 10`
:   Number of pixels per synthesized beam used to set the cell size.
    Larger values give finer pixel sampling but larger images.

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

    `"user"`
    :   Use a pre-made mask file supplied via `user_mask`. This is the
        portable option for CASA 6.7+ users: prepare a mask externally
        (e.g. with CARTA, an older CASA installation, or any FITS-aware
        tool) and point AJISAI at it.

    `"none"`
    :   No mask at all. Rarely useful.

`user_mask: Optional[str] = None`
:   Path to a pre-made mask file. Required when `mask_mode="user"`.

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

`minsnr: float = 1.5`
:   Minimum SNR for a gain solution to be accepted. Solutions below this
    are discarded; the corresponding visibilities are preserved unflagged
    because AJISAI uses `applymode="calonly"`.

`gaintype: str = "T"`
:   gaincal gain type. `"T"` averages parallel-hand polarizations for
    improved SNR; `"G"` solves them independently.

`combine: str = "scan"`
:   Solutions are combined across scan boundaries (within a solution
    interval).

`gaincal_interp: str = "linear,linear"`
:   Interpolation in time and frequency for the gain table.

`applycal_interp: str = "linearPD"`
:   Interpolation for applycal. `"linearPD"` preserves phase unwrapping.

`applymode: str = "calonly"`
:   applycal mode. `"calonly"` ensures that flagged solutions do NOT
    propagate to flagging the visibilities themselves — important for
    preserving data on low-SNR antennas.

`calwt: bool = True`
:   Calibrate the weights when applying.

## `SelfcalSchedule`

By default the schedule is:

```python
SelfcalSchedule(steps=(
    SelfcalStep("p", "inf",  label="phase_inf"),
    SelfcalStep("p", "6*IT", label="phase_6IT"),
    SelfcalStep("p", "3*IT", label="phase_3IT"),
    SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
))
```

The string `"N*IT"` in a `solint` means "N times the average
integration (dump) time" of the MS — AJISAI resolves it at runtime.

`SelfcalStep`
:   A single self-cal step. Fields:

    - `calmode`: `"p"` for phase, `"a"` for amplitude (which also
      preserves phase). Passed straight through to `gaincal`.
    - `solint`: solution interval, e.g. `"inf"`, `"60s"`, or `"6*IT"`.
    - `solnorm`: when `True`, the gain solutions are normalized to unit
      mean. Always set this for amplitude self-cal to preserve the
      absolute flux scale.
    - `label`: human-readable name used in the justification log and
      output filenames.

To customize the schedule:

```python
from ajisai import SelfcalSchedule, SelfcalStep

custom_schedule = SelfcalSchedule(steps=(
    SelfcalStep("p", "inf",  label="phase_inf"),
    SelfcalStep("p", "30s",  label="phase_30s"),
    SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
))

cfg = AJISAIConfig(vis="...", schedule=custom_schedule)
```
