"""
Full configuration template for AJISAI.

Every option is listed with its default value, so running this file as it is
(after setting ``vis``) does exactly what ``AJISAI(AJISAIConfig(vis=...)).run()``
does. Change only the lines you need; the allowed values are given in the
comments. Details: https://ajisai.readthedocs.io/en/latest/configuration.html

Run with the Python that has CASA (monolithic CASA's python3, or a Python with
modular CASA), or from inside the CASA prompt with
``exec(open("full_config_template.py").read())``.
"""
from ajisai import (
    AJISAI,
    AJISAIConfig,
    GainCalConfig,
    ImagingConfig,
    PlotConfig,
    SelfcalSchedule,
    SelfcalStep,
)

# --- Input data -------------------------------------------------------------
vis = "/path/to/data.ms"       # REQUIRED: a continuum-ready measurement set
field = None                   # FIELD_ID as a string ("0") or field name; None = first target field
projname = None                # None = derived from the basename of vis
results_dir = None             # None = ./ajisai_<projname>/

# --- Imaging (tclean) -------------------------------------------------------
imaging = ImagingConfig(
    weighting="briggs",
    robust=0.5,
    uvtaper=(),                # e.g. ("1arcsec",), ("100klambda",), ("2arcsec", "0.5arcsec", "30deg")
    cellpix=10,                # pixels per synthesized beam; sets the cell size
    deconvolver="multiscale",  # scales [0, 1, 3] x beam are derived from the data
    nterms=1,
    gridder="standard",
    pbcor=False,
    parallel=False,            # mpicasa
    niter=50000,
    gain=0.05,
    cyclefactor=1.5,
    threshold_factor=1.0,      # CLEAN threshold = threshold_factor x off-source RMS
    # Mask: "auto-multithresh" (default, any CASA) | "interactive" (CASA <= 6.6 only)
    #       | "user" (pre-made mask, set user_mask) | "none"
    mask_mode="auto-multithresh",
    user_mask=None,
    # auto-multithresh tunables (ALMA pipeline values for continuum)
    sidelobethreshold=2.0,
    noisethreshold=4.25,
    lownoisethreshold=1.5,
    minbeamfrac=0.3,
    growiterations=75,
)

# --- Gain calibration (gaincal / applycal) ----------------------------------
gaincal = GainCalConfig(
    gaintype="T",              # "T" averages the parallel hands for SNR | "G" solves them separately
    minsnr=1.5,                # solutions below this SNR are dropped (data stay unflagged: calonly)
    minblperant=4,
    combine="scan",
    gaincal_interp="linear,linear",
    applycal_interp="linearPD",
    applymode="calonly",       # flagged solutions never flag the visibilities
    calwt=True,
)

# --- Self-calibration schedule (fixed; every step always runs) --------------
# calmode: "p" (phase) | "a" (amplitude; phase is preserved)
# solint:  any CASA solint ("inf", "60s", ...) or "N*IT" = N x average integration time
# solnorm: True for amplitude steps, to preserve the flux scale
schedule = SelfcalSchedule(steps=(
    SelfcalStep("p", "inf",  label="phase_inf"),
    SelfcalStep("p", "6*IT", label="phase_6IT"),
    SelfcalStep("p", "3*IT", label="phase_3IT"),
    SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
))

# --- Diagnostic figures (selfcal_images.png) --------------------------------
plots = PlotConfig(
    images_cmap="inferno",
    images_gamma=1.0,          # top row: power-law stretch (1 = linear)
    images_gamma2=0.3,         # second row; None draws one row only
    images_fov_beam_factor=10.0,    # half-width of the field shown = N x beam
    images_fov_radius_arcsec=None,  # overrides images_fov_beam_factor
    images_center="phase",     # "phase" (phase center) | "peak" (brightest pixel)
)

cfg = AJISAIConfig(
    vis=vis,
    field=field,
    projname=projname,
    results_dir=results_dir,
    imaging=imaging,
    gaincal=gaincal,
    schedule=schedule,
    plots=plots,
    # Phase shift (off by default). With phase_center=None the brightest pixel of the
    # dirty image becomes the phase center; or give (ra, dec, frame), frame "ICRS" or
    # "J2000", e.g. ("16:25:45.0", "-24:12:23.0", "ICRS").
    phase_shift=False,
    phase_center=None,
    # Reference antenna: "hybrid" (default) | "geometric_center" | "flag_stats"
    #                    | "manual" (set refant_manual)
    refant_strategy="hybrid",
    refant_manual=None,
    refant_flag_threshold=0.25,     # antennas with a larger flagged fraction are excluded
    # Off-source RMS: "sigma_clip_excl" (default) | "sigma_clip" | "mad"
    rms_method="sigma_clip_excl",
    rms_exclude_beam_factor=5.0,    # sigma_clip_excl: exclude a central circle of N x beam
    rms_sigma=3.0,
    rms_maxiters=5,
    # Best image: "dynamic_range" (default) | "peak_snr"
    quality_metric="dynamic_range",
    on_iter_anomaly="log_only",     # anomalies are logged and never change the schedule
    verbose=True,
    show_banner=True,
)

aj = AJISAI(cfg).run()
print(aj.best_image, aj.best_metric_value)
