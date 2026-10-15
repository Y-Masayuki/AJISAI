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
# Every tclean of a run (dirty, initial CLEAN, each iteration, final image) uses
# the same cell, imsize, nterms, reffreq, scales and mask.
imaging = ImagingConfig(
    weighting="briggs",
    robust=0.5,
    uvtaper=(),                # e.g. ("1arcsec",), ("100klambda",), ("2arcsec", "0.5arcsec", "30deg")
    # Image geometry: None = derived from the data.
    cell=None,                 # e.g. "0.0062arcsec" or "6.2mas"
    imsize=None,               # N or (Nx, Ny); None = primary beam (smallest dish) down to 20%
    cell_rule="umax",          # "umax": 1/(5 u_max), longest unflagged baseline at the highest
                               # frequency | "percentile" (legacy v0.2: 90th-percentile beam / cellpix)
    cellpix=10,                # pixels per beam; cell_rule="percentile" only
    deconvolver="multiscale",  # "mtmfs" is used automatically when nterms > 1
    nterms=1,                  # > 1: mtmfs; statistics on .image.tt0
    reffreq=None,              # e.g. "227.0GHz"; None = middle of the frequency range
    scales=None,               # pixels; None = 0, b, 3b, 9b, ... (b = beam) up to the largest
                               # that fits the user mask region (0, b, 3b without one)
    gridder="standard",
    pbcor=False,
    parallel=False,            # mpicasa
    niter=50000,
    gain=0.05,
    cyclefactor=1.5,
    # CLEAN depth of the self-cal images:
    #   "pure_noise": threshold_nsigma x sigma_th (absolute; sigma_th from the (XX-YY)/2 dirty image)
    #   "image_rms" (legacy v0.2): threshold_factor x off-source RMS of the previous image
    threshold_mode="pure_noise",
    threshold_nsigma=3.0,
    threshold_factor=1.0,
    # Final image of the best iteration: CLEAN 3 -> 2 -> 1 -> 0.5 sigma_th until the aperture
    # flux converges; () = no final image (final.* = best iteration's image, as in v0.2).
    final_depth=(3.0, 2.0, 1.0, 0.5),
    final_aperture_arcsec=None,     # None = the CLEAN mask region (5 beams without a mask)
    # Mask: "auto-multithresh" (default, any CASA) | "interactive" (CASA <= 6.6 only; refused
    #       when caller="asebi") | "user" (CRTF region file or mask image, set user_mask) | "none"
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
    gaintype="T",              # "T" averages the parallel hands for SNR | "G" (mode="fixed" only)
    minsnr=3.0,                # floor; mode="data" gates on a predicted SNR of 5 (never raise to 5)
    minblperant=4,
    combine="scan",            # mode="data": only "spw" is used ("scan,spw" combines the spws)
    gaincal_interp="linear,linear",
    applycal_interp="linearPDperobs",  # "perobs": never interpolate across observations
    applymode="calonly",       # flagged solutions never flag the visibilities
    calwt=False,               # self-cal tables never change WEIGHT
    spwmap=(),                 # e.g. (0, 0, 0, 0) when combine includes "spw"
)

# --- Self-calibration stages -------------------------------------------------
# mode="data" (default): phase stages decided from the data (ASEBI SPEC v0.3 10.2):
#   one solution per EB, then intervals chosen from the measured solution SNR and phase
#   structure function and accepted on held-out integrations; at most three stages.
#   steps are not used. amplitude: "never" (the conditional amplitude stage comes later).
# mode="fixed" (v0.2): every step of steps runs; best by quality_metric.
#   calmode: "p" (phase) | "a" (amplitude; phase is preserved)
#   solint:  any CASA solint ("inf", "60s", ...) or "N*IT" = N x average integration time
#   solnorm: True for amplitude steps, to preserve the flux scale
schedule = SelfcalSchedule(
    mode="data",
    amplitude="never",
    steps=(                    # mode="fixed" only (this is the v0.2 ladder)
        SelfcalStep("p", "inf",  label="phase_inf"),
        SelfcalStep("p", "6*IT", label="phase_6IT"),
        SelfcalStep("p", "3*IT", label="phase_3IT"),
        SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
    ),
)

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
    #                 | "annulus" (legacy; requires rms_target_radius_arcsec)
    rms_method="sigma_clip_excl",
    rms_exclude_beam_factor=5.0,    # sigma_clip_excl: exclude a central circle of N x beam
    rms_target_radius_arcsec=None,  # annulus: inner radius in arcsec (must enclose the source)
    rms_sigma=3.0,
    rms_maxiters=5,
    # Best iteration of mode="fixed": "dynamic_range" (default) | "peak_snr"
    # (mode="data": the last accepted stage; image statistics are diagnostics)
    quality_metric="dynamic_range",
    on_iter_anomaly="log_only",     # anomalies are logged and never change the schedule
    # The input MS is never written. With J2000 frame codes in the input, a relabelled
    # (J2000 -> ICRS, frame code only) copy is used when this is True.
    relabel_j2000_to_icrs=True,
    caller="user",                  # "user" | "asebi" (refuses interactive masking)
    verbose=True,
    show_banner=True,
)

aj = AJISAI(cfg).run()
print(aj.best_image, aj.best_metric_value)
print(aj.final_image, aj.final_ms)
