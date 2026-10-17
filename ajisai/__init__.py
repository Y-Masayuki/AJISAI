"""
AJISAI - Automated Justification-based Imaging and Self-calibration for ALMA Infrastructure
============================================================================================

AJISAI is a fully automated, reproducible, and explainable self-calibration
pipeline for ALMA continuum data. It is designed around three principles:

    1. Fire-and-forget:    minimal user input; sensible defaults for all parameters.
    2. Justification-based: every parameter choice is logged to ``justification.json``
                           with a structured reason, so the pipeline is auditable
                           and the choices are reproducible in publications.
    3. Data-driven:        the phase self-cal stages and their solution intervals
                           are decided from quantities measured in the data and
                           checked on held-out integrations (schedule.mode="data",
                           default); the fixed v0.2 ladder remains as
                           schedule.mode="fixed".
    4. Read-only input:    the input MS is never written; every stage works on
                           copies, and each stage's MS is kept.

Quick start
-----------
Run from within CASA (or a Python environment with modular CASA installed)::

    from ajisai import AJISAI, AJISAIConfig

    cfg = AJISAIConfig(vis="/path/to/data.ms")
    aj = AJISAI(cfg).run()

    print(aj.best_image)              # image of the last accepted stage
    print(aj.justification)           # full structured rationale

Multi-field workflow
--------------------
For measurement sets containing multiple target fields::

    from ajisai import AJISAI, AJISAIConfig, list_target_fields

    vis = "/path/to/multi_field.ms"
    for f in list_target_fields(vis):
        cfg = AJISAIConfig(vis=vis, field=f["id"], projname=f["name"])
        AJISAI(cfg).run()

License: MIT
Author:  Masayuki Yamaguchi (Kyushu Univ./NAOJ) and the AJISAI development team
"""
# Re-exports from the implementation modules. Names live here in __init__.py
# so that ``from ajisai import X`` works regardless of the underlying file.
from .core import (
    # Main class and config
    AJISAI,
    AJISAIConfig,
    ImagingConfig,
    GainCalConfig,
    SelfcalStep,
    SelfcalSchedule,
    PlotConfig,
    FIXED_SCHEDULE_STEPS,
    # Entry point for a calling pipeline's measurement self-cal (ASEBI)
    selfcal_measurement,
    # MS helpers that the orchestrator uses internally
    list_target_fields,
    # Image / RMS helpers
    load_fits_image,
    compute_rms,
    compute_image_stats,
    # Refant selection
    select_refant,
    # Coordinate frame utility
    relabel_J2000_to_ICRS,
    # Self-cal bookkeeping and the final-image depth rule (pure functions)
    SOLVE_INFO_KEYS,
    final_depth_decision,
    aperture_flux_terms,
)
# Data-driven solution intervals (pure numpy; see ajisai.solint).
from . import solint
# Diagnostic figures (CASA-independent; see ajisai.plotting).
from .plotting import (
    plot_refant_selection,
    plot_selfcal_summary,
    plot_selfcal_images,
    plot_from_workdir,
)

# Helpers re-exported directly from ms_utils for advanced use.
from .ms_utils import (
    get_array_info,
    get_on_source_time,
    get_median_frequency,
    get_baseline_at_percentile,
    get_baseline_lengths,
    get_antenna_positions,
    get_antenna_flag_stats,
    get_dish_diameter_m,
    get_uv_max,
    cell_from_umax,
    pick_cell_imsize,
    rad_to_radec_from_imfit,
    icrs_to_j2000,
)

__version__ = "0.3.0.dev0"

__all__ = [
    "AJISAI",
    "AJISAIConfig",
    "ImagingConfig",
    "GainCalConfig",
    "SelfcalStep",
    "SelfcalSchedule",
    "PlotConfig",
    "FIXED_SCHEDULE_STEPS",
    "selfcal_measurement",
    "solint",
    "list_target_fields",
    "get_array_info",
    "get_on_source_time",
    "get_median_frequency",
    "get_baseline_at_percentile",
    "get_baseline_lengths",
    "get_antenna_positions",
    "get_antenna_flag_stats",
    "get_dish_diameter_m",
    "get_uv_max",
    "cell_from_umax",
    "pick_cell_imsize",
    "rad_to_radec_from_imfit",
    "icrs_to_j2000",
    "load_fits_image",
    "compute_rms",
    "compute_image_stats",
    "select_refant",
    "plot_refant_selection",
    "plot_selfcal_summary",
    "plot_selfcal_images",
    "plot_from_workdir",
    "relabel_J2000_to_ICRS",
    "SOLVE_INFO_KEYS",
    "final_depth_decision",
    "aperture_flux_terms",
    "__version__",
]
