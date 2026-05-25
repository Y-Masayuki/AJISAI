"""
AJISAI - Automated Justification-based Imaging and Self-calibration for ALMA Infrastructure
============================================================================================

AJISAI is a fully automated, reproducible, and explainable self-calibration
pipeline for ALMA continuum data. It is designed around three principles:

    1. Fire-and-forget:    minimal user input; sensible defaults for all parameters.
    2. Justification-based: every parameter choice is logged to ``justification.json``
                           with a structured reason, so the pipeline is auditable
                           and the choices are reproducible in publications.
    3. Deterministic:      a fixed self-cal schedule (3 phase + 1 amp by default),
                           never adaptive rollback - all iterations complete and
                           the best by dynamic range is selected at the end.

Quick start
-----------
Run from within CASA (or a Python environment with modular CASA installed)::

    from ajisai import AJISAI, AJISAIConfig

    cfg = AJISAIConfig(vis="/path/to/data.ms")
    aj = AJISAI(cfg).run()

    print(aj.best_image)              # path to the best image
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
from .core import (
    # Main class and config
    AJISAI,
    AJISAIConfig,
    ImagingConfig,
    GainCalConfig,
    SelfcalStep,
    SelfcalSchedule,
    # MS helpers (re-exported for advanced use)
    list_target_fields,
    get_array_info,
    get_on_source_time,
    get_median_frequency,
    get_baseline_at_percentile,
    get_baseline_lengths,
    get_antenna_positions,
    get_antenna_flag_stats,
    pick_cell_imsize,
    rad_to_radec_from_imfit,
    icrs_to_j2000,
    # Image / RMS helpers
    load_fits_image,
    compute_rms,
    compute_image_stats,
    # Refant selection
    select_refant,
    plot_refant_selection,
    # Coordinate frame utility
    relabel_J2000_to_ICRS,
)

__version__ = "0.1.0"

__all__ = [
    "AJISAI",
    "AJISAIConfig",
    "ImagingConfig",
    "GainCalConfig",
    "SelfcalStep",
    "SelfcalSchedule",
    "list_target_fields",
    "get_array_info",
    "get_on_source_time",
    "get_median_frequency",
    "get_baseline_at_percentile",
    "get_baseline_lengths",
    "get_antenna_positions",
    "get_antenna_flag_stats",
    "pick_cell_imsize",
    "rad_to_radec_from_imfit",
    "icrs_to_j2000",
    "load_fits_image",
    "compute_rms",
    "compute_image_stats",
    "select_refant",
    "plot_refant_selection",
    "relabel_J2000_to_ICRS",
    "__version__",
]
