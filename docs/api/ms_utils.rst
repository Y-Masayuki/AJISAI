``ajisai.ms_utils`` — measurement set queries
=============================================

.. automodule:: ajisai.ms_utils
   :no-members:

This module is a self-contained replacement for the small subset of
``analysisUtils`` functions AJISAI relies on. The functions here are
implemented natively on top of ``casatools`` and ``astropy``, with no
dependency on the 110,000-line analysisUtils source.

The math-only portions (``_smallest_5_smooth_at_least``,
``_round_to_sig_figs``, ``_baars_taper_factor``,
``_parse_uvtaper_to_image_fwhm``, etc.) are validated against the
analysisUtils golden values for the V883 Ori dataset. See
``_self_check_math()`` for the test suite.


MS metadata
-----------

.. autofunction:: ajisai.ms_utils.get_array_info

.. autofunction:: ajisai.ms_utils.get_on_source_time

.. autofunction:: ajisai.ms_utils.get_median_frequency


Baseline statistics
-------------------

.. autofunction:: ajisai.ms_utils.get_baseline_lengths

.. autofunction:: ajisai.ms_utils.get_baseline_at_percentile


Cell size and image size
------------------------

.. autofunction:: ajisai.ms_utils.pick_cell_imsize

.. autofunction:: ajisai.ms_utils.test_pick_cell_imsize


Antenna queries
---------------

.. autofunction:: ajisai.ms_utils.get_antenna_positions

.. autofunction:: ajisai.ms_utils.get_antenna_flag_stats


Coordinate utilities
--------------------

.. autofunction:: ajisai.ms_utils.rad_to_radec

.. autofunction:: ajisai.ms_utils.rad_to_radec_from_imfit

.. autofunction:: ajisai.ms_utils.icrs_to_j2000


Module-level helper for users
-----------------------------

.. autofunction:: ajisai.list_target_fields
