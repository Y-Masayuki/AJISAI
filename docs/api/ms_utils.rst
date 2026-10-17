``ajisai.ms_utils`` — measurement set queries
=============================================

.. automodule:: ajisai.ms_utils
   :no-members:

This module is a self-contained replacement for the small subset of
``analysisUtils`` functions AJISAI relies on. The functions here are
implemented natively on top of ``casatools`` and ``numpy``, with no
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

.. autofunction:: ajisai.ms_utils.get_uv_max

.. autofunction:: ajisai.ms_utils.cell_from_umax

.. autofunction:: ajisai.ms_utils.uv_max_from_maxima

.. autofunction:: ajisai.ms_utils.max_uv_of_unflagged_rows

.. autofunction:: ajisai.ms_utils.get_dish_diameter_m

.. autofunction:: ajisai.ms_utils.smallest_dish_diameter

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

.. autofunction:: ajisai.ms_utils.parse_ra_dec

.. autofunction:: ajisai.ms_utils.parse_ra

.. autofunction:: ajisai.ms_utils.parse_dec

.. autofunction:: ajisai.ms_utils.format_ra

.. autofunction:: ajisai.ms_utils.format_dec

.. autofunction:: ajisai.ms_utils.casa_direction_string

.. autofunction:: ajisai.ms_utils.convert_direction


Directions in the FIELD and SOURCE tables
-----------------------------------------

.. autofunction:: ajisai.ms_utils.direction_frames

.. autofunction:: ajisai.ms_utils.frame_of_code

.. autofunction:: ajisai.ms_utils.code_of_frame

.. autofunction:: ajisai.ms_utils.field_direction_frame

.. autofunction:: ajisai.ms_utils.field_id_by_name

.. autofunction:: ajisai.ms_utils.write_field_direction

.. autofunction:: ajisai.ms_utils.field_direction_offsets


Module-level helper for users
-----------------------------

.. autofunction:: ajisai.list_target_fields
