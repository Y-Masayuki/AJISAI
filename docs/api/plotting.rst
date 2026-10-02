``ajisai.plotting`` — diagnostic figures
========================================

.. automodule:: ajisai.plotting
   :no-members:

The three functions below are what ``run()`` calls at the end of a
pipeline; they are also importable from the top-level ``ajisai`` namespace.
See :doc:`../outputs` for what each figure shows and
:class:`ajisai.PlotConfig` for the options that ``run()`` passes through.


Figures written by ``run()``
----------------------------

.. autofunction:: ajisai.plotting.plot_selfcal_summary

.. autofunction:: ajisai.plotting.plot_refant_selection

.. autofunction:: ajisai.plotting.plot_selfcal_images


Gallery panels
--------------

.. autofunction:: ajisai.plotting.panels_from_metrics

.. autofunction:: ajisai.plotting.panels_from_fits_dir


Re-plotting an existing run
---------------------------

.. autofunction:: ajisai.plotting.plot_from_workdir


Lower-level pieces
------------------

.. autofunction:: ajisai.plotting.draw_refant_selection

.. autofunction:: ajisai.plotting.round_labels

.. autofunction:: ajisai.plotting.format_solint

.. autodata:: ajisai.plotting.PLOT_RC
   :no-value:

.. autodata:: ajisai.plotting.STATUS_COLORS
