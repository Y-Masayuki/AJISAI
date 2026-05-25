``ajisai`` — main package
=========================

.. automodule:: ajisai
   :no-members:

Public API
----------

The ``ajisai`` top-level package re-exports the names listed below from
``ajisai.core`` and ``ajisai.ms_utils``. They are the supported public
interface.

Main class and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ajisai.AJISAI
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: ajisai.AJISAIConfig
   :members:
   :exclude-members: __init__
   :show-inheritance:

.. autoclass:: ajisai.ImagingConfig
   :members:
   :exclude-members: __init__
   :show-inheritance:

.. autoclass:: ajisai.GainCalConfig
   :members:
   :exclude-members: __init__
   :show-inheritance:

.. autoclass:: ajisai.SelfcalStep
   :members:
   :exclude-members: __init__
   :show-inheritance:

.. autoclass:: ajisai.SelfcalSchedule
   :members:
   :exclude-members: __init__
   :show-inheritance:


Image and RMS utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ajisai.load_fits_image

.. autofunction:: ajisai.compute_rms

.. autofunction:: ajisai.compute_image_stats


Reference-antenna selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ajisai.select_refant

.. autofunction:: ajisai.plot_refant_selection


Coordinate frame utility
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ajisai.relabel_J2000_to_ICRS


``ajisai.core`` — internal pipeline implementation
---------------------------------------------------

The full implementation lives in ``ajisai.core``. Most users should use
the top-level ``ajisai`` namespace above. The members below are provided
for advanced use (subclassing :class:`AJISAI` to override strategy
methods, calling pipeline steps individually, etc.).

.. automodule:: ajisai.core
   :members:
   :undoc-members: False
   :show-inheritance:
   :private-members: False
   :exclude-members: AJISAI, AJISAIConfig, ImagingConfig, GainCalConfig,
                     SelfcalStep, SelfcalSchedule, load_fits_image,
                     compute_rms, compute_image_stats, select_refant,
                     plot_refant_selection, relabel_J2000_to_ICRS
