AJISAI
======

**Automated Justification-based Imaging and Self-calibration for ALMA Infrastructure**

AJISAI is a fully automated, reproducible, and explainable self-calibration
pipeline for ALMA continuum data. It is designed around three principles:

1. **Fire-and-forget.** Minimum required input is the measurement set path;
   all other parameters have sensible defaults derived from the data itself.

2. **Justification-based.** Every parameter choice (reference antenna,
   cell size, image size, solution intervals, masking strategy, ...) is
   recorded in a structured ``justification.json`` file with the rationale,
   so the run is auditable and the choices are reproducible in publications.

3. **Deterministic.** A fixed self-calibration schedule (three phase
   iterations plus one amplitude iteration by default) runs to completion;
   the best image is selected by dynamic range at the end. There is no
   adaptive rollback that would make the result depend on run order.

.. note::
   AJISAI requires CASA at runtime (either a monolithic CASA distribution or
   modular CASA installed via pip). See :doc:`installation` for details.


Minimal example
---------------

.. code-block:: python

   from ajisai import AJISAI, AJISAIConfig

   cfg = AJISAIConfig(vis="/path/to/data.ms")
   aj = AJISAI(cfg).run()

   print(aj.best_image)              # path to the best CLEAN image
   print(aj.best_metric_value)       # achieved dynamic range
   print(aj.justification)           # full structured rationale (also JSON on disk)


.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   tutorials/index
   configuration
   outputs

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Background

   design


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Citation
--------

If AJISAI helps your work, please cite (Yamaguchi et al., in prep.) and the
software itself via the Zenodo DOI (to be assigned at the first GitHub release).


License
-------

AJISAI is released under the MIT License. See the ``LICENSE`` file in the
repository for the full text.
