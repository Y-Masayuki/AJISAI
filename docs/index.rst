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

If AJISAI helps your work, please cite both the paper describing the method
and the specific version of the software you used.

**Paper** (the self-calibration procedure is described in Appendix B):
Yamaguchi, M., Machida, M. N., Tominaga, R. T., et al. 2026, ApJ, 1006, 232,
`doi:10.3847/1538-4357/ae819b <https://doi.org/10.3847/1538-4357/ae819b>`_

**Software** (v0.2.1):
Yamaguchi, M. 2026, AJISAI: Automated Justification-based Imaging and
Self-calibration for ALMA Infrastructure, v0.2.1, Zenodo,
`doi:10.5281/zenodo.23094438 <https://doi.org/10.5281/zenodo.23094438>`_

Each release has its own version DOI on Zenodo; cite the one matching the
version you used. The concept DOI
`10.5281/zenodo.23053992 <https://doi.org/10.5281/zenodo.23053992>`_ always
resolves to the latest release. You can also use the "Cite this repository"
button on GitHub, which reads ``CITATION.cff``.

.. code-block:: bibtex

   @article{yamaguchi2026v1094sco,
     author  = {Yamaguchi, Masayuki and Machida, Masahiro N. and Tominaga, Ryosuke T. and Sai, Jinshi and Muto, Takayuki and Takami, Michihiro and Liu, Hauyu Baobab and Shoshi, Ayumu and Tsukagoshi, Takashi and Ishibashi, Shu},
     title   = {A Hybrid Origin for the Multiple Ring-gap Structures in the Large Protoplanetary Disk V1094 Sco: A Low-mass Planet and Secular Gravitational Instability},
     journal = {The Astrophysical Journal},
     year    = {2026},
     volume  = {1006},
     pages   = {232},
     doi     = {10.3847/1538-4357/ae819b}
   }

   @software{yamaguchi2026ajisai,
     author    = {Yamaguchi, Masayuki},
     title     = {{AJISAI}: Automated Justification-based Imaging and Self-calibration for {ALMA} Infrastructure},
     version   = {v0.2.1},
     publisher = {Zenodo},
     year      = {2026},
     doi       = {10.5281/zenodo.23094438},
     url       = {https://doi.org/10.5281/zenodo.23094438}
   }


License
-------

AJISAI is released under the MIT License. See the ``LICENSE`` file in the
repository for the full text.
