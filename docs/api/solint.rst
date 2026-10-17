``ajisai.solint`` — data-driven solution intervals
==================================================

.. automodule:: ajisai.solint
   :no-members:

Pure numpy (no CASA at import). The rules and every number come from the
ASEBI specification v0.3, section 10.2 and appendices C.2-C.3; each function
names its section in a ``Spec:`` line. :doc:`../design` explains the method.


Rules and the loss table
------------------------

.. autodata:: ajisai.solint.LOSS_TABLE_SNR
.. autodata:: ajisai.solint.LOSS_TABLE_NEG_LN_C
.. autodata:: ajisai.solint.G1_SNR_MIN
.. autodata:: ajisai.solint.ACCEPT_FLOOR
.. autodata:: ajisai.solint.MAX_PHASE_STAGES

.. autofunction:: ajisai.solint.casa_loss


Closed forms
------------

.. autofunction:: ajisai.solint.box_coefficient
.. autofunction:: ajisai.solint.atmospheric_coefficient
.. autofunction:: ajisai.solint.residual_variance_closed
.. autofunction:: ajisai.solint.optimal_solint
.. autofunction:: ajisai.solint.variance_ratio


Measured quantities
-------------------

.. autoclass:: ajisai.solint.VisSummary
   :members: y0, track_phase

.. autofunction:: ajisai.solint.reduce_visibilities
.. autofunction:: ajisai.solint.geometry
.. autofunction:: ajisai.solint.noise_scale
.. autofunction:: ajisai.solint.snr_rates
.. autofunction:: ajisai.solint.lag_structure_function
.. autofunction:: ajisai.solint.fit_structure_function
.. autofunction:: ajisai.solint.kappa_geometry


Candidates, G1 and the prediction
---------------------------------

.. autofunction:: ajisai.solint.candidate_solints
.. autofunction:: ajisai.solint.solution_units
.. autofunction:: ajisai.solint.g1_gate
.. autofunction:: ajisai.solint.caltable_gate
.. autofunction:: ajisai.solint.predict_residual_variance


G2, the rules and the decision
------------------------------

.. autofunction:: ajisai.solint.solve_phases
.. autofunction:: ajisai.solint.make_splits
.. autofunction:: ajisai.solint.holdout_coherence
.. autofunction:: ajisai.solint.loss_correction
.. autofunction:: ajisai.solint.corrected_improvement
.. autofunction:: ajisai.solint.rule_a
.. autofunction:: ajisai.solint.rule_b
.. autofunction:: ajisai.solint.choose_solint
.. autofunction:: ajisai.solint.acceptance
.. autofunction:: ajisai.solint.evaluate_stage
.. autofunction:: ajisai.solint.combine_spw_decision
.. autofunction:: ajisai.solint.g3_short_baseline_ratio
