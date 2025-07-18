Optimization Overview
=====================

A generic material model :math:`\model` has parameters :math:`\parameters` and 
produces outputs (e.g., stress :math:`\stress`) as a function inputs (e.g. strain :math:`\strain` ):

.. math::
    :label: eq_model

    \stress = \model\left(\strain; \parameters\right) .

Calibration determines optimal parameters given: (a) experimental data and 
(b) an objective that indicates how close the model is to the data.
The observable experimental data is rarely at the level of the 
output of the model in :eq:`eq_model` (e.g. stresses at points).
Instead, the model needs to be embedded in a simulator of a boundary 
value problem (BVP) that emulates the experiment and gives the observables :math:`\observables`

.. math::
    :label: eq_bvp

    \observables = \qois\left(\controls;\parameters\right),

as a function of the experimental conditions/controls :math:`\controls` (such as temperature or applied velocities)
which we call *states*.
The dependence of the *quantities of interest* :math:`\qois` on 
the parameters :math:`\parameters` comes via the dependence of the boundary value 
problem on the selected model :math:`\model`.

The (calibration) objective compares the quantities of interest :math:`\qois`
with experimental data :math:`\hat{\qois}`.
Geometrically an objective can be thought of a distance, for example the mean squared error

.. math::
    :label: eq_objective

    {\Phi} = \left| \qois - \hat{\qois} \right|^2

where :math:`\left| \cdot \right|` is the L2-norm for functions or L2-norm for discrete data. 
The difference :math:`\qois - \hat{\qois}` is called the *residual*.
With the chosen objective :math:`\objective`, the calibration problem to find 
the optimal parameters :math:`\parameters^*` becomes:

.. math::
    :label: eq_calibration_min

    \parameters^* = \argmin_\parameters \objective,

where the minimization can be accomplished by a variety of gradient and/or global optimization techniques.

.. Typically, we use the :math:`\observables` produced by the experiment and the model (through the BVP).

A weighted least squares objective, such as 

.. math::
    :label: eq_weighted_least_squares

    \Phi = \sum_i \left|  \hat{\qois}_i - \qois_i \right|^2 w_i

(here for the discrete data case), gives control over what data or features 
in the data are emphasized in the calibration.
For instance, setting the weights :math:`w_i` equal to measurement
noise variance for each point :math:`i` guides the fit to be closer 
to data points with less uncertainty.
Alternately, a large weight for a particular point or set of points in
the experimental data can emphasize the importance of a feature such as yield.
Mixing residuals from different quantities of interest presents a scaling/conditioning
problem which can be handled by scaling the residuals for each dataset :math:`I` separately 

.. math::
    :label: eq_multi_data_least_squares

    \Phi = \sum_I s_I \left( \sum_i \left|  \hat{\qois}_{Ii} - \qois_{Ii} \right|^2 w_i \right)

with a scale factor :math:`s_I` per dataset :math:`I` so each residual is on par with the others.

A number of common issues complicate the calibration and are worth mentioning:

    #. Noise in the data 
    #. Discrepancy between the model and the data (the model cannot fit the data perfectly)
    #. Non-identifiability of the parameters (combinations of parameters produce identical output and/or the data is too simple to determine all the parameter uniquely)

These issues are investigated further in :ref:`Introduction Examples` where potential solutions to handling them are demonstrated.
More details on specifics to optimization can be found in :cite:p:`luenberger1984linear`
and a book with relevance for using optimization for calibration and inverse problems 
is :cite:p:`sun_model_cal`.