Surrogate Studies
=================

These are user-facing examples for building surrogate models with MatCal.
They are intended to be copied, modified, and used as templates for users'
own studies. Each example is self-contained and shows the complete workflow
from defining parameters and a high-fidelity model through training, testing,
and interrogating a surrogate.

Example problem
---------------

The examples use a simple 2D thermal boundary-value problem motivated by a
foam/metal layered component exposed to a high-temperature environment. A foam
layer separates two steel layers. The top steel layer is heated by a far-field
radiative source and by convection from hot surrounding gas.
The steel layers are 1 cm thick and the foam layer is 1.5 cms thick. 
They are all 2.5 cms wide. The uncertain model
inputs are:

* convective heat-transfer coefficient, ``H``;
* far-field source temperature, ``T_inf``; and
* surrounding air temperature, ``T_air``.

The surrogate models predict thermocouple temperature histories in the layered
component. The ``TC_top`` thermocouple is at the junction between the 
foam and steel near the top and the ``TC_bottom`` thermocouple 
is at the lower foam/steel junction. 
The examples use a common deterministic Halton test set so that the
standard and adaptive surrogate approaches can be compared quantitatively.

A schematic of the boundary value problem is shown below.

.. image:: boundary_value_problem.png
   :alt: Schematic of the layered material boundary value problem
   :align: center
   :width: 600px

Examples
--------

``plot_generate_surrogate_a_standard.py``
    Builds a standard PCA-based surrogate from a Latin-hypercube training study.
    This example predicts both ``TC_top`` and ``TC_bottom``. Near the end, it
    scores the surrogate on the common Halton test set used by the adaptive
    examples.

``plot_generate_surrogate_b1_adaptive_voronoi_GP.py``
    Builds a Voronoi adaptive surrogate for ``TC_bottom`` using a Gaussian
    Process backend. The adaptive sampling loop adds training points in regions
    where cross-validation indicates the surrogate needs improvement.

``plot_generate_surrogate_b2_adaptive_voronoi_RBF.py``
    Builds a Voronoi adaptive surrogate for ``TC_bottom`` using an RBF backend
    instead of a Gaussian Process backend. This example is useful when users want
    a deterministic local interpolator without Gaussian-process predictive
    uncertainty.

``plot_generate_surrogate_c_adaptive_sparse_grid.py``
    Builds a PyApprox sparse-grid adaptive surrogate for ``TC_bottom``. This
    example demonstrates an alternative adaptive strategy that can be effective
    for smooth responses in moderate-dimensional parameter spaces.


Template usage
--------------

To use these examples as templates for a new study:

#. Replace the ``Parameter`` definitions with the uncertain inputs for the new
   model.
#. Replace the SIERRA/Aria model definition with the user's high-fidelity model.
#. Update the independent variable and prediction locations, such as time,
   displacement, or spatial position.
#. Update the target response fields, such as temperature, force, stress, or
   displacement.
#. Choose a surrogate training strategy:
   
   * standard LHS sampling for a straightforward baseline;
   * Voronoi adaptive sampling for response-driven point placement;
   * sparse-grid adaptivity for smooth responses; or
   * Voronoi adaptive sampling with an RBF backend for deterministic local
     interpolation.

#. Keep the common-test-set pattern if comparing multiple surrogate approaches.
   The test set should be independent of the validation points used for plotting
   example error curves.