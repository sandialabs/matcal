Surrogate Verification
======================

These examples verify several surrogate modeling workflows available in MatCal.
The examples use the two-dimensional Peaks benchmark function from the
KFCV-Voronoi adaptive sampling paper by :cite:`voronoi_adaptive_surrogates`. 
This set of examples is meant to demonstrate features and behavior of the 
surrogates in MatCal. They are not meant to be used as templates for user MatCal files.

The Peaks function is a useful verification problem because most of the domain
is relatively smooth and low magnitude, while the important response behavior is
localized near a small region of the parameter space. This makes it a good test
for whether a sampling method can discover and refine important local features.

The verification examples compare three surrogate workflows:

* a pretrained RBF surrogate built from random samples;
* an adaptive Voronoi RBF surrogate;
* an adaptive sparse-grid surrogate.

Each example reports validation RMSE, validation maximum absolute error, and an
empirical convergence rate. The convergence rate is estimated by fitting the
power law

.. math::

   E(N) \approx C N^{-p},

where ``E`` is the validation error, ``N`` is the number of training samples,
``C`` is a fitted constant, and ``p`` is the observed convergence rate. Larger
values of ``p`` indicate faster observed convergence.

Verification Results
--------------------

The following results were obtained for the Peaks verification examples.

.. list-table:: Peaks surrogate verification results
   :header-rows: 1
   :widths: 32 14 18 22 18 18

   * - Method
     - Final samples
     - Final RMSE
     - Final max error
     - RMSE rate ``p``
     - Max-error rate ``p``
   * - Pretrained random-sampling RBF
     - 150
     - 5.606465e-01
     - 4.819553e+00
     - 0.3639
     - 0.1775
   * - Adaptive Voronoi RBF
     - 150
     - 7.177528e-02
     - 4.102925e-01
     - 1.4752
     - 1.7587
   * - Adaptive sparse grid
     - 769
     - 5.580570e-02
     - 3.334068e-01
     - 0.6893
     - 0.6184

The best maximum-error result for the adaptive Voronoi RBF case occurred before
the final sample count. In that run, the best maximum absolute error was

.. math::

   3.519908 \times 10^{-1}

at 123 samples. This illustrates why the adaptive surrogate object stores both
the error history and the best retained surrogate.

Discussion
----------

The pretrained random-sampling RBF surrogate has the weakest convergence in
this study. This is expected. Random sampling does not use any response
information when selecting training points. It has no knowledge of where the
Peaks function changes rapidly or where the most important local features are.
As a result, many samples may be spent in regions that are easy to approximate,
while the localized high-gradient region remains under-resolved. This produces
the slowest observed convergence rates:

.. math::

   p_\mathrm{RMSE} \approx 0.36,
   \qquad
   p_\mathrm{max} \approx 0.18.

The adaptive sparse-grid surrogate performs substantially better than random
sampling. Sparse grids use a structured approximation space and adaptive
refinement, so they are more systematic than purely random sampling. However,
sparse-grid methods are still constrained by their grid-based structure. They
tend to perform best when the response is globally smooth or when important
features can be efficiently represented by the sparse-grid basis. The Peaks
function has a localized region of strong response variation, so the sparse grid
must spend samples refining a structured grid representation of a local feature.
This is less direct than placing samples specifically in the high-error local
region. The observed sparse-grid convergence rates,

.. math::

   p_\mathrm{RMSE} \approx 0.69,
   \qquad
   p_\mathrm{max} \approx 0.62,

are better than random sampling but slower than the adaptive Voronoi RBF case.

The adaptive Voronoi RBF surrogate performs best for this particular function
when compared at similar sample counts. This behavior is expected for the Peaks
benchmark. The KFCV-Voronoi adaptive sampling method uses cross-validation error
to identify regions where the surrogate is performing poorly, then uses Voronoi
cell geometry to place new samples away from existing samples in those important
regions. Because the Peaks function has localized difficult behavior, this
sampling strategy can concentrate samples where they are most valuable. The
result is much faster observed convergence:

.. math::

   p_\mathrm{RMSE} \approx 1.48,
   \qquad
   p_\mathrm{max} \approx 1.76.

The adaptive Voronoi RBF method also reaches much lower error than the
pretrained random-sampling RBF surrogate using the same final sample count of
150.

These results should not be interpreted as a universal ranking of surrogate
methods. The best surrogate and sampling strategy are highly dependent on the
function being approximated. A sparse-grid surrogate may be very effective for a
smooth response with distributed global variation. A random or space-filling
pretrained design may be adequate when the response is simple or when adaptive
sampling is not possible. Voronoi adaptive sampling is particularly attractive
when important response features are localized and expensive model evaluations
should be concentrated in high-error regions.

In general, users should select a surrogate workflow based on:

* expected smoothness of the response;
* whether important behavior is localized or distributed;
* number of input parameters;
* cost of each model evaluation;
* whether adaptive sampling is practical;
* desired accuracy metric, such as RMSE or maximum absolute error.

Example Summary
---------------

``plot_a1_peaks_pretrained_random_rbf.py``
    Builds pretrained RBF surrogates from nested random training sets. This
    example demonstrates the baseline behavior of a non-adaptive sampling
    strategy.

``plot_a2_peaks_adaptive_voronoi_rbf.py``
    Builds an adaptive RBF surrogate using KFCV-Voronoi sampling. This example
    demonstrates response-aware sampling for a function with localized important
    features.

``plot_a3_peaks_adaptive_sparse_grid.py``
    Builds an adaptive sparse-grid surrogate using PyApprox through MatCal's
    sparse-grid adaptive surrogate study. This example demonstrates a structured
    adaptive approximation method.

Each example produces a convergence plot and diagnostic plots showing training
sample locations over the true Peaks function and the corresponding surrogate
absolute-error field.