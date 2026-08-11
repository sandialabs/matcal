Surrogate Verification
======================

These examples verify several surrogate modeling workflows available in MatCal.
The examples use the two-dimensional Peaks benchmark function from the
KFCV-Voronoi adaptive sampling paper by :cite:`voronoi_adaptive_surrogates`.
This set of examples is meant to demonstrate features and behavior of the
surrogates in MatCal. They are not meant to be used as templates for user
MatCal files.

The Peaks function is a useful verification problem because most of the domain
is relatively smooth and low magnitude, while the important response behavior is
localized near a small region of the parameter space. This makes it a good test
for whether a sampling method can discover and refine important local features.

The verification examples include several surrogate workflows:

* pretrained Gaussian-process, RBF, and random-forest surrogates built from
  nested random samples;
* adaptive Voronoi Gaussian-process, RBF, and random-forest surrogates;
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

The following results were obtained for the Peaks verification examples. The
Peaks function is continuous and mostly smooth over the domain, with localized
high-gradient behavior that makes it useful for testing whether adaptive
sampling can discover and refine important local features.

.. list-table:: Peaks surrogate verification results
   :header-rows: 1
   :widths: 34 12 16 18 22 16 18

   * - Method
     - Final samples
     - Final RMSE
     - Final max error
     - Best max error, sample count
     - RMSE rate ``p``
     - Max-error rate ``p``
   * - Pretrained random-sampling GP
     - 150
     - 5.111100e-01
     - 4.808070e+00
     - 4.393954e+00 at 140
     - 0.7237
     - 0.3678
   * - Pretrained random-sampling RBF
     - 150
     - 5.606465e-01
     - 4.819553e+00
     - 4.819553e+00 at 150
     - 0.3639
     - 0.1775
   * - Pretrained random-sampling random forest
     - 150
     - 1.000573e+00
     - 7.446087e+00
     - 7.334305e+00 at 80
     - 0.1169
     - 0.0447
   * - Adaptive Voronoi GP
     - 150
     - 3.016666e-02
     - 1.588163e-01
     - 1.588163e-01 at 150
     - 1.9845
     - 2.2087
   * - Adaptive Voronoi RBF
     - 150
     - 7.177528e-02
     - 4.102925e-01
     - 3.519908e-01 at 123
     - 1.4752
     - 1.7587
   * - Adaptive Voronoi random forest
     - 150
     - 5.086865e-01
     - 3.297064e+00
     - 3.170707e+00 at 144
     - 0.2570
     - 0.3117
   * - Adaptive sparse grid
     - 769
     - 5.580570e-02
     - 3.334068e-01
     - 3.334068e-01 at 769
     - 0.6893
     - 0.6184

For all three surrogate-regressor families tested with random and adaptive
sampling, the adaptive sampling methods produced lower final validation errors
than the corresponding pretrained random-sampling examples. The adaptive
methods also outperformed all of the random-sampling examples in final maximum
absolute error. This is expected for the Peaks benchmark because the most
important approximation difficulty is localized, and adaptive methods can use
response information to place new samples in high-error regions.

Discussion
----------

The pretrained random-sampling examples provide a baseline for comparing the
adaptive methods. It has no knowledge of where the Peaks function
changes rapidly or where the most important local features are. As a result,
many samples may be spent in regions that are easy to approximate, while the
localized high-gradient region remains under-resolved.

Among the pretrained random-sampling cases, the Gaussian-process and RBF
surrogates produced similar final maximum errors, with the Gaussian-process
surrogate giving a somewhat lower final RMSE and faster observed convergence
rate. The pretrained random-forest surrogate performed worst for this smooth
two-dimensional interpolation problem. This behavior is consistent with the fact
that random forests are not smooth interpolants; their predictions are assembled
from decision-tree ensembles and can be less effective for representing smooth
localized features than Gaussian-process or RBF surrogates.

The adaptive Voronoi methods performed substantially better than the
pretrained random-sampling methods at the same final sample count of 150. The
KFCV-Voronoi adaptive sampling method uses cross-validation error to identify
regions where the surrogate is performing poorly, then uses Voronoi cell
geometry to place new samples away from existing samples in those important
regions. Because the Peaks function has localized difficult behavior, this
sampling strategy can concentrate samples where they are most valuable.
The adaptive Voronoi Gaussian-process surrogate gave the best overall result in
this verification set. The adaptive Voronoi RBF surrogate also performed well. 

The adaptive Voronoi random-forest surrogate also improved substantially over
the pretrained random-sampling random-forest surrogate. 
Its errors are much lower than the corresponding pretrained random-forest
errors. However, the adaptive Voronoi random-forest surrogate did not match the
accuracy of the adaptive Voronoi Gaussian-process or RBF surrogates. This is
consistent with the expected behavior of random forests on a smooth
interpolation benchmark.

The adaptive sparse-grid surrogate also performed substantially better than the
random-sampling baselines. Sparse grids use a structured approximation space and
adaptive refinement, so they are more systematic than purely random sampling.
However, sparse-grid methods are still constrained by their grid-based
structure. They tend to perform best when the response is globally smooth or
when important features can be efficiently represented by the sparse-grid basis.
The Peaks function has a localized region of strong response variation, so the
sparse grid must spend samples refining a structured grid representation of a
local feature. Its best prediction errors were much lower than the pretrained
random-sampling errors, but the sparse-grid run required more samples than the
adaptive Voronoi runs. This is expected for this type of localized feature: the
Voronoi adaptive sampling methods can place samples directly in high-error
regions, while the sparse-grid method refines a structured approximation.

Overall, the results demonstrate the expected benefit of adaptive sampling for
this benchmark. All adaptive sampling methods performed better than all
pretrained random-sampling examples in final maximum absolute error. The
adaptive Voronoi Gaussian-process and RBF surrogates were especially effective
because they combined smooth surrogate models with sample placement targeted at
localized high-error regions.

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

Sometimes, the answers to these questions are unknown, so it may be
useful to start with a random sampling methdo. If the behavior of interest 
is not well predicted with the surrogates, it may be beneficial to 
change to adaptive surrogates. Adapative voronoi surrogates 
are useful for low dimensional problems (5 input parameters or less)
and sparse grid surrogates tend to perform decently with on the order
of 12 inputs. 

Example Summary
---------------

``plot_a1_peaks_pretrained_random_gp.py``
    Builds pretrained Gaussian-process surrogates from nested random training
    sets.

``plot_a2_peaks_pretrained_random_rbf.py``
    Builds pretrained RBF surrogates from nested random training sets.

``plot_a3_peaks_pretrained_random_random_forest.py``
    Builds pretrained random-forest surrogates from nested random training sets.

``plot_a4_peaks_adaptive_voronoi_gp.py``
    Builds an adaptive Gaussian-process surrogate using KFCV-Voronoi sampling.

``plot_a5_peaks_adaptive_voronoi_rbf.py``
    Builds an adaptive RBF surrogate using KFCV-Voronoi sampling.

``plot_a6_peaks_adaptive_voronoi_random_forest.py``
    Builds an adaptive random-forest surrogate using KFCV-Voronoi sampling.

``plot_a7_peaks_adaptive_sparse_grid.py``
    Builds an adaptive sparse-grid surrogate using PyApprox through MatCal's
    sparse-grid adaptive surrogate study.

Each example produces a convergence plot and diagnostic plots showing training
sample locations over the true Peaks function and the corresponding surrogate
absolute-error field.