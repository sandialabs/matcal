Verification Examples
---------------------
Each example produces a convergence plot and diagnostic plots showing training
sample locations over the true Peaks function and the corresponding surrogate
absolute-error field.

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

