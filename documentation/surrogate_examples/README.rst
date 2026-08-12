Study Examples
--------------
Each example produces a convergence plot, diagnostic plots showing training
sample locations over parameter space and the worst surrogate predictions and errors 
produced at the end of training. The surrogates are 
evaluated against a common-test set so that qualitative comparisons can be made.

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

