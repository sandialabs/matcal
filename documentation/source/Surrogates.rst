**********
Surrogates
**********
For complex physics or characterization tests, 
the computational cost of models used for calibration can 
make the use of some calibration methods impractical. 
The ability to use surrogates that accurately 
reproduce model results can facilitate more rigorous 
calibrations. This section covers the surrogate 
generation tools available within MatCal and 
how to incorporate them in the calibration process.

Gaussian Process Surrogates Theory 
==================================
These surrogates are based on those developed in :cite:`pca_gp_surrogates`.

In the cited work, surrogates were employed to approximate an expensive finite element (FE) model,
providing predictions of load and displacement based on a specific set of
constitutive model parameters. To enhance the surrogate representation of the full-field
displacements, principal component analysis (PCA) was utilized.
PCA is a dimension-reduction technique that transforms high-dimensional data into
a reduced basis, preserving the maximum varaince within the data. Throuhg PCA,
the high-dimensional data was effectively represented using only a few PCA modes and their
corresponding singular values (a.k.a. amplitudes). Surrogates were subsequently constructed for each PCA singular value.
Consequently, given a set of parameter values, the surrogate produced predictions of the PCA singular values, 
enabling the reconstruction of the high-dimensional displacement field with little information loss.

Response Dimensionality Reduction 
---------------------------------

:cite:`scikit-learn`


GP Surrogates of PCA Amplitudes
-------------------------------

Results Reconstruction
----------------------

Adaptive GP Surrogates
======================
The fundamental concepts used for adaptive GP surrogate construction are based on those developed
in :cite:`voronoi_adaptive_surrogates`.





