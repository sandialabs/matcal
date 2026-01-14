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
a reduced basis, preserving the maximum variance within the data. Through PCA, 
the high-dimensional data was effectively represented using only a few PCA modes and their 
corresponding singular values (a.k.a. amplitudes). Surrogates were subsequently constructed 
for each PCA singular value. Consequently, given a set of parameter values, the surrogate 
produced predictions of the PCA singular values, enabling the reconstruction of the 
high-dimensional displacement field with little information loss.

Response Dimensionality Reduction 
---------------------------------
Given a data matrix, :math:`\mathcal{A} \in \mathbb{R}^{u \times v}`, 
PCA is used to factorize the matrix into three new matrices, 
:math:`\mathcal{A} = USV^{T}`, where :math:`U \in \mathbb{R}^{u \times r}` and 
:math:`V^{T} \in \mathbb{R}^{r \times v}` are orthonormal bases with columns and rows, 
respectively, containing the left and right singular vectors of :math:`\mathcal{A}`. 
The matrix :math:`S \in \mathbb{R}^{r \times r}` is diagonal and contains the singular values, 
and :math:`r` is the rank of :math:`\mathcal{A}`. By retaining a number of principal components 
:math:`p \lt r`, matrix :math:`\mathcal{A}` can be expressed in a reduced-dimensional space. The 
retained components are written as :math:`V^{*T} \subset V^{T}, V^{*T} in \mathbb{R}^{p times v}`.
The reduction of :math:`\mathcal{A}` is written as 
:math:`\mathcal{A}^{pca} = \mathcal{A}'V^{*} \in \mathbb{R}^{u \times p}` and its reconstruction by 
:math:`\mathcal{A}^{recon} = \mathcal{A}^{pca}V^{*T}`.

The data matrix :math:`\mathcal{A}` is representative of a high-dimensional data set, such a full-field data. 
The dimension :math:`u` is the number of repeat measurements, or model evaluations with different 
parameter values, and :math:`v` is the number of measurements in each data set. 
Generally speaking, for this application, :math:`u \lt v`.

The scikit-learn implementation of PCA is utilized in MatCal :cite:`scikit-learn`.


GP Surrogates of PCA Amplitudes
-------------------------------

Results Reconstruction
----------------------

Adaptive GP Surrogates
======================
The fundamental concepts used for adaptive GP surrogate construction are based on those developed
in :cite:`voronoi_adaptive_surrogates`, 

This adaptive surrogate modeling approach iteratively selects new sample points to construct 
a surrogate more efficiently and with greater accuracy than traditional space-filling methods. 
It combines K-fold cross validation with Voronoi tessellations of the input space to assess local 
model error and identify regions where the surrogate performs poorly, guiding the refinement of the 
parameter space. A batch Voronoi adaptive sampling strategy is employed to select multiple informative 
samples simultaneously, reducing overall computational cost. By balancing exploration and exploitation 
through cross-validation error estimates and spatial partitioning, the method concentrates the sampling 
efforts to regions of high uncertainty or error, steadily improving surrogate accuracy while 
minimizing expensive model evaluations.
