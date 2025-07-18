Hierarchical Wavelet Decomposition
==================================

Hierarchical Wavelet Decomposition (HWD) attempts to represent large unstructured data as a series of pattens. 
Representing the data as patterns allows for compression of the residual size required to analyze the data, allowing 
larger problems to be run, and potentially allowing for more meaningful interpretations of the data than could be observed 
though direct observation of the unstructured data set. In MatCal HWD is primarily used to create a latent space on which full-field data is compressed.
Compressed full-field data allows HWD based parameter studies to use full-field data in the same way as an interpolated based objective, however at a much lower memory footprint.

HWD Theory
----------
HWD at its core is a linear projection of a data set onto a series of orthogonal bases that characterize the data seen in the data set. 
There are many forms of projection based methods, with Principal Component Analysis (PCA) being one of the most common methods uses. 
In equation :eq:`eq_bases_projection` shows the general setup for this type of method. 

.. math::
    :label: eq_bases_projection

    \underline{d} = \underline{\underline{Q}} \cdot \underline{c}

Where :math:`\underline{d}` is the unstructured full-field data, :math:`\underline{\underline{Q}}` is the matrix of bases vectors, and 
:math:`\underline{c}` are the weights of the bases vectors used to recreate the original field data. The data vector :math:`\underline{d}` is of 
length :math:`n`. This means that the bases matrix :math:`\underline{\underline{Q}}` is of shape :math:`n \times m` where :math:`m` is the number of 
bases vectors. It is desired that :math:`n >> m`, such that the compressed weights are much smaller than the initial data set, but effectively contain 
as much information as the initial data set. 

It is common to define :math:`\underline{\underline{Q}}` to be orthogonal. Which means that all bases vectors in :math:`\underline{\underline{Q}}` contain unique information. 
Orthogonality also, allows for a simple inversion of :math:`\underline{\underline{Q}}`, buy taking its transpose, and therefore allowing one to find the values for the weights 
by using equation :eq:`eq_bases_projection_inv`.

.. math::
    :label: eq_bases_projection_inv

    \underline{\underline{Q}}^T \underline{d} =  \underline{c}


The challenge is projection based methods is identifying an appropriate set of bases on to which to project the data. 
Often the approach taken is to generate a sufficient about of data and directly build a bases based off a corpus of pre-computed results. 
While this method generates great compression ratios for a given data set, it has a large upfront cost that can be intensive to generate. 
This means that for initial exploratory parameter studies methods that rely on a pre-computed corpus of data, like PCA, can be prohibitively expensive.
HWD attempts to provide a similar service to PCA, but without the requirement for a pre-computed corpus of data. 

HWD combines the approaches of Salloum :cite:p:`alpert_wavelet` and Christian :cite:p:`qr_full_field`. To generate a series of 
orthogonal wavelets at various length scales across the data. The core steps to generate the basis matrix using HWD consists of 3 steps:

#. Generation of hierarchical tree to represent of the physical space occupied by the full-field data
#. Generation of approximation functions at the various levels of the tree
#. Use of QR Decomposition to generate an orthogonal basis(:math:`\underline{\underline{Q}}`.) and a change of basis matrix(:math:`\underline{\underline{R}}`.)

Additional steps, or sub-steps my be required to properly process the data or generate a more informed bases, but at its core these are the main steps used in HWD. 
HWD builds upon :cite:p:`alpert_wavelet` and :cite:p:`qr_full_field`, by taking the QR factorization approach of :cite:p:`qr_full_field` and applying it to a 
hierarchical split domain from :cite:p:`alpert_wavelet`. 

As identified by :cite:p:`qr_full_field`, applying patterns directly to different unstructured data sets generates compressed forms that are similar, but are not the same. 
While similar but not the same may be sufficient for some applications, for the its use in applications like material calibration such a discrepancy is not useful. 
Their solution was to build their basis from an initial background series of polynomials. These polynomials are defined in space covering the domain of the data points, but not 
bound to the points themselves. these polynomials are evaluated across all points in a given data set to generate a moment matrix :math:`\underline{\underline{M}}`. 
Each column of the moment matrix is the evaluation of one polynomial in the series, with the rows being the evaluation result for a given point. 
The moment matrix is then decomposed in to an orthogonal basis(:math:`\underline{\underline{Q}}`.) and a change of basis matrix(:math:`\underline{\underline{R}}`.) via QR factorization.

The factorization of the moment matrix provides the :math:`\underline{\underline{Q}}` necessary to generate weights using :eq:`eq_bases_projection_inv`, but also supplies a method to map the weights from
one data set to an other. In equation :eq:`eq_mapped_weight`, the change of basis matrices are used to map the weights from data set 1 to be compatible with data set 2. 
The mapped weights now can be compared much more rigorously. 

.. math::
    :label: eq_mapped_weight

    \underline{c}_{12}= \underline{\underline{R}}_2 \underline{\underline{R}}_1^{-1} \underline{c}_1

Using equation :eq:`eq_mapped_weight` a calibration residual can be defined by labeling one of the data sets as the reference(or experimental) and subtracting it by 
the mapped weighted simulation data, like in equation :eq:`eq_qr_residual`.

.. math::
    :label: eq_qr_residual

    \underline{r}= c_{ref} - \underline{\underline{R}}_{ref} \underline{\underline{R}}_{sim}^{-1} \underline{c}_{sim}

HWD extends the work of :cite:p:`qr_full_field` by generating an initial moment matrix that takes into account both large scale and small scale patterns that may be present in the data. 
HWD does this by using a binary tree to successively split the physical space of the data, and then applying pattern functions(typically polynomials) to all the regions generated by the 
tree splitting. Generating the splits for each section of the domain can be done in a variety of ways, but is usually done by using clustering algorithms to split regions into regions 
of greatest likeness. 

Pattern functions are the functional forms chosen to populate the momentum matrix. Currently, polynomials are the only functions that have been extensively used, but
other functional forms are valid choices as well. Future research is planned to investigate the optimal choice of functional form used for the pattern functions if 
some data is available. 

By applying the pattern functions at different levels of spatial scale it is believed that important localized behavior should be captured better than if exclusively larger scale 
functions are used. In practice only one of the two regions generated by each split in the tree, has the pattern function applied to it. 
This is to preserve linear independence between the size scales, required to obtain the desired QR factorization. 

.. figure:: images/HWD_momentum_matrix.png
    :scale: 50%
    :alt: Example momentum matrix generated using a quadratic polynomial on one dimensional data. 
    :align: center
    
    This is an example illustration of a momentum matrix generated by HWD. The data in this example is ordered one dimensional data, and the pattern functions chosen are
    polynomials up to order 2. The domain has gone though 3 layers of splitting. The colored regions at the bottom indicate the different tiers of splitting, while the numbers 
    at the top indicate the order of polynomial used in that column. 

After the tiered momentum matrix is generated, it is run though the same flow as :cite:p:`qr_full_field`; a QR factorization is applied to the momentum matrix and a 
basis matrix(:math:`\underline{\underline{Q}}`) and change of basis matrix(:math:`\underline{\underline{R}}`) are formed. Then equations :eq:`eq_mapped_weight` and :eq:`eq_qr_residual` 
can be used to define a residual, the only additional complication is that all data sets that want to be compared need to use the same splitting/clustering instance. 
By using the same instance, it will ensure that the different data sets are partitioned in the same way and have compatible bases. 

MatCal HWD Implementation
-------------------------
MatCal implements HWD using polynomials through an external HWD library. The library can be found on the Sandia SRN at https://cee-gitlab.sandia.gov/mwkury/hierarchical-wavelet-decomposition.
MatCal currently uses a reduced polynomial pattern function to generate its moment matrix in the HWD library. 
Because full-field data is two dimensional in space, generating polynomials of arbitrary order can generate an excessive amount of possible bases, without adding 
too much additional value. The reduced polynomial pattern function cuts down on the number of polynomial terms generated by only keeping the isolated X and Y terms (e.g. :math:`X^3` or :math:`Y^2`)
and only generating the middle cross term for a even polynomial powers (e.g. if the power is 4, then :math:`X^4`, :math:`Y^4`, and :math:`X^2 Y^2` will be generated). 
This helps preserve the effects that cross both X and Y, while keeping the number of bases from exploding out of control. 

During the definition of an HWD objective, the user can supply the max polynomial order and the max sets of splits.
Higher numbers of splits will also balloon the number of bases generated, and users are advised not to go above 8. 

Due to the current naive implementation of the domain decomposition in the HWD library, the library does not make ideal choices when supplied 
point cloud geometries of complex shapes(such as those containing holes). Due to poor decomposition behavior, the the cross data set mapping is too poor 
to be used for material calibration.  While the HWD library will be improved to create better domain splitting choices, a stop gap measure 
leveraging the generalized moving least squares (GMLS) algorithm tools in MatCal and pycompadre :cite:p:`compadre_toolkit` is used to interpolate 
the experimental data sets on to the simulation data sets. 

Performing the interpolation in the beginning, allows the remapping step (equation :eq:`eq_mapped_weight`) to be skipped, and the calculated weights to be compared directly, equation :eq:`eq_interp_qr_residual`. 

.. math::
    :label: eq_interp_qr_residual

    \underline{r}= c_{ref} - \underline{c}_{sim}

In addition to the streamlined residual, performing an initial interpolation on to the simulation data points allows for an easy down-selection of bases vectors. 
Because the remap step potentially ties all weights together, removing that step allows us to ignore weights for bases that do not play an important role in 
describing data. MatCal supplies a tolerance ratio to the HWD library that governs the minimum magnitude of a basis weight required to be considered significant. 
Based on the first set of data supplied to the HWD library, the largest weight is identified and all bases with weights of lower magnitude than the tolerance ratio 
times the largest weight magnitude are discarded and only the remaining bases vectors and weights are used.
This allows for additional compression of the residual, which is important because it allows for larger models to be used in parameter studies. 
