Full-field Study Verification
=============================
In this set of examples, we perform
verification of MatCal studies
using full-field data tools and 
objectives. There are several 
examples studies included here. We provide 
these examples in order to show the current 
best practices when calibrating to full-field 
data for solid mechanics problems and to 
highlight what benefits it can provide. 
We also highlight some challenges that may 
occur when using full-field data. 
Once successful calibration results 
are obtained, we compare the calibration results
and the overall computational performance of the methods.

Full-field Verification Problem Material Model
----------------------------------------------
For this example set, we model
a non-standard specimen subject to 
uniaxial loading to generate synthetic
data for the calibration. Our goal
is to calibrate to these data and reproduce
the parameters used to generate the data. 
In this section we will describe the model
that we used to generate the synthetic data.
First we describe the material model and geometry.
We finish with a description of the 
generated data that will be used for calibration.

For the model
we use a plastic material model with the Hill48 orthotropic
yield surface :cite:p:`hill1948theory` and 
Voce hardening :cite:p:`voce1948relationship`. 
We define the material flow rule
using Voce hardening as

.. math::
    :label: eq_voce_flow

    \theta^2\left(\boldsymbol{\sigma}\right) - 
    \left[Y+A\left(1-\exp\left(-n\epsilon_p\right)\right)\right] 
    \le 0

where :math:`Y`, :math:`A` and :math:`n` are
calibration parameters, and :math:`\epsilon_p` is the 
equivalent plastic strain of the material point.
We use the Hill48 yield surface as 
our equivalent stress :math:`\theta^2\left(\boldsymbol{\sigma}\right)`
where :math:`\boldsymbol{\sigma}` is the 
material Cauchy stress.
The Hill48 yield surface is defined by

.. math::
    :label: eq_hill48

    \theta^2\left(\boldsymbol{\hat{\sigma}}\right) = 
    F\left(\hat{\sigma}_{22}-\hat{\sigma}_{33}\right)^2+
    G\left(\hat{\sigma}_{33}-\hat{\sigma}_{11}\right)^2+
    H\left(\hat{\sigma}_{11}-\hat{\sigma}_{22}\right)^2+
    2L\hat{\sigma}_{23}^2+
    2M\hat{\sigma}_{31}^2+
    2N\hat{\sigma}_{12}^2

where :math:`\boldsymbol{\hat{\sigma}}` is the 
Cauchy stress rotated to a local material 
coordinate system and :math:`F`, :math:`G`, :math:`H`
:math:`L`, :math:`M`, and :math:`N` are 
calibration parameters. These Hill48 
parameters are typically not calibrated since 
they can be calculated using the yield stresses 
of the material in the six directions relative
to the material directions. As a result, ratios 
for the material yield in each direction relative 
to some reference stress value are typically
used for calibration. The ratios are defined by 

.. math::
    :label: eq_hill48_normal_ratios

    R_{ii} = \frac{\sigma^y_{ii}}{\sigma_{ref}}

and 

.. math::
    :label: eq_hill48_shear_ratios

    R_{ij} = \sqrt(3)\frac{\tau^y_{ij}}{\sigma_{ref}}

where :math:`\sigma^y_{ii}` and :math:`\tau^y_{ij}` are the material's 
normal and shear stresses aligned with the material's orthotropic
material directions, and :math:`\sigma_{ref}` is an arbitrary
reference stress.  

We populate the material model with 
parameters that are representative
of an austenitic stainless steel rolled sheet. 
The chosen parameter values are listed 
in :numref:`chosen_mat_params`.


.. _chosen_mat_params:

.. list-table:: Chosen Material Parameters
    :header-rows: 1
    :widths: 25 25 25

    * - Parameter
      - Symbol
      - Value
    * - Elastic Modulus (GPa)
      - :math:`E`
      - 200 
    * - Poisson's Ratio 
      - :math:`\nu`
      - 0.27
    * - Yield Stress (MPa)
      - :math:`Y`
      - 200
    * - Voce Saturation Stress (MPa)
      - :math:`A`
      - 1500
    * - Voce Exponent
      - :math:`n`
      - 2 
    * - :math:`\sigma^y_{11}` Ratio
      - :math:`R_{11}`
      - 0.95
    * - :math:`\sigma^y_{22}` Ratio
      - :math:`R_{22}`
      - 1.00
    * - :math:`\sigma^y_{33}` Ratio
      - :math:`R_{33}`
      - 0.9
    * - :math:`\tau^y_{12}` Ratio
      - :math:`R_{12}`
      - 0.85
    * - :math:`\tau^y_{23}` Ratio
      - :math:`R_{23}`
      - 1.0
    * - :math:`\tau^y_{31}` Ratio
      - :math:`R_{31}`
      - 1.0
    
The synthetic material exhibits significant
hardening, low yield and relatively mild anisotropy 
in yield. The anisotropy was added since it 
is a large driver for adding full-field
data tools for calibration and validation activities. 
This is due to the fact that anisotropy can have a 
large effect on the deformation modes on deformed 
part while having less of an effect on the global 
load-displacement behavior. 

Full-field Verification Problem Geometry
----------------------------------------
The geometry for this set of examples was 
chosen in an effort to require full-field
data tools for an adequate calibration. 
The test geometry consists of a sheet 
with large notches and holes 
specifically placed such that depending 
on how the geometry is loaded the location 
of plastic localization changes. The 
thin sheet design was chosen to allow 
VFM to be used which requires 
a plane stress assumption. 

The change in 
plastic localization location is driven by 
both the material model and the geometry.
To achieve this, the two possible failure 
locations were designed to have similar 
lengths. These are shown in :numref:`ff_study_verification_geo`.

.. _ff_study_verification_geo:

.. figure:: figures/full_field_study_verification/study_verification_example_geometry.png
   :scale: 12%
   :align: center

   The geometry used for the full-field data 
   study verification examples. The specimen
   is loaded along the vertical axis shown with 
   the dashed black line.

The two dimensions :math:`L_s` and :math:`L_t`
are set so that they are approximately equal. 
The stress state in the region of :math:`L_s` is
intended to be shear dominated while the region 
near :math:`L_t` is primarily loaded in tension.
In :numref:`ff_study_verification_geo`, the material 
directions align with the global 
coordinate system shown. The material 22 direction
aligns with the Y axis and the material 11 direction
aligns with the X axis. This will be referred 
to as the ``0_degree`` state. We also simulate
another configuration that will be referred to 
as the ``90_degree`` state. In the ``90_degree``
state configuration, the material 11 direction
is aligned with the Y axis and the material 22 direction 
is aligned with the X axis. 
Full-field data is only output from the more 
finely meshed region of the specimen and only on 
the largest in plane surface.  This surface 
would be similar to what would have measurements
from a test with digital image correlation measurements.

Full-field Verification Problem Modeling Information
----------------------------------------------------
There are a few details worth noting about the model
used to generate the synthetic data. 

#.  The model is stopped once the load has 
    dropped 50% from peak load. This is to 
    save on simulation time. 
#.  Output is only requested on the region 
    of interest to keep memory usage low. This 
    can be done using SierraSM's newer output features.
    Use ``output mesh = exposed surface`` and ``include = {surface_name}``
    in your SierraSM Exodus output block to get output only 
    on a side set named "{surface_name}". 
#.  Tight residual tolerances and many time steps are 
    used in an attempt to ensure finite difference derivatives of 
    the calibration objectives
    with respect to the material parameters 
    are accurate. 
#.  Since this is a verification problem designed to 
    be computationally inexpensive, a mesh convergence
    study was not performed. A converged mesh is not required as long 
    as the calibration mesh matches the mesh used 
    to generate the synthetic data. 
#.  Symmetry was used on the Z plane. Due to 
    mesh asymmetry, the results across 
    the X plane were not symmetric which is believed to be 
    more representative of experimental data. 
    Although we could have enforced symmetry along the 
    X plane, we did not.

The first three points are important to note and should 
be considered for all calibrations.
The last two points are specific to our verification 
testing here. 

Full-field Verification Problem Results
---------------------------------------
As stated previously, the problem 
was designed such that loading it in 
different directions produce different 
regions of plastic localization. This is 
accomplished through the similar 
lengths of :math:`L_s` and :math:`L_t`
and the in-plane Hill48 yield ratios
:math:`R_{11}`, :math:`R_{22}`, :math:`R_{12}`. 
If this were modeled using a von Mises yield surface, 
the part would always localize and fail in the center
along :math:`L_t`. The stress in the component 
is shown at peak load for both the ``0_degree`` 
and ``90_degree`` states in :numref:`ff_study_verification_peak_load_stress`.

.. _ff_study_verification_peak_load_stress:

.. figure:: figures/full_field_study_verification/study_verification_example_stress_peak.png
   :scale: 15%
   :align: center

   The von Mises stress at peak load for each 
   state. The left is the ``0_degree`` state 
   and the right is the ``90_degree`` state. 



The stress fields look fairly similar with only a slight 
bias toward the :math:`L_s` region for the ``0_degree`` and the
:math:`L_t` region for the ``90_degree``. The localization 
is clear when looking at the plastic strains in the model 
after peak load as shown in :numref:`ff_study_verification_localized_eqps`.

.. _ff_study_verification_localized_eqps:

.. figure:: figures/full_field_study_verification/study_verification_example_localization.png
   :scale: 15%
   :align: center

   The equivalent plastic strain is shown 
   at the end of the simulation (10 seconds)
   for each model. The left is the ``0_degree`` state 
   and the right is the ``90_degree`` state. 


The load-displacement curves 
also exhibit noticeably different behavior. 
They are shown in :numref:`ff_study_verification_load_displacement`.
for both simulated configurations.
As expected, the specimen loaded in the material 
22 direction has higher load carrying capacity 
due to the larger :math:`R_{22}` value. Due 
to the different localization regions, the load displacement
curves unload at different rates. 

.. _ff_study_verification_load_displacement:

.. figure:: figures/full_field_study_verification/study_verification_example_load_displacement.png
   :scale: 100%
   :align: center

   The generated load displacement curves are shown 
   here for each simulation. 

Full-field Verification Study Examples
--------------------------------------
In this section, we present many studies 
in an attempt to perform the calibration.
The ultimate goal is to use one state, 
either the ``0_degree`` state or ``90_degree`` 
state, to perform the calibration an 
obtain calibrated parameters within 
a few percent of the input parameters 
presented in :numref:`chosen_mat_params`.
In practice, this is more difficult than 
it seems. The calibration focuses on 
only five of the eleven total material
parameters. We only calibrate :math:`Y`, :math:`A`, :math:`n`, 
:math:`R_{11}` and :math:`R_{12}`. We assume that
elastic parameters can be found 
in the literature and that only the in plane
Hill48 ratios can be calibrated 
with the sheet-like specimen. We also set 
our :math:`\sigma_{ref} = \sigma^2_{22}` which 
requires that :math:`R_{22}=1` which is a common 
practice. 

With the five calibration parameters 
chosen, we first verify that the objectives are all
sensitive to these parameters. 
We perform a MatCal :class:`~matcal.core.parameter_studies.ParameterStudy`
where each parameter
was changed -5% to +5% from its initial value 
and the full-field objectives in MatCal were evaluated 
for each parameter set. This was done to investigate
shape of the objective function near the preselected
parameter values used for the synthetic data generation. 
Ideally, the objective will be lowest at the values
specified in :numref:`chosen_mat_params` and smoothly 
increase away from them. As can be seen in 
:ref:`Objective Sensitivity Study`, the objectives
are all sensitive to the input parameters and 
return the lowest objective at, or in the case of 
VFM near, the parameters used for data generation.
This alone verifies that our implementation is behaving
as expected for all new full-field methods. However, 
we also wish to investigate how well these methods
work in a calibration study and what issues 
users may encounter.

As a result, we now attempt calibrations using different combinations
of data, algorithms and objectives. Most calibrations 
are attempted with only a single data set since one of the 
goals of including full-field data would be to reduce the number
of tests needed to identify parameters. Only with VFM 
do we use more than one data set because the VFM model is 
about 10x faster than the full finite element model and the objective  
function appears more convex than the others based 
on the sensitivity study results. Therefore, 
the following calibrations were attempted:
  
#. A standard load-displacement calibration using 
   a nonlinear least squares method
   with all five unknown parameters being calibrated.
   This study fails to provide an accurate answer and stalls due to 
   poor objective function gradients and Dakota exits with 
   ``FALSE CONVERGENCE``. This is likely because the load-displacement curve is 
   relatively insensitive to the :math:`R_{11}` and :math:`R_{12}` 
   parameters. We only used the ``0_degree`` state for this calibration.
#. A VFM calibration using 
   a nonlinear least squares method where only the ``0_degree`` state 
   is used and all five parameters are calibrated. 
   This calibration converges with ``RELATIVE FUNCTION CONVERGENCE``, 
   however, several parameters 
   have significant error. 
#. A VFM calibration using 
   a nonlinear least squares method where three data sets are used
   and all five parameters are calibrated. We added an additional 
   data set ``45_degree`` to the calibration along with
   the ``0_degree`` and ``90_degree``. This calibration converges with 
   ``RELATIVE FUNCTION CONVERGENCE``.
   This calibration provides an
   acceptable fit with all parameters identified below 5%. To 
   fit all parameters within a reasonable tolerance, VFM needs 
   all three loading directions. Any less, and the calibration 
   cannot identify all parameters.
#. A calibration using a standard load-displacement objective 
   and a full-field interpolation objective. A nonlinear least squares method
   is used to calibrate all five unknown parameters.
   This calibration fails with 
   ``FALSE CONVERGENCE``.
   The parameter's are improved 
   over the first calibration above, but the errors are still higher 
   than expected for verification purposes and the algorithm is likely 
   in a local minima as it balances the two objectives.
   We only used the ``0_degree`` state for this calibration.
#. A calibration using a standard load-displacement objective 
   and a polynomial HWD objective. A nonlinear least squares method
   is used to calibrate all five unknown parameters.
   This calibration completed with 
   ``FALSE CONVERGENCE``; 
   however, the parameters 
   have similar magnitude errors as those in the previous example. In 
   contrast, the objectives have both been significantly minimized. 
   This suggests the current use of the HWD weights 
   as objectives have had a small effect on the calibrated parameter
   results.
   We only used the ``0_degree`` state for this calibration.
#. A calibration using a standard load-displacement objective 
   and a full-field interpolation objective with only 
   the full-field data at peak load included in the 
   objective. A nonlinear least squares method
   is used to calibrate all five unknown parameters
   and the initial point is chosen at values that 
   are 4% away from the known solution.
   This calibration successfully completes with 
   ``RELATIVE FUNCTION CONVERGENCE``.
   The  calibrated parameter's are all within 0.1% from the known 
   solution. We only used the ``0_degree`` state for this calibration.
#. A calibration using a standard load-displacement objective 
   and a polynomial HWD objective with only 
   the full-field data at peak load included in the 
   objective. A nonlinear least squares method
   is used to calibrate all five unknown parameters
   and the initial point is chosen at values that 
   are 4% away from the known solution.
   This calibration completes with 
   ``FALSE CONVERGENCE``.
   The parameter  errors are relatively unchanged
   from the first HWD calibration example, reinforcing 
   that updates to the HWD objective will be needed
   to provide the desired verification results.
   We only used the ``0_degree`` state for this calibration.

In conclusion, the methods are verified to work as intended by the objective
sensitivity example and that gradient calibrations can be used with 
VFM and the full-field interpolation objective. Also, 
although the HWD example does provide satisfactory results, 
it cannot return parameter values within 1%. We believe 
the cause is due to mode switching that could be occurring 
for lower amplitude modes. This may make the objective landscape 
less amenable to gradient techniques. More work is need to 
improve their performance and provide well verified results. 

It is important to note that the examples in this series also show 
the common issues that can be encountered
when calibrating challenging problems. They indicate that attempting 
to calibrate models to limited data introduces complications to the 
objective landscape that makes calibration more difficult. For VFM, 
the solution is to add more data. The other methods require 
careful objective choice that improve the 
overall convexity of the objective and careful initial 
point selection. 
Calibration with HWD and full-field interpolation
may also be improved by adding more data sets to the objective 
as was done with the VFM calibration. However, this will increase the
computational cost of the 
calibration by a factor for each data set added. It is also
likely to introduce more local minima that optimization
routines will need to avoid. 
Without 
the ability to improve the objective landscape,
the use of nongradient optimization algorithms will help 
ensure minima are found at the cost of additional expense.
An aspect not investigated in this effort are 
different algorithm options and calibration setup. 
Adjustments in calibration algorithm options could improve
overall performance.

For this suite of examples, 
the full-field objective calibration converges well 
with an initial guess close to the known solution. Since 
the calibrations for all three objectives with 
a initial point far from the known solution provide calibrated 
parameters very near the known solution, quality calibrations 
could likely be obtained using a two step calibration 
where the second calibration is a repeat of the first calibration with 
two changes: (1) the initial point updated to the first calibration's calibrated 
parameter set and (2) the use of the full-field interpolation objective. Since VFM
calibrations are signifcantly cheaper and well posed, this should be the 
first choice for this first step. If VFM cannot be used 
due to its plane-stress constraint, HWD can be used for
memory intensive problems or the full-field interpolation 
objective if memory is not a problem.

Future releases will include 
a couple tools to help tackle calibration issues 
related to cost and objective function landscape:

#. An integration objective metric that can be applied 
   to objectives with large numbers of QoIs.
   Currently, only L2- and L1-norm metrics 
   are available. The load-displacement objective 
   maybe improved if the absolute value of the
   error is integrated and provided 
   as the objective value. This will not be valid with 
   least squares algorithms, so a different gradient 
   based algorithm will need to be used such as one of 
   Dakota's sequential quadratic programming algorithms.
   Many of Dakota's different gradient based methods 
   can be accessed through the
   :meth:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy.set_method`
   method.
#. Combined QoI extractors that will allow users to extract 
   specific times from the data and then extract full-field
   comparisons at these times. This may improve the objective 
   landscape away from the global minimum. For example, 
   we could extract the data at peak load for each data set 
   and then compare the full-field data at peak load 
   even if the peak load for the calibration evaluation 
   is far in time from the experimental data. 
#. More efficient global calibration algorithms that 
   build surrogates on the objectives as it searches the 
   parameter space
#. Tools for building surrogate models to replace 
   user models. Once built using the supplied user model, 
   the surrogate models can be used 
   in place of the full user model.
  
See the input decks for these calibration examples
with additional commentary and results in the following examples.
