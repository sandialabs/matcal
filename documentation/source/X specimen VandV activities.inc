X-specimen Verification and Validation Activities
===================================================
The verification and validation tests of the full-field objective optimization tools built in MatCal were based off the work done in the 
Experimodelment LDRD by Elizabeth Jones et al. :cite:p:`VFM_LDRD_Jones`. Using full-field data in junction with conventional probe 
measurements, and advanced optimization/UQ methods for material characterization allows for a much wider set of test geometries than is 
traditionally used. In traditional calibration, samples are typically designed to generate physical fields with homogeneous distributions 
(such as strain in a uniaxial tension sample). With the addition of full-field data from DIC or other sources, these limitations can be 
relaxed, and novel geometries can be used in the hopes of gaining more information per experiment run. 

With the larger possibility space for possible sample geometries, a new question arises, "what should my sample geometry be?".
The answer to that question is beyond the scope of this work. But two possible configurations are proposed by :cite:p:`VFM_LDRD_Jones`, the 
X- and D- specimen.

.. figure:: figures/Xspec.png
    :alt: X-specimen
    :align: center

    CAD illustration of X-Specimen from Experimodelment LDRD.

.. figure:: figures/Dspec.png
    :alt: D-specimen
    :align: center

    CAD illustration of D-Specimen from Experimodelment LDRD.

These specimens were designed with the Virtual Fields Method in mind, which is the most restrictive of all of MatCal's full-field
calibration objectives. Therefore, building off of the Experimodelment work is an appropriate way to test and compare all of MatCal's 
full-field tools. Additional considerations that were put into the design of these tests samples are summarized below, and more details
can be found in the SAND Report. 

#. Maximize Stress Heterogeneity: Create non-uniform stress states in order to generate more potential data per experiment. 
#. Maximize Range of Strain Rates: Sample as wide as possible domain of strains to inform the material model. 
#. Minimize Large Gradients in Stress or Strain Near Sample Edges: Data Collection using DIC becomes poor near edges, minimizing gradients helps.
#. Restrict Geometry to Planar: DIC only captures surface data, and VFM requires knowledge of the deformation through the sample. Keeping the geometry as planar as possible assists in preserving the assumption that the surface in-plane deformations can be applied though the thickness of the sample.
#. Uniaxial loading direction: This sample was designed to work with a uniaxial loading rig.

To test the full-field calibration capabilities of MatCal on a realistic problems, we will be using the X-Specimen in two testing applications:

#. Calibration of the X-specimen to synthetic data generated from a known model
#. Calibration of the X-specimen to data collected from the Experimodelment LDRD

Calibrating against synthetic data was done to ensure that the full implementation of the full-field calibration code was done correctly in MatCal on 
a scale of problem that users would want to run on. The use of synthetic data allows for direct comparison against a known solution, allowing for 
greater confidence in the test of correctness. 

In calibrating against the data collected in the Experimodelment LDRD we ensure that the full-field tools can handle real data generated from 
actual experimental equipment. 

X-specimen Synthetic Data Calibration Verification
--------------------------------------------------
A synthetic data case was chosen as the initial capability verification case because a realistic test could be defined, but with a known solution. 
With a known solution, the calibration performance of each full-field calibration method could be evaluated. 

A 304L stainless steel temperature dependent elastic-plastic J2 plasticity model was chosen for the base material model. 
The material parameters used were calibrated using conventional methods. 
The reference model was run using coupled SIERRA/SM and SIERRA/Aria, more details of the material model and 
its implementation can be found in the material model reference hand book :cite:p:`lame_manual`.
SIERRA/SM handled the solid mechanical aspects of the multiphysics solve,
while SIERRA/Aria took in the thermal energy generated from plastic work and evolved the thermal diffusion aspects of the problem. 
The parameters that were calibrated in the synthetic test were set at the values shown in :numref:`syn_params_x_specimen`

.. _syn_params_x_specimen:

.. list-table:: Synthetic Material Parameters of Interest
    :header-rows: 1
    :widths: 25 25
    :align: center

    * - Parameter
      - Value
    * - Yield Rate Coefficient
      - 10^3 (1/s)
    * - Yield Rate Exponent
      - 10.041
    * - Yield Stress
      - 301.43 (MPa)
    * - Hardening Exponent
      - 1.8153
    * - Temperature Hardening Coefficient 1
      - 3540.6 (MPa)
    * - Temperature Hardening Coefficient 2
      - 3.7741 (MPa/K)

These parameters were obtained in a multi-step traditional calibration to uniaxial tension
experiments performed on the material used to manufacture 
the X specimen. By using a material model calibrated to the material of interest, 
the synthetic data is more similar to the experimental data we will be comparing 
to next than if arbitrary 
parameters where chosen. 
Tests were performed with strain rates of :math:`1\times10^{-4} \frac{1}{s}`, 
:math:`3.2\times10^{-3} \frac{1}{s}`
and :math:`1\times10^{-1} \frac{1}{s}` so that 
material rate dependence in this rate range could be calibrated.
Due to heating from plastic work, a temperature dependent material model 
was required to capture the thermal softening at the higher rates. 
The calibrated model results are compared 
against the tension data in :numref:`fig_tension_calibration_slow`,
:numref:`fig_tension_calibration_middle`,
and :numref:`fig_tension_calibration_fast`.

.. _fig_tension_calibration_slow:

.. figure:: figures/x_specimen_tension_cal/x_specimen_data_tension_fit_slow.png
   :align: center

   :math:`\dot{\epsilon} = 1\times10^{-4} \frac{1}{s}` - stress in Pa

.. _fig_tension_calibration_middle:

.. figure:: figures/x_specimen_tension_cal/x_specimen_data_tension_fit_med.png
   :align: center

   Middle rate - stress in Pa


.. _fig_tension_calibration_fast:

.. figure:: figures/x_specimen_tension_cal/x_specimen_data_tension_fit_fast.png
   :align: center

   Fast rate - stress in Pa


The synthetic data generated from this model was performed on a hex mesh of the X-specimen with an element size of approximately .4mm. 
this mesh was found be be converged based on the nodal displacements on the mesh. To capture the rate dependent behavior of the 
plasticity model three uniaxial extension rates were used (:numref:`tab_x_syn_pull_rates`). The rates were selected to generate multiple 
orders of magnitude of strain-rates. 

.. _tab_x_syn_pull_rates:

.. list-table:: Synthetic Pull Rates
    :header-rows: 1
    :widths: 25 25
    :align: center

    * - Name
      - Extension Rate (mm/s)
    * - Slow
      - .005
    * - Medium
      - .154
    * - Fast
      - 5.0

To gauge the sensitivity of mesh refinement for each calibration method, two mesh refinement levels were run, a coarse and a fine mesh.
The average element size in the primary region if interest is shown in the :numref:`tab_x_mesh_sizes`. 
The coarse mesh is below the level required to guarantee converged behavior of the mesh displacements. 
Multiple mesh sizes were not used for the VFM calibrations. 
Since the disconnected hex model is being used, the VFM mesh size 
should be on the order of the source data discretization size. 

.. _tab_x_mesh_sizes:

.. list-table:: Test Mesh Sizes
    :header-rows: 1
    :widths: 25 25
    :align: center

    * - Name
      - Approximate Mesh Size (mm)
    * - Coarse
      - .8
    * - Fine
      - .4

For all full-field methods a nonlinear least squares method, Dakota's implementation of ``nl2sol``, was chosen to calibrate the material models. 
This method was chosen because it can achieve rapid convergence when sufficiently close to a minima, therefore converging or 
failing to converge to a solution in a small number of evaluations. 


VFM Synthetic Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^
The VFM calibration performed well for the synthetic calibration. 
All parameters were identified within 4% and the algorithm reached 
convergence quickly. The resulting parameters and errors 
are shown in :numref:`tab_x_sym_vfm_all_params` when calibrating
all parameters.

.. _tab_x_sym_vfm_all_params:

.. list-table:: VFM Calibration Results All Parameters
    :header-rows: 1
    :widths: 25 25 25
    :align: center

    * - Parameter
      - Calibrated Value
      - Percent Error
    * - Yield Rate Coefficient
      - 10^2.99 (1/s)
      - -0.4
    * - Yield Rate Exponent
      - 10.11
      - 0.7
    * - Yield Stress
      - 300.5 (MPa)
      - -0.3
    * - Hardening Exponent
      - 1.83
      - 0.8
    * - Temperature Hardening Coefficient 1
      - 3597.1 (MPa)
      - 1.6
    * - Temperature Hardening Coefficient 2
      - 3.91 (MPa/K)
      - 3.8

Since the hardening parameters are coupled and may not 
be well quantified by the data, we fix the 
Temperature Hardening Coefficient 2 to the truth value and 
re-ran the VFM calibration. The results improved noticeably. 
These results are presented in :numref:`tab_x_sym_vfm_fixed_H2`.

.. _tab_x_sym_vfm_fixed_H2:

.. list-table:: VFM Calibration Results Reduced Parameters
    :header-rows: 1
    :widths: 25 25 25
    :align: center

    * - Parameter
      - Calibrated Value
      - Percent Error
    * - Yield Rate Coefficient
      - 10^3.00 (1/s)
      - 0.06
    * - Yield Rate Exponent
      - 10.03
      - -0.074
    * - Yield Stress
      - 301.3 (MPa)
      - -0.05
    * - Hardening Exponent
      - 1.83
      - 0.959
    * - Temperature Hardening Coefficient 1
      - 3554.2 (MPa)
      - 0.386

Both calibrations converged with ``RELATIVE FUNCTION CONVERGENCE``
within 50 iterations indicating that the objective function 
was well defined and the derivatives of good quality. The small 
errors in the parameters indicate that for this problem, the 
plane stress assumption is valid. 

HWD Synthetic Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^
:numref:`tab_x_sym_hwd_0_24` and :numref:`tab_x_sym_hwd_6_8` contain the calibrated parameter results for two different HWD calibration setups. 
The runs summarized in :numref:`tab_x_sym_hwd_0_24`, show the results for HWD calibration with a no subdivision of the domain and 
using a high order polynomial(24).
The runs summarized in :numref:`tab_x_sym_hwd_6_8` show the results for a HWD calibration 
using a setup close to the recommended default settings for Polynomial HWD.
The recommended setup uses approximately 6 sets of subdivisions and a moderate order polynomial(8).

.. _tab_x_sym_hwd_0_24:

.. list-table:: HWD Synthetic Calibration Results D0 P24
    :header-rows: 2
    :widths: 25 25 25 25 25
    :align: center

    * - 
      - Coarse
      -
      - Fine
      - 
    * - Parameter
      - Calibrated Value
      - Percent Error
      - Calibrated Value
      - Percent Error
    * - Yield Rate Coefficient
      - 10^2.983 (1/s)
      - -0.567
      - 10^3.0176
      - 0.587
    * - Yield Rate Exponent
      - 10.095
      - -0.538
      - 10.032
      - -0.090
    * - Yield Stress
      - 301.62 (MPa)
      - 0.063
      - 301.93
      - 0.166
    * - Hardening Exponent
      - 1.8110
      - -0.237
      - 1.8109
      - -.242
    * - Temperature Hardening Coefficient 1
      - 3517.5 (MPa)
      - -0.652
      - 3532.53
      - -.228
    * - Temperature Hardening Coefficient 2
      - 3.7095 (MPa/K)
      - -1.711
      - 3.7586
      - -.411

.. _tab_x_sym_hwd_6_8:

.. list-table:: HWD Synthetic Calibration Results D6 P8
    :header-rows: 2
    :widths: 25 25 25 25 25
    :align: center

    * - 
      - Coarse
      -
      - Fine
      - 
    * - Parameter
      - Calibrated Value
      - Percent Error
      - Calibrated Value
      - Percent Error
    * - Yield Rate Coefficient
      - 10^ 2.9784 (1/s)
      - -0.720
      - 10^3.016 (1/s)
      - 0.533
    * - Yield Rate Exponent
      - 10.111
      -  0.697
      - 10.031
      - -0.0996
    * - Yield Stress
      - 300.38(MPa)
      - -0.348
      - 301.92 (MPa)
      - 0.163
    * - Hardening Exponent
      - 1.8146
      -  -0.039
      - 1.8117
      - -0.120
    * - Temperature Hardening Coefficient 1
      - 3517.4 (MPa)
      - -0.655
      - 3532.8 (MPa)
      - -0.2203
    * - Temperature Hardening Coefficient 2
      - 3.7045 (MPa/K)
      - -1.844
      - 3.7586 (MPa/K)
      - -0.411


Both configurations of HWD perform quite well across both levels of mesh refinement. With most parameters having less than one percent error with the largest error 
at around two percent. The largest error was present on the coarse no subdivisions configuration, which from the verification work would predict to be the poorest 
calibration configuration. For calibrating against noiseless synthetic data these results are very acceptable. From these results, it may seem that using either configuration 
of HWD would be appropriate for any calibration procedure, but in reality the no subdivisions case captures less detail of the full-field data than the six subdivision case. 
While both cases are sufficiently detailed to capture the some of the patterns present in the full-field data, and be can be used to calibrate. The six subdivision case 
captures more detail and would be more appropriate for a case where significant small scale behavior is paramount to the calibration, and not just larger trends. 

In addition, when a lower number of subdivision sets and polynomials  are used for HWD there is a risk of non-uniqueness in the latent space. 
If there are not sufficiently detailed modes in the latent space than multiple functions could map to the same values. For a simple example consider 
mapping a two dimensional line centered at the origin. If the only latent mode is the constant mode, both our line and a constant value of zero will 
have the same values in latent space. Some of this concern can be alleviated in many types of material calibration by the inclusion of traditional calibration metrics. 
These metrics will apply additional constraints that reduce the possibility of non-unique solutions to a calibration problem. 

The fine D6 P8 calibration has the best calibration results in addition to some of the best calibration efficiency amongst the models, :numref:`tab_x_sym_hwd_stats`.
Thus, while coarser meshes can be used with HWD, there can be significant benefits to calibration quality and efficiency by using appropriately resolved meshes. 

.. _tab_x_sym_hwd_stats:

.. list-table:: HWD Synthetic Calibration Stats
    :header-rows: 2
    :widths: 50 50 50 50 50 
    :align: center

    * - 
      - Coarse
      -
      - Fine
      - 
    * - Metric
      - D0P24
      - D6P8
      - D0P24
      - D6P8
    * - Exit Status
      - Relative Function Convergence
      - Relative Function Convergence
      - HPC Time Limit
      - Relative Function Convergence
    * - Iterations
      - 45
      - 23
      - 14
      - 10
    * - Wall Time(hr)
      - 29.0
      - 50
      - 72.0+
      - 29.4
    * - Average Total Model Time (CPU hr)
      - 41.1
      - 25.8
      - 176
      - 165


Interpolation Synthetic Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:numref:`tab_x_syn_interp` shows the results of the coarse mesh calibration to the synthetic data. 
Only coarse mesh data exists for the interpolation calibration. 
Running the calibration with the fine mesh causes an out of memory (OOM) error and the calibration is aborted. 
There are plans to address the OOM problems in MatCal for the interpolated calibration. 
Details about the memory concerns for interpolated calibrations are discussed in more detail at the end of this section. 

.. _tab_x_syn_interp:

.. list-table:: Interpolation Synthetic Calibration Results
    :header-rows: 2
    :widths: 25 25 25
    :align: center

    * - 
      - Coarse
      -
    * - Parameter
      - Calibrated Value
      - Percent Error
    * - Yield Rate Coefficient
      - 10^2.3457 (1/s)
      - -21.810
    * - Yield Rate Exponent
      - 9.3168
      - -7.212
    * - Yield Stress
      - 296.18 (MPa)
      - -1.742
    * - Hardening Exponent
      - 1.8522
      - 2.033
    * - Temperature Hardening Coefficient 1
      - 3496.6 (MPa)
      - -1.243
    * - Temperature Hardening Coefficient 2
      - 3.7046 (MPa/K)
      - -1.841

.. _tab_x_syn_interp_stats:

.. list-table:: Interpolation Synthetic Calibration Stats
    :header-rows: 2
    :widths: 50 50
    :align: center

    * - 
      - Coarse
    * - Metric
      - Value
    * - Exit Status
      - Relative Function Convergence
    * - Iterations
      - 17
    * - Wall Time(hr)
      - 23.5
    * - Average Total Model Time (CPU hr)
      - 22.9

The results for the coarse calibration vary dramatically based on which parameter is investigated. Outside of the yield rate coefficient and the yield rate exponent, the errors are within two percent. 
For a synthetic data problem without any noise, this is a bit high. We would expect that the percent error should be less than one percent. The yield rate coefficient and exponent are much worse
than the other parameters with percent errors of about 22 and 9 percent, respectively. 
Much of this error can be explained by the coarse mesh and the under resolution of 
sharp gradients in the displacement. Because the method relies on comparing displacement values directly, 
it needs very good representation of these localized displacements and this will drive up the memory cost of the simulation. 
If the errors arise due to poor mesh resolution,
the material parameters will need to adapt to attempt to bring the experimental and simulation results into better alignment.
This explains why the yield parameters are the most effected and are smaller. 
Smaller yield parameters for this model mean an earlier yield, which enables a greater amount of 'large' plastic deformation,
allowing for the material parameters to compensate for the mesh deficiencies. 

In the chosen model the value of the yield rate coefficient can dramatically effect the necessary values of other parameters. This dependence inherently makes the calibration more difficult. 
To test the interpolation on a simpler problem, we rerun the problem with the same model and data however the yield rate coefficient is specified to be the goal value. The results of the calibration 
are shown in :numref:`tab_x_syn_interp_fixed` along with the analogous HWD results. 

.. _tab_x_syn_interp_fixed:

.. list-table:: Interpolation Synthetic Calibration Results Fixed Yield Rate Coefficient
    :header-rows: 2
    :widths: 25 25 25 25 25
    :align: center

    * - 
      - Coarse: Interp
      -
      - Coarse: HWD D6 P8
      -
    * - Parameter
      - Calibrated Value
      - Percent Error
      - Calibrated Value
      - Percent Error
    * - Yield Rate Exponent
      - 10.040 
      - -0.010
      -  10.075 
      - 0.339
    * - Yield Stress
      - 297.641 (MPa)
      - -1.257
      - 300.96 (MPa)
      - -0.156
    * - Hardening Exponent
      - 1.8113
      - -0.220
      - 1.8159
      - -0.033
    * - Temperature Hardening Coefficient 1
      -  3467.04 (MPa)
      - -2.078
      - 3511.4 (MPa)
      - -0.824
    * - Temperature Hardening Coefficient 2
      - 3.6472 (MPa/K)
      - -3.362
      - 3.6844 (MPa/K)
      - -2.377


Compared to the full calibration, the overall error of the parameters is lower, with most of the reduction found in the yield parameters. However, a small increase in the error of the temperature hardening parameters is observed.
The general parameter error values are brought closer to the errors seen with an HWD calibration, which had much more acceptable performance at coarser mesh refinements than the interpolation objective. 

Interpolation Memory Considerations
"""""""""""""""""""""""""""""""""""
The source of the OOM problems comes from the shrinking of memory as the optimizer iterates, and the swelling of MatCal memory after an evaluation batch.

Interpolation residuals are the largest residuals for all the full-field methods implemented into MatCal. 
The residual has a length of the number of quantities of interest (QoI) times the number simulation nodes times the number of experimental time steps. 
For the fine mesh there are 2 QoIs, about 50500 surface nodes, and 100 time steps, leading to at least .1GB of data being occupied by the residual alone for one experimental data set.
With multiple experimental data sets and multiple calibration parameters one evaluation set can occupy well over 1GB of memory.  
Due to some interactions with MatCal and Dakota, the full history of the residual evaluations currently stay in memory.
Because memory is not currently released for large problems this can lead to memory congestion and the emergence of an OOM. 

These memory problems are exacerbated by MatCal's use of the exodus python library to import simulation results. Currently, the exodus library will load all 
data from an exodus file into memory. This can be a large amount of data that MatCal may or may not need, often many times larger than the residual. This coupled with the batch import system currently 
implemented in to MatCal, results in a large swelling of memory at the time of residual calculation. This temporary swelling of memory can exceed the available memory causing an OOM error. 

Future plans to improve the memory issue are:

#. Work with the Dakota team to improve the interface between Dakota and MatCal to ensure that we can release memory 
#. Reduce the required memory swelling, by limiting the number of fully loaded exodus files concurrently loaded in to memory. This can be done by moving QoI calculations to distributed processes or by serializing the load-calculate-unload workflow in MatCal. 

The following list contains guidance on how to minimize the memory issues in MatCal while the memory issue exists for the interpolation objective. 

#. Local Mesh Refinement: If possible only refine in regions that have high gradients to reduce the overall memory of the outputs, and the calibration residual. 
#. Reducing Simulation Output Time Steps: Each time step output is an additional set of all nodal values in the problem. If a simulation is outputting very often this can create large exodus imports into MatCal. Reducing the simulation output rate to a lower frequency can cut the memory requirements for a calibration dramatically.
#. Staggering Parameter Calibrations: In many optimization methods the evaluation batch size scales with the number of parameters being investigated. If possible, perform multiple calibrations where some of the parameters are frozen in each run, and the next run uses the updated results from the previous run.
#. Narrowing Region of Interest: The larger the surface used for the residual, the more memory it requires. By limiting the region used for calibration, less memory is lost each evaluation.
#. Down-selecting Experimental Frames: The residual scales with the number of experimental times, often experimental data has an excess of these recordings. By removing redundant frames of experimental data it is often possible to reduce the experimental data used and residual size by an order of magnitude. 


Full-field Calibration to X-specimen Experimental Data
------------------------------------------------------
In this section we show that MatCal's full-filed calibration tools can be used with real experimental data. 
The data for the these calibrations was collected for the Experimodelment LDRD, and details about its collection 
and processing can be found in the SAND Report :cite:p:`VFM_LDRD_Jones`. 
The set up for this experiment is the same as what was done for the synthetic data test, :ref:`X-specimen Synthetic Data Calibration Verification`,
but now we use actual experimental data where the parameter values from a given model are unknown and are not guaranteed to align with our proposed model form. 

The experiment data provided for this calibration is a collection of uniaxial tension data using the X-specimen. From the data collected in the LDRD 
we used two pull rates detailed in :numref:`tab_x_exp_pull_rates`. The fixture force-extension curves for the two rates are shown in :numref:`fig_x_exp_load_disp`, 
and the final frame of displacements across the sample is shown in :numref:`fig_x_exp_disp`. The displacement plots indicate that there is 
significant inhomogeneous displacement gradients though the model. 

.. _tab_x_exp_pull_rates:

.. list-table:: Experimental Pull Rates
    :header-rows: 1
    :widths: 25 25
    :align: center

    * - Name
      - Extension Rate (mm/s)
    * - Slow
      - .005
    * - Fast
      - 5.0

.. _fig_x_exp_load_disp:

.. figure:: figures/X_load_disp.png
    :scale: 45%
    :alt: Force-extension plots for the two deformation rates 
    :align: center

    Force-extension plots for the slow and fast rate experiments. 

.. _fig_x_exp_disp:

.. figure:: figures/X_disp.png
    :scale: 20%
    :alt: Final Displacements 
    :align: center

    Final displacements in the X and Y for the X-specimen for the two experimental rates chosen. 

Like in the synthetic section, we use Dakota's ``nl2sol`` algorithm to calibrate the material. 
This method was chosen for the following reasons:

#. Data sets generated from full-field calibrations can be large, and nl2sol has a small batch size requirement compared to a global optimization technique. 
#. Success was seen in the synthetic case with this methods, and having a consistent comparison allows for more direct comparison. 
#. Gradient methods can converge in a few number of evaluations compared to more exhaustive methods.
    The fine scale mesh has a relatively high computational and nl2sol can minimize the number of concurrent runs required and thus reduce the time required to wait in computing queues.

The only significant change made to the calibration is the removal of the Yield Rate Coefficient parameter.
Generally three extension rates separated by a several decades are required to identify the Yield Rate Coefficient parameter.
Only two rates are available, and thus the parameter was preassigned to be 3.0.

It will be shown in the following sections that a gradient decent method is not robust enough to calibrate this experimental data set. 
Due to the data and the material model chosen, there appear to be many local minima that the parameter set can 
get stuck in. A 'poor' starting point was initially selected for these parameters to emulate what might happen 
if an analyst were to use these tools directly with out any precalculations. 

.. note::
    Work is being conducted on more robust calibration techniques for this problem and will be documented in future releases. 

The major conclusions from performing full-field calibrations on real experimental data are:

#. MatCal's full-field objectives can correctly compare real full-field data sets to simulation predictions, enough to 
   Improve a model's fit from the initial guess. In this case the calibration method was not good enough to overcome the
   challenges present in the data and model form, but a better answer was obtained. 
#. If data appropriate for VFM calibrations is available then VFM is an excellent choice. 
   VFM is very fast allowing one to obtain results in a very short period of time.
   Due to VFM's formulation it has a strong sensitivity to material parameters it to find an appropriate minima efficiency.
#. Model form error becomes more apparent using full-field data in junction with probe data. 
   In several of the objective histories it was seen that the full-field and force-extension objectives often 
   oppose each other. This could be because there are incongruities between the field displacements and the load predictions 
   made by the model vs the data(the model is not of the correct form to capture both the field and probe data correctly).
   Because if this trait, we recommend using full-field data in a looped workflow with conventional data. 
   Where initially a calibration is done with conventional data alone, and then a second one is performed incorporating 
   full-field data. The differences in parameter calibrations should provide an indication to the degree of model form 
   misfit. Then these results can be used to revise the model form and then the loop can be repeated until the desired 
   level of agreement is reached. 
#. Mesh refinement is important for full-field data. Where conventional calibrations only require levels of refinement sufficient for 
   the convergence of QoIs, full-field data requires a level of mesh refinement sufficient for the field values. 
#. If experimental data and simulation data can be made small enough, it is recommended that a global calibration be used to initially fit the 
   data. These methods are less likely to get stuck poorly performing local minima, which is a problem with gradient decent methods. 
#. With nl2sol starting at an uninformed starting point, full-field + conventional objectives generate better results than conventional objectives alone.
    However, all methods other than VFM generated poor final results. 



Conventional Experimental Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A conventional calibration was preformed to observe its performance against that of the full-field calibrations. 
In this calibration only the force-extension data was used. The calibration predictions and statistics are shown in :numref:`tab_x_exp_conv` and :numref:`tab_x_exp_conv_stats`.
The results for this calibration are poor, due to stagnation of the objective in a local minima.
This is a the result from the choice of model form, complex data, and the use of a local calibration method. 

.. _tab_x_exp_conv:

.. list-table:: Conventional Experimental Calibration Results
    :header-rows: 2
    :widths: 25 25 
    :align: center

    * - 
      - Fine
    * - Parameter
      - Value
    * - Yield Rate Exponent
      - 18.567
    * - Yield Stress
      - 165.5 (MPa)
    * - Hardening Exponent
      - 5.978
    * - Temperature Hardening Coefficient 1
      - 2972.7 (MPa)
    * - Temperature Hardening Coefficient 2
      - 3.492 (MPa/K)

.. _tab_x_exp_conv_stats:

.. list-table:: Conventional Experimental Calibration Stats
    :header-rows: 2
    :widths: 50 50 
    :align: center

    * - 
      - Fine
    * - Parameter
      - Value
    * - Exit Status
      - Time Limit
    * - Iterations
      - 23
    * - Wall Time(hr)
      - 72.0+
    * - Average Total Model Time (CPU hr)
      - 282


VFM Experimental Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In contrast to the other methods, the VFM calibrations performed
fairly well. As was seen in :ref:`Objective Sensitivity Study`, 
the VFM objective has proven well suited for use with gradient methods
even with experimental data. 
The calibration stats for the calibration are shown in 
:numref:`tab_x_exp_vfm_stats`. 

.. _tab_x_exp_vfm_stats:

.. list-table:: VFM Experimental Calibration Stats
    :header-rows: 1
    :widths: auto
    :align: center

    * - Exit Status
      - Iterations
      - Wall Time (hr)
      - Average Total Model Time (CPU hr)
    * - Relative Function Convergence
      - 18
      - 5.6
      - 3.77
      

The calibrated converged
after 18 iterations and the objective function 
dropped by a factor of twenty. 
The calibrated parameters using VFM are shown in :numref:`tab_x_sym_vfm_fixed_yield_rate_exponent`.

.. _tab_x_sym_vfm_fixed_yield_rate_exponent:

.. list-table:: VFM Experimental Calibration Results 
    :header-rows: 1
    :widths: auto
    :align: center

    * - Parameter
      - Calibrated Value
      - Difference From Conventional(%)
    * - Yield Rate Exponent
      - 10.15 (1/s)
      - 1.1
    * - Yield Stress
      - 285.8 (MPa)
      - -5.2
    * - Hardening Exponent
      - 4.18
      - 130.1
    * - Temperature Hardening Coefficient 1
      - 3137.4 (MPa)
      - -11.4
    * - Temperature Hardening Coefficient 2
      - 0.56 (MPa/K)
      - -85

Although the parameters are different than the traditional calibration, 
we cannot know which parameter set is better without 
testing them in validation simulations.
Future work will include validation simulations where these 
parameters are used to predict experiments for 
different geometry under similar loading. 
The quality of the calibrated parameter will 
be determined by how well the parameter sets predict
the validation geometry behavior.

The VFM internal power plotted with the VFM external power 
are shown in :numref:`fig_VFM_x_exp_qois_slow`
and :numref:`fig_VFM_x_exp_qois_fast`

.. _fig_VFM_x_exp_qois_slow:

.. figure:: figures/VFM_x_exp_cal/VFM_0_slow_objective_0.png
    :alt: VFM Slow Experimental Calibration QoIs
    :align: center

    The VFM simulation internal virtual power is shown with the
    experiment external virtual power as a function of time 
    for the slow test.  

.. _fig_VFM_x_exp_qois_fast:

.. figure:: figures/VFM_x_exp_cal/VFM_0_fast_objective_0.png
    :alt: VFM Fast Experimental Calibration QoIs
    :align: center

    The VFM simulation internal virtual power is shown with the
    experiment external virtual power as a function of time 
    for the fast test.  

The VFM results show good alignment with their experiment QoIs, unlike all the other calibrations. 
The objective history is shown in :numref:`fig_VFM_x_exp_objective_history`.

.. _fig_VFM_x_exp_objective_history:

.. figure:: figures/VFM_x_exp_cal/objective_history_VFM_0.png
    :alt: VFM Experimental Calibration Objective History 
    :align: center

    The objective value is plotted for each evaluation.

The good agreement between the virtual power QoIs and
the low objective value suggest the global 
minimum was found for this calibration using the 
VFM techniques.


HWD Experimental Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HWD calibrations were performed on the same for configurations seen used in the :ref:`HWD Synthetic Calibration`.
The calibration predictions and statistics are shown in :numref:`tab_x_exp_hwd` and :numref:`tab_x_exp_hwd_stats`.
The results for this calibration are poor, due to stagnation of the objective in a local minima.
This is a the result from the choice of model form, complex data, and the use of a local calibration method. 

The addition of full-field data in the form of HWD was not sufficient to overcome the potential stagnation points present in the calibration setup. 
While the results can be improved, this work shows that HWD can be applied to real experimental data and can drive an objective down. 

.. _tab_x_exp_hwd:

.. list-table:: HWD Experimental Calibration Results
    :header-rows: 2
    :widths: 25 25 25 25 25
    :align: center

    * - 
      - Coarse
      -
      - Fine
      - 
    * - Parameter
      - D0/P24
      - D6/P8
      - D0/P24
      - D6/P8
    * - Yield Rate Exponent
      - 30.000
      - 30.000
      - 11.978
      - 17.007
    * - Yield Stress (MPa)
      - 100.0 
      - 100.0 
      - 234.5
      - 195.7
    * - Hardening Exponent
      - 3.531
      - 3.559
      - 4.187
      - 4.917
    * - Temperature Hardening Coefficient 1 (MPa)
      - 3290.7 
      - 3295.2 
      - 2884.6 
      - 3803.5 
    * - Temperature Hardening Coefficient 2 (MPa/K)
      - 0.000 
      - 0.000 
      - 4.215 
      - 5.052 

.. _tab_x_exp_hwd_stats:

.. list-table:: HWD Experimental Calibration Stats
    :header-rows: 2
    :widths: 50 50 50 50 50 
    :align: center

    * - 
      - Coarse
      -
      - Fine
      - 
    * - Metric
      - D0P24
      - D6P8
      - D0P24
      - D6P8
    * - Exit Status
      - Relative Function Convergence
      - Relative Function Convergence
      - Time Limit
      - False Convergence
    * - Iterations
      - 9
      - 12
      - 20
      - 14
    * - Wall Time(hr)
      - 15.5
      - 17.6
      - 72.0+
      - 68.3
    * - Average Total Model Time (CPU hr)
      - 43.3
      - 31.2
      - 246
      - 278


Interpolation Experimental Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interpolation calibrations were performed on the coarse X-specimen mesh, this was done to avoid OOM issues as was seen in the synthetic test case.
The calibration predictions and statistics are shown in :numref:`tab_x_exp_interp` and :numref:`tab_x_exp_interp_stats`.
The results for this calibration are poor, due to stagnation of the objective in a local minima.
This is a the result from the choice of model form, complex data, and the use of a local calibration method. 

The addition of full-field data in the form of interpolation was not sufficient to overcome the potential stagnation points present in the calibration setup. 
While the results can be improved, this work shows that GMLS interpolation can be applied to real experimental data and can drive an objective down.  

.. _tab_x_exp_interp:


.. list-table:: Interpolation Experimental Calibration Results
    :header-rows: 2
    :widths: 25 25 
    :align: center

    * - 
      - Coarse
    * - Parameter
      - P2/R2
    * - Yield Rate Exponent
      - 5.0
    * - Yield Stress (MPa)
      - 128.76
    * - Hardening Exponent
      - 7.214
    * - Temperature Hardening Coefficient 1 (MPa)
      - 5000.0
    * - Temperature Hardening Coefficient 2 (MPa/K)
      - 0.000

.. _tab_x_exp_interp_stats:

.. list-table:: HWD Experimental Calibration Stats
    :header-rows: 2
    :widths: 50 50 
    :align: center

    * - 
      - Coarse
    * - Parameter
      - P2/R2
    * - Exit Status
      - Relative Function Convergence
    * - Iterations
      - 12
    * - Wall Time(hr)
      - 16.1
    * - Average Total Model Time (CPU hr)
      - 27.2
