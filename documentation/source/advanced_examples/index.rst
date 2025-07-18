:orphan:

#################
Advanced Examples
#################
In this gallery of examples, we show how to use MatCal
to perform calibration with real experimental data. MatCal 
has several calibration, uncertainty quantification and
sensitivity study tools to use, and these examples demonstrate
how to use them when applied to data from different materials. As the features are 
developed and documented more feature examples will be added
here to highlight their use.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

304L annealed bar viscoplastic calibrations
###########################################
In this calibration example, we will calibrate 
a plasticity model using a rate dependent von Mises yield surface with 
Voce hardening :cite:p:`voce1948relationship`
to the Ductile Failure data for 304L bar material :cite:p:`laser_weld_paper`.
The experimental data set
used for this calibration 
was purposefully taken such that this viscoplastic
behavior
would be characterized. 

This example has been broken into six steps:

#. Experimental data analysis to aid in model form choice. 
   :ref:`304L bar data analysis`
#. Initial guess estimation for the plasticity parameters 
   using MatFit.
   :ref:`304L bar calibration initial point estimation`
#. Calibration of the material as the averaged 
   parameter set to all data. 
   :ref:`304L stainless steel viscoplastic calibration`
#. A mesh convergence study after the calibration to verify that the 
   calibration is valid. 
   :ref:`304L stainless steel mesh and time step convergence`
#. A verification that the model options chosen for the calibration
   capture what is necessary for the model to be accurate.
   :ref:`304L calibrated round tension model - effect of different model options`
#. Uncertainty quantification of the parameters 
   using MatCal's laplace study. 
   :ref:`304L stainless steel viscoplastic calibration uncertainty quantification` 
#. Validation of the calculated parameter 
   uncertainties by pushing samples from the
   uncertain parameter distributions back through 
   the models and comparing the results to the experimental data. 
   :ref:`304L stainless steel viscoplastic uncertainty quantification validation` 




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. BatchDataImporter     #. plot      #. determine_pt2_offset_yield">

.. only:: html

  .. image:: /advanced_examples/304L_viscoplastic_calibration/images/thumb/sphx_glr_plot_304L_a_data_analysis_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_304L_viscoplastic_calibration_plot_304L_a_data_analysis.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">304L bar data analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Running MatFit">

.. only:: html

  .. image:: /advanced_examples/304L_viscoplastic_calibration/images/thumb/sphx_glr_plot_304L_b_initial_point_estimation_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_304L_viscoplastic_calibration_plot_304L_b_initial_point_estimation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">304L bar calibration initial point estimation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="With our material model chosen and initial points determined,  we can setup a final full finite element calibration to  get a best fit to the available data.">

.. only:: html

  .. image:: /advanced_examples/304L_viscoplastic_calibration/images/thumb/sphx_glr_plot_304L_c_tension_calibration_cluster_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_304L_viscoplastic_calibration_plot_304L_c_tension_calibration_cluster.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">304L stainless steel viscoplastic calibration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Uniaxial Tension Models     #. RoundUniaxialTensionModel     #. ObjectiveResults     #. ParameterStudy">

.. only:: html

  .. image:: /advanced_examples/304L_viscoplastic_calibration/images/thumb/sphx_glr_plot_304L_d_tension_convergence_study_cluster_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_304L_viscoplastic_calibration_plot_304L_d_tension_convergence_study_cluster.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">304L stainless steel mesh and time step convergence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Uniaxial Tension Models     #. RoundUniaxialTensionModel     #. ObjectiveResults     #. ParameterStudy">

.. only:: html

  .. image:: /advanced_examples/304L_viscoplastic_calibration/images/thumb/sphx_glr_plot_304L_e_tension_model_option_effects_cluster_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_304L_viscoplastic_calibration_plot_304L_e_tension_model_option_effects_cluster.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">304L calibrated round tension model - effect of different model options</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Uniaxial Tension Models     #. PythonModel     #. LaplaceStudy      In this example, we will use MatCal&#x27;s LaplaceStudy to estimate the parameter uncertainty for the calibration. ">

.. only:: html

  .. image:: /advanced_examples/304L_viscoplastic_calibration/images/thumb/sphx_glr_plot_304L_f_tension_laplace_study_cluster_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_304L_viscoplastic_calibration_plot_304L_f_tension_laplace_study_cluster.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">304L stainless steel viscoplastic calibration uncertainty quantification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Uniaxial Tension Models     #. PythonModel     #. sample_multivariate_normal">

.. only:: html

  .. image:: /advanced_examples/304L_viscoplastic_calibration/images/thumb/sphx_glr_plot_304L_g_uq_validation_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_304L_viscoplastic_calibration_plot_304L_g_uq_validation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">304L stainless steel viscoplastic uncertainty quantification validation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /advanced_examples/304L_viscoplastic_calibration/plot_304L_a_data_analysis
   /advanced_examples/304L_viscoplastic_calibration/plot_304L_b_initial_point_estimation
   /advanced_examples/304L_viscoplastic_calibration/plot_304L_c_tension_calibration_cluster
   /advanced_examples/304L_viscoplastic_calibration/plot_304L_d_tension_convergence_study_cluster
   /advanced_examples/304L_viscoplastic_calibration/plot_304L_e_tension_model_option_effects_cluster
   /advanced_examples/304L_viscoplastic_calibration/plot_304L_f_tension_laplace_study_cluster
   /advanced_examples/304L_viscoplastic_calibration/plot_304L_g_uq_validation

6061T6 aluminum plate calibrations
##################################
In this calibration example, we will calibrate 
the Hill48 orthotropic
yield surface :cite:p:`hill1948theory` with 
Voce hardening :cite:p:`voce1948relationship`
to the Ductile Failure aluminum data for a rolled plate.
The rolling process tends to impart a texture 
to the material microstructure that results in 
orthotropic plasticity behavior. The experimental data set
used for this calibration 
was purposefully taken such that this orthotropic behavior
would be characterized. 

This example has been broken into several steps:

#. Experimental data analysis to verify the suspected orthotropic
   plasticity is present in the material. 
   :ref:`6061T6 aluminum data analysis`
#. Initial guess estimation for the plasticity parameters 
   using MatFit and engineering judgment.
   :ref:`6061T6 aluminum anisotropy calibration initial point estimation`
#. Calibration of the material as the averaged 
   parameter set to all data. 
   :ref:`6061T6 aluminum calibration with anisotropic yield`
#. Data analysis to examine the effect of temperature on 
   the material behavior and determine a material model 
   form that is temperature dependent. 
   :ref:`6061T6 aluminum temperature dependent data analysis`
#. Initial point estimation using MatFit for the material model temperature 
   dependence parameters.
   :ref:`6061T6 aluminum temperature calibration initial point estimation`
#. Calibration of the material model temperature dependence 
   using MatCal.
   :ref:`6061T6 aluminum temperature dependent calibration`
#. Uncertainty quantification of the parameters 
   using MatCal's laplace study. 
   :ref:`6061T6 aluminum calibration uncertainty quantification` 
#  Validation of the calculated parameter 
   uncertainties by pushing samples from the
   uncertain parameter distributions back through 
   the models and comparing the results to the experimental data.
   :ref:`6061T6 aluminum uncertainty quantification validation` 




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. BatchDataImporter     #. plot      #. determine_pt2_offset_yield">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_a_anisotropy_data_analysis_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_a_anisotropy_data_analysis.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum data analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we use MatFit and engineering judgement to estimate the  initial point for our calibration in  6061T6 aluminum calibration with anisotropic yield. See that example for more detail on material model  choice and experimental data review for the material.">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_b_anisotropy_initial_point_estimation_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_b_anisotropy_initial_point_estimation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum anisotropy calibration initial point estimation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Uniaxial Tension Models     #. RoundUniaxialTensionModel     #. GradientCalibrationStudy     #. UserFunctionWeighting">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_c_anisotropy_calibration_cluster.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum calibration with anisotropic yield</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="#. How the material anisotropy is affected by temperature. #. How the material plasticity is affected by temperature.">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_d_temperature_dependent_data_analysis_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_d_temperature_dependent_data_analysis.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum temperature dependent data analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Running MatFit     #. FileData    ">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_e_temperature_dependent_initial_point_estimation_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_e_temperature_dependent_initial_point_estimation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum temperature calibration initial point estimation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Uniaxial Tension Models     #. RoundUniaxialTensionModel     #. GradientCalibrationStudy     #. UserFunctionWeighting">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_f_temperature_dependent_calibration_cluster_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_f_temperature_dependent_calibration_cluster.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum temperature dependent calibration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Uniaxial Tension Models     #. RoundUniaxialTensionModel     #. ParameterStudy      We will perform this verification by running the model at  many temperatures over our temperature range and inspecting the results.  To do this, we will generate fictitious boundary condition data at  all temperatures of interest with independent states. As with the calibrations in this example suite, these data will have state variables of  temperature and direction. We will then run a  ParameterStudy with  the appropriate  RoundUniaxialTensionModel in the evaluation set. The study will run a single evaluation  with parameter values from the results of  6061T6 aluminum temperature dependent calibration and 6061T6 aluminum calibration with anisotropic yield. Once all states are complete, we will plot the result and  visually inspect the curves to verify the behavior is as desired.">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_g_temperature_dependent_verification_cluster_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_g_temperature_dependent_verification_cluster.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum temperature dependence verification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This uncertainty quantification study differs from  the one in 304L stainless steel viscoplastic calibration uncertainty quantification because many of the parameters are directly correlated to other parameters in the  model. Specifically, the temperature and anisotropy parameters, are  multipliers of the yield stress and Voce hardening parameters. As a result, we will assume all parameter uncertainty can be attributed to the yield stress and Voce hardening parameters alone. This will significantly  reduce the cost of the finite difference calculations needed  for the LaplaceStudy and ensure robustness  of the method. ">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_h_laplace_study_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_h_laplace_study.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum calibration uncertainty quantification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    #. Uniaxial Tension Models     #. Top Hat Shear Model     #. ParameterStudy     #. sample_multivariate_normal              To begin, we import the tools we need for this study and setup the  data and model as we did in 6061T6 aluminum calibration with anisotropic yield.">

.. only:: html

  .. image:: /advanced_examples/6061T6_anisotropic_calibration/images/thumb/sphx_glr_plot_6061T6_i_uq_validation_parameter_study_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_i_uq_validation_parameter_study.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6061T6 aluminum uncertainty quantification validation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_a_anisotropy_data_analysis
   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_b_anisotropy_initial_point_estimation
   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_c_anisotropy_calibration_cluster
   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_d_temperature_dependent_data_analysis
   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_e_temperature_dependent_initial_point_estimation
   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_f_temperature_dependent_calibration_cluster
   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_g_temperature_dependent_verification_cluster
   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_h_laplace_study
   /advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_i_uq_validation_parameter_study

SIERRA User Defined Model Studies
#################################
In this example sub-gallery, we walk 
through using the :class:`~matcal.sierra.models.UserDefinedSierraModel`
to perform MatCal studies. Careful model preparation is required 
to ensure a successful study that behaves as expected. 



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="MatCal has no native ability to perform physics calculations,  therefore, this needs to be done  by an outside program. For this case we use the Sandia  thermal-fluids code SIERRA/Aria. Prior to running this calibration, we created and tested a SIERRA/Aria input  file and mesh file that represents our experimental configuration. The SIERRA/Aria input file  is named &#x27;two_material_square.i&#x27; and the mesh file  is named &#x27;two_material_square.g&#x27;. After creating these  files, we prepare them for use in MatCal.">

.. only:: html

  .. image:: /advanced_examples/user_model_studies/images/thumb/sphx_glr_plot_user_supplied_calibration_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_user_model_studies_plot_user_supplied_calibration.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calibration of Two Different Material Conductivities</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="If you want more details about particular types of studies please see  MatCal Studies. ">

.. only:: html

  .. image:: /advanced_examples/user_model_studies/images/thumb/sphx_glr_plot_user_supplied_sensitivity_thumb.png
    :alt:

  :ref:`sphx_glr_advanced_examples_user_model_studies_plot_user_supplied_sensitivity.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Latin Hypercube Sampling to Obtain Local Material Sensitivities</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /advanced_examples/user_model_studies/plot_user_supplied_calibration
   /advanced_examples/user_model_studies/plot_user_supplied_sensitivity


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: advanced_examples_python.zip </advanced_examples/advanced_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: advanced_examples_jupyter.zip </advanced_examples/advanced_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
