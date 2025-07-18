:orphan:

.. _introduction_examples-index:

*********************
Introduction Examples
*********************
After a brief introduction to Python 
and object oriented programming,
this set of examples covers basic calibrations 
using either PythonModels (see :class:`~matcal.core.models.PythonModel`) or SIERRA models.
The first set focus on Python models where calibrations are performed 
to data sets with known solutions. These are intended to 
show MatCal's basic input file structure and usage. A few also touch on 
calibration issues to be weary of for calibrations with 
analytical objectives that can easily be visualized for 
demonstration purposes.

These are then followed with more applied examples 
with similar goals using a MatCal 
generated SIERRA model for a uniaxially loaded 
material point (:ref:`Uniaxial Loading Material Point Model`). 
First a successful calibration is shown with justifications for 
decisions made and tools used for that calibration. Next, 
common issues that may be encountered in applied problems are 
explored using the same data set from the successful calibration.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This chapter serves as a basic primer into using Python and working with object-oriented code. If you are familiar with these concepts, feel free to skip this chapter. ">

.. only:: html

  .. image:: /introduction_examples/images/thumb/sphx_glr_a_python_primer_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_a_python_primer.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Basic Python Overview</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This section applies many of the topics introduced in the previous chapter  to a simple calibration of  the equation of a line (y=\text{m}x+\text{b}) to data. While the example is simple it contains all of the fundamental topics necessary  for more advanced calibrations. ">

.. only:: html

  .. image:: /introduction_examples/images/thumb/sphx_glr_plot_basic_example_walk_through_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_plot_basic_example_walk_through.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Linear Python Model Example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The fit is well defined in the sense that the error contours  are closed, and the bowl is steep; however, the model discrepancy  and the constraint of fixed E clearly have effects on the fit.  For instance, the changeover point in the bilinear “plasticity”  model is ambiguous as evidenced by the elongated diagonal error contours.  Also extrapolating the fitted model will not follow the data.  This issue is common in fitting real material models to data  that spans large ranges of, for example, strain, strain-rate and temperature.">

.. only:: html

  .. image:: /introduction_examples/images/thumb/sphx_glr_plot_issue_example-discrepancy_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_plot_issue_example-discrepancy.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Model Discrepancy Issue Example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Model Identifiability Issue Example">

.. only:: html

  .. image:: /introduction_examples/images/thumb/sphx_glr_plot_issue_example-identifiability_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_plot_issue_example-identifiability.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Model Identifiability Issue Example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The fits for the low and high noise cases look similar; however,  if the error as a function of the parameters is examined you can see:  (a) the optimum has shifted and (b) the bowl is flatter for the high noise case.  The noise induces bias in the optimal parameters and  makes the problem harder to solve. With enough noise  the calibration becomes useless. Also, it is apparent from  the shape of the error contours the two parameters Y and H  have correlated effects on the error, i.e., combinations of  high Y and low H have the same error as low Y and high H.  This is common in physics models and presents a tradeoff in  calibration that in its extreme becomes another issue  (which will be explored in another example  targeting &quot;identifiability&quot;).">

.. only:: html

  .. image:: /introduction_examples/images/thumb/sphx_glr_plot_issue_example-noise_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_plot_issue_example-noise.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Data Noise Issue - Low and High Noise Example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The model is a bilinear function similar to elastic-plastic  response with the parameter &quot;Y&quot; controlling where change from one linear  trend to the other takes place and the parameter &quot;H&quot; controlling the slope of the second trend.  The slope of the initial trend is fixed, as if it were known (as the elastic modulus typically would be). The problem differs from the referenced example when we generate the data for the model. We will use the model to generate data with uncertainty from two sources: noise and parameter uncertainty.">

.. only:: html

  .. image:: /introduction_examples/images/thumb/sphx_glr_plot_parameter_uncertainty_quantification_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_plot_parameter_uncertainty_quantification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter uncertainty example - external noise and internal variability</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /introduction_examples/a_python_primer
   /introduction_examples/plot_basic_example_walk_through
   /introduction_examples/plot_issue_example-discrepancy
   /introduction_examples/plot_issue_example-identifiability
   /introduction_examples/plot_issue_example-noise
   /introduction_examples/plot_parameter_uncertainty_quantification

SIERRA/SM Material Point Model Practical Examples
=================================================
In this section, we present the calibration of our :class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel`
to uniaxial compression data for a 6061-T6 aluminum from the Ductile Failure project at Sandia 
:cite:p:`DE_L2_Ductile_Failure`.
Specifically, we calibrate to test specimens C-RD-01 and C-ST-01 from that dataset. Four calibrations are presented to highlight 
different approaches to practical material calibration and potential issues that may arise. The four calibrations are:

#. A successful calibration to true stress/strain data with data preprocessing and objective weighting.
#. A failed calibration to true stress/strain data without data preprocessing and the default objective weighting.
#. A failed calibration to true stress/time data that becomes successful with minor data processing.
#. A successful calibration with two other model forms demonstrating how MatCal can aid in 
   material model choice while also demonstrating overfitting and model form error.

We use these four examples to highlight three important issues:

#. The methods in MatCal are robust but designed for specific situations. Calibrations 1 and 
   2 in this example will show how the :class:`~matcal.core.objective.CurveBasedInterpolatedObjective` 
   class should be used and how its requirement of having data with a monotonically increasing independent variable
   limits data that it can operate on.
#. The objective function is a quantitative measure of the quality of the calibration. Translating
   what you think is a quality calibration to a quantitative measure may not always be intuitive
   and must be done carefully at times. Calibrations 1, 
   3, and 4 demonstrate the importance of careful objective function specification.
#. Since the objective is a quantitative measure of calibration quality,
   it can be used to determine the best model form for a dataset if a suite of 
   calibrations with different model forms is completed. If one model 
   form produces a lower objective value after calibration, it *potentially* means it is the better model form. 
   However, care must be taken to avoid over fitting, so this is not a guarantee of improved model form.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="As stated previously, we present the calibration of our  UniaxialLoadingMaterialPointModel to uniaxial compression data for a 6061-T6 aluminum from the Ductile Failure Project at Sandia :citeDE_L2_Ductile_Failure. This example and calibration will consist of three steps:">

.. only:: html

  .. image:: /introduction_examples/sierra_material_point_examples/images/thumb/sphx_glr_plot_sierra_material_point_calibration_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Successful Calibration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this section, we once again present the calibration of our UniaxialLoadingMaterialPointModel to the uniaxial compression data. However, instead of manipulating the data and weighting the residuals like we did in the  sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py example, we ignore the potential issues that were pointed out there and use the data as provided. Overall,  the exact same process is used in this calibration as  was used in the successful calibration; however, the data cleanup and residual weights modification is not  performed initially. See the sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py for more of those details. First, we import and view the data: ">

.. only:: html

  .. image:: /introduction_examples/sierra_material_point_examples/images/thumb/sphx_glr_plot_sierra_material_point_calibration_uncorrected_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration_uncorrected.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calibration With Unmodified Data and Objective: A Simple Calibration Gone Wrong</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this section, we approach the calibration of the  UniaxialLoadingMaterialPointModel to the Ductile Failure 6061-T6 compression data a little differently. For the more traditional approach see the sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py. MatCal&#x27;s objective tools allow significant flexibility in how the objective for calibrations are built. Also, the UniaxialLoadingMaterialPointModel allows  flexibility in the way the boundary conditions are derived from data. In this example, we highlight  this flexibility by using strain versus time data to define the model boundary condition  so that it simulates both loading and unloading. We then define an objective with time as the independent variable and true stress as the dependent variable in the  CurveBasedInterpolatedObjective. In addition to the plasticity parameters calibrated in the previous example, we add the  elastic modulus as a calibrated parameter for this study. Generally, this is not  recommended as the isotropic elastic  properties of metals are readily available in the literature. However, we want  to make use of the additional information provided to the objective when including  the elastic unloading portion of  the data in the model. A more practical use of calibrating to stress-time history  would be calibrating a model to cyclical loading that has cycle dependent behavior such as calibrating a model with isotropic and kinematic hardening.">

.. only:: html

  .. image:: /introduction_examples/sierra_material_point_examples/images/thumb/sphx_glr_plot_sierra_material_point_calibration_with_unloading_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration_with_unloading.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calibration With Unloading: A Stress vs. Time Calibration Needs Attention to Detail</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This last section will explore calibrating different model forms to the Ductile Failure  6061-T6 aluminum compression data. We will show how the final objective function values  for a set of calibrated models can be used to help decide which is the best model form  for the data. However, we will also show how overfitting a model could provide a better  objective value over another model form but be a poor model for the material. We do this  with three additional calibrations for the following three model forms:">

.. only:: html

  .. image:: /introduction_examples/sierra_material_point_examples/images/thumb/sphx_glr_plot_sierra_material_point_calibration_z_model_forms_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration_z_model_forms.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Exploring Different Model Forms: For Better and Worse...</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /introduction_examples/sierra_material_point_examples/plot_sierra_material_point_calibration
   /introduction_examples/sierra_material_point_examples/plot_sierra_material_point_calibration_uncorrected
   /introduction_examples/sierra_material_point_examples/plot_sierra_material_point_calibration_with_unloading
   /introduction_examples/sierra_material_point_examples/plot_sierra_material_point_calibration_z_model_forms

Surrogate Studies
=================
In this set of examples, we show how to build surrogate models
using MatCal. These models then can be used as replacements for more 
expensive models in MatCal studies.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to generate a basic surrogate from a MatCal study. This example will cover:">

.. only:: html

  .. image:: /introduction_examples/surrogate_studies/images/thumb/sphx_glr_plot_generate_surrogate_thumb.png
    :alt:

  :ref:`sphx_glr_introduction_examples_surrogate_studies_plot_generate_surrogate.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Surrogate Generation Example</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /introduction_examples/surrogate_studies/plot_generate_surrogate


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: introduction_examples_python.zip </introduction_examples/introduction_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: introduction_examples_jupyter.zip </introduction_examples/introduction_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
