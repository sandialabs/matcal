*********************
Full-field Data Tools
*********************

The proliferation of full-field data acquisition systems and tools, such as digital image correlation (DIC) and infrared (IR) cameras, provides a plethora of data to be used for model calibration and validation activities. 
Generally, full-field data is more complicated to work with than conventional probe data, because of this it has conventionally only been used for qualitative comparisons.
Full-field data sets are noticeably more difficult to work with than probe based data sets because of two main factors. The first factor is the additional dimensionality of the data, 
where conventional data only requires alignment in time(or some analogous measure), full-field data requires alignment in both space(typically two dimensional) and time.
The second factor is the sizes of data involved in full-field calculations. Where conventional experimental measurements and simulation predictions only produce data sets with a few columns and many rows,
full-field data sets produce data sets with many columns and many rows. This increase in data size takes what was a trivial memory requirement on modern computers, and makes it a significant concern.
These two challenges make writing a full-field calibration code difficult, and with few appropriate external libraries to help. 

Full-field data used in concert with probe measurements has the potential to provide much more insight in to the material behavior than with probe measurements alone. 
Potential benefits of using full-field data include:

#. Potential to reduce the number of experiments required to characterize data. Full-field data collection techniques extract more information per experiment, 
   potentially allowing researchers to gain the necessary amount of information to characterize a material with less experiments. 
#. Increased model form scrutiny. Probe measurements often provide an integrated, homogenized, or partial measurement of the changes happening during an experiment. 
   Full-field data provides many local experimental observations that provide a fuller measurement of the material evolution during an experiment.
   This fuller picture can reveal weaknesses in a material model's representation much more apparently than by probe measurements alone. 
   This clarity allows the analyst to identify potential weakness in their model and can take steps to address them. 
#. Enabling of fast 'solver free' calibration techniques. Full-field data often provides the degrees of freedom that are solved for in engineering software. 
   This information allows for the use of techniques that avoid much of the necessary calculations done in these codes, allowing for calibrations to finish in
   a fraction of the time compared to conventional calibrations. 

To leverage the benefits that full-field data can provide MatCal has three full-field data analysis methods:

#. Meshless Interpolation
#. Virtual Fields Method
#. Hierarchal Wavelet Decomposition

These methods range from common to novel and provide tools that quantitatively compare full-field data 
from two different sources which enables model calibration and validation.
In addition to providing calibration methods for full-field data, MatCal includes tools for easier manipulation 
and of surface full-field data. 

All of these tools can be used for the following tasks:

#. Material model calibration with MatCal calibration studies. 
   See :mod:`~matcal.dakota.local_calibration_studies` and
   :mod:`~matcal.dakota.global_calibration_studies`.
#. Model validation by comparing model results to experimental
   using MatCal objectives. 
   See :class:`~matcal.full_field.objective.InterpolatedFullFieldObjective`, 
   :class:`~matcal.full_field.objective.PolynomialHWDObjective` and
   :class:`~matcal.full_field.objective.MechanicalVFMObjective`.
#. Model parameter sensitivity and uncertainty quantification studies. 
   See :mod:`~matcal.dakota.sensitivity_studies`, :mod:`~matcal.dakota.uncertainty_quantification_studies`
   and :class:`~matcal.core.parameter_studies.LaplaceStudy`.


.. include:: full_field_data_alignment_warning.inc

.. include:: Full-field Interpolation.inc

.. include:: Virtual Fields Method.inc

.. include:: Polynomial HWD Theory.inc

.. include:: full_field_verification_examples/index.rst
   :start-after: :orphan:   

.. include:: full_field_study_verification_examples/index.rst
   :start-after: :orphan:   


Full-field Method Comparison and User Guide
===========================================
The examples shown in :ref:`Full-field Study Verification`
are useful for showing the pros and cons of each of the methods.
The results can be distilled down to the following observations
and guidelines.

Virtual Fields Method Use Guidelines
------------------------------------
When and how to use VFM:

#. Use VFM when the material can be easily identified in 
   characterization tests where a plane 
   stress assumption is valid.
#. Make sure to obtain full-field measurement 
   data for at least the in plane displacements.
#. Start with gradient methods for calibration 
   when using VFM. As seen in 
   :ref:`Objective Sensitivity Study`, the VFM objective has a sharp and 
   deep valley toward the optimum for its objective. 
#. If obtaining new experimental data, 
   work closely with your experimentalist to ensure 
   the the specimen is not rotated out of plane 
   and that the plane stress assumption can be applied 
   to the test specimen.

Issues to be aware of:

#. With the previous being stated, we note that 
   the VFM Models and objective can introduce errors in 
   calibrated parameters. In practice, 
   it is generally on the order of 1%, but can be as high as 
   5% as shown in the :ref:`Virtual Fields Calibration Verification - Three Data Sets`
#. Do not expect to be able to characterize more parameters with 
   fewer tests. The VFM tools in MatCal tend to over fit the 
   model to the available data. 
   As seen in :ref:`Virtual Fields Calibration Verification`, 
   the optimization will still calibrate all parameters but can 
   do so with large errors. We believe this is 
   due the plane stress assumption causing model form error.
   Be certain the data provided can adequately identify the 
   model parameters.

Polynomial HWD and Full-field Interpolation Use Guidelines
----------------------------------------------------------
These methods should be considered
only after it has been determined VFM is
not viable. This may be due to only 
having access to characterization tests
that do not conform to the plane stress assumption, 
attempting to fit a more complicated model 
to limited full-field test data than originally intended, 
or attempting to calibrate parameters that govern 
behavior after plastic localization. 

When and how to use:

#. Use default settings unless 
   you have performed your own verification 
   as done in :ref:`Polynomial HWD Verification - X Specimen Data`
   and :ref:`Full-field Interpolation Verification`
#. Consider what is added to the full-field objective 
   carefully as memory can be an issue. It is recommended
   to carefully select time steps of interest as was done 
   for these methods in :ref:`Full-field Study Verification`.
#. Carefully setup your model to ensure robustness 
   and timely execution. See 
   :ref:`Full-field Verification Problem Modeling Information`
#. Do not use full-field interpolation if you are 
   calibrating to many time steps or using a non-gradient methods. Running out 
   of memory is likely.
#. The full-field interpolation objective may produce better results 
   when using gradient methods than the Polynomial HWD 
   objective.
#. The recommended workflow would be to use a Polynomial HWD objective
   with a non-gradient method to identify areas in the parameter space
   where the objective is low. Then pick some of those locations 
   as initial points for starting gradient 
   calibrations with full-field interpolation and a single full-field 
   time step for comparison.
#. When using the full-field interpolation objective, 
   using :meth:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy.do_not_save_evaluation_cache`
   will help avoid out of memory errors. However, it will 
   also prohibit the use of restarts. Consider using the HWD objective
   if you need restarts and run out of memory with the full-field
   interpolation objective.

Issues to be aware of:

#. Do not expect these methods to work well with gradient 
   methods. It may be worth attempting a gradient 
   based calibration initially, however, 
   non-gradient methods will be more robust. 
   We suggest :class:`~matcal.dakota.local_calibration_studies.SolisWetsCalibrationStudy`
   if computational resources are limited
   and :class:`~matcal.dakota.global_calibration_studies.SingleObjectiveGACalibrationStudy`
   otherwise.
#. Do not use full-field interpolation unless the model output
   and experiment data file sizes are on the order of 10 megabytes. 
   With non-gradient methods for calibration, the objective 
   will likely be evaluated many times concurrently and the likelihood of 
   out-of-memory errors increases.
   Polynomial HWD will significantly reduce memory use and 
   should produce similar results.
#. Polynomial HWD must have enough points in each subdivision 
   to support the polynomial fit to that subdivision. 
   If the calibration performs poorly, consider 
   conducting a study such as 
   :ref:`Polynomial HWD Verification with Colocated Points`
   or :ref:`Polynomial HWD Verification - Analytical Function`
   to verify the transform is valid for your 
   discretization. 


