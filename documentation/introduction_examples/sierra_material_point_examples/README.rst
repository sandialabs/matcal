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
