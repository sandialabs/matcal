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

