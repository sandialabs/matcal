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

