"""
Exploring Different Model Forms: For Better and Worse...
================================================================

This last section will explore calibrating different model forms to the Ductile Failure 
6061-T6 aluminum compression data. We will show how the final objective function values 
for a set of calibrated models can be used to help decide which is the best model form 
for the data. However, we will also show how overfitting a model could provide a better 
objective value over another model form but be a poor model for the material. We do this 
with three additional calibrations for the following three model forms:

#. Double Voce hardening 
#. Power law hardening without Luders strain 
#. Power law hardening with Luders strain.

The flow rule for Double Voce hardening is defined by:

.. math::
    \\sigma_f = Y + A_1\\left[1-\\exp\\left(-b_1\\varepsilon_p\\right)\\right] + A_2\\left[1-\\exp\\left(-b_2\\varepsilon\\right)\\right]

where :math:`Y` is the yield, :math:`\\varepsilon_p` is the material equivalent plastic strain
and :math:`A_1`, :math:`b_1`, :math:`A_2`, :math:`b_2` are the parameters for the two Voce hardening model components that 
we are using for this calibration. 
This hardening model allows for a little more flexibility when fitting material data. Generally, it results in a lower yield stress
and one of the Voce components is used to model the low strain portion of the curve just before yield while the other Voce component 
is used to model the high strain portion of the curve. 

The flow rule for 
power law hardening is 

.. math::
    \\sigma_f = Y + A\\left<\\varepsilon_p-\\varepsilon_L\\right>^n

where :math:`Y` is once again the material yield, :math:`A` is the hardening modulus,  
:math:`n` is the hardening exponent and :math:`\\varepsilon_L` is the Luders strain.
Power law hardening is a commonly used engineering material model
that cannot simulate materials with a clear saturation stress such as the aluminum being studied
here. We will look at two forms of this model, one where the Luders strain is set to zero and 
another where we allow it to calibrate the Luders strain to the data even though Luders strain 
is clearly not present in this material data set.

We begin this example as we have the others by first importing the calibration tools and the data 
before setting up the models.
"""
# sphinx_gallery_thumbnail_number = 5

from matcal import *
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)

data_collection = BatchDataImporter("uniaxial_material_point_data/*.csv").batch

#%%
# For these calibrations, we will follow the 
# :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py`
# example for data manipulation and overall objective setup since we know that setup works well.
# Our goal here is only to observe the effect of model form on the calibration results. The data and 
# objective preparation
# from that example is shown again here.
#

for state_data_list in data_collection.values():
  for data in state_data_list:
    data['time'] = data['time'] - data['time'][0]

boundary_data = data_collection["matcal_default_state"][1]
boundary_data = boundary_data[["engineering_strain"]]
boundary_data.set_name("dataset 1 derived BC data")
boundary_data_collection = DataCollection('boundary_data', boundary_data)
boundary_data_collection = scale_data_collection(boundary_data_collection, "engineering_strain", -1)

data_collection = scale_data_collection(data_collection, "true_strain", -1)
data_collection = scale_data_collection(data_collection, "true_stress", -1000)

objective = CurveBasedInterpolatedObjective('true_strain','true_stress')

def remove_high_and_low_strain_from_residual(true_strains, true_stresses, residuals):
    import numpy as np
    weights = np.ones(len(residuals))
    weights[(-true_strains > 0.5) | (-true_strains < 0.0035)] = 0
    return weights*residuals

residual_weights = UserFunctionWeighting("true_strain", "true_stress", remove_high_and_low_strain_from_residual)

objective.set_field_weights(residual_weights)

#%% 
# Now we can create a couple :class:`~matcal.sierra.material.Material` classes 
# and the corresponding material file for the calibrations to follow. 
# The input deck for the SIERRA/SM material models of 
# interest for is shown below::
# 
#     begin function double_voce
#         type is analytic
#         evaluate expression = "{Y*1e3}+{A1*1e3}*(1-exp(-{b1}*x))+{A2*1e3}*(1-exp(-{b2}*x))"
#         differentiate expression = "{A1*b1*1e3}*(exp(-{b1}*x))+{A2*b2*1e3}*(exp(-{b2}*x))"
#     end
#
#     begin material j2_double_voce
#         density = 0.000254
#         begin parameters for model j2_plasticity
#         youngs modulus                = 9.9e6
#         poissons ratio                =   0.33
#         yield stress                  = {Y*1e3}
#
#         hardening model = user_defined
#         hardening function = double_voce
#         end
#     end
#
#     begin material j2_power_law
#         density = 0.000254
#         begin parameters for model j2_plasticity
#         youngs modulus                = 9.9e6
#         poissons ratio                =   0.33
#         yield stress                  = {Y*1e3}
#
#         hardening model = power_law
#         hardening constant = {A*1e3}
#         hardening exponent = {n}
#         luders strain = {epsilon_l}
#         end
#     end
# 
# As was done in the 
# :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py`
# example, the elastic
# properties and density are pulled from MMPDS :cite:p:`MMPDS10`. 
# With this SIERRA/SM input saved in the current directory as "sierra_sm_multiple_hardening_forms.inc", 
# we can create the two :class:`~matcal.sierra.material.Material` objects and the 
# two :class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel` objects we will be using 
# for the calibrations. 
#

j2_double_voce = Material("j2_double_voce", "sierra_sm_multiple_hardening_forms.inc", "j2_plasticity")
j2_power_law = Material("j2_power_law", "sierra_sm_multiple_hardening_forms.inc", "j2_plasticity")

mat_point_model_DV = UniaxialLoadingMaterialPointModel(j2_double_voce)
mat_point_model_DV.add_boundary_condition_data(boundary_data_collection)
mat_point_model_DV.set_name("compression_mat_point_DV")

mat_point_model_PL = UniaxialLoadingMaterialPointModel(j2_power_law)
mat_point_model_PL.add_boundary_condition_data(boundary_data_collection)
mat_point_model_PL.set_name("compression_mat_point_PL")
mat_point_model_PL.add_constants(epsilon_l=0.0)

# %%
# .. note::
#      We use :meth:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel.add_constants` to set the "epsilon_l" parameter to zero
#      for the first round of the calibration. Since it is not included in the parameters for this first study, it will
#      be set to the constant value zero as a model constant. When we use it as a study parameter for the second calibration study, 
#      the model constant value will be overridden by the values specified during the study as described in the 
#      method's documentation linked above.
#


#%%
# Next, we setup the parameters and two of the calibration 
# studies we will be performing. Once again, we will 
# use a :class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`
# to perform the calibrations. We setup the 
# double Voce calibration followed by the power law 
# model calibration without Luders strain. Then we run the calibrations and 
# review the results.

Y = Parameter('Y', 20, 60, 40)
A1 = Parameter('A1', 0, 25, 12)
b1 = Parameter('b1', 5, 30, 20)
A2 = Parameter('A2', 0, 10, 5)
b2 = Parameter('b2', 30, 5000, 600)

calibration_DV = GradientCalibrationStudy(Y, A1, b1, A2, b2)
calibration_DV.set_results_storage_options(results_save_frequency=6)

Y = Parameter('Y', 30, 60, 50)
A = Parameter('A', 1, 5000, 1000)
n = Parameter('n', 0, 1, 0.5)

calibration_PL = GradientCalibrationStudy(Y, A, n)
calibration_PL.set_results_storage_options(results_save_frequency=4)
calibration_DV.add_evaluation_set(mat_point_model_DV, objective, data_collection)
calibration_PL.add_evaluation_set(mat_point_model_PL, objective, data_collection)

calibration_DV.set_core_limit(6)
calibration_PL.set_core_limit(4)

results_DV = calibration_DV.launch()
make_standard_plots("true_strain")
print(results_DV.best)
#%%
# The double Voce calibration completes with the Dakota output::
#   
#   ***** RELATIVE FUNCTION CONVERGENCE *****
#
# indicating that the algorithm completed successfully. Once again, from 
# the plots it is clear that the model matches the experimental 
# data well, and the final objective function value of around 0.005542 
# indicates an improved fit over the 
# :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py`
# example.
# The objective function has decreased by 19.9% with the double Voce model over the single Voce model.
# Also, the calibrated parameter values, show that the saturation stress is still approximately 55 ksi while
# the yield has decreased as expected to 35.9 ksi. 
# All of these indicate that the double Voce model is an improved model form over the single Voce form. 
# However, this would be further supported with validation data where the double Voce model was shown to 
# be more predictive than the single Voce model.
results_PL = calibration_PL.launch()
make_standard_plots("true_strain")
print(results_PL.best)


#%%
# Similarly, the power law calibration completes with the Dakota output::
#
#   ***** X- AND RELATIVE FUNCTION CONVERGENCE *****
#
# indicating a successful calibration.
# As expected, the plots show that the model form does not match the data well.
# Additionally, the final objective function value of around 0.01492
# indicates the model form is noticeably worse than the single and double Voce model forms
# investigated previously. Finally, the calibration is forcing the yield to be much below the expected
# value of near 40 ksi. In fact, it would drive the yield lower, but the algorithm
# is hitting the specified lower bound. This is showing that the model is being "over-fit" to the data.
# Overfitting occurs when a model is matches the calibration data as well as possible but does not perform 
# well when predicting behavior in validation cases for the model. By dropping the yield so low, 
# this model would not do well in applications where the model was loaded near yield and 
# would likely over predict plastic strains.

mat_point_model_PL.set_name("compression_mat_point_PL_var")
epsilon_l = Parameter('epsilon_l', 0, 0.1, 0)
calibration_PL = GradientCalibrationStudy(Y, A, n, epsilon_l)
calibration_PL.set_results_storage_options(results_save_frequency=5)
calibration_PL.add_evaluation_set(mat_point_model_PL, objective, data_collection)
calibration_PL.set_core_limit(5)

results_PL_2 = calibration_PL.launch()
make_standard_plots("true_strain")
print(results_PL_2.best)
#%%
# Lastly, the power law calibration with Luders strain completes with the Dakota output::
#
#   ***** RELATIVE FUNCTION CONVERGENCE *****
#
# again, indicating a successful calibration.
# As expected, the plots once again show that the model does not match the data well.
# The final objective function value of around 0.01374
# indicates the model form is noticeably worse than the single and double Voce model forms
# investigated previously, but better than the power law calibration without luders strain.
# However, from the QoI plots it is clear that this is not the case. This is a clear example of 
# overfitting the data. The Luders strain parameter should be set to zero as it is not 
# a mechanism apparent in the data. However, the calibration determined that by using Luders strain 
# the overall objective could be reduced. In this case, overfitting is glaringly obvious, but in actual applications
# that may not be so.
# 
# In closing, we demonstrated how the objective function value of a calibrated model is a metric of material model form 
# quality for a given set of data. We also showed that more information is needed to appropriately
# select the best model form for a simulation or suite of simulations.
# In general, once a model has been calibrated use your knowledge of the material, its application,
# and hopefully some validation experiments and simulations to choose a final material model form.