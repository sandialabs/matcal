"""
Calibration With Unmodified Data and Objective: A Simple Calibration Gone Wrong
===============================================================================

In this section, we once again present the calibration of our :class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel`
to the uniaxial compression data. However, instead of manipulating the data and weighting the residuals like we did in the 
:ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py`
example, we ignore
the potential issues that were pointed out there and use the data as provided. Overall, 
the exact same process is used in this calibration as 
was used in the successful calibration; however, the data cleanup and residual weights modification is not 
performed initially. See the :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py`
for more of those details. First, we import and view the data: 
"""

# sphinx_gallery_thumbnail_number = 13

from matcal import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)

data_collection = BatchDataImporter("uniaxial_material_point_data/*.csv").batch 
data_collection = scale_data_collection(data_collection, "true_stress", 1000)
data_collection.plot("true_strain", "true_stress")
data_collection.plot("time", "true_strain")

#%%
# We then create our material model parameters 
# and the model we will be calibrating. 

Y = Parameter('Y', 30, 60, 50)
A = Parameter('A', 1, 500, 100)
b = Parameter('b', 5, 30, 20.001)

j2_voce = Material("j2_voce", "sierra_sm_voce_hardening.inc", "j2_plasticity")

mat_point_model = UniaxialLoadingMaterialPointModel(j2_voce)
mat_point_model.add_boundary_condition_data(data_collection)
mat_point_model.set_name("compression_mat_point")

#%%
# Next we setup the calibration and objective to be evaluated.

calibration = GradientCalibrationStudy(Y, A, b)
calibration.set_results_storage_options(results_save_frequency=4)
objective = CurveBasedInterpolatedObjective('true_strain','true_stress')
calibration.add_evaluation_set(mat_point_model, objective, data_collection)
calibration.set_core_limit(4)

#%%
# Finally, we launch the calibration study and review the results.

results = calibration.launch()
print(results.best)
make_standard_plots("true_strain")

#%%
# The calibration completes with the Dakota output::
#   
#       ***** X- AND RELATIVE FUNCTION CONVERGENCE *****

#%%
# indicating that the algorithm completed successfully. However, from 
# the results plots it is clear that the model is not quite matching the data how
# we would expect. The objective value is also about 100 times higher than in the successful
# calibration. Finally, the calibrated parameters do not seem to line up with expectations due to
# the higher than expected yield stress and and very low saturation stress. 
# This is due to two reasons. First, the unloading data is included in the objective, and
# the model doesn't seem to appropriately unload.  
# Second, the  
# :class:`~matcal.core.objective.CurveBasedInterpolatedObjective` requires data
# to have an independent field that is monotonically increasing. MatCal sorts 
# the data so that is true. By sorting these data with the unloading section present,
# it incorrectly mixes the data around the unloading region, and the interpolation is meaningless.
# Even if MatCal did not sort the data, NumPy interp would return bad interpolation data, 
# so without data preprocessing this calibration will not be performed correctly. 
#
# After realizing these issues, let's add residual weighting to remove the
# high strain data as we did in the previous example. Then we can
# create a new study with the updated evaluation set and re-run the calibration.
#
#.. warning::
#   You currently cannot run the same study twice. 
#   Even if objects that are fed into it have changed,
#   A new study must be made.
#

def remove_high_strain_from_residual(true_strains, true_stresses, residuals):
    import numpy as np
    weights = np.ones(len(residuals))
    weights[(true_strains > 0.5)] = 0
    return weights*residuals

residual_weights = UserFunctionWeighting("true_strain", "true_stress", remove_high_strain_from_residual)

objective.set_field_weights(residual_weights)

Y = Parameter('Y', 30, 60, 51)
A = Parameter('A', 1, 500, 110)
b = Parameter('b', 5, 30, 21)

#%%
# .. include:: ../../multiple_dakota_studies_in_python_instance_warning.rst

calibration = GradientCalibrationStudy(Y, A, b)
calibration.set_results_storage_options(results_save_frequency=4)
calibration.add_evaluation_set(mat_point_model, objective, data_collection)
calibration.set_core_limit(4)

results = calibration.launch()
print(results.best)
make_standard_plots("true_strain")
import matplotlib.pyplot as plt

#%%
# Once again, the calibration completes with the Dakota output::
#   
#   ***** RELATIVE FUNCTION CONVERGENCE *****
#
# indicating that the algorithm completed successfully. This time the 
# results even compare well to the data and the calibration produced parameters
# that intuitively match what one would expect. However, a strange artifact 
# is present in the simulation QoI true stress/strain curve. It still drops 
# in the middle of the curve. This is due to the function passed to the boundary conditions.  
# The experimental data has unloading at the end of the engineering strain time history,
# which resulted in the model partially unloading in the middle of its deformation. 
# Why did it unload in the middle of the displacement time history and not at the end?
# There is another harder-to-notice issue lurking here. If we 
# plot the simulation data, on top of the interpolated simulation QoIs
# it becomes clear.
state_name = data_collection.state_names[0]
best_sim_results = results.best_simulation_data(mat_point_model, state_name)
best_sim_qois = results.best_simulation_qois(mat_point_model, objective, state_name, 0)

plt.figure(figsize=(4,3), constrained_layout=True)
plt.plot(best_sim_qois["true_strain"], best_sim_qois["true_stress"], 'ko', 
         label='interpolated qois')
plt.plot(best_sim_results["true_strain"], best_sim_results["true_stress"], 
         label='simulation data')
plt.legend()
plt.xlabel("true strain")
plt.ylabel("true stress (psi)")
plt.show()

#%%
# You can see that the simulation data ends too early. 
# This occurs because MatCal prioritizes engineering strain over true strain in a data collection 
# for boundary condition generation and expects compressive strains to be negative. 
# The conversion of engineering strain to true strain is done using:
#
# .. math::
#   \epsilon_t = \log(\epsilon_e+1)
#
# where :math:`\epsilon_e` is the engineering strain.
# If we evaluate the above equation for the max engineering strain for 
# tension and then for compression, we see the difference in the final 
# applied true strain.
import numpy as np
max_strain = np.max(data_collection["matcal_default_state"][1]["engineering_strain"])
true_tension = np.log(max_strain+1)
true_compression = np.log(-max_strain+1)
print("Final applied true strain if assumed tension:",  true_tension)
print("Final applied true strain if assumed in compression:", true_compression)

#%%
# When the measure engineering strain is applied in compression a larger true-strain
# is applied to the model.
#
# This is illustrating the importance of data cleanup and processing when strain-time or 
# displacement-time data 
# is passed into a MatCal model 
# to generate boundary conditions. Only clean and well understood data should be  
# provided for boundary condition data when passing time-strain or time-displacement data to 
# the model. Also, any compression data used in the 
# :class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel`
# should always be supplied as negative for boundary condition generation and the objective.
# The fact 
# that the positive strain values appeared to behave correctly here is due to the way NumPy interp
# functions during extrapolation and the model form. In some cases, it may fail more noticeably.
#
# To get the correct behavior in this case, the engineering strain must be converted to negative
# since it was taken in compression. 
# Once again, let's convert the data to negative 
# since it is compressive data and update the residual weighting
# function for compressive data.
# We also select only the cleaner data, second data set 
# to be used to define the model boundary condition.
# We then make a new calibration and a new model
# with the updated boundary condition
# data to see how the results are affected.
data_collection = scale_data_collection(data_collection, "true_stress", -1)
data_collection = scale_data_collection(data_collection, "true_strain", -1)

boundary_data = data_collection["matcal_default_state"][1]
boundary_data = boundary_data[["engineering_strain"]]
boundary_data["engineering_strain"] *= -1
boundary_data.set_name("dataset 1 derived BC data")
boundary_data_collection = DataCollection('boundary_data', boundary_data)

mat_point_model = UniaxialLoadingMaterialPointModel(j2_voce)
mat_point_model.add_boundary_condition_data(boundary_data_collection)
mat_point_model.set_name("compression_mat_point")

def remove_high_strain_from_residual(true_strains, true_stresses, residuals):
    import numpy as np
    weights = np.ones(len(residuals))
    weights[(-true_strains > 0.5)] = 0
    return weights*residuals

residual_weights = UserFunctionWeighting("true_strain", "true_stress", remove_high_strain_from_residual)

objective.set_field_weights(residual_weights)


Y = Parameter('Y', 30, 60, 50.5)
A = Parameter('A', 1, 500, 100.25)
b = Parameter('b', 5, 30, 20.12)
#%%
# .. include:: ../../multiple_dakota_studies_in_python_instance_warning.rst

calibration = GradientCalibrationStudy(Y, A, b)
calibration.set_results_storage_options(results_save_frequency=4)
calibration.add_evaluation_set(mat_point_model, objective, data_collection)
calibration.set_core_limit(4)

results = calibration.launch()

#%%
# The results from this calibration reproduce the results from the 
# :ref:`Successful Calibration` and match the data well. Generally, 
# calibrations will require some data clean up and manipulation to 
# work well and provide the desired results. 
print(results.best)
make_standard_plots("true_strain")

#%%
# We can also plot the simulation data with the qois 
# again and show the model has been deformed enough 
# to not use NumPy interp's results padding.
best_sim_results = results.best_simulation_data(mat_point_model, state_name)
best_sim_qois = results.best_simulation_qois(mat_point_model, objective, state_name, 0)

plt.figure(figsize=(4,3), constrained_layout=True)
plt.plot(best_sim_qois["true_strain"], best_sim_qois["true_stress"], 'ko', 
         label='interpolated qois')
plt.plot(best_sim_results["true_strain"], best_sim_results["true_stress"], 
         label='simulation data')
plt.legend()
plt.xlabel("true strain")
plt.ylabel("true stress (psi)")
plt.show()