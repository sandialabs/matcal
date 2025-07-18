"""
6061T6 aluminum uncertainty quantification validation
-----------------------------------------------------
In this example, we will use MatCal's :class:`~matcal.core.parameter_studies.ParameterStudy`
to validate the estimated parameter uncertainty for the calibration. 
We do this by generating samples from the fitted covariance from 
:ref:`6061T6 aluminum calibration with anisotropic yield` and 
running the calibrated models with these samples. Then the 
model results are compared to the data to see how well the sampled parameter 
sets allow the models to represent the data uncertainty. 

.. note::
    Useful Documentation links:

    #. :ref:`Uniaxial Tension Models`
    #. :ref:`Top Hat Shear Model`
    #. :class:`~matcal.core.parameter_studies.ParameterStudy`
    #. :func:`~matcal.core.parameter_studies.sample_multivariate_normal`
            
To begin, we import the tools we need for this study and setup the 
data and model as we did in :ref:`6061T6 aluminum calibration with anisotropic yield`.
"""

from matcal import *
from matcal.sandia.computing_platforms import is_sandia_cluster, get_sandia_computing_platform

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)

tension_data_collection = BatchDataImporter("ductile_failure_aluminum_6061_data/" 
                                              "uniaxial_tension/processed_data/"
                                              "cleaned_[CANM]*.csv",).batch

down_selected_tension_data = DataCollection("down selected data")
for state in tension_data_collection.keys():
    for index, data in enumerate(tension_data_collection[state]):
        stresses = data["engineering_stress"]
        strains = data["engineering_strain"]    
        peak_index = np.argmax(stresses)
        peak_strain = strains[peak_index]
        peak_stress = stresses[peak_index]
        data_to_keep = (((strains>peak_strain) & (stresses > 0.89*peak_stress)) | 
                        (strains>0.005) & (strains < peak_strain))
        down_selected_tension_data.add(data[data_to_keep])

down_selected_tension_data = scale_data_collection(down_selected_tension_data, 
                                                   "engineering_stress", 1000)
down_selected_tension_data.remove_field("time")

material_filename = "hill_plasticity_temperature_dependent.inc"
material_model = "hill_plasticity"
material_name = "ductile_failure_6061T6"
sierra_material = Material(material_name, material_filename, material_model)

gauge_radius = 0.125
element_size = gauge_radius/8
geo_params = {"extensometer_length": 1.0,
              "gauge_length": 1.25,
              "gauge_radius": gauge_radius,
              "grip_radius": 0.25,
              "total_length": 4,
              "fillet_radius": 0.188,
              "taper": 0.0015,
              "necking_region":0.375,
              "element_size": element_size,
              "mesh_method":3,
              "grip_contact_length":1}

tension_model = RoundUniaxialTensionModel(sierra_material, **geo_params)            
tension_model.set_name("tension_model")
tension_model.add_boundary_condition_data(down_selected_tension_data)
tension_model.set_allowable_load_drop_factor(0.70)
tension_model.set_boundary_condition_scale_factor(1.5)

my_wcid = "fy220213"
if is_sandia_cluster():
  tension_model.run_in_queue(my_wcid, 1)
  tension_model.continue_when_simulation_fails()
  platform = get_sandia_computing_platform()
  num_cores = platform.get_processors_per_node()
else:
  num_cores = 8
tension_model.set_number_of_cores(num_cores)

top_hat_data_collection = BatchDataImporter("ductile_failure_aluminum_6061_data/" 
                                              "top_hat_shear/processed_data/cleaned_*.csv").batch
for state, state_data_list in top_hat_data_collection.items():
    for index, data in enumerate(state_data_list):
        max_load_arg = np.argmax(data["load"])
        data = data[data["time"] < data["time"][max_load_arg]]
        data = data[data["load"] > 0.005]
        top_hat_data_collection[state][index] = data[data["displacement"] < 0.02]
top_hat_data_collection.remove_field("time")

top_hat_geo_params = {"total_height":1.25,
        "base_height":0.75,
        "trapezoid_angle": 10.0,
        "top_width": 0.417*2,
        "base_width": 1.625, 
        "base_bottom_height": (0.75-0.425),
        "thickness":0.375, 
        "external_radius": 0.05,
        "internal_radius": 0.05,
        "hole_height": 0.3,
        "lower_radius_center_width":0.390*2,
        "localization_region_scale":0.0,
        "element_size":0.005, 
        "numsplits":1}

top_hat_model = TopHatShearModel(sierra_material, **top_hat_geo_params)
top_hat_model.set_name('top_hat_shear')
top_hat_model.set_allowable_load_drop_factor(0.05)
top_hat_model.add_boundary_condition_data(top_hat_data_collection)
top_hat_model.set_number_of_cores(num_cores*2)
if is_sandia_cluster():
  top_hat_model.run_in_queue(my_wcid, 1)
  top_hat_model.continue_when_simulation_fails()

tension_objective = CurveBasedInterpolatedObjective("engineering_strain", "engineering_stress")
tension_objective.set_name("engineering_stress_strain_obj")
top_hat_objective = CurveBasedInterpolatedObjective("displacement", "load")
top_hat_objective.set_name("load_displacement_obj")

RT_calibrated_params = matcal_load("anisotropy_parameters.serialized")
yield_stress = Parameter("yield_stress", 15, 50, 
        RT_calibrated_params.pop("yield_stress"))
hardening = Parameter("hardening", 0, 60, 
        RT_calibrated_params.pop("hardening"))
b = Parameter("b", 10, 40,
        RT_calibrated_params.pop("b"))
R22 = Parameter("R22", 0.8, 1.15, 
        RT_calibrated_params["R22"])
R33 = Parameter("R33", 0.8, 1.15, 
        RT_calibrated_params["R33"])
R12 = Parameter("R12", 0.8, 1.15, 
        RT_calibrated_params["R12"])
R23 = Parameter("R23", 0.8, 1.15, 
        RT_calibrated_params["R23"])
R31 = Parameter("R31", 0.8, 1.15,
        RT_calibrated_params["R31"])

pc = ParameterCollection("uncertain_params", yield_stress, hardening, b)

high_temp_calibrated_params = matcal_load("temperature_dependent_parameters.serialized")
tension_model.add_constants(**high_temp_calibrated_params,
                            **RT_calibrated_params)
top_hat_model.add_constants(**high_temp_calibrated_params,
                            **RT_calibrated_params)

results = matcal_load("laplace_study_results.joblib")

#%%
# After importing laplace study results, we can 
# sample parameters sets from the estimated parameter
# uncertainties using :func:`~matcal.core.parameter_studies.sample_multivariate_normal`.
num_samples = 50
uncertain_parameter_sets = sample_multivariate_normal(num_samples, 
                                                      results.mean.to_list(),
                                                      results.fitted_parameter_covariance, 
                                                      seed=1234, 
                                                      param_names=pc.get_item_names())

#%%
# Now we set up a study so we can 
# visualize the results by pushing the samples back through the models.
# We do so using a MatCal :class:`~matcal.core.parameter_studies.ParameterStudy`.
param_study = ParameterStudy(pc)
param_study.add_evaluation_set(tension_model, tension_objective, down_selected_tension_data)
param_study.add_evaluation_set(top_hat_model, top_hat_objective, top_hat_data_collection)
param_study.set_core_limit(250)
param_study.set_working_directory("UQ_sampling_study", remove_existing=True)
params_to_evaluate = zip(uncertain_parameter_sets["yield_stress"],
                         uncertain_parameter_sets["hardening"],
                         uncertain_parameter_sets["b"])

#%%
# Next, we add parameter evaluations for each of the samples. 
# We do so by organizing the data using Python's
# ``zip`` function and then loop over the result
# to add each parameter set sample to the study.
#
# .. Warning::
#    We add error catching to the addition of each parameter 
#    evaluation. There is a chance that parameters could be 
#    generated outside of our original bounds and we want the study to complete.
#    If this error is caught, we will see it in the MatCal output 
#    and know changes are needed. However, some results will still be output
#    and can be of use.
#
valid_runs = 0
for params in params_to_evaluate:
    y_eval    = params[0]
    A_eval    = params[1]
    b_eval    = params[2]
 
    try:
      param_study.add_parameter_evaluation(yield_stress=y_eval, hardening=A_eval,b=b_eval)
      print(f"Running evaluation {params}")
      valid_runs +=1                         
    except ValueError:
       print(f"Skipping evaluation with {params}. Parameters out of range. ")


#%%
# Next, we launch the study and plot the results.
# We use functions to simplify the plotting processes.
if valid_runs > 0:
    param_study_results = param_study.launch()
else:
    exit()

def compare_data_and_model(data, model_responses, indep_var, dep_var, 
                           plt_func=plt.plot, fig_label=None):
    if fig_label is not None:
        fig = plt.figure(fig_label)
    else:
        fig = None
    data.plot(indep_var, dep_var, plot_function=plt_func, ms=3, labels="data", 
            figure=fig, marker='o', linestyle='-', color="#bdbdbd", show=False)
    model_responses.plot(indep_var, dep_var, plot_function=plt_func,labels="models", 
                      figure=fig, linestyle='-', alpha=0.5)

all_tension_data = tension_data_collection
all_tension_data = scale_data_collection(all_tension_data, 
                                                  "engineering_stress", 1000)
all_sim_tension_data = param_study_results.simulation_history[tension_model.name]
compare_data_and_model(all_tension_data, 
                       all_sim_tension_data, 
                       "engineering_strain", "engineering_stress")

all_top_hat_sim_data =param_study_results.simulation_history[top_hat_model.name]
compare_data_and_model(top_hat_data_collection, 
                       all_top_hat_sim_data, 
                       "displacement", "load")

#%%
# In the plots, the simulation results for the simulated samples
# does not match the variation in the 
# data sets in the areas where the data were used for calibration, and 
# seem to be a poor representation of the uncertainty. Also,
# many of the parameter samples were rejected due to being out of bounds indicating
# an unacceptable results.
# A potential alternative uncertainty quantification option, 
# that is more computationally expensive, is to do data resampling. With data resampling, 
# random data sets for each model are chosen and the models are calibrated to this
# random selection. This is repeated for many sample selections. After many calibrations
# are completed, a population of valid parameter sets are obtained and can be used 
# as the uncertain parameter distributions for the parameters.  
