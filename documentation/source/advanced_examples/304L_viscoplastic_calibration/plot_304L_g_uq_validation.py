"""
304L stainless steel viscoplastic uncertainty quantification validation
------------------------------------------------------------------------
In this example, we will use MatCal's :class:`~matcal.core.parameter_studies.ParameterStudy`
to validate the estimated parameter uncertainty for the calibration. 
We do this by generating samples from the fitted covariance from 
:ref:`304L stainless steel viscoplastic calibration uncertainty quantification` and 
running the calibrated models with these samples. Then the 
model results are compared to the data to see how well the sampled parameter 
sets allow the models to represent the data uncertainty. 

.. note::
    Useful Documentation links:

    #. :ref:`Uniaxial Tension Models`
    #. :class:`~matcal.core.models.PythonModel`
    #. :func:`~matcal.core.parameter_studies.sample_multivariate_normal`

To begin, we reuse the data import, model preparation 
and objective specification for the tension model and rate 
models from :ref:`304L stainless steel viscoplastic calibration uncertainty quantification`.    
"""
import numpy as np
import matplotlib.pyplot as plt
from matcal import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)

tension_data = BatchDataImporter("ductile_failure_ASTME8_304L_data/*.dat", 
                                    file_type="csv", 
                                    fixed_states={"displacement_rate":2e-4, 
                                                  "temperature":530}).batch

tension_data = scale_data_collection(tension_data, "engineering_stress", 1000)
tension_data.remove_field("time")

down_selected_data = DataCollection("down selected data")
for state in tension_data.keys():
   for index, data in enumerate(tension_data[state]):
      down_selected_data.add(data[(data["engineering_stress"] > 36000) &
                                  (data["engineering_strain"] < 0.75)])

rate_data_collection = matcal_load("rate_data.joblib")

calibrated_params = matcal_load("voce_calibration_results.serialized")

Y_0 = Parameter("Y_0", 20, 60, 
                calibrated_params["Y_0"])
A = Parameter("A", 100, 400, 
              calibrated_params["A"])
b = Parameter("b", 0, 3, 
              calibrated_params["b"])
C = Parameter("C", -3, -0.5, calibrated_params["C"])
X = Parameter("X", 0.50, 1.75, 1.0)
params = ParameterCollection("laplace params", Y_0, A, b, C)

def JC_rate_dependence_model(Y_0, A, b, C, X, ref_strain_rate, rate, **kwargs):
    yield_stresses = np.atleast_1d(Y_0*X*(1+10**C*np.log(rate/ref_strain_rate)))
    yield_stresses[np.atleast_1d(rate) < ref_strain_rate] = Y_0
    return {"yield":yield_stresses}

rate_model = PythonModel(JC_rate_dependence_model)
rate_model.set_name("python_rate_model")

material_name = "304L_viscoplastic"
material_filename = "304L_viscoplastic_voce_hardening.inc"
sierra_material = Material(material_name, material_filename,
                            "j2_plasticity")

geo_params = {"extensometer_length": 0.75,
               "gauge_length": 1.25, 
               "gauge_radius": 0.125, 
               "grip_radius": 0.25, 
               "total_length": 4, 
               "fillet_radius": 0.188,
               "taper": 0.0015,
               "necking_region":0.375,
               "element_size": 0.01,
               "mesh_method":3, 
               "grip_contact_length":1}

astme8_model = RoundUniaxialTensionModel(sierra_material, **geo_params)            
astme8_model.add_boundary_condition_data(tension_data)       

from site_matcal.sandia.computing_platforms import is_sandia_cluster, get_sandia_computing_platform
from site_matcal.sandia.tests.utilities import MATCAL_WCID

cores_per_node = 24
if is_sandia_cluster():
    platform = get_sandia_computing_platform()
    cores_per_node = platform.processors_per_node

astme8_model.set_number_of_cores(cores_per_node)
if is_sandia_cluster():       
    astme8_model.run_in_queue(MATCAL_WCID, 1)
    astme8_model.continue_when_simulation_fails()
astme8_model.set_allowable_load_drop_factor(0.45)
astme8_model.set_name("ASTME8_tension_model")
astme8_model.add_constants(ref_strain_rate=1e-5)

X_calibrated = calibrated_params.pop("X")
rate_model.add_constants(ref_strain_rate=1e-5, X=X_calibrated)
astme8_model.add_constants(ref_strain_rate=1e-5)

rate_objective = Objective("yield")
astme8_objective = CurveBasedInterpolatedObjective("engineering_strain", "engineering_stress")

#%%
# With the models, data, and objectives created, 
# we import the :class:`~matcal.core.parameter_studies.LaplaceStudy` results from the previous step.
laplace_covariance = matcal_load("laplace_study_covariance.joblib")

#%%
# Next, we can sample
# the calculated parameter distribution using 
# :func:`~matcal.core.parameter_studies.sample_multivariate_normal` and evaluate 
# the parameter uncertainty as desired. 
num_samples=40
mean = [calibrated_params["Y_0"], calibrated_params["A"],
         calibrated_params["b"], calibrated_params["C"]]
uncertain_param_sets = sample_multivariate_normal(num_samples, 
                                                  mean,
                                                  laplace_covariance, 
                                                  12345, 
                                                  params.get_item_names())

#%%
# We save the parameter samples to be used or plotted later.
matcal_save("laplace_uq_validation_results.joblib", uncertain_param_sets)

#%%
# Now we set up a study so we can 
# visualize the results by pushing the samples back through the models.
# We do so using a MatCal :class:`~matcal.core.parameter_studies.ParameterStudy`.
param_study = ParameterStudy(Y_0, A, b, C)
param_study.add_evaluation_set(astme8_model, astme8_objective, tension_data)
param_study.add_evaluation_set(rate_model, rate_objective, rate_data_collection)
param_study.set_core_limit(250)
sampling_dir = "UQ_sampling_study"
param_study.set_working_directory(sampling_dir, remove_existing=True)

#%%
# Next, we add parameter evaluations for each of the samples. 
# We do so by organizing the data using Python's
# ``zip`` function and then loop over the result
# to add each parameter set sample to the study.
#
# .. note::
#    We add error catching to the addition of each parameter 
#    evaluation. There is a chance that parameters could be 
#    generated outside of our original bounds and we want the study to complete.
#    If this error is caught, we will see it in the MatCal output 
#    and know changes are needed. However, some results will still be output
#    and can be of use.
#
params_to_evaluate = zip(uncertain_param_sets["Y_0"], uncertain_param_sets["A"],
                         uncertain_param_sets["b"], uncertain_param_sets["C"])

for Y_0, A_eval, b_eval, C_eval in params_to_evaluate:
    try:
      param_study.add_parameter_evaluation(Y_0=Y_0, A=A_eval, b=b_eval, C=C_eval)
      print(f"Running evaluation with Y_0={Y_0}, A={A_eval}, b={b_eval}, and "
          f"C={C_eval}.")
                               
    except ValueError:
       print(f"Skipping evaluation with Y={Y_0}, A={A_eval}, b={b_eval}, and "
            f"C={C_eval}. Parameters out of range.")

#%%
# Next, we launch the study and plot the results.
# Once again, we use plotting functions from 
# the previous examples to simplify the plotting processes.
param_study_results = param_study.launch()
astme_results = param_study_results.simulation_history[astme8_model.name]
rate_results = param_study_results.simulation_history[rate_model.name]

def compare_data_and_model(data, model_responses, indep_var, dep_var, 
                           plt_func=plt.plot, fig_label=""):
  fig = plt.figure(fig_label, constrained_layout=True, figsize=figsize)
  data.plot(indep_var, dep_var, plot_function=plt_func, ms=3, labels="data", 
            figure=fig, marker='o', linestyle='-', color="#bdbdbd", show=False)
  model_responses.plot(indep_var, dep_var, plot_function=plt_func,labels="models", 
                      figure=fig, linestyle='-', alpha=0.5)
  plt.xlabel("Engineering Strain (.)")
  plt.ylabel("Engineering Stress (psi)")
  

compare_data_and_model(tension_data, astme_results, 
                       "engineering_strain", "engineering_stress", 
                       fig_label="tension model")

def make_single_plot(data_collection, state, cur_idx, label, 
                     color, marker, **kwargs):
    data = data_collection[state][cur_idx]
    plt.semilogx(state["rate"], data["yield"][0],
                marker=marker, label=label, color=color, 
                **kwargs)

def plot_dc_by_state(data_collection, label=None, color=None,
                     marker='o', best_index=None, only_label_first=False, **kwargs):
    for state in data_collection:
        if best_index is None:
            for idx, data in enumerate(data_collection[state]):
                make_single_plot(data_collection, state, idx, label, 
                                 color, marker, **kwargs)
                if ((color is not None and label is not None) or
                    only_label_first):
                    label = None
        else:
            make_single_plot(data_collection, state, best_index, label, 
                             color, marker, **kwargs)
    plt.xlabel("engineering strain rate (1/s)")
    plt.ylabel("yield stress (ksi)")

plt.figure(constrained_layout=True, figsize=figsize)
plot_dc_by_state(rate_data_collection, label='experiments', 
                 color="k", markersize=10)
plot_dc_by_state(rate_results, label='rate model', marker='x',
                 only_label_first=True)
plt.legend()

#%%
# These figure show the model results from the 40 samples. 
# For the tension model, the results appear to be good estimate of parameter 
# uncertainty. The simulations encapsulate all data, without exhibiting 
# too much variability. While the python rate dependence model results do not 
# completely encapsulates all 
# data, the results seem to be an adequate measure of overall uncertainty.
