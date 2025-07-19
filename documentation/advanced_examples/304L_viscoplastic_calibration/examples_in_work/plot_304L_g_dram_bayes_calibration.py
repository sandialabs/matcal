"""
304L stainless steel viscoplastic calibration
---------------------------------------------

With our material model chosen and initial points determined, 
we can setup a final full finite element calibration to 
get a best fit to the available data.

.. note::
    Useful Documentation links:

    #. :ref:`Uniaxial Tension Models`
    #. :class:`~matcal.sierra.models.RoundUniaxialTensionModel`
    #. :class:`~matcal.core.models.PythonModel`
    #. :class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`
    #. :class:`~matcal.core.residuals.UserFunctionWeighting`

To begin, we import all the tools we will use.
We will be using MatPlotLib, NumPy and MatCal.
"""

import numpy as np
import matplotlib.pyplot as plt
from matcal import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)

#%%
# Next, we import the data using a :class:`~matcal.core.data_importer.BatchDataImporter`.
# Since we are using a rate dependent material model, we assign a displacement rate 
# state variable to the data using the ``fixed_states`` keyword argument. We also
# assign an initial temperature through ``fixed_states``.
tension_data = BatchDataImporter("ductile_failure_ASTME8_304L_data/*.dat", 
                                    file_type="csv", 
                                    fixed_states={"displacement_rate":2e-4, 
                                                  "temperature":530}).batch

rate_data = matcal_load("rate_data.joblib")
rate_data.set_name("ductile failure small tension")
rate_data_collection = DataCollection("yield vs rate", rate_data)
rate_data_collection = scale_data_collection(rate_data_collection, "yield", 1000)

#%%
# We then manipulate the data to fit our needs and modeling choices. First, 
# we scale the data from ksi to psi units. Then we remove the time field 
# as this has consequences for the finite element model boundary conditions. 
# See :ref:`Uniaxial tension solid mechanics boundary conditions`.
tension_data = scale_data_collection(tension_data, "engineering_stress", 1000)
tension_data.remove_field("time")

calibrated_params = matcal_load("voce_calibration_results.serialized")
Y_0_val = calibrated_params["Y_0"]
Y_0 = Parameter("Y_0", Y_0_val*0.9, Y_0_val*1.1, Y_0_val, 
                distribution="uniform_uncertain")

A_val = calibrated_params["A"]
A = Parameter("A", A_val*0.9, A_val*1.1, A_val, 
                distribution="uniform_uncertain")

b_val = calibrated_params["b"]
b = Parameter("b", 1.5, 2.5, b_val, 
                distribution="uniform_uncertain")

C_val = calibrated_params["C"]
C = Parameter("C", C_val*1.1, C_val*0.9, C_val,
                distribution="uniform_uncertain")

#%%
# Now we can define the models to be calibrated. 
# We will start with the Python function for the 
# rate-dependence Python model.
def JC_rate_dependence_model(Y_0, C, X, ref_strain_rate, **unused):
    import numpy as np
    strain_rates = np.logspace(-4,4, 1000)
    yield_stresses = Y_0*X*(1+10**C*np.log(strain_rates/ref_strain_rate))*1000
    yield_stresses[strain_rates < ref_strain_rate] = Y_0
    return {"rate":strain_rates, "yield":yield_stresses}

#%%
# We then create the model and add the reference
# strain rate constant to the model.
rate_model = PythonModel(JC_rate_dependence_model)
rate_model.set_name("python_rate_model")
X_val = calibrated_params["X"]
rate_model.add_constants(ref_strain_rate=1e-5, X=X_val)


#%%
# After preparing the models and data, we must define the objectives to be minimized. 
# For this calibration, we will need a separate objective for each model and 
# data set to be compared. Both will use the
# :class:`~matcal.core.objective.CurveBasedInterpolatedObjective`,
# but will differ in the fields that they use for
# interpolation and residual calculation. For the 
# rate dependence model,
# we will be calibrating the yield stress from the model to each measured yield 
# at each rate. For the tension model, we will be calibrating to the 
# measured engineering stress-strain curve. Therefore,
# we create the objectives shown below.
rate_objective = DirectCurveBasedInterpolatedObjective("rate", "yield")
astme8_objective = DirectCurveBasedInterpolatedObjective("engineering_strain", "engineering_stress")


#%%
# To perform the calibration, we will use 
# the :class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`.
# First, we create the calibration
# study object with the :class:`~matcal.core.parameters.Parameter` objects that we made earlier.
# We then add the evaluation sets which will be 
# combined to form the full objective. In this case, each evaluation 
# set has a single objective, model and data/data_collection. 
# As a result, MatCal will track two objectives for this problem.
#
# .. note ::
#   MatCal can also accept multiple objectives passed to a single evaluation set in the form of an
#   :class:`~matcal.core.objective.ObjectiveCollection`. 
#   You can also add evaluation sets for a given 
#   model multiple times. This is useful when you have different types 
#   of data from the experiments and 
#   must use different objectives on these data sets. 
#   An example would be calibrating to both stress-strain and temperature-time data.
#   Sometimes the experimental data is not collocated in time and supplied in different files.
#   In such a case, you could calibrate
#   to both by adding two evaluation sets for the model, 
#   one for stress-strain and another for temperature-time.
#
# After adding the evaluation sets, we need to set the study core limit. 
# MatCal takes advantage of 
# multiple cores in two layers. Most models can be run on several cores, all studies can run 
# evaluation sets in parallel (all models for a combined objective 
# evaluation can be run concurrently), and most 
# studies can run several combined objective evaluations concurrently. 
# For this case, we need 1 core for the python model and 
# 36 cores for the tension model in each combined objective evaluation. 
# The study itself supports objective evaluation 
# concurrently up to :math:`n+1` where :math:`n` is the number of parameters. 
# See the 
# study specific documentation for the objective evaluation concurrency for other methods.
# For this case, the study will perform seven concurrent combined
# objective evaluations, so this study can use at most 37*6 cores. 
# As a result, we set the core limit to 112 because
# we have computational resources where we can use 112 cores concurrently
# and do not want to attempt to run more processes than available cores. 
# If you have fewer cores, 
# set the limit to what is available and MatCal will not use 
# more than what is specified. If no core limit is set,
# MatCal will default to 1. For parallel jobs, you must specify the limit.

surrogate_filename = "tension_model_surrogate.joblib"
my_surrogate = load_matcal_surrogate(surrogate_filename)
astme8_sur_model = MatCalSurrogateModel(my_surrogate)
astme8_sur_model.add_constants(C=C_val)
dram_study = DramBayesianCalibrationStudy(Y_0, A, b)
dram_study.do_not_report_evaluation_history()
dram_study.set_number_of_samples(125000)
dram_study.add_evaluation_set(astme8_sur_model, astme8_objective, tension_data)
#dram_study.add_evaluation_set(rate_model, rate_objective, rate_data_collection)
dram_study.set_proposal_covariance(*list(np.ones(3)))
dram_study.run_in_serial()

results = dram_study.launch()
matcal_save('dram_results.joblib', results)


calc_map = results["parameters"]
calc_mean = results["mean"]
calc_sd = results["stddev"]

print("MEAN: ", calc_mean)
print("SD  : ", calc_sd)
print("MAP : ", calc_map)


#%%
# We can now run the calibration. After it finishes, we will plot 
# MatCal's standard plots which include plotting the simulation QoIs versus the experimental data
# QoIs, the objectives versus evaluation and the objectives versus the parameter values. 
# We also print and save the final parameter values. 
#results = sampling_study.launch()
#matcal_save("lhs_results.joblib", results["parameters"])
