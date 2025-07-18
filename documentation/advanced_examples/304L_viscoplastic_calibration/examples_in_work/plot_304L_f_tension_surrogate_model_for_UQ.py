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
from matcal.sandia.computing_platforms import is_sandia_cluster

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
#%%
# We then manipulate the data to fit our needs and modeling choices. First, 
# we scale the data from ksi to psi units. Then we remove the time field 
# as this has consequences for the finite element model boundary conditions. 
# See :ref:`Uniaxial tension solid mechanics boundary conditions`.
tension_data = scale_data_collection(tension_data, "engineering_stress", 1000)
tension_data.remove_field("time")

calibrated_params = matcal_load("voce_calibration_results.serialized")
Y_0_val = calibrated_params["Y_0"]
Y_0 = Parameter("Y_0", Y_0_val*0.66, Y_0_val*1.33)

A_val = calibrated_params["A"]
A = Parameter("A", A_val*0.66, A_val*1.33)

b = Parameter("b", 0.5, 3.5)
C = Parameter("C", -3, -1)

#%%
# Now we can define the models to be calibrated. 
# We will start with the Python function for the 
# rate-dependence Python model.
def JC_rate_dependence_model(Y_0, C, X, ref_strain_rate, **unused):
    import numpy as np
    strain_rates = np.logspace(-4,4, 1000)
    yield_stresses = Y_0*X*(1+10**C*np.log(strain_rates/ref_strain_rate))
    yield_stresses[strain_rates < ref_strain_rate] = Y_0
    return {"rate":strain_rates, "yield":yield_stresses}

#%%
# We then create the model and add the reference
# strain rate constant to the model.
#rate_model = PythonModel(JC_rate_dependence_model)
#rate_model.set_name("python_rate_model")
#X_val = voce_params["X"]
#rate_model.add_constants(ref_strain_rate=1e-5, X=X_val)

#%%
# In the ``JC_rate_dependence_model`` function, you can see that the correction factor :math:`X`
# is a simple multiplier on :math:`Y_0`. This allows the calibration algorithm to compensate
# for any discrepancy between the 0.2\% offset yield in the
# experimental measurements and the material
# model yield. The correction factor is not actually used in the SIERRA/SM material model.
#
# With the rate model defined, we can now build the MatCal standard model for the 
# ASTME8 tension specimen. MatCal's :class:`~matcal.sierra.models.RoundUniaxialTensionModel` 
# does not enforce the requirements of the ASTME8 test specification, 
# and will build the model according 
# to the geometry and input provided. It significantly simplifies
# generating a model of the test for calibration. 
# The primary inputs to create the model are:
# the geometry for the specimen, a material model input file, 
# and data for boundary condition generation. 
# For more details on the model and its features see 
# :ref:`MatCal Generated SIERRA Standard Models`
# and :ref:`Uniaxial Tension Models`. 
#
# First, we create the :class:`~matcal.sierra.material.Material` object. 
# We write the material file that will be used to create the 
# MatCal :class:`~matcal.sierra.material.Material`.
material_name = "304L_viscoplastic"
material_filename = "304L_viscoplastic_voce_hardening.inc"
sierra_material = Material(material_name, material_filename,
                            "j2_plasticity")

#%%
# Next, we create the tension model using the
# :class:`~matcal.sierra.models.RoundUniaxialTensionModel`
# which takes the material object we created and geometry parameters as input.
# It is convenient to put the geometry parameters in a dictionary and then unpack that
# dictionary when initializing the model as shown below. After the model is initialized,
# the model's options can be set and modified as desired. Here we pass the entire 
# data collection into the model for boundary condition generation. Since our 
# data collection no longer has the test displacement-time history, the model will 
# deform the specimen to the maximum displacement in the data over 
# the correct time to achieve the desired engineering strain rate. 
# We study the effects of boundary condition choice in more detail in 
# :ref:`304L calibrated round tension model - effect of different model options`.
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

#%%
# We set the cores the model uses to be platform dependent.
# On a local machine it will run on 36 cores. If its on a cluster,
# it will run in the queue on 112.
astme8_model.set_number_of_cores(36)
if is_sandia_cluster():       
    astme8_model.run_in_queue("fy220213", 0.5)
    astme8_model.set_number_of_cores(112)
    astme8_model.continue_when_simulation_fails()
astme8_model.set_allowable_load_drop_factor(0.8)
astme8_model.set_name("ASTME8_tension_model")

#%%
# We also add the reference strain rate constant to the
# SIERRA model.
astme8_model.add_constants(ref_strain_rate=1e-5)

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
astme8_objective = SimulationResultsSynchronizer("engineering_strain", np.linspace(0,0.8,400), 
                                                 "engineering_stress")


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
sampling_study = LhsSensitivityStudy(Y_0, A, b, C)
sampling_study.add_evaluation_set(astme8_model, astme8_objective, 
                               states=tension_data.states)
sampling_study.set_core_limit(112)
sampling_study.use_overall_objective()
sampling_study.set_number_of_samples(1000)
sampling_study.set_seed(12345)
study_dir = "finite_element_model_lhs_runs"
sampling_study.set_working_directory(study_dir, remove_existing=True)
#%%
# However, if we are on a cluster where the models are run in a queue, 
# we set the limit based on the number of jobs that can run concurrently 
# because there is some overhead for job monitoring and results processing.
# For our case, that is only fourteen seven python models run on the parent node 
# and then seven finite element models run on children nodes with job monitoring
# and post processing on the parent node.
if is_sandia_cluster():
    sampling_study.set_core_limit(250)

#%%
# We can now run the calibration. After it finishes, we will plot 
# MatCal's standard plots which include plotting the simulation QoIs versus the experimental data
# QoIs, the objectives versus evaluation and the objectives versus the parameter values. 
# We also print and save the final parameter values. 
#results = sampling_study.launch()
#matcal_save("lhs_results.joblib", results["parameters"])

import os
init_dir = os.getcwd()
os.chdir(study_dir)

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
kernel = C(1.0, constant_value_bounds=(1e-2, 1e3)) * \
        RBF(np.ones(4), length_scale_bounds=(1e-2, 1e3))

surrogate_generator = SurrogateGenerator(sampling_study, regressor_type="Gaussian Process", 
                                         surrogate_type="PCA Multiple Regressors",
                                         decomp_var=0.999, 
                                         n_restarts_optimizer=20, 
                                         training_fraction=0.9,
                                         alpha=1e-5, 
                                         normalize_y=True, kernel=kernel)
surrogate = surrogate_generator.generate("tension_model_surrogate")
os.chdir(init_dir)
