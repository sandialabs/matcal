"""
Calibration With Unloading: A Stress vs. Time Calibration Needs Attention to Detail
===================================================================================

In this section, we approach the calibration of the  :class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel`
to the Ductile Failure 6061-T6 compression data a little differently. For the more traditional approach see the
:ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py`.
MatCal's objective tools allow significant flexibility in how the objective for calibrations are built.
Also, the :class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel` allows 
flexibility in the way the boundary conditions are derived from data. In this example, we highlight 
this flexibility by using strain versus time data to define the model boundary condition 
so that it simulates both loading and unloading.
We then define an objective with time as the independent variable and true stress as the dependent variable in the 
:class:`~matcal.core.objective.CurveBasedInterpolatedObjective`. In addition to the plasticity
parameters calibrated in the previous example, we add the 
elastic modulus as a calibrated parameter for this study. Generally, this is not 
recommended as the isotropic elastic 
properties of metals are readily available in the literature. However, we want 
to make use of the additional information provided to the objective when including 
the elastic unloading portion of 
the data in the model. A more practical use of calibrating to stress-time history 
would be calibrating a model to cyclical loading that has cycle dependent behavior
such as calibrating a model with isotropic and kinematic hardening.

Once again, the overall approach to this calibration is the same as the previous examples
and includes:

    #. Data overview, analysis and preprocessing.
    #. Model selection and preparation.
    #. Calibration execution and results review.
 
To begin, import all of MatCal's calibration tools:
"""
# sphinx_gallery_thumbnail_number = 8
from matcal import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)

#%%
# Then load the data using the :class:`~matcal.core.data_importer.BatchDataImporter` and 
# perform minor data manipulation. Remove the time offsets as was done in
# the previous examples and scale the true stress data from ksi to psi. Finally, 
# we must preprocess the strain-time functions in the data appropriately so that they are 
# interpreted correctly by the :class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel`.
# Since the data as provided by the experimentalists are positive, MatCal 
# would interpret the engineering strain function as a tensile strain which would be invalid.
# This can be corrected in two ways, (1) we can make the data negative before
# passing it to the model so that it applies the engineering strain-time function
# correctly in compression or (2) remove the engineering strain
# data from the data collection and MatCal will calculate the appropriate tensile boundary
# condition from the positive true strain data. This is valid for this calibration due to the simple 
# material model we are currently calibrating but could be invalid for models 
# with tension/compression asymmetry. For this study, we will
# select the latter method and work only with the true stress/strain data
# by completely removing the engineering strain data from the data set. 
#
# .. note:: 
#    We recommend making the data negative if the test was compressive to set
#    a standard process for calibration regardless of model form. We deviate
#    from that recommendation here to demonstrate how to manipulate data in MatCal. 

data_collection = BatchDataImporter("uniaxial_material_point_data/*.csv").batch
for state_data_list in data_collection.values():
  for data in state_data_list:
    data['time'] = data['time'] - data['time'][0]
    len_data = len(data["time"])
    data = data[["true_strain", "true_stress", "time"]][::int(len_data/200)]
data_collection = scale_data_collection(data_collection, "true_strain", 1)
data_collection = scale_data_collection(data_collection, "true_stress", 1000)

data_collection.plot("time", "true_strain")

#%%
# As done in the previous examples, we will need to calibrate the :math:`Y`, :math:`A`, and :math:`b` parameters
# for the J2 plasticity model with Voce hardening. 
# We define our parameters below with the addition of the modulus of 
# elasticity parameter :math:`E` as discussed earlier. 

Y = Parameter('Y', 30, 60, 50, units='ksi')
A = Parameter('A', 1, 500, 100.001, units='ksi')
b = Parameter('b', 5, 30, 20, units='ksi')
E = Parameter('E', 5000, 120000, 10000, units='ksi')

#%% 
# The :class:`~matcal.sierra.material.Material` class 
# definition is the same as the previous examples, but the material file
# input deck is modified for the elastic modulus parameter as shown below::
# 
#    begin material j2_voce
#      density = 0.000254
#      begin parameters for model j2_plasticity
#        youngs modulus                = {E*1e3}
#        poissons ratio                =   0.33
#        yield stress                  = {Y*1e3}
#
#        hardening model = voce
#        hardening modulus = {A*1e3}
#        exponential coefficient = {b} 
#       end
#    end
# 
# With this SIERRA/SM input saved in the current directory as "sierra_sm_voce_hardening_with_elastic_modulus.inc", 
# we can create the updated :class:`~matcal.sierra.material.Material` and 
# :class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel` objects. 

j2_voce = Material("j2_voce", "sierra_sm_voce_hardening_with_elastic_modulus.inc", "j2_plasticity")

mat_point_model = UniaxialLoadingMaterialPointModel(j2_voce)
mat_point_model.add_boundary_condition_data(data_collection)
mat_point_model.set_name("compression_mat_point")

#%%
# Next, we create the study and the objective before adding 
# the desired evaluation set to the study and running it. 
# The objective specification here is similar to the previous example, 
# but now has time as the independent variable.

calibration = GradientCalibrationStudy(Y, A, b, E)
calibration.set_results_storage_options(results_save_frequency=5)
objective = CurveBasedInterpolatedObjective('time','true_stress')
calibration.add_evaluation_set(mat_point_model, objective, data_collection)
calibration.set_core_limit(5)
results = calibration.launch()
print(results.best)
make_standard_plots("time")
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(8,8)

#%%
# The calibration completes and indicates success with the following output.::
#   
#   ***** RELATIVE FUNCTION CONVERGENCE *****
#
# However, the result plots once again show a fit that appears suboptimal and the 
# parameter values are not near the expected values. The yield is much lower than 
# the expected value of ~42 ksi and the elastic modulus is very high at ~21000 ksi.
# Closer inspection of the experiment/simulation QoI plot shows that this is due to 
# a relatively minor difference between the stress-time data for the 
# two experimental data sets used in the objective. This discrepancy is shown 
# in the zoomed in figure below where an 8 second
# delay before deformation initiates is observed for the C-ST-01 sample. Since we are using the 
# strain-time history from the C-RD-01 sample with deformation beginning near 
# time = 0, the stress-time history for the model will not match the
# C-ST-01 data. As a result, the objective is poorly defined which leads to 
# a low-quality fit. 
import matplotlib.pyplot as plt
zoom_fig = plt.figure("zoom_view", constrained_layout=True)
data_collection.plot("time", "true_strain", figure=zoom_fig)
plt.xlim([0, 35])
plt.ylim([0, 0.05])

#%% 
# We can once again use MatCal tools to correct the data. 
# Since we only want to modify one of the data sets, we do not 
# import them all with the :class:`~matcal.core.data_importer.BatchDataImporter`. 
# Instead, we import them individually using the :func:`~matcal.core.data_importer.FileData` 
# utility and modify only the data set with the issue as shown below. A :class:`~matcal.core.data.Scaling`
# object is then used to apply the correct 8 second offset to the RD data set. Then 
# we plot the data to verify the results are as intended.
data_RD = FileData("uniaxial_material_point_data/true_engineering_stress_strain_compression_C-RD-01.csv")
data_RD['time'] = data_RD['time'] - data_RD['time'][0]
len_data_RD = len(data_RD["time"])
data_RD = data_RD[["time", "true_stress", "true_strain"]][1:len_data_RD+1:int(len_data_RD/200)]
data_RD_offset_scaling = Scaling("time", offset = -8)
data_RD = data_RD_offset_scaling.apply_to_data(data_RD)

data_ST = FileData("uniaxial_material_point_data/true_engineering_stress_strain_compression_C-ST-01.csv")
data_ST['time'] = data_ST['time'] - data_ST['time'][0]
len_data_ST = len(data_ST["time"])
data_ST = data_ST[["time", "true_stress", "true_strain"]][:len_data_ST+1:int(len_data_ST/200)]

data_collection = DataCollection("compression data", data_RD, data_ST)
data_collection = scale_data_collection(data_collection, "true_strain", 1)
data_collection = scale_data_collection(data_collection, "true_stress", 1000)

data_collection.plot("time", "true_strain")

#%%
# With the data appropriately modified we can re-run the 
# calibration study and review the results. We re-initialize
# the model and study with the updated data collection and
# then launch the updated calibration.

mat_point_model = UniaxialLoadingMaterialPointModel(j2_voce)
mat_point_model.add_boundary_condition_data(data_collection)
mat_point_model.set_name("compression_mat_point")

Y = Parameter('Y', 30, 60, 51, units='ksi')
A = Parameter('A', 1, 500, 100, units='ksi')
b = Parameter('b', 5, 30, 20, units='ksi')
E = Parameter('E', 5000, 120000, 10000, units='ksi')
#%%
#
# .. include:: ../../multiple_dakota_studies_in_python_instance_warning.rst
#

calibration = GradientCalibrationStudy(Y, A, b, E)
calibration.set_results_storage_options(results_save_frequency=5)
calibration.add_evaluation_set(mat_point_model, objective, data_collection)
calibration.set_core_limit(5)
results = calibration.launch()
print(results.best)
make_standard_plots("time")
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(8,8)

#%%
# The calibration completes and indicates success with the following output::
#   
#   ***** RELATIVE FUNCTION CONVERGENCE *****
#
# The result plots show good agreement between the simulation and experiments. Additionally,  
# the calibrated parameter values match the stresses observed in the data with a yield stress 
# of ~42 ksi and a saturation stress of ~55 ksi. Note that the elastic modulus seems low at 8,177 ksi versus the 
# expected ~10,000 ksi for aluminum.
# As mentioned earlier, the elastic modulus should generally not be calibrated to tension or 
# compression data since it is well known and documented in the literature. 
