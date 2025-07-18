"""
6061T6 aluminum temperature calibration initial point estimation
----------------------------------------------------------------
In this example, we use MatFit to estimate the initial point for our 
material model temperature dependence functions. For more on the chosen 
model form see :ref:`6061T6 aluminum temperature dependent data analysis`.

.. note::
    Useful Documentation links:

    #. :ref:`Running MatFit`
    #. :class:`~matcal.core.data_importer.FileData`    

We begin by importing the data metrics that are required for MatFit
that were calculated in the previously referenced example.  
We will use the FileData tool to perform the import, so we import 
all of MatCal's tools. We also import MatFit tools, NumPy, matplotlib and glob before 
setting our preferred plotting defaults. 
"""
from matcal import *
from matfit.models import Voce
from matfit.fitting import MatFit

from glob import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)
#%%
# Since we want to keep these data identifiable by state, 
# we write a function to perform the import, create the correct 
# state from the filename and then add the data to a 
# :class:`~matcal.core.data.DataCollection`.
files = glob("*_matfit_metrics.csv")
metrics_collection = DataCollection("matfit metrics")
for filename in files:
    filename_split = filename.split("_")
    temperature = float(filename_split[1])
    direction = filename_split[3]
    new_state = State(f"temperature_{temperature:0.0f}_direction_{direction}", 
                      temperature=temperature, direction=direction)
    new_data = FileData(filename, state=new_state)
    metrics_collection.add(new_data)

#%%
# With all the required data available, 
# we can perform the MatFit initial point 
# estimate. For MatFit, three steps 
# are required. We must put the material 
# data metrics into the correct data form, 
# specify the parameters we want to 
# calibrate and, finally, run the 
# MatFit calibration. 
# We perform each of these steps in separate 
# functions as shown below. 
# In the first function, we define the 
# required material data metrics and 
# properties that MatFit needs in a dictionary 
# with the correct keys. 
# The elastic constants will not be 
# calibrated so they are specified 
# as global constants. The remaining 
# material data metrics are averaged 
# from all the data for a given state. 
# We are not concerned with uncertainty quantification 
# at this point and the average should give 
# an acceptable initial point for the full 
# MatCal calibration to follow.
youngs_modulus=10.3e6
poissons_ratio=0.33
def prepare_material_metrics(state):
    state_data = metrics_collection[state][0]
    material_metrics = {"ultimate_strength":np.average(state_data["ultimate_stress"]),
    "strain_at_ultimate":np.average(state_data["strain_at_ultimate_stress"]),
    "elongation":np.average(state_data["failure_strain"]),
    "yield_stress":np.average(state_data["yield_stress"]),
    "youngs_modulus":youngs_modulus,
    "poissons_ratio":poissons_ratio}
    
    return material_metrics

#%%
# The next function defines the parameters 
# we wish to calibrate using MatFit. Since we want to 
# define temperature dependent functions for 
# the yield and Voce hardening parameters, 
# these are the three parameters for which we 
# define MatFit parameters. 
#
# .. note::
#       The MatFit parameters are defined as dictionaries 
#       where certain keywords are required 
#       in order to be valid. See :cite:p:`matfit`.

def prepare_matfit_parameters():
    hardening_modulus = dict(value=1.0, lower=0, upper=100.0e6, 
                            calibrate=True)
    exponential_coefficient=dict(value=15.0, lower=0.0, upper=300, 
                                calibrate=True)
    yield_stress=dict(value=40e3, lower=0.0, upper=100e3, 
                                     calibrate=True)
    voce_parameters = dict(hardening_modulus=hardening_modulus,
                           exponential_coefficient=exponential_coefficient,
                           yield_stress=yield_stress)
    return voce_parameters

#%%
# In the final function, 
# We run MatFit for our Voce hardening model. 
# First it calls the preceding two functions 
# to prepare the material metrics and parameters. 
# It then calibrates the Voce material model parameters
# and returns the calibration results.
def matfit_single_state(state):
    material_metrics = prepare_material_metrics(state)
    voce_parameters = prepare_matfit_parameters()
    voce_model = Voce(material_metrics, voce_parameters, name='Voce')
    MF = MatFit(voce_model)
    MF.fit(solver_settings=dict(method='trf'))
    solution = MF.get_solution()
    param_return_tuple = (solution["yield_stress"], 
                          solution["hardening_modulus"], 
                          solution["exponential_coefficient"])
    return param_return_tuple

#%%
# We now are able to estimate the Voce hardening 
# parameters and yield stresses for our material at 
# the higher temperatures.  We will only use the 
# data for the :math:`R_{11}` direction since
# this direction's yield stress is the reference stress
# for the material's Hill yield. Any small errors in the other 
# directions' Voce hardening parameters will be corrected
# when the full calibration is performed. 
# We call the ``matfit_single_state`` function 
# on each temperature for the :math:`R_{11}`
# material direction and store the parameters for each 
# temperature.
y_672_ip, A_672_ip,  b_672_ip  = matfit_single_state("temperature_672_direction_R11")
y_852_ip, A_852_ip,  b_852_ip  = matfit_single_state("temperature_852_direction_R11")
y_1032_ip, A_1032_ip, b_1032_ip = matfit_single_state("temperature_1032_direction_R11")

#%%
# The temperature dependence functions for the parameters 
# will scale the room temperature values using a piecewise-linear 
# function. 
# As a result, we will need the room temperature (533 R)
# parameter values, so we create a dictionary storing these parameters 
# that resulted from the calibration in
# :ref:`6061T6 aluminum calibration with anisotropic yield`.
RT_calibrated_params = matcal_load("anisotropy_parameters.serialized")

#%%
# Now we can use the MatFit data and the room temperature 
# parameters to create our temperature dependent scaling functions for the 
# yield stress and Voce hardening parameters.
# For each parameter, we created an array that contains the room temperature 
# calibration value and the MatFit estimates for the high temperature 
# ordered from lowest temperature to highest. The array for each parameter
# is normalized by the room temperature value for that parameter. 
# The resulting array is the scaling value for each parameter at each 
# temperature where data are available.
yields = np.array([RT_calibrated_params["yield_stress"]*1e3, y_672_ip, y_852_ip, y_1032_ip])
yield_scale_factors = yields/1000/RT_calibrated_params["yield_stress"]

As = np.array([RT_calibrated_params["hardening"]*1e3, A_672_ip, A_852_ip, A_1032_ip])
A_scale_factors=As/1000/RT_calibrated_params["hardening"]

bs = np.array([RT_calibrated_params["b"], b_672_ip, b_852_ip, b_1032_ip])
b_scale_factors=bs/RT_calibrated_params["b"]

#%%
# We now plot the scaling functions to verify 
# they meet our expectations. 
plt.figure()
plt.plot([533, 672, 852, 1032], yield_scale_factors, label='yield stress')
plt.plot([533, 672, 852, 1032], A_scale_factors, label='Voce hardening modulus')
plt.plot([533, 672, 852, 1032], b_scale_factors, label='Voce exponential coefficient')
plt.ylabel("temperature scaling function (.)")
plt.xlabel("temperature (R)")
plt.legend()
plt.show()

#%%
# In the plot, we can see that the yield and Voce saturation stress 
# (referred to as hardening modulus in LAME and MatFit)
# generally decrease wth increasing temperature as expected. 
# The Voce exponential coefficient generally increases as the temperature 
# increases. At 852 R, the exponential coefficient function increases significantly 
# before reducing again at 1032 R. Ideally, this function should be 
# monotonically increasing, however, this may not be an issue. 
# We will move forward with this as our initial estimate for the functions 
# and verify this does not cause undesirable behavior once the MatCal 
# calibration is complete. We print the scale factors at each function 
# below and write them to a file
# so that they can be seen and imported into :ref:`6061T6 aluminum temperature dependent calibration`
# as the initial point for the calibration.
print(yield_scale_factors)
print(A_scale_factors)
print(b_scale_factors)
output_params = {"Y_scale_factor_672":yield_scale_factors[1] ,
                 "Y_scale_factor_852":yield_scale_factors[2], 
                 "Y_scale_factor_1032":yield_scale_factors[3],
                 "A_scale_factor_672":A_scale_factors[1], 
                 "A_scale_factor_852":A_scale_factors[2], 
                 "A_scale_factor_1032":A_scale_factors[3],
                 "b_scale_factor_672":b_scale_factors[1], 
                 "b_scale_factor_852":b_scale_factors[2], 
                 "b_scale_factor_1032":b_scale_factors[3]}
matcal_save("temperature_parameters_initial.serialized", output_params)