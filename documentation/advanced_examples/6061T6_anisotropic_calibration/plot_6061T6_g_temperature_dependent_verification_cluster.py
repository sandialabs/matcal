"""
6061T6 aluminum temperature dependence verification
---------------------------------------------------
In this example, we verify our calibrated temperature dependence 
functions do not produce unwanted behavior between 
the temperatures at which they were calibrated.

.. note::
    Useful Documentation links:

    #. :ref:`Uniaxial Tension Models`
    #. :class:`~matcal.sierra.models.RoundUniaxialTensionModel`
    #. :class:`~matcal.core.parameter_studies.ParameterStudy`
    
We will perform this verification by running the model at 
many temperatures over our temperature range and inspecting the results. 
To do this, we will generate fictitious boundary condition data at 
all temperatures of interest with independent states. As with the calibrations
in this example suite, these data will have state variables of 
``temperature`` and ``direction``. We will then run a 
:class:`~matcal.core.parameter_studies.ParameterStudy` with 
the appropriate 
:class:`~matcal.sierra.models.RoundUniaxialTensionModel`
in the evaluation set. The study will run a single evaluation 
with parameter values from the results of 
:ref:`6061T6 aluminum temperature dependent calibration`
and
:ref:`6061T6 aluminum calibration with anisotropic yield`.
Once all states are complete, we will plot the result and 
visually inspect the curves to verify the behavior is as desired.

Once again, we begin by importing the tools needed for the calibration and 
setting our default plotting options.
"""

from matcal import *
from site_matcal.sandia.computing_platforms import is_sandia_cluster, get_sandia_computing_platform
from site_matcal.sandia.tests.utilities import MATCAL_WCID

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)

#%%
# Next, we create our fictitious data using NumPy and 
# the MatCal :func:`~matcal.core.data.convert_dictionary_to_data` function. 
# We want to sample the material model at many 
# temperatures over our temperature range and
# choose to run the model from 533 to 1033 R in 
# intervals of 10 R. We create a data set 
# for each temperature that strains the material 
# to an engineering strain of approximately 0.3
# and has zero values for engineering stress.
# The stress values will not be used but are required
# for the evaluation set. We only create states for the :math:`R_{11}`
# direction since the other directions will have similar 
# responses.
temps = np.linspace(533.0, 1033.0, 51)
bc_data = DataCollection("bc data")
for temp in temps:
    state = State(f"temperature_{temp}", temperature=temp, direction="R11")
    data =  convert_dictionary_to_data({"engineering_strain":[0.0, 0.3], 
                                        "engineering_stress":[0.0, 0.0]})
    data.set_state(state)
    bc_data.add(data)

# %%
# With the fictitious boundary condition data created, 
# we create the :class:`~matcal.sierra.models.RoundUniaxialTensionModel`
# as we did in :ref:`6061T6 aluminum temperature dependent calibration`
# and add the :class:`~matcal.core.data.DataCollection` that we created
# as the model model boundary condition data.   
material_filename = "hill_plasticity_temperature_dependent.inc"
material_model = "hill_plasticity"
material_name = "ductile_failure_6061T6"
sierra_material = Material(material_name, material_filename, material_model)

gauge_radius = 0.125
element_size = gauge_radius/8
geo_params = {"extensometer_length": 0.5,
               "gauge_length": 0.75, 
               "gauge_radius": gauge_radius, 
               "grip_radius": 0.25, 
               "total_length": 3.2, 
               "fillet_radius": 0.25,
               "taper": 0.0015,
               "necking_region":0.375,
               "element_size": element_size,
               "mesh_method":3, 
               "grip_contact_length":0.8}

model = RoundUniaxialTensionModel(sierra_material, **geo_params)            
model.set_name("tension_model")
model.add_boundary_condition_data(bc_data)
model.set_allowable_load_drop_factor(0.70)

if is_sandia_cluster():       
    platform = get_sandia_computing_platform()   
    model.set_number_of_cores(platform.get_processors_per_node())
    model.run_in_queue(MATCAL_WCID, 0.5)
    model.continue_when_simulation_fails()
else:
    model.set_number_of_cores(8)

#%%
# We now create our parameters for our parameter 
# study. The parameters are the parameters 
# from :ref:`6061T6 aluminum temperature dependent calibration`
# and
# :ref:`6061T6 aluminum calibration with anisotropic yield` with 
# their current value set to their calibration values.
RT_calibrated_params = matcal_load("anisotropy_parameters.serialized")

yield_stress = Parameter("yield_stress", 15, 50, 
                         RT_calibrated_params["yield_stress"])
hardening = Parameter("hardening", 0, 60, 
        RT_calibrated_params["hardening"])
b = Parameter("b", 10, 40,
        RT_calibrated_params["b"])
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

high_temp_calibrated_params = matcal_load("temperature_dependent_parameters.serialized")

y_scale_factor_672_calibrated = high_temp_calibrated_params["Y_scale_factor_672"]
y_scale_factor_852_calibrated = high_temp_calibrated_params["Y_scale_factor_852"]
y_scale_factor_1032_calibrated = high_temp_calibrated_params["Y_scale_factor_1032"]

Y_scale_factor_672  = Parameter("Y_scale_factor_672", 0.85, 1, 
                                y_scale_factor_672_calibrated)
Y_scale_factor_852  = Parameter("Y_scale_factor_852", 0.45, 0.85, 
                                y_scale_factor_852_calibrated)
Y_scale_factor_1032 = Parameter("Y_scale_factor_1032", 0.05, 0.45, 
                                y_scale_factor_1032_calibrated)

A_scale_factor_672_calibrated = high_temp_calibrated_params["A_scale_factor_672"]
A_scale_factor_852_calibrated = high_temp_calibrated_params["A_scale_factor_852"]
A_scale_factor_1032_calibrated = high_temp_calibrated_params["A_scale_factor_1032"]

A_scale_factor_672  = Parameter("A_scale_factor_672", 0.0, 
                                2*A_scale_factor_672_calibrated)
A_scale_factor_852  = Parameter("A_scale_factor_852", 0.0, 
                                2*A_scale_factor_852_calibrated)
A_scale_factor_1032 = Parameter("A_scale_factor_1032", 0.0, 
                                2*A_scale_factor_1032_calibrated)

b_scale_factor_672_calibrated = high_temp_calibrated_params["b_scale_factor_672"]
b_scale_factor_852_calibrated = high_temp_calibrated_params["b_scale_factor_852"]
b_scale_factor_1032_calibrated = high_temp_calibrated_params["b_scale_factor_1032"]

b_scale_factor_672  = Parameter("b_scale_factor_672", 0.0, 
                                3*b_scale_factor_672_calibrated, 
                                b_scale_factor_672_calibrated)
b_scale_factor_852  = Parameter("b_scale_factor_852", 0.0, 
                                3*b_scale_factor_852_calibrated, 
                                b_scale_factor_852_calibrated)
b_scale_factor_1032 = Parameter("b_scale_factor_1032", 0.0, 
                                3*b_scale_factor_1032_calibrated, 
                                b_scale_factor_1032_calibrated)

#%%
# To simplify setting up the parameter study, 
# we put all the parameters in a :class:`~matcal.core.parameters.ParameterCollection`.
pc = ParameterCollection("all_params", 
                         yield_stress, 
                         hardening,
                         b,
                         R22,
                         R33,
                         R12,
                         R23, 
                         R31,
                         Y_scale_factor_672, 
                         A_scale_factor_672, 
                         b_scale_factor_672, 
                         Y_scale_factor_852, 
                         A_scale_factor_852, 
                         b_scale_factor_852, 
                         Y_scale_factor_1032, 
                         A_scale_factor_1032, 
                         b_scale_factor_1032)

#%%
# Now we can create our parameter study
# and add an evaluation set. An objective 
# is required, but will not be used for this example except 
# for results access by name when the study is complete. 
study = ParameterStudy(pc)
study.set_core_limit(60)
obj = CurveBasedInterpolatedObjective("engineering_strain", "engineering_stress")
obj.set_name('objective')
study.add_evaluation_set(model, obj, bc_data)

#%%
# Parameter studies require the user to set 
# parameter sets to be evaluated and will not 
# run the parameter current values by default. 
# As a result, we pass the current values 
# from our parameter collection as a parameter set 
# to be evaluated and then run the study. 
study.add_parameter_evaluation(**pc.get_current_value_dict())
results = study.launch()

#%%
# When the study finishes, 
# we retrieve the simulation 
# results
sim_dc = results.simulation_history[model.name]
#%%
# We then can plot the results 
# using :meth:`~matcal.core.data.DataCollection.plot` 
# and color the results according to temperature 
# as was done in :ref:`6061T6 aluminum temperature dependent calibration`.
cmap = cm.get_cmap("RdYlBu")
def get_colors(data_dc):
    colors = {}
    for state_name in data_dc.state_names:
        temp = data_dc.states[state_name]["temperature"]
        colors[temp] = cmap(1.0-(temp-533.0)/(1032.0-533.0))
    return colors 
colors = get_colors(sim_dc)

fig = plt.figure(constrained_layout=True)
for state_name in sim_dc.state_names:
    state = sim_dc.states[state_name]
    temperature = state["temperature"]
    sim_dc.plot("engineering_strain", "engineering_stress", labels="suppress",
                state=state, color=colors[temperature], show=False, figure=fig, 
                linestyle="-")
plt.xlabel("engineering strain (.)")
plt.ylabel("engineering stress (psi)")
  
plt.show()

#%%
# As can be seen in the plot, the curves at the different
# temperatures do not cross which would result if  
# the material was stronger at a higher temperature than 
# some lower temperature. Since the results
# do not exhibit this crossing behavior, the fit 
# is acceptable. Although, this is not a rigorous check to ensure
# the material is always weaker at lower temperatures, it is 
# enough to provide some confidence that the fit is useable 
# for most circumstances.
