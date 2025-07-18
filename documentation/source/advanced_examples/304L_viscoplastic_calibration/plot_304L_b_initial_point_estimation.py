"""
304L bar calibration initial point estimation
---------------------------------------------
In this example, we estimate an initial point for our full finite element
model calibration to data 
from :cite:p:`laser_weld_paper`. 
We will use MatFit on the ASTME8 tension data to provide the initial point for the 
next example, 
:ref:`304L stainless steel viscoplastic calibration`.

.. note::
    Useful Documentation links:

    #. :ref:`Running MatFit`

First, import all needed tools. 
We will be using tools from NumPy, 
MatPlotLib, MatFit and MatCal for this 
example.
"""
import numpy as np
from matcal import *
from matfit.models import Voce
from matfit.fitting import MatFit
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)

#%%
# We import the data using the 
# :class:`~matcal.core.data_importer.BatchDataImporter`
tension_data = BatchDataImporter("ductile_failure_ASTME8_304L_data/*.dat", 
                                    file_type="csv").batch

#%%
# To use MatFit, we need to extract certain quantities of interest (QoIs)
# from each engineering stress strain curve. We need 
# the yield stress, ultimate stress, strain at ultimate stress and 
# the elongation strain for each test. We extract those from the 
# :class:`~matcal.core.data.DataCollection` below and store each
# QoI set in a list to be used with MatFit. We use NumPy and
# MatCal's :class:`~matcal.core.data_analysis.determine_pt2_offset_yield`
# to determine these QoIs from the data.
qoi_sets = []
steel_elastic_mod = 29e3
for state, data_sets in tension_data.items():
    for data in data_sets:
        yield_pt = determine_pt2_offset_yield(data, steel_elastic_mod)
        yield_stress = yield_pt[1]
        ultimate_stress = np.max(data["engineering_stress"])
        argmax = np.argmax(data["engineering_stress"])
        strain_at_ultimate = data["engineering_strain"][argmax]
        elongation_strain = np.max(data["engineering_strain"])
        qoi_sets.append([yield_stress, ultimate_stress, strain_at_ultimate, elongation_strain])

#%%
# Next, we write a function that will take those QoIs and provide
# an estimate for a Voce material model :cite:p:`voce1948relationship` using MatFit.
# The function returns the Voce material parameters of 
# saturation stress (:math:`A`) and Voce exponent (:math:`b`) in a solution dictionary. 
# Since we already have an estimate for the yield, we will only be calibrating 
# :math:`A` and :math:`b` with MatFit. MatFit requires specific formatting 
# of input parameters. See the MatFit documentation for more information
# :cite:p:`matfit`. The bounds for our two calibrated parameters are estimated
# from the stress-strain curves and previous experience with the model 
# for austenitic stainless steels.
def get_voce_params(yield_stress, ultimate_stress, strain_at_ultimate, elongation_strain):
    material_specification = dict(
        ultimate_strength = ultimate_stress,
        strain_at_ultimate = strain_at_ultimate,
        elongation = elongation_strain,
        yield_stress = yield_stress,
        youngs_modulus=steel_elastic_mod,
        poissons_ratio=0.27,
        density=7.41e-4)

    voce_parameters = dict(
        hardening_modulus=dict(value=200, lower=0, upper=1e3, calibrate=True),
        exponential_coefficient=dict(value=2.0, lower=0.0, upper=5, calibrate=True),
        )
    voce_model = Voce(material_specification, voce_parameters, name='Voce')
    MF = MatFit(voce_model)
    MF.fit(solver_settings=dict(method='trf'))
    solution = MF.get_solution()
    return solution

#%%
# Next, we write another function to take the QoIs and calculate our 
# Voce material parameters. We will store those in a dictionary for 
# further analysis.
voce_params = {"Ys":[], "As":[], "bs":[]}
for qoi_set in qoi_sets:
    voce_params["Ys"].append(qoi_set[0])
    solution = get_voce_params(*qoi_set)

    voce_params["As"].append(solution['hardening_modulus'])
    voce_params["bs"].append(solution['exponential_coefficient'])

#%%
# First, we make histograms of each parameter. 
# We want to ensure the parameters are as expected and 
# try to understand the cause of any multi-modal behavior.
figsize=[4,3]
plt.figure("Ys", figsize, constrained_layout=True)
plt.hist(voce_params["Ys"], density=True, alpha=0.8)
plt.xlabel("Y (ksi)")
plt.ylabel("PDF")

plt.figure("As", figsize, constrained_layout=True)
plt.hist(voce_params["As"], density=True, alpha=0.8)
plt.xlabel("A (Ksi)")
plt.ylabel("PDF")

plt.figure("bs", figsize, constrained_layout=True)
plt.hist(voce_params["bs"], density=True, alpha=0.8)
plt.xlabel("b")
plt.ylabel("PDF")

#%%
# From these plots there is some slight grouping. However, 
# the parameter values are not spread out over a large range 
# indicating MatFit has provided a good initial guess for the parameters. 
# We can plot the data collection and verify that two groupings of the data are 
# present. We do this with MatCal's :meth:`~matcal.core.data.DataCollection.plot`
# method for :class:`~matcal.core.data.DataCollection` objects.
tension_fig = plt.figure("data", (5,4), constrained_layout=True)
tension_data.plot("engineering_strain", "engineering_stress", 
                  figure=tension_fig, labels='ASTME8 data', 
                  color="#bdbdbd")
plt.xlabel("engineering strain")
plt.ylabel("engineering stress (ksi)")

#%%
# In this plot, two groupings of the data can be seen since there are two 
# groups with different elongation strains. This verifies the 
# results seen in the histograms. Since these tension specimens were 
# extracted from a large diameter bar, the different groupings likely 
# correspond to extraction location and the resulting groupings in stress-strain
# behavior are expected.
#
# Since we are ignoring any material inhomogeneity for this calibration, 
# we will take the average of all calculated values and save that 
# as the initial point for our full finite element model calibration.
voce_initial_point = {}
voce_initial_point["Y_0"] = np.average(voce_params["Ys"])
voce_initial_point["A"] = np.average(voce_params["As"])
voce_initial_point["b"] = np.average(voce_params["bs"])

print(voce_initial_point)
matcal_save("voce_initial_point.serialized", voce_initial_point)
