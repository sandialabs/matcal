"""
6061T6 aluminum anisotropy calibration initial point estimation
---------------------------------------------------------------

In this example, we use MatFit and engineering judgement to estimate the 
initial point for our calibration in 
:ref:`6061T6 aluminum calibration with anisotropic yield`.
See that example for more detail on material model 
choice and experimental data review for the material.

.. note::
    Useful Documentation links:

    #. :ref:`Running MatFit`
    #. :class:`~matcal.core.data_importer.FileData`    

First import all needed tools. 
We will be using tools from NumPy, 
MatPlotLib, MatFit and MatCal for this 
example.
"""
import numpy as np
import matplotlib.pyplot as plt

from matcal import *
from matfit.models import Voce
from matfit.fitting import MatFit
# sphinx_gallery_thumbnail_number = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

#%%
# First, we use the 
# :func:`~matcal.core.data_importer.FileData` 
# function to read in the relevant engineering
# stress-strain curves.  
# MatFit will use these to estimate the Voce 
# hardening parameters for the data.
# MatFit's algorithms need the 0.2% offset yield stress, 
# the ultimate stress, the strain at the
# ultimate stress and the failure strain.
# We estimated these values by manipulating the 
# the raw engineering stress-strain data,
# and saving the quantities in CSV files.
# See the :ref:`6061T6 aluminum data analysis`
# example to see how we extracted these data 
# from the engineering stress strain curves. 

all_RD_metrics_CA = FileData("aluminum_6061_data/uniaxial_tension/"
                             "RD_aluminum_AL_6061_tension_stress_metrics_CA.csv")
all_LT_metrics_CA = FileData("aluminum_6061_data/uniaxial_tension/"
                             "LT_aluminum_AL_6061_tension_stress_metrics_CA.csv")
all_ST_metrics_CA = FileData("aluminum_6061_data/uniaxial_tension/"
                             "ST_aluminum_AL_6061_tension_stress_metrics_CA.csv")
all_RD_metrics_NM = FileData("aluminum_6061_data/uniaxial_tension/"
                             "RD_aluminum_AL_6061_tension_stress_metrics_NM.csv")
all_LT_metrics_NM = FileData("aluminum_6061_data/uniaxial_tension/"
                             "LT_aluminum_AL_6061_tension_stress_metrics_NM.csv")
all_ST_metrics_NM = FileData("aluminum_6061_data/uniaxial_tension/"
                             "ST_aluminum_AL_6061_tension_stress_metrics_NM.csv")

#%%
# With the necessary data loaded, 
# we create a function to estimate 
# the Voce hardening material parameters from  
# tension test metrics 
# using MatFit.
# This function takes in a single set of 
# material data metrics and returns 
# a single MatFit solution for the 
# Voce hardening parameters.
def get_voce_params(metrics):
    material_specification = dict(
        ultimate_strength=metrics["ultimate_stress"],
        strain_at_ultimate=metrics["strain_at_ultimate_stress"],
        elongation=metrics['max_strain'],
        yield_stress=metrics['yield'],
        youngs_modulus=10e3,
        poissons_ratio=0.33,
        density=0.00026)
    
    voce_parameters = dict(
        hardening_modulus=dict(value=1.0, lower=0, upper=3000.0, calibrate=True),
        exponential_coefficient=dict(value=15.0, lower=0.0, upper=100, calibrate=True),
        )
    voce_model = Voce(material_specification, voce_parameters, name='Voce')
    MF = MatFit(voce_model)
    MF.fit(solver_settings=dict(method='trf'))
    solution = MF.get_solution()
    return solution

#%%
# With the preceding function available, 
# we create an additional function to 
# loop over a set of uniaxial tension 
# data metrics, pass them to the 
# ``get_voce_params`` function and then 
# extract the desired
# material parameters from the 
# MatFit result. The yield stress and 
# hardening parameters are stored in lists
# for later processing.
def get_voce_params_for_metric_list(metric_list):
    Ys =[]
    As = []
    bs = []
    for metrics in metric_list:
        solution = get_voce_params(metrics)
        As.append(solution['hardening_modulus'])
        bs.append(solution['exponential_coefficient'])
        Ys.append(metrics["yield"])
    return Ys,As,bs

#%%
# Next, we apply the ``get_voce_params_for_metric_list``` 
# function to our
# engineering stress-strain metrics.
rd_Ys_CA, rd_As_CA, rd_bs_CA = get_voce_params_for_metric_list(all_RD_metrics_CA)
lt_Ys_CA, lt_As_CA, lt_bs_CA = get_voce_params_for_metric_list(all_LT_metrics_CA)
st_Ys_CA, st_As_CA, st_bs_CA= get_voce_params_for_metric_list(all_ST_metrics_CA)

rd_Ys_NM, rd_As_NM, rd_bs_NM = get_voce_params_for_metric_list(all_RD_metrics_NM)
lt_Ys_NM, lt_As_NM, lt_bs_NM = get_voce_params_for_metric_list(all_LT_metrics_NM)
st_Ys_NM, st_As_NM, st_bs_NM= get_voce_params_for_metric_list(all_ST_metrics_NM)

#%%
# Although it may be interesting to compare 
# the results from the different test labs (CA vs NM), 
# we assume the test lab has no affect on the 
# tension data results and combine the
# data using list summation.
rd_Ys = rd_Ys_CA+rd_Ys_NM
lt_Ys = lt_Ys_CA+lt_Ys_NM
st_Ys = st_Ys_CA+st_Ys_NM

rd_As = rd_As_CA+rd_As_NM
lt_As = lt_As_CA+lt_As_NM
st_As = st_As_CA+st_As_NM

rd_bs = rd_Ys_CA+rd_bs_NM
lt_bs = lt_Ys_CA+lt_bs_NM
st_bs = st_Ys_CA+st_bs_NM

#%%
# We can now estimate some 
# of the Hill yield parameters. 
# If we assume the yield stress 
# from the LT tests (aligned with the R11 direction)
# is the reference stress for the Hill
# ratios, we can set :math:`R_{11}=1.0` 
# and can estimate :math:`R_{22}` and :math:`R_{33}` from 
# the yield stress values in the RD and ST directions, respectively. 
# See :ref:`6061T6 aluminum data analysis` for more information 
# on the chosen material coordinate system.
R22s = []
R33s = []
for lt_Y in lt_Ys:
    for rd_Y in rd_Ys:
        R22s.append(rd_Y/lt_Y)
    for st_Y in st_Ys:
        R33s.append(st_Y/lt_Y)

#%%
# By looping over each yield stress for each direction, we get
# many estimates for the Hill :math:`R_{22}` and :math:`R_{33}`
# ratios.
# Since we need 
# one value for our
# calibration initial point,
# we average the values to arrive at our initial point
# estimate.
print("Y estimate:", np.average(lt_Ys))
print("R11 estimate:", 1.0) 
print("R22 estimate:", np.average(R22s)) 
print("R33 estimate:", np.average(R33s))
print("A estimate:", np.average(rd_As+lt_As+st_As))
print("b estimate:", np.average(rd_bs+lt_bs+st_bs))

#%%
# We can also plot histograms 
# of the estimated parameters 
# to see if there are any apparent trends 
# or modes in the data.
figsize=[4,3]
plt.figure("Ys", figsize, constrained_layout=True)  
plt.hist(lt_Ys, density=True, alpha=0.8)
plt.xlabel("Y (MPa)")
plt.ylabel("PDF")

plt.figure("R22,R33", figsize, constrained_layout=True)  
plt.hist(R22s, density=True, alpha=0.8, label="$R_{22}$")
plt.hist(R33s, density=True, alpha=0.8, label="$R_{33}$")
plt.xlabel("Hill normal ratio values")
plt.ylabel("PDF")
plt.legend()

plt.figure("As", figsize, constrained_layout=True)  
plt.hist(rd_As+lt_As+st_As, density=True, alpha=0.8)
plt.xlabel("A (MPa)")
plt.ylabel("PDF")

plt.figure("bs", figsize, constrained_layout=True)  
plt.hist(rd_bs+lt_bs+st_bs, density=True, alpha=0.8)
plt.xlabel("b")
plt.ylabel("PDF")

#%%
# The most apparent feature of the data
# is the bimodal distribution for the Voce 
# exponent :math:`b`. This is likely due to anisotropy 
# in the hardening and failure of this material. For the 
# sake of this example, we are ignoring this feature in 
# the data. However, 
# depending on the application, 
# the material model and calibration may need to account 
# for this behavior.
#
# The only three remaining parameters are the 
# Hill shear ratios :math:`R_{12}`, :math:`R_{23}` and 
# :math:`R_{31}`. Estimating these ratios
# cannot be done analytically because
# the shear yield strength cannot be analytically determined 
# from the top hat shear tests used to characterize the material's 
# shear behavior.
# However, we can make a rough guess for the ratios in 
# a similar fashion to what was done for the normal 
# Hill ratios.
# We will look at the load for each specimen when the load-displacement
# slope begins to deviate from linear. By inspecting 
# the data, the deviation from linear appears to occur around a displacement of 
# 0.005 inches. We extract the loads at this displacement 
# for each specimen and categorize them by their loading direction. 
# We then assume that the :math:`R_{12}` ratio
# (aligned with the RTS/TRS directions) will have a value of 1.0 since 
# it has the highest load at this displacement.
# Now we can estimate what the :math:`R_{23}` and :math:`R_{31}` Hill shear ratio values
# will be relative to :math:`R_{12}` by dividing the
# extracted loads for the STR/TSR and RST/SRT directions 
# by the RTS/TRS load.
# The load at 0.005" displacement extracted in the previous example 
# is saved to a file.  
# Once again, we import that data using :func:`~matcal.core.data_importer.FileData`. 
all_top_hat_12_metrics = FileData("aluminum_6061_data/top_hat_shear/"
                                   "RTS_TRS_aluminum_AL_6061_top_hat_metrics.csv")
all_top_hat_23_metrics = FileData("aluminum_6061_data/top_hat_shear/"
                                   "RST_SRT_aluminum_AL_6061_top_hat_metrics.csv")
all_top_hat_31_metrics = FileData("aluminum_6061_data/top_hat_shear/"
                                   "STR_TSR_aluminum_AL_6061_top_hat_metrics.csv")

#%%
# With the load data imported, we estimate  :math:`R_{23}` and  :math:`R_{31}` similarly to how 
# R22 and R33 were estimated.
R23s = []
R31s = []
for load_R12 in all_top_hat_12_metrics["load_at_0.005_in"]:
    for load_23 in all_top_hat_23_metrics["load_at_0.005_in"]:
        R23s.append(load_23/load_R12)
    for load_31 in all_top_hat_31_metrics["load_at_0.005_in"]:
        R31s.append(load_31/load_R12)

#%%
# We then plot the histograms
# and output an average to obtain 
# a single initial point.
plt.figure("R23,R31", figsize, constrained_layout=True)  
plt.hist(R23s, density=True, alpha=0.8, label="$R_{23}$")
plt.hist(R31s, density=True, alpha=0.8, label="$R_{31}$")
plt.ylabel("Hill shear ratio values")
plt.ylabel("PDF")
plt.legend()
plt.show()

print("R23 estimate:", np.average(R23s))
print("R31 estimate:", np.average(R31s))

#%% 
# We now have a complete initial point 
# for our calibration using the finite element 
# models that MatCal provides for a uniaxial 
# tension test and shear top hat test. We will 
# perform this calibration in the next example. 
# See :ref:`6061T6 aluminum calibration with anisotropic yield`
