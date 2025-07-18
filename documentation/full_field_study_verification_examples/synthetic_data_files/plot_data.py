''' 
Virtual Fields Calibration Verification
==================================
In this example, we verify that a calibration 
using MatCal's VFM 
tools will reproduce the 
parameters used in the test model 
to generate the synthetic data described 
in :ref:`Full-field Study Verification`.
Due to the numerical methods 
used in optimization process and the 
errors introduced by the plane stress
assumption inherent in VFM, we expect 
there to be some error in the parameters, 
however, an acceptable result would 
be errors less than 1%. 

To begin we import the MatCal tools necessary for this study
and import the data that will be used for the calibration.
'''
from matcal import *
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})


state_0_degree = State("0_degree", angle=0)
synthetic_data_0 = FieldSeriesData("synthetic_surf_results_0_degree.e", 
                                   state=state_0_degree)
synthetic_data_0.set_name("0_degree")
synthetic_data_0["displacement"] *= 1000
synthetic_data_0["load"] /= 1000

state_90_degree = State("90_degree", angle=90)
synthetic_data_90 = FieldSeriesData("synthetic_surf_results_90_degree.e",
                                    state=state_90_degree)
synthetic_data_90.set_name("90_degree")
synthetic_data_90["displacement"] *= 1000
synthetic_data_90["load"] /= 1000


print(synthetic_data_0["time"][np.argmax(synthetic_data_0["load"])])
print(synthetic_data_90["time"][np.argmax(synthetic_data_90["load"])])

print(synthetic_data_0["load"][-1]/np.max(synthetic_data_0["load"]))
print(synthetic_data_90["load"][-1]/np.max(synthetic_data_90["load"]))
# %% 
# Since VFM requires a 
# plane stress assumption, 
# we must calibrate to 
# portions of the data 
# that most closely adhere 
# to this assumption. 
# For this problem, we must 
# ensure that the data doesn't
# include significant plastic localization. 
# To investigate this, we plot the 
# data load-displacement curve. If 
# the data shows structural load loss, 
# we know the specimen has necked. 
# To plot the data, we add the 
# ``synthetic_data`` 
# :class:`~matcal.full_field.data.FieldData`
# object to a :class:`~matcal.core.data.DataCollection`
# and use the :meth:`~matcal.core.data.DataCollection.plot`
# method to plot the load-displacement curve. 

dc = DataCollection("synthetic", synthetic_data_0, synthetic_data_90)
import matplotlib.pyplot as plt
fig = plt.figure("load-disp", figsize=(5,4), constrained_layout=True)
dc.plot("displacement", "load",figure=fig, show=False, linestyle='-.')
plt.figure("load-disp")
plt.title("")
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (kN)")

def plot_field(data, field, ax):
    c = ax.scatter(1e3*(data.spatial_coords[:,0]), 
                   1e3*(data.spatial_coords[:,1]), 
                   c="#bdbdbd", marker='.', s=1, alpha=0.5)
    c = ax.scatter(1e3*(data.spatial_coords[:,0]+data["U"][-1, :]), 
                   1e3*(data.spatial_coords[:,1]+data["V"][-1, :]), 
                   c=1e3*data[field][-1, :], marker='.', s=3)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    direction = data.state.name.replace("_", " ")
    ax.set_title(f"{direction}")
    ax.set_aspect('equal')
    fig.colorbar(c, ax=ax, label=f"{field} mm")

fig, axes = plt.subplots(2,2, figsize=(10,4), constrained_layout=True)
plot_field(synthetic_data_0, "U", axes[0,0])
plot_field(synthetic_data_0, "V", axes[0,1])
plot_field(synthetic_data_90, "U", axes[1,0])
plot_field(synthetic_data_90, "V", axes[1,1])

plt.show()

