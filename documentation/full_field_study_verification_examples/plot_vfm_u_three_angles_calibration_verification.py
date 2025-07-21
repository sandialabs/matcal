''' 
Virtual Fields Calibration Verification - Three Data Sets
=========================================================
In this example, we repeat the study performed 
in :ref:`Virtual Fields Calibration Verification` one more time
but now include two additional data sets. We include the ``90_degree``
data set as described in :ref:`Full-field Verification Problem Results`, 
and include a ``45_degree`` data set. The ``45_degree``
data set was generated with 
the material direction rotated 45 degrees away from 
``0_degree`` orientation about the Z axis. We believe this additional 
data set will constrain the objective such that the best parameters
possible using the VFM method are found.

Since the problem setup and data manipulation is 
nearly identical to :ref:`Virtual Fields Calibration Verification`, 
we only add additional commentary and discussion on the 
study results at the end.
'''
from matcal import *

state_0_degree = State("0_degree", angle=0)
synthetic_data_0 = FieldSeriesData("../../../docs_support_files/synthetic_surf_results_0_degree.e", 
                                   state=state_0_degree)

state_45_degree = State("45_degree", angle=45)
synthetic_data_45 = FieldSeriesData("../../../docs_support_files/synthetic_surf_results_45_degree.e", 
                                   state=state_45_degree)

state_90_degree = State("90_degree", angle=90)
synthetic_data_90 = FieldSeriesData("../../../docs_support_files/synthetic_surf_results_90_degree.e",
                                    state=state_90_degree)

dc = DataCollection("synthetic", synthetic_data_0, synthetic_data_45, synthetic_data_90)
import matplotlib.pyplot as plt
dc.plot("displacement", "load",figure=plt.figure())

dc["0_degree"][0] = synthetic_data_0[synthetic_data_0["displacement"] < 0.036]
dc["45_degree"][0] = synthetic_data_45[synthetic_data_45["displacement"] < 0.0325]
dc["90_degree"][0] = synthetic_data_90[synthetic_data_90["displacement"] < 0.040]

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

fig, axes = plt.subplots(3,2, figsize=(10,15), constrained_layout=True)
plot_field(synthetic_data_0, "U", axes[0,0])
plot_field(synthetic_data_0, "V", axes[0,1])
plot_field(synthetic_data_45, "U", axes[1,0])
plot_field(synthetic_data_45, "V", axes[1,1])
plot_field(synthetic_data_90, "U", axes[2,0])
plot_field(synthetic_data_90, "V", axes[2,1])

plt.show()

mat_file_string = """begin material test_material
  density = 1
  begin parameters for model hill_plasticity
    youngs modulus  = {elastic_modulus*1e9}
    poissons ratio  = {poissons}
    yield_stress    = {yield_stress*1e6}

    hardening model = voce
    hardening modulus = {A*1e6}
    exponential coefficient = {n}

    R11 = {R11}
    R22 = {R22}
    R33 = {R33}
    R12 = {R12}
    R23 = {R23}
    R31 = {R31}

    coordinate system = rectangular_coordinate_system
    direction for rotation = 1
    alpha = 0
    second direction for rotation = 3
    second alpha = {angle}
  end
end
"""

with open("modular_plasticity.inc", 'w') as fn:
    fn.write(mat_file_string)

material = Material("test_material", "modular_plasticity.inc", "hill_plasticity")
vfm_model = VFMUniaxialTensionHexModel(material, 
                                       "synthetic_data_files/test_mesh_surf.g", 
                                       0.0625*0.0254)
vfm_model.add_boundary_condition_data(dc)
vfm_model.set_name("test_model")
vfm_model.set_number_of_cores(36)
vfm_model.set_number_of_time_steps(450)
vfm_model.set_displacement_field_names(x_displacement="U", y_displacement="V")
vfm_model.add_constants(elastic_modulus=200, poissons=0.27, R22=1.0, 
                        R33=0.9, R23=1.0, R31=1.0)
from site_matcal.sandia.computing_platforms import is_sandia_cluster
from site_matcal.sandia.tests.utilities import MATCAL_WCID

if is_sandia_cluster():       
    vfm_model.run_in_queue(MATCAL_WCID, 10.0/60.0)
    vfm_model.continue_when_simulation_fails()

vfm_objective = MechanicalVFMObjective()
vfm_objective.set_name("vfm_objective")

Y = Parameter("yield_stress", 100, 500.0)
A = Parameter("A", 100, 4000)
n = Parameter("n", 1, 10)
R11 = Parameter("R11", 0.8, 1.1)
R12 = Parameter("R12", 0.8, 1.1)

param_collection = ParameterCollection("hill voce", Y, A, n, R11, R12)

study = GradientCalibrationStudy(param_collection)
study.set_results_storage_options(results_save_frequency=len(param_collection)+1)
study.set_core_limit(48)
study.add_evaluation_set( vfm_model, vfm_objective, dc)
study.do_not_save_evaluation_cache()
study.set_working_directory("vfm_three_angles", remove_existing=True)

results = study.launch()

calibrated_params = results.best.to_dict()
print(calibrated_params)

goal_results = {"yield_stress":200,
                "A":1500,
                "n":2,
                "R11":0.95, 
                "R12":0.85}

def pe(result, goal):
    return (result-goal)/goal*100

for param in goal_results.keys():
    print(f"Parameter {param} error: {pe(calibrated_params[param], goal_results[param])}")
#%%
# This calibration also completes
# with ``RELATIVE FUNCTION CONVERGENCE``
# indicating the algorithm found a local
# minima and based on our objective 
# sensitivity study it is likely a global minimum
# for the VFM objective and model.
# Additionally, all yield related parameters are 
# identified within 1% error. However, 
# The hardening parameters have up to 5%
# error. This is due to the model form error
# and correlation of these parameters. They
# are negatively correlated so the :math:`n`
# decreased approximately 4% while the :math:`A`
# increased approximately 5%. These changes
# are relatively minor and are due to the model
# form error introduced by the plane stress
# assumption. Overall the results indicate
# the VFM problem is well formulated for gradient 
# methods and can provide adequate calibrations
# if over fitting is avoided.
#
# When we plot the results, we now see 
# that all yield parameters are identified
# quickly with clear minima. The objective verses hardening parameters
# plots 
# show evidence of a slight trough in the objective for these 
# parameters. This is indicating some over fitting 
# of these parameters is occurring due to the model 
# form error introduced due to VFM's plane 
# stress assumption constrain.
import os
init_dir = os.getcwd()
os.chdir("vfm_three_angles")
make_standard_plots("time")
os.chdir(init_dir)

# sphinx_gallery_thumbnail_number = 2
