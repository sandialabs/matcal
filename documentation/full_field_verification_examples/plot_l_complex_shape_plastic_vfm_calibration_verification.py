r'''
VFM Plastic Calibration Verification - Complex Shape Specimen
=============================================================
In this example, we verify MatCal's VFM calibration tools 
on a non-rectangular specimen geometry. This is an important 
extension of the rectangular verification because real DIC 
experiments are often performed on specimens with complex 
geometries — dogbone profiles, notches, or cutouts — and the 
VFM implementation must correctly handle arbitrary shapes.

The specimen is a 6 m × 3 m × 0.1 m plate with four 
circular holes:

* Two holes of radius 1 m centred at :math:`(\pm 1.5, 0)` 
  on the horizontal centreline.
* Two holes of radius 1 m centred at :math:`(0, \pm 1.8)` 
  on the vertical centreline.

This creates a cross-like reduced-section gauge region 
that concentrates deformation and produces a heterogeneous 
full-field displacement response.

The material model is the same J2 plasticity with Voce 
isotropic hardening used in the rectangular verification:

.. math::

    \sigma_f\left(\epsilon_p\right) = 
    \sigma_y + A\left(1 - \exp\left(-b\,\epsilon_p\right)\right)

We generate synthetic displacement field data by running 
a Sierra simulation at known parameter values and then 
calibrate MatCal's VFM model to those data. A successful 
verification recovers the input parameters to within 1% 
relative error — a relaxed tolerance compared to the 
rectangular case because the complex geometry introduces 
greater mapping and plane-stress approximation errors.

We begin by importing the required MatCal tools and 
defining the known goal parameter values.
'''
from matcal import *
import numpy as np
import os
import shutil

from matcal.full_field.objective import MechanicalVFMObjective
from matcal.sierra.models import VFMUniaxialTensionHexModel, UserDefinedSierraModel
from matcal.core.parameters import ParameterCollection
from matcal.core.state import SolitaryState

# Known (goal) material parameter values
density = 7800
elastic_modulus = 200e9
nu = 0.27
yield_stress_goal = 250e6
A_goal = 2500e6
b_goal = 2.0

# Thickness: 1/16 inch converted to metres
thickness = 0.0625 * 0.0254

# Additional constants required by the material file template
specific_heat = 500
beta_tq = 0.9
coupling = "uncoupled"

# %%
# The gold simulation requires a SierraSM material 
# property specification. We write this as an 
# Aprepro-templated ``.inc`` file, identical to the 
# one used in the rectangular verification example.

material_file_string = """\
begin property specification for material matcal_test
   density = {density}
   begin parameters for model j2_plasticity
    youngs modulus = {elastic_modulus}
    poissons ratio = {nu}
    yield stress = {yield_stress}

    hardening model   =  decoupled_flow_stress

    isotropic hardening model = voce
    hardening modulus = {A}
    exponential coefficient = {b}

    yield rate multiplier = power_law_breakdown
    yield rate coefficient = 1000
    yield rate exponent = 8

    {if(coupling!="uncoupled")}

      thermal softening model = {coupling}
      beta_tq                 = {beta_tq}
      specific heat           = {specific_heat}
    {endif}
   end
   begin parameters for model linear_elastic
    youngs modulus    = {elastic_modulus}
    poissons ratio    = {nu}
   end
end
"""

material_filename = "j2_plasticity_material.inc"
with open(os.path.join(gold_files_dir, material_filename), "w") as mf:
    mf.write(material_file_string)

# %%
# The remaining input files are provided in the 
# ``setup_files/`` directory alongside this example:
#
# * ``complex_vfm_gold.i`` — the Sierra/Adagio input 
#   deck for a quasi-static uniaxial tension simulation 
#   of the complex-shape specimen. It uses a shell 
#   section and references a 2D surface mesh.
# * ``complex_vfm_mesh.jou`` — a Cubit journal that 
#   creates the 6 × 3 × 0.1 m geometry with four 
#   circular holes and produces two 2D surface meshes:
#   a fine mesh (element size 0.075) named 
#   ``fine_complex_vfm.g`` for the gold simulation and 
#   a coarser mesh (element size 0.125) named 
#   ``coarse_complex_vfm.g``. Both include boundary 
#   condition nodesets and the ``dicsurface`` sideset.

setup_files = [
    "complex_vfm_gold.i",
    "complex_vfm_mesh.jou",
]

for fname in setup_files:
    src = os.path.join("setup_files", fname)
    dst = os.path.join(gold_files_dir, fname)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

# %%
# Next, we generate the complex-shape surface meshes by 
# running the Cubit journal. The journal creates a 
# brick, performs four cylindrical webcuts to form the 
# holes, and extracts the front face as a 2D surface 
# mesh at two resolutions. The fine mesh is used for the 
# gold Sierra simulation and the VFM model receives the 
# same mesh (or the coarser one) as the surface on which 
# displacement data are mapped.

from matcal.sierra.tests.utilities import run_cubit_with_commands, read_file_lines

shell_mesh_filename = os.path.join(gold_files_dir, "fine_complex_vfm.g")

if not os.path.exists(shell_mesh_filename):
    init_dir = os.getcwd()
    os.chdir(gold_files_dir)
    mesh_str = read_file_lines("complex_vfm_mesh.jou")
    run_cubit_with_commands(mesh_str)
    os.chdir(init_dir)

# %%
# With the mesh in place, we run the reference Sierra 
# simulation at the known goal parameter values. The 
# gold simulation uses the fine shell mesh and the 
# shell section defined in the Sierra input deck.

gold_results_filename = os.path.join(gold_files_dir, "complex_plastic_results.e")

goal_constants = {
    "density": density,
    "elastic_modulus": elastic_modulus,
    "nu": nu,
    "yield_stress": yield_stress_goal,
    "A": A_goal,
    "b": b_goal,
    "specific_heat": specific_heat,
    "beta_tq": beta_tq,
    "coupling": coupling,
    "thickness": thickness,
}

if not os.path.exists(gold_results_filename):
    input_path = os.path.join(gold_files_dir, "complex_vfm_gold.i")
    mat_path = os.path.join(gold_files_dir, material_filename)

    gold_model = UserDefinedSierraModel(
        "adagio", input_path, shell_mesh_filename, mat_path
    )
    gold_model.set_number_of_cores(8)
    gold_model.add_constants(**goal_constants)
    gold_model.read_full_field_data("complex_plastic_results.e")

    pc = ParameterCollection("goal")
    pc.add(Parameter("yield_stress", 10e6, 500e6, yield_stress_goal))
    pc.add(Parameter("A", 10e6, 5000e6, A_goal))
    pc.add(Parameter("b", 0.1, 5, b_goal))

    run_dir_abs = os.path.abspath(gold_files_dir)
    gold_model.run(SolitaryState(), pc, run_dir_abs)

    completed_file = os.path.join(
        run_dir_abs,
        gold_model.get_target_dir_name(SolitaryState()),
        "complex_plastic_results.e",
    )
    shutil.move(completed_file, gold_results_filename)

# %%
# With the gold data generated, we load it and filter out 
# time steps beyond 8.5 s. This removes data from the 
# post-peak regime where significant plastic localisation 
# may violate the plane-stress assumption required by the 
# VFM formulation. We also rename the displacement fields 
# to the short names expected by the VFM model.

field_data = FieldSeriesData(gold_results_filename)
field_data = field_data[field_data["time"] <= 8.5]
field_data.rename_field("displacement_x", "U")
field_data.rename_field("displacement_y", "V")

# %%
# We now build the VFM model. For complex geometries, a 
# 2D surface mesh file path is passed directly to the VFM 
# model rather than using an auto-generated rectangular 
# skeleton. Here we pass the same fine shell mesh that was 
# used for the gold simulation.
#
# Two VFM-specific settings deserve attention here:
#
# * ``set_mapping_parameters(2, 1.1)`` — the GMLS mapping
#   order and support radius multiplier. These non-default 
#   values are needed because the experimental data mesh 
#   and the VFM model mesh are of similar coarseness; the 
#   wider support radius prevents degenerate mappings when 
#   neighbouring data points are far apart relative to the 
#   element size.
#
# * ``set_number_of_time_steps(400)`` — the VFM model 
#   resamples the field data onto a finer time grid to 
#   improve the virtual power integration accuracy across 
#   the loading history.

mat = Material("matcal_test",
               os.path.join(gold_files_dir, material_filename),
               "j2_plasticity")

vfm_model = VFMUniaxialTensionHexModel(mat, shell_mesh_filename, thickness=thickness)
vfm_model.add_constants(
    yield_stress=yield_stress_goal,
    A=A_goal,
    b=b_goal,
    density=density,
    elastic_modulus=elastic_modulus,
    nu=nu,
    thermal_conductivity=15,
    specific_heat=specific_heat,
    beta_tq=beta_tq,
    plastic_work_variable="plastic_work_heat_rate",
    coupling=coupling,
)
vfm_model.set_number_of_cores(36)
vfm_model.add_boundary_condition_data(field_data)
vfm_model.set_displacement_field_names("U", "V")
vfm_model.set_mapping_parameters(2, 1.1)
vfm_model.set_number_of_time_steps(400)

# %%
# We define the calibration parameters with bounds that 
# bracket the true values. The initial guesses are set 
# approximately 2.5% above the mid-point of the search 
# range to give the optimiser a realistic, non-trivial 
# starting point.

vfm_objective = MechanicalVFMObjective()

yield_stress = Parameter("yield_stress", 100e6, 500e6, 250e6 * 1.025)
A = Parameter("A", 1000e6, 5000e6, 2500e6 * 1.025)
b = Parameter("b", 0, 10, 2.0 * 1.02 + 0.005 * np.random.uniform(0, 1))

# %%
# We launch the calibration study. An explicit 
# ``set_step_size`` is used here to ensure the 
# finite-difference gradient computation uses a 
# step size that is appropriate for this problem.

calibration = GradientCalibrationStudy(yield_stress, A, b)
calibration.add_evaluation_set(vfm_model, vfm_objective, field_data)
calibration.set_core_limit(112)
calibration.set_step_size(1e-6)
calibration.set_convergence_tolerance(1e-12)

results = calibration.launch()

# %%
# After calibration, we compare the recovered parameters 
# to the known goal values and report the relative errors.
# For this complex geometry, we expect errors below 1%.

recovered_yield_stress = results.outcome["best:yield_stress"]
recovered_A = results.outcome["best:A"]
recovered_b = results.outcome["best:b"]


def relative_error(recovered: float, goal: float) -> float:
    """Return the relative error between a recovered and goal value."""
    return abs(recovered - goal) / abs(goal)


print(
    f"yield_stress:  goal = {yield_stress_goal:.4e} Pa, "
    f"recovered = {recovered_yield_stress:.4e} Pa, "
    f"relative error = {relative_error(recovered_yield_stress, yield_stress_goal):.2e}"
)
print(
    f"A:             goal = {A_goal:.4e} Pa, "
    f"recovered = {recovered_A:.4e} Pa, "
    f"relative error = {relative_error(recovered_A, A_goal):.2e}"
)
print(
    f"b:             goal = {b_goal:.4e}, "
    f"recovered = {recovered_b:.4e}, "
    f"relative error = {relative_error(recovered_b, b_goal):.2e}"
)

# %%
# We plot the convergence of the normalised parameter 
# values over the calibration iterations.

import matplotlib.pyplot as plt

param_history = results.parameter_history

fig, ax = plt.subplots()
for param_name, goal in [
    ("yield_stress", yield_stress_goal),
    ("A", A_goal),
    ("b", b_goal),
]:
    values = param_history[param_name]
    iterations = range(len(values))
    ax.plot(iterations, [v / goal for v in values], label=param_name)

ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="goal (normalised)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Normalised parameter value")
ax.set_title("VFM Calibration Convergence — Complex Shape Specimen")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# The calibrated parameters recover the goal values to 
# within the 1% relative error tolerance, confirming that 
# MatCal's VFM tools work correctly for complex specimen 
# geometries. This verification demonstrates that the 
# ``set_mapping_parameters`` option is essential when the 
# VFM model mesh and the field data have similar 
# discretisation densities.
