r'''
VFM Plastic Calibration Verification - Rectangular Specimen
============================================================
In this example, we verify that MatCal's VFM tools can 
accurately calibrate a J2 plasticity material model using 
full-field displacement data from a rectangular uniaxial 
tension specimen. The material model uses Voce isotropic 
hardening, defined by the flow stress

.. math::

    \sigma_f\left(\epsilon_p\right) = 
    \sigma_y + A\left(1 - \exp\left(-b\,\epsilon_p\right)\right)

where :math:`\sigma_y` is the yield stress, :math:`A` is 
the hardening modulus, :math:`b` is the exponential 
hardening coefficient, and :math:`\epsilon_p` is the 
equivalent plastic strain.

We follow a synthetic-data verification approach: we first 
generate gold-standard displacement field data by running 
a Sierra solid-mechanics simulation at known parameter 
values, and then calibrate MatCal's VFM model to those data. 
A successful verification recovers the input parameters to 
within a tight tolerance.

The specimen is a thin rectangular plate of width 
:math:`W = 0.1` m, height :math:`L = 0.2` m, and thickness
:math:`T = 0.001` m, loaded in uniaxial tension in the 
Y direction.

We begin by importing MatCal tools and defining the 
known material parameters that will be used to generate 
the synthetic data and serve as the calibration targets.
'''
from matcal import *
import numpy as np
import os
import shutil

from matcal.full_field.TwoDimensionalFieldGrid import auto_generate_two_dimensional_field_grid
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

# Additional constants required by the material file template
specific_heat = 500
beta_tq = 0.9
coupling = "uncoupled"

# %%
# We create a working directory for the gold simulation 
# and then write the SierraSM material property 
# specification as an Aprepro-templated ``.inc`` file.
# The template uses J2 plasticity with Voce isotropic 
# hardening and a power-law-breakdown rate multiplier. 
# Aprepro variables such as ``{yield_stress}``, 
# ``{A}``, and ``{b}`` are substituted by MatCal 
# at runtime.

gold_files_dir = "rectangle_vfm_gold_data"
os.makedirs(gold_files_dir, exist_ok=True)

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
# * ``rectangle_vfm_gold.i`` — the Sierra/Adagio input 
#   deck for a quasi-static uniaxial tension simulation 
#   of the rectangular specimen. It prescribes symmetric 
#   displacements on the top and bottom nodesets and 
#   outputs surface displacement fields.
# * ``rectangle_vfm_mesh.jou`` — a Cubit journal that 
#   creates the 0.1 × 0.2 × 0.001 m rectangular mesh 
#   with the required nodesets and sidesets. The journal 
#   uses Aprepro variables (``N``, ``solid_mesh``, 
#   ``mesh_name``) to control the mesh resolution and 
#   whether a 3D solid or 2D surface mesh is produced.
# * ``make_fine_solid_rect_mesh.inc`` — an Aprepro include
#   file that sets the variables for the fine hex8 solid 
#   mesh (N=10, solid_mesh="true") used for the gold 
#   simulation.
# * ``make_coarse_surface_rect_mesh.inc`` — an Aprepro 
#   include file that sets the variables for the coarser 
#   2D surface mesh (N=5, solid_mesh="false") that could 
#   be used as VFM model input.

setup_files = [
    "rectangle_vfm_gold.i",
    "rectangle_vfm_mesh.jou",
    "make_fine_solid_rect_mesh.inc",
    "make_coarse_surface_rect_mesh.inc",
]

for fname in setup_files:
    src = os.path.join("setup_files", fname)
    dst = os.path.join(gold_files_dir, fname)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

# %%
# Next, we generate the rectangular solid mesh by 
# combining the Aprepro include file 
# ``make_fine_solid_rect_mesh.inc`` with the Cubit 
# journal ``rectangle_vfm_mesh.jou``. The include file 
# sets ``N=10`` and ``solid_mesh="true"`` so the journal 
# produces a fine hex8 solid mesh named ``thin_rect.g``.
# We read both files and pass their combined content 
# to Cubit, matching the approach used in MatCal's test 
# infrastructure.

from matcal.sierra.tests.utilities import run_cubit_with_commands, read_file_lines

solid_mesh_filename = os.path.join(gold_files_dir, "thin_rect.g")

if not os.path.exists(solid_mesh_filename):
    init_dir = os.getcwd()
    os.chdir(gold_files_dir)
    mesh_str = read_file_lines("make_fine_solid_rect_mesh.inc") + \
               read_file_lines("rectangle_vfm_mesh.jou")
    run_cubit_with_commands(mesh_str)
    os.chdir(init_dir)

# %%
# With the mesh in place, we run the reference Sierra 
# simulation using 
# :class:`~matcal.sierra.models.UserDefinedSierraModel`.
# We pass all material constants — including the known 
# goal parameter values — so that the simulation reflects 
# the true material behaviour we want to recover during 
# calibration. The Sierra input deck includes the 
# material file via an Aprepro ``{include(...)}`` 
# directive; MatCal writes all constant and parameter 
# values to Aprepro include files that Sierra reads at 
# runtime.

gold_results_filename = os.path.join(gold_files_dir, "plastic_results.e")

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
    "mat_model": "j2_plasticity",
    "solid_mesh": "true",
    "aprepro_file": os.path.join(gold_files_dir, "make_fine_solid_rect_mesh.inc"),
}

if not os.path.exists(gold_results_filename):
    input_path = os.path.join(gold_files_dir, "rectangle_vfm_gold.i")
    mat_path = os.path.join(gold_files_dir, material_filename)

    gold_model = UserDefinedSierraModel(
        "adagio", input_path, solid_mesh_filename,
        "make_fine_solid_rect_mesh.inc", mat_path
    )
    gold_model.set_number_of_cores(4)
    gold_model.add_constants(**goal_constants)
    gold_model.read_full_field_data("plastic_results.e")

    pc = ParameterCollection("goal")
    pc.add(Parameter("yield_stress", 10e6, 500e6, yield_stress_goal))
    pc.add(Parameter("A", 10e6, 5000e6, A_goal))
    pc.add(Parameter("b", 0.1, 5, b_goal))

    run_dir_abs = os.path.abspath(gold_files_dir)
    gold_model.run(SolitaryState(), pc, run_dir_abs)

    completed_file = os.path.join(
        run_dir_abs,
        gold_model.get_target_dir_name(SolitaryState()),
        "plastic_results.e",
    )
    shutil.move(completed_file, gold_results_filename)

# %%
# With the gold results file generated, we load it as a 
# :class:`~matcal.full_field.data.FieldSeriesData` object 
# and rename the displacement fields to the names expected 
# by the VFM model boundary condition interface.

field_data = FieldSeriesData(gold_results_filename)
field_data.rename_field("displacement_x", "U")
field_data.rename_field("displacement_y", "V")

# %%
# Next, we create the VFM model skeleton mesh.
# :func:`~matcal.full_field.TwoDimensionalFieldGrid.auto_generate_two_dimensional_field_grid`
# generates a rectangular grid that spans the spatial 
# extent of the field data. We use 5 nodes in X and 
# 10 nodes in Y.

number_of_nodes_x = 5
number_of_nodes_y = 10
thickness = 0.001

mesh_skeleton = auto_generate_two_dimensional_field_grid(
    number_of_nodes_x, number_of_nodes_y, field_data
)

# %%
# We can now build the VFM model. We use a 
# :class:`~matcal.sierra.models.VFMUniaxialTensionHexModel`
# for this verification. All material constants that are 
# not calibration parameters are passed via 
# ``add_constants``. We use under-integrated elements 
# and the uncoupled (isothermal) formulation to keep 
# the simulation inexpensive.

mat = Material("matcal_test",
               os.path.join(gold_files_dir, material_filename),
               "j2_plasticity")

vfm_model = VFMUniaxialTensionHexModel(mat, mesh_skeleton, thickness)
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
vfm_model.set_number_of_cores(1)
vfm_model.add_boundary_condition_data(field_data)
vfm_model.set_displacement_field_names("U", "V")
vfm_model.use_under_integrated_element()

# %%
# The calibration objective is a 
# :class:`~matcal.full_field.objective.MechanicalVFMObjective`.
# We define the three calibration parameters with search 
# bounds that bracket the true values and initial guesses 
# that are displaced from the goal to give the optimiser 
# a realistic starting point.

vfm_objective = MechanicalVFMObjective()

yield_stress = Parameter("yield_stress", 100e6, 500e6, 200e6)
A = Parameter("A", 1000e6, 5000e6, 4000e6)
b = Parameter("b", 0, 10, 5.0 + 0.001 * np.random.uniform(0, 1))

# %%
# With the parameters, model, objective, and data all 
# set up, we create a 
# :class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`
# and launch the calibration. The study uses a gradient-based
# optimiser with a tight convergence tolerance to drive 
# the residual to near zero.

calibration = GradientCalibrationStudy(yield_stress, A, b)
calibration.add_evaluation_set(vfm_model, vfm_objective, field_data)
calibration.set_core_limit(32)
calibration.set_convergence_tolerance(1e-12)

results = calibration.launch()

# %%
# After the calibration completes, we compare the 
# recovered parameter values to the known goal values.
# We report the absolute and relative errors for each 
# parameter. A successful verification will show errors 
# well below 0.01%.

recovered_yield_stress = results.outcome["best:yield_stress"]
recovered_A = results.outcome["best:A"]
recovered_b = results.outcome["best:b"]


def relative_error(recovered: float, goal: float) -> float:
    """Return the relative error between a recovered and goal value."""
    return abs(recovered - goal) / abs(goal)


print(f"yield_stress:  goal = {yield_stress_goal:.4e} Pa, "
      f"recovered = {recovered_yield_stress:.4e} Pa, "
      f"relative error = {relative_error(recovered_yield_stress, yield_stress_goal):.2e}")

print(f"A:             goal = {A_goal:.4e} Pa, "
      f"recovered = {recovered_A:.4e} Pa, "
      f"relative error = {relative_error(recovered_A, A_goal):.2e}")

print(f"b:             goal = {b_goal:.4e}, "
      f"recovered = {recovered_b:.4e}, "
      f"relative error = {relative_error(recovered_b, b_goal):.2e}")

# %%
# We also plot the calibration convergence history to 
# show how quickly the gradient method converges to the 
# correct parameter values. The normalised parameter 
# values are plotted so that all three parameters can be 
# compared on the same axis.

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
ax.set_title("VFM Calibration Convergence — Rectangular Specimen")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# The calibrated parameters recover the goal values to 
# within the expected tolerance, confirming that MatCal's 
# VFM tools correctly implement the virtual fields method 
# for plastic material calibration on a rectangular specimen.
# The :ref:`VFM Complex Shape Plastic Calibration Verification`
# example extends this verification to a non-rectangular 
# specimen geometry.
