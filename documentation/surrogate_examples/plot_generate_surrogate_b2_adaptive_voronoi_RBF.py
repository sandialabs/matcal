"""
Voronoi Adaptive Surrogate Example with RBF Backend
==================================================

This example demonstrates how to generate a Voronoi adaptive surrogate using
MatCal with an RBF backend surrogate regressor.

This study is a follow-on example to
:ref:`Surrogate Generation Example` and uses the same layered foam/metal thermal
boundary-value problem. The primary difference from the Gaussian Process
Voronoi adaptive example is that this example uses a deterministic RBF
interpolator as the backend surrogate regressor.

The RBF backend can be useful when users want a local deterministic interpolator
instead of a Gaussian Process. Because RBF regressors do not 
provide predictive standard deviations, this
example uses deterministic original-response-space cross-validation and
convergence metrics rather than NLPD-based metrics.
"""
# sphinx_gallery_thumbnail_number = 2

import matcal as mc
import numpy as np

# %%
# Define the uncertain boundary-condition parameters.
conv_heat_transfer_coeff = mc.Parameter("H", 1, 100)  # W / (m^2 K)
far_field_temperature = mc.Parameter("T_inf", 500, 1000)  # K
air_temperature = mc.Parameter("T_air", 400, 800)  # K

# %%
# Define the high-fidelity SIERRA/Aria model.
my_hifi_model = mc.UserDefinedSierraModel(
    "aria",
    "aria_model/metal_foam_layers.i",
    "aria_model/test_block.g",
    "aria_model/include",
)
my_hifi_model.set_results_filename("results/results.csv")
my_hifi_model.set_number_of_cores(1)

# %%
# Common settings used across the surrogate examples. Keeping these values the
# same allows quantitative comparison between standard, Gaussian Process
# adaptive, sparse-grid adaptive, and RBF adaptive surrogates.
COMMON_TEST_SAMPLE_COUNT = 250
COMMON_TEST_SEED = 12345
TRAINING_SEED = 54321

# %%
# These validation points are used only for FE-versus-surrogate plots and
# signed-error curves. They are not manually inserted into the common Halton test
# set used for surrogate scoring.
VALIDATION_PARAMETER_SETS = [
    {"H": 10, "T_inf": 600, "T_air": 500},
    {"H": 20, "T_inf": 815, "T_air": 634},
]

# %%
# Define the independent variable and prediction locations for the surrogate.
n_prediction_points = 200
time_start = 0
time_end = 60 * 60 * 2
indep_field_vals = np.linspace(time_start, time_end, n_prediction_points)

# %%
# Create the Voronoi adaptive surrogate study. Adaptive surrogate studies in
# MatCal currently build a surrogate for one response field at a time. We choose
# TC_bottom to match the other adaptive examples.
study = mc.VoronoiAdaptiveSurrogateStudy(
    conv_heat_transfer_coeff,
    far_field_temperature,
    air_temperature,
)
study.set_independent_variable("time", indep_field_vals)
study.set_target_field_name("TC_bottom")
study.add_evaluation_set(my_hifi_model)

# %%
# Configure the backend surrogate to use RBF interpolation. 
# Any SciPy RBFInterpolator kwargs can be passed through to 
# the interpolator with this method.
study.set_surrogate_options(
    regressor_type="RBF",
    decomp_var=0.999,
)

# %%
# Configure cross-validation used by the Voronoi adaptive sampler. The
# ``sum_abs`` metric is a deterministic original-response-space error metric.
# RBF does not provide Gaussian Process predictive standard deviations, so
# metrics such as NLPD are not meaningful. We therefore use deterministic
# cross-validation and convergence metrics below.
study.set_cross_validation_options(
    kfold_splits=10,
    kfold_regions_for_loo=3,
    loo_seed_candidate_count=3,
    batch_size=3,
    cv_metric="sum_abs",
)

# %%
# Use the same deterministic common Halton test set as the other examples.
from pathlib import Path

COMMON_TEST_RESULTS_FILE = Path("common_surrogate_test") / "final_results.joblib"

study.set_number_of_test_samples(COMMON_TEST_SAMPLE_COUNT)

if COMMON_TEST_RESULTS_FILE.exists():
    print(f"Using existing common surrogate test data: {COMMON_TEST_RESULTS_FILE}")
    study.set_test_data(str(COMMON_TEST_RESULTS_FILE))
else:
    print(
        "Common surrogate test data was not found. "
        "The adaptive study will generate its own Halton test set using "
        f"{COMMON_TEST_SAMPLE_COUNT} samples and seed {COMMON_TEST_SEED}."
    )
# %%
# Set stopping criteria. The maximum absolute error goal is evaluated on the
# common test set.
study.set_error_stopping_criteria(max_abs_error_goal=3)

#%%
# To stop when the test-set error stagnates between adaptive batches,
# use
# :meth:`~matcal.core.adaptive_surrogates.VoronoiAdaptiveSurrogateStudy.set_convergence_criteria`.
# For deterministic metrics such as ``"max_error"`` and ``"rmse"``, this
# convergence check uses original-response-space errors on the stored test set.
# Here we set ``eps`` very small because we primarily want the adaptive loop to stop
# based on the requested error goal or maximum training-sample limit.
study.set_convergence_criteria(
    eps=1e-12,
)

# %%
# Use the same initial and maximum sample counts as the Gaussian Process
# Voronoi adaptive example for direct comparison.
study.set_number_of_initial_samples(100)
study.set_max_training_samples(300)

# %%
# Retain the best surrogate object according to maximum absolute test error.
# Histories, test parameters, and test responses are always stored.
study.set_surrogate_storage_options(
    best_n_surrogates=1,
    score_metric="max_error",
)
study.set_surrogate_save_filename(
    "layered_metal_bc_voronoi_rbf_adaptive_surrogate.joblib"
)

# %%
# Set standard study options.
from site_matcal.sandia.computing_platforms import  get_sandia_computing_platform
platform = get_sandia_computing_platform()
cores_per_node = platform.get_processors_per_node()
study.set_core_limit(cores_per_node)

# %%
# Use different seeds for training and testing so the adaptive training samples
# are not the same as the common test samples.
study.set_seed(TRAINING_SEED)
study.set_test_group_random_seed(COMMON_TEST_SEED)
study.set_working_directory(
    "voronoi_rbf_adaptive_surrogate",
    remove_existing=True,
)

# %%
# Launch the adaptive surrogate study.
study_results = study.launch()

# %%
# Access the retained adaptive surrogate.
surrogate = study.surrogate
best_surrogate_index = surrogate.best_surrogate_iteration_index

print("Best retained surrogate iteration:", best_surrogate_index)
print("Stored surrogate score record:")
print(surrogate.stored_surrogate_scores[best_surrogate_index])
print("Best retained surrogate R2 score:\n", surrogate.score(best_surrogate_index))

# %%
# Evaluate the retained surrogate at two validation parameter sets. These
# validation points are independent visual checks and are not part of the common
# test set used for surrogate scoring.
H = VALIDATION_PARAMETER_SETS[0]["H"]
T_inf = VALIDATION_PARAMETER_SETS[0]["T_inf"]
T_air = VALIDATION_PARAMETER_SETS[0]["T_air"]

H2 = VALIDATION_PARAMETER_SETS[1]["H"]
T_inf2 = VALIDATION_PARAMETER_SETS[1]["T_inf"]
T_air2 = VALIDATION_PARAMETER_SETS[1]["T_air"]

prediction = surrogate(
    [[H, T_inf, T_air], [H2, T_inf2, T_air2]],
    surrogate_index="best",
    batch_evaluate=True,
)

# %%
# Run the high-fidelity model at the same two validation parameter sets for
# visual comparison.
param_study = mc.ParameterStudy(
    conv_heat_transfer_coeff,
    far_field_temperature,
    air_temperature,
)
my_objective = mc.SimulationResultsSynchronizer(
    "time",
    indep_field_vals,
    "TC_top",
    "TC_bottom",
)
param_study.add_evaluation_set(my_hifi_model, my_objective)
param_study.set_core_limit(cores_per_node)
param_study.add_parameter_evaluation(H=H, T_inf=T_inf, T_air=T_air)
param_study.add_parameter_evaluation(H=H2, T_inf=T_inf2, T_air=T_air2)
results = param_study.launch()

# %%
# Plot the RBF adaptive surrogate prediction against the high-fidelity model
# results for the bottom thermocouple.
fe_data1 = results.simulation_history[my_hifi_model.name]["matcal_default_state"][0]
fe_data2 = results.simulation_history[my_hifi_model.name]["matcal_default_state"][1]

import matplotlib.pyplot as plt

plt.close("all")
plt.figure(constrained_layout=True)

plt.plot(
    prediction["time"],
    prediction["TC_bottom"][0, :],
    ".",
    label="bottom prediction 1",
    color="tab:green",
)
plt.plot(
    prediction["time"],
    prediction["TC_bottom"][1, :],
    ".",
    label="bottom prediction 2",
    color="tab:red",
)

plt.plot(
    fe_data1["time"],
    fe_data1["TC_bottom"],
    label="bottom FE results 1",
    color="lightgreen",
)
plt.plot(
    fe_data2["time"],
    fe_data2["TC_bottom"],
    label="bottom FE results 2",
    color="orangered",
)

plt.xlabel("time (s)")
plt.ylabel("temperature (K)")
plt.legend(ncols=2)
plt.title("Voronoi Adaptive RBF Surrogate Predictions")
plt.show()

# %%
# Plot signed surrogate error, surrogate prediction minus finite-element result,
# for each validation point.
interp_prediction_bot1 = np.interp(
    fe_data1["time"],
    prediction["time"],
    prediction["TC_bottom"][0, :],
)
interp_prediction_bot2 = np.interp(
    fe_data2["time"],
    prediction["time"],
    prediction["TC_bottom"][1, :],
)

plt.figure(constrained_layout=True)

plt.plot(
    fe_data1["time"],
    interp_prediction_bot1 - fe_data1["TC_bottom"],
    label="bottom TC error 1",
    color="tab:green",
)
plt.plot(
    fe_data2["time"],
    interp_prediction_bot2 - fe_data2["TC_bottom"],
    label="bottom TC error 2",
    color="tab:red",
)

plt.xlabel("time (s)")
plt.ylabel("temperature error (K)")
plt.legend(ncols=2)
plt.title("Voronoi Adaptive RBF Surrogate Signed Error")
plt.show()

# %%
# Print a quantitative common-test-set summary. This mirrors the summaries in
# the other surrogate examples so the methods can be compared directly.
print("\n=== Voronoi adaptive RBF surrogate common-test summary ===")
print(f"Best retained surrogate iteration: {best_surrogate_index}")
print(
    "Training samples for best retained surrogate:",
    surrogate.sample_count_history[best_surrogate_index],
)
print(
    "Common-test RMSE for best retained surrogate:",
    surrogate.rmse_history[best_surrogate_index],
)
print(
    "Common-test maximum absolute error for best retained surrogate:",
    surrogate.max_error_history[best_surrogate_index],
)
print(
    "Common-test R2 for best retained surrogate:",
    surrogate.score(best_surrogate_index),
)

print("Final batch max error:", surrogate.max_error_history[-1])
print("Final batch training samples:", surrogate.sample_count_history[-1])

# %%
# Plot the adaptive error history.
fig, ax = surrogate.plot_error_history(
    metrics=("rmse", "max_error"),
    error_units="K",
    ylabel="common_test_error",
    title="Voronoi Adaptive RBF Surrogate Error Convergence",
)
plt.show()

# %%
# Plot the worst stored common-test samples for the best retained surrogate.
figures, axes_groups, worst_test_indices = surrogate.plot_worst_N(
    N=5,
    surrogate_index="best",
    metric="max_error",
    error_type="signed",
    independent_variable_units="s",
    target_field_units="K",
)
print("Worst test sample indices:", worst_test_indices)
plt.show()

# %%
# Visualize the adaptive training samples in parameter space.
parameter_names = ["H", "T_inf", "T_air"]

training_parameters = np.column_stack([
    study_results.parameter_history[name]
    for name in parameter_names
])

cumulative_sample_counts = np.asarray(surrogate.sample_count_history, dtype=int)
n_training_samples = training_parameters.shape[0]

cumulative_sample_counts = cumulative_sample_counts[
    cumulative_sample_counts <= n_training_samples
]

if cumulative_sample_counts.size == 0 or cumulative_sample_counts[-1] < n_training_samples:
    cumulative_sample_counts = np.append(cumulative_sample_counts, n_training_samples)

batch_ids = np.zeros(n_training_samples, dtype=int)
start_index = 0
for batch_index, stop_index in enumerate(cumulative_sample_counts):
    batch_ids[start_index:stop_index] = batch_index
    start_index = stop_index

from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111, projection="3d")

n_batches = int(batch_ids.max()) + 1
cmap = plt.get_cmap("viridis", n_batches)
bounds = np.arange(-0.5, n_batches + 0.5, 1)
norm = BoundaryNorm(bounds, cmap.N)

scatter = ax.scatter(
    training_parameters[:, 0],
    training_parameters[:, 1],
    training_parameters[:, 2],
    c=batch_ids,
    cmap=cmap,
    norm=norm,
    s=45,
    edgecolor="black",
    linewidth=0.35,
    alpha=0.9,
)

ax.set_xlabel("H")
ax.set_ylabel(r"$T_\infty$")
ax.set_zlabel(r"$T_\mathrm{air}$")
ax.set_title("Voronoi adaptive RBF training samples by batch")

colorbar = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    pad=0.1,
    shrink=0.75,
)
colorbar.set_label("adaptive batch")
colorbar.set_ticks(np.arange(n_batches))
colorbar.set_ticklabels(["initial"] + [str(i) for i in range(1, n_batches)])

plt.show()