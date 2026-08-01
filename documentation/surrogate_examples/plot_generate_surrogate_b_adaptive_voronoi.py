"""
Voronoi Adaptive Surrogate Example
==================================

This example demonstrates how to generate a surrogate
using a MatCal study that performs adaptive sampling
for training the surrogate.
This study is a follow-on example to 
:ref:`Surrogate Generation Example`
and uses the same boundary value problem that from example.
The primary difference is that a Matcal adaptive surrogate 
study is used for surrogate training. 
In this example we use a :class:`~matcal.core.adaptive_surrogates.VoronoiAdaptiveSurrogateStudy`
to create the surrogate. 

We re-create the model and parameters from 
:ref:`Surrogate Generation Example`
that are needed to perform the study.
"""
# sphinx_gallery_thumbnail_number = 6
import matcal as mc
import numpy as np

conv_heat_transfer_coeff = mc.Parameter("H", 1, 100) # W / (m^2 K)
far_field_temperature = mc.Parameter("T_inf", 500, 1000) # K
air_temperature = mc.Parameter("T_air", 400, 800) # K

my_hifi_model = mc.UserDefinedSierraModel('aria', "aria_model/metal_foam_layers.i", 
                                          "aria_model/test_block.g", "aria_model/include")
my_hifi_model.set_results_filename("results/results.csv")
my_hifi_model.set_number_of_cores(2)
from site_matcal.sandia.tests.utilities import MATCAL_WCID
from site_matcal.sandia.computing_platforms import is_sandia_cluster

if is_sandia_cluster():
    my_hifi_model.run_in_queue(MATCAL_WCID, 0.25)
    my_hifi_model.continue_when_simulation_fails()
    my_hifi_model.set_number_of_cores(12)

#
# %%
# We set some common values that will be used 
# across the surrogate examples as part of this
# example set. This is to ensure valid comparisons
# are made between different methods.
COMMON_TEST_SAMPLE_COUNT = 250
COMMON_TEST_SEED = 12345
TRAINING_SEED = 54321

#%%
# These are demonstration/validation points used for FE-vs-surrogate plots.
# They are not inserted into the common Halton test set used for scoring.
VALIDATION_PARAMETER_SETS = [
    {"H": 10, "T_inf": 600, "T_air": 500},
    {"H": 20, "T_inf": 815, "T_air": 634},
]

#%%
# With the model and parameters created, 
# we must still define the independent variable
# for the surrogate and the values 
# at which we want the surrogate to produce a response.
# However, we do not create the objective. 
# This will automatically happen inside the study.
# This is done because only one response can be 
# used to build the surrogate for because the 
# adaptive training technique uses the sensitivity 
# of the response to the input parameters to 
# adaptively choose where to add training samples.
n_prediction_points = 200
time_start = 0
time_end = 60 * 60 * 2
indep_field_vals = np.linspace(time_start, time_end, n_prediction_points)

#%%
# We can now create the study. As stated previously, 
# only one response can be reproduced with an adaptive surrogate study.
# As a result, the study requires the specification of the 
# independent field, the values for the independent field, 
# and the target field for which the surrogate will 
# predict the response. 
# For this study, we choose the bottom thermocouple
# response as the target field because it had the highest 
# error for both of the test cases from the non-adaptive
# surrogate example.
study = mc.VoronoiAdaptiveSurrogateStudy(conv_heat_transfer_coeff, far_field_temperature,
                                        air_temperature)
study.set_independent_variable("time", indep_field_vals)
study.set_target_field_name("TC_bottom")
study.add_evaluation_set(my_hifi_model)
study.set_surrogate_options( n_restarts_optimizer=20, alpha=1e-5, normalize_y=True, decomp_var=0.999)

study.set_cross_validation_options(
    nsplits=10,
    nmax_folds=3,
    nmax_loo=1,
    batch_size=1,
    cv_scale=1.0,
    cv_metric="sum_abs",
    group_kfold=False,
)

#%%
# We must also specify how many samples to run for generating test data. 
# These adaptive surrogates use Halton sampling for test data generation.
study.set_number_of_test_samples(COMMON_TEST_SAMPLE_COUNT)

#%%
# Next we set a stopping criteria. 
# We are hoping to increase the accuracy of the 
# surrogate to be within 1.5 K or less for all time for all test cases. 
# We do so with the 
# :meth:`~matcal.core.adaptive_surrogates.VoronoiAdaptiveSurrogateStudy.set_error_stopping_criteria`
# that sets a stopping criteria based on the test sample error.
study.set_error_stopping_criteria(max_abs_error_goal=1.5)
#%%
# To set stopping criteria on the convergence of test errors 
# after each iteration, alternatively use 
# :meth:`~matcal.core.adaptive_surrogates.VoronoiAdaptiveSurrogateStudy.set_convergence_criteria`.
# Here we set it arbitrarily small, because we are attempting to 
# drive the error down in the surrogate 
# and have a desired max error goal.
study.set_convergence_criteria(eps=1e-12)

# Now we set some basic surrogate training options for the study.
# First, we set the number of training samples for the initial surrogate. 
study.set_number_of_initial_samples(100)
#%%
# Next, we set the max number of training samples to
# the same number of samples that were 
# used for the non-adaptive surrogate.
# In theory, adaptivity should improve the 
# prediction with the same or fewer samples.
study.set_max_training_samples(500)

#%%
# Next, we set the surrogate save options and filename.
# The adaptive surrogate can retain only selected surrogate model objects to
# keep the saved ``.joblib`` file small. Score histories and test data are
# always stored. Here we retain the best surrogate according to the maximum
# test-sample error.
# Different options are available, see
# :meth:`~matcal.core.adaptive_surrogates.VoronoiAdaptiveSurrogateStudy.set_surrogate_storage_options`.
study.set_surrogate_storage_options(
    best_n_surrogates=1,
    score_metric="max_error",
)
study.set_surrogate_save_filename("layered_metal_bc_voronoi_adaptive_surrogate.joblib")

#%%
# Finally, we set the standard study options like seeds, core use, and working
# directory.
if is_sandia_cluster():
    study.set_core_limit(250)
else:
    study.set_core_limit(112)

#%%
# If setting the seed, ensure the test group seed is different than the study
# seed. If not, the training data will include samples from the test data.
study.set_seed(TRAINING_SEED)
study.set_test_group_random_seed(COMMON_TEST_SEED)
study.set_working_directory("voronoi_adaptive_surrogate", remove_existing=True)

#%%
# With our study defined, we run it and wait for it to complete.
study_results = study.launch()

#%%
# We can now access our surrogate using the
# :meth:`~matcal.core.adaptive_surrogates.VoronoiAdaptiveSurrogateStudy.surrogate`
# property.
# The surrogate is a :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate`
# object.
surrogate = study.surrogate

#%%
# The `study_results` variable is a :class:`~matcal.core.study_base.StudyResults`
# object with the training results stored in it.
#
# While the surrogate is being trained, the adaptive surrogate stores the
# testing error histories for each candidate surrogate. The retained surrogate
# object below is the best surrogate according to the storage metric selected
# above.
best_surrogate_index = surrogate.best_surrogate_iteration_index

print("Best retained surrogate iteration:", best_surrogate_index)
print("Stored surrogate score record:")
print(surrogate.stored_surrogate_scores[best_surrogate_index])
print("Best retained surrogate R2 score:\n", surrogate.score(best_surrogate_index))

#%%
# The retained-surrogate test scores and error histories indicate that the
# surrogate can be used to predict the selected response. The adaptive surrogate
# also stores the common test parameters and responses used to score each retained
# candidate surrogate.
#
# Now we use the retained surrogate to make predictions at two validation
# parameter sets. These validation points are used only for the FE-versus-surrogate
# plots and signed-error curves below. They are not manually added to the common
# Halton test set used to score the adaptive surrogate during training.
# The order of the parameters is the same order that they were passed into the
# parameter collection or study, but this can be verified by calling
# :meth:`~matcal.core.surrogates.MatCalMultiModalPCASurrogate.parameter_order`.
# By default, the surrogate will not allow evaluations outside of the parameter
# ranges provided in the adaptive surrogate study used for training.
#
# We evaluate the surrogate and resulting error similar to 
# as was done in the previous non-adaptive surrogate example
# so that we can see if the surrogate has a more accurate prediction.
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

param_study = mc.ParameterStudy(conv_heat_transfer_coeff, far_field_temperature,
                                 air_temperature)
my_objective = mc.SimulationResultsSynchronizer('time', indep_field_vals,
                                                 "TC_top", "TC_bottom")
param_study.add_evaluation_set(my_hifi_model, my_objective)
param_study.set_core_limit(16)
param_study.add_parameter_evaluation(H=H, T_inf=T_inf, T_air=T_air)
param_study.add_parameter_evaluation(H=H2, T_inf=T_inf2, T_air=T_air2)
results = param_study.launch()

#%%
# With both the finite element model results and the surrogate model results
# obtained, we can plot them together for comparison.
# Note that we can only plot the bottom thermocouple because adaptive
# surrogates are specific to a given response.
fe_data1 = results.simulation_history[my_hifi_model.name]["matcal_default_state"][0]
fe_data2 = results.simulation_history[my_hifi_model.name]["matcal_default_state"][1]

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(constrained_layout=True)
plt.plot(prediction['time'], prediction['TC_bottom'][0, :], '.',
         label="bottom prediction 1", color='tab:green')
plt.plot(prediction['time'], prediction['TC_bottom'][1, :], '.',
         label="bottom prediction 2", color='tab:red')

plt.plot(fe_data1['time'], fe_data1['TC_bottom'],
         label="bottom FE results 1", color='lightgreen')
plt.plot(fe_data2['time'], fe_data2['TC_bottom'],
         label="bottom FE results 2", color='orangered')
plt.xlabel("time (s)")
plt.ylabel("temperature (K)")

plt.legend(ncols=2)
plt.title("Multiple Surrogate Predictions")

plt.show()

#%%
# Similarly, we can plot the surrogate model error. First, we interpolate the
# surrogate results to the finite element model times. Next, we calculate and
# plot the signed error for each prediction.
interp_prediction_bot1 = np.interp(
    fe_data1['time'],
    prediction['time'],
    prediction['TC_bottom'][0, :],
)
interp_prediction_bot2 = np.interp(
    fe_data2['time'],
    prediction['time'],
    prediction['TC_bottom'][1, :],
)

plt.figure(constrained_layout=True)
plt.plot(fe_data1['time'], interp_prediction_bot1 - fe_data1['TC_bottom'],
         label="bottom TC error 1", color='tab:green')
plt.plot(fe_data2['time'], interp_prediction_bot2 - fe_data2['TC_bottom'],
         label="bottom TC error 2", color='tab:red')
plt.xlabel("time (s)")
plt.ylabel("temperature error (K)")

plt.legend(ncols=2)
plt.title("Multiple Surrogate Predictions")

plt.show()

#%%
# The Voronoi adaptive surrogate stores the test-error history. This can be
# useful to evaluate convergence rate and to assess if better performance is
# likely with additional training samples.
print("\n=== Voronoi adaptive surrogate common-test summary ===")
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

print("\nVoronoi adaptive full history:")
print("Training sample counts:")
print(surrogate.sample_count_history)

print("Final batch max error:", surrogate.max_error_history[-1])
print("Final batch training samples:", surrogate.sample_count_history[-1])

#%%
# Plot the error metric as a function of the number of model training samples.
fig, ax = surrogate.plot_error_history(
    metrics="max_error",
    error_units="K",
    metric_styles={
        "max_error": {
            "color": "tab:red",
            "linestyle": "None",
            "marker": "o",
        },
    },
    ylabel="max_test_sample_error",
    title="Surrogate error convergence",
)
plt.show()

#%%
# The adaptive surrogate stores the test parameters and responses used to score
# each candidate surrogate during training. We can use the retained best
# surrogate to inspect which test samples were hardest to predict.
#
# :meth:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate.plot_worst_N`
# plots the worst test samples. For each test sample, the left column compares
# the surrogate prediction against the stored test data. The right column shows
# the signed error, surrogate minus test data.
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

surrogate = study.surrogate
#%%
# We can also visualize how the adaptive training samples were placed in the
# parameter space. The color indicates the adaptive batch in which each training
# point was added. Batch 0 corresponds to the initial training design.
parameter_names = ["H", "T_inf", "T_air"]

training_parameters = np.column_stack([
    study_results.parameter_history[name] for name in parameter_names
])

cumulative_sample_counts = np.asarray(surrogate.sample_count_history, dtype=int)
n_training_samples = training_parameters.shape[0]

# Guard against any bookkeeping mismatch by clipping cumulative counts to the
# number of stored training samples.
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

import matplotlib.pyplot as plt
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
ax.set_title("Voronoi adaptive training samples by batch")

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

#%%
# Optional: plot pairwise projections of the same batch-colored training points.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

projection_pairs = [
    (0, 1, "H", r"$T_\infty$"),
    (0, 2, "H", r"$T_\mathrm{air}$"),
    (1, 2, r"$T_\infty$", r"$T_\mathrm{air}$"),
]

for ax, (i, j, xlabel, ylabel) in zip(axes, projection_pairs):
    sc = ax.scatter(
        training_parameters[:, i],
        training_parameters[:, j],
        c=batch_ids,
        cmap=cmap,
        norm=norm,
        s=40,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_title(f"{xlabel} vs. {ylabel}")

colorbar = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    shrink=0.9,
)
colorbar.set_label("adaptive batch")
colorbar.set_ticks(np.arange(n_batches))
colorbar.set_ticklabels(["initial"] + [str(i) for i in range(1, n_batches)])

fig.suptitle("Voronoi adaptive training samples by batch")
plt.show()


#%%
