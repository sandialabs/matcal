r"""
Tang Verification: Adaptive Voronoi RBF Surrogate
=================================================

This example verifies MatCal's KFCV-Voronoi adaptive surrogate workflow on a
Tang-style discontinuous benchmark function. The function is nonlinear and has a
flat region in the all-negative quadrant.

This is one of the discontinuous-function surrogate verification examples. These
examples are meant to demonstrate features and behavior of the surrogates in
MatCal. They are not meant to be used as templates for user MatCal files.

This example combines:

* K-fold cross-validation Voronoi adaptive sampling;
* a 4 by 4 initial grid;
* MatCal's RBF surrogate machinery.

The example produces two figures.

1. A convergence plot of validation RMSE and validation maximum absolute error
   versus number of training samples. This is the Sphinx-Gallery thumbnail.
2. Diagnostic plots at 50, 100, and 150 samples showing the true function and
   the absolute surrogate error field with training samples overlaid.
"""

# The convergence plot is the first figure created by this example.
# sphinx_gallery_thumbnail_number = 1

###############################################################################
# Imports
# -------

import os

import matplotlib.pyplot as plt

from includes.paper_tang_verification_common import (
    INDEPENDENT_FIELD,
    INDEPENDENT_VALUES,
    RANDOM_SEED,
    TARGET_FIELD,
    PaperTangInitialGridVoronoiStudy,
    adaptive_history_arrays,
    collect_adaptive_surrogates_at_sample_counts,
    collect_training_samples_for_surrogate_count_map,
    copy_validation_results_with_qoi_alias,
    make_model,
    make_parameters,
    plot_convergence_history,
    plot_function_and_surrogate_error_at_counts,
    print_convergence_summary,
    run_fixed_validation_set,
)


###############################################################################
# User options
# ------------

WORKING_DIRECTORY = os.path.abspath("paper_tang_adaptive_voronoi_rbf")
FIGURE_DIRECTORY = os.path.join(WORKING_DIRECTORY, "figures")

MAX_TRAINING_SAMPLES = 150
CONVERGENCE_PLOT_SAMPLE_LIMIT = MAX_TRAINING_SAMPLES
SAMPLE_COUNTS_TO_PLOT = (50, 100, 150)

RBF_SURROGATE_OPTIONS = {
    "regressor_type": "RBF",
    "neighbors": 50,
    "decomp_var": 0.99,
}

KFCV_OPTIONS = {
    "nsplits": 10,
    "nmax_folds": 3,
    "nmax_loo": 1,
    "batch_size": 1,
    "cv_scale": 1.0,
    "cv_metric": "sum_abs",
    "group_kfold": False,
}


###############################################################################
# Create output directories.
# --------------------------

os.makedirs(WORKING_DIRECTORY, exist_ok=True)
os.makedirs(FIGURE_DIRECTORY, exist_ok=True)


###############################################################################
# Create the MatCal parameter set and analytic Tang model.
# --------------------------------------------------------

parameters = make_parameters()
model = make_model()


###############################################################################
# Create the adaptive Voronoi study.
# ----------------------------------

study = PaperTangInitialGridVoronoiStudy(*parameters)


###############################################################################
# Configure the scalar response.
# ------------------------------

study.set_independent_variable(INDEPENDENT_FIELD, INDEPENDENT_VALUES)
study.set_target_field_name(TARGET_FIELD)
study.add_evaluation_set(model)


###############################################################################
# Generate or load and attach a fixed validation set.
# ---------------------------------------------------

print("Generating or loading fixed Tang validation set.")
validation_results = run_fixed_validation_set(
    parameters,
    model,
    study.results_synchronizer,
    os.path.join("fixed_tang_validation_set"),
)

validation_results = copy_validation_results_with_qoi_alias(
    validation_results,
    model_name=study._get_model_names()[0],
    objective_name=study.results_synchronizer.name,
)

study.set_test_data(validation_results)

# The Voronoi training path also passes this object to the internal
# SurrogateGenerator, so it is set explicitly for this verification example.
study._test_eval_info = validation_results


###############################################################################
# Configure adaptive sampling.
# ----------------------------

study.set_cross_validation_options(**KFCV_OPTIONS)

study.set_voronoi_sampling_options(
    voronoi_type="full",
    finite_only=False,
    iterative_updates=True,
    thin=None,
    random_selection=None,
)


###############################################################################
# Configure the internal RBF surrogate.
# -------------------------------------

study.set_surrogate_options(**RBF_SURROGATE_OPTIONS)


###############################################################################
# Configure stopping behavior.
# ----------------------------

study.set_max_training_samples(MAX_TRAINING_SAMPLES - 1)

study.set_error_stopping_criteria(
    rmse_goal=1.0e-300,
    max_abs_error_goal=1.0e-300,
)

study.set_convergence_criteria(
    eps=0.0,
    convergence_metric="rmse",
)


###############################################################################
# Retain every surrogate for diagnostic plotting.
# -----------------------------------------------

study.set_surrogate_storage_options(
    best_n_surrogates=1,
    save_every_n_batches=1,
    score_metric="max_error",
)


###############################################################################
# Configure reproducibility and output files.
# -------------------------------------------

study.set_seed(RANDOM_SEED)
study.set_test_group_random_seed(12345)

study.set_surrogate_save_filename(
    os.path.join(
        WORKING_DIRECTORY,
        "tang_adaptive_voronoi_rbf_surrogate.joblib",
    )
)

study.set_working_directory(
    os.path.join(WORKING_DIRECTORY, "adaptive_training"),
    remove_existing=True,
)


###############################################################################
# Launch the adaptive study.
# --------------------------

print("Running adaptive Voronoi RBF Tang verification.")
study_results = study.launch()


###############################################################################
# Extract and report convergence histories.
# -----------------------------------------

sample_counts, rmse_history, max_error_history = adaptive_history_arrays(
    study.surrogate,
)

print_convergence_summary(
    "adaptive Voronoi RBF Tang",
    sample_counts,
    rmse_history,
    max_error_history,
)


###############################################################################
# Plot convergence versus number of training samples.
# ---------------------------------------------------

plot_convergence_history(
    sample_counts,
    rmse_history,
    max_error_history,
    method_name="adaptive Voronoi RBF",
    figure_directory=FIGURE_DIRECTORY,
    filename="tang_adaptive_voronoi_rbf_convergence.png",
    max_sample_count=CONVERGENCE_PLOT_SAMPLE_LIMIT,
)


###############################################################################
# Collect retained surrogates and training samples at selected counts.
# --------------------------------------------------------------------

surrogates_by_count = collect_adaptive_surrogates_at_sample_counts(
    study.surrogate,
    sample_counts_to_collect=SAMPLE_COUNTS_TO_PLOT,
)

training_samples_by_count = collect_training_samples_for_surrogate_count_map(
    study_results,
    surrogates_by_count,
)


###############################################################################
# Plot true function and surrogate error fields.
# ----------------------------------------------

plot_function_and_surrogate_error_at_counts(
    surrogates_by_count,
    training_samples_by_count,
    sample_counts_to_plot=SAMPLE_COUNTS_TO_PLOT,
    figure_directory=FIGURE_DIRECTORY,
    filename="tang_adaptive_voronoi_rbf_error_fields.png",
    method_name="adaptive Voronoi RBF",
    n_grid=150,
)

print(f"\nFigures saved to: {FIGURE_DIRECTORY}")
plt.show()