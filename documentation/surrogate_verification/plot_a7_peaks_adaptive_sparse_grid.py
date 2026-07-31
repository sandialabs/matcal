r"""
Paper Peaks Verification: Adaptive Sparse-Grid Surrogate
========================================================

This example verifies MatCal's adaptive sparse-grid surrogate workflow on the
2D Peaks benchmark function from :cite:`voronoi_adaptive_surrogates`. This example is 
part of the verification example set for surrogates. This set of examples is
meant to demonstrate features and behavior of the surrogates in MatCal. 
These examples are not meant to be used as templates for user MatCal files.

This example uses :class:`matcal.core.adaptive_surrogates.SparseGridAdaptiveSurrogateStudy`.
The sparse-grid surrogate is built through PyApprox.

The example produces two figures.

1. A convergence plot of validation RMSE and validation maximum absolute error
   versus number of training samples. This is the Sphinx-Gallery thumbnail.
2. Diagnostic plots at 250, 500, and 750 samples showing the true Peaks function
   and the absolute surrogate error field with training samples overlaid.
"""

# The convergence plot is the first figure created by this example.
# sphinx_gallery_thumbnail_number = 1

###############################################################################
# Imports
# -------

import os

import matplotlib.pyplot as plt

import matcal as mc

from includes.paper_peaks_verification_common import (
    INDEPENDENT_FIELD,
    INDEPENDENT_VALUES,
    RANDOM_SEED,
    TARGET_FIELD,
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


try:
    import pyapprox  
    HAS_PYAPPROX = True
except Exception:
    HAS_PYAPPROX = False


###############################################################################
# User options
# ------------

WORKING_DIRECTORY = os.path.abspath("paper_peaks_adaptive_sparse_grid")
FIGURE_DIRECTORY = os.path.join(WORKING_DIRECTORY, "figures")

SAMPLE_COUNTS_TO_PLOT = (250, 500, 750)
MAX_TRAINING_SAMPLES = 750
CONVERGENCE_PLOT_SAMPLE_LIMIT = None

###############################################################################
# Check optional dependency.
# --------------------------
#
# PyApprox is required for the sparse-grid adaptive surrogate.

if not HAS_PYAPPROX:
    raise ImportError(
        "PyApprox is not available. Install PyApprox to run the adaptive "
        "sparse-grid Peaks verification example."
    )


###############################################################################
# Create output directories.
# --------------------------

os.makedirs(WORKING_DIRECTORY, exist_ok=True)
os.makedirs(FIGURE_DIRECTORY, exist_ok=True)


###############################################################################
# Create MatCal parameters and analytic model.
# --------------------------------------------

parameters = make_parameters()
model = make_model()


###############################################################################
# Create the sparse-grid adaptive surrogate study.
# ------------------------------------------------

study = mc.SparseGridAdaptiveSurrogateStudy(*parameters)


###############################################################################
# Configure the scalar response.
# ------------------------------

study.set_independent_variable(INDEPENDENT_FIELD, INDEPENDENT_VALUES)
study.set_target_field_name(TARGET_FIELD)
study.add_evaluation_set(model)


###############################################################################
# Generate and attach a fixed validation set.
# -------------------------------------------

print("Generating fixed validation set.")
validation_results = run_fixed_validation_set(
    parameters,
    model,
    study.results_synchronizer,
    os.path.join("fixed_validation_set"),
)

validation_results = copy_validation_results_with_qoi_alias(
    validation_results,
    model_name=study._get_model_names()[0],
    objective_name=study.results_synchronizer.name,
)

study.set_test_data(validation_results)


###############################################################################
# Configure sparse-grid stopping behavior.
# ----------------------------------------
#
# Sparse-grid adaptive refinement can add samples in batches. 
# The sparse grid surrogate generally requires more samples to 
# converge for this type of problem so we allow 750 samples for training.

study.set_max_training_samples(MAX_TRAINING_SAMPLES)

study.set_error_stopping_criteria(
    rmse_goal=1.0e-300,
    max_abs_error_goal=1.0e-300,
)


###############################################################################
# Configure the sparse-grid basis and refinement limits.
# ------------------------------------------------------
#
# A piecewise quadratic basis is a robust default for the Peaks function because
# the response has localized high-gradient behavior.

study.set_sparse_grid_basis(
    basis_type="piecewise",
    piecewise_degree=2,
)

study.set_sparse_grid_adaptivity_limits(
    max_level=20,
    pnorm=1.0,
)


###############################################################################
# Retain every sparse-grid surrogate for diagnostic plotting.
# -----------------------------------------------------------
#
# The 250, 500, and 750 sample error-field plots require evaluating retained
# sparse-grid surrogates near those sample counts. If the sparse-grid adaptive
# batches do not land exactly on these sample counts, see the note below.

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
        "peaks_adaptive_sparse_grid_surrogate.joblib",
    )
)

study.set_working_directory(
    os.path.join(WORKING_DIRECTORY, "adaptive_sparse_grid_training"),
    remove_existing=True,
)


###############################################################################
# Launch the adaptive sparse-grid study.
# --------------------------------------

print("Running adaptive sparse-grid Peaks verification.")
study_results = study.launch()


###############################################################################
# Extract and report convergence histories.
# -----------------------------------------

sample_counts, rmse_history, max_error_history = adaptive_history_arrays(
    study.surrogate,
)

print_convergence_summary(
    "adaptive sparse grid",
    sample_counts,
    rmse_history,
    max_error_history,
)


###############################################################################
# Plot convergence versus number of training samples.
# ---------------------------------------------------
#
# This is figure 1 and is used as the Sphinx-Gallery thumbnail.

plot_convergence_history(
    sample_counts,
    rmse_history,
    max_error_history,
    method_name="adaptive sparse grid",
    figure_directory=FIGURE_DIRECTORY,
    filename="peaks_adaptive_sparse_grid_convergence.png",
)


###############################################################################
# Collect retained surrogates and samples at selected counts.
# -----------------------------------------------------------
#
# Sparse-grid adaptive batches may not land exactly on 250, 500, and 750 samples.
# The helper below selects the nearest retained sparse-grid surrogate for each
# requested diagnostic sample count.

surrogates_by_count = collect_adaptive_surrogates_at_sample_counts(
    study.surrogate,
    sample_counts_to_collect=(250, 500, 750),
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
    sample_counts_to_plot=(250, 500, 750),
    figure_directory=FIGURE_DIRECTORY,
    filename="peaks_adaptive_sparse_grid_error_fields.png",
    method_name="adaptive sparse grid",
    n_grid=150,
)

print(f"\nFigures saved to: {FIGURE_DIRECTORY}")
plt.show()