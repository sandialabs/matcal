r"""
Paper Peaks Verification: Pretrained Random-Sampling Random Forest Surrogate
===========================================================================

This example verifies a pretrained random forest surrogate on the 2D Peaks
benchmark function used by Kaminsky, Wang, and Pant.

This is a **pretrained** surrogate example:

* all training samples are chosen before surrogate construction;
* samples are independent uniform random samples over the Peaks domain;
* a separate random forest surrogate is trained at each requested sample count;
* all surrogates are scored on the same fixed validation set.

Random forests can model nonlinear responses, but they are not smooth
interpolants. Their predictions are assembled from an ensemble of decision
trees, so their behavior can differ substantially from Gaussian process or RBF
surrogates.

The example produces two figures.

1. A convergence plot of validation RMSE and validation maximum absolute error
   versus number of training samples. This is the Sphinx-Gallery thumbnail.
2. Diagnostic plots at 50, 100, and 150 samples showing the true Peaks function
   and the absolute surrogate error field with training samples overlaid.
"""

# The convergence plot is the first figure created by this example.
# sphinx_gallery_thumbnail_number = 1

###############################################################################
# Imports
# -------

import os

import matplotlib.pyplot as plt
import numpy as np

import matcal as mc

from includes.paper_peaks_verification_common import (
    INDEPENDENT_FIELD,
    INDEPENDENT_VALUES,
    RANDOM_SEED,
    TARGET_FIELD,
    get_field_score,
    make_model,
    make_objective,
    make_parameters,
    make_uniform_random_samples,
    plot_convergence_history,
    plot_function_and_surrogate_error_at_counts,
    print_convergence_summary,
    run_fixed_validation_set,
    run_parameter_study_for_samples,
)


###############################################################################
# User options
# ------------

WORKING_DIRECTORY = os.path.abspath("paper_peaks_pretrained_random_random_forest")
FIGURE_DIRECTORY = os.path.join(WORKING_DIRECTORY, "figures")

MAX_TRAINING_SAMPLES = 150
CONVERGENCE_PLOT_SAMPLE_LIMIT = MAX_TRAINING_SAMPLES
SAMPLE_COUNTS_TO_PLOT = (50, 100, 150)

SAMPLE_COUNTS = tuple(
    sorted(set(range(20, MAX_TRAINING_SAMPLES + 1, 20)) | set(SAMPLE_COUNTS_TO_PLOT))
)

RANDOM_FOREST_SURROGATE_OPTIONS = {
    "regressor_type": "Random Forest",
    "n_estimators": 200,
    "random_state": RANDOM_SEED,
    "decomp_var": 0.99,
}


###############################################################################
# Training helper
# ---------------

def train_and_score_random_forest(
    parameters,
    model,
    objective,
    validation_results,
    samples,
    sample_count,
):
    """
    Train and score one pretrained random-sampling random forest surrogate.
    """
    training_samples = samples[:sample_count]

    training_directory = os.path.join(
        WORKING_DIRECTORY,
        "random_training",
        f"n_{int(sample_count)}",
    )

    training_results = run_parameter_study_for_samples(
        parameters,
        model,
        objective,
        training_samples,
        training_directory,
    )

    surrogate_generator = mc.SurrogateGenerator(
        training_results,
        interpolation_field=INDEPENDENT_FIELD,
        interpolation_locations=INDEPENDENT_VALUES,
        training_fraction=1.0,
        test_eval_info=validation_results,
        regressor_type=RANDOM_FOREST_SURROGATE_OPTIONS["regressor_type"],
        n_estimators=RANDOM_FOREST_SURROGATE_OPTIONS["n_estimators"],
        random_state=RANDOM_FOREST_SURROGATE_OPTIONS["random_state"],
    )

    surrogate_generator.set_fields_of_interest(TARGET_FIELD)
    surrogate_generator.set_PCA_details(
        decomp_var=RANDOM_FOREST_SURROGATE_OPTIONS["decomp_var"],
    )

    save_name = os.path.join(
        training_directory,
        f"random_forest_surrogate_{int(sample_count)}",
    )

    surrogate = surrogate_generator.generate(save_name)

    rmse = get_field_score(surrogate.rmse_errors["test"], TARGET_FIELD)
    max_error = get_field_score(surrogate.max_errors["test"], TARGET_FIELD)

    return surrogate, rmse, max_error


###############################################################################
# Create output directories.
# --------------------------

os.makedirs(WORKING_DIRECTORY, exist_ok=True)
os.makedirs(FIGURE_DIRECTORY, exist_ok=True)


###############################################################################
# Create the MatCal model components.
# -----------------------------------

parameters = make_parameters()
model = make_model()
objective = make_objective()


###############################################################################
# Generate or load one fixed validation set.
# ------------------------------------------

print("Generating or loading fixed validation set.")
validation_results = run_fixed_validation_set(
    parameters,
    model,
    objective,
    os.path.join("fixed_validation_set"),
)


###############################################################################
# Generate nested random training samples.
# ----------------------------------------

print(f"Generating {MAX_TRAINING_SAMPLES} nested random training samples.")
random_samples = make_uniform_random_samples(
    MAX_TRAINING_SAMPLES,
    seed=RANDOM_SEED,
)


###############################################################################
# Train and score pretrained random forest surrogates.
# ----------------------------------------------------

rmse_history = []
max_error_history = []

surrogates_by_count = {}
training_samples_by_count = {}

for sample_count in SAMPLE_COUNTS:
    print(
        "\nTraining pretrained random-sampling random forest "
        f"with {sample_count} samples."
    )

    surrogate, rmse, max_error = train_and_score_random_forest(
        parameters,
        model,
        objective,
        validation_results,
        random_samples,
        sample_count,
    )

    rmse_history.append(rmse)
    max_error_history.append(max_error)

    if sample_count in SAMPLE_COUNTS_TO_PLOT:
        surrogates_by_count[int(sample_count)] = {
            "surrogate": surrogate,
            "surrogate_index": None,
        }
        training_samples_by_count[int(sample_count)] = random_samples[:sample_count]

    print(f"  validation RMSE:               {rmse:.6e}")
    print(f"  validation max absolute error: {max_error:.6e}")

sample_counts = np.asarray(SAMPLE_COUNTS, dtype=int)
rmse_history = np.asarray(rmse_history, dtype=float)
max_error_history = np.asarray(max_error_history, dtype=float)


###############################################################################
# Report convergence statistics.
# ------------------------------

print_convergence_summary(
    "pretrained random-sampling random forest",
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
    method_name="pretrained random-sampling random forest",
    figure_directory=FIGURE_DIRECTORY,
    filename="peaks_pretrained_random_random_forest_convergence.png",
    max_sample_count=CONVERGENCE_PLOT_SAMPLE_LIMIT,
)


###############################################################################
# Plot true function and surrogate error fields.
# ----------------------------------------------

plot_function_and_surrogate_error_at_counts(
    surrogates_by_count,
    training_samples_by_count,
    sample_counts_to_plot=SAMPLE_COUNTS_TO_PLOT,
    figure_directory=FIGURE_DIRECTORY,
    filename="peaks_pretrained_random_random_forest_error_fields.png",
    method_name="pretrained random-sampling random forest",
    n_grid=150,
)

print(f"\nFigures saved to: {FIGURE_DIRECTORY}")
plt.show()