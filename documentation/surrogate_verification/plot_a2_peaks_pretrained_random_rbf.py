r"""
Paper Peaks Verification: Pretrained Random-Sampling RBF Surrogate
==================================================================

This example verifies a pretrained radial basis function (RBF) surrogate on the
2D Peaks benchmark function used by :cite:`voronoi_adaptive_surrogates`. This example is 
part of the verification example set for surrogates. This set of examples is
meant to demonstrate features and behavior of the surrogates in MatCal. 
They are not meant to be used as templates for user MatCal files.

This is a **pretrained** surrogate example:

* all training samples are chosen before surrogate construction;
* samples are independent uniform random samples over the Peaks domain;
* a separate RBF surrogate is trained at each requested sample count;
* all surrogates are scored on the same fixed validation set.

The example produces two figures.

1. A convergence plot of validation RMSE and validation maximum absolute error
   versus number of training samples. This is the Sphinx-Gallery thumbnail.
2. Diagnostic plots at 50, 100, and 150 samples showing the true Peaks function
   and the absolute surrogate error field with training samples overlaid.

The empirical convergence rate is estimated by fitting

.. math::

    E(N) \approx C N^{-p}.
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

MAX_TRAINING_SAMPLES = 150
CONVERGENCE_PLOT_SAMPLE_LIMIT = MAX_TRAINING_SAMPLES
SAMPLE_COUNTS_TO_PLOT = (50, 100, 150)

WORKING_DIRECTORY = os.path.abspath("paper_peaks_pretrained_random_rbf")
FIGURE_DIRECTORY = os.path.join(WORKING_DIRECTORY, "figures")

SAMPLE_COUNTS = tuple(
    sorted(set(range(20, MAX_TRAINING_SAMPLES + 1, 20)) | set(SAMPLE_COUNTS_TO_PLOT))
)

RBF_SURROGATE_OPTIONS = {
    "regressor_type": "RBF",
    "neighbors": 50,
    "decomp_var": 0.99,
}


###############################################################################
# Training helper
# ---------------

def train_and_score_random_rbf(
    parameters,
    model,
    objective,
    validation_results,
    samples,
    sample_count,
):
    """
    Train and score one pretrained random-sampling RBF surrogate.
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
        regressor_type=RBF_SURROGATE_OPTIONS["regressor_type"],
        neighbors=RBF_SURROGATE_OPTIONS["neighbors"],
    )

    surrogate_generator.set_fields_of_interest(TARGET_FIELD)
    surrogate_generator.set_PCA_details(
        decomp_var=RBF_SURROGATE_OPTIONS["decomp_var"],
    )

    save_name = os.path.join(
        training_directory,
        f"random_rbf_surrogate_{int(sample_count)}",
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
#
# The helper functions create:
#
# * two bounded parameters, ``x1`` and ``x2``;
# * a MatCal ``PythonModel`` wrapping the Peaks function;
# * a ``SimulationResultsSynchronizer`` for the scalar one-point response.

parameters = make_parameters()
model = make_model()
objective = make_objective()


###############################################################################
# Generate one fixed validation set.
# ----------------------------------
#
# Every surrogate is scored on the same validation set so that differences in
# error are due to training sample count and not due to changing test data.

print("Generating fixed validation set.")
validation_results = run_fixed_validation_set(
    parameters,
    model,
    objective,
    os.path.join("fixed_validation_set"),
)


###############################################################################
# Generate nested random training samples.
# ----------------------------------------
#
# The 100-sample surrogate uses the first 100 samples of the same random sample
# sequence used by the 150-sample surrogate.

print(f"Generating {MAX_TRAINING_SAMPLES} nested random training samples.")
random_samples = make_uniform_random_samples(
    MAX_TRAINING_SAMPLES,
    seed=RANDOM_SEED,
)


###############################################################################
# Train and score the pretrained RBF surrogates.
# ----------------------------------------------
#
# ``surrogates_by_count`` and ``training_samples_by_count`` are also populated
# so the 50, 100, and 150 sample diagnostic error-field plots can be made later.

rmse_history = []
max_error_history = []

surrogates_by_count = {}
training_samples_by_count = {}

for sample_count in SAMPLE_COUNTS:
    print(f"\nTraining pretrained random-sampling RBF with {sample_count} samples.")

    surrogate, rmse, max_error = train_and_score_random_rbf(
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
    "pretrained random-sampling RBF",
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
    method_name="pretrained random-sampling RBF",
    figure_directory=FIGURE_DIRECTORY,
    filename="peaks_pretrained_random_rbf_convergence.png",
)


###############################################################################
# Plot true function and surrogate error fields.
# ----------------------------------------------
#
# The first row shows the true Peaks function with samples overlaid. The second
# row shows the absolute surrogate error field for the corresponding surrogate.

plot_function_and_surrogate_error_at_counts(
    surrogates_by_count,
    training_samples_by_count,
    sample_counts_to_plot=SAMPLE_COUNTS_TO_PLOT,
    figure_directory=FIGURE_DIRECTORY,
    filename="peaks_pretrained_random_rbf_error_fields.png",
    method_name="pretrained random-sampling RBF",
    n_grid=150,
)

print(f"\nFigures saved to: {FIGURE_DIRECTORY}")
plt.show()