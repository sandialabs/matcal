r"""
Compare Surrogate Convergence for a Discontinuous Piecewise Tang Function
=========================================================================

This example compares surrogate convergence for three sampling/surrogate
strategies:

* Halton space-filling sampling with MatCal PCA/Gaussian-process surrogates
* Voronoi adaptive sampling with MatCal PCA/Gaussian-process surrogates
* Sparse-grid adaptive surrogates through PyApprox

The Halton and Voronoi paths use MatCal's PCA/GP surrogate machinery with the
same Gaussian-process settings. The Sparse Grid path uses
:class:`matcal.core.adaptive_surrogates.SparseGridAdaptiveSurrogateStudy`, which
does not use MatCal's PCA/GP surrogate machinery.

Discontinuous piecewise Tang benchmark function
-----------------------------------------------

The benchmark function used here is a modified, piecewise Tang-style function.
For a point :math:`x \in \mathbb{R}^d`, define the polynomial

.. math::

    p(x)
    =
    \frac{1}{2}
    \sum_{i=1}^{d}
    \left(
    x_i^4 - 16 x_i^2 + 5 x_i
    \right).

The response used in this example is

.. math::

    f(x)
    =
    \begin{cases}
    0,
    &
    x_i < 0 \quad \text{for all } i = 1,\ldots,d,
    \\[6pt]
    p(x),
    &
    \text{otherwise}.
    \end{cases}

Equivalently, the function is forced to zero in the all-negative orthant and is
equal to the Tang-style polynomial elsewhere.

This modification introduces discontinuities along portions of the coordinate
planes that bound the all-negative orthant. For example, if all coordinates
except :math:`x_k` are negative, then crossing from :math:`x_k < 0` to
:math:`x_k \ge 0` changes the response from zero to the polynomial value
:math:`p(x)`, which is generally nonzero. As a result, this benchmark is
intentionally challenging for smooth surrogate models such as Gaussian-process
surrogates and sparse-grid polynomial surrogates.

The generated plots include:

* test MSE versus number of training samples;
* adaptive sample-selection time versus number of training samples;
* adaptive training points colored by batch;
* true function and absolute-error fields for the best surrogates, where
  "best" is selected by native-space maximum absolute test error.
"""

# sphinx_gallery_thumbnail_number = 1

import copy
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import matcal as mc

from matcal.core.adaptive_surrogates import _setup_pyapprox_adaptive_sparse_grid_fitter
from matcal.core.serializer_wrapper import matcal_save


try:
    import pyapprox  # noqa: F401
    HAS_PYAPPROX = True
except Exception:
    HAS_PYAPPROX = False


# =============================================================================
# User options
# =============================================================================

N_DIMENSIONS = 2
N_INITIAL_SAMPLES = 10 * N_DIMENSIONS
MAX_TRAINING_SAMPLES = 300
N_TEST_SAMPLES = 500

WORKING_DIRECTORY = os.path.abspath("compare_halton_voronoi_sparse_grid")
FIGURE_DIRECTORY = os.path.join(WORKING_DIRECTORY, "figures")
RESULTS_FILENAME = os.path.join(
    WORKING_DIRECTORY,
    "halton_voronoi_sparse_grid_results.pkl",
)

TARGET_FIELD = "f"
INDEPENDENT_FIELD = "response_location"
INDEPENDENT_VALUES = np.array([0.0])

PARAMETER_NAMES = [f"x{i + 1}" for i in range(N_DIMENSIONS)]
PARAMETER_BOUNDS = [(-5.0, 5.0) for _ in range(N_DIMENSIONS)]

# MatCal GP surrogate settings used for both Halton and Voronoi paths.
GP_SURROGATE_OPTIONS = {
    "regressor_type": "Gaussian Process",
    "n_restarts_optimizer": 20,
    "alpha": 1.0e-5,
    "normalize_y": True,
    "decomp_var": 4,
}

# Voronoi adaptive sampling settings.
VORONOI_CV_OPTIONS = {
    "nsplits": 5,
    "nmax_folds": 3,
    "nmax_loo": 5,
    "cv_metric": "sum_abs",
    "group_kfold": False,
}

VORONOI_SAMPLING_OPTIONS = {
    "voronoi_type": "full",
    "finite_only": False,
    "iterative_updates": True,
}

# Sparse-grid settings. These are PyApprox sparse-grid settings, not GP settings.
SPARSE_GRID_BASIS_OPTIONS = {
    "basis_type": "piecewise",
    "piecewise_degree": 2,
}

SPARSE_GRID_ADAPTIVITY_OPTIONS = {
    "max_level": 20,
    "pnorm": 1.0,
}


# =============================================================================
# Analytic model
# =============================================================================

def tang_function(samples):
    """
    Tang-style nonlinear benchmark function.

    This is adapted from the function used in ``compare_voronoi_batch_sampling.py``.
    It is nonlinear and has a flat region in the all-negative quadrant, making it
    useful for exercising adaptive sampling.
    """
    samples = np.asarray(samples, dtype=float)
    samples = np.atleast_2d(samples)

    values = np.zeros(samples.shape[0])
    for dim_index in range(samples.shape[1]):
        x = samples[:, dim_index]
        values += x**4 - 16.0 * x**2 + 5.0 * x

    all_negative = np.all(samples < 0.0, axis=1)
    values[all_negative] = 0.0

    return 0.5 * values


def analytic_python_model(**parameters):
    """
    MatCal PythonModel wrapper around :func:`tang_function`.

    The adaptive surrogate interface expects a response curve associated with an
    independent variable. For this scalar analytic function, we return a
    one-point response curve.
    """
    parameter_vector = np.array(
        [parameters[name] for name in PARAMETER_NAMES],
        dtype=float,
    )

    value = tang_function(parameter_vector)[0]

    return {
        INDEPENDENT_FIELD: INDEPENDENT_VALUES.copy(),
        TARGET_FIELD: np.array([value], dtype=float),
    }


# =============================================================================
# Timed adaptive study subclasses
# =============================================================================

class TimedVoronoiAdaptiveSurrogateStudy(mc.VoronoiAdaptiveSurrogateStudy):
    """
    Voronoi adaptive surrogate study that records adaptive sample-selection time.

    The recorded timing is the time spent inside the Voronoi/KFCV/LOO
    sample-selection step. It does not include model-evaluation time or surrogate
    training time.
    """

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self.adaptive_sample_selection_times = []

    def _create_voronoi_tess_and_choose_new_samples(
        self,
        iteration,
        training_params,
        training_data,
    ):
        start_time = time.perf_counter()
        new_points = super()._create_voronoi_tess_and_choose_new_samples(
            iteration,
            training_params,
            training_data,
        )
        elapsed_time = time.perf_counter() - start_time
        self.adaptive_sample_selection_times.append(elapsed_time)
        return new_points


class TimedSparseGridAdaptiveSurrogateStudy(mc.SparseGridAdaptiveSurrogateStudy):
    """
    Sparse-grid adaptive surrogate study that records sample-selection time.

    The recorded timing is only for ``fitter.step_samples()``. It does not
    include model-evaluation time, ``fitter.step_values()``, surrogate scoring,
    or file I/O.
    """

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self.adaptive_sample_selection_times = []

    def _perform_adaptive_surrogate_batch_sampling(self):
        n_qois = len(self._independent_variable_values)

        if self._surrogate_save_filename is None:
            self.set_surrogate_save_filename(
                f"{self._get_model_names()[0]}_sparse_grid_surrogate.joblib"
            )

        fitter = _setup_pyapprox_adaptive_sparse_grid_fitter(
            self._number_parameters,
            n_qois,
            bounds=self._bounds,
            basis_type=self._sg_basis_type,
            piecewise_degree=self._sg_piecewise_degree,
            max_level=self._sg_max_level,
            pnorm=self._sg_pnorm,
        )

        while True:
            start_time = time.perf_counter()
            new_samples = fitter.step_samples()
            elapsed_time = time.perf_counter() - start_time
            self.adaptive_sample_selection_times.append(elapsed_time)

            if new_samples is None:
                print("No more admissible sparse-grid indices. Stopping.")
                break

            new_vals = self._matcal_evaluate_parameter_sets_batch_adaptive_training(
                new_samples
            )

            if new_vals.ndim != 2 or new_vals.shape[1] != n_qois:
                raise RuntimeError(
                    "Batch evaluation must return array with shape "
                    f"(nsamples, n_qois). Got {new_vals.shape}."
                )

            fitter.step_values(new_vals.T)

            result = fitter.result()
            self._surrogate._add_iteration(
                result,
                self._results.number_of_evaluations,
            )

            matcal_save(self._surrogate_save_filename, self._surrogate)

            training_batch_number = len(self._surrogate.sample_count_history)
            if self._stopping_criterion_met(training_batch_number):
                break

        return self._results


# =============================================================================
# Helper functions
# =============================================================================

def make_parameters():
    """
    Create the MatCal parameters used by all studies.
    """
    parameters = []
    for name, bounds in zip(PARAMETER_NAMES, PARAMETER_BOUNDS):
        lower, upper = bounds
        parameters.append(mc.Parameter(name, lower, upper))
    return parameters


def make_model():
    """
    Create the MatCal PythonModel used by all studies.
    """
    model = mc.PythonModel(analytic_python_model)
    model.set_name("tang_analytic_model")
    return model


def make_objective():
    """
    Create the response synchronizer for the scalar analytic response.
    """
    return mc.SimulationResultsSynchronizer(
        INDEPENDENT_FIELD,
        INDEPENDENT_VALUES,
        TARGET_FIELD,
    )


def get_field_score(score_dict, field_name=TARGET_FIELD):
    """
    Extract a scalar field score from a MatCal surrogate score dictionary.
    """
    value = score_dict[field_name]
    return float(np.asarray(value).squeeze())


def copy_test_results_with_qoi_alias(test_results, model_name, objective_name):
    """
    Return a copy of ``test_results`` whose ``qoi_history`` contains an alias
    matching the adaptive study's internally generated objective name.

    Adaptive surrogate studies create their own
    ``SimulationResultsSynchronizer`` inside ``add_evaluation_set``. If an
    external test set was generated with a different synchronizer instance, the
    QoI-history key will not match the adaptive study's synchronizer name.
    """
    patched_results = copy.deepcopy(test_results)

    expected_key = f"{model_name}:{objective_name}"
    if expected_key in patched_results.qoi_history:
        return patched_results

    model_prefix = f"{model_name}:"
    candidate_keys = [
        key for key in patched_results.qoi_history.keys()
        if key.startswith(model_prefix)
    ]

    if len(candidate_keys) == 0:
        raise KeyError(
            "Could not find a QoI-history entry in the supplied test results "
            f"for model '{model_name}'. Available QoI-history keys are "
            f"{list(patched_results.qoi_history.keys())}."
        )

    if len(candidate_keys) > 1:
        raise KeyError(
            "Found multiple QoI-history entries for model "
            f"'{model_name}', so the test-data alias is ambiguous. "
            f"Candidate keys are {candidate_keys}. Expected key is "
            f"'{expected_key}'."
        )

    source_key = candidate_keys[0]
    patched_results.qoi_history[expected_key] = patched_results.qoi_history[source_key]

    print(
        "Aliased fixed-test QoI history:\n"
        f"  source:   {source_key}\n"
        f"  expected: {expected_key}"
    )

    return patched_results


def run_fixed_test_study(parameters, model, objective):
    """
    Generate the fixed test set used by all comparisons.
    """
    test_study = mc.HaltonStudy(*parameters)
    test_study.add_evaluation_set(model, objective)
    test_study.set_number_of_samples(N_TEST_SAMPLES)
    test_study.set_seed(12345)
    test_study.set_working_directory(
        os.path.join(WORKING_DIRECTORY, "fixed_test_set"),
        remove_existing=True,
    )
    return test_study.launch()


def train_halton_surrogate(
    parameters,
    model,
    objective,
    test_results,
    nsamples,
    seed,
    working_directory,
):
    """
    Generate ``nsamples`` Halton training points, train a MatCal GP surrogate,
    and return the surrogate plus its native-space test MSE and max error.
    """
    start_time = time.perf_counter()

    working_directory = os.path.abspath(working_directory)
    os.makedirs(working_directory, exist_ok=True)

    halton_study = mc.HaltonStudy(*parameters)
    halton_study.add_evaluation_set(model, objective)
    halton_study.set_number_of_samples(int(nsamples))
    halton_study.set_seed(seed)
    halton_study.set_working_directory(working_directory, remove_existing=True)

    halton_results = halton_study.launch()

    surrogate_generator = mc.SurrogateGenerator(
        halton_results,
        interpolation_field=INDEPENDENT_FIELD,
        interpolation_locations=INDEPENDENT_VALUES,
        training_fraction=1.0,
        test_eval_info=test_results,
        regressor_type=GP_SURROGATE_OPTIONS["regressor_type"],
        n_restarts_optimizer=GP_SURROGATE_OPTIONS["n_restarts_optimizer"],
        alpha=GP_SURROGATE_OPTIONS["alpha"],
        normalize_y=GP_SURROGATE_OPTIONS["normalize_y"],
    )
    surrogate_generator.set_fields_of_interest(TARGET_FIELD)
    surrogate_generator.set_PCA_details(
        decomp_var=GP_SURROGATE_OPTIONS["decomp_var"],
    )

    save_name = os.path.join(
        working_directory,
        f"halton_gp_surrogate_{int(nsamples)}",
    )
    surrogate = surrogate_generator.generate(save_name)

    rmse = get_field_score(surrogate.rmse_errors["test"], TARGET_FIELD)
    mse = rmse**2

    # Native-space maximum absolute error. This is used to select the best
    # Halton surrogate.
    max_error = get_field_score(surrogate.max_errors["test"], TARGET_FIELD)

    elapsed_time = time.perf_counter() - start_time

    return surrogate, mse, max_error, elapsed_time


def run_halton_comparison(parameters, model, objective, test_results, sample_counts):
    """
    Train Halton GP surrogates at specified sample counts.

    The best Halton surrogate is selected using the native-space ``max_error``
    values returned here.
    """
    halton_mse = []
    halton_max_errors = []
    halton_times = []
    halton_surrogates = []

    for sample_count in sample_counts:
        print(f"Training Halton GP surrogate with {sample_count} samples.")

        working_directory = os.path.join(
            WORKING_DIRECTORY,
            "halton_training",
            f"n_{int(sample_count)}",
        )

        surrogate, mse, max_error, elapsed_time = train_halton_surrogate(
            parameters,
            model,
            objective,
            test_results,
            int(sample_count),
            seed=54321,
            working_directory=working_directory,
        )

        halton_surrogates.append(surrogate)
        halton_mse.append(mse)
        halton_max_errors.append(max_error)
        halton_times.append(elapsed_time)

        print(f"    Halton test MSE:       {mse:.6e}")
        print(f"    Halton test max error: {max_error:.6e}")
        print(f"    Halton total train/score time: {elapsed_time:.3f} s")

    return (
        np.asarray(halton_mse),
        np.asarray(halton_max_errors),
        np.asarray(halton_times),
        halton_surrogates,
    )


def run_voronoi_adaptive_study(parameters, model, objective, test_results):
    """
    Run the Voronoi adaptive surrogate study using the fixed test set.
    """
    study = TimedVoronoiAdaptiveSurrogateStudy(*parameters)

    study.set_independent_variable(INDEPENDENT_FIELD, INDEPENDENT_VALUES)
    study.set_target_field_name(TARGET_FIELD)
    study.add_evaluation_set(model)

    test_results_for_study = copy_test_results_with_qoi_alias(
        test_results,
        model_name=study._get_model_names()[0],
        objective_name=study.results_synchronizer.name,
    )

    study.set_test_data(test_results_for_study)

    # VoronoiAdaptiveSurrogateStudy passes ``_test_eval_info`` into
    # ``_fit_surrogate_model``. When test data are supplied externally, that
    # attribute is not populated by the internal test-sampling path, so set it
    # explicitly for this example.
    study._test_eval_info = test_results_for_study

    study.set_number_of_initial_samples(N_INITIAL_SAMPLES)
    study.set_max_training_samples(MAX_TRAINING_SAMPLES)

    # Make error goals intentionally tight so this example primarily stops due
    # to MAX_TRAINING_SAMPLES rather than early error convergence.
    study.set_error_stopping_criteria(
        rmse_goal=1.0e-14,
        max_abs_error_goal=1.0e-14,
    )

    study.set_convergence_criteria(
        eps=1.0e-14,
        convergence_metric="rmse",
    )

    study.set_cross_validation_options(**VORONOI_CV_OPTIONS)
    study.set_voronoi_sampling_options(**VORONOI_SAMPLING_OPTIONS)

    # These options are passed to MatCal's SurrogateGenerator and GP regressor.
    study.set_surrogate_options(**GP_SURROGATE_OPTIONS)

    study.set_surrogate_storage_options(
        best_n_surrogates=1,
        score_metric="max_error",
    )

    study.set_seed(54321)
    study.set_test_group_random_seed(12345)

    study.set_surrogate_save_filename(
        os.path.join(WORKING_DIRECTORY, "voronoi_adaptive_gp_surrogate.joblib")
    )
    study.set_working_directory(
        os.path.join(WORKING_DIRECTORY, "voronoi_adaptive_training"),
        remove_existing=True,
    )

    study_results = study.launch()
    return study, study_results


def run_sparse_grid_adaptive_study(parameters, model, objective, test_results):
    """
    Run the SparseGridAdaptiveSurrogateStudy using the same fixed test set.

    Sparse-grid surrogates do not use the MatCal PCA/GP surrogate options.
    Instead, sparse-grid basis and admissibility options are controlled through
    ``set_sparse_grid_basis`` and ``set_sparse_grid_adaptivity_limits``.
    """
    if not HAS_PYAPPROX:
        print("PyApprox is not available. Skipping sparse-grid comparison.")
        return None, None

    study = TimedSparseGridAdaptiveSurrogateStudy(*parameters)

    study.set_independent_variable(INDEPENDENT_FIELD, INDEPENDENT_VALUES)
    study.set_target_field_name(TARGET_FIELD)
    study.add_evaluation_set(model)

    test_results_for_study = copy_test_results_with_qoi_alias(
        test_results,
        model_name=study._get_model_names()[0],
        objective_name=study.results_synchronizer.name,
    )

    study.set_test_data(test_results_for_study)

    study.set_max_training_samples(MAX_TRAINING_SAMPLES)

    study.set_error_stopping_criteria(
        rmse_goal=1.0e-14,
        max_abs_error_goal=1.0e-14,
    )

    study.set_sparse_grid_basis(**SPARSE_GRID_BASIS_OPTIONS)
    study.set_sparse_grid_adaptivity_limits(**SPARSE_GRID_ADAPTIVITY_OPTIONS)

    study.set_surrogate_storage_options(
        best_n_surrogates=1,
        score_metric="max_error",
    )

    study.set_seed(54321)
    study.set_test_group_random_seed(12345)

    study.set_surrogate_save_filename(
        os.path.join(WORKING_DIRECTORY, "sparse_grid_adaptive_surrogate.joblib")
    )

    study.set_working_directory(
        os.path.join(WORKING_DIRECTORY, "sparse_grid_adaptive_training"),
        remove_existing=True,
    )

    study_results = study.launch()
    return study, study_results


def plot_convergence_results(
    voronoi_sample_counts,
    voronoi_mse,
    halton_sample_counts,
    halton_mse,
    voronoi_selection_times,
    output_directory,
    sparse_grid_sample_counts=None,
    sparse_grid_mse=None,
    sparse_grid_selection_times=None,
    voronoi_max_errors=None,
    halton_max_errors=None,
    sparse_grid_max_errors=None,
):
    """
    Make convergence and adaptive-sampling-cost comparison plots.

    The figure contains three subplots:

    1. test MSE versus number of training samples;
    2. native-space maximum absolute test error versus number of training samples;
    3. adaptive sample-selection time versus number of training samples.

    Halton has no adaptive sample-selection step, so only Voronoi and, when
    available, Sparse Grid are shown on the timing plot.
    """
    os.makedirs(output_directory, exist_ok=True)

    voronoi_sample_counts = np.asarray(voronoi_sample_counts, dtype=int)
    voronoi_mse = np.asarray(voronoi_mse, dtype=float)

    halton_sample_counts = np.asarray(halton_sample_counts, dtype=int)
    halton_mse = np.asarray(halton_mse, dtype=float)

    voronoi_selection_times = np.asarray(voronoi_selection_times, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), constrained_layout=True)

    # -------------------------------------------------------------------------
    # MSE versus number of training samples
    # -------------------------------------------------------------------------
    axes[0].plot(
        voronoi_sample_counts,
        voronoi_mse,
        color="tab:blue",
        linestyle="-",
        marker="o",
        linewidth=2,
        label="Voronoi adaptive GP",
    )

    axes[0].plot(
        halton_sample_counts,
        halton_mse,
        color="tab:green",
        linestyle="--",
        marker="s",
        linewidth=2,
        label="Halton space-filling GP",
    )

    if sparse_grid_sample_counts is not None and sparse_grid_mse is not None:
        sparse_grid_sample_counts = np.asarray(sparse_grid_sample_counts, dtype=int)
        sparse_grid_mse = np.asarray(sparse_grid_mse, dtype=float)

        axes[0].plot(
            sparse_grid_sample_counts,
            sparse_grid_mse,
            color="tab:orange",
            linestyle="-.",
            marker="^",
            linewidth=2,
            label="Sparse-grid adaptive",
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of training samples")
    axes[0].set_ylabel("test MSE")
    axes[0].set_title("Discontinuous function MSE convergence")
    axes[0].grid(True, which="both", alpha=0.35)
    axes[0].legend()

    # -------------------------------------------------------------------------
    # Max absolute test error versus number of training samples
    # -------------------------------------------------------------------------
    if voronoi_max_errors is not None:
        voronoi_max_errors = np.asarray(voronoi_max_errors, dtype=float)

        axes[1].plot(
            voronoi_sample_counts,
            voronoi_max_errors,
            color="tab:blue",
            linestyle="-",
            marker="o",
            linewidth=2,
            label="Voronoi adaptive GP",
        )

    if halton_max_errors is not None:
        halton_max_errors = np.asarray(halton_max_errors, dtype=float)

        axes[1].plot(
            halton_sample_counts,
            halton_max_errors,
            color="tab:green",
            linestyle="--",
            marker="s",
            linewidth=2,
            label="Halton space-filling GP",
        )

    if sparse_grid_sample_counts is not None and sparse_grid_max_errors is not None:
        sparse_grid_max_errors = np.asarray(sparse_grid_max_errors, dtype=float)

        axes[1].plot(
            sparse_grid_sample_counts,
            sparse_grid_max_errors,
            color="tab:orange",
            linestyle="-.",
            marker="^",
            linewidth=2,
            label="Sparse-grid adaptive",
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("number of training samples")
    axes[1].set_ylabel("native-space max absolute test error")
    axes[1].set_title("Discontinuous function max-error convergence")
    axes[1].grid(True, which="both", alpha=0.35)
    axes[1].legend()

    # -------------------------------------------------------------------------
    # Adaptive sample-selection time versus number of training samples
    # -------------------------------------------------------------------------
    if voronoi_selection_times.size > 0:
        voronoi_timing_x = voronoi_sample_counts[:voronoi_selection_times.size]

        axes[2].plot(
            voronoi_timing_x,
            voronoi_selection_times,
            color="tab:blue",
            linestyle="-",
            marker="o",
            linewidth=2,
            label="Voronoi sample-selection time",
        )

    if (
        sparse_grid_sample_counts is not None
        and sparse_grid_selection_times is not None
    ):
        sparse_grid_selection_times = np.asarray(
            sparse_grid_selection_times,
            dtype=float,
        )

        n_time = min(
            len(sparse_grid_sample_counts),
            len(sparse_grid_selection_times),
        )

        if n_time > 0:
            axes[2].plot(
                sparse_grid_sample_counts[:n_time],
                sparse_grid_selection_times[:n_time],
                color="tab:orange",
                linestyle="-.",
                marker="^",
                linewidth=2,
                label="Sparse-grid sample-selection time",
            )

    axes[2].set_yscale("log")
    axes[2].set_xlabel("number of training samples before batch")
    axes[2].set_ylabel("sample-selection time (s)")
    axes[2].set_title("Adaptive sample-selection cost")
    axes[2].grid(True, which="both", alpha=0.35)
    axes[2].legend()

    figure_path = os.path.join(
        output_directory,
        "halton_voronoi_sparse_grid_convergence.png",
    )
    fig.savefig(figure_path, dpi=300)

    return fig, axes


def _batch_ids_from_sample_counts(n_training_samples, sample_count_history):
    sample_count_history = np.asarray(sample_count_history, dtype=int)
    sample_count_history = sample_count_history[
        sample_count_history <= n_training_samples
    ]

    if sample_count_history.size == 0 or sample_count_history[-1] < n_training_samples:
        sample_count_history = np.append(sample_count_history, n_training_samples)

    batch_ids = np.zeros(n_training_samples, dtype=int)
    start_index = 0
    for batch_index, stop_index in enumerate(sample_count_history):
        batch_ids[start_index:stop_index] = batch_index
        start_index = stop_index

    return batch_ids


def plot_training_points_by_batch(
    study_results,
    surrogate,
    output_directory,
    method_label,
    filename_prefix,
):
    """
    Plot adaptive training points in parameter space, colored by batch.

    For two-dimensional examples, this produces a 2D scatter plot. For
    three-dimensional examples, this produces a 3D scatter plot. For higher
    dimensions, this produces pairwise projections using seaborn.
    """
    os.makedirs(output_directory, exist_ok=True)

    training_parameters = np.column_stack([
        study_results.parameter_history[name] for name in PARAMETER_NAMES
    ])

    n_training_samples = training_parameters.shape[0]
    batch_ids = _batch_ids_from_sample_counts(
        n_training_samples,
        surrogate.sample_count_history,
    )

    n_batches = int(batch_ids.max()) + 1
    cmap = plt.get_cmap("viridis", n_batches)

    if training_parameters.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

        scatter = ax.scatter(
            training_parameters[:, 0],
            training_parameters[:, 1],
            c=batch_ids,
            cmap=cmap,
            s=50,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
        )

        ax.set_xlabel(PARAMETER_NAMES[0])
        ax.set_ylabel(PARAMETER_NAMES[1])
        ax.set_title(f"{method_label} training samples by batch")
        ax.grid(True, alpha=0.35)

        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("batch")
        colorbar.set_ticks(np.arange(n_batches))
        colorbar.set_ticklabels([str(i) for i in range(n_batches)])

    elif training_parameters.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            training_parameters[:, 0],
            training_parameters[:, 1],
            training_parameters[:, 2],
            c=batch_ids,
            cmap=cmap,
            s=50,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
        )

        ax.set_xlabel(PARAMETER_NAMES[0])
        ax.set_ylabel(PARAMETER_NAMES[1])
        ax.set_zlabel(PARAMETER_NAMES[2])
        ax.set_title(f"{method_label} training samples by batch")

        colorbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.75)
        colorbar.set_label("batch")
        colorbar.set_ticks(np.arange(n_batches))
        colorbar.set_ticklabels([str(i) for i in range(n_batches)])

    else:
        import pandas as pd
        import seaborn as sns

        dataframe = pd.DataFrame(
            training_parameters,
            columns=PARAMETER_NAMES,
        )
        dataframe["batch"] = batch_ids

        grid = sns.pairplot(
            dataframe,
            vars=PARAMETER_NAMES,
            hue="batch",
            corner=True,
            plot_kws={"s": 35, "edgecolor": "black", "linewidth": 0.25},
            diag_kind="hist",
        )
        fig = grid.figure
        fig.suptitle(
            f"{method_label} training samples by batch",
            y=1.02,
        )

    figure_path = os.path.join(
        output_directory,
        f"{filename_prefix}_training_points_by_batch.png",
    )
    fig.savefig(figure_path, dpi=300)

    return fig


def _select_retained_surrogate_for_range_enforcement(surrogate, surrogate_index):
    """
    Return the retained underlying surrogate object, when applicable.

    Voronoi adaptive surrogates store MatCal PCA/GP surrogate objects inside an
    AdaptiveSurrogate container. The container itself does not enforce parameter
    ranges for the underlying PCA/GP surrogate; the retained surrogate object
    does. This helper retrieves that retained object when possible.
    """
    if surrogate_index == "best" and hasattr(surrogate, "best_surrogate"):
        return surrogate.best_surrogate

    if surrogate_index == "latest" and hasattr(surrogate, "current_surrogate"):
        return surrogate.current_surrogate

    if surrogate_index is not None and hasattr(surrogate, "_select_surrogate"):
        try:
            return surrogate._select_surrogate(surrogate_index)
        except Exception:
            return None

    return None


def _set_range_enforcement_if_available(obj, enforce):
    """
    Set parameter-range enforcement if the object supports it.
    """
    if obj is not None and hasattr(obj, "enforce_training_data_parameter_range"):
        obj.enforce_training_data_parameter_range(enforce)


def evaluate_scalar_surrogate_on_points(surrogate, points, surrogate_index=None):
    """
    Evaluate a scalar-response surrogate on a batch of parameter points.

    This helper supports both direct MatCal PCA/GP surrogate objects and
    adaptive surrogate containers. For the final contour diagnostic, parameter
    range enforcement is temporarily disabled because the plotting grid includes
    the exact user-specified parameter bounds, while the trained surrogate's
    internal valid range is usually based on the extrema of sampled points and
    may be slightly inside those bounds.
    """
    points = np.asarray(points, dtype=float)

    kwargs = {"batch_evaluate": True}
    if surrogate_index is not None:
        kwargs["surrogate_index"] = surrogate_index

    retained_surrogate = _select_retained_surrogate_for_range_enforcement(
        surrogate,
        surrogate_index,
    )

    # Temporarily disable range enforcement on both the container and the
    # retained underlying surrogate if either supports it.
    _set_range_enforcement_if_available(surrogate, False)
    _set_range_enforcement_if_available(retained_surrogate, False)

    try:
        prediction = surrogate(points, **kwargs)
    finally:
        # Restore default range enforcement after plotting evaluation.
        _set_range_enforcement_if_available(retained_surrogate, True)
        _set_range_enforcement_if_available(surrogate, True)

    values = np.asarray(prediction[TARGET_FIELD], dtype=float)

    return values.reshape(points.shape[0], -1)[:, 0]


def make_2d_prediction_grid(n_grid=150):
    """
    Create a two-dimensional prediction grid over the parameter bounds.
    """
    bounds = np.asarray(PARAMETER_BOUNDS, dtype=float)

    x_values = np.linspace(bounds[0, 0], bounds[0, 1], n_grid)
    y_values = np.linspace(bounds[1, 0], bounds[1, 1], n_grid)

    xx, yy = np.meshgrid(x_values, y_values)

    points = np.column_stack((
        xx.ravel(),
        yy.ravel(),
    ))

    return xx, yy, points


def plot_best_surrogate_function_and_errors(
    best_surrogates,
    output_directory,
    n_grid=150,
):
    """
    Plot the true function and absolute-error fields for the best surrogates.

    The best surrogates should be selected using native-space maximum absolute
    error. This function assumes the benchmark problem is two-dimensional.
    """
    if N_DIMENSIONS != 2:
        print(
            "Skipping true-function/error contour plot because "
            f"N_DIMENSIONS={N_DIMENSIONS}. This plot is only implemented for 2D."
        )
        return None, None

    os.makedirs(output_directory, exist_ok=True)

    xx, yy, grid_points = make_2d_prediction_grid(n_grid=n_grid)
    true_values = tang_function(grid_points)
    true_grid = true_values.reshape(n_grid, n_grid)

    n_methods = len(best_surrogates)
    n_columns = n_methods + 1

    fig, axes = plt.subplots(
        1,
        n_columns,
        figsize=(5.0 * n_columns, 4.5),
        constrained_layout=True,
    )

    if n_columns == 1:
        axes = np.asarray([axes])

    true_contour = axes[0].contourf(
        xx,
        yy,
        true_grid,
        levels=30,
        cmap="viridis",
    )
    axes[0].set_title("True function")
    axes[0].set_xlabel(PARAMETER_NAMES[0])
    axes[0].set_ylabel(PARAMETER_NAMES[1])
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(true_contour, ax=axes[0], label=TARGET_FIELD)

    for axis, (method_name, metadata) in zip(axes[1:], best_surrogates.items()):
        surrogate = metadata["surrogate"]
        surrogate_index = metadata.get("surrogate_index", None)
        max_error = metadata.get("max_error", np.nan)
        sample_count = metadata.get("sample_count", None)

        predicted_values = evaluate_scalar_surrogate_on_points(
            surrogate,
            grid_points,
            surrogate_index=surrogate_index,
        )

        absolute_error = np.abs(predicted_values - true_values)
        error_grid = absolute_error.reshape(n_grid, n_grid)

        error_contour = axis.contourf(
            xx,
            yy,
            error_grid,
            levels=30,
            cmap="magma",
        )

        error_field_max = float(np.nanmax(absolute_error))

        title = method_name
        if sample_count is not None:
            title += f"\nN={sample_count}"

        if np.isfinite(max_error):
            title += f"\ntest max error={max_error:.3g}"

        title += f"\nfield max error={error_field_max:.3g}"

        axis.set_title(title)
        axis.set_xlabel(PARAMETER_NAMES[0])
        axis.set_ylabel(PARAMETER_NAMES[1])
        axis.grid(True, alpha=0.25)

        fig.colorbar(error_contour, ax=axis, label="absolute error")

    figure_path = os.path.join(
        output_directory,
        "best_surrogate_true_function_and_error_fields.png",
    )
    fig.savefig(figure_path, dpi=300)

    return fig, axes


# =============================================================================
# Run comparison
# =============================================================================

if __name__ == "__main__":
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    os.makedirs(FIGURE_DIRECTORY, exist_ok=True)

    parameters = make_parameters()
    model = make_model()
    objective = make_objective()

    print("Generating fixed test set.")
    test_results = run_fixed_test_study(parameters, model, objective)

    print("Running Voronoi adaptive GP surrogate study.")
    voronoi_study, voronoi_study_results = run_voronoi_adaptive_study(
        parameters,
        model,
        objective,
        test_results,
    )

    voronoi_surrogate = voronoi_study.surrogate

    voronoi_sample_counts = np.asarray(
        voronoi_surrogate.sample_count_history,
        dtype=int,
    )

    voronoi_rmse = np.asarray(
        voronoi_surrogate.rmse_history,
        dtype=float,
    )
    voronoi_mse = voronoi_rmse**2

    voronoi_selection_times = np.asarray(
        voronoi_study.adaptive_sample_selection_times,
        dtype=float,
    )

    print("Running Sparse Grid adaptive surrogate study.")
    sparse_grid_study, sparse_grid_study_results = run_sparse_grid_adaptive_study(
        parameters,
        model,
        objective,
        test_results,
    )

    sparse_grid_sample_counts = None
    sparse_grid_rmse = None
    sparse_grid_mse = None
    sparse_grid_selection_times = None
    sparse_grid_surrogate = None

    if sparse_grid_study is not None:
        sparse_grid_surrogate = sparse_grid_study.surrogate

        sparse_grid_sample_counts = np.asarray(
            sparse_grid_surrogate.sample_count_history,
            dtype=int,
        )

        sparse_grid_rmse = np.asarray(
            sparse_grid_surrogate.rmse_history,
            dtype=float,
        )

        sparse_grid_mse = sparse_grid_rmse**2

        sparse_grid_selection_times = np.asarray(
            sparse_grid_study.adaptive_sample_selection_times,
            dtype=float,
        )

    # Train Halton GP surrogates at all sample counts used by either adaptive
    # method. This gives a space-filling GP baseline over the same sample-count
    # range as both adaptive methods.
    if sparse_grid_sample_counts is not None:
        halton_sample_counts = np.unique(
            np.concatenate((
                voronoi_sample_counts,
                sparse_grid_sample_counts,
            ))
        )
    else:
        halton_sample_counts = voronoi_sample_counts.copy()

    halton_sample_counts = halton_sample_counts[
        halton_sample_counts > 0
    ]

    print("Training Halton GP surrogates at adaptive sample counts.")
    (
        halton_mse,
        halton_max_errors,
        halton_total_times,
        halton_surrogates,
    ) = run_halton_comparison(
        parameters,
        model,
        objective,
        test_results,
        halton_sample_counts,
    )

    results = {
        "halton_sample_counts": halton_sample_counts,
        "halton_mse": halton_mse,
        "halton_max_errors": halton_max_errors,
        "halton_total_train_score_times": halton_total_times,

        "voronoi_sample_counts": voronoi_sample_counts,
        "voronoi_mse": voronoi_mse,
        "voronoi_rmse": voronoi_rmse,
        "voronoi_sample_selection_times": voronoi_selection_times,

        "sparse_grid_sample_counts": sparse_grid_sample_counts,
        "sparse_grid_mse": sparse_grid_mse,
        "sparse_grid_rmse": sparse_grid_rmse,
        "sparse_grid_sample_selection_times": sparse_grid_selection_times,

        "gp_surrogate_options": GP_SURROGATE_OPTIONS,
        "voronoi_cv_options": VORONOI_CV_OPTIONS,
        "voronoi_sampling_options": VORONOI_SAMPLING_OPTIONS,
        "sparse_grid_basis_options": SPARSE_GRID_BASIS_OPTIONS,
        "sparse_grid_adaptivity_options": SPARSE_GRID_ADAPTIVITY_OPTIONS,
    }

    with open(RESULTS_FILENAME, "wb") as results_file:
        pickle.dump(results, results_file)

    plot_convergence_results(
        voronoi_sample_counts,
        voronoi_mse,
        halton_sample_counts,
        halton_mse,
        voronoi_selection_times,
        FIGURE_DIRECTORY,
        sparse_grid_sample_counts=sparse_grid_sample_counts,
        sparse_grid_mse=sparse_grid_mse,
        sparse_grid_selection_times=sparse_grid_selection_times,
        voronoi_max_errors=voronoi_surrogate.max_error_history,
        halton_max_errors=halton_max_errors,
        sparse_grid_max_errors=(
            sparse_grid_surrogate.max_error_history
            if sparse_grid_surrogate is not None
            else None
        ),
    )

    plot_training_points_by_batch(
        voronoi_study_results,
        voronoi_surrogate,
        FIGURE_DIRECTORY,
        method_label="Voronoi adaptive GP",
        filename_prefix="voronoi_adaptive_gp",
    )

    if sparse_grid_study_results is not None and sparse_grid_surrogate is not None:
        plot_training_points_by_batch(
            sparse_grid_study_results,
            sparse_grid_surrogate,
            FIGURE_DIRECTORY,
            method_label="Sparse-grid adaptive",
            filename_prefix="sparse_grid_adaptive",
        )

    # Select best surrogates by native-space maximum absolute error and plot
    # their approximation errors over the 2D parameter space.
    best_surrogates = {}

    best_halton_position = int(np.argmin(halton_max_errors))
    best_halton_surrogate = halton_surrogates[best_halton_position]
    best_halton_sample_count = int(halton_sample_counts[best_halton_position])
    best_halton_max_error = float(halton_max_errors[best_halton_position])

    best_surrogates["Halton GP"] = {
        "surrogate": best_halton_surrogate,
        "surrogate_index": None,
        "max_error": best_halton_max_error,
        "sample_count": best_halton_sample_count,
    }

    best_voronoi_index = voronoi_surrogate.best_surrogate_iteration_index
    best_surrogates["Voronoi adaptive GP"] = {
        "surrogate": voronoi_surrogate,
        "surrogate_index": "best",
        "max_error": float(voronoi_surrogate.max_error_history[best_voronoi_index]),
        "sample_count": int(voronoi_surrogate.sample_count_history[best_voronoi_index]),
    }

    if sparse_grid_surrogate is not None:
        best_sparse_index = sparse_grid_surrogate.best_surrogate_iteration_index
        best_surrogates["Sparse Grid adaptive"] = {
            "surrogate": sparse_grid_surrogate,
            "surrogate_index": "best",
            "max_error": float(sparse_grid_surrogate.max_error_history[best_sparse_index]),
            "sample_count": int(sparse_grid_surrogate.sample_count_history[best_sparse_index]),
        }

    plot_best_surrogate_function_and_errors(
        best_surrogates,
        FIGURE_DIRECTORY,
        n_grid=150,
    )

    print("\nComparison complete.")
    print(f"Results saved to: {RESULTS_FILENAME}")
    print(f"Figures saved to: {FIGURE_DIRECTORY}")

    print("\nFinal errors:")
    print(f"  Voronoi final MSE:     {voronoi_mse[-1]:.6e}")
    print(f"  Halton final MSE:      {halton_mse[-1]:.6e}")

    if sparse_grid_mse is not None:
        print(f"  Sparse Grid final MSE: {sparse_grid_mse[-1]:.6e}")

    print("\nBest selected Halton GP surrogate by native-space max error:")
    print(f"  sample count:    {best_halton_sample_count}")
    print(f"  MSE:             {halton_mse[best_halton_position]:.6e}")
    print(f"  max error:       {best_halton_max_error:.6e}")

    print("\nBest retained Voronoi surrogate by native-space max error:")
    print(f"  iteration index: {best_voronoi_index}")
    print(
        "  sample count:    "
        f"{voronoi_surrogate.sample_count_history[best_voronoi_index]}"
    )
    print(
        "  RMSE:            "
        f"{voronoi_surrogate.rmse_history[best_voronoi_index]:.6e}"
    )
    print(
        "  MSE:             "
        f"{voronoi_surrogate.rmse_history[best_voronoi_index]**2:.6e}"
    )
    print(
        "  max error:       "
        f"{voronoi_surrogate.max_error_history[best_voronoi_index]:.6e}"
    )

    if sparse_grid_surrogate is not None:
        print("\nBest retained Sparse Grid surrogate by native-space max error:")
        print(f"  iteration index: {best_sparse_index}")
        print(
            "  sample count:    "
            f"{sparse_grid_surrogate.sample_count_history[best_sparse_index]}"
        )
        print(
            "  RMSE:            "
            f"{sparse_grid_surrogate.rmse_history[best_sparse_index]:.6e}"
        )
        print(
            "  MSE:             "
            f"{sparse_grid_surrogate.rmse_history[best_sparse_index]**2:.6e}"
        )
        print(
            "  max error:       "
            f"{sparse_grid_surrogate.max_error_history[best_sparse_index]:.6e}"
        )

    plt.show()