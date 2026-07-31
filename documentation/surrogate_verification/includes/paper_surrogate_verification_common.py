"""
Shared utilities for surrogate verification examples.

This module supports both:

* the smooth 2D Peaks benchmark function; and
* the discontinuous Tang-style benchmark function.

The benchmark-specific wrapper modules
``paper_peaks_verification_common.py`` and
``paper_tang_verification_common.py`` can re-export these utilities with
benchmark-specific defaults.
"""
import copy
import os
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import matcal as mc
from matcal.core.study_base import StudyResults


# =============================================================================
# Shared user options
# =============================================================================

TARGET_FIELD = "y"
INDEPENDENT_FIELD = "response_location"
INDEPENDENT_VALUES = np.array([0.0])

N_VALIDATION_SAMPLES = 2500

RANDOM_SEED = 54321
VALIDATION_SEED = 12345

INITIAL_GRID_POINTS_PER_DIMENSION = 4
N_INITIAL_SAMPLES = INITIAL_GRID_POINTS_PER_DIMENSION**2


# =============================================================================
# Benchmark definitions
# =============================================================================

@dataclass(frozen=True)
class BenchmarkSpec:
    """
    Container for benchmark-specific configuration.

    :param name: Long benchmark name used in text.
    :type name: str

    :param short_name: Short lowercase benchmark identifier used in filenames.
    :type short_name: str

    :param title_name: Benchmark name used in figure titles.
    :type title_name: str

    :param model_name: MatCal model name.
    :type model_name: str

    :param parameter_names: Parameter names.
    :type parameter_names: tuple[str, ...]

    :param parameter_bounds: Parameter bounds.
    :type parameter_bounds: tuple[tuple[float, float], ...]

    :param function: Callable benchmark function.
    :type function: Callable
    """
    name: str
    short_name: str
    title_name: str
    model_name: str
    parameter_names: tuple
    parameter_bounds: tuple
    function: Callable


# =============================================================================
# Smooth Peaks benchmark function
# =============================================================================

def peaks_function(samples):
    r"""
    Evaluate the 2D Peaks benchmark function from Kaminsky et al.

    .. math::

        y =
        3(1-x_1)^2 e^{-x_1^2-(x_2+1)^2}
        -10\left(\frac{x_1}{5}-x_1^3-x_2^5\right)e^{-x_1^2-x_2^2}
        -\frac{1}{3} e^{-(x_1+1)^2-x_2^2}.

    :param samples: Sample locations with shape ``(n_samples, 2)``.
    :type samples: numpy.ndarray

    :return: Function values with shape ``(n_samples,)``.
    :rtype: numpy.ndarray
    """
    samples = np.asarray(samples, dtype=float)
    samples = np.atleast_2d(samples)

    x1 = samples[:, 0]
    x2 = samples[:, 1]

    term_1 = 3.0 * (1.0 - x1) ** 2 * np.exp(
        -(x1**2) - (x2 + 1.0) ** 2
    )

    term_2 = -10.0 * (
        x1 / 5.0 - x1**3 - x2**5
    ) * np.exp(-(x1**2) - x2**2)

    term_3 = -(1.0 / 3.0) * np.exp(
        -((x1 + 1.0) ** 2) - x2**2
    )

    return term_1 + term_2 + term_3


def peaks_python_model(**parameters):
    """
    MatCal PythonModel wrapper for the scalar Peaks function.
    """
    sample = np.array(
        [[parameters["x1"], parameters["x2"]]],
        dtype=float,
    )

    value = peaks_function(sample)[0]

    return {
        INDEPENDENT_FIELD: INDEPENDENT_VALUES.copy(),
        TARGET_FIELD: np.array([value], dtype=float),
    }


# =============================================================================
# Discontinuous Tang-style benchmark function
# =============================================================================

def tang_function(samples):
    """
    Tang-style nonlinear benchmark function.

    This is adapted from the function used in
    ``compare_voronoi_batch_sampling.py``. It is nonlinear and has a flat region
    in the all-negative quadrant, making it useful for exercising adaptive
    sampling.

    :param samples: Sample locations with shape ``(n_samples, n_dimensions)``.
    :type samples: numpy.ndarray

    :return: Function values with shape ``(n_samples,)``.
    :rtype: numpy.ndarray
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


def tang_python_model(**parameters):
    """
    MatCal PythonModel wrapper for the scalar Tang-style function.
    """
    sample = np.array(
        [[parameters["x1"], parameters["x2"]]],
        dtype=float,
    )

    value = tang_function(sample)[0]

    return {
        INDEPENDENT_FIELD: INDEPENDENT_VALUES.copy(),
        TARGET_FIELD: np.array([value], dtype=float),
    }


PEAKS_BENCHMARK = BenchmarkSpec(
    name="smooth Peaks benchmark",
    short_name="peaks",
    title_name="Peaks",
    model_name="paper_peaks_function_model",
    parameter_names=("x1", "x2"),
    parameter_bounds=((-5.0, 5.0), (-5.0, 5.0)),
    function=peaks_function,
)

TANG_BENCHMARK = BenchmarkSpec(
    name="discontinuous Tang-style benchmark",
    short_name="tang",
    title_name="Tang",
    model_name="paper_tang_function_model",
    parameter_names=("x1", "x2"),
    parameter_bounds=((-5.0, 5.0), (-5.0, 5.0)),
    function=tang_function,
)


# =============================================================================
# MatCal setup helpers
# =============================================================================

def make_parameters_for_benchmark(benchmark):
    """
    Create MatCal parameters for a benchmark domain.

    :param benchmark: Benchmark configuration.
    :type benchmark: BenchmarkSpec

    :return: List of MatCal parameters.
    :rtype: list
    """
    return [
        mc.Parameter(name, lower, upper)
        for name, (lower, upper) in zip(
            benchmark.parameter_names,
            benchmark.parameter_bounds,
        )
    ]


def make_model_for_benchmark(benchmark):
    """
    Create the MatCal PythonModel for a benchmark.

    :param benchmark: Benchmark configuration.
    :type benchmark: BenchmarkSpec

    :return: MatCal PythonModel.
    :rtype: matcal.core.models.PythonModel
    """
    if benchmark is PEAKS_BENCHMARK:
        model_function = peaks_python_model
    elif benchmark is TANG_BENCHMARK:
        model_function = tang_python_model
    else:
        raise ValueError(
            f"Unsupported benchmark '{benchmark}'. Add a top-level PythonModel "
            "wrapper before using this benchmark."
        )

    model = mc.PythonModel(model_function)
    model.set_name(benchmark.model_name)
    return model


def make_objective():
    """
    Create the response synchronizer used by fixed-sample studies.
    """
    return mc.SimulationResultsSynchronizer(
        INDEPENDENT_FIELD,
        INDEPENDENT_VALUES,
        TARGET_FIELD,
    )


def _cached_final_results_filename(working_directory):
    """
    Return the expected MatCal final-results filename for a study directory.
    """
    return os.path.join(
        os.path.abspath(working_directory),
        "final_results.joblib",
    )


def _load_cached_study_results(results_filename):
    """
    Load cached MatCal StudyResults from disk.

    :param results_filename: Path to ``final_results.joblib``.
    :type results_filename: str

    :return: Loaded StudyResults object.
    :rtype: matcal.core.study_base.StudyResults

    :raises RuntimeError: If the file does not contain a StudyResults object.
    """
    cached_results = mc.matcal_load(results_filename)

    if not isinstance(cached_results, StudyResults):
        raise RuntimeError(
            f"Cached validation results file '{results_filename}' did not "
            "contain a MatCal StudyResults object."
        )

    return cached_results


def copy_validation_results_with_qoi_alias(
    validation_results,
    model_name,
    objective_name,
):
    """
    Return a copy of validation results with a QoI-history alias matching the
    adaptive study's internally generated objective name.

    Cached validation sets can contain a different
    ``SimulationResultsSynchronizer`` name than the current adaptive study
    expects. The simulation history and parameter history are still valid. This
    helper simply adds an additional key to ``qoi_history`` that points to the
    same QoI data under the expected name.

    :param validation_results: Validation StudyResults.
    :type validation_results: matcal.core.study_base.StudyResults

    :param model_name: Name of the model used by the adaptive study.
    :type model_name: str

    :param objective_name: Name of the adaptive study's results synchronizer.
    :type objective_name: str

    :return: Copied and patched validation StudyResults.
    :rtype: matcal.core.study_base.StudyResults
    """
    patched_results = copy.deepcopy(validation_results)

    expected_key = f"{model_name}:{objective_name}"

    if expected_key in patched_results.qoi_history:
        return patched_results

    model_prefix = f"{model_name}:"

    candidate_keys = [
        key
        for key in patched_results.qoi_history.keys()
        if key.startswith(model_prefix)
    ]

    if len(candidate_keys) == 0:
        raise KeyError(
            "Could not find a QoI-history entry in the cached validation "
            f"results for model '{model_name}'. Available QoI-history keys are "
            f"{list(patched_results.qoi_history.keys())}."
        )

    if len(candidate_keys) > 1:
        raise KeyError(
            "Found multiple QoI-history entries for model "
            f"'{model_name}', so the validation-data alias is ambiguous. "
            f"Candidate keys are {candidate_keys}. Expected key is "
            f"'{expected_key}'."
        )

    source_key = candidate_keys[0]
    patched_results.qoi_history[expected_key] = patched_results.qoi_history[source_key]

    print(
        "Aliased fixed-validation QoI history:\n"
        f"  source:   {source_key}\n"
        f"  expected: {expected_key}"
    )

    return patched_results


def run_fixed_validation_set(
    parameters,
    model,
    objective,
    working_directory,
    n_samples=N_VALIDATION_SAMPLES,
    seed=VALIDATION_SEED,
    force_rerun=False,
):
    """
    Generate or load the fixed validation set used for surrogate scoring.

    If ``final_results.joblib`` already exists in ``working_directory``, this
    function loads and returns that cached validation set instead of launching a
    new MatCal study.

    :param parameters: MatCal parameters.
    :type parameters: list

    :param model: MatCal model.
    :type model: object

    :param objective: MatCal synchronizer/objective used to extract the response.
    :type objective: object

    :param working_directory: Working directory for the validation study.
    :type working_directory: str

    :param n_samples: Number of validation samples to generate if no cache exists.
    :type n_samples: int

    :param seed: Validation-set random seed.
    :type seed: int

    :param force_rerun: If ``True``, regenerate the validation set even if a
        cached ``final_results.joblib`` file exists.
    :type force_rerun: bool

    :return: Validation StudyResults.
    :rtype: matcal.core.study_base.StudyResults
    """
    working_directory = os.path.abspath(working_directory)
    cached_results_filename = _cached_final_results_filename(working_directory)

    if os.path.exists(cached_results_filename) and not force_rerun:
        print(
            "Loading cached fixed validation set from:\n"
            f"  {cached_results_filename}"
        )
        return _load_cached_study_results(cached_results_filename)

    print(
        "Cached fixed validation set was not found. Generating a new one in:\n"
        f"  {working_directory}"
    )

    validation_study = mc.HaltonStudy(*parameters)
    validation_study.add_evaluation_set(model, objective)
    validation_study.set_number_of_samples(int(n_samples))
    validation_study.set_seed(seed)
    validation_study.set_working_directory(
        working_directory,
        remove_existing=True,
    )

    return validation_study.launch()


def run_parameter_study_for_samples(
    parameters,
    model,
    objective,
    samples,
    working_directory,
    benchmark,
):
    """
    Evaluate the analytic model at prescribed sample locations.

    :param parameters: MatCal parameters.
    :type parameters: list

    :param model: MatCal model.
    :type model: object

    :param objective: MatCal synchronizer/objective.
    :type objective: object

    :param samples: Prescribed samples.
    :type samples: numpy.ndarray

    :param working_directory: Study working directory.
    :type working_directory: str

    :param benchmark: Benchmark configuration.
    :type benchmark: BenchmarkSpec

    :return: MatCal StudyResults.
    :rtype: matcal.core.study_base.StudyResults
    """
    samples = np.asarray(samples, dtype=float)
    working_directory = os.path.abspath(working_directory)
    os.makedirs(os.path.dirname(working_directory), exist_ok=True)

    parameter_study = mc.ParameterStudy(*parameters)
    parameter_study.add_evaluation_set(model, objective)

    for sample in samples:
        parameter_values = {
            name: float(value)
            for name, value in zip(benchmark.parameter_names, sample)
        }
        parameter_study.add_parameter_evaluation(**parameter_values)

    parameter_study.set_working_directory(
        working_directory,
        remove_existing=True,
    )

    return parameter_study.launch()


def make_uniform_random_samples_for_benchmark(nsamples, seed, benchmark):
    """
    Generate independent uniform random samples over a benchmark domain.

    :param nsamples: Number of samples.
    :type nsamples: int

    :param seed: Random seed.
    :type seed: int

    :param benchmark: Benchmark configuration.
    :type benchmark: BenchmarkSpec

    :return: Random samples.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)

    bounds = np.asarray(benchmark.parameter_bounds, dtype=float)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    unit_samples = rng.random((int(nsamples), len(benchmark.parameter_names)))

    return lower_bounds + unit_samples * (upper_bounds - lower_bounds)


# =============================================================================
# Initial-grid Voronoi adaptive studies
# =============================================================================

class BenchmarkInitialGridVoronoiStudy(mc.VoronoiAdaptiveSurrogateStudy):
    """
    Voronoi adaptive surrogate study with a deterministic tensor-product
    initial grid.

    Subclasses must set the ``benchmark`` class attribute.
    """

    benchmark = None
    initial_grid_points_per_dimension = INITIAL_GRID_POINTS_PER_DIMENSION

    def _make_initial_grid(self):
        if self.benchmark is None:
            raise RuntimeError(
                "BenchmarkInitialGridVoronoiStudy subclasses must set the "
                "'benchmark' class attribute."
            )

        values_by_dimension = [
            np.linspace(lower, upper, self.initial_grid_points_per_dimension)
            for lower, upper in self.benchmark.parameter_bounds
        ]

        mesh = np.meshgrid(*values_by_dimension)

        return np.column_stack([
            dimension_values.ravel()
            for dimension_values in mesh
        ])

    def _run_initial_training_samples(self):
        initial_samples = self._make_initial_grid()

        self._populate_parameter_evaluations(initial_samples)
        self._matcal_evaluate_parameter_sets_batch(
            self._parameter_sets_to_evaluate
        )

        return self._train_surrogate_with_current_results()


class PaperPeaksInitialGridVoronoiStudy(BenchmarkInitialGridVoronoiStudy):
    """
    Voronoi adaptive surrogate study for the Peaks benchmark with a 4 by 4
    initial grid.
    """
    benchmark = PEAKS_BENCHMARK


class PaperTangInitialGridVoronoiStudy(BenchmarkInitialGridVoronoiStudy):
    """
    Voronoi adaptive surrogate study for the Tang-style benchmark with a 4 by 4
    initial grid.
    """
    benchmark = TANG_BENCHMARK


# =============================================================================
# Scoring helpers
# =============================================================================

def get_field_score(score_dict, field_name=TARGET_FIELD):
    """
    Extract a scalar field score from a MatCal score dictionary.
    """
    return float(np.asarray(score_dict[field_name]).squeeze())


def fit_power_law_convergence(sample_counts, errors):
    r"""
    Fit a power-law convergence model.

    The fitted model is

    .. math::

        E(N) \approx C N^{-p},

    where ``E`` is the error, ``N`` is the number of training samples,
    ``C`` is a fitted constant, and ``p`` is the empirical convergence rate.

    :param sample_counts: Training sample counts.
    :type sample_counts: numpy.ndarray

    :param errors: Error values corresponding to ``sample_counts``.
    :type errors: numpy.ndarray

    :return: Tuple ``(C, p)``. Returns ``(nan, nan)`` if the fit cannot be made.
    :rtype: tuple[float, float]
    """
    sample_counts = np.asarray(sample_counts, dtype=float)
    errors = np.asarray(errors, dtype=float)

    mask = (
        np.isfinite(sample_counts)
        & np.isfinite(errors)
        & (sample_counts > 0.0)
        & (errors > 0.0)
    )

    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan

    slope, intercept = np.polyfit(
        np.log(sample_counts[mask]),
        np.log(errors[mask]),
        deg=1,
    )

    p = -float(slope)
    C = float(np.exp(intercept))

    return C, p


def estimate_power_law_convergence_rate(sample_counts, errors):
    r"""
    Estimate the convergence rate ``p`` from

    .. math::

        E(N) \approx C N^{-p}.
    """
    _, p = fit_power_law_convergence(sample_counts, errors)
    return p


def print_convergence_summary(method_name, sample_counts, rmse, max_errors):
    """
    Print final and best errors plus empirical convergence rates.
    """
    sample_counts = np.asarray(sample_counts, dtype=int)
    rmse = np.asarray(rmse, dtype=float)
    max_errors = np.asarray(max_errors, dtype=float)

    rmse_rate = estimate_power_law_convergence_rate(sample_counts, rmse)
    max_error_rate = estimate_power_law_convergence_rate(sample_counts, max_errors)

    best_index = int(np.nanargmin(max_errors))

    print(f"\n{method_name} convergence summary:")
    print(f"  final sample count:         {sample_counts[-1]}")
    print(f"  final RMSE:                 {rmse[-1]:.6e}")
    print(f"  final max absolute error:   {max_errors[-1]:.6e}")
    print(f"  best max absolute error:    {max_errors[best_index]:.6e}")
    print(f"  best max-error sample count:{sample_counts[best_index]}")
    print(f"  estimated RMSE rate p:      {rmse_rate:.4f}")
    print(f"  estimated max-error rate p: {max_error_rate:.4f}")


def adaptive_history_arrays(adaptive_surrogate):
    """
    Extract sample-count, RMSE, and max-error histories from an AdaptiveSurrogate.
    """
    sample_counts = np.asarray(
        adaptive_surrogate.sample_count_history,
        dtype=int,
    )
    rmse = np.asarray(
        adaptive_surrogate.rmse_history,
        dtype=float,
    )
    max_errors = np.asarray(
        adaptive_surrogate.max_error_history,
        dtype=float,
    )

    return sample_counts, rmse, max_errors


def parameter_matrix_from_results_for_benchmark(results, benchmark):
    """
    Extract a parameter matrix from MatCal StudyResults.

    :param results: MatCal study results.
    :type results: matcal.core.study_base.StudyResults

    :param benchmark: Benchmark configuration.
    :type benchmark: BenchmarkSpec

    :return: Parameter matrix with shape ``(n_samples, n_parameters)``.
    :rtype: numpy.ndarray
    """
    return np.column_stack([
        np.asarray(results.parameter_history[name], dtype=float)
        for name in benchmark.parameter_names
    ])


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_convergence_history_for_benchmark(
    sample_counts,
    rmse,
    max_errors,
    method_name,
    figure_directory,
    filename,
    benchmark,
    max_sample_count=None,
):
    """
    Plot RMSE and maximum absolute error versus training sample count.

    The plot also includes empirical power-law convergence fits of the form

    .. math::

        E(N) \approx C N^{-p}.
    """
    os.makedirs(figure_directory, exist_ok=True)

    sample_counts = np.asarray(sample_counts, dtype=int)
    rmse = np.asarray(rmse, dtype=float)
    max_errors = np.asarray(max_errors, dtype=float)

    if max_sample_count is None:
        mask = np.ones(sample_counts.shape, dtype=bool)
    else:
        mask = sample_counts <= int(max_sample_count)

    plot_counts = sample_counts[mask]
    plot_rmse = rmse[mask]
    plot_max_errors = max_errors[mask]

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    ax.semilogy(
        plot_counts,
        plot_rmse,
        color="tab:blue",
        linestyle="-",
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="RMSE",
    )

    ax.semilogy(
        plot_counts,
        plot_max_errors,
        color="tab:red",
        linestyle="--",
        marker="s",
        linewidth=1.8,
        markersize=4,
        label="max absolute error",
    )

    rmse_C, rmse_p = fit_power_law_convergence(plot_counts, plot_rmse)
    max_C, max_p = fit_power_law_convergence(plot_counts, plot_max_errors)

    fit_counts = np.linspace(
        float(np.min(plot_counts)),
        float(np.max(plot_counts)),
        200,
    )

    if np.isfinite(rmse_C) and np.isfinite(rmse_p):
        ax.semilogy(
            fit_counts,
            rmse_C * fit_counts ** (-rmse_p),
            color="tab:blue",
            linestyle=":",
            linewidth=2.0,
            label=rf"RMSE fit, $p={rmse_p:.2f}$",
        )

    if np.isfinite(max_C) and np.isfinite(max_p):
        ax.semilogy(
            fit_counts,
            max_C * fit_counts ** (-max_p),
            color="tab:red",
            linestyle=":",
            linewidth=2.0,
            label=rf"max-error fit, $p={max_p:.2f}$",
        )

    ax.set_xlabel("number of training samples")
    ax.set_ylabel("validation error")
    ax.set_title(f"{benchmark.title_name} verification: {method_name}")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()

    figure_path = os.path.join(figure_directory, filename)
    fig.savefig(figure_path, dpi=300)

    return fig, ax


def _select_retained_surrogate_for_range_enforcement(
    surrogate,
    surrogate_index,
):
    """
    Return a retained underlying surrogate object, when applicable.
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
    Enable or disable parameter-range enforcement if supported by the object.
    """
    if obj is not None and hasattr(obj, "enforce_training_data_parameter_range"):
        obj.enforce_training_data_parameter_range(enforce)


def evaluate_scalar_surrogate_on_points(
    surrogate,
    points,
    surrogate_index=None,
):
    """
    Evaluate a scalar-response surrogate on a batch of parameter points.

    Range enforcement is temporarily disabled because the diagnostic grid
    includes exact parameter bounds, while a trained surrogate's internal
    sampled range may lie slightly inside those bounds.
    """
    points = np.asarray(points, dtype=float)

    kwargs = {"batch_evaluate": True}
    if surrogate_index is not None:
        kwargs["surrogate_index"] = surrogate_index

    retained_surrogate = _select_retained_surrogate_for_range_enforcement(
        surrogate,
        surrogate_index,
    )

    _set_range_enforcement_if_available(surrogate, False)
    _set_range_enforcement_if_available(retained_surrogate, False)

    try:
        prediction = surrogate(points, **kwargs)
    finally:
        _set_range_enforcement_if_available(retained_surrogate, True)
        _set_range_enforcement_if_available(surrogate, True)

    values = np.asarray(prediction[TARGET_FIELD], dtype=float)

    return values.reshape(points.shape[0], -1)[:, 0]


def make_2d_prediction_grid_for_benchmark(n_grid, benchmark):
    """
    Create a two-dimensional prediction grid over a benchmark parameter domain.
    """
    bounds = np.asarray(benchmark.parameter_bounds, dtype=float)

    x_values = np.linspace(bounds[0, 0], bounds[0, 1], n_grid)
    y_values = np.linspace(bounds[1, 0], bounds[1, 1], n_grid)

    xx, yy = np.meshgrid(x_values, y_values)

    grid_points = np.column_stack((
        xx.ravel(),
        yy.ravel(),
    ))

    return xx, yy, grid_points


def _get_surrogate_for_sample_count(
    surrogates_by_count,
    sample_count,
):
    """
    Return the surrogate metadata for a requested sample count.
    """
    if sample_count not in surrogates_by_count:
        available_counts = sorted(surrogates_by_count.keys())
        raise KeyError(
            f"No surrogate is available for sample_count={sample_count}. "
            f"Available sample counts are {available_counts}."
        )

    return surrogates_by_count[sample_count]


def plot_function_and_surrogate_error_at_counts_for_benchmark(
    surrogates_by_count,
    training_samples_by_count,
    sample_counts_to_plot,
    figure_directory,
    filename,
    method_name,
    benchmark,
    n_grid=150,
):
    """
    Plot true benchmark function and absolute surrogate error fields.

    The figure has two rows and one column per requested sample count.

    The first row shows the true benchmark function with the training samples
    used by that surrogate overlaid. The second row shows the absolute surrogate
    error field,

    .. math::

        |f(x) - \hat{f}(x)|,

    with the same training samples overlaid.
    """
    os.makedirs(figure_directory, exist_ok=True)

    xx, yy, grid_points = make_2d_prediction_grid_for_benchmark(
        n_grid=n_grid,
        benchmark=benchmark,
    )

    true_values = benchmark.function(grid_points)
    true_grid = true_values.reshape(n_grid, n_grid)

    n_counts = len(sample_counts_to_plot)

    fig, axes = plt.subplots(
        2,
        n_counts,
        figsize=(5.4 * n_counts, 9.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    if n_counts == 1:
        axes = np.asarray(axes).reshape(2, 1)

    function_contours = []
    error_contours = []

    for column_index, sample_count in enumerate(sample_counts_to_plot):
        requested_sample_count = int(sample_count)

        metadata = _get_surrogate_for_sample_count(
            surrogates_by_count,
            requested_sample_count,
        )

        surrogate = metadata["surrogate"]
        surrogate_index = metadata.get("surrogate_index", None)

        actual_sample_count = int(
            metadata.get("actual_sample_count", requested_sample_count)
        )

        training_samples = np.asarray(
            training_samples_by_count[requested_sample_count],
            dtype=float,
        )

        predicted_values = evaluate_scalar_surrogate_on_points(
            surrogate,
            grid_points,
            surrogate_index=surrogate_index,
        )

        absolute_error = np.abs(predicted_values - true_values)
        error_grid = absolute_error.reshape(n_grid, n_grid)

        # ---------------------------------------------------------------------
        # Top row: true function and samples.
        # ---------------------------------------------------------------------
        function_axis = axes[0, column_index]

        function_contour = function_axis.contourf(
            xx,
            yy,
            true_grid,
            levels=40,
            cmap="coolwarm",
        )
        function_contours.append(function_contour)

        function_axis.scatter(
            training_samples[:, 0],
            training_samples[:, 1],
            s=26,
            c="black",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.9,
        )

        if actual_sample_count == requested_sample_count:
            sample_count_label = f"N={actual_sample_count}"
        else:
            sample_count_label = (
                f"requested N={requested_sample_count}\n"
                f"nearest retained N={actual_sample_count}"
            )

        function_axis.set_title(
            f"True {benchmark.title_name} function\n{sample_count_label}"
        )
        function_axis.set_xlabel(r"$x_1$")
        function_axis.set_ylabel(r"$x_2$")
        function_axis.grid(True, color="black", alpha=0.18, linewidth=0.5)

        # ---------------------------------------------------------------------
        # Bottom row: absolute surrogate error and samples.
        # ---------------------------------------------------------------------
        error_axis = axes[1, column_index]

        error_contour = error_axis.contourf(
            xx,
            yy,
            error_grid,
            levels=40,
            cmap="magma",
        )
        error_contours.append(error_contour)

        error_axis.scatter(
            training_samples[:, 0],
            training_samples[:, 1],
            s=26,
            c="cyan",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.9,
        )

        max_error = float(np.nanmax(absolute_error))

        error_axis.set_title(
            f"Absolute surrogate error\n{sample_count_label}, "
            f"field max={max_error:.3e}"
        )
        error_axis.set_xlabel(r"$x_1$")
        error_axis.set_ylabel(r"$x_2$")
        error_axis.grid(True, color="white", alpha=0.18, linewidth=0.5)

    for axis in axes.ravel():
        axis.set_xlim(benchmark.parameter_bounds[0])
        axis.set_ylim(benchmark.parameter_bounds[1])

    function_colorbar = fig.colorbar(
        function_contours[-1],
        ax=axes[0, :],
        location="right",
        shrink=0.88,
        pad=0.015,
    )
    function_colorbar.set_label(f"{benchmark.title_name} function value")

    error_colorbar = fig.colorbar(
        error_contours[-1],
        ax=axes[1, :],
        location="right",
        shrink=0.88,
        pad=0.015,
    )
    error_colorbar.set_label("absolute surrogate error")

    fig.suptitle(
        f"{benchmark.title_name} verification: {method_name}",
        fontsize=15,
    )

    figure_path = os.path.join(figure_directory, filename)
    fig.savefig(figure_path, dpi=300)

    return fig, axes


# =============================================================================
# Adaptive-surrogate collection helpers
# =============================================================================

def _nearest_adaptive_surrogate_record(
    adaptive_surrogate,
    requested_sample_count,
):
    """
    Find the retained adaptive-surrogate record whose sample count is nearest to
    a requested sample count.

    Ties are resolved by selecting the smaller sample count.
    """
    requested_sample_count = int(requested_sample_count)

    retained_records = [
        record
        for record in adaptive_surrogate.surrogate_records
        if int(record["iteration_index"]) in adaptive_surrogate.stored_surrogates
    ]

    if len(retained_records) == 0:
        raise RuntimeError(
            "No adaptive surrogate records are retained. Use "
            "set_surrogate_storage_options with best_n_surrogates or "
            "save_every_n_batches to retain at least one surrogate."
        )

    nearest_record = min(
        retained_records,
        key=lambda record: (
            abs(int(record["sample_count"]) - requested_sample_count),
            int(record["sample_count"]),
        ),
    )

    return nearest_record


def collect_adaptive_surrogates_at_sample_counts(
    adaptive_surrogate,
    sample_counts_to_collect=(50, 100, 150),
):
    """
    Collect retained adaptive surrogates nearest to requested sample counts.

    This function is intentionally nearest-count based rather than exact-count
    based. Some adaptive methods, especially sparse-grid adaptive refinement, do
    not necessarily generate surrogates at exactly the requested sample counts.
    """
    surrogates_by_count = {}

    for requested_sample_count in sample_counts_to_collect:
        requested_sample_count = int(requested_sample_count)

        record = _nearest_adaptive_surrogate_record(
            adaptive_surrogate,
            requested_sample_count,
        )

        iteration_index = int(record["iteration_index"])
        actual_sample_count = int(record["sample_count"])

        surrogates_by_count[requested_sample_count] = {
            "surrogate": adaptive_surrogate,
            "surrogate_index": iteration_index,
            "requested_sample_count": requested_sample_count,
            "actual_sample_count": actual_sample_count,
        }

        print(
            "Selected retained adaptive surrogate for diagnostic plot: "
            f"requested N={requested_sample_count}, "
            f"actual N={actual_sample_count}, "
            f"iteration={iteration_index}."
        )

    return surrogates_by_count


def collect_training_samples_for_surrogate_count_map_for_benchmark(
    study_results,
    surrogates_by_count,
    benchmark,
):
    """
    Collect training samples that correspond to selected adaptive surrogates.
    """
    all_training_samples = parameter_matrix_from_results_for_benchmark(
        study_results,
        benchmark,
    )

    training_samples_by_count = {}

    for requested_sample_count, metadata in surrogates_by_count.items():
        actual_sample_count = int(metadata["actual_sample_count"])

        if all_training_samples.shape[0] < actual_sample_count:
            raise RuntimeError(
                f"Requested training samples through actual sample count "
                f"{actual_sample_count}, but only "
                f"{all_training_samples.shape[0]} samples are available."
            )

        training_samples_by_count[int(requested_sample_count)] = (
            all_training_samples[:actual_sample_count]
        )

    return training_samples_by_count


def collect_training_samples_at_counts_for_benchmark(
    study_results,
    sample_counts_to_collect,
    benchmark,
):
    """
    Collect training sample arrays from an adaptive study result.
    """
    all_training_samples = parameter_matrix_from_results_for_benchmark(
        study_results,
        benchmark,
    )

    samples_by_count = {}
    for sample_count in sample_counts_to_collect:
        sample_count = int(sample_count)

        if all_training_samples.shape[0] < sample_count:
            raise RuntimeError(
                f"Requested {sample_count} samples, but only "
                f"{all_training_samples.shape[0]} are available."
            )

        samples_by_count[sample_count] = all_training_samples[:sample_count]

    return samples_by_count