"""
Shared utilities for Peaks surrogate verification examples.

These examples use the 2D Peaks function from:

Kaminsky, A. L., Wang, Y., and Pant, K.,
"An Efficient Batch K-Fold Cross-Validation Voronoi Adaptive Sampling
Technique for Global Surrogate Modeling,"
Journal of Mechanical Design, 143(1), 011706, 2021.
"""
import copy
import os

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

PARAMETER_NAMES = ["x1", "x2"]
PARAMETER_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]

N_VALIDATION_SAMPLES = 2500

RANDOM_SEED = 54321
VALIDATION_SEED = 12345

INITIAL_GRID_POINTS_PER_DIMENSION = 4
N_INITIAL_SAMPLES = INITIAL_GRID_POINTS_PER_DIMENSION**2


# =============================================================================
# Peaks benchmark function
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
# MatCal setup helpers
# =============================================================================

def make_parameters():
    """
    Create MatCal parameters for the Peaks domain.
    """
    return [
        mc.Parameter(name, lower, upper)
        for name, (lower, upper) in zip(PARAMETER_NAMES, PARAMETER_BOUNDS)
    ]


def make_model():
    """
    Create the MatCal PythonModel for the Peaks function.
    """
    model = mc.PythonModel(peaks_python_model)
    model.set_name("paper_peaks_function_model")
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

    MatCal writes completed study results to ``final_results.joblib`` inside the
    study working directory.

    :param working_directory: Study working directory.
    :type working_directory: str

    :return: Path to the cached final results file.
    :rtype: str
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
    expects. For example, a cached file may contain

    ``paper_peaks_function_model:SimulationResultsSynchronizer_0``

    while the current study expects

    ``paper_peaks_function_model:SimulationResultsSynchronizer_1``.

    The simulation history and parameter history are still valid. This helper
    simply adds an additional key to ``qoi_history`` that points to the same QoI
    data under the expected name.

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
    new MatCal study. This avoids regenerating the same validation/test data each
    time the example is run.

    Set ``force_rerun=True`` to ignore the cached file and regenerate the
    validation set.

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
):
    """
    Evaluate the analytic model at prescribed sample locations.
    """
    samples = np.asarray(samples, dtype=float)
    working_directory = os.path.abspath(working_directory)
    os.makedirs(os.path.dirname(working_directory), exist_ok=True)

    parameter_study = mc.ParameterStudy(*parameters)
    parameter_study.add_evaluation_set(model, objective)

    for sample in samples:
        parameter_study.add_parameter_evaluation(
            x1=float(sample[0]),
            x2=float(sample[1]),
        )

    parameter_study.set_working_directory(
        working_directory,
        remove_existing=True,
    )

    return parameter_study.launch()


def make_uniform_random_samples(nsamples, seed):
    """
    Generate independent uniform random samples over the Peaks domain.
    """
    rng = np.random.default_rng(seed)

    lower_bounds = np.asarray([b[0] for b in PARAMETER_BOUNDS], dtype=float)
    upper_bounds = np.asarray([b[1] for b in PARAMETER_BOUNDS], dtype=float)

    unit_samples = rng.random((int(nsamples), len(PARAMETER_NAMES)))

    return lower_bounds + unit_samples * (upper_bounds - lower_bounds)


# =============================================================================
# Paper-style initial grid for Voronoi adaptive studies
# =============================================================================

class PaperPeaksInitialGridVoronoiStudy(mc.VoronoiAdaptiveSurrogateStudy):
    """
    Voronoi adaptive surrogate study with the paper's 4 by 4 initial grid.

    The base MatCal Voronoi adaptive study initializes with generated samples.
    This subclass overrides only the initial training samples so the Peaks
    adaptive example starts from the same 4 by 4 grid used in the paper's 2D
    low-dimensional examples.
    """

    def _make_paper_initial_grid(self):
        x1_values = np.linspace(
            PARAMETER_BOUNDS[0][0],
            PARAMETER_BOUNDS[0][1],
            INITIAL_GRID_POINTS_PER_DIMENSION,
        )
        x2_values = np.linspace(
            PARAMETER_BOUNDS[1][0],
            PARAMETER_BOUNDS[1][1],
            INITIAL_GRID_POINTS_PER_DIMENSION,
        )

        xx, yy = np.meshgrid(x1_values, x2_values)

        return np.column_stack((xx.ravel(), yy.ravel()))

    def _run_initial_training_samples(self):
        initial_samples = self._make_paper_initial_grid()

        self._populate_parameter_evaluations(initial_samples)
        self._matcal_evaluate_parameter_sets_batch(
            self._parameter_sets_to_evaluate
        )

        return self._train_surrogate_with_current_results()


# =============================================================================
# Scoring and plotting helpers
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


def plot_convergence_history(
    sample_counts,
    rmse,
    max_errors,
    method_name,
    figure_directory,
    filename,
    max_sample_count=None,
):
    """
    Plot RMSE and maximum absolute error versus training sample count.

    The plot also includes empirical power-law convergence fits of the form

    .. math::

        E(N) \\approx C N^{-p}.

    :param sample_counts: Training sample counts.
    :type sample_counts: numpy.ndarray

    :param rmse: Validation RMSE values.
    :type rmse: numpy.ndarray

    :param max_errors: Validation maximum absolute error values.
    :type max_errors: numpy.ndarray

    :param method_name: Label for the plot title.
    :type method_name: str

    :param figure_directory: Directory where the figure is saved.
    :type figure_directory: str

    :param filename: Figure filename.
    :type filename: str

    :param max_sample_count: Optional maximum sample count to show. If ``None``,
        all supplied history points are plotted.
    :type max_sample_count: int or None

    :return: Matplotlib ``(fig, ax)`` pair.
    :rtype: tuple
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
    ax.set_title(f"Peaks verification: {method_name}")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()

    figure_path = os.path.join(figure_directory, filename)
    fig.savefig(figure_path, dpi=300)

    return fig, ax


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

def parameter_matrix_from_results(results):
    """
    Extract a parameter matrix from MatCal StudyResults.

    :param results: MatCal study results.
    :type results: matcal.core.study_base.StudyResults

    :return: Parameter matrix with shape ``(n_samples, 2)``.
    :rtype: numpy.ndarray
    """
    return np.column_stack([
        np.asarray(results.parameter_history[name], dtype=float)
        for name in PARAMETER_NAMES
    ])


def _select_retained_surrogate_for_range_enforcement(
    surrogate,
    surrogate_index,
):
    """
    Return a retained underlying surrogate object, when applicable.

    Adaptive surrogate containers store retained surrogate objects internally.
    For plotting on the full parameter-domain grid, range enforcement may need
    to be disabled on both the container and the retained surrogate.
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

    This helper supports both direct MatCal PCA/RBF surrogate objects and
    adaptive surrogate containers. Range enforcement is temporarily disabled
    because the diagnostic grid includes exact parameter bounds, while a trained
    surrogate's internal sampled range may lie slightly inside those bounds.

    :param surrogate: Direct surrogate or AdaptiveSurrogate container.
    :type surrogate: object

    :param points: Parameter points with shape ``(n_points, 2)``.
    :type points: numpy.ndarray

    :param surrogate_index: Optional retained-surrogate selector for adaptive
        surrogate containers.
    :type surrogate_index: int or str or None

    :return: Predicted scalar values with shape ``(n_points,)``.
    :rtype: numpy.ndarray
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


def make_2d_prediction_grid(n_grid=150):
    """
    Create a two-dimensional prediction grid over the Peaks parameter domain.

    :param n_grid: Number of grid points per dimension.
    :type n_grid: int

    :return: Tuple ``(xx, yy, grid_points)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    bounds = np.asarray(PARAMETER_BOUNDS, dtype=float)

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


def plot_function_and_surrogate_error_at_counts(
    surrogates_by_count,
    training_samples_by_count,
    sample_counts_to_plot=(50, 100, 150),
    figure_directory="figures",
    filename="peaks_function_and_surrogate_error_fields.png",
    method_name="surrogate",
    n_grid=150,
):
    """
    Plot true Peaks function and absolute surrogate error fields.

    The figure has two rows and one column per requested sample count.

    The first row shows the true Peaks function with the training samples used
    by that surrogate overlaid. The second row shows the absolute surrogate
    error field,

    .. math::

        |f(x) - \\hat{f}(x)|,

    with the same training samples overlaid.

    :param surrogates_by_count: Dictionary keyed by sample count. Each value
        must contain ``"surrogate"`` and may contain ``"surrogate_index"``.
    :type surrogates_by_count: dict[int, dict]

    :param training_samples_by_count: Dictionary keyed by sample count. Each
        value is an array of training samples with shape ``(sample_count, 2)``.
    :type training_samples_by_count: dict[int, numpy.ndarray]

    :param sample_counts_to_plot: Sample counts to show.
    :type sample_counts_to_plot: tuple[int]

    :param figure_directory: Directory where the figure is saved.
    :type figure_directory: str

    :param filename: Figure filename.
    :type filename: str

    :param method_name: Label used in the figure title.
    :type method_name: str

    :param n_grid: Number of grid points per dimension.
    :type n_grid: int

    :return: Matplotlib ``(fig, axes)`` pair.
    :rtype: tuple
    """
    os.makedirs(figure_directory, exist_ok=True)

    xx, yy, grid_points = make_2d_prediction_grid(n_grid=n_grid)

    true_values = peaks_function(grid_points)
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
            f"True Peaks function\n{sample_count_label}"
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
        axis.set_xlim(PARAMETER_BOUNDS[0])
        axis.set_ylim(PARAMETER_BOUNDS[1])

    function_colorbar = fig.colorbar(
        function_contours[-1],
        ax=axes[0, :],
        location="right",
        shrink=0.88,
        pad=0.015,
    )
    function_colorbar.set_label("Peaks function value")

    error_colorbar = fig.colorbar(
        error_contours[-1],
        ax=axes[1, :],
        location="right",
        shrink=0.88,
        pad=0.015,
    )
    error_colorbar.set_label("absolute surrogate error")

    fig.suptitle(
        f"Peaks verification: {method_name}",
        fontsize=15,
    )

    figure_path = os.path.join(figure_directory, filename)
    fig.savefig(figure_path, dpi=300)

    return fig, axes


def _nearest_adaptive_surrogate_record(
    adaptive_surrogate,
    requested_sample_count,
):
    """
    Find the retained adaptive-surrogate record whose sample count is nearest to
    a requested sample count.

    Ties are resolved by selecting the smaller sample count.

    :param adaptive_surrogate: Adaptive surrogate container.
    :type adaptive_surrogate: matcal.core.adaptive_surrogates.AdaptiveSurrogate

    :param requested_sample_count: Desired sample count.
    :type requested_sample_count: int

    :return: Nearest retained surrogate record.
    :rtype: collections.OrderedDict
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
    based. Some adaptive methods, especially sparse-grid adaptive refinement, do not
    necessarily generate surrogates at exactly the requested sample counts.

    The returned dictionary is keyed by the requested sample count. Each value
    stores both the requested sample count and the actual retained surrogate
    sample count.

    :param adaptive_surrogate: Adaptive surrogate container.
    :type adaptive_surrogate: matcal.core.adaptive_surrogates.AdaptiveSurrogate

    :param sample_counts_to_collect: Desired sample counts.
    :type sample_counts_to_collect: tuple[int]

    :return: Dictionary keyed by requested sample count.
    :rtype: dict[int, dict]
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


def collect_training_samples_for_surrogate_count_map(
    study_results,
    surrogates_by_count,
):
    """
    Collect training samples that correspond to selected adaptive surrogates.

    The input ``surrogates_by_count`` is expected to come from
    :func:`collect_adaptive_surrogates_at_sample_counts`. It is keyed by the
    requested sample count, but each metadata dictionary contains the actual
    retained surrogate sample count.

    :param study_results: Adaptive study training results.
    :type study_results: matcal.core.study_base.StudyResults

    :param surrogates_by_count: Dictionary keyed by requested sample count.
    :type surrogates_by_count: dict[int, dict]

    :return: Dictionary keyed by requested sample count. Each value contains the
        training samples used by the nearest retained surrogate.
    :rtype: dict[int, numpy.ndarray]
    """
    all_training_samples = parameter_matrix_from_results(study_results)

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


def collect_training_samples_at_counts(
    study_results,
    sample_counts_to_collect=(50, 100, 150),
):
    """
    Collect training sample arrays from an adaptive study result.

    :param study_results: Adaptive study training results.
    :type study_results: matcal.core.study_base.StudyResults

    :param sample_counts_to_collect: Desired sample counts.
    :type sample_counts_to_collect: tuple[int]

    :return: Dictionary keyed by sample count.
    :rtype: dict[int, numpy.ndarray]
    """
    all_training_samples = parameter_matrix_from_results(study_results)

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